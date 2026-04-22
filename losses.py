import math
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=num_classes).to(labels.device).float()


def get_curriculum_class_weights(
    epoch: int,
    num_classes: int,
    start_epoch: int = 10,
    decay_rate: float = 0.1,
    min_weight: float = 0.1,
) -> torch.Tensor:
    """
    Same curriculum weighting logic used in train_v3.py.
    """
    if epoch < start_epoch:
        return torch.ones(num_classes)

    decay_factor = torch.exp(torch.tensor(-decay_rate * (epoch - start_epoch))).item()
    base_weights = torch.tensor([0.3, 0.8, 0.8, 1.0, 1.0, 1.0], dtype=torch.float32)
    decay_multipliers = torch.tensor([1.0, 0.5, 0.5, 0.3, 0.3, 0.3], dtype=torch.float32)

    if num_classes != 6:
        # Keep behavior sensible for arbitrary ordinal class counts.
        # Easiest classes (lower indices) get stronger decay than hardest classes.
        idx = torch.arange(num_classes, dtype=torch.float32)
        frac = idx / max(1.0, float(num_classes - 1))
        base_weights = 0.3 + 0.7 * frac
        decay_multipliers = 1.0 - 0.7 * frac

    weights = base_weights * (1.0 - decay_multipliers * (1.0 - decay_factor))
    weights = torch.clamp(weights, min=min_weight)
    return weights


class AuxWeightController:
    """
    Epoch-scheduled + ratio-adaptive auxiliary weight controller,
    mirroring train_v3.py behavior.
    """

    def __init__(
        self,
        aux_weight_start: float = 0.5,
        aux_weight_end: float = 0.05,
        aux_weight_decay_start: int = 1,
        aux_weight_decay_end: int = 15,
        aux_weight_schedule: str = "linear",
        aux_ratio_adapt: bool = True,
        aux_target_ratio: float = 0.05,
        ratio_ema_beta: float = 0.9,
    ):
        self.aux_weight_start = float(aux_weight_start)
        self.aux_weight_end = float(aux_weight_end)
        self.aux_weight_decay_start = int(aux_weight_decay_start)
        self.aux_weight_decay_end = int(aux_weight_decay_end)
        self.aux_weight_schedule = str(aux_weight_schedule)
        self.aux_ratio_adapt = bool(aux_ratio_adapt)
        self.aux_target_ratio = float(aux_target_ratio)
        self.ratio_ema_beta = float(ratio_ema_beta)
        self.ratio_ema: Optional[float] = None

    def scheduled_weight(self, epoch: int) -> float:
        if epoch <= self.aux_weight_decay_start:
            return self.aux_weight_start
        if epoch >= self.aux_weight_decay_end:
            return self.aux_weight_end

        t = (epoch - self.aux_weight_decay_start) / max(
            1, (self.aux_weight_decay_end - self.aux_weight_decay_start)
        )

        if self.aux_weight_schedule == "linear":
            return self.aux_weight_start + (self.aux_weight_end - self.aux_weight_start) * t
        if self.aux_weight_schedule == "cosine":
            return self.aux_weight_end + (self.aux_weight_start - self.aux_weight_end) * 0.5 * (
                1.0 + math.cos(math.pi * t)
            )
        if self.aux_weight_schedule == "exp":
            safe = max(1e-8, self.aux_weight_start)
            ratio = self.aux_weight_end / safe
            return safe * (ratio**t)
        return self.aux_weight_start + (self.aux_weight_end - self.aux_weight_start) * t

    def adapt_weight(
        self,
        base_weight: float,
        main_loss: torch.Tensor,
        aux_loss: torch.Tensor,
    ) -> float:
        if not self.aux_ratio_adapt:
            return float(base_weight)

        m = float(aux_loss.item())
        c = float(main_loss.item())
        if m + c < 1e-9:
            return float(base_weight)

        current_ratio = m / (m + c)
        if self.ratio_ema is None:
            self.ratio_ema = current_ratio
        else:
            self.ratio_ema = self.ratio_ema_beta * self.ratio_ema + (1.0 - self.ratio_ema_beta) * current_ratio

        factor = self.aux_target_ratio / max(1e-8, self.ratio_ema)
        adjusted = base_weight * factor
        return float(min(max(adjusted, 0.0), 10.0))


class EMDOrdinalLoss(nn.Module):
    """
    Earth Mover's Distance (squared) for ordinal classification.
    Uses CDF distance between predicted class distribution and one-hot target.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_prob = F.softmax(logits, dim=1)
        target_dist = _one_hot(target, self.num_classes)

        pred_cdf = pred_prob.cumsum(dim=1)
        target_cdf = target_dist.cumsum(dim=1)
        emd_sq = (pred_cdf - target_cdf).pow(2).sum(dim=1)

        if class_weights is not None:
            class_weights = class_weights.to(logits.device, dtype=logits.dtype)
            emd_sq = emd_sq * class_weights[target.long()]

        return emd_sq.mean()


class OrdinalNegWeightedSupConLoss(nn.Module):
    """
    Ordinal-aware supervised contrastive loss with optional domain awareness.

    Positives:
      - same class only
    Negatives:
      - weighted by ordinal distance
      - optionally boosted for cross-domain negatives
    """

    def __init__(
        self,
        temperature: float = 0.07,
        neg_strength: float = 1.0,
        max_label_distance: Optional[int] = None,
        domain_aware: bool = False,
        domain_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.neg_strength = float(neg_strength)
        self.max_label_distance = None if max_label_distance is None else int(max_label_distance)
        self.domain_aware = bool(domain_aware)
        self.domain_weight = float(domain_weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = embeddings.device
        labels = labels.view(-1).to(device)
        z = F.normalize(embeddings, dim=1)
        n = labels.size(0)

        sim = torch.matmul(z, z.t()) / max(self.temperature, 1e-6)

        logits_mask = torch.ones((n, n), device=device, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)

        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = labels_equal & logits_mask
        neg_mask = (~labels_equal) & logits_mask

        yi = labels.unsqueeze(1).float()
        yj = labels.unsqueeze(0).float()
        dist = torch.abs(yi - yj)
        if self.max_label_distance is None:
            maxd = max(1.0, float(labels.max().item() - labels.min().item()))
        else:
            maxd = max(1.0, float(self.max_label_distance))
        d_norm = (dist / maxd).clamp(min=0.0, max=1.0)

        w_neg = 1.0 + self.neg_strength * d_norm
        w_neg = w_neg * neg_mask.float()

        if self.domain_aware:
            if domain_labels is None:
                raise ValueError("domain_labels must be provided when domain_aware=True.")
            domain_labels = domain_labels.view(-1).to(device)
            cross_domain = (domain_labels.unsqueeze(0) != domain_labels.unsqueeze(1)).float()
            w_neg = w_neg * (1.0 + self.domain_weight * cross_domain)

        denom_weights = torch.ones_like(sim, device=device)
        denom_weights = denom_weights * pos_mask.float() + w_neg

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim) * logits_mask.float() * denom_weights
        denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8
        log_prob = sim - torch.log(denom)

        pos_count = pos_mask.sum(dim=1)
        pos_count = torch.clamp(pos_count, min=1)
        sum_pos = (log_prob * pos_mask.float()).sum(dim=1)
        mean_log_prob_pos = sum_pos / pos_count.float()

        loss_vec = -mean_log_prob_pos
        if sample_weights is not None:
            sw = sample_weights.view(-1).to(device).float()
            sw = sw / (sw.mean() + 1e-8)
            return (loss_vec * sw).mean()
        return loss_vec.mean()


class SCLEMDLoss(nn.Module):
    """
    Composite wrapper:
      total = emd_weight * EMD(logits, labels) + supcon_weight * SCL(embeddings, labels)
    """

    def __init__(
        self,
        num_classes: int,
        emd_weight: float = 1.0,
        supcon_weight: float = 0.5,
        temperature: float = 0.07,
        neg_strength: float = 1.0,
        max_label_distance: Optional[int] = None,
        domain_aware: bool = False,
        domain_weight: float = 1.0,
    ):
        super().__init__()
        num_classes = int(num_classes)
        self.emd_weight = float(emd_weight)
        self.supcon_weight = float(supcon_weight)

        self.emd = EMDOrdinalLoss(num_classes=num_classes)
        if max_label_distance is None:
            max_label_distance = num_classes - 1
        self.supcon = OrdinalNegWeightedSupConLoss(
            temperature=temperature,
            neg_strength=neg_strength,
            max_label_distance=max_label_distance,
            domain_aware=domain_aware,
            domain_weight=domain_weight,
        )

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        emd_loss = self.emd(logits, labels, class_weights=class_weights)
        supcon_loss = self.supcon(
            embeddings,
            labels,
            domain_labels=domain_labels,
            sample_weights=sample_weights,
        )
        total = self.emd_weight * emd_loss + self.supcon_weight * supcon_loss

        if return_dict:
            return {
                "total": total,
                "emd": emd_loss,
                "supcon": supcon_loss,
            }
        return total


class TrainV3StyleLoss(nn.Module):
    """
    Loss wrapper aligned with train_v3.py innovations.

    Supports:
      - main loss: "ce" | "emd" | "ord_contrastive"
      - optional auxiliary ordinal contrastive
      - curriculum class weighting
      - scheduled + ratio-adapted auxiliary weight
      - optional domain-aware contrastive
    """

    def __init__(
        self,
        num_classes: int,
        main_loss: str = "emd",
        aux_loss: str = "contrastive",
        temperature: float = 0.07,
        neg_strength: float = 1.0,
        max_label_distance: Optional[int] = None,
        domain_aware: bool = False,
        domain_weight: float = 1.0,
        use_curriculum: bool = False,
        curr_start: int = 10,
        curr_decay: float = 0.1,
        curr_min_weight: float = 0.1,
        aux_weight_start: float = 0.5,
        aux_weight_end: float = 0.05,
        aux_weight_decay_start: int = 1,
        aux_weight_decay_end: int = 15,
        aux_weight_schedule: str = "linear",
        aux_ratio_adapt: bool = True,
        aux_target_ratio: float = 0.05,
        ratio_ema_beta: float = 0.9,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.main_loss = str(main_loss)
        self.aux_loss = str(aux_loss)
        self.use_curriculum = bool(use_curriculum)
        self.curr_start = int(curr_start)
        self.curr_decay = float(curr_decay)
        self.curr_min_weight = float(curr_min_weight)

        if max_label_distance is None:
            max_label_distance = self.num_classes - 1

        self.emd = EMDOrdinalLoss(num_classes=self.num_classes)
        self.supcon = OrdinalNegWeightedSupConLoss(
            temperature=temperature,
            neg_strength=neg_strength,
            max_label_distance=max_label_distance,
            domain_aware=domain_aware,
            domain_weight=domain_weight,
        )
        self.aux_controller = AuxWeightController(
            aux_weight_start=aux_weight_start,
            aux_weight_end=aux_weight_end,
            aux_weight_decay_start=aux_weight_decay_start,
            aux_weight_decay_end=aux_weight_decay_end,
            aux_weight_schedule=aux_weight_schedule,
            aux_ratio_adapt=aux_ratio_adapt,
            aux_target_ratio=aux_target_ratio,
            ratio_ema_beta=ratio_ema_beta,
        )

    def _get_class_weights(self, epoch: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not self.use_curriculum:
            return None
        cw = get_curriculum_class_weights(
            epoch=epoch,
            num_classes=self.num_classes,
            start_epoch=self.curr_start,
            decay_rate=self.curr_decay,
            min_weight=self.curr_min_weight,
        )
        return cw.to(device=device, dtype=dtype)

    def _classification_aware_sample_weights(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, labels.view(-1, 1)).squeeze(1)
        sw = 1.0 + (1.0 - pt)
        if class_weights is not None:
            sw = sw * class_weights[labels.long()]
        return sw

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
        domain_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        class_weights = self._get_class_weights(epoch, logits.device, logits.dtype)

        if self.main_loss == "ce":
            main_loss = F.cross_entropy(logits, labels, weight=class_weights)
        elif self.main_loss == "emd":
            main_loss = self.emd(logits, labels, class_weights=class_weights)
        elif self.main_loss == "ord_contrastive":
            sw_main = self._classification_aware_sample_weights(logits, labels, class_weights)
            main_loss = self.supcon(
                embeddings,
                labels,
                domain_labels=domain_labels,
                sample_weights=sw_main,
            )
        else:
            raise ValueError(f"Unknown main_loss: {self.main_loss}")

        aux_loss = torch.tensor(0.0, device=logits.device)
        aux_weight = 0.0

        if self.main_loss != "ord_contrastive" and self.aux_loss == "contrastive":
            sw_aux = self._classification_aware_sample_weights(logits, labels, class_weights)
            aux_loss = self.supcon(
                embeddings,
                labels,
                domain_labels=domain_labels,
                sample_weights=sw_aux,
            )
            base = self.aux_controller.scheduled_weight(epoch)
            aux_weight = self.aux_controller.adapt_weight(base, main_loss=main_loss, aux_loss=aux_loss)

        total = main_loss + aux_weight * aux_loss

        if not return_dict:
            return total

        if class_weights is None:
            class_weights = torch.empty(0, device=logits.device, dtype=logits.dtype)
        return {
            "total": total,
            "main": main_loss,
            "aux": aux_loss,
            "aux_weight": torch.tensor(aux_weight, device=logits.device, dtype=logits.dtype),
            "class_weights": class_weights,
        }


def make_emd_only_loss(num_classes: int) -> EMDOrdinalLoss:
    """
    Preset A (simple):
      Replace CE with EMD.
    """
    return EMDOrdinalLoss(num_classes=int(num_classes))


def make_full_ordinal_loss(
    num_classes: int,
    use_domains: bool = False,
    domain_weight: float = 0.5,
    use_curriculum: bool = True,
    curr_start: int = 3,
    **kwargs,
) -> TrainV3StyleLoss:
    """
    Preset B (recommended):
      Main = EMD
      Aux  = ordinal contrastive
      + curriculum
      + scheduled/adaptive aux weight
      + optional domain-aware behavior
    """
    config = {
        "num_classes": int(num_classes),
        "main_loss": "emd",
        "aux_loss": "contrastive",
        "temperature": 0.07,
        "neg_strength": 1.0,
        "domain_aware": bool(use_domains),
        "domain_weight": float(domain_weight),
        "use_curriculum": bool(use_curriculum),
        "curr_start": int(curr_start),
        "curr_decay": 0.1,
        "curr_min_weight": 0.1,
        "aux_weight_start": 0.5,
        "aux_weight_end": 0.05,
        "aux_weight_decay_start": 1,
        "aux_weight_decay_end": 15,
        "aux_weight_schedule": "linear",
        "aux_ratio_adapt": True,
        "aux_target_ratio": 0.05,
    }
    config.update(kwargs)
    return TrainV3StyleLoss(
        **config,
    )


def build_loss(
    loss_name: str,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    """
    Factory helper.

    Supported names:
      - "emd"
      - "emd_only"     (alias of emd)
      - "ordinal_supcon"
      - "scl_emd"
      - "scl_emd_wd"  (same as scl_emd, keep name for convenience)
      - "train_v3_style"
      - "full_ordinal" (preset aligned with train_v3.py best setting)
    """
    key = loss_name.lower().strip()
    if key in {"emd", "emd_only"}:
        return EMDOrdinalLoss(num_classes=num_classes)
    if key == "ordinal_supcon":
        if "max_label_distance" not in kwargs:
            kwargs["max_label_distance"] = int(num_classes) - 1
        return OrdinalNegWeightedSupConLoss(**kwargs)
    if key in {"scl_emd", "scl_emd_wd"}:
        if "max_label_distance" not in kwargs:
            kwargs["max_label_distance"] = int(num_classes) - 1
        return SCLEMDLoss(num_classes=num_classes, **kwargs)
    if key == "train_v3_style":
        return TrainV3StyleLoss(num_classes=num_classes, **kwargs)
    if key == "full_ordinal":
        return make_full_ordinal_loss(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown loss_name: {loss_name}")
