import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class ABMILISUP(nn.Module):
    """
    Gated ABMIL (two-layer gated variant) for slide-level ISUP classification
    using pre-extracted patch features (e.g., UNI/UNI2 1536-D embeddings).

    See https://www.nature.com/articles/s41591-024-02857-3#Sec13 -> evaluation on PANDA WSI-level ISUP classification.

    Matches the UNI Methods config:
      - Input dropout: p = 0.10 on the *input embeddings* X
      - First FC maps input_dim -> 512
      - Gated attention hidden dims = 384 (V and U branches)
      - Dropout p = 0.25 *after each intermediate layer* (V and U branches, and
        the classifier hidden layer)
      - Classifier: 512 -> 384 -> num_classes

    Inputs
    ------
    X    : Tensor [B, N, D] or [N, D]
    mask : Optional[BoolTensor] [B, N], True for valid patches (handles padding)

    Outputs
    -------
    logits : [B, num_classes]
    attn   : [B, N]  (attention weights over patches, sum to 1 over valid N)
    z      : [B, 512] (slide embeddings)
    """
    def __init__(
        self,
        input_dim: int = 1536,   # your UNI features are 1536-D
        num_classes: int = 6,    # ISUP 0-5
        proj_dim: int = 512,     # fixed by paper
        attn_hidden: int = 384,  # fixed by paper
        p_input_dropout: float = 0.10,
        p_mid_dropout: float = 0.25,
        use_proj_layernorm: bool = False,
        use_embed_layernorm: bool = False,
        use_l2_norm_embed: bool = False,
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(p_input_dropout)
        self.use_proj_layernorm = use_proj_layernorm
        self.use_embed_layernorm = use_embed_layernorm
        self.use_l2_norm_embed = use_l2_norm_embed

        # First FC: input_dim -> 512
        self.proj = nn.Linear(input_dim, proj_dim)
        self.proj_ln = nn.LayerNorm(proj_dim) if use_proj_layernorm else None

        # Gated attention:
        # a_i ∝ exp( w^T [tanh(V h_i) ⊙ σ(U h_i)] )
        self.attn_V = nn.Linear(proj_dim, attn_hidden)
        self.attn_U = nn.Linear(proj_dim, attn_hidden)
        self.attn_w = nn.Linear(attn_hidden, 1)

        # Dropout p=0.25 after each intermediate layer (here: after V and U branches)
        self.attn_do = nn.Dropout(p_mid_dropout)

        # Classifier: 512 -> 384 -> C   (dropout after hidden)
        self.cls_fc1 = nn.Linear(proj_dim, attn_hidden)
        self.cls_do = nn.Dropout(p_mid_dropout)
        self.cls_fc2 = nn.Linear(attn_hidden, num_classes)
        self.embed_ln = nn.LayerNorm(proj_dim) if use_embed_layernorm else None

        # Nonlinearities
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None, return_h: bool = False):
        squeeze_back = False
        if X.dim() == 2:  # [N, D] -> [1, N, D]
            X = X.unsqueeze(0)
            squeeze_back = True

        B, N, D = X.shape
        device = X.device
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # Dropout on *input embeddings* (p=0.10)
        X = self.input_dropout(X)

        # Project to 512
        H = self.relu(self.proj(X))   # [B, N, 512]
        if self.proj_ln is not None:
            H = self.proj_ln(H)

        # Gated attention with hidden=384
        Vh = torch.tanh(self.attn_V(H))           # [B, N, 384]
        Uh = torch.sigmoid(self.attn_U(H))        # [B, N, 384]

        # Dropout after each intermediate layer (V and U branches)
        Vh = self.attn_do(Vh)
        Uh = self.attn_do(Uh)

        A = self.attn_w(Vh * Uh).squeeze(-1)      # [B, N]
        A = A.masked_fill(~mask, torch.finfo(A.dtype).min)  # mask pads
        A = F.softmax(A, dim=1)                   # attention weights over patches

        # Slide embedding z = Σ_i a_i * h_i
        z = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # [B, 512]
        if self.embed_ln is not None:
            z = self.embed_ln(z)
        if self.use_l2_norm_embed:
            z = F.normalize(z, dim=1)

        # Classifier 512 -> 384 -> C with dropout after hidden
        h_pre = self.relu(self.cls_fc1(z))           # [B, 384]
        h = self.cls_do(h_pre)                        # dropout p=0.25
        logits = self.cls_fc2(h)                  # [B, C]

        if squeeze_back:
            logits = logits.squeeze(0)
            A = A.squeeze(0)
            z = z.squeeze(0)

        if return_h:
            return logits, A, z, h_pre


        return logits, A, z

    @torch.no_grad()
    def embed(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Return only the slide embedding z (useful for downstream analysis/UMAP).
        """
        self.eval()
        outputs = self.forward(X, mask)
        # Handle both ABMILISUP (3 outputs) and DomainAdversarialABMIL (4 outputs)
        z = outputs[2]  # z is always the third output
        return z


# -------------------- Domain Adversarial Components --------------------

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DomainAdversarialABMIL(ABMILISUP):
    """
    Gated ABMIL with a Domain Adversarial component, now supporting CDANN-style
    conditioning by concatenating z with a conditional vector (e.g., class probs
    or one-hot labels) before the domain classifier.
    """
    def __init__(
        self,
        input_dim: int = 1536,
        num_classes: int = 6,
        proj_dim: int = 512,
        attn_hidden: int = 384,
        p_input_dropout: float = 0.10,
        p_mid_dropout: float = 0.25,
        num_domains: int = 2,
        grl_alpha: float = 1.0,
        use_proj_layernorm: bool = True,
        use_embed_layernorm: bool = True,
        use_l2_norm_embed: bool = True,
        domain_cond_dim: int = 0,  # NEW: CDANN conditional vector dimension
    ):
        # Use normalization for better training stability
        super().__init__(
            input_dim, num_classes, proj_dim, attn_hidden,
            p_input_dropout, p_mid_dropout,
            use_proj_layernorm=use_proj_layernorm,
            use_embed_layernorm=use_embed_layernorm,
            use_l2_norm_embed=use_l2_norm_embed,
        )
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.domain_cond_dim = int(domain_cond_dim) if domain_cond_dim is not None else 0
        in_dim = proj_dim + self.domain_cond_dim
        # Domain classifier accepts [z_rev, cond] if cond provided
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_mid_dropout),
            nn.Linear(256, num_domains)
        )

    def domain_forward(self, z: torch.Tensor, cond: torch.Tensor | None = None):
        """
        Domain head forward with optional CDANN conditioning.
        Gradients are reversed by GRL before concatenation.
        Ensures input dimensionality always matches the classifier by padding
        a zero conditional vector when cond is None or mismatched.
        """
        z_rev = self.grl(z)
        if self.domain_cond_dim > 0:
            # If no conditioning is provided, use zeros of required dim
            if cond is None:
                cond = torch.zeros(z_rev.size(0), self.domain_cond_dim, device=z_rev.device, dtype=z_rev.dtype)
            else:
                if cond.dim() == 1:
                    cond = cond.unsqueeze(1)
                # Adjust cond width to expected domain_cond_dim (pad zeros or truncate)
                if cond.size(1) != self.domain_cond_dim:
                    if cond.size(1) < self.domain_cond_dim:
                        pad = self.domain_cond_dim - cond.size(1)
                        cond = torch.cat([cond, cond.new_zeros(cond.size(0), pad)], dim=1)
                    else:
                        cond = cond[:, : self.domain_cond_dim]
            x = torch.cat([z_rev, cond], dim=1)
        else:
            x = z_rev
        return self.domain_classifier(x)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None, return_h: bool = False):
        # Get the base model's outputs. We need the slide embedding 'z'.
        # The base forward returns (logits, attn, z) or (logits, attn, z, h_pre)
        base_outputs = super().forward(X, mask, return_h=return_h)
        logits, attn, z = base_outputs[0], base_outputs[1], base_outputs[2]

        # Default forward uses zero-conditioning if a conditional dim is set
        domain_logits = self.domain_forward(z, None)

        # Return the main logits, attention, slide embedding, and domain logits
        if return_h:
            return logits, attn, z, domain_logits, base_outputs[3]
        return logits, attn, z, domain_logits