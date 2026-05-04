"""
trainer/loss.py
─────────────────────────────────────────────────────────────────
Four-term CVAE objective:

  L = λ1·L_recon + λ2·L_PC + λ3·L_orth + λ4_w·KL

Expected inputs:
  smri        : (B, 1,  64, 64, 64)
  components  : (B, K,  64, 64, 64)
  ica_stacked : (B, K,  64, 64, 64)
  mu          : (B, latent_dim)
  logvar      : (B, latent_dim)

The updated dataloader returns:
  (smri, ica_stacked)

The updated model returns:
  out["components"], out["mu"], out["logvar"]
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Term 1 — Reconstruction fidelity
# ─────────────────────────────────────────────────────────────────────────────

def reconstruction_loss(
    smri: torch.Tensor,
    components: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    MSE between sMRI and sum of generated components.

    smri:
      (B, 1, D, H, W)

    components:
      (B, K, D, H, W)

    mask:
      optional, (B, 1, D, H, W)
    """
    assert smri.ndim == 5, \
        f"Expected smri shape (B,1,D,H,W), got {tuple(smri.shape)}"

    assert components.ndim == 5, \
        f"Expected components shape (B,K,D,H,W), got {tuple(components.shape)}"

    assert smri.shape[0] == components.shape[0], \
        f"Batch mismatch: smri={smri.shape[0]}, components={components.shape[0]}"

    assert smri.shape[1] == 1, \
        f"Expected smri channel dim 1, got {smri.shape[1]}"

    pred = components.sum(dim=1, keepdim=True)   # (B, 1, D, H, W)

    assert pred.shape == smri.shape, \
        f"Prediction shape {tuple(pred.shape)} does not match smri {tuple(smri.shape)}"

    diff = smri - pred

    if mask is not None:
        assert mask.shape == smri.shape, \
            f"Expected mask shape {tuple(smri.shape)}, got {tuple(mask.shape)}"

        mask = mask.float()
        numer = (diff * mask).pow(2).sum(dim=(1, 2, 3, 4))
        denom = mask.sum(dim=(1, 2, 3, 4)).clamp(min=1.0)
        return (numer / denom).mean()

    return diff.pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Term 2 — Structure-function alignment
# ─────────────────────────────────────────────────────────────────────────────

def pearson_alignment_loss(
    components: torch.Tensor,
    ica_stacked: torch.Tensor,
    eps: float = 1e-8,
) -> tuple:
    """
    Pearson alignment between generated component ŝ_ik and ICA map c_ik.

    components:
      (B, K, D, H, W)

    ica_stacked:
      (B, K, D, H, W)

    returns:
      loss scalar
      rho : (B, K), detached
    """
    assert components.ndim == 5, \
        f"Expected components shape (B,K,D,H,W), got {tuple(components.shape)}"

    assert ica_stacked.ndim == 5, \
        f"Expected ica_stacked shape (B,K,D,H,W), got {tuple(ica_stacked.shape)}"

    assert components.shape == ica_stacked.shape, \
        f"components shape {tuple(components.shape)} must match ica_stacked shape {tuple(ica_stacked.shape)}"

    B, K = components.shape[:2]
    V = components.shape[2] * components.shape[3] * components.shape[4]

    s = components.reshape(B, K, V)
    c = ica_stacked.reshape(B, K, V)

    s_centered = s - s.mean(dim=2, keepdim=True)
    c_centered = c - c.mean(dim=2, keepdim=True)

    num = (s_centered * c_centered).sum(dim=2)
    denom = s_centered.norm(dim=2) * c_centered.norm(dim=2)
    rho = num / denom.clamp(min=eps)

    loss = (1.0 - rho).mean()

    return loss, rho.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Term 3 — Soft orthogonality
# ─────────────────────────────────────────────────────────────────────────────

def orthogonality_loss(
    components: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Soft orthogonality between generated components.

    components:
      (B, K, D, H, W)
    """
    assert components.ndim == 5, \
        f"Expected components shape (B,K,D,H,W), got {tuple(components.shape)}"

    B, K = components.shape[:2]

    S = components.reshape(B, K, -1)              # (B, K, V)
    G = torch.bmm(S, S.transpose(1, 2))           # (B, K, K)

    frob2 = S.pow(2).sum(dim=(1, 2), keepdim=True).clamp(min=eps)
    G_norm = G / frob2

    target = torch.eye(
        K,
        device=components.device,
        dtype=components.dtype,
    ).unsqueeze(0) / K

    return (G_norm - target).pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Term 4 — KL divergence
# ─────────────────────────────────────────────────────────────────────────────

def kl_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    KL(q(z|x) || p(z)) for diagonal Gaussian posterior.
    """
    assert mu.shape == logvar.shape, \
        f"mu shape {tuple(mu.shape)} must match logvar shape {tuple(logvar.shape)}"

    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss
# ─────────────────────────────────────────────────────────────────────────────

class CVAELoss(nn.Module):
    """
    L = λ1·L_recon + λ2·L_PC + λ3·L_orth + λ4_w·KL

    KL weight is annealed from 0 to λ4 using a cosine warmup.
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        lambda3: float = 0.1,
        lambda4: float = 0.001,
        warmup_epochs: int = 20,
    ):
        super().__init__()

        self.l1 = lambda1
        self.l2 = lambda2
        self.l3 = lambda3
        self.l4 = lambda4
        self.warmup = warmup_epochs

    def _kl_weight(self, epoch: int) -> float:
        """
        Cosine ramp:
          epoch = 0       → 0
          epoch >= warmup → lambda4
        """
        if self.warmup <= 0:
            return self.l4

        if epoch >= self.warmup:
            return self.l4

        return self.l4 * 0.5 * (
            1.0 - math.cos(math.pi * epoch / self.warmup)
        )

    def forward(
        self,
        smri: torch.Tensor,
        components: torch.Tensor,
        ica_stacked: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        mask: torch.Tensor = None,
        epoch: int = 0,
    ) -> tuple:
        """
        Returns:
          total    : scalar
          l_recon  : scalar
          l_pc     : scalar
          l_orth   : scalar
          l_kl     : scalar
          rho      : (B, K), detached
        """
        l_recon = reconstruction_loss(
            smri=smri,
            components=components,
            mask=mask,
        )

        l_pc, rho = pearson_alignment_loss(
            components=components,
            ica_stacked=ica_stacked,
        )

        l_orth = orthogonality_loss(
            components=components,
        )

        l_kl = kl_loss(
            mu=mu,
            logvar=logvar,
        )

        kl_weight = self._kl_weight(epoch)

        total = (
            self.l1 * l_recon
            + self.l2 * l_pc
            + self.l3 * l_orth
            + kl_weight * l_kl
        )

        return total, l_recon, l_pc, l_orth, l_kl, rho