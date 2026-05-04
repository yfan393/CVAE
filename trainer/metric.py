"""
trainer/metric.py
─────────────────────────────────────────────────────────────────
Evaluation metrics.

Expected inputs:
  smri        : (B, 1, 64, 64, 64)
  components  : (B, K, 64, 64, 64)
  ica_stacked : (B, K, 64, 64, 64)
"""

import torch
import numpy as np
from typing import Optional, Tuple
from sklearn.metrics import mutual_info_score
from joblib import Parallel, delayed

N_JOBS = -1
MI_BINS = 32


def _check_component_pair(
    components: torch.Tensor,
    ica_stacked: torch.Tensor,
):
    assert components.ndim == 5, \
        f"Expected components shape (B,K,D,H,W), got {tuple(components.shape)}"

    assert ica_stacked.ndim == 5, \
        f"Expected ica_stacked shape (B,K,D,H,W), got {tuple(ica_stacked.shape)}"

    assert components.shape == ica_stacked.shape, \
        f"components shape {tuple(components.shape)} must match ica_stacked shape {tuple(ica_stacked.shape)}"


@torch.no_grad()
def compute_recon(
    smri: torch.Tensor,
    components: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    RECON = ||smri - sum_k components_k||² / ||smri||²
    """
    assert smri.ndim == 5, \
        f"Expected smri shape (B,1,D,H,W), got {tuple(smri.shape)}"

    assert components.ndim == 5, \
        f"Expected components shape (B,K,D,H,W), got {tuple(components.shape)}"

    assert smri.shape[0] == components.shape[0], \
        f"Batch mismatch: smri={smri.shape[0]}, components={components.shape[0]}"

    pred = components.sum(dim=1, keepdim=True)

    assert pred.shape == smri.shape, \
        f"Prediction shape {tuple(pred.shape)} does not match smri {tuple(smri.shape)}"

    if mask is not None:
        assert mask.shape == smri.shape, \
            f"Expected mask shape {tuple(smri.shape)}, got {tuple(mask.shape)}"

        mask = mask.float()
        diff = (smri - pred) * mask
        numer = diff.pow(2).sum(dim=(1, 2, 3, 4))
        denom = (smri * mask).pow(2).sum(dim=(1, 2, 3, 4)).clamp(min=1e-8)
    else:
        diff = smri - pred
        numer = diff.pow(2).sum(dim=(1, 2, 3, 4))
        denom = smri.pow(2).sum(dim=(1, 2, 3, 4)).clamp(min=1e-8)

    return (numer / denom).mean().item()


@torch.no_grad()
def compute_pearson(
    components: torch.Tensor,
    ica_stacked: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[float, float, torch.Tensor]:
    """
    Spatial Pearson correlation between generated components and stacked ICA.

    components:
      (B, K, D, H, W)

    ica_stacked:
      (B, K, D, H, W)

    returns:
      PC, PC_025, rho
    """
    _check_component_pair(components, ica_stacked)

    B, K = components.shape[:2]
    V = components.shape[2] * components.shape[3] * components.shape[4]

    s = components.reshape(B, K, V)
    c = ica_stacked.reshape(B, K, V)

    s_c = s - s.mean(dim=2, keepdim=True)
    c_c = c - c.mean(dim=2, keepdim=True)

    num = (s_c * c_c).sum(dim=2)
    denom = s_c.norm(dim=2) * c_c.norm(dim=2)
    rho = (num / denom.clamp(min=eps)).cpu()

    pc = rho.mean().item()
    pc_025 = (rho > 0.25).float().mean().item()

    return pc, pc_025, rho


def _mi_one(
    s_flat: np.ndarray,
    c_flat: np.ndarray,
    bins: int,
) -> float:
    lo_s, hi_s = s_flat.min(), s_flat.max() + 1e-8
    lo_c, hi_c = c_flat.min(), c_flat.max() + 1e-8

    s_d = np.digitize(s_flat, np.linspace(lo_s, hi_s, bins + 1))
    c_d = np.digitize(c_flat, np.linspace(lo_c, hi_c, bins + 1))

    return float(mutual_info_score(s_d, c_d))


@torch.no_grad()
def compute_mi(
    components: torch.Tensor,
    ica_stacked: torch.Tensor,
    bins: int = MI_BINS,
    n_jobs: int = N_JOBS,
) -> Tuple[float, float, np.ndarray]:
    """
    Mutual information between each generated component and each ICA map.

    inputs should be CPU tensors.
    """
    _check_component_pair(components, ica_stacked)

    B, K = components.shape[:2]

    s_np = components.detach().cpu().numpy().reshape(B * K, -1)
    c_np = ica_stacked.detach().cpu().numpy().reshape(B * K, -1)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_mi_one)(s_np[i], c_np[i], bins)
        for i in range(B * K)
    )

    mi_matrix = np.array(results, dtype=np.float32).reshape(B, K)

    mi_mean = float(mi_matrix.mean())
    mi_02 = float((mi_matrix > 0.2).mean())

    return mi_mean, mi_02, mi_matrix


@torch.no_grad()
def compute_isc(
    all_components: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[float, torch.Tensor]:
    """
    Inter-subject consistency.

    all_components:
      (N, K, D, H, W)
    """
    assert all_components.ndim == 5, \
        f"Expected all_components shape (N,K,D,H,W), got {tuple(all_components.shape)}"

    N, K = all_components.shape[:2]
    V = all_components.shape[2] * all_components.shape[3] * all_components.shape[4]

    S = all_components.reshape(N, K, V)
    group_mean = S.mean(dim=0, keepdim=True)

    S_c = S - S.mean(dim=2, keepdim=True)
    mean_c = group_mean - group_mean.mean(dim=2, keepdim=True)

    num = (S_c * mean_c).sum(dim=2)
    denom = S_c.norm(dim=2) * mean_c.norm(dim=2)

    rho = num / denom.clamp(min=eps)

    isc_k = rho.mean(dim=0)

    return isc_k.mean().item(), isc_k.cpu()


@torch.no_grad()
def compute_all_metrics(
    smri: torch.Tensor,
    components: torch.Tensor,
    ica_stacked: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    compute_mi_flag: bool = True,
) -> dict:
    """
    Compute RECON, PC, PC_025, and optionally MI.

    ISC should still be computed separately after accumulating the full
    test-set component tensor.
    """
    recon = compute_recon(
        smri=smri,
        components=components,
        mask=mask,
    )

    pc, pc_025, rho = compute_pearson(
        components=components,
        ica_stacked=ica_stacked,
    )

    out = {
        "RECON": recon,
        "PC": pc,
        "PC_025": pc_025,
        "rho": rho,
    }

    if compute_mi_flag:
        mi, mi_02, mi_mat = compute_mi(
            components=components,
            ica_stacked=ica_stacked,
        )

        out["MI"] = mi
        out["MI_02"] = mi_02
        out["mi_mat"] = mi_mat

    return out