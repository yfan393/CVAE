"""
utils/model_utils.py
─────────────────────────────────────────────────────────────────
Shared utilities used by trainer, evaluate, and ablation modules.

Centralising here eliminates duplicate definitions that previously
appeared across trainer.py, evaluate.py, and run_ablations.py.

Contents
────────
load_model      — load C3DVAE from checkpoint (single definition)
is_bad          — NaN/Inf detector
format_eval_table     — formatted string table matching paper Table 1
format_ablation_table — formatted string table matching paper ablation table
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (single definition, used by evaluate and ablation)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: str, device: torch.device):
    """
    Load a C3DVAE from a checkpoint written by CVAETrainer
    Returns the model in eval mode on `device`.
    """
    from model.cvae import C3DVAE   # deferred to avoid circular import

    ckpt  = torch.load(path, map_location=device)
    cfg   = ckpt.get('config', {})
    model = C3DVAE(
        num_components = cfg.get('num_components', 53),
        latent_dim     = cfg.get('latent_dim',     512),
        cond_dim       = cfg.get('cond_dim',        64),
        dropout        = 0.0,   # always disable dropout at inference
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Tensor health check
# ─────────────────────────────────────────────────────────────────────────────

def is_bad(t: torch.Tensor) -> bool:
    """True if tensor contains any NaN or Inf value."""
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


# ─────────────────────────────────────────────────────────────────────────────
# Formatted metric tables  (paper Table 1 format)
# ─────────────────────────────────────────────────────────────────────────────

# Reference thresholds from Luo et al. (2020) and paper Table 1
_METRIC_REFS: Dict[str, tuple] = {
    'RECON':  ('≤ 0.10', '↓'),
    'PC':     ('> 0.25', '↑'),
    'PC_025': ('> 0.62', '↑'),
    'MI':     ('> 0.20', '↑'),
    'MI_02':  ('> 0.62', '↑'),
    'ISC':    ('> 0.50', '↑'),
}


def format_eval_table(metrics: Dict[str, float], N: int) -> str:
    """
    Print a formatted table matching paper Table 1.

    Example:
      Evaluation metrics  (test set, N = 140)
      ────────────────────────────────────────────────────
        Metric       Value        Ref       Dir
      ────────────────────────────────────────────────────
        RECON        0.07234   ≤ 0.10      ↓
        PC           0.31102   > 0.25      ↑
        ...
      ────────────────────────────────────────────────────
    """
    sep   = '─' * 54
    lines = [f"\nEvaluation metrics  (test set, N = {N:,})",
             sep,
             f"  {'Metric':<12} {'Value':>10}   {'Ref':<10} {'Dir'}",
             sep]
    for k, (ref, direction) in _METRIC_REFS.items():
        if k in metrics:
            lines.append(f"  {k:<12} {metrics[k]:>10.5f}   {ref:<10} {direction}")
    lines.append(sep)
    return '\n'.join(lines)


def format_ablation_table(summary_df: pd.DataFrame) -> str:
    """
    Print a formatted ablation results table.

    Input: DataFrame with columns [ablation, RECON, PC, PC_025, MI, MI_02, ISC]
    """
    cols    = ['ablation', 'RECON', 'PC', 'PC_025', 'MI', 'MI_02', 'ISC']
    present = [c for c in cols if c in summary_df.columns]
    df      = summary_df[present].copy()
    for col in present:
        if col != 'ablation':
            df[col] = df[col].apply(lambda x: f'{float(x):.5f}')

    sep   = '─' * 88
    lines = ['\nAblation results', sep]
    lines.append('  '.join(f'{c:<18}' if c == 'ablation' else f'{c:>10}'
                            for c in present))
    lines.append(sep)
    for _, row in df.iterrows():
        lines.append('  '.join(
            f'{str(row[c]):<18}' if c == 'ablation' else f'{str(row[c]):>10}'
            for c in present))
    lines.append(sep)
    return '\n'.join(lines)
