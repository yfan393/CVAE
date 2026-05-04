"""
explore/eda.py
─────────────────────────────────────────────────────────────────
Exploratory Data Analysis (EDA) for the UK Biobank C3D-VAE dataset.

Produces (saved to eda_results/):
  smri_stats.csv          — per-subject sMRI intensity statistics
  ica_stats.csv           — per-subject, per-component ICA statistics
  smri_distribution.png   — histogram of sMRI voxel intensities
  ica_distribution.png    — histogram of ICA Z-score values (all components)
  smri_mean_std.png       — population mean and std sMRI volume
  ica_mean_activation.png — mean max-activation per component (bar chart)
  ica_sparsity.png        — fraction of non-zero voxels per component
  subject_coverage.png    — number of subjects per split

Run BEFORE training to:
  • Verify data loading works
  • Check intensity ranges (detect un-normalised data)
  • Identify any anomalous subjects (extreme outliers)
  • Understand ICA sparsity (informs noise threshold choice)

Usage:
  python explore/eda.py --num_subjects 100 --save_dir eda_results/
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nibabel.nifti1 import Nifti1Image
import nibabel as nib
from nilearn import plotting

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader.datasets import UKBBData

SMRI_AFFINE = np.array([
    [-1.5,  0.0,  0.0,   90.0],
    [ 0.0,  1.5,  0.0, -126.0],
    [ 0.0,  0.0,  1.5,  -72.0],
    [ 0.0,  0.0,  0.0,    1.0]
], dtype=np.float32)

def analyse_smri(dataset: UKBBData, save_dir: Path) -> pd.DataFrame:
    """
    Per-subject sMRI statistics: mean, std, min, max, fraction non-zero.
    Helps detect:
      • Un-normalised intensities (should be roughly centred after mean sub)
      • All-zero subjects (loading failure)
      • Extreme outliers (poor registration / skull not stripped)
    """
    records = []
    vols    = []

    for i in range(len(dataset)):
        smri, _, _ = dataset[i]
        v = smri[0].numpy()   # (64,64,64)
        vols.append(v)
        records.append({
            'subject_id': dataset.subjects[i],
            'mean':       float(v.mean()),
            'std':        float(v.std()),
            'min':        float(v.min()),
            'max':        float(v.max()),
            'frac_nonzero': float((v != 0).mean()),
        })

    df = pd.DataFrame(records)
    df.to_csv(save_dir / 'smri_stats.csv', index=False)

    print(f"\n=== sMRI statistics (N={len(df)}) ===")
    print(df[['mean','std','min','max','frac_nonzero']].describe().to_string())

    # Histogram of all voxel intensities (sampled for speed)
    sample = np.concatenate([v.ravel()[::100] for v in vols])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sample, bins=80, color='steelblue', alpha=0.85, edgecolor='white')
    ax.set_xlabel('sMRI voxel intensity (after resampling, before mean-sub)')
    ax.set_ylabel('Count (sampled)')
    ax.set_title(f'sMRI intensity distribution  (N={len(df)} subjects, 1:100 sample)')
    plt.tight_layout()
    fig.savefig(save_dir / 'smri_distribution.png', dpi=150)
    plt.close(fig)
    print("  → smri_distribution.png")

    # Population mean and std volume
    all_v  = np.stack(vols, axis=0)    # (N, 64, 64, 64)
    mu_vol = all_v.mean(axis=0)
    sd_vol = all_v.std(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    plotting.plot_epi(
        Nifti1Image(mu_vol.astype(np.float32), SMRI_AFFINE),
        axes=axes[0], title='Population mean sMRI',
        display_mode='z', cut_coords=8, colorbar=True)
    plotting.plot_epi(
        Nifti1Image(sd_vol.astype(np.float32), SMRI_AFFINE),
        axes=axes[1], title='Population std sMRI',
        display_mode='z', cut_coords=8, colorbar=True)
    plt.tight_layout()
    fig.savefig(save_dir / 'smri_mean_std.png', dpi=150)
    plt.close(fig)
    print("  → smri_mean_std.png")

    return df


def analyse_ica(dataset: UKBBData, save_dir: Path) -> pd.DataFrame:
    """
    Per-subject, per-component ICA statistics.
    Helps detect:
      • Components with very low activation (possibly noise-only)
      • Components with extreme Z-scores (potentially artefacts)
      • Highly sparse maps (check threshold is appropriate)
    """
    K       = 53
    records = []
    max_act = np.zeros((len(dataset), K))
    sparsity= np.zeros((len(dataset), K))
    all_z   = []

    for i in range(len(dataset)):
        _, ica, _ = dataset[i]   # (53, 64, 64, 64)
        ica_np    = ica.numpy()
        for k in range(K):
            v = ica_np[k]
            max_act[i, k]  = np.abs(v).max()
            sparsity[i, k] = (v != 0).mean()
            records.append({
                'subject_id': dataset.subjects[i],
                'component': k,
                'max_abs':   float(np.abs(v).max()),
                'mean':      float(v.mean()),
                'std':       float(v.std()),
                'sparsity':  float((v != 0).mean()),
            })
        # Sample voxels for distribution plot
        all_z.append(ica_np[:, ::50, ::50, ::50].ravel())
    
    df = pd.DataFrame(records)
    df.to_csv(save_dir / 'ica_stats.csv', index=False)

    print(f"\n=== ICA statistics (N={len(dataset)}, K={K}) ===")
    print(df[['max_abs','mean','std','sparsity']].describe().to_string())

    # ICA Z-score distribution
    sample_z = np.concatenate(all_z)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sample_z[np.abs(sample_z) < 10], bins=100,
            color='teal', alpha=0.85, edgecolor='white')
    ax.axvline( 0.2, color='red', linestyle='--', lw=1, label='Threshold +0.2')
    ax.axvline(-0.2, color='red', linestyle='--', lw=1, label='Threshold -0.2')
    ax.set_xlabel('ICA Z-score (|Z| < 10 shown)')
    ax.set_ylabel('Count (1:50³ sample)')
    ax.set_title('ICA Z-score distribution — peak near 0 expected (sparse maps)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_dir / 'ica_distribution.png', dpi=150)
    plt.close(fig)
    print("  → ica_distribution.png")

    # Mean max-activation per component
    mean_max = max_act.mean(axis=0)   # (K,)
    fig, ax  = plt.subplots(figsize=(18, 4))
    colors   = ['#F44336' if v < 1.0 else '#2196F3' for v in mean_max]
    ax.bar(range(K), mean_max, color=colors, alpha=0.85)
    ax.axhline(1.0, color='orange', linestyle='--', lw=1, label='Z=1.0 reference')
    ax.set_xlabel('ICA component index k')
    ax.set_ylabel('Mean max |Z| across subjects')
    ax.set_title('Mean max activation per ICA component\n'
                 'Red bars (< 1.0) may indicate noise-only or low-signal components')
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_dir / 'ica_mean_activation.png', dpi=150)
    plt.close(fig)
    print("  → ica_mean_activation.png")

    # Sparsity per component
    mean_spar = sparsity.mean(axis=0)
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.bar(range(K), mean_spar * 100, color='steelblue', alpha=0.85)
    ax.set_xlabel('ICA component index k')
    ax.set_ylabel('Mean % non-zero voxels (post-threshold)')
    ax.set_title('ICA component sparsity after |Z| < 0.2 suppression\n'
                 'Lower = more focused spatial activation')
    plt.tight_layout()
    fig.savefig(save_dir / 'ica_sparsity.png', dpi=150)
    plt.close(fig)
    print("  → ica_sparsity.png")

    return df


def analyse_splits(save_dir: Path):
    """Show subject counts per split to verify 72/8/20 distribution."""
    import pandas as pd
    from data_loader.datasets import CSV_PATH
    df = pd.read_csv(CSV_PATH, index_col=0)
    N  = len(pd.unique(df['subject_id']))

    splits = {}
    for sp in ['train', 'valid', 'test']:
        ds = UKBBData(split=sp)
        splits[sp] = len(ds)

    print(f"\n=== Split distribution (total unique subjects: {N}) ===")
    for sp, n in splits.items():
        print(f"  {sp:<8}: {n:>5}  ({100*n/N:.1f}%)")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(list(splits.keys()), list(splits.values()),
           color=['steelblue', 'orange', 'green'], alpha=0.85)
    for sp, n in splits.items():
        ax.text(list(splits.keys()).index(sp), n + 2, str(n),
                ha='center', fontsize=10)
    ax.set_ylabel('Number of subjects')
    ax.set_title('Subject count per split (target: 72/8/20)')
    plt.tight_layout()
    fig.savefig(save_dir / 'subject_coverage.png', dpi=150)
    plt.close(fig)
    print("  → subject_coverage.png")


def main():
    parser = argparse.ArgumentParser(description='EDA for C3D-VAE dataset')
    parser.add_argument('--num_subjects', type=int, default=100,
                        help='Number of training subjects to analyse')
    parser.add_argument('--save_dir', default='eda_results')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training dataset ({args.num_subjects} subjects) …")
    ds = UKBBData(split='train', num_subjects=args.num_subjects)

    smri_df = analyse_smri(ds, save_dir)
    ica_df  = analyse_ica(ds,  save_dir)
    analyse_splits(save_dir)

    # Summary report
    print(f"\n{'═'*55}")
    print(f"  EDA SUMMARY")
    print(f"{'═'*55}")
    print(f"  Subjects analysed   : {len(ds)}")
    print(f"  sMRI mean intensity : {smri_df['mean'].mean():.4f} ± {smri_df['mean'].std():.4f}")
    print(f"  sMRI frac non-zero  : {smri_df['frac_nonzero'].mean():.3f}")
    outliers = smri_df[smri_df['std'] < smri_df['std'].quantile(0.05)]
    if len(outliers):
        print(f"  ⚠  {len(outliers)} subjects with very low std — check: "
              f"{outliers['subject_id'].tolist()[:5]}")
    zero_subs = smri_df[smri_df['frac_nonzero'] < 0.01]
    if len(zero_subs):
        print(f"  ⚠  {len(zero_subs)} near-zero sMRI subjects — possible "
              f"loading failure: {zero_subs['subject_id'].tolist()}")
    print(f"  ICA mean sparsity   : {ica_df['sparsity'].mean():.4f}")
    low_act = ica_df.groupby('component')['max_abs'].mean()
    print(f"  ICA components with mean max_abs < 1.0: "
          f"{(low_act < 1.0).sum()} / 53")
    print(f"{'═'*55}")
    print(f"\nAll EDA outputs → {save_dir}/")


if __name__ == '__main__':
    main()
