# C3D-VAE: Learning Structure–Function Relationships in the Human Brain

A Conditional 3D Variational Autoencoder that generates subject-specific
structural MRI spatial maps corresponding to fMRI ICA functional networks,
trained without ground-truth structural component supervision.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0 (for `torch.compile` and bfloat16 AMP).

---

## Project Structure

```
cvae_project/
├── main.py                         Training entry point
├── requirements.txt                Python dependencies
├── .gitignore
│
├── model/
│   └── cvae.py                     C3DVAE architecture (encoder, ICA encoder, decoder)
│
├── trainer/
│   ├── loss.py                     Four-term training objective (ELBO variant)
│   ├── metric.py                   Evaluation metrics (RECON, PC, MI, ISC)
│   └── trainer.py                  CVAETrainer with AMP, compile, monitoring
│
├── data_loader/
│   ├── datasets.py                 UKBBData — lazy load, GM mask, 72/8/20 split
│   ├── data_loaders.py             UKBB DataLoader wrapper
│   └── ukbb.csv                    Subject manifest (NOT committed — see below)
│
├── explore/
│   └── eda.py                      Exploratory analysis (run before training)
│
├── evaluate/
│   └── evaluate.py                 Full test-set evaluation + all plots
│
├── ablation/
│   └── run_ablations.py            Four ablation studies (H1–H3)
│
├── utils/
│   └── model_utils.py              Shared: load_model, is_bad, table formatters
│
├── scripts/
│   ├── slurm_train.sh              SLURM job: training
│   ├── slurm_eval.sh               SLURM job: evaluation + ablations
│   └── slurm_eda.sh                SLURM job: exploratory analysis (CPU)
│
└── config/
    ├── runs/C3DVAE.json            Hyperparameters (epochs, lr, λ weights)
    ├── models/C3DVAE.json          Architecture (latent_dim, cond_dim)
    └── data/ukbb.json              Data (batch_size, num_subjects)
```

---

## Subject Manifest (ukbb.csv)

Create `data_loader/ukbb.csv` with columns:

| Column       | Type | Description |
|-------------|------|-------------|
| `subject_id` | int  | Unique subject identifier |
| `ica_path`   | str  | Path to NIfTI file, shape (x, y, z, 53) |
| `smri_path`  | str  | Path to T1w NIfTI, shape (121, 145, 121) |

To generate GM masks: `fast -t 1 -n 3 -o subj_fast subj_T1.nii.gz`

---

## Recommended Workflow

### Step 0 — Exploratory Analysis (CPU, run first)

```bash
# Local
python explore/eda.py --num_subjects 100 --save_dir eda_results/

# SLURM
sbatch scripts/slurm_eda.sh
```

Produces intensity histograms, ICA sparsity charts, split coverage plots.
**Always run before training** to catch loading failures and normalisation issues.

### Step 1 — Training

```bash
# Local (single GPU)
python main.py \
    --config       config/runs/C3DVAE.json \
    --model_config config/models/C3DVAE.json \
    --data_config  config/data/ukbb.json

# SLURM
sbatch scripts/slurm_train.sh
```

Training saves to `saved/C3DVAE/`:
- `last.pth`, `model_best.pth` — checkpoints
- `train.csv`, `valid.csv` — per-epoch metrics (weighted, recon, pc_loss, orth, kl, isc)
- `loss_curves.png` — updated every epoch; `loss_curves_ep{N}.png` at milestones
- `epoch_{N}/` — per-epoch visualisations (every `vis_every` epochs)

### Step 2 — Evaluation

```bash
python evaluate/evaluate.py \
    --checkpoint saved/C3DVAE/model_best.pth \
    --num_subjects 140 \
    --save_dir eval_results/

# SLURM
sbatch scripts/slurm_eval.sh
```

Produces:
- `metrics.csv`  
- `rho_histogram.png`, `isc_per_component.png`, `latent_pca.png`
- `reconstruction_subj{i}.png` — sMRI / sum / residual
- `components/subj{i}/component_{k}.png` — ICA vs generated (axial + coronal)

### Step 3 — Ablation Studies

```bash
python ablation/run_ablations.py \
    --checkpoint saved/C3DVAE/model_best.pth \
    --ablation all \
    --save_dir ablation_results/
```

Produces `ablation_summary.csv`

---

## Architecture

```
sMRI Xi (1×64³)  → SMRIEncoder → μ_φ, log σ²_φ → zi ∈ ℝ^512
                                                    │
ICA c̃_ik (1×64³) → ICAEncoder (shared) → e_ik ∈ ℝ^64
                                                    │
              [zi ‖ e_ik] → ComponentDecoder (shared) → ŝ_ik (1×64³)
```

K=53 components decoded in one batched forward pass. ~17M trainable parameters.

---

## Loss Function

```
L = λ1·L_recon  +  λ2·L_PC  +  λ3·L_orth  +  λ4_w·KL
```

| Term | Default λ | Description |
|------|-----------|-------------|
| `L_recon` | 1.0 | Normalised L2: ‖Xi − Σ ŝ_ik‖²/‖Xi‖² |
| `L_PC`    | 0.5 | Spatial Pearson alignment: (1 − ρ_ik) |
| `L_orth`  | 0.1 | Gram-matrix orthogonality |
| `KL`      | 0.001 | VAE regulariser (cosine-annealed) |

---

## Evaluation Metrics (Table 1)

| Metric   | Direction | Reference (Luo et al. 2020) |
|----------|-----------|------------------------------|
| RECON    | ↓         | ≤ 0.10                       |
| PC       | ↑         | > 0.25 (significance)        |
| PC_025   | ↑         | > 0.62                       |
| MI       | ↑         | > 0.20                       |
| MI_02    | ↑         | > 0.62                       |
| ISC      | ↑         | > 0.50                       |

---

## Training Monitoring

The trainer emits warnings for:
- **NaN/Inf in loss** — halts immediately with per-term diagnostics
- **KL < 1e-4 for 5 epochs** — possible posterior collapse
- **Component std < 1e-5** — possible mode collapse (all components identical)
- **GM mask coverage < 10%** — possible GM path configuration error

`valid.csv` tracks all 6 metrics per epoch including ISC, enabling early
detection of training issues before full evaluation.

---

## Performance Features

- **AMP (bfloat16/float16)**: ~2× memory reduction, ~1.5× speedup
- **torch.compile**: kernel fusion (first step is slow; disable with `"compile": false`)
- **Lazy loading**: NIfTI files loaded once on first access, cached in RAM
- **Vectorised metrics**: RECON, PC, ISC all O(1) matrix ops over (B, K, V)
- **Parallel MI**: joblib parallelises all B×K MI calls across CPU cores

---

## Key Design Decisions

**Why Pearson, not cosine similarity?**
Pearson = cosine on mean-centred vectors. ICA maps are Z-scored (≈zero mean);
generated components have non-zero intensity offsets from sMRI background signal.
Cosine is sensitive to these offsets; Pearson is not. See `trainer/loss.py`.

**Why encode sMRI, not the 53 ICA maps?**
The sMRI contains the subject's full anatomical fingerprint as a dense signal.
The 53 ICA maps are already low-dimensional sparse statistical maps.
See paper Section 4.2 and `model/cvae.py`.

**Why shared decoder?**
Sharing weights across all 53 components enforces the biological universality
prior and reduces parameters by 53×. Components differ only via their ICA
conditioning vector `e_ik`.
