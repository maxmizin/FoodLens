# View the Research Paper
🔗 Click the image to read the full paper

[![FoodLens paper preview](./FoodLens%20-%20Preview.png)](./FoodLens%20-%20Maximilian%20Mizin.pdf)

# FoodLens: Selective Prediction for Nut Allergen Detection

Calibrated DeBERTa-v3-base for nut allergen classification from ingredient text, featuring confidence-based selective abstention.

![Python](https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

# TLDR

FoodLens classifies food products into three allergen risk categories: safe, trace, contain. With calibrated abstention at threshold τ=0.80, we achieve 0.789 macro F1 at 99.17% coverage, abstaining from only 2 of 240 test samples.

# Results

## Performance on Test Set

| Method | Macro F1 | Accuracy | Coverage | Abstain Rate |
|--------|----------|----------|----------|--------------|
| Regex Baseline | 0.709 | 0.946 | 1.000 | 0.000 |
| DeBERTa (Non-Abstaining) | 0.767 | 0.950 | 1.000 | 0.000 |
| **FoodLens (Abstaining τ=0.80)** | **0.789** | **0.954** | **0.992** | 0.008 |

## Per-Class F1 Scores
- Safe: 0.979
- Trace: 0.545
- Contain: 0.842

## Calibration
- Method: Temperature scaling (T=10.0)
- ECE: 0.0136

# Quickstart

Set up environment:
```bash
pip install -r requirements.txt
git lfs install
```

Build paper:
```bash
make -C paper pdf
```

# Reproduce our paper results

Evaluation is frozen and verified with provided CSV files and SHA256 checksums in `results/audits/`.

## Verify Data Integrity
```bash

# Windows
certutil -hashfile data/frozen_splits/test.csv SHA256
```

## Run Evaluation Pipeline
```bash
# Evaluate frozen model (no retraining)
python scripts/freeze_and_evaluate_best.py --use-frozen-splits --no-train

# Run abstention analysis
python scripts/complete_evaluate_pipeline.py --abstain --taus 0.60 0.70 0.80 0.90

# Optimize abstention thresholds
python scripts/optimize_abstention_methods.py --frozen --report results/abstention_recheck
```

## Full Training (Optional)
To retrain from scratch:
```bash
python scripts/retrain_complete_pipeline.py
```

# Data and Models

## Data
- **Frozen splits**: `data/frozen_splits/` (train/val/test CSVs) with `SHA256SUMS.txt`
- **Samples**: `data/samples/` provides example inputs used in the paper
- **Integrity check**: recompute hashes with `certutil -hashfile <file> SHA256`

## Models
- `models/final_backbone/`: frozen DeBERTa config, tokenizer, and calibration JSON
- `models/regex/`: baseline pattern list used for comparison experiments
- Large checkpoint binaries are tracked via Git LFS; see `.gitattributes`

# Paper Build

```bash
# Build PDF
make -C paper pdf

# Clean build artifacts
make -C paper clean

# Validate all references
make -C paper check
```

Paper output: `paper/paper.pdf`

# Artifact Map

## Paper
- `paper/tex/main.tex` – LaTeX source
- `paper/figs/` – figure assets referenced from the manuscript
- `paper/tables/` – CSV and TeX tables imported into the paper

## Results
- `results/final_eval/` – frozen evaluation outputs for abstaining and non-abstaining models
- `results/verified_final/` – selective prediction metrics with bootstrap confidence intervals
- `results/model_comparison_summary.csv` – headline metrics for three model variants
- `results/per_class_comparison_summary.csv` – per-class macro metrics

# Citation

```bibtex
@software{foodlens2025,
  title={FoodLens: Selective Prediction for Nut Allergen Detection},
  author={Mizin, Max},
  year={2025},
  version={1.0.0},
  license={MIT}
}

@article{foodlens2025paper,
  title={FoodLens: Reliable Allergen Risk Assessment},
  author={Mizin, Max},
  year={2025}
}
```

# License

- **Code**: MIT (see `LICENSE`)
- **Paper**: CC BY 4.0 (see `LICENSE_PAPER.txt`)
- **Data**: Not redistributed here; contact maintainers for access details

# Contact

Email: mizinmax22@gmail.com

# Repository Structure

```
FoodLens/
├── README.md                  # Project overview and replication guide
├── CONTRIBUTING.md            # Contribution guidelines
├── requirements.txt           # Python dependencies
├── CITATION.cff               # Citation metadata
├── LICENSE                    # MIT license for code
├── LICENSE_PAPER.txt          # CC BY 4.0 license for manuscript
│
├── data/
│   ├── frozen_splits/         # Train/val/test CSVs + SHA256SUMS.txt
│   └── samples/               # Example ingredient snippets
│
├── models/
│   ├── final_backbone/        # Final DeBERTa configuration + tokenizer
│   └── regex/                 # Baseline pattern file
│
├── results/
│   ├── final_eval/            # Frozen evaluation outputs (CSV + confusion matrices)
│   ├── verified_final/        # Bootstrap summaries for released models
│   ├── model_comparison_summary.csv
│   └── per_class_comparison_summary.csv
│
├── dist/
│   ├── fairness_analysis_table.csv
│   ├── fairness_table.tex
│   ├── full_test_predictions.csv
│   └── harder_subset_*        # CSVs for auxiliary analyses
│
├── paper/
│   ├── tex/                   # LaTeX source (main.tex)
│   ├── figs/                  # Figures referenced in the paper
│   └── tables/                # Tables imported by LaTeX
│
└── scripts/
    ├── complete_evaluate_pipeline.py
    ├── compute_ece_from_saved_data.py
    ├── config.py
    ├── freeze_and_evaluate_best.py
    ├── optimize_abstention_methods.py
    ├── retrain_complete_pipeline.py
    ├── verify_accuracy_from_scratch.py
    └── verify_all_metrics.py
```

# Build Status

Paper build PASS - 2025-10-27 16:00
