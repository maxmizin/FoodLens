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

Generate checksums:
```bash
python scripts/export/write_checksums.py
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
- **Frozen splits**: `data/frozen_splits/` with SHA256 checksums
- **Full dataset**: Deposited on KiltHub with DOI
- **Verify**: Use SHA256SUMS.txt to verify integrity

See `data/README_DATA.md` for details.

## Models
- **Configs and tokenizers**: Included in `models/`
- **Model weights**: Hosted separately with DOI links in KiltHub record
- **Size**: ~500MB per checkpoint

See `models/README_MODELS.md` for details.

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
- `paper/paper.pdf` - Main paper
- `paper/tex/main.tex` - LaTeX source
- `paper/README_figure_map.md` - Figure provenance

## Results and Audits
- `results/audits/finality_report_complete.md` - Complete audit report
- `results/model_comparison_summary.csv` - Summary metrics
- `results/per_class_comparison_summary.csv` - Per-class metrics

## Figures
All figures referenced in paper:
- `paper/figs/model_comparison_bar_chart.png`
- `paper/figs/per_class_comparison_bar_chart.png`
- `paper/figs/comprehensive_model_comparison.png`

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
- **Data**: To be specified at KiltHub deposition

# Contact

Email: mizinmax22@gmail.com

# Repository Structure

```
ModelExpansion/
├── README.md                    # This file
├── CONTRIBUTING.md              # Contribution guidelines
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT license
├── LICENSE_PAPER.txt            # CC BY 4.0 license
├── CITATION.cff                 # Citation metadata
├── SHA256SUMS.txt               # File integrity checksums
│
├── data/                        # Data files
│   ├── frozen_splits/          # Train/val/test splits
│   ├── samples/                 # Sample data
│   └── README_DATA.md           # Data documentation
│
├── models/                      # Model checkpoints
│   ├── final_backbone/         # Trained DeBERTa model
│   ├── regex/                  # Baseline patterns
│   └── README_MODELS.md        # Model documentation
│
├── results/                     # Evaluation results
│   ├── audits/                 # Audit reports
│   ├── final_eval/             # Final evaluation
│   └── abstention_comparison/  # Abstention analysis
│
├── paper/                       # Complete paper package
│   ├── tex/                    # LaTeX source (main.tex, refs/)
│   ├── figs/                   # All figures for paper
│   ├── tables/                 # All tables for paper
│   ├── snippets/               # Method comparisons
│   ├── Makefile                # Build rules
│   ├── README_figure_map.md    # Figure provenance
│   └── paper.pdf               # Compiled paper (after build)
│
├── scripts/                     # Python scripts
│   ├── paper/                  # Paper utilities
│   ├── export/                 # Export scripts
│   └── [evaluation scripts]
│
├── docs/                        # Documentation
└── dist/                        # Distribution packages
```

# Build Status

Paper build PASS - 2025-10-27 16:00
