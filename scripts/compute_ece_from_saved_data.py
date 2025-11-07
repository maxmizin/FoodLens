#!/usr/bin/env python3
"""
Compute ECE (Expected Calibration Error) from saved logits and labels.
This will verify the actual ECE values reported in the paper.
"""

import numpy as np
import pandas as pd
from scipy.special import softmax
import json
from pathlib import Path

# Paths
RESULTS_DIR = Path('results/verified_final')
MODEL_DIR = Path('models/final_backbone')

def compute_ece(y_true, y_pred, confidences, n_bins=10):
    """
    Compute Expected Calibration Error.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        confidences: confidence scores (max probability)
        n_bins: number of bins
    
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = (y_pred == y_true)
    ece = 0
    bin_metrics = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_metrics.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'prop': prop_in_bin,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin
            })
    
    return ece, bin_metrics

print("="*80)
print("ECE VERIFICATION FROM SAVED DATA")
print("="*80)

# Load calibration parameters
calibration_file = MODEL_DIR / 'calibration.json'
if calibration_file.exists():
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    temperature = calib_data['temperature']
    print(f"\nLoaded calibration: T = {temperature:.6f}")
else:
    print("\n[ERROR] Calibration.json not found!")
    temperature = 1.0

# Load data
print("\nLoading saved data...")
test_probs = np.load(RESULTS_DIR / 'test_probs.npy')  # Already temperature-scaled
test_preds_df = pd.read_csv(RESULTS_DIR / 'test_preds.csv')

print(f"  Test probs shape: {test_probs.shape}")
print(f"  Test preds shape: {test_preds_df.shape}")

# Extract predictions and labels
test_preds = test_preds_df['prediction'].values
test_labels = test_preds_df['label'].values
test_confidences = test_probs.max(axis=1)

print(f"  Test set size: {len(test_labels)}")

print(f"\nConfidences: min={test_confidences.min():.3f}, max={test_confidences.max():.3f}, mean={test_confidences.mean():.3f}")

# Compute ECE
print("\nComputing ECE with 10 bins...")
ece, bin_metrics = compute_ece(test_labels, test_preds, test_confidences, n_bins=10)

print(f"\n{'='*80}")
print(f"RESULT: ECE = {ece:.6f}")
print(f"{'='*80}")

# Show bin breakdown
print("\nBin-by-bin calibration:")
for bm in bin_metrics:
    if bm['prop'] > 0.01:  # Only show bins with >1% of data
        print(f"  [{bm['bin_lower']:.2f}, {bm['bin_upper']:.2f}]: "
              f"prop={bm['prop']:.3f}, acc={bm['accuracy']:.3f}, conf={bm['confidence']:.3f}, diff={abs(bm['accuracy']-bm['confidence']):.3f}")

# Compare to paper values
print("\n" + "="*80)
print("COMPARISON TO PAPER VALUES")
print("="*80)
print(f"\nPAPER CLAIMS:")
print(f"  1. Abstract: ECE=0.014 (with temp scaling)")
print(f"  2. Section 4.3: ECE drops from 0.042 to 0.014")
print(f"  3. Table 3: FoodLens ECE=0.014")
print(f"  4. Section 5.6.2 caption: ECE=0.476 (with abstention)")
print(f"\nACTUAL ECE FROM CODE:")
print(f"  Computed ECE = {ece:.6f}")
print(f"\nVERDICT:")
if abs(ece - 0.014) < 0.001:
    print("  ✓ MATCHES 0.014 claimed in abstract, Section 4.3, Table 3")
elif abs(ece - 0.476) < 0.001:
    print("  ⚠ MATCHES 0.476 from Section 5.6.2 caption")
else:
    print(f"  X DOES NOT MATCH either reported value!")
    print(f"  This is a CRITICAL DISCREPANCY requiring correction.")

print("\n" + "="*80)
print("="*80)

