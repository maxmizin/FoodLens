#!/usr/bin/env python3
"""
Generate calibration reliability diagram with STANDARDIZED ECE calculation.
Uses the SAME 10-bin fixed-width method as validation set calibration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.special import softmax

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "verified_final"
MODEL_DIR = PROJECT_ROOT / "models" / "final_backbone"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figs"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Global style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.25,
})

def compute_ece_standard(y_true, y_pred, confidences, n_bins=10):
    """
    STANDARD ECE calculation with fixed 10 equal-width bins.
    Matches the validation set calibration procedure.
    
    This is the AUTHORITATIVE ECE calculation method used throughout the paper.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = (y_pred == y_true)
    ece = 0.0
    bin_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # CRITICAL: Use > and <= to match validation calibration
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_data.append({
                'bin_center': (bin_lower + bin_upper) / 2,
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'confidence': avg_confidence_in_bin,
                'accuracy': accuracy_in_bin,
                'count': in_bin.sum(),
                'proportion': prop_in_bin
            })
    
    return ece, bin_data

def load_data():
    """Load test set predictions with temperature scaling applied."""
    # Load temperature-scaled probabilities (T=1.45 already applied)
    probs_path = RESULTS_DIR / 'test_probs.npy'
    preds_path = RESULTS_DIR / 'test_preds.csv'
    
    if probs_path.exists() and preds_path.exists():
        probs = np.load(probs_path)
        df = pd.read_csv(preds_path)
        y_true = df['label'].values
        y_pred = df['prediction'].values
        confidences = probs.max(axis=1)
        
        print(f"  Loaded {len(y_true)} test samples")
        print(f"  Confidence range: [{confidences.min():.3f}, {confidences.max():.3f}]")
        print(f"  Mean confidence: {confidences.mean():.3f}")
        
        return y_true, y_pred, probs, confidences
    else:
        raise FileNotFoundError(f"Required files not found in {RESULTS_DIR}")

def load_uncalibrated_data():
    """Load uncalibrated test set predictions (before temperature scaling)."""
    logits_path = RESULTS_DIR / 'test_logits.npy'
    preds_path = RESULTS_DIR / 'test_preds.csv'
    
    if logits_path.exists():
        logits = np.load(logits_path)
        probs_uncalibrated = softmax(logits, axis=1)
        df = pd.read_csv(preds_path)
        y_true = df['label'].values
        y_pred_uncal = probs_uncalibrated.argmax(axis=1)
        confidences_uncal = probs_uncalibrated.max(axis=1)
        
        return y_true, y_pred_uncal, probs_uncalibrated, confidences_uncal
    else:
        return None, None, None, None

def main():
    print("="*80)
    print("GENERATING STANDARDIZED CALIBRATION RELIABILITY DIAGRAM")
    print("="*80)
    
    # Load calibration info
    import json
    calib_file = MODEL_DIR / 'calibration.json'
    if calib_file.exists():
        with open(calib_file, 'r') as f:
            calib_info = json.load(f)
        temperature = calib_info.get('temperature', 1.45)
        print(f"\nCalibration temperature: T = {temperature:.4f}")
    else:
        temperature = 1.45
        print(f"\n[WARNING] Using default temperature T = {temperature}")
    
    # Load calibrated data
    print("\n[1] Loading calibrated test set data...")
    y_true, y_pred, probs_cal, conf_cal = load_data()
    
    # Load uncalibrated data
    print("\n[2] Loading uncalibrated test set data...")
    y_true_uncal, y_pred_uncal, probs_uncal, conf_uncal = load_uncalibrated_data()
    
    # Compute ECE - NO ABSTENTION (calibrated)
    print("\n[3] Computing ECE (calibrated, no abstention)...")
    ece_calibrated, bins_calibrated = compute_ece_standard(y_true, y_pred, conf_cal, n_bins=10)
    print(f"  ECE (after T={temperature:.2f}): {ece_calibrated:.6f}")
    
    # Compute ECE - UNCALIBRATED (if available)
    if conf_uncal is not None:
        print("\n[4] Computing ECE (uncalibrated)...")
        ece_uncalibrated, bins_uncalibrated = compute_ece_standard(
            y_true_uncal, y_pred_uncal, conf_uncal, n_bins=10)
        print(f"  ECE (before calibration): {ece_uncalibrated:.6f}")
    else:
        ece_uncalibrated = None
        bins_uncalibrated = None
        print("\n[4] Uncalibrated data not available, skipping")
    
    # Compute ECE - WITH ABSTENTION
    print("\n[5] Computing ECE (with abstention tau>=0.5)...")
    tau = 0.5
    keep = conf_cal >= tau
    n_kept = keep.sum()
    print(f"  Samples above tau={tau}: {n_kept}/{len(y_true)} ({100*n_kept/len(y_true):.1f}%)")
    
    if n_kept > 0:
        ece_abstain, bins_abstain = compute_ece_standard(
            y_true[keep], y_pred[keep], conf_cal[keep], n_bins=10)
        print(f"  ECE (with abstention): {ece_abstain:.6f}")
    else:
        ece_abstain = ece_calibrated
        bins_abstain = bins_calibrated
        print("  [WARNING] No samples above threshold, using no-abstention values")
    
    # Create figure
    print("\n[6] Generating figure...")
    fig = plt.figure(figsize=(7, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[1, 0], sharex=ax_main)
    
    # Main calibration plot
    ax_main.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, 
                 label='Perfect Calibration', zorder=1)
    
    # Extract bin centers and values for plotting
    conf_centers_cal = [b['bin_center'] for b in bins_calibrated]
    acc_values_cal = [b['accuracy'] for b in bins_calibrated]
    
    ax_main.plot(conf_centers_cal, acc_values_cal, 'o-', color='#3498db', 
                linewidth=2.5, markersize=8, 
                label=f'No Abstention (ECE={ece_calibrated:.3f})', 
                alpha=0.85, zorder=3)
    
    if bins_abstain != bins_calibrated:
        conf_centers_abs = [b['bin_center'] for b in bins_abstain]
        acc_values_abs = [b['accuracy'] for b in bins_abstain]
        ax_main.plot(conf_centers_abs, acc_values_abs, 's-', color='#2ecc71', 
                    linewidth=2.5, markersize=8, 
                    label=f'With Abstention tau>={tau} (ECE={ece_abstain:.3f})', 
                    alpha=0.85, zorder=2)
    
    ax_main.set_ylabel('Empirical Accuracy', fontweight='bold', fontsize=11)
    ax_main.set_title('Calibration Reliability Diagram (Test Set)', 
                      fontweight='bold', fontsize=12, pad=15)
    ax_main.set_xlim([0, 1])
    ax_main.set_ylim([0, 1])
    ax_main.grid(alpha=0.25, linestyle='--')
    ax_main.legend(loc='lower right', framealpha=0.95, fontsize=9)
    ax_main.set_aspect('equal')
    plt.setp(ax_main.get_xticklabels(), visible=False)
    
    # Histogram showing confidence distribution
    ax_hist.hist(conf_cal, bins=30, alpha=0.6, color='#95a5a6', 
                edgecolor='black', linewidth=0.5)
    ax_hist.axvline(x=tau, color='#2ecc71', linestyle='--', linewidth=2, 
                    alpha=0.7, label=f'tau={tau}')
    ax_hist.set_xlabel('Predicted Confidence', fontweight='bold', fontsize=11)
    ax_hist.set_ylabel('Count', fontweight='bold', fontsize=9)
    ax_hist.set_xlim([0, 1])
    ax_hist.grid(axis='y', alpha=0.25)
    ax_hist.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # Save
    out_png = OUTPUT_DIR / 'calibration_reliability_fixed.png'
    out_pdf = OUTPUT_DIR / 'calibration_reliability_fixed.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved {out_png.name}")
    print(f"  [OK] Saved {out_pdf.name}")
    
    # Save ECE values to CSV for reference
    ece_summary = pd.DataFrame([
        {'metric': 'ECE_uncalibrated', 'value': ece_uncalibrated if ece_uncalibrated else np.nan},
        {'metric': 'ECE_calibrated_no_abstention', 'value': ece_calibrated},
        {'metric': 'ECE_calibrated_with_abstention', 'value': ece_abstain},
        {'metric': 'temperature', 'value': temperature},
        {'metric': 'abstention_threshold', 'value': tau},
    ])
    ece_csv = TABLE_DIR / 'ece_test_set.csv'
    ece_summary.to_csv(ece_csv, index=False)
    print(f"  [OK] Saved ECE summary to {ece_csv.name}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTest Set ECE (10-bin standard method):")
    if ece_uncalibrated:
        print(f"  Before calibration:     ECE = {ece_uncalibrated:.6f}")
    print(f"  After calibration (T={temperature:.2f}): ECE = {ece_calibrated:.6f}")
    print(f"  With abstention (tau>={tau}):  ECE = {ece_abstain:.6f}")
    print(f"\n[SUCCESS] Figure generated with standardized ECE calculation")
    print(f"[SUCCESS] Matches validation set calibration method (10 equal-width bins)")
    print("="*80)

if __name__ == '__main__':
    main()

