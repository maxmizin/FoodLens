#!/usr/bin/env python3
"""
Generate proper Safety Curve: Accuracy vs Coverage with abstention.

This script computes and plots a robust accuracy-coverage curve for the FoodLens
abstention system, using real prediction data with bootstrap confidence intervals.

Author: Max Mizin
Date: 2025-10-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import sys

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "verified_final"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figs"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"

# Ensure output dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RNG_SEED = 1337
np.random.seed(RNG_SEED)


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load predictions, labels, and confidence scores.
    
    Returns:
        y_true: (N,) true labels
        y_pred: (N,) predicted labels  
        p_max: (N,) maximum softmax probability per prediction
    """
    # Try CSV first
    csv_path = RESULTS_DIR / "test_preds.csv"
    probs_path = RESULTS_DIR / "test_probs.npy"
    
    if not csv_path.exists() or not probs_path.exists():
        print(f"ERROR: Required files not found!")
        print(f"  Expected: {csv_path}")
        print(f"  Expected: {probs_path}")
        sys.exit(1)
    
    # Load predictions and labels
    df = pd.read_csv(csv_path)
    if 'prediction' not in df.columns or 'label' not in df.columns:
        print(f"ERROR: CSV must have 'prediction' and 'label' columns")
        print(f"  Found: {df.columns.tolist()}")
        sys.exit(1)
    
    y_pred = df['prediction'].values
    y_true = df['label'].values
    
    # Load probabilities and compute max confidence
    probs = np.load(probs_path)
    if probs.shape[0] != len(y_true):
        print(f"ERROR: Mismatch between CSV ({len(y_true)}) and probs ({probs.shape[0]})")
        sys.exit(1)
    
    p_max = probs.max(axis=1)
    
    print(f"[OK] Loaded {len(y_true)} predictions")
    print(f"     Labels: {np.unique(y_true)}")
    print(f"     Confidence range: [{p_max.min():.4f}, {p_max.max():.4f}]")
    
    return y_true, y_pred, p_max


def compute_accuracy_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_max: np.ndarray,
    thresholds: np.ndarray,
    n_bootstrap: int = 1000
) -> pd.DataFrame:
    """
    Compute accuracy on kept predictions vs coverage for each threshold.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        p_max: confidence scores
        thresholds: array of confidence thresholds to sweep
        n_bootstrap: number of bootstrap samples for CI
        
    Returns:
        DataFrame with columns: threshold, coverage, acc_kept, acc_lo, acc_hi
    """
    results = []
    n = len(y_true)
    
    print(f"Computing accuracy-coverage curve over {len(thresholds)} thresholds...")
    
    for tau in thresholds:
        keep = p_max >= tau
        n_kept = keep.sum()
        coverage = n_kept / n
        
        if n_kept == 0:
            # Can't compute accuracy if we abstain on everything
            continue
        
        # Base accuracy on kept predictions
        acc_kept = (y_pred[keep] == y_true[keep]).mean()
        
        # Bootstrap CI
        if n_kept >= 10:  # Need sufficient samples for bootstrap
            boot_accs = []
            rng = np.random.RandomState(RNG_SEED)
            for _ in range(n_bootstrap):
                idx_boot = rng.choice(n_kept, size=n_kept, replace=True)
                y_true_boot = y_true[keep][idx_boot]
                y_pred_boot = y_pred[keep][idx_boot]
                boot_accs.append((y_pred_boot == y_true_boot).mean())
            
            boot_accs = np.array(boot_accs)
            acc_lo = np.percentile(boot_accs, 2.5)
            acc_hi = np.percentile(boot_accs, 97.5)
        else:
            # Too few samples for reliable CI
            acc_lo = acc_kept
            acc_hi = acc_kept
        
        results.append({
            'threshold': tau,
            'coverage': coverage,
            'acc_kept': acc_kept,
            'acc_lo': acc_lo,
            'acc_hi': acc_hi
        })
    
    df = pd.DataFrame(results)
    print(f"[OK] Computed {len(df)} valid threshold points")
    return df


def find_operating_point(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Find optimal operating point that balances accuracy and coverage.
    Uses product: acc * coverage (maximize both simultaneously)
    
    FIXED: Use tau=0.80 as specified in paper, not data-driven optimization.
    
    Returns:
        (threshold, coverage, accuracy)
    """
    # Use tau=0.80 as specified in paper
    tau_target = 0.80
    
    # Find closest threshold to target
    distances = (df['threshold'] - tau_target).abs()
    idx_best = distances.idxmin()
    row_best = df.loc[idx_best]
    
    return row_best['threshold'], row_best['coverage'], row_best['acc_kept']


def compute_auacc(df: pd.DataFrame) -> float:
    """
    Compute area under accuracy-coverage curve using trapezoidal rule.
    Normalized to [0,1] scale.
    """
    # Sort by coverage ASCENDING for proper integration
    df_sorted = df.sort_values('coverage', ascending=True).copy()
    
    # Remove any NaN values
    df_sorted = df_sorted.dropna(subset=['coverage', 'acc_kept'])
    
    if len(df_sorted) < 2:
        return 0.0
    
    # Compute raw integral (use trapezoid for numpy >= 2.0)
    try:
        auacc_raw = np.trapezoid(df_sorted['acc_kept'], df_sorted['coverage'])
    except AttributeError:
        auacc_raw = np.trapz(df_sorted['acc_kept'], df_sorted['coverage'])
    
    # Normalize by coverage range to get value in [0,1]
    cov_range = df_sorted['coverage'].iloc[-1] - df_sorted['coverage'].iloc[0]
    if cov_range <= 0:
        return 0.0
    
    auacc_normalized = auacc_raw / cov_range
    
    return auacc_normalized


def plot_safety_curve(df: pd.DataFrame, tau_opt: float, cov_opt: float, acc_opt: float, auacc: float):
    """
    Generate publication-quality safety curve figure with main plot and zoomed inset.
    
    Args:
        df: DataFrame with accuracy-coverage data
        tau_opt: optimal threshold
        cov_opt: coverage at optimal point
        acc_opt: accuracy at optimal point
        auacc: area under curve
    """
    # Set publication style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.0
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Sort by coverage for plotting
    df_plot = df.sort_values('coverage')
    
    # Baseline: no abstention (coverage = 1.0)
    df_full = df[df['coverage'] >= 0.99]
    if len(df_full) > 0:
        acc_no_abstain = df_full['acc_kept'].mean()
    else:
        acc_no_abstain = df['acc_kept'].iloc[0]
    
    # ========== MAIN PLOT (Full Range) ==========
    # Main curve with CI
    ax1.fill_between(df_plot['coverage'], df_plot['acc_lo'], df_plot['acc_hi'],
                     alpha=0.2, color='#2ecc71', label='95% CI (Bootstrap)')
    
    ax1.plot(df_plot['coverage'], df_plot['acc_kept'], '-',
            color='#2ecc71', linewidth=2.5, label='Accuracy on Kept', marker='o', 
            markersize=3, markevery=max(1, len(df_plot)//15))
    
    # No-abstention baseline
    ax1.axhline(y=acc_no_abstain, color='#3498db', linestyle='--', linewidth=2,
               alpha=0.7, label=f'No Abstention ({acc_no_abstain:.3f})')
    
    # Operating point
    ax1.plot(cov_opt, acc_opt, 'r*', markersize=18, 
            label=f'Operating (τ={tau_opt:.2f})', zorder=10, markeredgecolor='darkred', markeredgewidth=1)
    ax1.axvline(x=cov_opt, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax1.axhline(y=acc_opt, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Labels
    ax1.set_xlabel('Coverage (Fraction of Predictions Made)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy on Kept Predictions', fontsize=11, fontweight='bold')
    ax1.set_title('Safety Curve: Accuracy vs Coverage (Full Range)', fontsize=12, fontweight='bold', pad=10)
    
    # Set appropriate range
    cov_min = max(0, df_plot['coverage'].min() - 0.05)
    ax1.set_xlim([cov_min, 1.02])
    acc_min = max(0.85, df_plot['acc_kept'].min() - 0.02)
    ax1.set_ylim([acc_min, 1.02])
    
    # Grid and legend
    ax1.grid(alpha=0.25, linestyle='--', linewidth=0.5)
    ax1.legend(loc='lower right', framealpha=0.95, fontsize=9)
    
    # Add AUACC annotation
    ax1.text(0.03, 0.97, f'AUACC = {auacc:.3f}',
            transform=ax1.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.7, edgecolor='orange', linewidth=2))
    
    # ========== ZOOMED INSET (High Coverage) ==========
    # Filter to high-coverage region
    df_zoom = df_plot[df_plot['coverage'] >= 0.90].copy()
    
    if len(df_zoom) > 0:
        # Zoomed curve with CI
        ax2.fill_between(df_zoom['coverage'], df_zoom['acc_lo'], df_zoom['acc_hi'],
                         alpha=0.25, color='#2ecc71')
        
        ax2.plot(df_zoom['coverage'], df_zoom['acc_kept'], '-',
                color='#2ecc71', linewidth=2.5, marker='o', markersize=4, markevery=1)
        
        # Baseline
        ax2.axhline(y=acc_no_abstain, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Operating point if in range
        if cov_opt >= 0.90:
            ax2.plot(cov_opt, acc_opt, 'r*', markersize=16, zorder=10,
                    markeredgecolor='darkred', markeredgewidth=1)
            ax2.axvline(x=cov_opt, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
            ax2.axhline(y=acc_opt, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
        
        ax2.set_xlabel('Coverage', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax2.set_title('Zoomed: High Coverage Region (0.90–1.0)', fontsize=12, fontweight='bold', pad=10)
        
        ax2.set_xlim([0.89, 1.01])
        acc_zoom_min = max(0.90, df_zoom['acc_kept'].min() - 0.01)
        acc_zoom_max = min(1.01, df_zoom['acc_kept'].max() + 0.01)
        ax2.set_ylim([acc_zoom_min, acc_zoom_max])
        
        ax2.grid(alpha=0.25, linestyle='--', linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data\nin high coverage\nregion', 
                ha='center', va='center', fontsize=10, transform=ax2.transAxes)
        ax2.set_xlim([0.9, 1.0])
        ax2.set_ylim([0.9, 1.0])
    
    plt.tight_layout()
    
    # Save both PNG and PDF
    png_path = OUTPUT_DIR / 'safety_curve_fixed.png'
    pdf_path = OUTPUT_DIR / 'safety_curve_fixed.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved {png_path}")
    print(f"[OK] Saved {pdf_path}")


def plot_logscale_diagnostic(df: pd.DataFrame, tau_opt: float, cov_opt: float, acc_opt: float):
    """
    Create diagnostic plot with log-scale y-axis to emphasize small differences near accuracy=1.0.
    Plots log(1 - accuracy) vs coverage to highlight safety improvements.
    """
    plt.rcParams['font.size'] = 10
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by coverage
    df_plot = df.sort_values('coverage').copy()
    
    # Compute error rate (1 - accuracy) and its log
    df_plot['error_rate'] = 1.0 - df_plot['acc_kept']
    df_plot['error_rate'] = df_plot['error_rate'].clip(lower=1e-5)  # Avoid log(0)
    df_plot['log_error'] = np.log10(df_plot['error_rate'])
    
    # Plot
    ax.plot(df_plot['coverage'], df_plot['log_error'], '-o',
            color='#e74c3c', linewidth=2.5, markersize=4, 
            markevery=max(1, len(df_plot)//15), label='log₁₀(Error Rate)')
    
    # Operating point
    error_opt = 1.0 - acc_opt
    log_error_opt = np.log10(max(error_opt, 1e-5))
    ax.plot(cov_opt, log_error_opt, 'g*', markersize=18, 
            label=f'Operating (τ={tau_opt:.2f})', zorder=10,
            markeredgecolor='darkgreen', markeredgewidth=1)
    ax.axvline(x=cov_opt, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Coverage (Fraction of Predictions Made)', fontsize=11, fontweight='bold')
    ax.set_ylabel('log₁₀(1 - Accuracy) = log₁₀(Error Rate)', fontsize=11, fontweight='bold')
    ax.set_title('Diagnostic: Log-Scale Error Rate vs Coverage\n(Emphasizes small improvements near perfect accuracy)',
                fontsize=12, fontweight='bold', pad=10)
    
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', framealpha=0.95, fontsize=10)
    
    # Add secondary y-axis with actual error rates
    ax2 = ax.secondary_yaxis('right', functions=(lambda x: 10**x, lambda x: np.log10(x)))
    ax2.set_ylabel('Error Rate (1 - Accuracy)', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    out_path = OUTPUT_DIR / 'safety_curve_logscale.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved diagnostic plot to {out_path}")


def main():
    """Main execution."""
    print("=" * 70)
    print("FoodLens Safety Curve Generation (FIXED)")
    print("=" * 70)
    print()
    
    # Load data
    y_true, y_pred, p_max = load_data()
    
    # Define threshold sweep - extend lower to capture more coverage range
    # Use finer granularity in high-confidence region where action happens
    thresholds = np.concatenate([
        np.linspace(0.0, 0.5, 20),    # Coarse in low range
        np.linspace(0.5, 0.95, 50),   # Fine in mid range
        np.linspace(0.95, 0.999, 30)  # Very fine near 1.0
    ])
    
    # Compute accuracy-coverage curve
    df = compute_accuracy_coverage_curve(y_true, y_pred, p_max, thresholds, n_bootstrap=1000)
    
    # Save raw data
    table_path = TABLE_DIR / 'safety_curve_points.csv'
    df.to_csv(table_path, index=False, float_format='%.6f')
    print(f"[OK] Saved raw data to {table_path}")
    
    # Find operating point
    tau_opt, cov_opt, acc_opt = find_operating_point(df)
    
    # Compute AUACC
    auacc = compute_auacc(df)
    
    # Print summary in requested format
    print()
    print("=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Operating tau={tau_opt:.3f}, Coverage={cov_opt:.3f}, Accuracy={acc_opt:.3f}, AUACC={auacc:.3f}")
    print()
    print(f"Detailed Breakdown:")
    print(f"  Threshold tau* = {tau_opt:.4f}")
    print(f"  Coverage       = {cov_opt:.4f} ({cov_opt*100:.1f}%)")
    print(f"  Accuracy       = {acc_opt:.4f} ({acc_opt*100:.1f}%)")
    print(f"  AUACC          = {auacc:.4f}")
    print()
    
    # Generate plots
    print("Generating figures...")
    plot_safety_curve(df, tau_opt, cov_opt, acc_opt, auacc)
    plot_logscale_diagnostic(df, tau_opt, cov_opt, acc_opt)
    
    print()
    print("=" * 70)
    print("Safety curve generation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

