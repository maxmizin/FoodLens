#!/usr/bin/env python3
"""
Generate comparison graph showing FoodLens vs DeBERTa performance
on easy vs hard samples for the research paper.
"""

import matplotlib.pyplot as plt
import numpy as np
import json

# Set publication-quality parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Data from results
# Easy samples (full test set, 240 samples, 93.75% baseline accuracy)
easy_deberta = {
    'F1': 0.767,
    'Accuracy': 0.950,
    'Coverage': 1.000
}

easy_foodlens = {
    'F1': 0.789,
    'Accuracy': 0.954,
    'Coverage': 0.988
}

# Hard samples (minority classes, 25 samples, 60% baseline accuracy)
# Baseline at τ < 0.95
hard_deberta = {
    'F1': 0.476,
    'Accuracy': 0.600,
    'Coverage': 1.000
}

# With abstention at τ = 0.95
hard_foodlens = {
    'F1': 0.510,
    'Accuracy': 0.650,
    'Coverage': 0.800
}

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Metrics to compare
metrics = ['F1 Score', 'Accuracy', 'Coverage']
x = np.arange(len(metrics))
width = 0.35

# Easy samples subplot
easy_deberta_vals = [easy_deberta['F1'], easy_deberta['Accuracy'], easy_deberta['Coverage']]
easy_foodlens_vals = [easy_foodlens['F1'], easy_foodlens['Accuracy'], easy_foodlens['Coverage']]

bars1 = ax1.bar(x - width/2, easy_deberta_vals, width, label='DeBERTa (No Abstention)', 
                color='#FF7F0E', alpha=0.8, edgecolor='black', linewidth=0.8)
bars2 = ax1.bar(x + width/2, easy_foodlens_vals, width, label='FoodLens (Abstaining)', 
                color='#2CA02C', alpha=0.8, edgecolor='black', linewidth=0.8)

ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('(A) Easy Samples\n(Full Test Set: 240 samples, 93.75% baseline accuracy)', 
              fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend(loc='lower right', framealpha=0.95)
ax1.set_ylim([0, 1.05])
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# Add improvement annotation
ax1.text(0.5, 0.15, f'F1 Improvement: +{(easy_foodlens["F1"] - easy_deberta["F1"])*100:.1f}%\n(minimal gain)',
         transform=ax1.transAxes, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3, edgecolor='black'),
         fontsize=9, fontweight='bold')

# Hard samples subplot
hard_deberta_vals = [hard_deberta['F1'], hard_deberta['Accuracy'], hard_deberta['Coverage']]
hard_foodlens_vals = [hard_foodlens['F1'], hard_foodlens['Accuracy'], hard_foodlens['Coverage']]

bars3 = ax2.bar(x - width/2, hard_deberta_vals, width, label='DeBERTa (No Abstention)', 
                color='#FF7F0E', alpha=0.8, edgecolor='black', linewidth=0.8)
bars4 = ax2.bar(x + width/2, hard_foodlens_vals, width, label='FoodLens (Abstaining)', 
                color='#2CA02C', alpha=0.8, edgecolor='black', linewidth=0.8)

ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('(B) Hard Samples\n(Minority Classes: 25 samples, 60% baseline accuracy)', 
              fontweight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend(loc='lower right', framealpha=0.95)
ax2.set_ylim([0, 1.05])
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# Add value labels on bars
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# Add improvement annotation
ax2.text(0.5, 0.15, f'F1 Improvement: +{(hard_foodlens["F1"] - hard_deberta["F1"])*100:.1f}%\n(substantial gain)',
         transform=ax2.transAxes, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5, edgecolor='black'),
         fontsize=9, fontweight='bold')

plt.tight_layout()

# Save figure
output_path = 'paper/figs/easy_vs_hard_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved comparison figure to {output_path}")

# Also save summary statistics to JSON
summary = {
    "easy_samples": {
        "description": "Full test set (240 samples, 93.75% baseline accuracy)",
        "deberta": easy_deberta,
        "foodlens": easy_foodlens,
        "f1_improvement_pct": round((easy_foodlens['F1'] - easy_deberta['F1']) * 100, 2)
    },
    "hard_samples": {
        "description": "Minority classes (25 samples, 60% baseline accuracy)",
        "deberta": hard_deberta,
        "foodlens": hard_foodlens,
        "f1_improvement_pct": round((hard_foodlens['F1'] - hard_deberta['F1']) * 100, 2)
    }
}

with open('paper/figs/easy_vs_hard_comparison_data.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved comparison data to paper/figs/easy_vs_hard_comparison_data.json")

# Don't show plot, just save
plt.close()

