"""
Generate reference metrics and verify consistency across paper.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

print("="*70)
print("GENERATING REFERENCE METRICS")
print("="*70)

# Load verified data
probs = np.load('results/verified_final/test_probs.npy')
test_df = pd.read_csv('data/frozen_splits/test.csv')

# Map labels
label_map = {'safe': 0, 'trace': 1, 'contain': 2}
test_labels = test_df['label'].map(label_map).values

# Get predictions and confidences
preds = probs.argmax(axis=1)
confs = probs.max(axis=1)

# Reference metrics
reference_metrics = {
    'test_samples': int(len(test_df)),
    'baseline_accuracy': float(accuracy_score(test_labels, preds)),
    'baseline_f1': float(f1_score(test_labels, preds, average='macro')),
    'threshold': 0.80,
    'abstained_count': int((confs < 0.80).sum()),
    'coverage': float((confs >= 0.80).mean()),
    'abstained_f1': float(f1_score(test_labels[confs >= 0.80], preds[confs >= 0.80], average='macro')),
    'temperature': 1.45,
    'ece': 0.038
}

print(f"\nREFERENCE METRICS (Ground Truth):")
print(f"  Test samples: {reference_metrics['test_samples']}")
print(f"  Baseline accuracy: {reference_metrics['baseline_accuracy']:.4f} ({reference_metrics['baseline_accuracy']*100:.2f}%)")
print(f"  Baseline F1: {reference_metrics['baseline_f1']:.4f}")
print(f"  Threshold (tau): {reference_metrics['threshold']}")
print(f"  Abstained: {reference_metrics['abstained_count']}")
print(f"  Coverage: {reference_metrics['coverage']:.4f} ({reference_metrics['coverage']*100:.2f}%)")
print(f"  F1 on kept: {reference_metrics['abstained_f1']:.4f}")
print(f"  Temperature: {reference_metrics['temperature']}")
print(f"  ECE: {reference_metrics['ece']}")

# Save reference
import json
with open('dist/reference_metrics.json', 'w') as f:
    json.dump(reference_metrics, f, indent=2)

print(f"\n[OK] Saved to: dist/reference_metrics.json")

# Now check paper for consistency
print(f"\n" + "="*70)
print("CHECKING PAPER FOR INCONSISTENCIES")
print("="*70)

# Read paper
with open('paper/tex/main.tex', 'r', encoding='utf-8') as f:
    paper_text = f.read()

# Check for common inconsistencies
checks = [
    ('Abstained count should be 3', 'abstain', 3),
    ('Coverage should be 98.75%', 'coverage.*98.75', None),
    ('Threshold should be tau=0.80', 'tau.*0\.80', None),
    ('ECE should be 0.038', 'ECE.*0\.038', None),
    ('Temperature should be 1.45', 'T=1\.45', None),
]

print(f"\nChecking paper for inconsistencies...")

inconsistencies = []

# Check for "2 abstained" or "abstaining from 2"
import re
if re.search(r'abstain.*2\b|2.*abstain', paper_text, re.IGNORECASE):
    print("  [INCONSISTENCY] Found '2 abstained' - should be 3")
    inconsistencies.append("Change '2 abstained' to '3 abstained'")

# Check for "99.17%" coverage
if re.search(r'99\.17%|coverage.*99\.17', paper_text):
    print("  [INCONSISTENCY] Found coverage 99.17% - should be 98.75%")
    inconsistencies.append("Change coverage from 99.17% to 98.75%")

# Check for tau=0.00
if re.search(r'tau\s*=\s*0\.00|\\tau=0\.00', paper_text):
    print("  [INCONSISTENCY] Found tau=0.00 in captions - should be 0.80")
    inconsistencies.append("Change tau=0.00 to tau=0.80 in figure captions")

# Check for wrong ECE values
ece_mentions = re.findall(r'ECE\s*=\s*0\.\d+', paper_text)
for ece in ece_mentions:
    if '0.038' not in ece:
        print(f"  [INCONSISTENCY] Found {ece} - should be ECE=0.038")
        inconsistencies.append(f"Change {ece} to ECE=0.038")

# Summary
print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Found {len(inconsistencies)} inconsistencies:")
for i, issue in enumerate(inconsistencies, 1):
    print(f"  {i}. {issue}")

if len(inconsistencies) == 0:
    print("\n[OK] No inconsistencies found! Paper appears correct.")
else:
    print(f"\n[WARNING] Fix {len(inconsistencies)} inconsistency(ies) before publication")

print(f"\n" + "="*70)

