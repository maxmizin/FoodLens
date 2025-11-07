#!/usr/bin/env python3
"""
Create ethical flow diagram with proper layout and no cropping.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors
COLORS = {
    'process': '#003865',     # CMU blue
    'decision': '#C41230',    # CMU red
    'safe': '#27ae60',        # green
    'warning': '#f39c12',     # orange
}

def main():
    print("[BUILD] Ethical Flow Diagram...")
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define nodes with explicit positions and sizes
    nodes = [
        {'x': 0.8, 'y': 4, 'w': 1.6, 'h': 1.0, 'text': 'Input\nIngredient\nText', 
         'color': COLORS['process']},
        {'x': 3.0, 'y': 4, 'w': 1.6, 'h': 1.0, 'text': 'DeBERTa\nModel\n(Calibrated)', 
         'color': COLORS['process']},
        {'x': 5.2, 'y': 4, 'w': 1.6, 'h': 1.0, 'text': 'Confidence\nEstimate\np_max', 
         'color': COLORS['process']},
        # Decision node (diamond shape)
        {'x': 7.6, 'y': 3.5, 'w': 1.8, 'h': 2.0, 'text': 'Abstention\nGate\n(τ = 0.80)', 
         'color': COLORS['decision'], 'shape': 'diamond'},
        # Outcomes
        {'x': 7.6, 'y': 1.2, 'w': 1.6, 'h': 1.0, 'text': 'Human\nReview\nRequired', 
         'color': COLORS['warning']},
        {'x': 7.6, 'y': 6.5, 'w': 1.6, 'h': 1.0, 'text': 'Automated\nDecision\n(High Conf)', 
         'color': COLORS['safe']},
    ]
    
    # Draw nodes
    for node in nodes:
        x, y, w, h = node['x'], node['y'], node['w'], node['h']
        
        if node.get('shape') == 'diamond':
            # Diamond for decision node
            points = [
                [x + w/2, y + h],      # top
                [x + w, y + h/2],      # right
                [x + w/2, y],          # bottom
                [x, y + h/2],          # left
            ]
            polygon = mpatches.Polygon(points, closed=True,
                                      facecolor=node['color'],
                                      edgecolor='black', linewidth=2, alpha=0.85)
            ax.add_patch(polygon)
            ax.text(x + w/2, y + h/2, node['text'], ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white',
                   bbox=dict(pad=0, facecolor='none', edgecolor='none'))
        else:
            # Rectangle for regular nodes
            rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                 facecolor=node['color'], edgecolor='black',
                                 linewidth=2, alpha=0.85)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, node['text'], ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
    
    # Draw arrows with labels
    arrows = [
        # Main flow
        {'start': (2.4, 4.5), 'end': (3.0, 4.5), 'label': '', 'color': 'black', 'style': '-'},
        {'start': (4.6, 4.5), 'end': (5.2, 4.5), 'label': '', 'color': 'black', 'style': '-'},
        {'start': (6.8, 4.5), 'end': (7.6, 4.5), 'label': 'Check\nThreshold', 
         'color': 'black', 'style': '-'},
        
        # Decision branches
        {'start': (8.5, 5.3), 'end': (8.5, 6.5), 'label': 'p ≥ τ\nConfident', 
         'color': COLORS['safe'], 'style': '-'},
        {'start': (8.5, 3.7), 'end': (8.5, 2.2), 'label': 'p < τ\nUncertain', 
         'color': COLORS['decision'], 'style': '--'},
    ]
    
    for arrow in arrows:
        arrow_patch = FancyArrowPatch(arrow['start'], arrow['end'],
                                     arrowstyle='->', mutation_scale=20,
                                     linewidth=2.5, color=arrow['color'],
                                     linestyle=arrow.get('style', '-'))
        ax.add_patch(arrow_patch)
        
        if arrow['label']:
            mid_x = (arrow['start'][0] + arrow['end'][0]) / 2
            mid_y = (arrow['start'][1] + arrow['end'][1]) / 2
            offset = 0.4 if 'Threshold' in arrow['label'] else 0.3
            ax.text(mid_x + offset, mid_y, arrow['label'], fontsize=8,
                   fontweight='bold', ha='left', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.9, edgecolor='gray', linewidth=1))
    
    # Add risk mitigation note at bottom
    ax.text(5.5, 0.3, 'Risk Mitigation: Defer uncertain predictions to human expertise',
           fontsize=9, ha='center', style='italic', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#CFC493',
                    alpha=0.8, edgecolor='black', linewidth=1.5))
    
    # Title (removed duplicate y parameter)
    ax.text(5.5, 5.8, 'Ethical Decision Flow: Abstention as Safety Mechanism',
           fontsize=12, ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save with extra padding
    out_png = OUTPUT_DIR / 'ethical_flow_diagram_fixed.png'
    out_pdf = OUTPUT_DIR / 'ethical_flow_diagram_fixed.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"  [OK] Saved {out_png}")
    print(f"  [OK] Saved {out_pdf}")
    print("  [PASS] Ethical flow diagram created with proper padding")

if __name__ == '__main__':
    main()

