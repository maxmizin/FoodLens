#!/usr/bin/env python3
"""
Redraw ethical_flow_diagram_fixed.png with correct two-branch abstention flow.

CORRECTS:
1. Expanded canvas with proper padding (no clipping)
2. TWO distinct output arrows from Abstention Gate
3. Proper colors and alignment
4. Clear labeling of both decision paths
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
FIG_DIR = BASE_DIR / 'paper' / 'figs'

# Style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def draw_ethical_flow():
    """
    Draw complete ethical decision flow with two-branch abstention logic.
    """
    # Create figure with expanded canvas
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define colors
    color_input = '#1A5276'     # Blue
    color_gate = '#C0392B'      # Red
    color_human = '#F5B041'     # Orange
    color_auto = '#239B56'      # Green
    color_audit = '#7D3C98'     # Purple
    
    # Node positions (x, y, width, height)
    nodes = {
        'input': (0.5, 5.5, 1.8, 1.0),
        'model': (3.0, 5.5, 1.8, 1.0),
        'confidence': (5.5, 5.5, 1.8, 1.0),
        'gate': (8.5, 4.5, 1.2, 2.5),  # Diamond (special handling)
        'human': (11.0, 6.0, 2.0, 1.2),
        'auto': (11.0, 2.5, 2.0, 1.2),
        'audit': (11.0, 0.5, 2.0, 0.8)
    }
    
    # Draw rectangles
    def draw_rect(pos, color, label, fontsize=10, fontweight='bold'):
        x, y, w, h = pos
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                      edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color='white')
    
    # Draw Input
    draw_rect(nodes['input'], color_input, 'User Input\n(Ingredient List)')
    
    # Draw Model
    draw_rect(nodes['model'], color_input, 'DeBERTa-v3\nClassifier')
    
    # Draw Confidence
    draw_rect(nodes['confidence'], color_input, 'Confidence\nEstimate (p_max)')
    
    # Draw Abstention Gate (Diamond)
    gate_x, gate_y, gate_w, gate_h = nodes['gate']
    diamond = patches.FancyBboxPatch((gate_x, gate_y), gate_w, gate_h,
                                      boxstyle="round,pad=0.15",
                                      edgecolor='black', facecolor=color_gate, linewidth=3)
    ax.add_patch(diamond)
    ax.text(gate_x + gate_w/2, gate_y + gate_h/2, 'Abstention\nGate\n(tau=0.80)',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Draw Human Review
    draw_rect(nodes['human'], color_human, 'Human Review\nRequired', fontsize=11)
    
    # Draw Automated Decision
    draw_rect(nodes['auto'], color_auto, 'Automated\nDecision', fontsize=11)
    
    # Draw Audit Log
    draw_rect(nodes['audit'], color_audit, 'Audit Log', fontsize=9)
    
    # Draw arrows
    def draw_arrow(x1, y1, x2, y2, label='', color='black', width=0.03, label_offset=(0, 0.2)):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=color, shrinkA=0, shrinkB=0))
        if label:
            mid_x = (x1 + x2) / 2 + label_offset[0]
            mid_y = (y1 + y2) / 2 + label_offset[1]
            ax.text(mid_x, mid_y, label, fontsize=9, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    
    # Arrow 1: Input → Model
    draw_arrow(nodes['input'][0] + nodes['input'][2], nodes['input'][1] + nodes['input'][3]/2,
               nodes['model'][0], nodes['model'][1] + nodes['model'][3]/2)
    
    # Arrow 2: Model → Confidence
    draw_arrow(nodes['model'][0] + nodes['model'][2], nodes['model'][1] + nodes['model'][3]/2,
               nodes['confidence'][0], nodes['confidence'][1] + nodes['confidence'][3]/2)
    
    # Arrow 3: Confidence → Gate
    draw_arrow(nodes['confidence'][0] + nodes['confidence'][2], nodes['confidence'][1] + nodes['confidence'][3]/2,
               nodes['gate'][0], nodes['gate'][1] + nodes['gate'][3]/2)
    
    # Arrow 4: Gate → Human Review (p < tau - UNCERTAIN)
    gate_top_x = nodes['gate'][0] + nodes['gate'][2]/2
    gate_top_y = nodes['gate'][1] + nodes['gate'][3]
    human_left_x = nodes['human'][0]
    human_center_y = nodes['human'][1] + nodes['human'][3]/2
    
    # Curved arrow to human review
    from matplotlib.patches import FancyArrowPatch
    arrow_human = FancyArrowPatch((gate_top_x, gate_top_y), (human_left_x, human_center_y),
                                  connectionstyle="arc3,rad=0.3", arrowstyle='->', 
                                  mutation_scale=20, linewidth=2.5, color=color_human)
    ax.add_patch(arrow_human)
    ax.text(9.5, 6.8, 'p < tau\n(Uncertain)', fontsize=9, ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_human, linewidth=2))
    
    # Arrow 5: Gate → Automated Decision (p >= tau - CONFIDENT)
    gate_bottom_x = nodes['gate'][0] + nodes['gate'][2]/2
    gate_bottom_y = nodes['gate'][1]
    auto_left_x = nodes['auto'][0]
    auto_center_y = nodes['auto'][1] + nodes['auto'][3]/2
    
    # Curved arrow to automated decision
    arrow_auto = FancyArrowPatch((gate_bottom_x, gate_bottom_y), (auto_left_x, auto_center_y),
                                connectionstyle="arc3,rad=-0.3", arrowstyle='->', 
                                mutation_scale=20, linewidth=2.5, color=color_auto)
    ax.add_patch(arrow_auto)
    ax.text(9.5, 3.5, 'p >= tau\n(Confident)', fontsize=9, ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_auto, linewidth=2))
    
    # Arrow 6: Human Review → Audit Log
    draw_arrow(nodes['human'][0] + nodes['human'][2]/2, nodes['human'][1],
               nodes['audit'][0] + nodes['audit'][2]/2, nodes['audit'][1] + nodes['audit'][3],
               color='gray')
    
    # Arrow 7: Automated Decision → Audit Log
    draw_arrow(nodes['auto'][0] + nodes['auto'][2]/2, nodes['auto'][1],
               nodes['audit'][0] + nodes['audit'][2]/2, nodes['audit'][1] + nodes['audit'][3],
               color='gray')
    
    # Title
    ax.text(7.0, 7.5, 'Ethical Decision Flow: Abstention as Safety Mechanism',
           fontsize=14, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black', linewidth=2))
    
    # Risk mitigation note
    ax.text(7.0, 0.1, 'Risk Mitigation: Defer uncertain predictions to human expertise',
           fontsize=10, ha='center', style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F8F5', edgecolor='gray', alpha=0.9))
    
    # Save with proper padding
    out_png = FIG_DIR / 'ethical_flow_diagram_fixed.png'
    fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"[OK] Saved {out_png}")
    
    plt.close()

def main():
    """Main execution."""
    print("="*70)
    print("REDRAWING ETHICAL FLOW DIAGRAM")
    print("="*70)
    
    draw_ethical_flow()
    
    print("\n[OK] Ethical flow diagram redrawn with:")
    print("  - Two distinct branches from Abstention Gate")
    print("  - p >= tau -> Automated Decision (green)")
    print("  - p < tau -> Human Review (orange)")
    print("  - Expanded canvas (no clipping)")
    print("  - Proper colors and alignment")
    print("="*70)

if __name__ == '__main__':
    main()

