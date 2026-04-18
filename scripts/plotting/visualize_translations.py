#!/usr/bin/env python3
"""Visualize translational automorphisms on the lattice."""

import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def read_positions(filename):
    sites = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            sid = int(parts[0])
            sub = int(parts[2])
            x, y = float(parts[3]), float(parts[4])
            sites[sid] = (x, y, sub)
    return sites

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'test_bfg_2x2'
    
    sites = read_positions(os.path.join(data_dir, 'positions.dat'))
    with open(os.path.join(data_dir, 'automorphism_results', 'max_clique.json')) as f:
        translations = json.load(f)
    
    n_sites = len(sites)
    n_trans = len(translations)
    
    # Identify identity
    identity = list(range(n_sites))
    non_identity = [t for t in translations if t != identity]
    
    # Sublattice colors
    sub_colors = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', 3: '#f39c12'}
    sub_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    
    # Arrow colors for each non-identity translation
    arrow_cmap = plt.cm.Set1
    arrow_colors = [arrow_cmap(i / max(1, len(non_identity) - 1)) for i in range(len(non_identity))]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Left panel: all translations overlaid ---
    ax = axes[0]
    ax.set_title(f'Translation Automorphisms ({n_trans} total, {len(non_identity)} non-identity)', fontsize=13)
    
    xs = [sites[i][0] for i in range(n_sites)]
    ys = [sites[i][1] for i in range(n_sites)]
    
    for sid in range(n_sites):
        x, y, sub = sites[sid]
        ax.scatter(x, y, c=sub_colors.get(sub, 'gray'), s=200, zorder=5, edgecolors='black', linewidths=1.2)
        ax.annotate(str(sid), (x, y), ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)
    
    legend_arrows = []
    for tidx, sigma in enumerate(non_identity):
        color = arrow_colors[tidx]
        # Compute the displacement vector (from site 0)
        dx = sites[sigma[0]][0] - sites[0][0]
        dy = sites[sigma[0]][1] - sites[0][1]
        label = f'T{tidx+1}: ({dx:.2f}, {dy:.2f})'
        
        # Draw arrows for a few representative sites (one per sublattice)
        drawn_subs = set()
        for sid in range(n_sites):
            sub = sites[sid][2]
            if sub in drawn_subs:
                continue
            drawn_subs.add(sub)
            x0, y0 = sites[sid][0], sites[sid][1]
            x1, y1 = sites[sigma[sid]][0], sites[sigma[sid]][1]
            adx, ady = x1 - x0, y1 - y0
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2.0,
                                      connectionstyle='arc3,rad=0.1'))
        
        legend_arrows.append(Line2D([0], [0], color=color, lw=2, label=label))
    
    # Sublattice legend
    sub_patches = [mpatches.Patch(color=sub_colors[s], label=f'Sublattice {sub_names[s]}')
                   for s in sorted(set(sites[i][2] for i in range(n_sites)))]
    
    leg1 = ax.legend(handles=sub_patches, loc='upper left', fontsize=9, title='Sublattices')
    ax.add_artist(leg1)
    ax.legend(handles=legend_arrows, loc='upper right', fontsize=9, title='Translations')
    
    ax.set_aspect('equal')
    margin = 0.3
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.grid(True, alpha=0.3)
    
    # --- Right panel: individual translations ---
    ax2 = axes[1]
    ax2.set_title('Translation Group Structure', fontsize=13)
    
    # Show group multiplication table
    # Map each translation to an index
    trans_tuples = [tuple(t) for t in translations]
    trans_idx = {t: i for i, t in enumerate(trans_tuples)}
    
    # Compose permutations
    def compose(a, b):
        return tuple(a[b[i]] for i in range(len(a)))
    
    n = len(translations)
    table = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            prod = compose(translations[i], translations[j])
            table[i, j] = trans_idx[prod]
    
    # Labels
    labels = ['e']
    for tidx, sigma in enumerate(non_identity):
        dx = sites[sigma[0]][0] - sites[0][0]
        dy = sites[sigma[0]][1] - sites[0][1]
        labels.append(f'T{tidx+1}')
    
    # Reorder: identity first
    order = [trans_tuples.index(tuple(identity))]
    for i in range(n):
        if i not in order:
            order.append(i)
    
    reordered_table = np.zeros((n, n), dtype=int)
    reorder_map = {old: new for new, old in enumerate(order)}
    for i in range(n):
        for j in range(n):
            reordered_table[i, j] = reorder_map[table[order[i], order[j]]]
    
    im = ax2.imshow(reordered_table, cmap='Set3', vmin=0, vmax=n-1)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel('Right operand', fontsize=11)
    ax2.set_ylabel('Left operand', fontsize=11)
    
    # Annotate cells
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, labels[reordered_table[i, j]], ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_title(f'Cayley Table (Z{" × Z".join(str(o) for _, o in sorted(set()))})', fontsize=13) if False else None
    
    # Determine group structure from orders
    from collections import Counter
    orders = []
    for t in translations:
        tt = tuple(t)
        power = tt
        for o in range(1, n+1):
            if power == tuple(identity):
                orders.append(o)
                break
            power = compose(list(power), t)
    
    order_counts = Counter(orders)
    if n == 1:
        group_str = 'Z₁ (trivial)'
    elif all(o == n for o in orders if o > 1):
        group_str = f'Z_{n}'
    else:
        # Factor group
        with open(os.path.join(data_dir, 'automorphism_results', 'minimal_generators.json')) as f:
            gens = json.load(f)
        gen_orders = [g['order'] for g in gens]
        group_str = ' × '.join(f'Z_{o}' for o in sorted(gen_orders))
    
    ax2.set_title(f'Cayley Table ({group_str})', fontsize=13)
    
    plt.tight_layout()
    out_path = os.path.join(data_dir, 'automorphism_results', 'translation_group.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")
    plt.close()

if __name__ == '__main__':
    main()
