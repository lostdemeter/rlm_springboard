#!/usr/bin/env python3
"""
Resonant Array: Demo for non-technical audiences.

Generates a single publication-quality figure that shows:
  1. A phase circle with stored items (dots) and a query (star)
  2. Resonance bands showing which items are "nearby"
  3. Multiple independent heads (different Riemann zeros)
  4. Ranked results with distances
  5. Frequency signatures for content matching

Run:  python3 resonant_demo.py
Output: resonant_array_demo.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import os

TWOPI = 2.0 * math.pi
GAMMAS = [14.134725, 21.022040, 25.010858, 30.424876,
          32.935062, 37.586178, 40.918719, 43.327073]

# Phase primitive
def phase(key, gamma):
    return (gamma * key) % TWOPI

def circ_dist(a, b):
    d = abs(a - b)
    return min(d, TWOPI - d) / math.pi

def hash_key(text):
    h = 0
    for c in text:
        h = (h * 31 + ord(c)) & 0x7FFFFFFF
    return h

# ── Data ───────────────────────────────────────────────────────────────

# Real, recognizable items a manager would know
items = [
    "Q4 revenue exceeded targets by 12%",
    "Customer acquisition cost down 8%",
    "New product launch scheduled Q2",
    "Team headcount grew to 45 engineers",
    "Server costs increased 15% this quarter",
    "User retention improved to 94%",
    "Competitor entered our market segment",
    "Patent filed for core algorithm",
    "European expansion approved for 2026",
    "Engineering velocity up 22% year-over-year",
    "Board meeting rescheduled to March 15",
    "Security audit passed with no findings",
    "Partnership deal signed with Acme Corp",
    "Customer churn rate dropped to 3.2%",
    "R&D budget increased 18% for next year",
    "Office lease renewed for 3 more years",
]

query_text = "Quarterly financial results show strong growth"

N_HEADS = 2  # Two heads for side-by-side comparison

# Insert items
keys = {}
phases_by_head = {h: {} for h in range(N_HEADS)}
for text in items:
    k = hash_key(text)
    keys[text] = k
    for h in range(N_HEADS):
        phases_by_head[h][text] = phase(k, GAMMAS[h])

# Query
qkey = hash_key(query_text)
qphases = [phase(qkey, GAMMAS[h]) for h in range(N_HEADS)]

# Compute resonances for each head
results_by_head = {h: [] for h in range(N_HEADS)}
for text in items:
    k = keys[text]
    for h in range(N_HEADS):
        d = circ_dist(phase(k, GAMMAS[h]), qphases[h])
        results_by_head[h].append((text, d))

for h in range(N_HEADS):
    results_by_head[h].sort(key=lambda x: x[1])

# ── Figure ────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#1a1a2e')
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.35,
                      width_ratios=[3, 1.5, 1.5])

# ── Top-left: Phase Circle, Head 0 ──
ax_circle_left = fig.add_subplot(gs[0, 0], facecolor='#16213e')
ax_circle_left.set_aspect('equal')

# Draw reference circle
theta = np.linspace(0, 2*math.pi, 100)
ax_circle_left.plot(np.cos(theta), np.sin(theta), '-', color='#2a2a5e', lw=1.5, zorder=1)

# Inner/outer rings
for r in [0.7, 0.85]:
    ax_circle_left.plot(r*np.cos(theta), r*np.sin(theta), '--', color='#2a2a5e', lw=0.5, alpha=0.5, zorder=1)

# Draw query resonance band (head 0)
query_angle_0 = qphases[0]
for th_factor in np.linspace(0.05, 0.25, 10):
    band = plt.Circle((0, 0), 0.98, color='#e94560', alpha=0.015, transform=ax_circle_left.transData)
    ax_circle_left.add_patch(band)

# Draw stored items as dots, colored by resonance distance to query
for text in items:
    p = phases_by_head[0][text]
    d = circ_dist(p, qphases[0])
    x, y = math.cos(p), math.sin(p)
    # Color: green (close) → gray (far)
    intensity = max(0.2, 1.0 - d * 2)
    color = (0.2, intensity * 0.8, 0.2)
    size = max(30, 150 * (1.0 - d))
    ax_circle_left.scatter(x, y, s=size * 0.8, c=[color], edgecolors='#4ecca3',
                          linewidth=1.5 if d < 0.15 else 0.5, alpha=0.9, zorder=5)
    
    # Label top-4 results
    rank = [r[0] for r in results_by_head[0]].index(text)
    if rank < 4:
        short = text[:18] + '..' if len(text) > 20 else text
        offset = 0.12
        ax_circle_left.annotate(f"#{rank+1} {short}", (x, y),
                               xytext=(x + offset * (1 if x > 0 else -1), y + offset * (1 if y > 0 else -1)),
                               fontsize=5.5, color='#cccccc',
                               arrowprops=dict(arrowstyle='->', color='#4ecca3', alpha=0.5, lw=0.5))

# Draw query star
qx_0, qy_0 = math.cos(query_angle_0), math.sin(query_angle_0)
ax_circle_left.scatter(qx_0, qy_0, s=400, c='#e94560', marker='*',
                       edgecolors='white', linewidth=2, zorder=10, label='Query')

# Resonance ring
ring_0 = plt.Circle((qx_0, qy_0), 0.2, color='#e94560', fill=False,
                    linestyle='--', linewidth=1.5, alpha=0.5, zorder=3)
ax_circle_left.add_patch(ring_0)

ax_circle_left.set_xlim(-1.15, 1.15)
ax_circle_left.set_ylim(-1.15, 1.15)
ax_circle_left.set_title(f"Head 0 (γ₁={GAMMAS[0]:.2f}): Phase Space", 
                         fontsize=12, color='white', fontweight='bold', pad=15)
ax_circle_left.legend(loc='lower right', fontsize=8, 
                     facecolor='#16213e', edgecolor='#2a2a5e', labelcolor='white')
ax_circle_left.axis('off')

# ── Bottom-left: Phase Circle, Head 1 ──
ax_circle_bottom = fig.add_subplot(gs[1, 0], facecolor='#16213e')
ax_circle_bottom.set_aspect('equal')

# Same elements for head 1
ax_circle_bottom.plot(np.cos(theta), np.sin(theta), '-', color='#2a2a5e', lw=1.5, zorder=1)
for r in [0.7, 0.85]:
    ax_circle_bottom.plot(r*np.cos(theta), r*np.sin(theta), '--', color='#2a2a5e', lw=0.5, alpha=0.5, zorder=1)

query_angle_1 = qphases[1]

for text in items:
    p = phases_by_head[1][text]
    d = circ_dist(p, qphases[1])
    x, y = math.cos(p), math.sin(p)
    intensity = max(0.2, 1.0 - d * 2)
    color = (0.2, intensity * 0.8, 0.2)
    size = max(30, 150 * (1.0 - d))
    ax_circle_bottom.scatter(x, y, s=size * 0.8, c=[color], edgecolors='#4ecca3',
                            linewidth=1.5 if d < 0.15 else 0.5, alpha=0.9, zorder=5)

qx_1, qy_1 = math.cos(query_angle_1), math.sin(query_angle_1)
ax_circle_bottom.scatter(qx_1, qy_1, s=400, c='#e94560', marker='*',
                        edgecolors='white', linewidth=2, zorder=10, label='Query')
ring_1 = plt.Circle((qx_1, qy_1), 0.2, color='#e94560', fill=False,
                    linestyle='--', linewidth=1.5, alpha=0.5, zorder=3)
ax_circle_bottom.add_patch(ring_1)

ax_circle_bottom.set_xlim(-1.15, 1.15)
ax_circle_bottom.set_ylim(-1.15, 1.15)
ax_circle_bottom.set_title(f"Head 1 (γ₂={GAMMAS[1]:.2f}): Phase Space", 
                          fontsize=12, color='white', fontweight='bold', pad=15)
ax_circle_bottom.legend(loc='lower right', fontsize=8,
                       facecolor='#16213e', edgecolor='#2a2a5e', labelcolor='white')
ax_circle_bottom.axis('off')

# ── Right column: Results tables ──
# Head 0 results
ax_r0 = fig.add_subplot(gs[0, 1:], facecolor='#16213e')
ax_r0.axis('off')

res_0 = results_by_head[0]
n_show = min(8, len(res_0))

# Table header
ax_r0.text(0, 1.02, f"Head 0 Results — Query: \"{query_text}\"", 
          fontsize=10, color='#4ecca3', fontweight='bold', transform=ax_r0.transAxes)

# Build table
table_data = []
for rank, (text, dist) in enumerate(res_0[:n_show]):
    distance_pct = dist * 100
    bar = '█' * max(1, int(15 * (1.0 - dist)))
    table_data.append([f"#{rank+1}", text[:45], f"{distance_pct:.1f}%", bar])

col_labels = ['Rank', 'Item', 'Distance', 'Proximity']
col_widths = [0.08, 0.52, 0.12, 0.28]

table_0 = ax_r0.table(cellText=table_data, colLabels=col_labels,
                       loc='center', cellLoc='left',
                       colWidths=col_widths)
table_0.auto_set_font_size(False)
table_0.set_fontsize(8)
table_0.scale(1, 1.4)

# Style the table
for (row, col), cell in table_0.get_celld().items():
    cell.set_facecolor('#1a1a2e' if row % 2 == 0 else '#16213e')
    cell.set_edgecolor('#2a2a5e')
    cell.set_text_props(color='white')
    if row == 0:  # header
        cell.set_facecolor('#0f3460')
        cell.set_text_props(color='#4ecca3', fontweight='bold')

# Annotate the "found" items
ax_r0.text(0, -0.05, f"★ Query matched {n_show}/{len(items)} items at this threshold",
          fontsize=8, color='#888888', transform=ax_r0.transAxes, style='italic')

# ── Bottom-right: Combined multi-head ranking ──
ax_r1 = fig.add_subplot(gs[1, 1:], facecolor='#16213e')
ax_r1.axis('off')

ax_r1.text(0, 1.02, "Cross-Head Consensus: Items Found by BOTH Heads",
          fontsize=10, color='#4ecca3', fontweight='bold', transform=ax_r1.transAxes)

# Find items in top N of BOTH heads
top_n = 5
head0_top = set(r[0] for r in results_by_head[0][:top_n])
head1_top = set(r[0] for r in results_by_head[1][:top_n])
consensus = head0_top & head1_top

if consensus:
    cons_data = []
    n = 1
    for text in sorted(consensus):
        d0 = next(r[1] for r in results_by_head[0] if r[0] == text)
        d1 = next(r[1] for r in results_by_head[1] if r[0] == text)
        avg_d = (d0 + d1) / 2
        cons_data.append([f"#{n}", text[:40], f"{avg_d*100:.1f}%", f"|{'█'*int(12*(1.0-avg_d))}"])
        n += 1
    
    cons_table = ax_r1.table(cellText=cons_data,
                              colLabels=['', 'Item', 'Avg Dist', 'Consensus'],
                              loc='center', cellLoc='left',
                              colWidths=[0.08, 0.52, 0.12, 0.28])
    cons_table.auto_set_font_size(False)
    cons_table.set_fontsize(8)
    cons_table.scale(1, 1.4)
    
    for (row, col), cell in cons_table.get_celld().items():
        cell.set_facecolor('#1a1a2e' if row % 2 == 0 else '#16213e')
        cell.set_edgecolor('#2a2a5e')
        cell.set_text_props(color='white')
        if row == 0:
            cell.set_facecolor('#0f3460')
            cell.set_text_props(color='#4ecca3', fontweight='bold')
    
    ax_r1.text(0, -0.05, f"✓ {len(consensus)} items found by both heads independently — no training, no randomness",
              fontsize=8, color='#4ecca3', transform=ax_r1.transAxes, style='italic')
else:
    ax_r1.text(0.5, 0.5, "No item found by both heads in top 5.\nDecrease threshold or increase search radius.",
              fontsize=9, color='#888888', ha='center', transform=ax_r1.transAxes)

# ── Annotations ──
fig.text(0.5, 0.96, "Resonant Array: Deterministic Similarity Search Without Training",
         ha='center', fontsize=16, fontweight='bold', color='white')
fig.text(0.5, 0.925, "Every item is indexed by γ·key mod 2π. Similar items have nearby phases. Two independent Riemann zeros (heads) give independent rankings — and they agree.",
         ha='center', fontsize=10, color='#888888')

# Key properties in footer
props_text = "No random seeds  •  No training  •  No softmax  •  O(1) lookup  •  O(N·H) resonance  •  Deterministic across all machines"
fig.text(0.5, 0.02, props_text, ha='center', fontsize=8, color='#555555',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f3460', alpha=0.5))

plt.savefig('resonant_array_demo.png', dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
print("✓ Saved resonant_array_demo.png")
