"""
Figure 2: SRAM budget for NeuronFabric 334K model (FP32 vs BF16W Adam)
Output: sram_budget.pdf  (also saves sram_budget.png for preview)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── data ──────────────────────────────────────────────────────────────────────
variants = ["FP32 Adam", "BF16W Adam"]

weights_mb  = [1.34, 0.67]   # weights only
moments_mb  = [2.67, 2.67]   # m + v (always FP32)
total_mb    = [w + m for w, m in zip(weights_mb, moments_mb)]  # [4.00, 3.34]

limit_mb = 4.00   # ZCU102 BRAM

# ── colours ───────────────────────────────────────────────────────────────────
C_WEIGHT  = "#4C72B0"   # blue  – weights
C_MOMENT  = "#DD8452"   # orange – moments
C_LIMIT   = "#CC3311"   # red   – BRAM limit line

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.8))

x = np.arange(len(variants))
bar_w = 0.45

bars_w = ax.bar(x, weights_mb, bar_w, label="Weights", color=C_WEIGHT)
bars_m = ax.bar(x, moments_mb, bar_w, bottom=weights_mb, label="Moments (m + v)", color=C_MOMENT)

# BRAM limit line
ax.axhline(limit_mb, color=C_LIMIT, linewidth=1.8, linestyle="--", label=f"ZCU102 BRAM limit ({limit_mb:.1f} MB)")

# total labels above each bar
for i, total in enumerate(total_mb):
    color = C_LIMIT if total >= limit_mb else "black"
    marker = " ✗" if total >= limit_mb else " ✓"
    ax.text(x[i], total + 0.05, f"{total:.2f} MB{marker}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=color)

# value labels inside bars
for bar, val in zip(bars_w, weights_mb):
    ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
            f"{val:.2f} MB", ha="center", va="center", color="white", fontsize=9)

for bar, bot, val in zip(bars_m, weights_mb, moments_mb):
    ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
            f"{val:.2f} MB", ha="center", va="center", color="white", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=11)
ax.set_ylabel("On-chip SRAM (MB)", fontsize=11)
ax.set_ylim(0, 4.8)
ax.set_title("Figure 2: SRAM Budget — NeuronFabric 334K Model\n(ZCU102 BRAM = 4.0 MB)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out_dir = "."   # save alongside this script (docs/paper/fig/)
plt.savefig(f"{out_dir}/sram_budget.pdf", bbox_inches="tight")
plt.savefig(f"{out_dir}/sram_budget.png", dpi=150, bbox_inches="tight")
print("Saved: sram_budget.pdf  sram_budget.png")
