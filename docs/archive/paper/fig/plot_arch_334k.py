"""
Figure: NeuronFabric 334K — Shakespeare FPGA Architecture Diagram
Generates arch_334k.pdf and arch_334k.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, (ax_arch, ax_sram) = plt.subplots(
    1, 2,
    figsize=(13, 7),
    gridspec_kw={"width_ratios": [2, 1]}
)
fig.patch.set_facecolor("white")

# ── Colour palette ──────────────────────────────────────────────
C_OUTER   = "#1a1a2e"   # dark navy  – outer chip border
C_EMBED   = "#16213e"   # section bg
C_CORE    = "#0f3460"   # transformer core bg
C_LAYER   = "#533483"   # single-layer box
C_OUT     = "#16213e"
C_TITLE   = "#e94560"   # accent red
C_TEXT    = "#f0f0f0"
C_ARROW   = "#e0e0e0"
C_BF16W   = "#4c9be8"
C_FP32    = "#e8734c"
C_LIMIT   = "#e84c4c"
C_MOMENTS = "#f0a500"
C_OK      = "#2ecc71"
C_WARN    = "#e74c3c"

def box(ax, x, y, w, h, facecolor, edgecolor="white", lw=1.0, radius=0.03, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=facecolor, edgecolor=edgecolor,
                       linewidth=lw, alpha=alpha)
    ax.add_patch(p)

def txt(ax, x, y, s, size=9, color=C_TEXT, ha="center", va="center", bold=False, wrap=False):
    weight = "bold" if bold else "normal"
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, wrap=wrap,
            fontfamily="monospace")

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5))

# ── ARCH PANEL ──────────────────────────────────────────────────
ax = ax_arch
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")
ax.set_facecolor("#0a0a1a")
fig.patch.set_facecolor("#0a0a1a")

# Outer chip border
box(ax, 0.3, 0.3, 9.4, 13.4, C_OUTER, edgecolor=C_TITLE, lw=2.0, radius=0.08)

# Title
txt(ax, 5, 13.2, "NeuronFabric 334K — Shakespeare FPGA Config", size=13, bold=True, color=C_TITLE)
txt(ax, 5, 12.75, "embed=88   heads=4   ff=264   layers=4   vocab=256", size=11, color="#aaaacc")

# ── INPUT SECTION ──
box(ax, 0.6, 11.3, 8.8, 1.2, C_EMBED, edgecolor="#334466", lw=0.8, radius=0.04)
txt(ax, 1.5, 12.05, "INPUT", size=10, bold=True, color="#aaaacc", ha="left")
txt(ax, 5, 11.85, "token [0..255]  ──►  Embedding SRAM", size=12, color=C_TEXT)
txt(ax, 5, 11.5,  "vocab × embed = 256 × 88   [ 88 KB ]", size=11, color="#99bbdd")

arrow(ax, 5, 11.3, 5, 10.95)

# ── TRANSFORMER CORE ──
box(ax, 0.6, 3.5, 8.8, 7.6, C_CORE, edgecolor="#5544aa", lw=1.0, radius=0.05)
txt(ax, 1.5, 10.85, "TRANSFORMER CORE  (× 4 identical layers)", size=10, bold=True, color="#aaaacc", ha="left")

# Single layer box
box(ax, 1.0, 3.9, 8.0, 6.6, C_LAYER, edgecolor="#8866cc", lw=0.8, radius=0.04)

rows = [
    (9.4,  "LayerNorm (88)"),
    (8.85, "Attention:  4 heads × headDim=22"),
    (8.35, "  Wq Wk Wv: [88×22]   Wo: [22×88]   fan-in: 88"),
    (7.8,  "+ residual"),
    (7.2,  "LayerNorm (88)"),
    (6.65, "FF:  W1 [88 → 264]   W2 [264 → 88]   fan-in: 264"),
    (6.1,  "+ residual"),
    (5.1,  "Adam state per weight:  m (FP32) + v (FP32)"),
]
for (ypos, label) in rows:
    size = 10.0 if label.startswith(" ") else 11.0
    color = "#88ccff" if "Adam" in label else ("#aaddaa" if "residual" in label else C_TEXT)
    txt(ax, 5, ypos, label, size=size, color=color)

txt(ax, 5, 4.3, "× 4", size=16, bold=True, color=C_TITLE)

arrow(ax, 5, 3.5, 5, 3.15)

# ── OUTPUT SECTION ──
box(ax, 0.6, 1.9, 8.8, 1.1, C_OUT, edgecolor="#334466", lw=0.8, radius=0.04)
txt(ax, 1.5, 2.7, "OUTPUT", size=10, bold=True, color="#aaaacc", ha="left")
txt(ax, 5, 2.35, "logits = layerOut · Embedding.T   (weight-tied, no extra params)", size=11, color=C_TEXT)

arrow(ax, 5, 1.9, 5, 1.55)

# ── ZCU102 note ──
box(ax, 0.6, 0.5, 8.8, 0.9, "#1a0a0a", edgecolor="#663333", lw=0.8, radius=0.04)
txt(ax, 5, 0.95, "Target: Xilinx ZCU102   BRAM = 4.0 MB   |   334K BF16W = 3.34 MB  ✓  (660 KB headroom)", size=7.5, color="#ffaaaa")

# ── SRAM PANEL ──────────────────────────────────────────────────
ax2 = ax_sram
ax2.set_facecolor("#0a0a1a")
ax2.spines[:].set_visible(False)
ax2.tick_params(colors=C_TEXT)
for spine in ax2.spines.values():
    spine.set_edgecolor("#334466")

# Stacked bar: weights + moments
labels    = ["FP32 Adam", "BF16W Adam"]
weights   = [1.34, 0.67]   # MB
moments   = [2.67, 2.67]   # MB (FP32 in both)
total     = [w + m for w, m in zip(weights, moments)]
x         = [0, 1]

bars_w = ax2.bar(x, weights, color=C_BF16W, label="Weights", width=0.5, zorder=3)
bars_m = ax2.bar(x, moments, bottom=weights, color=C_MOMENTS, label="Moments (m+v)", width=0.5, zorder=3)

# Limit line
ax2.axhline(4.0, color=C_LIMIT, linewidth=2, linestyle="--", zorder=4, label="ZCU102 limit (4.0 MB)")

# Value labels on bars
for i, (w, m, t) in enumerate(zip(weights, moments, total)):
    ax2.text(i, w / 2, f"{w:.2f} MB", ha="center", va="center",
             fontsize=11, color="white", fontweight="bold", fontfamily="monospace")
    ax2.text(i, w + m / 2, f"{m:.2f} MB", ha="center", va="center",
             fontsize=11, color="white", fontweight="bold", fontfamily="monospace")
    ok = "✓" if t <= 4.0 else "✗"
    col = C_OK if t <= 4.0 else C_WARN
    ax2.text(i, t + 0.08, f"{t:.2f} MB {ok}", ha="center", va="bottom",
             fontsize=12, color=col, fontweight="bold", fontfamily="monospace")

ax2.set_xticks(x)
ax2.set_xticklabels(labels, color=C_TEXT, fontsize=12, fontfamily="monospace")
ax2.set_ylabel("SRAM (MB)", color=C_TEXT, fontsize=12, fontfamily="monospace")
ax2.set_ylim(0, 5.0)
ax2.set_title("SRAM Budget\n334K params, ZCU102", color=C_TITLE, fontsize=13,
               fontweight="bold", fontfamily="monospace")
ax2.tick_params(axis="y", colors=C_TEXT, labelsize=11)
ax2.yaxis.set_tick_params(labelcolor=C_TEXT)
ax2.set_facecolor("#0a0a1a")

legend = ax2.legend(loc="upper right", fontsize=11,
                    facecolor="#1a1a2e", edgecolor="#334466", labelcolor=C_TEXT)

plt.tight_layout(pad=1.5)

out_base = "c:/repos/_Neuro/neuro-fabric-research/docs/paper/fig/arch_334k"
plt.savefig(out_base + ".pdf", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig(out_base + ".png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_base}.pdf / .png")
