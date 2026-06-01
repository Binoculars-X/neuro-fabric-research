"""
Figure 2: Shakespeare 334K — Validation Loss Curves (exp003, 80K samples)
Generates fig2_loss_curves.pdf and fig2_loss_curves.png
NeuronFabric v1.0.2, BUG-006 fixed
"""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def parse_log(path):
    samples, train_loss, val_loss = [], [], []
    with open(path, encoding='utf-16', errors='replace') as f:
        for line in f:
            line = line.replace(',', '')
            m = re.match(r'\s*(\d+)\s+([\d.]+)\s+([\d.]+)', line)
            if m:
                samples.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
    return samples, train_loss, val_loss

s1, tl1, vl1 = parse_log(r'C:\repos\_Neuro\neuro-fabric\run\results\exp-gpu-adam-shakespeare-334k-80k.neuro.log')
s2, tl2, vl2 = parse_log(r'C:\repos\_Neuro\neuro-fabric\run\results\exp-cpu-adam-bf16w-shakespeare-334k-80k.neuro.log')

best1 = min(vl1)
best2 = min(vl2)

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot([x/1000 for x in s1], vl1, color='#1f77b4', linewidth=1.5,
        label=f'GPU Adam FP32  (best {best1:.4f})')
ax.plot([x/1000 for x in s2], vl2, color='#ff7f0e', linewidth=1.5, linestyle='--',
        label=f'CPU Adam BF16W (best {best2:.4f})')

ax.set_xlabel('Training samples (thousands)', fontsize=11)
ax.set_ylabel('Validation loss', fontsize=11)
ax.set_title('Shakespeare 334K — Validation Loss (NeuronFabric v1.0.2)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 80)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}K'))

plt.tight_layout()

base = r'C:\repos\_Neuro\neuro-fabric-research\docs\paper\fig\fig2_loss_curves'
plt.savefig(base + '.pdf', dpi=200, bbox_inches='tight')
print(f'Saved: {base}.pdf')
plt.savefig(base + '.png', dpi=150, bbox_inches='tight')
print(f'Saved: {base}.png')
