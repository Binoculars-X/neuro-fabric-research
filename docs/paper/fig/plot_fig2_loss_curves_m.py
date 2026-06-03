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
    found_marker = found_header = False
    col_train = col_eval = None
    version = ''
    for enc in ('utf-16', 'utf-8', 'latin-1'):
        try:
            with open(path, encoding=enc, errors='replace') as f:
                lines = f.readlines()
            break
        except Exception:
            continue
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if not version:
            m = re.search(r'(v\d+\.\d+\.\d+)(?:\+([0-9a-f]{7}))', s)
            if m:
                version = f'{m.group(1)}+{m.group(2)}'
            else:
                m2 = re.search(r'(v\d+\.\d+\.\d+)', s)
                if m2:
                    version = m2.group(1)
        if not found_marker:
            if s == '#DATA':
                found_marker = True
            continue
        if not found_header:
            if s.startswith('-'):
                continue
            headers = [h.strip().lower() for h in re.split(r'\s{2,}', s)]
            col_train = headers.index('train loss') if 'train loss' in headers else 2
            col_eval  = headers.index('eval loss')  if 'eval loss'  in headers else 3
            found_header = True
            continue
        if s.startswith('-'):
            continue
        parts = s.replace(',', '').split()
        try:
            samples.append(int(parts[0]))
            train_loss.append(float(parts[col_train]))
            val_loss.append(float(parts[col_eval]))
        except (ValueError, IndexError):
            continue
    return samples, train_loss, val_loss, version

s1, tl1, vl1, _       = parse_log(r'C:\repos\_Neuro\neuro-fabric-research\docs\journal\experiments\preprint\gpu-fp32-shakespeare-334k-b1-80k\checkpoint.neuro.log')
s2, tl2, vl2, version = parse_log(r'C:\repos\_Neuro\neuro-fabric-research\docs\journal\experiments\preprint\cpu-bf16w-shakespeare-334k-b1-80k\checkpoint.neuro.log')

best1 = min(vl1)
best2 = min(vl2)

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot([x/1000 for x in s1], vl1, color='#ff7f0e', linewidth=1.5, linestyle='--',
        label=f'GPU Adam FP32 oracle (best {best1:.4f})')
ax.plot([x/1000 for x in s2], vl2, color='#1f77b4', linewidth=1.5,
        label=f'CPU Adam BF16W (best {best2:.4f})')

ax.set_xlabel('Training samples (thousands)', fontsize=11)
ax.set_ylabel('Validation loss', fontsize=11)
ax.set_title('Shakespeare 334K — Validation Loss', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 80)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}K'))

plt.tight_layout()

if version:
    fig.text(0.98, 0.01, version, ha='right', va='bottom', fontsize=8,
             color='#333333', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#cccccc', edgecolor='none', alpha=0.85))

base = r'C:\repos\_Neuro\neuro-fabric-research\docs\paper\fig\fig2_loss_curves'
plt.savefig(base + '.pdf', dpi=200, bbox_inches='tight')
print(f'Saved: {base}.pdf')
plt.savefig(base + '.png', dpi=150, bbox_inches='tight')
print(f'Saved: {base}.png')
