import re
import matplotlib.pyplot as plt

log_file = "exp-gpu-adam-tinystories-700k-1500k.neuro.log"

samples, train_loss, eval_loss = [], [], []

with open(log_file, encoding='utf-16') as f:
    for line in f:
        m = re.match(r'\s*([\d,]+)\s+([\d.]+)\s+([\d.]+)', line)
        if m:
            s = int(m.group(1).replace(',', ''))
            if s > 0:
                samples.append(s / 1000)
                train_loss.append(float(m.group(2)))
                eval_loss.append(float(m.group(3)))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(samples, train_loss, color='steelblue', linewidth=1.2, label='Train loss')
ax.plot(samples, eval_loss, color='darkorange', linewidth=1.2, linestyle='--', label=f'Eval loss (best {min(eval_loss):.4f})')
ax.set_xlabel('Training samples (thousands)')
ax.set_ylabel('Loss')
ax.set_title('TinyStories 700K — GPU Adam FP32 (1.5M samples)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('exp004_loss_curves.pdf')
plt.savefig('exp004_loss_curves.png', dpi=150)
print(f"Best eval loss: {min(eval_loss):.4f} @ {samples[eval_loss.index(min(eval_loss))]:.0f}K samples")
print("Saved: exp004_loss_curves.pdf / .png")
