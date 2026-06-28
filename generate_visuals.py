import matplotlib.pyplot as plt
import numpy as np

labels = ['Accuracy', 'Precision\n(REAL)', 'Recall\n(REAL)', 'Precision\n(FAKE)', 'Recall\n(FAKE)']
values = [94.94, 95, 93, 94, 97]
colors = ['#4f46e5', '#06b6d4', '#06b6d4', '#f59e0b', '#f59e0b']

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylim(85, 100)
ax.set_ylabel('Score (%)', fontsize=13)
ax.set_title('Fake News Detector — Model Performance', fontsize=15, fontweight='bold', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=90, color='red', linestyle='--', alpha=0.4, label='90% threshold')
ax.legend()

plt.tight_layout()
plt.savefig('performance_chart.png', dpi=150, bbox_inches='tight')
print("Performance chart saved!")
