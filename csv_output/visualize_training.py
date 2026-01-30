"""
Visualize training results and generate performance analysis
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load training history
with open('models/training_history.json', 'r') as f:
    history = json.load(f)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('AeroGuard Training Results - 674 Flight Data Recorder Files', 
             fontsize=16, fontweight='bold')

# 1. Training and Validation Loss
ax1 = axes[0, 0]
epochs = range(1, len(history['train_loss']) + 1)
ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2, alpha=0.8)
ax1.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2, alpha=0.8)
ax1.axvline(x=history['best_epoch'] + 1, color='red', linestyle='--', 
           label=f'Best Model (Epoch {history["best_epoch"] + 1})', alpha=0.7)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Negative Log-Likelihood Loss', fontsize=12)
ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Validation RMSE
ax2 = axes[0, 1]
ax2.plot(epochs, history['val_rmse'], color='green', linewidth=2, alpha=0.8)
ax2.axhline(y=history['best_rmse'], color='red', linestyle='--', 
           label=f'Best RMSE: {history["best_rmse"]:.2f} flights', alpha=0.7)
ax2.axvline(x=history['best_epoch'] + 1, color='red', linestyle='--', alpha=0.7)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('RMSE (Remaining Flights)', fontsize=12)
ax2.set_title('Validation Performance', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Loss convergence (zoomed in on last 50 epochs)
ax3 = axes[1, 0]
zoom_start = max(0, len(epochs) - 50)
ax3.plot(epochs[zoom_start:], history['train_loss'][zoom_start:], 
        label='Train Loss', linewidth=2, alpha=0.8)
ax3.plot(epochs[zoom_start:], history['val_loss'][zoom_start:], 
        label='Validation Loss', linewidth=2, alpha=0.8)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('Loss Convergence (Last 50 Epochs)', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Training metrics summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
TRAINING SUMMARY
{'='*50}

Dataset Statistics:
• Total FDR Files Processed: 674 flights
• Training Sequences: 515
• Validation Sequences: 129
• Features Extracted: 70 (4 engines × sensors)
• Sequence Length: 30 flights

Model Architecture:
• Total Parameters: 656,450
• LSTM Hidden Dim: 128
• LSTM Layers: 2
• Attention Heads: 4

Training Configuration:
• Batch Size: 64
• Learning Rate: 0.001
• Optimizer: Adam
• Scheduler: ReduceLROnPlateau

Final Results:
• Best Validation RMSE: {history['best_rmse']:.2f} flights
• Best Epoch: {history['best_epoch'] + 1}/100
• Final Train Loss: {history['train_loss'][-1]:.4f}
• Final Val Loss: {history['val_loss'][-1]:.4f}

Model Saved To:
• models/best_model.pth (7.9 MB)
"""

ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
        fontsize=11, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('models/training_results.png', dpi=150, bbox_inches='tight')
print("✅ Training visualization saved to: models/training_results.png")

# Print summary to console
print("\n" + "="*60)
print("AEROGUARD TRAINING COMPLETE")
print("="*60)
print(f"✅ Processed 674 flight data recorder CSV files")
print(f"✅ Trained on 515 flight sequences (30 flights each)")
print(f"✅ Best Validation RMSE: {history['best_rmse']:.2f} flights")
print(f"✅ Model: 656K parameters, saved to models/best_model.pth")
print("="*60)
