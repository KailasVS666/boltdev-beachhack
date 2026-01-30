"""
Quick training script with optimized hyperparameters
Attempts to improve RMSE beyond 313 by tuning learning rate, batch size, and architecture
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from train_fdr_improved import main as train_improved
import config

# Override config with tuned hyperparameters
config.LEARNING_RATE = 0.0005  # Lower learning rate for better convergence
config.BATCH_SIZE = 32  # Smaller batch for better generalization
config.LSTM_HIDDEN_DIM = 256  # Bigger model
config.NUM_EPOCHS = 150  # More epochs
config.EARLY_STOPPING_PATIENCE = 25  # More patience

print("=" * 60)
print("OPTIMIZED TRAINING RUN")
print("=" * 60)
print(f"Learning Rate: {config.LEARNING_RATE}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"LSTM Hidden: {config.LSTM_HIDDEN_DIM}")
print(f"Max Epochs: {config.NUM_EPOCHS}")
print("=" * 60)

if __name__ == '__main__':
    history = train_improved()
