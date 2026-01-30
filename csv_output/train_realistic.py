"""
Train with EGT-based realistic failure labels (flight 464 = failure point)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import torch
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import json

sys.path.append('.')
from aeroguard_model import AeroGuardModel, train_model
from utils import setup_logging
from train_fdr_improved import extract_engine_features_IMPROVED, create_sequences
import config

logger = setup_logging('INFO')

print("=" * 60)
print("TRAINING WITH EGT-BASED REALISTIC FAILURE LABELS")
print("Failure identified at: Flight 464 (666200405030835.csv)")
print("=" * 60)

# Load realistic labels from our analysis
labels_df = pd.read_csv('realistic_labels.csv')
print(f"\nLoaded {len(labels_df)} realistic RUL labels")
print(f"RUL range: {labels_df['rul'].min()} to {labels_df['rul'].max()}")
print(f"Mean health: {labels_df['health_score'].mean():.3f}")
print(f"Failure health: {labels_df['health_score'].min():.3f}")

# Process all FDR files
data_dir = Path('data')
csv_files = sorted(list(data_dir.glob('*.csv')))
print(f"\nProcessing {len(csv_files)} CSV files...")

all_features = []
for idx, csv_file in enumerate(tqdm(csv_files, desc="Extracting features")):
    try:
        df = pd.read_csv(csv_file)
        if len(df) < 10:
            continue
        
        features = extract_engine_features_IMPROVED(df)
        if features.isna().sum() < len(features) * 0.7:
            all_features.append(features)
    except Exception as e:
        logger.warning(f"Skipped {csv_file.name}: {e}")
        continue

features_df = pd.DataFrame(all_features)
features_df = features_df.fillna(features_df.mean())

# Use the realistic RUL labels
rul_labels = labels_df['rul'].values[:len(features_df)]

print(f"\nFeatures extracted: {features_df.shape}")
print(f"Using realistic RUL labels based on EGT degradation")

# Normalize
print("\nNormalizing features...")
scaler = RobustScaler()
features_normalized = scaler.fit_transform(features_df.values)

model_dir = Path('models_realistic')
model_dir.mkdir(exist_ok=True)
joblib.dump(scaler, model_dir / 'feature_scaler.pkl')

# Create sequences
print("\nCreating time sequences...")
X_sequences, y_rul = create_sequences(features_normalized, rul_labels, sequence_length=30)

# Split
print("\nSplitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    X_sequences, y_rul, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training: {X_train.shape[0]} sequences")
print(f"Validation: {X_val.shape[0]} sequences")

# Train
print("\nTraining model...")
num_features = X_train.shape[2]
model = AeroGuardModel(
    num_features=num_features,
    causal_hidden_dim=64,
    lstm_hidden_dim=128,
    lstm_layers=2,
    attention_heads=4
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

history = train_model(
    model=model,
    train_data={'X': X_train, 'y': y_train},
    val_data={'X': X_val, 'y': y_val},
    num_epochs=100,
    batch_size=64,
    learning_rate=0.001,
    save_dir=model_dir
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE WITH REALISTIC LABELS!")
print(f"Best RMSE: {history['best_rmse']:.2f} flights")
print(f"Previous (chronological labels): 313.02 flights")
if history['best_rmse'] < 313.02:
    improvement = ((313.02 - history['best_rmse']) / 313.02 * 100)
    print(f"Improvement: {improvement:.1f}% BETTER!")
else:
    diff = history['best_rmse'] - 313.02
    print(f"Difference: +{diff:.1f} flights (realistic labels may trade RMSE for realism)")
print("=" * 60)

# Save
with open(model_dir / 'training_history.json', 'w') as f:
    json.dump({
        'train_loss': [float(x) for x in history['train_loss']],
        'val_loss': [float(x) for x in history['val_loss']],
        'val_rmse': [float(x) for x in history['val_rmse']],
        'best_epoch': int(history['best_epoch']),
        'best_rmse': float(history['best_rmse']),
        'Label_type': 'realistic_egt_based',
        'failure_flight': 464
    }, f, indent=2)

print(f"\nModel saved to {model_dir}/best_model.pth")
