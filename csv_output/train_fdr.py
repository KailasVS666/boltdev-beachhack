"""
AeroGuard Training Script for Flight Data Recorder (FDR) CSV Files

This script trains the AeroGuard model on real flight recorder data.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from aeroguard_model import AeroGuardModel, train_model
from utils import setup_logging
import config

# Setup logging
logger = setup_logging('INFO')


def load_fdr_file(filepath: Path) -> pd.DataFrame:
    """Load a single FDR CSV file"""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


def extract_engine_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract relevant engine health features from FDR data
    
    Key sensors for each engine (1-4):
    - EGT (Exhaust Gas Temperature)
    - N1 (Fan Speed)
    - N2 (Core Speed)
    - FF (Fuel Flow)
    - OIT (Oil Temperature)
    - OIP (Oil Pressure)
    """
    
    features = {}
    
    # Extract features for each of 4 engines
    for engine_num in range(1, 5):
        # Temperature sensors
        if f'EGT_{engine_num}' in df.columns:
            features[f'EGT_{engine_num}_mean'] = df[f'EGT_{engine_num}'].mean()
            features[f'EGT_{engine_num}_std'] = df[f'EGT_{engine_num}'].std()
            features[f'EGT_{engine_num}_max'] = df[f'EGT_{engine_num}'].max()
        
        # Speed sensors
        for sensor in [f'N1_{engine_num}', f'N2_{engine_num}']:
            if sensor in df.columns:
                features[f'{sensor}_mean'] = df[sensor].mean()
                features[f'{sensor}_std'] = df[sensor].std()
                features[f'{sensor}_max'] = df[sensor].max()
        
        # Fuel flow
        if f'FF_{engine_num}' in df.columns:
            features[f'FF_{engine_num}_mean'] = df[f'FF_{engine_num}'].mean()
            features[f'FF_{engine_num}_std'] = df[f'FF_{engine_num}'].std()
        
        # Oil parameters
        for oil_sensor in [f'OIT_{engine_num}', f'OIP_{engine_num}']:
            if oil_sensor in df.columns:
                features[f'{oil_sensor}_mean'] = df[oil_sensor].mean()
                features[f'{oil_sensor}_std'] = df[oil_sensor].std()
    
    # Flight parameters (environmental confounders)
    for param in ['ALT', 'TAS', 'MACH', 'TAT', 'SAT']:
        if param in df.columns:
            features[f'{param}_mean'] = df[param].mean()
            features[f'{param}_std'] = df[param].std()
    
    return pd.Series(features)


def process_fdr_dataset(data_dir: Path, max_files: int = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Process all FDR CSV files and create training dataset
    
    Args:
        data_dir: Directory containing CSV files
        max_files: Maximum number of files to process (for testing)
    
    Returns:
        features_df: DataFrame of extracted features
        rul_labels: RUL labels (flight sequence number)
    """
    logger.info("=" * 60)
    logger.info("PROCESSING FDR DATASET")
    logger.info("=" * 60)
    
    # Get all CSV files (excluding Python files)
    csv_files = sorted([f for f in data_dir.glob('*.csv')])
    
    if max_files:
        csv_files = csv_files[:max_files]
    
    logger.info(f"Found {len(csv_files)} FDR files to process")
    
    all_features = []
    all_rul = []
    
    # Process each flight
    for idx, csv_file in enumerate(tqdm(csv_files, desc="Processing flights")):
        df = load_fdr_file(csv_file)
        
        if df is None or len(df) < 10:
            continue
        
        # Extract features
        features = extract_engine_features(df)
        
        if features.isna().sum() > len(features) * 0.5:
            # Skip if more than 50% features are missing
            continue
        
        # Calculate RUL: higher for earlier flights, lower for later
        # Assuming flights are chronologically ordered by filename
        rul = len(csv_files) - idx
        
        all_features.append(features)
        all_rul.append(rul)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    rul_labels = np.array(all_rul)
    
    logger.info(f"Processed {len(features_df)} valid flights")
    logger.info(f"Feature dimensions: {features_df.shape}")
    logger.info(f"RUL range: {rul_labels.min()} to {rul_labels.max()}")
    
    return features_df, rul_labels


def create_sequences(features: np.ndarray, rul: np.ndarray, 
                     sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create time sequences from flight-level features
    
    Args:
        features: Flight-level features (n_flights, n_features)
        rul: RUL labels (n_flights,)
        sequence_length: Length of sequences for LSTM
    
    Returns:
        X_sequences: (n_sequences, sequence_length, n_features)
        y_rul: (n_sequences,) - RUL at the end of each sequence
    """
    n_flights, n_features = features.shape
    
    X_sequences = []
    y_sequences = []
    
    # Create overlapping sequences
    for i in range(len(features) - sequence_length):
        seq = features[i:i+sequence_length]
        target_rul = rul[i+sequence_length-1]  # RUL at end of sequence
        
        X_sequences.append(seq)
        y_sequences.append(target_rul)
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    logger.info(f"Created {len(X_sequences)} sequences of length {sequence_length}")
    
    return X_sequences, y_sequences


def main():
    """Main training function"""
    
    logger.info("=" * 60)
    logger.info("AEROGUARD FDR TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Configuration
    data_dir = Path('data')
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    # Step 1: Process FDR data
    logger.info("\n[1/5] Loading and processing FDR files...")
    features_df, rul_labels = process_fdr_dataset(data_dir, max_files=None)
    
    # Handle missing values
    features_df = features_df.fillna(features_df.mean())
    
    # Step 2: Normalize features
    logger.info("\n[2/5] Normalizing features...")
    scaler = RobustScaler()
    features_normalized = scaler.fit_transform(features_df.values)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, model_dir / 'feature_scaler.pkl')
    logger.info(f"Saved scaler to {model_dir / 'feature_scaler.pkl'}")
    
    # Step 3: Create sequences
    logger.info("\n[3/5] Creating time sequences...")
    X_sequences, y_rul = create_sequences(
        features_normalized, 
        rul_labels, 
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    # Step 4: Train/val split
    logger.info("\n[4/5] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_sequences, y_rul,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    logger.info(f"Training set: {X_train.shape[0]} sequences")
    logger.info(f"Validation set: {X_val.shape[0]} sequences")
    
    # Step 5: Train model
    logger.info("\n[5/5] Training AeroGuard model...")
    
    num_features = X_train.shape[2]
    model = AeroGuardModel(
        num_features=num_features,
        causal_hidden_dim=64,
        lstm_hidden_dim=128,
        lstm_layers=2,
        attention_heads=4
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    history = train_model(
        model=model,
        train_data={'X': X_train, 'y': y_train},
        val_data={'X': X_val, 'y': y_val},
        num_epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        save_dir=model_dir
    )
    
    # Save final results
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best RMSE: {history['best_rmse']:.2f} cycles")
    logger.info(f"Best epoch: {history['best_epoch'] + 1}")
    logger.info(f"Model saved to: {model_dir / 'best_model.pth'}")
    logger.info("=" * 60)
    
    # Save training history
    import json
    with open(model_dir / 'training_history.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_rmse': [float(x) for x in history['val_rmse']],
            'best_epoch': int(history['best_epoch']),
            'best_rmse': float(history['best_rmse'])
        }
        json.dump(history_serializable, f, indent=2)
    
    logger.info(f"Training history saved to: {model_dir / 'training_history.json'}")
    
    return history


if __name__ == '__main__':
    history = main()
