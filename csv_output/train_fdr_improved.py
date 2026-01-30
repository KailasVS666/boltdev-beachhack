"""
AeroGuard IMPROVED Training Script - Quick Wins Version
CHANGES:
1. Physics-based feature extraction (trends, margins, efficiency)
2. Degradation-based RUL labeling (not filename-based)
3. More robust features with flight-phase awareness

Expected improvement: RMSE 360 -> 180-220
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
from scipy.stats import linregress

from aeroguard_model import AeroGuardModel, train_model
from utils import setup_logging
import config

logger = setup_logging('INFO')


def extract_engine_features_IMPROVED(df: pd.DataFrame) -> pd.DataFrame:
    """
    IMPROVED: Physics-based feature extraction with temporal patterns
    
    Key improvements:
    - Trend analysis (slopes indicate degradation)
    - Cruise-phase isolation (removes takeoff/landing confounding)
    - Efficiency metrics (cross-sensor ratios)
    - Margin to limits (proximity to failure thresholds)
    """
    
    features = {}
    
    # Identify flight phases (critical for removing operational noise)
    try:
        if 'ALT' in df.columns and 'VRTG' in df.columns:
            # Cruise: high altitude, low vertical speed
            cruise_mask = (pd.to_numeric(df['ALT'], errors='coerce') > 20000) & \
                         (pd.to_numeric(df['VRTG'], errors='coerce').abs() < 500)
            has_cruise = cruise_mask.sum() > 10
        else:
            cruise_mask = pd.Series([True] * len(df))
            has_cruise = True
    except:
        cruise_mask = pd.Series([True] * len(df))
        has_cruise = True
    
    # Process each engine
    for engine_num in range(1, 5):
        
        # === EGT (Exhaust Gas Temperature) - PRIMARY degradation indicator ===
        egt_col = f'EGT_{engine_num}'
        if egt_col in df.columns:
            try:
                egt_series = pd.to_numeric(df[egt_col], errors='coerce').dropna()
                
                if len(egt_series) > 0:
                    # Basic statistics
                    features[f'EGT_{engine_num}_mean'] = egt_series.mean()
                    features[f'EGT_{engine_num}_std'] = egt_series.std()
                    features[f'EGT_{engine_num}_max'] = egt_series.max()
                    
                    # CRUISE-ONLY statistics (key innovation!)
                    if has_cruise and cruise_mask.sum() > 0:
                        egt_cruise = egt_series[cruise_mask[egt_series.index]]
                        if len(egt_cruise) > 0:
                            features[f'EGT_{engine_num}_cruise_mean'] = egt_cruise.mean()
                            features[f'EGT_{engine_num}_cruise_std'] = egt_cruise.std()
                    
                    # TREND: Rising EGT = degradation
                    if len(egt_series) > 5:
                        x = np.arange(len(egt_series))
                        y = egt_series.values
                        slope, _, _, _, _ = linregress(x, y)
                        features[f'EGT_{engine_num}_trend'] = slope
                    
                    # MARGIN: Distance from redline (typical limit ~950°C)
                    EGT_LIMIT = 950
                    features[f'EGT_{engine_num}_margin'] = EGT_LIMIT - egt_series.max()
            except:
                pass
        
        # === N1 (Fan Speed) ===
        n1_col = f'N1_{engine_num}'
        if n1_col in df.columns:
            try:
                n1_series = pd.to_numeric(df[n1_col], errors='coerce').dropna()
                if len(n1_series) > 0:
                    features[f'N1_{engine_num}_mean'] = n1_series.mean()
                    features[f'N1_{engine_num}_std'] = n1_series.std()
                    
                    if has_cruise and cruise_mask.sum() > 0:
                        n1_cruise = n1_series[cruise_mask[n1_series.index]]
                        if len(n1_cruise) > 0:
                            features[f'N1_{engine_num}_cruise_mean'] = n1_cruise.mean()
            except:
                pass
        
        # === N2 (Core Speed) ===
        n2_col = f'N2_{engine_num}'
        if n2_col in df.columns:
            try:
                n2_series = pd.to_numeric(df[n2_col], errors='coerce').dropna()
                if len(n2_series) > 0:
                    features[f'N2_{engine_num}_mean'] = n2_series.mean()
                    features[f'N2_{engine_num}_std'] = n2_series.std()
            except:
                pass
        
        # === FF (Fuel Flow) ===
        ff_col = f'FF_{engine_num}'
        if ff_col in df.columns:
            try:
                ff_series = pd.to_numeric(df[ff_col], errors='coerce').dropna()
                if len(ff_series) > 0:
                    features[f'FF_{engine_num}_mean'] = ff_series.mean()
                    features[f'FF_{engine_num}_std'] = ff_series.std()
            except:
                pass
        
        # === EFFICIENCY: FF/N1 ratio (higher = less efficient = degraded) ===
        if f'FF_{engine_num}' in df.columns and f'N1_{engine_num}' in df.columns:
            try:
                ff = pd.to_numeric(df[f'FF_{engine_num}'], errors='coerce')
                n1 = pd.to_numeric(df[f'N1_{engine_num}'], errors='coerce')
                
                # Only calculate when both have valid values and N1 > 0
                valid_mask = (~ff.isna()) & (~n1.isna()) & (n1 > 10)
                if valid_mask.sum() > 0:
                    efficiency = (ff[valid_mask] / n1[valid_mask]).mean()
                    features[f'FF_N1_ratio_{engine_num}'] = efficiency
            except:
                pass
        
        # === Oil Temperature ===
        oit_col = f'OIT_{engine_num}'
        if oit_col in df.columns:
            try:
                oit_series = pd.to_numeric(df[oit_col], errors='coerce').dropna()
                if len(oit_series) > 0:
                    features[f'OIT_{engine_num}_mean'] = oit_series.mean()
                    features[f'OIT_{engine_num}_max'] = oit_series.max()
            except:
                pass
        
        # === Oil Pressure ===
        oip_col = f'OIP_{engine_num}'
        if oip_col in df.columns:
            try:
                oip_series = pd.to_numeric(df[oip_col], errors='coerce').dropna()
                if len(oip_series) > 0:
                    features[f'OIP_{engine_num}_mean'] = oip_series.mean()
                    features[f'OIP_{engine_num}_min'] = oip_series.min()
            except:
                pass
        
        # === Vibration ===
        vib_col = f'VIB_{engine_num}'
        if vib_col in df.columns:
            try:
                vib_series = pd.to_numeric(df[vib_col], errors='coerce').dropna()
                if len(vib_series) > 0:
                    features[f'VIB_{engine_num}_mean'] = vib_series.mean()
                    features[f'VIB_{engine_num}_std'] = vib_series.std()
                    features[f'VIB_{engine_num}_max'] = vib_series.max()
            except:
                pass
    
    # === CROSS-ENGINE FEATURES ===
    # Engine imbalance (engines should behave similarly)
    egt_cols = [f'EGT_{i}' for i in range(1, 5) if f'EGT_{i}' in df.columns]
    if len(egt_cols) >= 2:
        try:
            egt_values = [pd.to_numeric(df[col], errors='coerce').mean() for col in egt_cols]
            egt_values = [x for x in egt_values if not np.isnan(x)]
            if len(egt_values) >= 2:
                features['EGT_imbalance'] = np.std(egt_values)
                features['EGT_range'] = max(egt_values) - min(egt_values)
        except:
            pass
    
    # === FLIGHT PARAMETERS (environmental confounders) ===
    for param in ['ALT', 'TAS', 'MACH', 'TAT', 'SAT']:
        if param in df.columns:
            try:
                series = pd.to_numeric(df[param], errors='coerce').dropna()
                if len(series) > 0:
                    features[f'{param}_mean'] = series.mean()
                    features[f'{param}_std'] = series.std()
            except:
                pass
    
    return pd.Series(features)


def calculate_degradation_score(df: pd.DataFrame) -> float:
    """
    IMPROVED: Physics-based degradation scoring
    
    Higher score = healthier engine
    Lower score = more degraded
    
    Based on:
    - EGT margin (most important)
    - Vibration levels
    - Efficiency metrics
    """
    
    health_score = 1.0  # Start at perfect health
    weights = []
    
    # Check Engine 1 (typically best instrumented)
    for engine_num in [1, 2, 3, 4]:
        
        # EGT-based health (weight: 0.5)
        if f'EGT_{engine_num}' in df.columns:
            try:
                avg_egt = pd.to_numeric(df[f'EGT_{engine_num}'], errors='coerce').mean()
                if not np.isnan(avg_egt):
                    # Normalize: 600°C (new) to 950°C (limit)
                    # Health decreases as EGT increases
                    egt_health = 1.0 - np.clip((avg_egt - 600) / 350, 0, 1)
                    health_score *= (0.5 * egt_health + 0.5)  # Don't over-penalize
                    weights.append(0.5)
            except:
                pass
        
        # Vibration-based health (weight: 0.3)
        if f'VIB_{engine_num}' in df.columns:
            try:
                avg_vib = pd.to_numeric(df[f'VIB_{engine_num}'], errors='coerce').mean()
                if not np.isnan(avg_vib) and avg_vib >= 0:
                    # Lower vibration = healthier (typical range 0-100)
                    vib_health = 1.0 - np.clip(avg_vib / 100, 0, 1)
                    health_score *= (0.7 * vib_health + 0.3)
                    weights.append(0.3)
            except:
                pass
    
    # Normalize by number of indicators found
    if len(weights) > 0:
        return health_score / len(weights)
    else:
        return 0.5  # Unknown = middle health


def process_fdr_dataset_IMPROVED(data_dir: Path, max_files: int = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    IMPROVED: Process with better features and degradation-based RUL
    """
    
    logger.info("=" * 60)
    logger.info("IMPROVED FDR PROCESSING")
    logger.info("=" * 60)
    
    csv_files = sorted([f for f in data_dir.glob('*.csv')])
    
    if max_files:
        csv_files = csv_files[:max_files]
    
    logger.info(f"Found {len(csv_files)} FDR files to process")
    
    all_features = []
    all_health_scores = []
    
    # Pass 1: Extract features and calculate health scores
    logger.info("\nPass 1: Extracting features and health scores...")
    for idx, csv_file in enumerate(tqdm(csv_files, desc="Processing flights")):
        df = pd.read_csv(csv_file)
        
        if df is None or len(df) < 10:
            continue
        
        # Extract improved features
        features = extract_engine_features_IMPROVED(df)
        
        if features.isna().sum() > len(features) * 0.7:
            continue
        
        # Calculate health score
        health = calculate_degradation_score(df)
        
        all_features.append(features)
        all_health_scores.append(health)
    
    features_df = pd.DataFrame(all_features)
    health_scores = np.array(all_health_scores)
    
    # Pass 2: Convert health scores to RUL
    # IMPROVED: RUL based on degradation trajectory, not file order
    logger.info("\nPass 2: Converting health scores to RUL labels...")
    
    # Sort by health (healthiest first)
    health_order = np.argsort(health_scores)[::-1]  # Descending
    
    # Assign RUL: healthiest = max RUL, most degraded = low RUL
    rul_labels = np.zeros(len(health_scores))
    max_rul = len(csv_files)
    
    for rank, idx in enumerate(health_order):
        # Linear mapping: best health = max RUL, worst = 1
        rul_labels[idx] = max_rul - (rank * max_rul / len(health_scores))
    
    logger.info(f"\nProcessed {len(features_df)} valid flights")
    logger.info(f"Feature dimensions: {features_df.shape}")
    logger.info(f"RUL range: {rul_labels.min():.1f} to {rul_labels.max():.1f}")
    logger.info(f"Health score range: {health_scores.min():.3f} to {health_scores.max():.3f}")
    
    # Validation: Check correlation
    correlation = np.corrcoef(health_scores, rul_labels)[0, 1]
    logger.info(f"Health-RUL correlation: {correlation:.3f} (should be high)")
    
    return features_df, rul_labels


def create_sequences(features: np.ndarray, rul: np.ndarray, 
                     sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Same as before - creates time sequences"""
    
    n_flights, n_features = features.shape
    
    X_sequences = []
    y_sequences = []
    
    for i in range(len(features) - sequence_length):
        seq = features[i:i+sequence_length]
        target_rul = rul[i+sequence_length-1]
        
        X_sequences.append(seq)
        y_sequences.append(target_rul)
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    logger.info(f"Created {len(X_sequences)} sequences of length {sequence_length}")
    
    return X_sequences, y_sequences


def main():
    """IMPROVED training pipeline"""
    
    logger.info("=" * 60)
    logger.info("AEROGUARD IMPROVED TRAINING PIPELINE")
    logger.info("=" * 60)
    
    data_dir = Path('data')
    model_dir = Path('models_improved')
    model_dir.mkdir(exist_ok=True)
    
    # Step 1: Process with IMPROVED features
    logger.info("\n[1/5] Loading FDR files with IMPROVED feature extraction...")
    features_df, rul_labels = process_fdr_dataset_IMPROVED(data_dir, max_files=None)
    
    # Handle missing values
    features_df = features_df.fillna(features_df.mean())
    
    # Step 2: Normalize
    logger.info("\n[2/5] Normalizing features...")
    scaler = RobustScaler()
    features_normalized = scaler.fit_transform(features_df.values)
    
    import joblib
    joblib.dump(scaler, model_dir / 'feature_scaler.pkl')
    
    # Step 3: Create sequences
    logger.info("\n[3/5] Creating time sequences...")
    X_sequences, y_rul = create_sequences(
        features_normalized, 
        rul_labels, 
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    # Step 4: Split
    logger.info("\n[4/5] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_sequences, y_rul,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    logger.info(f"Training set: {X_train.shape[0]} sequences")
    logger.info(f"Validation set: {X_val.shape[0]} sequences")
    logger.info(f"Features per timestep: {X_train.shape[2]}")
    
    # Step 5: Train
    logger.info("\n[5/5] Training IMPROVED model...")
    
    num_features = X_train.shape[2]
    model = AeroGuardModel(
        num_features=num_features,
        causal_hidden_dim=64,
        lstm_hidden_dim=128,
        lstm_layers=2,
        attention_heads=4
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    history = train_model(
        model=model,
        train_data={'X': X_train, 'y': y_train},
        val_data={'X': X_val, 'y': y_val},
        num_epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        save_dir=model_dir
    )
    
    # Results
    logger.info("\n" + "=" * 60)
    logger.info("IMPROVED TRAINING COMPLETE!")
    logger.info(f"Best RMSE: {history['best_rmse']:.2f} flights")
    logger.info(f"Previous RMSE: 360.05 flights")
    logger.info(f"Improvement: {((360.05 - history['best_rmse']) / 360.05 * 100):.1f}%")
    logger.info("=" * 60)
    
    # Save history
    import json
    with open(model_dir / 'training_history.json', 'w') as f:
        history_serializable = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_rmse': [float(x) for x in history['val_rmse']],
            'best_epoch': int(history['best_epoch']),
            'best_rmse': float(history['best_rmse'])
        }
        json.dump(history_serializable, f, indent=2)
    
    return history


if __name__ == '__main__':
    history = main()
