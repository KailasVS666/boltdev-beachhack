"""
AeroGuard Data Processor
Load, preprocess, and engineer features from turbofan sensor data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional, List
from pathlib import Path

import config
from utils import logger, validate_sensor_data


class TurbofanDataProcessor:
    """
    Process NASA C-MAPSS turbofan degradation data for AeroGuard model
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize data processor
        
        Args:
            data_path: Path to turbofan dataset file
        """
        self.data_path = data_path
        self.scaler = RobustScaler()  # Robust to outliers
        self.feature_columns = None
        self.max_rul = 125  # Cap RUL at 125 cycles (common practice)
        
    def load_turbofan_data(
        self, 
        filepath: Optional[Path] = None,
        dataset_type: str = 'train'
    ) -> pd.DataFrame:
        """
        Load NASA C-MAPSS format data
        
        Args:
            filepath: Path to data file (space-separated text file)
            dataset_type: 'train' or 'test'
        
        Returns:
            DataFrame with sensor readings
        """
        filepath = filepath or self.data_path
        
        if filepath is None:
            raise ValueError("No data path provided")
        
        logger.info(f"Loading {dataset_type} data from {filepath}")
        
        # NASA C-MAPSS format: space-separated, no headers
        df = pd.read_csv(
            filepath,
            sep=r'\s+',  # Multiple spaces
            header=None,
            names=config.SENSOR_COLUMNS
        )
        
        logger.info(f"Loaded {len(df)} records for {df['unit_number'].nunique()} units")
        
        # Validate data
        is_valid, errors = validate_sensor_data(df)
        if not is_valid:
            logger.warning(f"Data validation warnings: {errors}")
        
        return df
    
    def calculate_rul(
        self, 
        df: pd.DataFrame, 
        max_rul: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate Remaining Useful Life (RUL) for each observation
        
        For training data: RUL = max_cycle - current_cycle (capped at max_rul)
        For test data: RUL must be provided separately
        
        Args:
            df: DataFrame with 'unit_number' and 'time_cycles'
            max_rul: Maximum RUL value (for piecewise linear degradation)
        
        Returns:
            DataFrame with 'RUL' column added
        """
        max_rul = max_rul or self.max_rul
        
        logger.info("Calculating RUL targets")
        
        # Group by engine unit
        df_rul = df.copy()
        
        # Calculate max cycle for each unit
        max_cycles = df_rul.groupby('unit_number')['time_cycles'].max()
        
        # RUL = max_cycle - current_cycle
        df_rul['RUL'] = df_rul.apply(
            lambda row: max_cycles[row['unit_number']] - row['time_cycles'],
            axis=1
        )
        
        # Cap RUL at max_rul (piecewise linear degradation model)
        # Assumption: no significant degradation in early life
        df_rul['RUL'] = df_rul['RUL'].clip(upper=max_rul)
        
        logger.info(f"RUL calculated - Range: [{df_rul['RUL'].min():.0f}, {df_rul['RUL'].max():.0f}]")
        
        return df_rul
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for enhanced predictive power
        
        Features:
        - Rolling statistics (mean, std, slope)
        - Sensor deltas (rate of change)
        - Physics-based ratios
        - Interaction terms
        
        Args:
            df: DataFrame with raw sensor data
        
        Returns:
            DataFrame with additional engineered features
        """
        logger.info("Engineering features")
        
        df_feat = df.copy()
        
        # Sort by unit and time for rolling calculations
        df_feat = df_feat.sort_values(['unit_number', 'time_cycles'])
        
        # 1. Rolling statistics (5-cycle window)
        window = 5
        rolling_cols = config.TEMPERATURE_SENSORS + config.PRESSURE_SENSORS + config.SPEED_SENSORS
        
        for col in rolling_cols:
            # Rolling mean
            df_feat[f'{col}_roll_mean_{window}'] = df_feat.groupby('unit_number')[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Rolling std
            df_feat[f'{col}_roll_std_{window}'] = df_feat.groupby('unit_number')[col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # 2. Rate of change (delta from previous cycle)
        for col in rolling_cols:
            df_feat[f'{col}_delta'] = df_feat.groupby('unit_number')[col].diff()
        
        # 3. Physics-based derived features
        # Thermal efficiency proxy
        if all(col in df_feat.columns for col in ['T30', 'T2']):
            df_feat['thermal_efficiency'] = (df_feat['T30'] - df_feat['T2']) / (df_feat['T2'] + 1e-8)
        
        # Pressure ratio
        if all(col in df_feat.columns for col in ['P30', 'P2']):
            df_feat['pressure_ratio'] = df_feat['P30'] / (df_feat['P2'] + 1e-8)
        
        # Specific fuel consumption proxy
        if all(col in df_feat.columns for col in ['phi', 'Nc']):
            df_feat['sfc_proxy'] = df_feat['phi'] / (df_feat['Nc'] + 1e-8)
        
        # Temperature spread (indicates imbalance)
        if all(col in df_feat.columns for col in ['T30', 'T50']):
            df_feat['temp_spread'] = np.abs(df_feat['T30'] - df_feat['T50'])
        
        # 4. Operational regime indicators (binned operational settings)
        if 'op_setting_1' in df_feat.columns:
            df_feat['altitude_bin'] = pd.cut(df_feat['op_setting_1'], bins=5, labels=False)
        
        if 'op_setting_2' in df_feat.columns:
            df_feat['mach_bin'] = pd.cut(df_feat['op_setting_2'], bins=5, labels=False)
        
        # Fill NaN from delta/rolling calculations
        df_feat = df_feat.fillna(method='bfill').fillna(0)
        
        num_new_features = len(df_feat.columns) - len(df.columns)
        logger.info(f"Created {num_new_features} engineered features")
        
        return df_feat
    
    def remove_environmental_noise(
        self, 
        df: pd.DataFrame,
        causal_adjustments: Optional[Dict[str, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Apply causal adjustments to remove environmental confounding
        
        This is a placeholder for integration with causal_engine.py
        The causal engine will compute adjustments for each sensor
        
        Args:
            df: DataFrame with features
            causal_adjustments: Dictionary of {sensor: adjustment_values}
        
        Returns:
            DataFrame with causally-adjusted sensor readings
        """
        if causal_adjustments is None:
            logger.info("No causal adjustments provided - using raw sensor data")
            return df
        
        logger.info("Applying causal adjustments to remove environmental noise")
        
        df_adjusted = df.copy()
        
        for sensor, adjustments in causal_adjustments.items():
            if sensor in df_adjusted.columns:
                df_adjusted[f'{sensor}_adjusted'] = df_adjusted[sensor] - adjustments
        
        return df_adjusted
    
    def build_sequences(
        self, 
        df: pd.DataFrame,
        sequence_length: int = config.SEQUENCE_LENGTH,
        stride: int = config.SEQUENCE_STRIDE,
        target_col: str = 'RUL'
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create time-series sequences for LSTM input
        
        Args:
            df: DataFrame with features and target
            sequence_length: Number of time steps to look back
            stride: Sliding window stride
            target_col: Name of target column
        
        Returns:
            (X_sequences, y_targets, feature_names)
            X_sequences: shape (num_sequences, sequence_length, num_features)
            y_targets: shape (num_sequences,)
        """
        logger.info(f"Building sequences with length={sequence_length}, stride={stride}")
        
        # Exclude metadata columns
        exclude_cols = ['unit_number', 'time_cycles', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_columns = feature_cols
        
        X_sequences = []
        y_targets = []
        
        # Process each engine unit separately
        for unit in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit].sort_values('time_cycles')
            
            unit_features = unit_data[feature_cols].values
            unit_targets = unit_data[target_col].values
            
            # Create sliding windows
            for i in range(0, len(unit_data) - sequence_length + 1, stride):
                X_sequences.append(unit_features[i:i+sequence_length])
                y_targets.append(unit_targets[i+sequence_length-1])  # Predict at end of sequence
        
        X_sequences = np.array(X_sequences)
        y_targets = np.array(y_targets)
        
        logger.info(f"Created {len(X_sequences)} sequences with {len(feature_cols)} features")
        
        return X_sequences, y_targets, feature_cols
    
    def normalize_features(
        self, 
        X: np.ndarray, 
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize features using RobustScaler
        
        Args:
            X: Feature array (sequences, timesteps, features) or (samples, features)
            fit: Whether to fit the scaler (True for training, False for test)
        
        Returns:
            Normalized feature array
        """
        original_shape = X.shape
        
        # Reshape to 2D for scaling
        if len(original_shape) == 3:
            X_reshaped = X.reshape(-1, original_shape[-1])
        else:
            X_reshaped = X
        
        if fit:
            logger.info("Fitting scaler on training data")
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)
        
        # Reshape back to original
        if len(original_shape) == 3:
            X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled
    
    def prepare_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = config.VALIDATION_SPLIT,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and validation sets
        
        Args:
            X: Feature sequences
            y: Target values
            test_size: Fraction for validation
            random_state: Random seed
        
        Returns:
            (X_train, X_val, y_train, y_val)
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        logger.info(f"Train/Val split: {len(X_train)} / {len(X_val)} samples")
        
        return X_train, X_val, y_train, y_val
    
    def process_pipeline(
        self, 
        train_filepath: Path,
        test_filepath: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Complete data processing pipeline
        
        Args:
            train_filepath: Path to training data
            test_filepath: Optional path to test data
        
        Returns:
            Dictionary with processed train/val/test data
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA PROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # 1. Load training data
        train_df = self.load_turbofan_data(train_filepath, 'train')
        
        # 2. Calculate RUL
        train_df = self.calculate_rul(train_df)
        
        # 3. Engineer features
        train_df = self.engineer_features(train_df)
        
        # 4. Build sequences
        X_train_full, y_train_full, feature_names = self.build_sequences(train_df)
        
        # 5. Normalize
        X_train_full = self.normalize_features(X_train_full, fit=True)
        
        # 6. Train/val split
        X_train, X_val, y_train, y_val = self.prepare_train_test_split(
            X_train_full, y_train_full
        )
        
        result = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_names': feature_names,
            'scaler': self.scaler
        }
        
        # Process test data if provided
        if test_filepath is not None:
            test_df = self.load_turbofan_data(test_filepath, 'test')
            test_df = self.engineer_features(test_df)
            
            # Note: For test data, RUL might need to be loaded separately
            # This depends on the dataset format
            
            result['test_df'] = test_df
        
        logger.info("=" * 60)
        logger.info("DATA PROCESSING COMPLETE")
        logger.info("=" * 60)
        
        return result


# ==================== HELPER FUNCTIONS ====================
def load_rul_test_labels(filepath: Path) -> pd.DataFrame:
    """
    Load RUL labels for test dataset (separate file in C-MAPSS format)
    
    Args:
        filepath: Path to RUL test labels file
    
    Returns:
        DataFrame with unit_number and RUL
    """
    rul_labels = pd.read_csv(filepath, header=None, names=['RUL'])
    rul_labels['unit_number'] = rul_labels.index + 1
    return rul_labels


def merge_test_rul(test_df: pd.DataFrame, rul_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Merge RUL labels with test data
    
    For each unit, add RUL to the LAST cycle, then calculate backwards
    
    Args:
        test_df: Test dataset
        rul_labels: RUL values for last cycle of each unit
    
    Returns:
        Test DataFrame with RUL column
    """
    test_df_rul = test_df.copy()
    
    for unit in test_df_rul['unit_number'].unique():
        unit_data = test_df_rul[test_df_rul['unit_number'] == unit]
        max_cycle = unit_data['time_cycles'].max()
        
        # Get RUL at last cycle
        rul_at_end = rul_labels[rul_labels['unit_number'] == unit]['RUL'].values[0]
        
        # Calculate RUL for all cycles
        test_df_rul.loc[test_df_rul['unit_number'] == unit, 'RUL'] = (
            rul_at_end + (max_cycle - test_df_rul.loc[test_df_rul['unit_number'] == unit, 'time_cycles'])
        )
    
    return test_df_rul


if __name__ == '__main__':
    # Example usage
    processor = TurbofanDataProcessor()
    
    # Assuming NASA C-MAPSS data in data directory
    train_file = config.DATA_DIR / 'train_FD001.txt'
    
    if train_file.exists():
        data = processor.process_pipeline(train_file)
        print(f"\nProcessed data shapes:")
        print(f"  X_train: {data['X_train'].shape}")
        print(f"  y_train: {data['y_train'].shape}")
        print(f"  X_val: {data['X_val'].shape}")
        print(f"  y_val: {data['y_val'].shape}")
        print(f"  Features: {len(data['feature_names'])}")
    else:
        print(f"Sample data not found at {train_file}")
        print("Please download NASA C-MAPSS dataset")
