
































































































"""
AeroGuard Predictive Model
Main ML architecture with causal sensor fusion and probabilistic RUL prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path

import config
from utils import logger, save_model_checkpoint


# ==================== DATASET ====================
class TurbofanDataset(Dataset):
    """PyTorch dataset for turbofan sequences"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Sequences (num_samples, seq_length, num_features)
            y: Targets (num_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== MODEL COMPONENTS ====================
class CausalAttentionLayer(nn.Module):
    """
    Attention mechanism weighted by causal relationships
    Implements the "Causal Sensor Fusion" concept
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 4):
        """
        Args:
            feature_dim: Dimension of input features
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Multi-head attention components
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.fc_out = nn.Linear(feature_dim, feature_dim)
        
        # Learnable causal weights (can be initialized from causal graph)
        self.causal_weights = nn.Parameter(torch.ones(1, 1, feature_dim))
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        
        Returns:
            Attention-weighted features
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
        
        # Final linear layer
        out = self.fc_out(out)
        
        # Weight by learned causal importance
        out = out * self.causal_weights
        
        return out


class TemporalFeatureExtractor(nn.Module):
    """
    Bidirectional LSTM to capture degradation patterns over time
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = config.LSTM_HIDDEN_DIM,
        num_layers: int = config.LSTM_LAYERS,
        dropout: float = config.DROPOUT_RATE
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Bidirectional doubles the output dim
        self.output_dim = hidden_dim * 2
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        
        Returns:
            LSTM outputs and final hidden state
        """
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last time step of forward and backward
        # hidden: (num_layers * 2, batch, hidden_dim)
        hidden_fwd = hidden[-2, :, :]  # Forward LSTM last layer
        hidden_bwd = hidden[-1, :, :]  # Backward LSTM last layer
        
        # Concatenate forward and backward
        final_hidden = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        final_hidden = self.dropout(final_hidden)
        
        return lstm_out, final_hidden


class ProbabilisticHead(nn.Module):
    """
    Output layer for probabilistic RUL prediction with uncertainty
    Uses Monte Carlo Dropout for uncertainty quantification
    """
    
    def __init__(self, input_dim: int, dropout: float = config.DROPOUT_RATE):
        """
        Args:
            input_dim: Dimension of input features
            dropout: Dropout rate for MC Dropout
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4)
        self.fc_mean = nn.Linear(input_dim // 4, 1)
        self.fc_log_var = nn.Linear(input_dim // 4, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (batch, features)
        
        Returns:
            (mean, log_variance) predictions
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Predict mean and log variance
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        
        return mean, log_var


# ==================== MAIN MODEL ====================
class AeroGuardModel(nn.Module):
    """
    Complete AeroGuard predictive maintenance model
    
    Architecture:
    1. Causal Attention Layer (sensor fusion)
    2. Temporal Feature Extraction (Bi-LSTM)
    3. Probabilistic Output Head (mean + uncertainty)
    """
    
    def __init__(
        self,
        num_features: int,
        causal_hidden_dim: int = config.CAUSAL_GNN_HIDDEN_DIM,
        lstm_hidden_dim: int = config.LSTM_HIDDEN_DIM,
        lstm_layers: int = config.LSTM_LAYERS,
        attention_heads: int = config.ATTENTION_HEADS,
        dropout: float = config.DROPOUT_RATE
    ):
        """
        Args:
            num_features: Number of input features
            causal_hidden_dim: Hidden dim for causal layer
            lstm_hidden_dim: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_features = num_features
        
        # 1. Input projection
        self.input_projection = nn.Linear(num_features, causal_hidden_dim)
        
        # 2. Causal sensor fusion
        self.causal_attention = CausalAttentionLayer(causal_hidden_dim, attention_heads)
        
        # 3. Temporal feature extraction
        self.temporal_extractor = TemporalFeatureExtractor(
            causal_hidden_dim, lstm_hidden_dim, lstm_layers, dropout
        )
        
        # 4. Probabilistic output
        self.probabilistic_head = ProbabilisticHead(
            self.temporal_extractor.output_dim, dropout
        )
        
        logger.info(f"AeroGuard model initialized: {num_features} features → "
                   f"Causal({causal_hidden_dim}) → LSTM({lstm_hidden_dim}×{lstm_layers}) → Probabilistic")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, num_features)
        
        Returns:
            (mean_rul, log_var_rul)
        """
        # Project input
        x = self.input_projection(x)
        
        # Causal attention fusion
        x = self.causal_attention(x)
        
        # Temporal extraction
        _, hidden = self.temporal_extractor(x)
        
        # Probabilistic prediction
        mean, log_var = self.probabilistic_head(hidden)
        
        return mean.squeeze(-1), log_var.squeeze(-1)
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        num_samples: int = config.MC_DROPOUT_SAMPLES
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout for uncertainty estimation
        
        Args:
            x: Input tensor (batch, seq_len, features)
            num_samples: Number of MC dropout samples
        
        Returns:
            (mean_predictions, std_predictions)
        """
        self.train()  # Enable dropout
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                mean, _ = self.forward(x)
                predictions.append(mean.cpu().numpy())
        
        predictions = np.array(predictions)  # (num_samples, batch)
        
        # Calculate statistics
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        self.eval()  # Disable dropout
        
        return mean_pred, std_pred


# ==================== LOSS FUNCTION ====================
class GaussianNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for Gaussian distribution
    Encourages accurate mean and uncertainty predictions
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, mean, log_var, target):
        """
        Args:
            mean: Predicted mean RUL
            log_var: Predicted log variance
            target: True RUL
        
        Returns:
            NLL loss
        """
        # var = exp(log_var)
        var = torch.exp(log_var)
        
        # NLL = 0.5 * (log(2π) + log(var) + (target - mean)^2 / var)
        loss = 0.5 * (torch.log(var) + (target - mean) ** 2 / var)
        
        return loss.mean()


# ==================== TRAINING FUNCTIONS ====================
def train_epoch(
    model: AeroGuardModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        mean, log_var = model(batch_x)
        loss = criterion(mean, log_var, batch_y)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(
    model: AeroGuardModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Validate model"""
    model.eval()
    total_loss = 0
    total_mse = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            mean, log_var = model(batch_x)
            loss = criterion(mean, log_var, batch_y)
            mse = F.mse_loss(mean, batch_y)
            
            total_loss += loss.item()
            total_mse += mse.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_rmse = np.sqrt(total_mse / len(val_loader))
    
    return avg_loss, avg_rmse


def train_model(
    model: AeroGuardModel,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    num_epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: Path = config.MODEL_DIR
) -> Dict:
    """
    Complete training loop
    
    Args:
        model: AeroGuard model
        train_data: Dict with 'X' and 'y' for training
        val_data: Dict with 'X' and 'y' for validation
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
    
    Returns:
        Training history
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    
    model = model.to(device)
    
    # Create datasets
    train_dataset = TurbofanDataset(train_data['X'], train_data['y'])
    val_dataset = TurbofanDataset(val_data['X'], val_data['y'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = GaussianNLLLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'best_epoch': 0,
        'best_rmse': float('inf')
    }
    
    # Early stopping
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_rmse = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val RMSE: {val_rmse:.4f}")
        
        # Save best model
        if val_rmse < history['best_rmse']:
            history['best_rmse'] = val_rmse
            history['best_epoch'] = epoch
            
            save_path = save_dir / 'best_model.pth'
            save_model_checkpoint(
                model, optimizer, epoch,
                {'val_rmse': val_rmse, 'val_loss': val_loss},
                save_path
            )
            
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE - Best RMSE: {history['best_rmse']:.4f} "
               f"at epoch {history['best_epoch']+1}")
    logger.info("=" * 60)
    
    return history


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("AeroGuard Model - Architecture Demo")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 32
    seq_length = 30
    num_features = 50
    
    dummy_x = torch.randn(batch_size, seq_length, num_features)
    dummy_y = torch.randn(batch_size) * 50 + 50  # RUL ~ 50±50
    
    # Initialize model
    model = AeroGuardModel(num_features=num_features)
    
    # Forward pass
    mean, log_var = model(dummy_x)
    
    print(f"\nInput shape: {dummy_x.shape}")
    print(f"Output mean shape: {mean.shape}")
    print(f"Output log_var shape: {log_var.shape}")
    
    # Uncertainty prediction
    mean_pred, std_pred = model.predict_with_uncertainty(dummy_x, num_samples=10)
    
    print(f"\nMC Dropout uncertainty:")
    print(f"  Mean RUL: {mean_pred.mean():.2f} ± {std_pred.mean():.2f}")
    
    print("\n" + "=" * 60)
    print("Model ready for training")
    print("=" * 60)
