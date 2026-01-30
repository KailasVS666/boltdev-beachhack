"""
AeroGuard Utility Functions
Helper functions for visualization, logging, unit conversions, and data manipulation
"""

import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import config


# ==================== LOGGING SETUP ====================
def setup_logging(log_level: str = config.LOG_LEVEL) -> logging.Logger:
    """
    Configure logging for AeroGuard system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('AeroGuard')


logger = setup_logging()


# ==================== UNIT CONVERSIONS ====================
def rankine_to_celsius(temp_r: float) -> float:
    """Convert temperature from Rankine to Celsius"""
    return (temp_r - 491.67) * 5/9


def rankine_to_fahrenheit(temp_r: float) -> float:
    """Convert temperature from Rankine to Fahrenheit"""
    return temp_r - 459.67


def psia_to_kpa(pressure_psia: float) -> float:
    """Convert pressure from psia to kPa"""
    return pressure_psia * 6.89476


def rpm_to_rps(rpm: float) -> float:
    """Convert rotational speed from RPM to revolutions per second"""
    return rpm / 60


# ==================== DATA FORMATTING ====================
def format_alert_message(
    engine_id: int,
    rul_mean: float,
    rul_std: float,
    confidence: float,
    top_sensors: List[Tuple[str, float]],
    recommended_action: str,
    parts_list: List[str]
) -> str:
    """
    Generate human-readable maintenance alert message
    
    Args:
        engine_id: Engine unit number
        rul_mean: Mean remaining useful life (cycles)
        rul_std: Standard deviation of RUL
        confidence: Confidence level (0-1)
        top_sensors: List of (sensor_name, contribution) tuples
        recommended_action: Suggested inspection/action
        parts_list: List of parts to stage
    
    Returns:
        Formatted alert message
    """
    priority = get_priority_level(rul_mean)
    rul_lower = max(0, rul_mean - 1.96 * rul_std)  # 95% CI lower bound
    rul_upper = rul_mean + 1.96 * rul_std
    
    message = f"""
╔══════════════════════════════════════════════════════════════════
║ AEROGUARD PREDICTIVE MAINTENANCE ALERT
╠══════════════════════════════════════════════════════════════════
║ Priority: {priority}
║ Engine Unit: #{engine_id}
║ 
║ REMAINING USEFUL LIFE ESTIMATE:
║   Mean: {rul_mean:.0f} flight cycles
║   Range: {rul_lower:.0f} - {rul_upper:.0f} cycles ({confidence*100:.0f}% confidence)
║   
║ PRIMARY INDICATORS:
"""
    
    for sensor_name, contribution in top_sensors:
        message += f"║   • {sensor_name}: {contribution*100:.1f}% contribution to risk\n"
    
    message += f"""║
║ RECOMMENDED ACTION:
║   {recommended_action}
║
║ PARTS TO STAGE:
"""
    
    if parts_list:
        for part in parts_list:
            message += f"║   • {part}\n"
    else:
        message += "║   • (To be determined based on inspection)\n"
    
    message += """║
║ NOTE: This is an ADVISORY alert. Maintenance action requires
║       verification by certified technician per maintenance manual.
╚══════════════════════════════════════════════════════════════════
"""
    
    return message


def get_priority_level(rul: float) -> str:
    """Determine alert priority based on RUL"""
    if rul < config.RUL_CRITICAL:
        return "⚠️  CRITICAL"
    elif rul < config.RUL_HIGH:
        return "🔶 HIGH"
    elif rul < config.RUL_MEDIUM:
        return "🔷 MEDIUM"
    else:
        return "🟢 LOW"


# ==================== RISK CALCULATIONS ====================
def calculate_risk_envelope(
    rul_mean: float,
    rul_std: float,
    target_flights: int,
    confidence_level: float = 0.80
) -> Dict[str, float]:
    """
    Calculate probability of failure within target flight cycles
    
    Args:
        rul_mean: Mean RUL prediction
        rul_std: Standard deviation of RUL
        target_flights: Number of flights to assess risk for
        confidence_level: Desired confidence level
    
    Returns:
        Dictionary with risk metrics
    """
    from scipy import stats
    
    # Probability that actual RUL < target_flights
    z_score = (target_flights - rul_mean) / (rul_std + 1e-8)
    prob_failure = stats.norm.cdf(z_score)
    
    # Safe operating window (conservative estimate)
    z_critical = stats.norm.ppf(confidence_level)
    safe_flights = max(0, rul_mean - z_critical * rul_std)
    
    return {
        'probability_failure': prob_failure,
        'safe_flight_budget': safe_flights,
        'confidence_level': confidence_level,
        'uncertainty_ratio': rul_std / (rul_mean + 1e-8)
    }


# ==================== VISUALIZATION ====================
def plot_sensor_degradation(
    sensor_data: pd.DataFrame,
    sensor_name: str,
    current_cycle: int,
    baseline_mean: Optional[float] = None,
    baseline_std: Optional[float] = None,
    save_path: Optional[Path] = None
):
    """
    Plot sensor time series with degradation pattern
    
    Args:
        sensor_data: DataFrame with 'time_cycles' and sensor columns
        sensor_name: Name of sensor to plot
        current_cycle: Current flight cycle
        baseline_mean: Normal operating mean
        baseline_std: Normal operating standard deviation
        save_path: Path to save plot
    """
    plt.style.use(config.PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=config.FIGURE_DPI)
    
    cycles = sensor_data['time_cycles']
    values = sensor_data[sensor_name]
    
    # Plot sensor values
    ax.plot(cycles, values, linewidth=2, label=f'{sensor_name} Actual', color='#1976d2')
    
    # Plot baseline if provided
    if baseline_mean is not None:
        ax.axhline(baseline_mean, color='#388e3c', linestyle='--', 
                   label='Baseline Mean', linewidth=1.5)
        
        if baseline_std is not None:
            ax.fill_between(cycles, 
                           baseline_mean - 2*baseline_std,
                           baseline_mean + 2*baseline_std,
                           alpha=0.2, color='#388e3c', label='±2σ Normal Range')
    
    # Mark current position
    ax.axvline(current_cycle, color='#d32f2f', linestyle=':', 
               label='Current Cycle', linewidth=2)
    
    ax.set_xlabel('Flight Cycles', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{sensor_name} Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Sensor Degradation Pattern: {sensor_name}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    
    return fig


def plot_rul_prediction(
    actual_rul: Optional[np.ndarray],
    predicted_rul: np.ndarray,
    uncertainty: np.ndarray,
    save_path: Optional[Path] = None
):
    """
    Plot RUL predictions with uncertainty bands
    
    Args:
        actual_rul: Ground truth RUL (if available)
        predicted_rul: Predicted RUL values
        uncertainty: Uncertainty (std) for each prediction
        save_path: Path to save plot
    """
    plt.style.use(config.PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=config.FIGURE_DPI)
    
    cycles = np.arange(len(predicted_rul))
    
    # Plot predictions with uncertainty
    ax.plot(cycles, predicted_rul, linewidth=2, label='Predicted RUL', color='#1976d2')
    ax.fill_between(cycles,
                    predicted_rul - 1.96 * uncertainty,
                    predicted_rul + 1.96 * uncertainty,
                    alpha=0.3, color='#1976d2', label='95% Confidence Interval')
    
    # Plot actual if available
    if actual_rul is not None:
        ax.plot(cycles, actual_rul, linewidth=2, label='Actual RUL', 
                color='#388e3c', linestyle='--')
    
    # Add threshold lines
    ax.axhline(config.RUL_CRITICAL, color='#d32f2f', linestyle=':', 
               label=f'Critical ({config.RUL_CRITICAL} cycles)', linewidth=1.5)
    ax.axhline(config.RUL_HIGH, color='#f57c00', linestyle=':', 
               label=f'High Priority ({config.RUL_HIGH} cycles)', linewidth=1.5)
    
    ax.set_xlabel('Sequence Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Remaining Useful Life (cycles)', fontsize=12, fontweight='bold')
    ax.set_title('RUL Prediction with Uncertainty', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    
    return fig


# ==================== AUDIT LOGGING ====================
def log_audit_event(
    event_type: str,
    details: Dict[str, Any],
    user_id: Optional[str] = None
):
    """
    Log audit event for regulatory compliance
    
    Args:
        event_type: Type of event (ALERT_GENERATED, FEEDBACK_RECEIVED, etc.)
        details: Dictionary with event details
        user_id: ID of user triggering event
    """
    if not config.ENABLE_AUDIT_LOGGING:
        return
    
    audit_entry = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'user_id': user_id or 'SYSTEM',
        'details': details
    }
    
    # Append to audit log
    with open(config.AUDIT_LOG_PATH, 'a') as f:
        f.write(json.dumps(audit_entry) + '\n')
    
    logger.info(f"Audit: {event_type} - {details.get('summary', 'No summary')}")


# ==================== DATA VALIDATION ====================
def validate_sensor_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate sensor data for completeness and sanity checks
    
    Args:
        df: DataFrame with sensor data
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required columns
    required_cols = ['unit_number', 'time_cycles'] + config.TEMPERATURE_SENSORS + config.PRESSURE_SENSORS
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check for NaN values
    if df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"NaN values found in columns: {nan_cols}")
    
    # Sanity checks on physical ranges
    if 'T2' in df.columns and (df['T2'] < 400 or df['T2'] > 600).any():
        errors.append("T2 (inlet temp) outside expected range [400-600°R]")
    
    if 'P2' in df.columns and (df['P2'] < 0 or df['P2'] > 20).any():
        errors.append("P2 (inlet pressure) outside expected range [0-20 psia]")
    
    is_valid = len(errors) == 0
    return is_valid, errors


# ==================== FILE I/O ====================
def save_model_checkpoint(model, optimizer, epoch: int, metrics: Dict, filepath: Path):
    """Save model checkpoint with training state"""
    import torch
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_model_checkpoint(filepath: Path):
    """Load model checkpoint"""
    import torch
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath)
    logger.info(f"Checkpoint loaded from {filepath} (epoch {checkpoint['epoch']})")
    return checkpoint
