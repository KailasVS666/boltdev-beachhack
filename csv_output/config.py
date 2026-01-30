"""
AeroGuard Configuration
Hyperparameters, thresholds, and safety margins for the predictive maintenance system
"""

import os
from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ALERT_DIR = BASE_DIR / "alerts"
FEEDBACK_DB = BASE_DIR / "feedback.db"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, ALERT_DIR]:
    directory.mkdir(exist_ok=True)

# ==================== DATA CONFIGURATION ====================
# NASA C-MAPSS Turbofan Dataset column names
SENSOR_COLUMNS = [
    'unit_number', 'time_cycles', 
    'op_setting_1', 'op_setting_2', 'op_setting_3',  # Operational settings (altitude, Mach, TRA)
    'T2', 'T24', 'T30', 'T50',  # Temperatures
    'P2', 'P15', 'P30',  # Pressures
    'Nf', 'Nc',  # Fan and core speeds
    'epr', 'Ps30', 'phi',  # Engine metrics
    'NRf', 'NRc', 'BPR',  # Corrected speeds and bypass ratio
    'farB', 'htBleed',  # Fuel-air ratio and bleed enthalpy
    'Nf_dmd', 'PCNfR_dmd',  # Demanded speeds
    'W31', 'W32'  # Coolant bleeds
]

# Feature groups for sensor fusion
TEMPERATURE_SENSORS = ['T2', 'T24', 'T30', 'T50']
PRESSURE_SENSORS = ['P2', 'P15', 'P30', 'Ps30']
SPEED_SENSORS = ['Nf', 'Nc', 'NRf', 'NRc']
OPERATIONAL_SETTINGS = ['op_setting_1', 'op_setting_2', 'op_setting_3']
DERIVED_METRICS = ['epr', 'phi', 'BPR', 'farB', 'htBleed']

# Sequence parameters for LSTM
SEQUENCE_LENGTH = 30  # Look back 30 flight cycles
SEQUENCE_STRIDE = 1   # Sliding window stride

# ==================== MODEL HYPERPARAMETERS ====================
# Architecture
CAUSAL_GNN_HIDDEN_DIM = 64
LSTM_HIDDEN_DIM = 128
LSTM_LAYERS = 2
ATTENTION_HEADS = 4
DROPOUT_RATE = 0.3  # For MC Dropout uncertainty

# Training
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
VALIDATION_SPLIT = 0.2

# Monte Carlo Dropout for uncertainty
MC_DROPOUT_SAMPLES = 50  # Number of forward passes for uncertainty estimation

# ==================== CAUSAL INFERENCE ====================
# Causal discovery algorithm settings
CAUSAL_ALGORITHM = 'PC'  # PC algorithm for constraint-based learning
CAUSAL_ALPHA = 0.05  # Significance level for conditional independence tests
CAUSAL_MAX_COND_VARS = 3  # Maximum conditioning set size

# Physics-informed causal constraints
# These are known relationships that should be enforced
KNOWN_CAUSAL_EDGES = [
    ('op_setting_1', 'T2'),  # Altitude affects inlet temp
    ('op_setting_2', 'epr'),  # Mach number affects engine pressure ratio
    ('T30', 'T50'),  # HPC temp affects LPT temp
    ('P30', 'epr'),  # HPC pressure affects EPR
    ('Nf', 'NRf'),  # Fan speed affects corrected fan speed
    ('Nc', 'NRc'),  # Core speed affects corrected core speed
]

# Environmental confounders to adjust for
ENVIRONMENTAL_CONFOUNDERS = ['op_setting_1', 'op_setting_2', 'op_setting_3']

# ==================== RISK ASSESSMENT ====================
# Probabilistic thresholds
CONFIDENCE_LEVELS = {
    'conservative': 0.90,  # 90% confidence (safety-critical)
    'standard': 0.80,      # 80% confidence (normal operations)
    'optimistic': 0.70     # 70% confidence (for planning)
}

DEFAULT_CONFIDENCE_LEVEL = 'conservative'

# RUL thresholds for alert generation
RUL_CRITICAL = 30    # < 30 flights remaining → Critical alert
RUL_HIGH = 60        # < 60 flights remaining → High priority
RUL_MEDIUM = 100     # < 100 flights remaining → Medium priority

# Uncertainty thresholds
MAX_ACCEPTABLE_UNCERTAINTY = 0.3  # Reject predictions with σ/μ > 0.3

# ==================== EXPLAINABILITY (XAI) ====================
# SHAP settings
SHAP_BACKGROUND_SAMPLES = 100  # Number of background samples for SHAP
TOP_K_FEATURES = 5  # Show top 5 contributing sensors in alerts

# Alert generation
MIN_CONTRIBUTION_THRESHOLD = 0.10  # Only show sensors contributing >10% to risk

# ==================== MAINTENANCE INTEGRATION ====================
# Safety margins (regulatory compliance)
FAA_SAFETY_MARGIN = 0.20  # Add 20% margin to conservative estimates
PARTS_LEAD_TIME_DAYS = 14  # Typical parts procurement time

# Alert priority scoring weights
PRIORITY_WEIGHTS = {
    'risk_score': 0.50,        # 50% weight on predicted risk
    'uncertainty': 0.20,       # 20% weight on prediction confidence
    'flight_schedule': 0.20,   # 20% weight on upcoming flights
    'parts_availability': 0.10 # 10% weight on parts in stock
}

# Work order templates
WORK_ORDER_PRIORITY = {
    'CRITICAL': 'AOG (Aircraft on Ground) - Immediate action required',
    'HIGH': 'Complete before next flight',
    'MEDIUM': 'Schedule within 7 days',
    'LOW': 'Include in next scheduled maintenance'
}

# ==================== FEEDBACK LOOP ====================
# Outcome categories
FEEDBACK_OUTCOMES = [
    'CONFIRMED',                 # Physical inspection validated alert
    'REJECTED_FALSE_POSITIVE',   # No issue found
    'REJECTED_ALREADY_KNOWN',    # Issue was already on schedule
    'DEFERRED',                  # Alert acknowledged, addressed later
    'UNDER_INVESTIGATION'        # Requires further analysis
]

# Retraining settings
RETRAINING_FREQUENCY_DAYS = 90  # Quarterly retraining
MIN_FEEDBACK_SAMPLES = 50  # Minimum feedback samples before retraining
FEEDBACK_REVIEW_REQUIRED = True  # Expert review before model update

# Alert precision targets
TARGET_PRECISION = 0.75  # Target: 75% of alerts should be confirmed
TARGET_RECALL = 0.90     # Target: Catch 90% of actual failures

# ==================== REGULATORY COMPLIANCE ====================
# Audit trail
ENABLE_AUDIT_LOGGING = True
AUDIT_LOG_PATH = BASE_DIR / "audit_log.json"

# Human-in-the-loop requirements
REQUIRE_HUMAN_SIGNOFF = True  # All work orders must be approved by certified tech
ADVISORY_ONLY_MODE = True     # System is advisory, not autonomous

# Data retention (for regulatory compliance)
ALERT_RETENTION_YEARS = 7
FEEDBACK_RETENTION_YEARS = 10

# ==================== VISUALIZATION ====================
# Plotting style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 150
COLOR_PALETTE = {
    'critical': '#d32f2f',  # Red
    'high': '#f57c00',      # Orange
    'medium': '#fbc02d',    # Yellow
    'low': '#388e3c',       # Green
    'normal': '#1976d2'     # Blue
}

# ==================== LOGGING ====================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = BASE_DIR / "aeroguard.log"
