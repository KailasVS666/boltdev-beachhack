# AeroGuard: Predictive Maintenance for Aviation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**AeroGuard** is a high-reliability predictive maintenance system designed to bridge the "Trust Gap" in aviation AI by providing explainable, probabilistic RUL (Remaining Useful Life) predictions that work within existing FAA/EASA regulatory frameworks.

## 🎯 Core Mission

Transform raw turbofan sensor data into actionable maintenance insights while maintaining:
- **Explainability**: Every prediction shows which sensors contributed and why
- **Uncertainty Quantification**: Risk envelopes instead of dangerous countdown timers
- **Regulatory Compliance**: Advisory-only system requiring human sign-off
- **Non-Punitive Learning**: "One-way valve" feedback loop treats rejections as valuable calibration data

## ✨ Key Features

### 1. **Causal Sensor Fusion**
- Removes environmental noise (altitude, temperature, flight conditions) from degradation signals
- Uses domain knowledge + data-driven causal discovery
- Detects "invisible chains" (cross-system correlations like landing style → gear + engine stress)

### 2. **Explainable Maintenance Alerts**
- SHAP-based feature importance shows exactly which sensors triggered alerts
- Maps sensor anomalies to specific inspection actions (e.g., "T50 elevated → borescope LPT blades")
- Human-readable reports with parts staging recommendations

### 3. **Probabilistic Risk Assessment**
- Monte Carlo Dropout for uncertainty quantification
- Outputs: "80% confidence of reaching limits in 45-60 flights" (not "fails in 50 flights")
- FAA safety margins (20%) automatically applied

### 4. **Automated Work Order Generation**
- Converts predictions into draft work orders
- Checks parts inventory and lead times
- Priority scoring based on risk, schedule, and parts availability
- Requires certified technician sign-off

### 5. **One-Way Valve Feedback Loop**
- Technicians confirm or reject alerts (non-punitive)
- False positives analyzed for systematic issues
- Quarterly retraining with expert review
- Maintains audit trail for regulatory compliance

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AeroGuard System Flow                     │
└─────────────────────────────────────────────────────────────┘

Raw Sensor Data (26 channels: temps, pressures, speeds, etc.)
         │
         ▼
┌─────────────────────┐
│  Causal Engine      │ ← Removes environmental confounding
│  - DAG Construction │   (altitude, Mach, throttle effects)
│  - Counterfactuals  │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Engineering│ ← Rolling stats, deltas, physics ratios
│  - 50+ features     │
└─────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  AeroGuard ML Model                     │
│  ┌───────────────────────────────────┐  │
│  │ Causal Attention (sensor fusion) │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │ Bi-LSTM (temporal degradation)   │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │ Probabilistic Head (mean + σ)    │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  XAI Layer (SHAP)   │ ← Feature importance, sensor contributions
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Maintenance Alert  │ ← Human-readable with recommended actions
│  + Draft Work Order │   Parts list, priority, deadline
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Technician Review  │ → Feedback (Confirmed / Rejected)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Feedback Database  │ → Quarterly retraining (with expert review)
└─────────────────────┘
```

## 📁 Project Structure

```
csv_output/
├── config.py                    # Hyperparameters, thresholds, safety margins
├── utils.py                     # Logging, visualization, unit conversions
├── data_processor.py            # Data loading, feature engineering, RUL calculation
├── causal_engine.py             # Causal inference, environmental noise removal
├── aeroguard_model.py           # Main ML model (Causal Attention + LSTM + Probabilistic)
├── explainability.py            # SHAP integration, alert generation
├── maintenance_integration.py   # Work orders, parts inventory, risk envelopes
├── feedback_system.py           # One-way valve feedback loop
├── requirements.txt             # Python dependencies
├── demo_notebook.ipynb          # End-to-end demonstration
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the project directory
cd csv_output

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

AeroGuard is designed for **NASA C-MAPSS turbofan degradation dataset**:
- Download from: [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- Place files in `data/` directory:
  - `train_FD001.txt`, `train_FD002.txt`, etc.
  - `test_FD001.txt`, `RUL_FD001.txt`, etc.

Dataset format: 26 columns (unit ID, time cycles, 3 operational settings, 21 sensors)

### Training

```python
from data_processor import TurbofanDataProcessor
from aeroguard_model import AeroGuardModel, train_model
from causal_engine import CausalEngine

# 1. Process data
processor = TurbofanDataProcessor()
data = processor.process_pipeline('data/train_FD001.txt')

# 2. Apply causal adjustments
causal_engine = CausalEngine()
# (Causal adjustments integrated in data pipeline)

# 3. Train model
model = AeroGuardModel(num_features=len(data['feature_names']))
history = train_model(
    model,
    train_data={'X': data['X_train'], 'y': data['y_train']},
    val_data={'X': data['X_val'], 'y': data['y_val']}
)

print(f"Best validation RMSE: {history['best_rmse']:.2f}")
```

### Prediction & Alert Generation

```python
from explainability import ExplainabilityEngine
from maintenance_integration import MaintenanceIntegration

# 1. Generate explanation
explainer = ExplainabilityEngine(model, data['X_train'], data['feature_names'])
explanation = explainer.explain_prediction(test_sequence)

# 2. Create maintenance alert
alert = explainer.generate_maintenance_alert(
    engine_id=3,
    explanation=explanation
)

print(alert['alert_message'])

# 3. Generate work order
maintenance = MaintenanceIntegration()
work_order = maintenance.generate_work_order(
    alert,
    aircraft_id='N12345',
    flights_scheduled=7
)

print(maintenance.format_work_order_for_print(work_order))
```

### Feedback Loop

```python
from feedback_system import FeedbackSystem

feedback = FeedbackSystem()

# Record alert
feedback.record_alert(alert, work_order['work_order_id'])

# Technician submits feedback
feedback.submit_feedback(
    alert_id=alert['alert_id'],
    outcome='CONFIRMED',  # or 'REJECTED_FALSE_POSITIVE', etc.
    technician_id='TECH-001',
    technician_notes='LPT blade erosion confirmed. Replaced per manual.',
    actual_finding='Erosion on blades 3-7',
    parts_replaced=['LPT-BLADE-001']
)

# Check performance
summary = feedback.get_performance_summary()
print(f"Precision: {summary['precision']:.1%}")
```

## 📊 Model Performance

Trained on NASA C-MAPSS FD001 (typical targets):
- **RMSE**: ~18-22 cycles (state-of-the-art baseline)
- **Alert Precision**: Target 75% (configurable)
- **Alert Recall**: Target 90%
- **Uncertainty Calibration**: Reliability diagrams used for validation

## 🔬 Technical Details

### Causal Inference
- **Algorithm**: PC (constraint-based) + domain knowledge
- **Environmental Confounders**: Operational settings (altitude, Mach, throttle)
- **Counterfactual Analysis**: "What if flight was at sea level?"

### Neural Architecture
- **Input**: 30-cycle sequences × 50+ features
- **Causal Attention**: 4-head multi-head attention with learned causal weights
- **Bi-LSTM**: 2 layers × 128 hidden units (captures forward/backward degradation)
- **Probabilistic Output**: Gaussian NLL loss for mean + variance
- **MC Dropout**: 50 samples for uncertainty quantification

### Explainability (SHAP)
- **Background Samples**: 100 from training set
- **Aggregation**: Sum absolute SHAP values across time steps
- **Top-K**: Show top 5 contributing sensors (>10% contribution threshold)

### Risk Calculation
- **Confidence Levels**: Conservative (90%), Standard (80%), Optimistic (70%)
- **FAA Safety Margin**: 20% reduction in safe flight budget
- **Uncertainty Threshold**: Reject predictions with σ/μ > 0.3

## 🛡️ Regulatory Compliance

- ✅ **Advisory Only**: System does NOT make autonomous maintenance decisions
- ✅ **Human Sign-Off**: All work orders require certified technician approval
- ✅ **Audit Trail**: Complete logging of alerts, feedback, and retraining events
- ✅ **Data Retention**: 7 years (alerts), 10 years (feedback) per regulations
- ✅ **Explainability**: Every prediction includes sensor-level evidence
- ✅ **Safety Margins**: FAA-compliant conservative estimates

## 🔄 Retraining Workflow

1. **Accumulate Feedback**: Minimum 50 samples required
2. **Expert Review**: Domain expert reviews rejection patterns
3. **Prepare Dataset**: Confirmed + rejected samples with labels
4. **Retrain Model**: New version trained on augmented dataset
5. **A/B Testing**: Shadow mode validation before deployment
6. **Approval & Deployment**: Expert sign-off required

## 📈 Future Enhancements

- [ ] Integration with real-time ACARS/FOQA data streams
- [ ] Multi-engine correlation (fleet-wide pattern detection)
- [ ] Advanced causal discovery (SCM, do-calculus)
- [ ] Transformer-based temporal modeling
- [ ] Integration with airline MRO systems (SAP, Maintenix)
- [ ] Mobile app for technician feedback
- [ ] Anomaly detection for novel failure modes

## 📚 Dataset Information

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**

The dataset simulates turbofan engine degradation under various operational conditions:

| Subset | Train Units | Test Units | Fault Modes | Operating Conditions |
|--------|-------------|------------|-------------|---------------------|
| FD001  | 100         | 100        | HPC         | Sea level           |
| FD002  | 260         | 259        | HPC         | Six conditions      |
| FD003  | 100         | 100        | HPC, Fan    | Sea level           |
| FD004  | 248         | 249        | HPC, Fan    | Six conditions      |

**Sensors (21 total)**:
- Temperatures: T2, T24, T30, T50 (Inlet → LPC → HPC → LPT)
- Pressures: P2, P15, P30, Ps30
- Speeds: Nf, Nc, NRf, NRc
- Performance: epr, phi, BPR, farB, htBleed
- Coolant: W31, W32

## 🤝 Contributing

Contributions welcome! Focus areas:
- Enhanced causal discovery algorithms
- Integration with real aircraft data formats
- Improved uncertainty calibration
- Additional failure mode coverage

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- NASA Ames Prognostics Center of Excellence for C-MAPSS dataset
- FAA Advisory Circulars on predictive maintenance
- SHAP library (Lundberg & Lee)
- Aviation maintenance professionals who provided domain expertise

## 📧 Contact

For questions or collaboration: [Your contact info]

---

**Disclaimer**: AeroGuard is a research/demonstration system. Any production deployment must undergo rigorous validation, certification, and regulatory approval per applicable aviation standards.
