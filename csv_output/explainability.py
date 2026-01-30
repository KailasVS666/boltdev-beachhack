"""
AeroGuard Explainability Module
Generate human-readable maintenance alerts with SHAP-based explanations
"""

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime

import config
from utils import logger, format_alert_message, plot_sensor_degradation


class ExplainabilityEngine:
    """
    XAI (Explainable AI) layer for AeroGuard
    
    Provides:
    1. SHAP-based feature importance
    2. Sensor-level contribution analysis
    3. Human-readable alert generation
    4. Visualization of degradation patterns
    """
    
    def __init__(
        self, 
        model,
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize explainability engine
        
        Args:
            model: Trained AeroGuard model
            background_data: Background dataset for SHAP (subset of training data)
            feature_names: Names of input features
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer = None
        
        if background_data is not None:
            self._initialize_shap_explainer()
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer with background data"""
        logger.info("Initializing SHAP explainer")
        
        # Limit background samples for computational efficiency
        if len(self.background_data) > config.SHAP_BACKGROUND_SAMPLES:
            indices = np.random.choice(
                len(self.background_data),
                config.SHAP_BACKGROUND_SAMPLES,
                replace=False
            )
            background_subset = self.background_data[indices]
        else:
            background_subset = self.background_data
        
        # Convert to torch tensor
        background_tensor = torch.FloatTensor(background_subset)
        
        # Create SHAP Deep Explainer
        # Wrapper for model that returns only mean prediction
        def model_predict(x):
            self.model.eval()
            with torch.no_grad():
                mean, _ = self.model(x)
            return mean.cpu().numpy()
        
        self.explainer = shap.DeepExplainer(
            lambda x: torch.tensor(model_predict(x)),
            background_tensor
        )
        
        logger.info("SHAP explainer initialized")
    
    def explain_prediction(
        self, 
        sequence: np.ndarray,
        actual_rul: Optional[float] = None
    ) -> Dict:
        """
        Generate SHAP explanation for a single prediction
        
        Args:
            sequence: Input sequence (seq_length, num_features)
            actual_rul: Ground truth RUL (if available)
        
        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            logger.warning("SHAP explainer not initialized")
            return {}
        
        # Expand to batch dimension
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        # Get prediction with uncertainty
        self.model.eval()
        mean_pred, std_pred = self.model.predict_with_uncertainty(
            sequence_tensor, 
            num_samples=config.MC_DROPOUT_SAMPLES
        )
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(sequence_tensor)
        
        # Aggregate SHAP values across time steps (sum absolute values)
        # shap_values shape: (batch=1, seq_length, num_features)
        feature_importance = np.abs(shap_values[0]).sum(axis=0)  # Sum over time
        
        # Sort features by importance
        top_indices = np.argsort(feature_importance)[::-1]
        
        explanation = {
            'predicted_rul_mean': mean_pred[0],
            'predicted_rul_std': std_pred[0],
            'actual_rul': actual_rul,
            'feature_importance': feature_importance,
            'top_features': [
                {
                    'name': self.feature_names[idx] if self.feature_names else f'Feature_{idx}',
                    'importance': feature_importance[idx],
                    'contribution_pct': feature_importance[idx] / feature_importance.sum() * 100
                }
                for idx in top_indices[:config.TOP_K_FEATURES]
                if feature_importance[idx] / feature_importance.sum() > config.MIN_CONTRIBUTION_THRESHOLD
            ]
        }
        
        return explanation
    
    def generate_maintenance_alert(
        self,
        engine_id: int,
        explanation: Dict,
        sensor_history: Optional[pd.DataFrame] = None,
        confidence_level: str = config.DEFAULT_CONFIDENCE_LEVEL
    ) -> Dict:
        """
        Generate structured maintenance alert from explanation
        
        Args:
            engine_id: Engine unit number
            explanation: Output from explain_prediction()
            sensor_history: Historical sensor data for this engine
            confidence_level: 'conservative' | 'standard' | 'optimistic'
        
        Returns:
            Alert dictionary with message, priority, parts, etc.
        """
        rul_mean = explanation['predicted_rul_mean']
        rul_std = explanation['predicted_rul_std']
        top_features = explanation['top_features']
        
        # Determine risk level
        if rul_mean < config.RUL_CRITICAL:
            priority = 'CRITICAL'
            recommended_action = "Immediate inspection required before next flight"
        elif rul_mean < config.RUL_HIGH:
            priority = 'HIGH'
            recommended_action = "Schedule detailed inspection within 24 hours"
        elif rul_mean < config.RUL_MEDIUM:
            priority = 'MEDIUM'
            recommended_action = "Schedule inspection within next maintenance window (7 days)"
        else:
            priority = 'LOW'
            recommended_action = "Monitor closely. Include in routine scheduled maintenance."
        
        # Map sensors to specific inspection actions
        inspection_actions = self._map_sensors_to_actions(top_features)
        
        # Identify required parts
        parts_list = self._identify_required_parts(top_features)
        
        # Format sensor contributions for alert
        sensor_contributions = [
            (feat['name'], feat['contribution_pct'] / 100)
            for feat in top_features
        ]
        
        # Get confidence level probability
        confidence = config.CONFIDENCE_LEVELS[confidence_level]
        
        # Generate human-readable message
        alert_message = format_alert_message(
            engine_id=engine_id,
            rul_mean=rul_mean,
            rul_std=rul_std,
            confidence=confidence,
            top_sensors=sensor_contributions,
            recommended_action=inspection_actions,
            parts_list=parts_list
        )
        
        alert = {
            'alert_id': f"AG-{engine_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'engine_id': engine_id,
            'priority': priority,
            'rul_mean': float(rul_mean),
            'rul_std': float(rul_std),
            'confidence_level': confidence_level,
            'top_sensors': sensor_contributions,
            'recommended_action': inspection_actions,
            'parts_list': parts_list,
            'alert_message': alert_message,
            'requires_human_approval': config.REQUIRE_HUMAN_SIGNOFF
        }
        
        return alert
    
    def _map_sensors_to_actions(self, top_features: List[Dict]) -> str:
        """
        Map sensor anomalies to specific maintenance actions
        
        Args:
            top_features: List of top contributing sensors
        
        Returns:
            Recommended inspection/maintenance action
        """
        actions = []
        
        for feature in top_features[:3]:  # Top 3 sensors
            sensor = feature['name']
            
            # Temperature sensors
            if 'T50' in sensor or 'LPT' in sensor:
                actions.append("Borescope inspection of LPT (Low-Pressure Turbine) blades")
            elif 'T30' in sensor or 'HPC' in sensor:
                actions.append("Inspect HPC (High-Pressure Compressor) for fouling or damage")
            elif 'T24' in sensor or 'LPC' in sensor:
                actions.append("Check LPC (Low-Pressure Compressor) efficiency")
            elif 'T2' in sensor:
                actions.append("Verify inlet temperature sensors and environmental controls")
            
            # Pressure sensors
            elif 'P30' in sensor:
                actions.append("Pressure test HPC outlet, check for leaks")
            elif 'P15' in sensor or 'bypass' in sensor.lower():
                actions.append("Inspect bypass duct for blockages or damage")
            elif 'P2' in sensor:
                actions.append("Check inlet pressure sensors and air filtration")
            
            # Speed sensors
            elif 'Nc' in sensor or 'core' in sensor.lower():
                actions.append("Inspect core rotor bearings and balance")
            elif 'Nf' in sensor or 'fan' in sensor.lower():
                actions.append("Inspect fan blades and bearings")
            
            # Derived metrics
            elif 'epr' in sensor.lower():
                actions.append("Full engine pressure ratio check across all stages")
            elif 'phi' in sensor or 'fuel' in sensor.lower():
                actions.append("Inspect fuel nozzles and flow control systems")
        
        if not actions:
            actions = ["Comprehensive engine diagnostic inspection per maintenance manual"]
        
        # Remove duplicates and join
        unique_actions = list(dict.fromkeys(actions))
        return " | ".join(unique_actions)
    
    def _identify_required_parts(self, top_features: List[Dict]) -> List[str]:
        """
        Identify parts that should be staged based on sensor anomalies
        
        Args:
            top_features: Top contributing sensors
        
        Returns:
            List of part descriptions
        """
        parts = []
        
        for feature in top_features[:3]:
            sensor = feature['name']
            
            if 'T50' in sensor or 'LPT' in sensor:
                parts.append("LPT blade set (PN: TBD)")
                parts.append("LPT nozzle guide vanes (PN: TBD)")
            elif 'T30' in sensor or 'HPC' in sensor:
                parts.append("HPC blade set (PN: TBD)")
                parts.append("HPC seals (PN: TBD)")
            elif 'P30' in sensor or 'P15' in sensor:
                parts.append("Pressure seals and gaskets (PN: TBD)")
            elif 'Nf' in sensor or 'Nc' in sensor:
                parts.append("Rotor bearings (PN: TBD)")
                parts.append("Fan blade set (PN: TBD)")
            elif 'fuel' in sensor.lower() or 'phi' in sensor:
                parts.append("Fuel nozzles (PN: TBD)")
        
        # Remove duplicates
        unique_parts = list(dict.fromkeys(parts))
        
        return unique_parts if unique_parts else ["To be determined based on inspection findings"]
    
    def visualize_explanation(
        self,
        explanation: Dict,
        save_path: Optional[Path] = None
    ):
        """
        Create visualization of feature importance
        
        Args:
            explanation: Explanation dictionary
            save_path: Path to save figure
        """
        top_features = explanation['top_features']
        
        if not top_features:
            logger.warning("No features to visualize")
            return
        
        # Extract data
        feature_names = [f['name'] for f in top_features]
        contributions = [f['contribution_pct'] for f in top_features]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6), dpi=config.FIGURE_DPI)
        
        colors = [
            config.COLOR_PALETTE['critical'] if c > 30 else
            config.COLOR_PALETTE['high'] if c > 20 else
            config.COLOR_PALETTE['medium'] if c > 10 else
            config.COLOR_PALETTE['normal']
            for c in contributions
        ]
        
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, contributions, color=colors, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Contribution to Risk (%)', fontsize=12, fontweight='bold')
        ax.set_title('Top Contributing Sensors (SHAP Analysis)', 
                    fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(contributions):
            ax.text(v + 0.5, i, f'{v:.1f}%', 
                   va='center', fontsize=10)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Explanation visualization saved to {save_path}")
        
        return fig
    
    def create_alert_report(
        self,
        alert: Dict,
        sensor_history: Optional[pd.DataFrame] = None,
        save_dir: Optional[Path] = None
    ) -> Path:
        """
        Create comprehensive alert report with visualizations
        
        Args:
            alert: Alert dictionary
            sensor_history: Historical sensor data
            save_dir: Directory to save report
        
        Returns:
            Path to saved report
        """
        if save_dir is None:
            save_dir = config.ALERT_DIR
        
        alert_id = alert['alert_id']
        report_dir = save_dir / alert_id
        report_dir.mkdir(exist_ok=True)
        
        # Save alert details as JSON
        import json
        with open(report_dir / 'alert_details.json', 'w') as f:
            json.dump(alert, f, indent=2)
        
        # Save formatted message
        with open(report_dir / 'alert_message.txt', 'w') as f:
            f.write(alert['alert_message'])
        
        logger.info(f"Alert report created: {report_dir}")
        
        return report_dir


# ==================== BATCH PROCESSING ====================
def generate_alerts_for_dataset(
    model,
    test_sequences: np.ndarray,
    test_rul: np.ndarray,
    engine_ids: np.ndarray,
    feature_names: List[str],
    background_data: np.ndarray,
    output_dir: Path = config.ALERT_DIR
) -> List[Dict]:
    """
    Generate alerts for all engines in test dataset
    
    Args:
        model: Trained AeroGuard model
        test_sequences: Test sequences (num_samples, seq_len, features)
        test_rul: Actual RUL values
        engine_ids: Engine unit numbers
        feature_names: Feature names
        background_data: Background data for SHAP
        output_dir: Output directory for alerts
    
    Returns:
        List of generated alerts
    """
    logger.info(f"Generating alerts for {len(test_sequences)} test samples")
    
    explainer = ExplainabilityEngine(model, background_data, feature_names)
    
    alerts = []
    
    for i, (sequence, rul, engine_id) in enumerate(zip(test_sequences, test_rul, engine_ids)):
        # Generate explanation
        explanation = explainer.explain_prediction(sequence, actual_rul=rul)
        
        # Only generate alert if RUL is below threshold
        if explanation['predicted_rul_mean'] < config.RUL_MEDIUM:
            alert = explainer.generate_maintenance_alert(
                engine_id=int(engine_id),
                explanation=explanation
            )
            
            alerts.append(alert)
            
            logger.info(f"Alert generated for Engine #{engine_id}: "
                       f"RUL={explanation['predicted_rul_mean']:.0f}±{explanation['predicted_rul_std']:.0f}, "
                       f"Priority={alert['priority']}")
    
    logger.info(f"Generated {len(alerts)} alerts total")
    
    return alerts


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("AeroGuard Explainability Engine - Demo")
    print("=" * 60)
    
    # Create dummy explanation
    dummy_explanation = {
        'predicted_rul_mean': 45.0,
        'predicted_rul_std': 8.5,
        'actual_rul': 50.0,
        'feature_importance': np.random.rand(10),
        'top_features': [
            {'name': 'T50_LPT_Temp', 'importance': 0.35, 'contribution_pct': 35.0},
            {'name': 'Nc_Core_Speed', 'importance': 0.22, 'contribution_pct': 22.0},
            {'name': 'P30_HPC_Pressure', 'importance': 0.18, 'contribution_pct': 18.0},
            {'name': 'phi_Fuel_Flow', 'importance': 0.15, 'contribution_pct': 15.0},
            {'name': 'T30_HPC_Temp', 'importance': 0.10, 'contribution_pct': 10.0},
        ]
    }
    
    # Initialize engine (without model for demo)
    explainer = ExplainabilityEngine(model=None)
    
    # Generate alert
    alert = explainer.generate_maintenance_alert(
        engine_id=3,
        explanation=dummy_explanation
    )
    
    print("\n" + alert['alert_message'])
    
    print("\n" + "=" * 60)
    print("Explainability Engine Ready")
    print("=" * 60)
