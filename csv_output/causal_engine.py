"""
AeroGuard Causal Inference Engine
Distinguish true degradation signals from environmental noise using causal reasoning
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression

import config
from utils import logger


class CausalEngine:
    """
    Causal inference system to separate equipment wear from environmental confounding
    
    Uses:
    1. Domain knowledge (physics-based causal graph)
    2. Data-driven causal discovery (PC algorithm)
    3. Do-calculus for intervention effects
    """
    
    def __init__(self):
        """Initialize causal engine with domain knowledge"""
        self.causal_graph = nx.DiGraph()
        self.causal_models = {}  # Store regression models for each sensor
        self._build_domain_knowledge_graph()
        
    def _build_domain_knowledge_graph(self):
        """
        Construct initial causal graph based on turbofan physics
        
        Nodes: Sensors and operational settings
        Edges: Known causal relationships
        """
        logger.info("Building domain knowledge causal graph")
        
        # Add all nodes (sensors + operational settings)
        all_sensors = (
            config.OPERATIONAL_SETTINGS +
            config.TEMPERATURE_SENSORS +
            config.PRESSURE_SENSORS +
            config.SPEED_SENSORS +
            config.DERIVED_METRICS
        )
        
        self.causal_graph.add_nodes_from(all_sensors)
        
        # Add known causal edges (from physics)
        for parent, child in config.KNOWN_CAUSAL_EDGES:
            self.causal_graph.add_edge(parent, child)
        
        # Add additional physics-based relationships
        physics_edges = [
            # Operational settings affect measurements
            ('op_setting_1', 'P2'),   # Altitude → inlet pressure
            ('op_setting_2', 'T2'),   # Mach number → inlet temp
            ('op_setting_3', 'Nf'),   # Throttle → fan speed
            ('op_setting_3', 'Nc'),   # Throttle → core speed
            
            # Thermodynamic chain
            ('P2', 'P15'),            # Inlet → bypass
            ('P2', 'P30'),            # Inlet → HPC
            ('T2', 'T24'),            # Inlet temp → LPC temp
            ('T24', 'T30'),           # LPC → HPC
            
            # Speed relationships
            ('Nf', 'BPR'),            # Fan speed affects bypass ratio
            ('Nc', 'T30'),            # Core speed affects HPC temp
            ('Nc', 'P30'),            # Core speed affects HPC pressure
        ]
        
        self.causal_graph.add_edges_from(physics_edges)
        
        logger.info(f"Causal graph initialized: {self.causal_graph.number_of_nodes()} nodes, "
                   f"{self.causal_graph.number_of_edges()} edges")
    
    def learn_causal_structure(
        self, 
        df: pd.DataFrame,
        alpha: float = config.CAUSAL_ALPHA
    ) -> nx.DiGraph:
        """
        Refine causal graph using constraint-based learning (PC algorithm)
        
        This augments domain knowledge with data-driven discoveries
        
        Args:
            df: DataFrame with sensor measurements
            alpha: Significance level for conditional independence tests
        
        Returns:
            Updated causal graph
        """
        logger.info("Refining causal structure from data (PC algorithm)")
        
        # Note: Full PC algorithm implementation is complex
        # Here we implement a simplified version using correlation + regression
        
        # For production, use libraries like: causal-learn, py-causal, or pgmpy
        
        sensor_cols = [col for col in df.columns 
                      if col not in ['unit_number', 'time_cycles', 'RUL']]
        
        # Calculate partial correlations to find conditional independencies
        # If X ⊥ Y | Z (X independent of Y given Z), remove edge X→Y
        
        logger.info(f"Analyzing {len(sensor_cols)} sensors for causal relationships")
        
        # Placeholder: In production, use proper causal discovery
        # For now, keep domain knowledge graph as-is
        
        return self.causal_graph
    
    def estimate_causal_effect(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str]
    ) -> float:
        """
        Estimate causal effect of treatment on outcome, adjusting for confounders
        
        Uses backdoor adjustment (regression with confounders)
        
        Args:
            df: DataFrame with measurements
            treatment: Treatment variable (e.g., 'op_setting_1')
            outcome: Outcome variable (e.g., 'T2')
            confounders: List of confounding variables
        
        Returns:
            Estimated causal coefficient
        """
        # Backdoor adjustment: Regress outcome on treatment + confounders
        X = df[[treatment] + confounders].values
        y = df[outcome].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Coefficient for treatment is the causal effect
        causal_effect = model.coef_[0]
        
        return causal_effect
    
    def remove_environmental_confounding(
        self,
        df: pd.DataFrame,
        sensors_to_adjust: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Remove confounding effects of environmental variables on sensor readings
        
        This implements the "Causal Sensor Fusion" concept:
        For each sensor, regress out the effects of operational settings
        
        Args:
            df: DataFrame with sensor data
            sensors_to_adjust: List of sensors to adjust (default: temps and pressures)
        
        Returns:
            Dictionary mapping sensor names to adjustment values
        """
        logger.info("Removing environmental confounding from sensors")
        
        if sensors_to_adjust is None:
            sensors_to_adjust = config.TEMPERATURE_SENSORS + config.PRESSURE_SENSORS
        
        adjustments = {}
        confounders = config.ENVIRONMENTAL_CONFOUNDERS
        
        for sensor in sensors_to_adjust:
            if sensor not in df.columns:
                continue
            
            # Regress sensor on operational settings
            X = df[confounders].values
            y = df[sensor].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Predicted values = effect of environment
            env_effect = model.predict(X)
            
            # Adjustment = remove environmental effect
            adjustments[sensor] = env_effect
            
            # Store model for later use
            self.causal_models[sensor] = model
        
        logger.info(f"Computed causal adjustments for {len(adjustments)} sensors")
        
        return adjustments
    
    def counterfactual_analysis(
        self,
        df: pd.DataFrame,
        sensor: str,
        intervention: Dict[str, float]
    ) -> np.ndarray:
        """
        Perform counterfactual reasoning: "What if operational settings were different?"
        
        Example: "What would T2 be if altitude was at sea level?"
        
        Args:
            df: DataFrame with current observations
            sensor: Sensor to predict
            intervention: Dict of {variable: intervened_value}
        
        Returns:
            Counterfactual sensor values
        """
        if sensor not in self.causal_models:
            logger.warning(f"No causal model for {sensor}, returning observed values")
            return df[sensor].values
        
        model = self.causal_models[sensor]
        
        # Create counterfactual data with intervention
        df_cf = df[config.ENVIRONMENTAL_CONFOUNDERS].copy()
        
        for var, value in intervention.items():
            if var in df_cf.columns:
                df_cf[var] = value
        
        # Predict under intervention
        counterfactual_values = model.predict(df_cf.values)
        
        return counterfactual_values
    
    def detect_cross_system_correlation(
        self,
        df: pd.DataFrame,
        system_groups: Dict[str, List[str]],
        threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        Detect "invisible chains" - correlations across different systems
        
        Example: Landing style (high descent rate) → gear stress + engine stress
        
        Args:
            df: DataFrame with sensor data
            system_groups: Dictionary mapping system names to sensor lists
            threshold: Correlation threshold for flagging
        
        Returns:
            List of (sensor1, sensor2, correlation) tuples
        """
        logger.info("Detecting cross-system correlations")
        
        cross_correlations = []
        
        # Check all pairs across different systems
        system_names = list(system_groups.keys())
        
        for i, sys1 in enumerate(system_names):
            for sys2 in system_names[i+1:]:
                
                for sensor1 in system_groups[sys1]:
                    if sensor1 not in df.columns:
                        continue
                    
                    for sensor2 in system_groups[sys2]:
                        if sensor2 not in df.columns:
                            continue
                        
                        # Calculate correlation
                        corr = df[[sensor1, sensor2]].corr().iloc[0, 1]
                        
                        if abs(corr) > threshold:
                            cross_correlations.append((sensor1, sensor2, corr))
        
        logger.info(f"Found {len(cross_correlations)} strong cross-system correlations")
        
        return cross_correlations
    
    def explain_causal_chain(
        self,
        source: str,
        target: str
    ) -> List[str]:
        """
        Find causal path from source to target in the causal graph
        
        Args:
            source: Source variable
            target: Target variable
        
        Returns:
            List of variables in causal path
        """
        if not self.causal_graph.has_node(source) or not self.causal_graph.has_node(target):
            return []
        
        try:
            path = nx.shortest_path(self.causal_graph, source, target)
            return path
        except nx.NetworkXNoPath:
            return []
    
    def process_data_with_causal_adjustments(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Main interface: Process data to add causally-adjusted sensor readings
        
        Args:
            df: Raw sensor data
        
        Returns:
            DataFrame with additional '_causal_adj' columns
        """
        logger.info("Processing data with causal adjustments")
        
        # Get adjustments
        adjustments = self.remove_environmental_confounding(df)
        
        # Create adjusted columns
        df_causal = df.copy()
        
        for sensor, adjustment in adjustments.items():
            df_causal[f'{sensor}_causal_adj'] = df_causal[sensor] - adjustment
        
        # Detect anomalous cross-system patterns
        system_groups = {
            'temperature': config.TEMPERATURE_SENSORS,
            'pressure': config.PRESSURE_SENSORS,
            'speed': config.SPEED_SENSORS
        }
        
        cross_corrs = self.detect_cross_system_correlation(df, system_groups)
        
        if cross_corrs:
            logger.info(f"Flagged cross-system correlations:")
            for s1, s2, corr in cross_corrs[:5]:  # Show top 5
                logger.info(f"  {s1} ↔ {s2}: {corr:.3f}")
        
        return df_causal


# ==================== STANDALONE FUNCTIONS ====================
def simulate_intervention(
    df: pd.DataFrame,
    intervention_var: str,
    intervention_value: float,
    target_sensors: List[str]
) -> pd.DataFrame:
    """
    Simulate what sensor readings would be under a specific intervention
    
    Useful for: "If we limit flights to sea level, how would temps change?"
    
    Args:
        df: Historical data
        intervention_var: Variable to intervene on (e.g., 'op_setting_1')
        intervention_value: Value to set it to
        target_sensors: Sensors to predict
    
    Returns:
        DataFrame with simulated values
    """
    engine = CausalEngine()
    
    results = {}
    
    for sensor in target_sensors:
        counterfactual = engine.counterfactual_analysis(
            df, 
            sensor, 
            {intervention_var: intervention_value}
        )
        results[f'{sensor}_cf'] = counterfactual
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("AeroGuard Causal Inference Engine - Demo")
    print("=" * 60)
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    demo_data = pd.DataFrame({
        'op_setting_1': np.random.uniform(0, 1, n_samples),  # Altitude
        'op_setting_2': np.random.uniform(0, 1, n_samples),  # Mach
        'op_setting_3': np.random.uniform(0, 1, n_samples),  # Throttle
    })
    
    # Simulate causal relationships
    demo_data['T2'] = 500 + 50 * demo_data['op_setting_1'] + np.random.normal(0, 5, n_samples)
    demo_data['T30'] = 1500 + 100 * demo_data['op_setting_3'] + 0.5 * demo_data['T2'] + np.random.normal(0, 10, n_samples)
    demo_data['P2'] = 14 - 5 * demo_data['op_setting_1'] + np.random.normal(0, 0.5, n_samples)
    
    # Initialize engine
    engine = CausalEngine()
    
    # Remove environmental confounding
    adjustments = engine.remove_environmental_confounding(demo_data)
    
    print(f"\nCausal adjustments computed for {len(adjustments)} sensors")
    print(f"Example: T2 adjustment range: [{adjustments['T2'].min():.2f}, {adjustments['T2'].max():.2f}]")
    
    # Counterfactual analysis
    cf_t2 = engine.counterfactual_analysis(
        demo_data,
        'T2',
        {'op_setting_1': 0}  # Sea level
    )
    
    print(f"\nCounterfactual: T2 at sea level = {cf_t2.mean():.2f}°R")
    print(f"Actual average T2 = {demo_data['T2'].mean():.2f}°R")
    
    print("\n" + "=" * 60)
    print("Causal Engine Ready")
    print("=" * 60)
