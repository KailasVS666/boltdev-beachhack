
import pandas as pd
import numpy as np
import json
import os
import glob
from src.config import SENSOR_FEATURES

def find_steady_state_periods(df, duration_min=10, sampling_rate_hz=1):
    """
    Identifies periods of 'Steady State' telemetry.
    Steady State = Low variance in key parameters (ALT, VEL, TH, N1) for > duration_min.
    """
    window_size = int(duration_min * 60 * sampling_rate_hz)
    
    # Calculate rolling variance/std for stability check
    # We focus on Cruise parameters: ALT, VEL
    
    if 'ALT' not in df.columns or 'VEL' not in df.columns:
        return []
        
    # Normalize or use percentage change to handle different scales?
    # Simple relative range check: (max - min) / mean < threshold
    
    rolling_std_alt = df['ALT'].rolling(window=window_size).std()
    rolling_std_vel = df['VEL'].rolling(window=window_size).std()
    
    # Thresholds for stability (e.g., ALT var < 50ft, VEL var < 10 knots)
    # These are heuristic values; ideally from config.
    stable_mask = (rolling_std_alt < 20.0) & (rolling_std_vel < 5.0)
    
    steady_periods = []
    
    # Find contiguous regions
    # Identify change points
    changes = stable_mask.ne(stable_mask.shift()).cumsum()
    
    for gid, group in stable_mask.groupby(changes):
        if group.iloc[0]: # If this group is True (Stable)
            if len(group) >= window_size:
                start_idx = group.index[0]
                end_idx = group.index[-1]
                steady_periods.append({
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "duration_seconds": len(group) / sampling_rate_hz,
                    "avg_alt": float(df.loc[start_idx:end_idx, 'ALT'].mean()),
                    "avg_vel": float(df.loc[start_idx:end_idx, 'VEL'].mean())
                })
    
    return steady_periods

def extract_logs_to_json(csv_dir="csv_output", output_dir="json_logs"):
    """
    Reads CSVs, finds steady states, and anomalies, and exports JSON logs.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    processed_count = 0
    
    for f in csv_files:
        if "narratives" in f or "events" in f: continue
        
        try:
            print(f"Processing {os.path.basename(f)}...")
            df = pd.read_csv(f)
            
            # Ensure columns exist (mapping logic from sfi_validation could be reused here if needed)
            # For now assume mostly standard columns or minimal set
            
            # 1. Steady State Extraction
            steady_periods = find_steady_state_periods(df)
            
            # 2. Anomaly Candidates (Simple logic: exceeding thresholds)
            # This mimics "Active Alert" extraction
            anomalies = []
            if 'EGT' in df.columns:
                 high_egt = df[df['EGT'] > 900]
                 if not high_egt.empty:
                     anomalies.append({
                         "type": "EGT_OVERHEAT",
                         "count": len(high_egt),
                         "first_timestamp_idx": int(high_egt.index[0])
                     })
            
            # 3. Construct JSON Log
            log_data = {
                "source_file": os.path.basename(f),
                "total_rows": len(df),
                "steady_state_periods": steady_periods,
                "anomalies_detected": anomalies,
                "status": "PROCESSED"
            }
            
            out_name = os.path.basename(f).replace(".csv", "_log.json")
            with open(os.path.join(output_dir, out_name), 'w') as jf:
                json.dump(log_data, jf, indent=2)
                
            processed_count += 1
            
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    print(f"Generated JSON logs for {processed_count} files in '{output_dir}/'")

if __name__ == "__main__":
    extract_logs_to_json()
