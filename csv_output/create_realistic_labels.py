"""
Simple but Realistic RUL Labeling:
Find the flight with worst degradation and use it as the failure point
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('.')

def simple_health_score(csv_file):
    """Calculate simple health score from a flight"""
    try:
        df = pd.read_csv(csv_file)
        
        # Use EGT (exhaust gas temp) as primary degradation indicator
        egt_scores = []
        for i in range(1, 5):
            col = f'EGT_{i}'
            if col in df.columns:
                egt_mean = df[col].mean()
                egt_scores.append(egt_mean)
        
        if egt_scores:
            # Higher EGT = worse health
            avg_egt = np.mean(egt_scores)
            # Normalize: 450°C = good (1.0), 600°C = bad (0.0)
            health = max(0, min(1, (600 - avg_egt) / 150))
            return health
        
        return 0.5  # Default midpoint
    except:
        return 0.5

# Find all CSV files
csv_files = sorted(list(Path('data').glob('*.csv')))
print(f"Analyzing {len(csv_files)} flights...")

# Calculate health for each flight
health_scores = []
for f in tqdm(csv_files):
    health = simple_health_score(f)
    health_scores.append(health)

# Find worst health (lowest score) = failure point
failure_idx = np.argmin(health_scores)
print(f"\nFailure point identified: Flight {failure_idx} (file: {csv_files[failure_idx].name})")
print(f"Health score at failure: {health_scores[failure_idx]:.3f}")

# Create RUL labels
ruls = []
for idx in range(len(csv_files)):
    rul = abs(failure_idx - idx)  # Distance from failure
    ruls.append(rul)

print(f"RUL range: {min(ruls)} to {max(ruls)}")
print(f"Mean RUL: {np.mean(ruls):.1f}")

# Save
df = pd.DataFrame({
    'filename': [f.name for f in csv_files],
    'health_score': health_scores,
    'rul': ruls
})
df.to_csv('realistic_labels.csv', index=False)
print("\nSaved to realistic_labels.csv")
