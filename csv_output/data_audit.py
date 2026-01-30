"""
AeroGuard Data Quality Audit Script

This script analyzes your FDR CSV files to understand:
1. Data availability and completeness
2. Sensor coverage across files
3. Temporal consistency
4. Feature extraction feasibility
5. RUL label validation

Run: python data_audit.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def audit_fdr_data(data_dir='data', output_dir='audit_reports'):
    """Comprehensive data quality analysis"""
    
    print("=" * 60)
    print("AEROGUARD FDR DATA QUALITY AUDIT")
    print("=" * 60)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    csv_files = sorted(list(Path(data_dir).glob('*.csv')))
    
    if len(csv_files) == 0:
        print(f"❌ No CSV files found in {data_dir}")
        return None
    
    print(f"\n📊 Found {len(csv_files)} FDR files")
    print(f"Date range: {csv_files[0].stem} to {csv_files[-1].stem}")
    
    # Initialize statistics collectors
    stats = {
        'total_files': len(csv_files),
        'column_coverage': defaultdict(int),
        'records_per_file': [],
        'missing_data_pct': [],
        'sensor_ranges': defaultdict(lambda: {'min': [], 'max': [], 'mean': []}),
        'file_info': [],
    }
    
    # Key sensors we're looking for
    target_sensors = {
        'engines': ['EGT_1', 'EGT_2', 'EGT_3', 'EGT_4', 
                   'N1_1', 'N1_2', 'N1_3', 'N1_4',
                   'N2_1', 'N2_2', 'N2_3', 'N2_4',
                   'FF_1', 'FF_2', 'FF_3', 'FF_4',
                   'OIT_1', 'OIT_2', 'OIT_3', 'OIT_4',
                   'OIP_1', 'OIP_2', 'OIP_3', 'OIP_4',
                   'VIB_1', 'VIB_2', 'VIB_3', 'VIB_4'],
        'flight': ['ALT', 'TAS', 'MACH', 'TAT', 'SAT', 'VRTG'],
        'time': ['GMT_HOUR', 'GMT_MINUTE', 'GMT_SEC', 'DATE_YEAR', 'DATE_MONTH', 'DATE_DAY']
    }
    
    print("\n🔍 Processing files...")
    
    for csv_file in tqdm(csv_files, desc="Scanning files"):
        try:
            df = pd.read_csv(csv_file)
            
            # Basic file info
            file_info = {
                'filename': csv_file.name,
                'records': len(df),
                'columns': len(df.columns),
                'missing_pct': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
            }
            stats['file_info'].append(file_info)
            stats['records_per_file'].append(len(df))
            stats['missing_data_pct'].append(file_info['missing_pct'])
            
            # Column coverage
            for col in df.columns:
                stats['column_coverage'][col] += 1
            
            # Sensor ranges (only for numeric columns)
            for sensor_group in target_sensors.values():
                for sensor in sensor_group:
                    if sensor in df.columns:
                        try:
                            values = pd.to_numeric(df[sensor], errors='coerce').dropna()
                            if len(values) > 0:
                                stats['sensor_ranges'][sensor]['min'].append(values.min())
                                stats['sensor_ranges'][sensor]['max'].append(values.max())
                                stats['sensor_ranges'][sensor]['mean'].append(values.mean())
                        except:
                            pass
        
        except Exception as e:
            print(f"\n⚠️ Error processing {csv_file.name}: {e}")
            continue
    
    # Generate report
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    
    # 1. File statistics
    print(f"\n📁 FILE STATISTICS:")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Records per file: {np.mean(stats['records_per_file']):.0f} ± {np.std(stats['records_per_file']):.0f}")
    print(f"   Min records: {np.min(stats['records_per_file'])}")
    print(f"   Max records: {np.max(stats['records_per_file'])}")
    print(f"   Missing data: {np.mean(stats['missing_data_pct']):.1f}% average")
    
    # 2. Column availability
    print(f"\n📋 COLUMN AVAILABILITY:")
    print(f"   Unique columns across all files: {len(stats['column_coverage'])}")
    print(f"\n   Top 20 most common columns:")
    sorted_cols = sorted(stats['column_coverage'].items(), key=lambda x: -x[1])
    for col, count in sorted_cols[:20]:
        pct = 100 * count / stats['total_files']
        status = "✅" if pct > 95 else "⚠️" if pct > 50 else "❌"
        print(f"   {status} {col:20s}: {count:4d}/{stats['total_files']} ({pct:5.1f}%)")
    
    # 3. Target sensor availability
    print(f"\n🎯 TARGET SENSOR AVAILABILITY:")
    for group_name, sensors in target_sensors.items():
        print(f"\n   {group_name.upper()} sensors:")
        for sensor in sensors:
            count = stats['column_coverage'].get(sensor, 0)
            pct = 100 * count / stats['total_files']
            status = "✅" if pct > 95 else "⚠️" if pct > 50 else "❌"
            print(f"   {status} {sensor:15s}: {count:4d}/{stats['total_files']} ({pct:5.1f}%)")
    
    # 4. Sensor range analysis
    print(f"\n📊 SENSOR RANGE ANALYSIS (for available sensors):")
    print(f"\n   {'Sensor':<15} {'Min':>10} {'Mean':>10} {'Max':>10} {'Files':>8}")
    print(f"   {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    
    for sensor, ranges in sorted(stats['sensor_ranges'].items()):
        if len(ranges['mean']) > 10:  # Only show sensors with good coverage
            avg_min = np.mean(ranges['min'])
            avg_mean = np.mean(ranges['mean'])
            avg_max = np.mean(ranges['max'])
            num_files = len(ranges['mean'])
            
            print(f"   {sensor:<15} {avg_min:10.1f} {avg_mean:10.1f} {avg_max:10.1f} {num_files:8d}")
    
    # 5. Data quality warnings
    print(f"\n⚠️ DATA QUALITY WARNINGS:")
    warnings = []
    
    # Check for low coverage of critical sensors
    critical_sensors = ['EGT_1', 'N1_1', 'FF_1', 'ALT']
    for sensor in critical_sensors:
        count = stats['column_coverage'].get(sensor, 0)
        pct = 100 * count / stats['total_files']
        if pct < 80:
            warnings.append(f"   ❌ Low coverage for {sensor}: only {pct:.1f}% of files")
    
    # Check for high missing data
    if np.mean(stats['missing_data_pct']) > 20:
        warnings.append(f"   ⚠️ High missing data rate: {np.mean(stats['missing_data_pct']):.1f}% average")
    
    # Check for inconsistent file sizes
    if np.std(stats['records_per_file']) > 0.5 * np.mean(stats['records_per_file']):
        warnings.append(f"   ⚠️ High variability in flight durations (records per file)")
    
    if warnings:
        for warning in warnings:
            print(warning)
    else:
        print("   ✅ No major issues detected")
    
    # 6. Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    # Count available engine sensors per engine
    engine_sensor_availability = {}
    for engine_num in range(1, 5):
        available = sum(1 for s in ['EGT', 'N1', 'N2', 'FF', 'OIT', 'OIP', 'VIB'] 
                       if stats['column_coverage'].get(f'{s}_{engine_num}', 0) > 0.8 * stats['total_files'])
        engine_sensor_availability[engine_num] = available
    
    best_engine = max(engine_sensor_availability, key=engine_sensor_availability.get)
    print(f"   1. Focus on Engine {best_engine} (best sensor coverage: {engine_sensor_availability[best_engine]}/7 key sensors)")
    
    if np.mean(stats['records_per_file']) > 100:
        print(f"   2. ✅ Good temporal resolution ({np.mean(stats['records_per_file']):.0f} records/flight)")
        print(f"      → Use within-flight time-series features (trends, slopes)")
    else:
        print(f"   3. ⚠️ Low temporal resolution ({np.mean(stats['records_per_file']):.0f} records/flight)")
        print(f"      → Limited within-flight pattern detection")
    
    if 'GMT_HOUR' in stats['column_coverage'] and 'DATE_YEAR' in stats['column_coverage']:
        print(f"   3. ✅ Timestamp data available")
        print(f"      → Can reconstruct flight chronology")
    
    # Save detailed report
    report_file = Path(output_dir) / 'data_quality_report.txt'
    with open(report_file, 'w') as f:
        f.write("AEROGUARD FDR DATA QUALITY AUDIT REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Files analyzed: {stats['total_files']}\n\n")
        
        f.write("COLUMN COVERAGE\n")
        f.write("-" * 60 + "\n")
        for col, count in sorted_cols:
            pct = 100 * count / stats['total_files']
            f.write(f"{col:30s} {count:4d}/{stats['total_files']} ({pct:5.1f}%)\n")
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Records per file distribution
    axes[0, 0].hist(stats['records_per_file'], bins=30, edgecolor='black')
    axes[0, 0].set_title('Records per File Distribution')
    axes[0, 0].set_xlabel('Number of Records')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(stats['records_per_file']), color='red', 
                      linestyle='--', label=f'Mean: {np.mean(stats["records_per_file"]):.0f}')
    axes[0, 0].legend()
    
    # Plot 2: Missing data distribution
    axes[0, 1].hist(stats['missing_data_pct'], bins=30, edgecolor='black', color='orange')
    axes[0, 1].set_title('Missing Data Distribution')
    axes[0, 1].set_xlabel('Missing Data (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(stats['missing_data_pct']), color='red', 
                      linestyle='--', label=f'Mean: {np.mean(stats["missing_data_pct"]):.1f}%')
    axes[0, 1].legend()
    
    # Plot 3: Top sensors availability
    top_sensors = sorted_cols[:15]
    sensor_names = [s[0] for s in top_sensors]
    sensor_coverage = [100 * s[1] / stats['total_files'] for s in top_sensors]
    
    axes[1, 0].barh(sensor_names, sensor_coverage, color='steelblue')
    axes[1, 0].set_xlabel('Coverage (%)')
    axes[1, 0].set_title('Top 15 Sensor Availability')
    axes[1, 0].axvline(95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
    axes[1, 0].legend()
    axes[1, 0].invert_yaxis()
    
    # Plot 4: EGT range across files (if available)
    if 'EGT_1' in stats['sensor_ranges'] and len(stats['sensor_ranges']['EGT_1']['mean']) > 0:
        egt_data = stats['sensor_ranges']['EGT_1']['mean']
        axes[1, 1].plot(egt_data, alpha=0.7, linewidth=0.5)
        axes[1, 1].set_title('EGT_1 Mean Values Across Flights')
        axes[1, 1].set_xlabel('Flight Index (chronological)')
        axes[1, 1].set_ylabel('EGT (°C)')
        axes[1, 1].axhline(np.mean(egt_data), color='red', linestyle='--', 
                          label=f'Overall Mean: {np.mean(egt_data):.1f}°C')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'EGT_1 data not available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    viz_file = Path(output_dir) / 'data_quality_visualizations.png'
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"   Saved visualizations to: {viz_file}")
    
    plt.close()
    
    print("\n" + "=" * 60)
    print("AUDIT COMPLETE!")
    print("=" * 60)
    
    return stats


def validate_current_labels(data_dir='data', output_dir='audit_reports'):
    """
    Validate whether current filename-based RUL labels correlate with
    physics-based degradation indicators
    """
    
    print("\n" + "=" * 60)
    print("RUL LABEL VALIDATION")
    print("=" * 60)
    
    csv_files = sorted(list(Path(data_dir).glob('*.csv')))
    
    labels_filename = []
    labels_egt_based = []
    labels_multivariate = []
    
    print("\n🔍 Analyzing degradation indicators...")
    
    for idx, csv_file in enumerate(tqdm(csv_files, desc="Processing")):
        try:
            df = pd.read_csv(csv_file)
            
            # Current labeling (by file order)
            rul_filename = len(csv_files) - idx
            labels_filename.append(rul_filename)
            
            # Physics-based estimation (if EGT available)
            if 'EGT_1' in df.columns:
                avg_egt = pd.to_numeric(df['EGT_1'], errors='coerce').mean()
                # Normalize: typical range 600-950°C, higher = worse
                health_egt = 1.0 - np.clip((avg_egt - 600) / 350, 0, 1)
                rul_egt = health_egt * len(csv_files)
                labels_egt_based.append(rul_egt)
            else:
                labels_egt_based.append(np.nan)
            
            # Multivariate health score
            health_score = 1.0
            
            if 'EGT_1' in df.columns:
                avg_egt = pd.to_numeric(df['EGT_1'], errors='coerce').mean()
                health_score *= (1.0 - np.clip((avg_egt - 600) / 350, 0, 1))
            
            if 'VIB_1' in df.columns:
                avg_vib = pd.to_numeric(df['VIB_1'], errors='coerce').mean()
                health_score *= (1.0 - np.clip((avg_vib - 0) / 100, 0, 1))  # Adjust range
            
            rul_multi = health_score * len(csv_files)
            labels_multivariate.append(rul_multi)
            
        except Exception as e:
            labels_egt_based.append(np.nan)
            labels_multivariate.append(np.nan)
    
    # Remove NaNs for correlation analysis
    labels_egt_valid = [x for x in labels_egt_based if not np.isnan(x)]
    labels_multi_valid = [x for x in labels_multivariate if not np.isnan(x)]
    
    print(f"\n📊 RESULTS:")
    print(f"   Files analyzed: {len(csv_files)}")
    print(f"   Files with EGT data: {len(labels_egt_valid)} ({100*len(labels_egt_valid)/len(csv_files):.1f}%)")
    
    if len(labels_egt_valid) > 10:
        # Calculate correlation
        filename_subset = labels_filename[:len(labels_egt_valid)]
        correlation = np.corrcoef(filename_subset, labels_egt_valid)[0, 1]
        
        print(f"\n🔍 LABEL VALIDATION:")
        print(f"   Correlation (filename-based vs EGT-based RUL): {correlation:.3f}")
        
        if abs(correlation) > 0.7:
            print(f"   ✅ GOOD: Labels are reasonably aligned with physics")
        elif abs(correlation) > 0.3:
            print(f"   ⚠️ MODERATE: Some alignment but significant noise")
        else:
            print(f"   ❌ POOR: Current labels DO NOT match physical degradation!")
            print(f"   → Recommend using physics-based or maintenance record labels")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(filename_subset, labels_egt_valid, alpha=0.5)
        axes[0].set_xlabel('Filename-based RUL')
        axes[0].set_ylabel('EGT-based RUL')
        axes[0].set_title(f'RUL Label Comparison (r={correlation:.3f})')
        axes[0].plot([0, max(filename_subset)], [0, max(filename_subset)], 
                    'r--', alpha=0.5, label='Perfect correlation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Time series
        axes[1].plot(labels_filename, label='Filename-based', alpha=0.7)
        axes[1].plot(labels_egt_valid, label='EGT-based', alpha=0.7)
        axes[1].set_xlabel('Flight Index')
        axes[1].set_ylabel('RUL (flights)')
        axes[1].set_title('RUL Labels Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_file = Path(output_dir) / 'label_validation.png'
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"\n📊 Saved visualization to: {viz_file}")
        plt.close()
    else:
        print(f"   ⚠️ Insufficient EGT data for validation")
    
    return {
        'filename_labels': labels_filename,
        'egt_labels': labels_egt_based,
        'multivariate_labels': labels_multivariate
    }


if __name__ == '__main__':
    # Run audit
    stats = audit_fdr_data(data_dir='data', output_dir='audit_reports')
    
    if stats:
        # Run label validation
        label_analysis = validate_current_labels(data_dir='data', output_dir='audit_reports')
        
        print("\n✅ Audit complete! Check 'audit_reports/' directory for detailed results.")
