"""
================================================================================
200 EXTREME CONDITION SAMPLES FOR MACHINE FAILURE PREDICTION
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

# ============================================================================
# GENERATE 200 EXTREME CONDITION SAMPLES
# ============================================================================

print("Generating 200 extreme condition samples...")

extreme_samples = []

# ============================================================================
# PATTERN 1: EXTREME OVERHEATING (40 samples)
# ============================================================================

for i in range(40):
    air_temp = np.random.uniform(306, 310)
    process_temp = np.random.uniform(318, 322)
    speed = np.random.uniform(2700, 3000)
    torque = np.random.uniform(46, 52)
    tool_wear = np.random.uniform(200, 260)
    
    extreme_samples.append({
        'UDI': 10000 + i,
        'Product ID': f'M{np.random.randint(10000, 99999)}',
        'Type': np.random.choice(['M', 'H']),
        'Air temperature [K]': round(air_temp, 1),
        'Process temperature [K]': round(process_temp, 1),
        'Rotational speed [rpm]': int(speed),
        'Torque [Nm]': round(torque, 1),
        'Tool wear [min]': int(tool_wear),
        'Machine failure': 1,
        'TWF': 0,
        'HDF': 1,
        'PWF': 0,
        'OSF': 0,
        'RNF': 0
    })

# ============================================================================
# PATTERN 2: EXTREME TOOL WEAR + HIGH TORQUE (40 samples)
# ============================================================================

for i in range(40, 80):
    air_temp = np.random.uniform(301, 306)
    process_temp = np.random.uniform(313, 318)
    speed = np.random.uniform(2200, 2600)
    torque = np.random.uniform(48, 52)
    tool_wear = np.random.uniform(230, 260)
    
    extreme_samples.append({
        'UDI': 10000 + i,
        'Product ID': f'M{np.random.randint(10000, 99999)}',
        'Type': 'M',
        'Air temperature [K]': round(air_temp, 1),
        'Process temperature [K]': round(process_temp, 1),
        'Rotational speed [rpm]': int(speed),
        'Torque [Nm]': round(torque, 1),
        'Tool wear [min]': int(tool_wear),
        'Machine failure': 1,
        'TWF': 1,
        'HDF': 0,
        'PWF': 0,
        'OSF': 0,
        'RNF': 0
    })

# ============================================================================
# PATTERN 3: EXTREME SPEED + HIGH TEMPERATURE (40 samples)
# ============================================================================

for i in range(80, 120):
    air_temp = np.random.uniform(304, 309)
    process_temp = np.random.uniform(316, 321)
    speed = np.random.uniform(2800, 3000)
    torque = np.random.uniform(43, 48)
    tool_wear = np.random.uniform(180, 240)
    
    extreme_samples.append({
        'UDI': 10000 + i,
        'Product ID': f'H{np.random.randint(10000, 99999)}',
        'Type': 'H',
        'Air temperature [K]': round(air_temp, 1),
        'Process temperature [K]': round(process_temp, 1),
        'Rotational speed [rpm]': int(speed),
        'Torque [Nm]': round(torque, 1),
        'Tool wear [min]': int(tool_wear),
        'Machine failure': 1,
        'TWF': 0,
        'HDF': 0,
        'PWF': 1,
        'OSF': 0,
        'RNF': 0
    })

# ============================================================================
# PATTERN 4: EXTREME OVERSTRAIN (40 samples)
# ============================================================================

for i in range(120, 160):
    air_temp = np.random.uniform(302, 307)
    process_temp = np.random.uniform(314, 319)
    speed = np.random.uniform(2500, 2900)
    torque = np.random.uniform(47, 52)
    tool_wear = np.random.uniform(200, 250)
    
    extreme_samples.append({
        'UDI': 10000 + i,
        'Product ID': f'M{np.random.randint(10000, 99999)}',
        'Type': 'M',
        'Air temperature [K]': round(air_temp, 1),
        'Process temperature [K]': round(process_temp, 1),
        'Rotational speed [rpm]': int(speed),
        'Torque [Nm]': round(torque, 1),
        'Tool wear [min]': int(tool_wear),
        'Machine failure': 1,
        'TWF': 0,
        'HDF': 0,
        'PWF': 0,
        'OSF': 1,
        'RNF': 0
    })

# ============================================================================
# PATTERN 5: MULTIPLE FAILURE MODES (40 samples)
# ============================================================================

for i in range(160, 200):
    air_temp = np.random.uniform(305, 310)
    process_temp = np.random.uniform(317, 322)
    speed = np.random.uniform(2600, 3000)
    torque = np.random.uniform(45, 52)
    tool_wear = np.random.uniform(210, 260)
    
    # Randomly assign multiple failure modes
    twf = np.random.choice([0, 1], p=[0.6, 0.4])
    hdf = np.random.choice([0, 1], p=[0.5, 0.5])
    pwf = np.random.choice([0, 1], p=[0.7, 0.3])
    osf = np.random.choice([0, 1], p=[0.7, 0.3])
    rnf = np.random.choice([0, 1], p=[0.9, 0.1])
    
    extreme_samples.append({
        'UDI': 10000 + i,
        'Product ID': f'{np.random.choice(["M", "L", "H"])}{np.random.randint(10000, 99999)}',
        'Type': np.random.choice(['M', 'L', 'H']),
        'Air temperature [K]': round(air_temp, 1),
        'Process temperature [K]': round(process_temp, 1),
        'Rotational speed [rpm]': int(speed),
        'Torque [Nm]': round(torque, 1),
        'Tool wear [min]': int(tool_wear),
        'Machine failure': 1,
        'TWF': twf,
        'HDF': hdf,
        'PWF': pwf,
        'OSF': osf,
        'RNF': rnf
    })

# ============================================================================
# CREATE DATAFRAME
# ============================================================================

df_extreme = pd.DataFrame(extreme_samples)

# Add some normal samples for context (optional)
normal_samples = []
for i in range(50):
    normal_samples.append({
        'UDI': 20000 + i,
        'Product ID': f'M{np.random.randint(10000, 99999)}',
        'Type': np.random.choice(['M', 'L', 'H']),
        'Air temperature [K]': round(np.random.uniform(296, 302), 1),
        'Process temperature [K]': round(np.random.uniform(305, 312), 1),
        'Rotational speed [rpm]': int(np.random.uniform(1200, 1800)),
        'Torque [Nm]': round(np.random.uniform(32, 42), 1),
        'Tool wear [min]': int(np.random.exponential(50)),
        'Machine failure': 0,
        'TWF': 0,
        'HDF': 0,
        'PWF': 0,
        'OSF': 0,
        'RNF': 0
    })

df_normal = pd.DataFrame(normal_samples)

# Combine extreme and normal samples
df_extreme_with_context = pd.concat([df_extreme, df_normal], ignore_index=True)

# Shuffle
df_extreme_with_context = df_extreme_with_context.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================================================
# SAVE TO CSV
# ============================================================================

df_extreme_with_context.to_csv('data/extreme_conditions_200.csv', index=False)
print(f"Saved 200 extreme condition samples + 50 normal samples to 'data/extreme_conditions_200.csv'")

# ============================================================================
# DISPLAY SUMMARY
# ============================================================================

print("\n" + "="*80)
print("EXTREME CONDITIONS SUMMARY")
print("="*80)

print(f"\nTotal samples: {len(df_extreme_with_context)}")
print(f"Failures: {len(df_extreme_with_context[df_extreme_with_context['Machine failure'] == 1])}")
print(f"Normal: {len(df_extreme_with_context[df_extreme_with_context['Machine failure'] == 0])}")
print(f"Failure Rate: {df_extreme_with_context['Machine failure'].mean()*100:.2f}%")

print("\nFailure Mode Distribution:")
print(f"  TWF (Tool Wear Failure): {df_extreme_with_context['TWF'].sum()}")
print(f"  HDF (Heat Dissipation Failure): {df_extreme_with_context['HDF'].sum()}")
print(f"  PWF (Power Failure): {df_extreme_with_context['PWF'].sum()}")
print(f"  OSF (Overstrain Failure): {df_extreme_with_context['OSF'].sum()}")
print(f"  RNF (Random Failure): {df_extreme_with_context['RNF'].sum()}")

print("\nFeature Statistics (Extreme Samples):")
extreme_only = df_extreme_with_context[df_extreme_with_context['Machine failure'] == 1]
print(extreme_only[['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']].describe())

# ============================================================================
# DISPLAY SAMPLE EXTREME ROWS
# ============================================================================

print("\n" + "="*80)
print("SAMPLE EXTREME ROWS (First 10)")
print("="*80)
print(df_extreme.head(10).to_string())

# ============================================================================
# SAVE AS CSV WITH PROPER FORMATTING
# ============================================================================

# Also save just the extreme samples (without normal)
df_extreme_only = df_extreme
df_extreme_only.to_csv('data/extreme_conditions_only_200.csv', index=False)
print("\nSaved extreme-only samples to 'data/extreme_conditions_only_200.csv'")