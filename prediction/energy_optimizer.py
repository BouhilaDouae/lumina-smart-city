import numpy as np
import pandas as pd

def traffic_to_dimming_level(traffic_flow):
    """
    Converts predicted traffic flow (pedestrians + vehicles) to discrete dimming levels:
      - 0 to 5 people -> 30%
      - 5 to 20 people -> 60%
      - > 20 people -> 100%
    """
    if traffic_flow <= 5.0:
        return 30.0
    elif traffic_flow <= 20.0:
        return 60.0
    else:
        return 100.0

def calculate_lamp_power(brightness_pct, p_max=150.0, alpha=0.1):
    """
    Calculates power draw (W) of a municipal LED lamp based on its dimming brightness %.
    Formula: P = P_max * (alpha + (1 - alpha) * (brightness / 100))
    alpha: standby coefficient (0.1 means 10% standby draw even when dimmed to minimum)
    """
    return p_max * (alpha + (1.0 - alpha) * (brightness_pct / 100.0))

def run_energy_optimization(df_sim, rate_per_kwh=0.12, co2_kg_per_kwh=0.533):
    """
    Evaluates energy savings of STGCN Adaptive Dimming vs. Fixed Lighting Baseline (100%).
    
    df_sim: DataFrame containing actual and predicted pedestrian/vehicle counts.
    Returns:
        metrics_df: DataFrame with total energy, cost, CO2, and savings comparisons.
        detailed_df: DataFrame with hourly lamp metrics.
    """
    # Initialize list for records
    records = []
    
    # Max power and standby
    p_max = 150.0 # W
    alpha = 0.1
    
    for idx, row in df_sim.iterrows():
        # Get actual and predicted values
        actual_ped = row.get('Target_Pedestrians', row.get('Pedestrians', 0.0))
        actual_veh = row.get('Target_Vehicles', row.get('Vehicles', 0.0))
        pred_ped = row.get('Pred_Pedestrians', actual_ped)
        pred_veh = row.get('Pred_Vehicles', actual_veh)
        
        # Calculate combined flow for actual and predicted
        actual_flow = actual_ped + actual_veh
        pred_flow = pred_ped + pred_veh
        
        # 1. Fixed Baseline Policy (constant 100%)
        b_fixed = 100.0
        p_fixed = calculate_lamp_power(b_fixed, p_max, alpha)
        
        # 2. Rule-Based Control (RBC) Policy (uses actual flow for perfect foresight baseline)
        b_rbc = traffic_to_dimming_level(actual_flow)
        p_rbc = calculate_lamp_power(b_rbc, p_max, alpha)
        
        # 3. STGCN Adaptive AI Policy (uses predicted flow)
        b_adaptive = traffic_to_dimming_level(pred_flow)
        p_adaptive = calculate_lamp_power(b_adaptive, p_max, alpha)
        
        # Safety Check:
        # A safety violation occurs if actual traffic is high (flow > 20)
        # but the policy set brightness below 100%.
        violation_fixed = 0
        violation_rbc = 1 if actual_flow > 20.0 and b_rbc < 100.0 else 0
        violation_adaptive = 1 if actual_flow > 20.0 and b_adaptive < 100.0 else 0
        
        records.append({
            'Timestamp': row.get('Timestamp'),
            'Node_ID': row.get('Node_ID'),
            'Actual_Flow': actual_flow,
            'Pred_Flow': pred_flow,
            # Brightness levels
            'Brightness_Fixed': b_fixed,
            'Brightness_RBC': b_rbc,
            'Brightness_Adaptive': b_adaptive,
            # Power (W)
            'Power_Fixed': p_fixed,
            'Power_RBC': p_rbc,
            'Power_Adaptive': p_adaptive,
            # Energy (kWh) in 1 hour
            'kWh_Fixed': p_fixed / 1000.0,
            'kWh_RBC': p_rbc / 1000.0,
            'kWh_Adaptive': p_adaptive / 1000.0,
            # Safety Violations
            'Violation_Fixed': violation_fixed,
            'Violation_RBC': violation_rbc,
            'Violation_Adaptive': violation_adaptive
        })
        
    detailed_df = pd.DataFrame(records)
    
    # Calculate global aggregated metrics
    total_slots = len(detailed_df)
    
    policies = {
        'Fixed Lighting (100%)': 'Fixed',
        'Rule-Based Dimming (RBC)': 'RBC',
        'STGCN Adaptive Dimming': 'Adaptive'
    }
    
    summary_data = []
    
    for name, key in policies.items():
        total_kwh = detailed_df[f'kWh_{key}'].sum()
        total_cost = total_kwh * rate_per_kwh
        total_co2 = total_kwh * co2_kg_per_kwh
        total_violations = detailed_df[f'Violation_{key}'].sum()
        violation_rate = (total_violations / total_slots) * 100.0 if total_slots > 0 else 0.0
        avg_brightness = detailed_df[f'Brightness_{key}'].mean()
        
        summary_data.append({
            'Policy': name,
            'Total Energy (kWh)': round(total_kwh, 2),
            'Total Cost ($)': round(total_cost, 2),
            'CO2 Emissions (kg)': round(total_co2, 2),
            'Safety Violations': int(total_violations),
            'Safety Violation Rate (%)': round(violation_rate, 3),
            'Average Brightness (%)': round(avg_brightness, 2)
        })
        
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate savings relative to Fixed Lighting (100%)
    baseline_kwh = summary_df.loc[summary_df['Policy'] == 'Fixed Lighting (100%)', 'Total Energy (kWh)'].values[0]
    baseline_cost = summary_df.loc[summary_df['Policy'] == 'Fixed Lighting (100%)', 'Total Cost ($)'].values[0]
    baseline_co2 = summary_df.loc[summary_df['Policy'] == 'Fixed Lighting (100%)', 'CO2 Emissions (kg)'].values[0]
    
    summary_df['Energy Savings (%)'] = (((baseline_kwh - summary_df['Total Energy (kWh)']) / baseline_kwh) * 100.0).round(2)
    summary_df['Financial Savings ($)'] = (baseline_cost - summary_df['Total Cost ($)']).round(2)
    summary_df['CO2 Avoided (kg)'] = (baseline_co2 - summary_df['CO2 Emissions (kg)']).round(2)
    
    return summary_df, detailed_df
