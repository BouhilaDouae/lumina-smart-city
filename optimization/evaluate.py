import pandas as pd
import numpy as np

def evaluate_dimming_policies(df_sim, rate_per_kwh=0.12, co2_g_per_kwh=533.0):
    """
    Evaluates dimming policies on total energy, cost, CO2 emissions, and safety violations.
    
    rate_per_kwh: float, grid tariff ($/kWh)
    co2_g_per_kwh: float, carbon intensity of electricity (g CO2 per kWh)
    """
    metrics = []
    
    policies = {
        'Static Baseline': {
            'kwh': 'kWh_Base',
            'violation': 'Violation_Base'
        },
        'Rule-Based Control (RBC)': {
            'kwh': 'kWh_RBC',
            'violation': 'Violation_RBC'
        },
        'Adaptive AI Control (AIC)': {
            'kwh': 'kWh_AIC',
            'violation': 'Violation_AIC'
        }
    }
    
    total_slots = len(df_sim)
    
    for name, cols in policies.items():
        total_kwh = df_sim[cols['kwh']].sum()
        total_cost = total_kwh * rate_per_kwh
        total_co2 = total_kwh * co2_g_per_kwh
        
        # Safety Violations count
        total_violations = df_sim[cols['violation']].sum()
        violation_rate = (total_violations / total_slots) * 100.0 if total_slots > 0 else 0.0
        
        metrics.append({
            'Policy': name,
            'Total Energy (kWh)': total_kwh,
            'Total Cost ($)': total_cost,
            'CO2 Emissions (kg)': total_co2 / 1000.0,
            'Safety Violations': total_violations,
            'Safety Violation Rate (%)': violation_rate
        })
        
    df_metrics = pd.DataFrame(metrics)
    
    # Calculate savings relative to the Static Baseline
    baseline_kwh = df_metrics.loc[df_metrics['Policy'] == 'Static Baseline', 'Total Energy (kWh)'].values[0]
    baseline_cost = df_metrics.loc[df_metrics['Policy'] == 'Static Baseline', 'Total Cost ($)'].values[0]
    baseline_co2 = df_metrics.loc[df_metrics['Policy'] == 'Static Baseline', 'CO2 Emissions (kg)'].values[0]
    
    df_metrics['Energy Savings (%)'] = ((baseline_kwh - df_metrics['Total Energy (kWh)']) / baseline_kwh) * 100.0
    df_metrics['Cost Savings ($)'] = baseline_cost - df_metrics['Total Cost ($)']
    df_metrics['CO2 Avoided (kg)'] = baseline_co2 - df_metrics['CO2 Emissions (kg)']
    
    return df_metrics
