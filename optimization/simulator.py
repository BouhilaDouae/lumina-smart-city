import os
import sys
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from prediction.data_loader import load_traffic_data, load_weather_data
from prediction.graph_builder import build_graphs, clean_load_excel
from prediction.feature_engineering import build_spatio_temporal_features
from optimization.dimming_policies import static_baseline_policy, rule_based_control_policy, adaptive_ai_control_policy

def get_spatio_temporal_features():
    return [
        'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
        'Is_Weekend', 'Is_Rush', 'Is_Night', 'Ramadan_Effect',
        'Ped_Lag_1', 'Veh_Lag_1', 'Ped_Lag_2', 'Veh_Lag_2',
        'Ped_Lag_3', 'Veh_Lag_3', 'Ped_Lag_48', 'Veh_Lag_48',
        'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality',
        'Spatial_Lag_Ped', 'Spatial_Lag_Veh'
    ]

def get_hourly_aligned_data(data_dir="data"):
    """
    Loads traffic, weather, and lamp metadata and aligns them to an hourly grid
    across the 30-day period.
    """
    print("Loading datasets for simulation...")
    # Load raw datasets
    df_traffic_raw = load_traffic_data(data_dir)
    df_weather = load_weather_data(data_dir)
    
    # Load physical graphs to extract centralities
    road_g, _, _ = build_graphs(data_dir)
    
    # Engineering spatio-temporal features
    print("Engineering spatio-temporal features...")
    df_features = build_spatio_temporal_features(df_traffic_raw, road_g)
    
    # Aggregate 30-min data to hourly to match weather & energy logs
    print("Aggregating traffic data to hourly intervals...")
    df_features['Hour_Timestamp'] = df_features['Timestamp'].dt.floor('h')
    
    # Define aggregation rules
    agg_rules = {
        'Target_Pedestrians': 'sum',
        'Target_Vehicles': 'sum',
        'Is_Weekend': 'first',
        'Is_Rush': 'max',
        'Is_Night': 'first',
        'Ramadan_Effect': 'first',
        'degree_centrality': 'first',
        'closeness_centrality': 'first',
        'betweenness_centrality': 'first',
        'eigenvector_centrality': 'first',
        'Spatial_Lag_Ped': 'mean',
        'Spatial_Lag_Veh': 'mean'
    }
    
    # Add lag features
    for col in get_spatio_temporal_features():
        if col not in agg_rules and col in df_features.columns:
            agg_rules[col] = 'mean'
            
    df_hourly = df_features.groupby(['Node_ID', 'Hour_Timestamp']).agg(agg_rules).reset_index()
    df_hourly = df_hourly.rename(columns={'Hour_Timestamp': 'Timestamp'})
    
    # Merge weather data
    df_hourly = pd.merge(df_hourly, df_weather, on='Timestamp', how='inner')
    
    return df_hourly

def generate_out_of_fold_predictions(df_hourly):
    """
    Generates realistic predictions for all 30 days using K-Fold cross-validation
    so that predictions reflect real-world model accuracy and errors.
    """
    print("Generating out-of-fold predictions using XGBoost...")
    features = get_spatio_temporal_features()
    
    # Add time elements needed by HA/Ridge models if any
    df_hourly['Hour'] = df_hourly['Timestamp'].dt.hour
    df_hourly['Day_of_Week'] = df_hourly['Timestamp'].dt.dayofweek
    
    # 5-fold cross-validation based on days to prevent temporal leakage
    df_hourly['Day'] = df_hourly['Timestamp'].dt.day
    days = sorted(df_hourly['Day'].unique())
    num_folds = 5
    fold_size = len(days) // num_folds
    
    df_hourly['Pred_Pedestrians'] = 0.0
    df_hourly['Pred_Vehicles'] = 0.0
    
    for fold in range(num_folds):
        val_days = days[fold*fold_size : (fold+1)*fold_size] if fold < num_folds - 1 else days[fold*fold_size:]
        train_df = df_hourly[~df_hourly['Day'].isin(val_days)]
        val_df = df_hourly[df_hourly['Day'].isin(val_days)]
        
        # Train and predict Pedestrians
        xgb_ped = XGBRegressor(n_estimators=80, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb_ped.fit(train_df[features], train_df['Target_Pedestrians'])
        df_hourly.loc[df_hourly['Day'].isin(val_days), 'Pred_Pedestrians'] = np.clip(xgb_ped.predict(val_df[features]), 0, None)
        
        # Train and predict Vehicles
        xgb_veh = XGBRegressor(n_estimators=80, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb_veh.fit(train_df[features], train_df['Target_Vehicles'])
        df_hourly.loc[df_hourly['Day'].isin(val_days), 'Pred_Vehicles'] = np.clip(xgb_veh.predict(val_df[features]), 0, None)
        
    return df_hourly

def run_dimming_simulation(data_dir="data"):
    """
    Simulates power draw, energy consumption, and safety parameters for all 80 lamps
    across the 30-day period under the three control policies.
    """
    df_hourly = get_hourly_aligned_data(data_dir)
    df_hourly = generate_out_of_fold_predictions(df_hourly)
    
    # Load lamp predictive maintenance metadata to map Lamp_ID to Node_ID
    pm_file = os.path.join(data_dir, "25_Predictive_Maintenance.xlsx")
    df_pm = clean_load_excel(pm_file)
    
    # Map Lamp_ID to Node_ID and max power
    lamp_metadata = df_pm[['Lamp_ID', 'Node_ID', 'Sector']].copy()
    lamp_metadata['Max_Power_W'] = 150.0  # standard municipal LED lamp max power
    
    # We will simulate each lamp
    records = []
    
    print("Running simulation loop over all lamps and timestamps...")
    # Group hourly traffic and weather by Node_ID for fast lookup
    node_groups = {node: group.sort_values(by='Timestamp').set_index('Timestamp') 
                   for node, group in df_hourly.groupby('Node_ID')}
    
    # Standby coefficient alpha
    alpha = 0.1
    
    # Loop over lamps
    for idx, lamp in lamp_metadata.iterrows():
        lamp_id = lamp['Lamp_ID']
        node_id = lamp['Node_ID']
        sector = lamp['Sector']
        p_max = lamp['Max_Power_W']
        
        if node_id not in node_groups:
            continue
            
        node_data = node_groups[node_id]
        
        for ts, row in node_data.iterrows():
            # Get actual counts
            actual_ped = row['Target_Pedestrians']
            actual_veh = row['Target_Vehicles']
            
            # Get predicted counts
            pred_ped = row['Pred_Pedestrians']
            pred_veh = row['Pred_Vehicles']
            
            # 1. Static Baseline Policy (Constant 100%)
            b_base = static_baseline_policy(row)
            
            # 2. Rule-Based Control (RBC) Policy (Uses actual traffic for perfect foresight baseline)
            b_rbc = rule_based_control_policy(row, ped_col='Target_Pedestrians', veh_col='Target_Vehicles')
            
            # 3. Adaptive AI Control (AIC) Policy (Uses out-of-fold predicted traffic)
            b_aic = adaptive_ai_control_policy(row, pred_ped, pred_veh)
            
            # Power calculation function
            def calc_power(b):
                return p_max * (alpha + (1.0 - alpha) * (b / 100.0))
                
            p_base = calc_power(b_base)
            p_rbc = calc_power(b_rbc)
            p_aic = calc_power(b_aic)
            
            # Safety Violations Check:
            # A safety violation occurs if actual traffic is high (ped > 30 or veh > 15) or weather boost is required
            # but brightness is set below the safe limit (100% for high traffic / fog).
            is_high_traffic = (actual_ped > 30 or actual_veh > 15)
            is_bad_weather = row.get('Recommended_Brightness_Boost_pct', 0.0) > 0.0
            
            violation_base = 0
            violation_rbc = 1 if (is_high_traffic or is_bad_weather) and b_rbc < 80.0 else 0
            # For AI, if predictions were too low resulting in low brightness when actual was high
            violation_aic = 1 if (is_high_traffic or is_bad_weather) and b_aic < 80.0 else 0
            
            records.append({
                'Timestamp': ts,
                'Lamp_ID': lamp_id,
                'Node_ID': node_id,
                'Sector': sector,
                'Actual_Pedestrians': actual_ped,
                'Actual_Vehicles': actual_veh,
                'Pred_Pedestrians': pred_ped,
                'Pred_Vehicles': pred_veh,
                # Brightness %
                'Brightness_Base': b_base,
                'Brightness_RBC': b_rbc,
                'Brightness_AIC': b_aic,
                # Power Draw (W)
                'Power_Base': p_base,
                'Power_RBC': p_rbc,
                'Power_AIC': p_aic,
                # Energy consumed in 1 hour (kWh)
                'kWh_Base': p_base / 1000.0,
                'kWh_RBC': p_rbc / 1000.0,
                'kWh_AIC': p_aic / 1000.0,
                # Safety violations
                'Violation_Base': violation_base,
                'Violation_RBC': violation_rbc,
                'Violation_AIC': violation_aic
            })
            
    df_sim = pd.DataFrame(records)
    print(f"Simulation completed! Generated {df_sim.shape[0]} simulated lamp-hour rows.")
    return df_sim

if __name__ == "__main__":
    df_sim = run_dimming_simulation()
    print("Sample simulation results:")
    print(df_sim.head(3))
