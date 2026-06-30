import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from prediction.data_loader import load_traffic_data, load_weather_data
from prediction.graph_builder import build_graphs
from prediction.feature_engineering import build_spatio_temporal_features
from prediction.stgcn_model import train_stgcn_model, load_and_predict_stgcn
from prediction.energy_optimizer import run_energy_optimization

def run_system_evaluation(data_dir="data", output_dir="scratch", force_retrain=False):
    print("=" * 60)
    # Ensure stdout is UTF-8 to avoid console errors
    sys.stdout.reconfigure(encoding='utf-8')
    print("      LUMINA SMART GRID - SYSTEM-WIDE POLICY EVALUATION ENGINE")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[Step 1] Loading raw datasets...")
    df_raw = load_traffic_data(data_dir)
    df_weather = load_weather_data(data_dir)
    road_g, _, _ = build_graphs(data_dir)
    
    # 2. Feature Engineering
    print("\n[Step 2] Engineering spatio-temporal and graph features...")
    df_features = build_spatio_temporal_features(df_raw, road_g)
    
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
    lags = [
        'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
        'Ped_Lag_1', 'Veh_Lag_1', 'Ped_Lag_2', 'Veh_Lag_2',
        'Ped_Lag_3', 'Veh_Lag_3', 'Ped_Lag_48', 'Veh_Lag_48'
    ]
    for col in lags:
        if col in df_features.columns:
            agg_rules[col] = 'mean'
            
    df_hourly = df_features.groupby(['Node_ID', 'Hour_Timestamp']).agg(agg_rules).reset_index()
    df_hourly = df_hourly.rename(columns={'Hour_Timestamp': 'Timestamp'})
    
    # Merge weather data
    df_hourly = pd.merge(df_hourly, df_weather, on='Timestamp', how='inner')
    
    # 3. Train/Test Split (83% Train, 17% Test)
    # 5 months train, 1 month test
    print("\n[Step 3] Splitting dataset chronologically...")
    unique_timestamps = sorted(df_hourly['Timestamp'].unique())
    split_idx = int(len(unique_timestamps) * 0.833)
    split_time = unique_timestamps[split_idx]
    
    df_train = df_hourly[df_hourly['Timestamp'] < split_time].copy().reset_index(drop=True)
    df_test = df_hourly[df_hourly['Timestamp'] >= split_time].copy().reset_index(drop=True)
    
    print(f"Train period: {df_train['Timestamp'].min()} to {df_train['Timestamp'].max()} ({df_train.shape[0]} samples)")
    print(f"Test period: {df_test['Timestamp'].min()} to {df_test['Timestamp'].max()} ({df_test.shape[0]} samples)")
    
    # 4. Generate Predictions using STGCN
    print("\n[Step 4] Training STGCN models...")
    if force_retrain:
        pred_peds = train_stgcn_model(df_train, df_test, 'Target_Pedestrians', epochs=15)
        pred_vehs = train_stgcn_model(df_train, df_test, 'Target_Vehicles', epochs=15)
    else:
        pred_peds = load_and_predict_stgcn(df_train, df_test, 'Target_Pedestrians')
        pred_vehs = load_and_predict_stgcn(df_train, df_test, 'Target_Vehicles')
        
    df_test['Pred_Pedestrians'] = pred_peds
    df_test['Pred_Vehicles'] = pred_vehs
    
    # Calculate GNN Accuracy Metrics on test set
    ped_mae = np.mean(np.abs(df_test['Target_Pedestrians'] - df_test['Pred_Pedestrians']))
    ped_rmse = np.sqrt(np.mean((df_test['Target_Pedestrians'] - df_test['Pred_Pedestrians'])**2))
    ped_r2 = 1 - (np.sum((df_test['Target_Pedestrians'] - df_test['Pred_Pedestrians'])**2) / np.sum((df_test['Target_Pedestrians'] - df_test['Target_Pedestrians'].mean())**2))
    
    veh_mae = np.mean(np.abs(df_test['Target_Vehicles'] - df_test['Pred_Vehicles']))
    veh_rmse = np.sqrt(np.mean((df_test['Target_Vehicles'] - df_test['Pred_Vehicles'])**2))
    veh_r2 = 1 - (np.sum((df_test['Target_Vehicles'] - df_test['Pred_Vehicles'])**2) / np.sum((df_test['Target_Vehicles'] - df_test['Target_Vehicles'].mean())**2))
    
    print("\nSTGCN Model Accuracy (Test Set):")
    print(f"Pedestrians: MAE={ped_mae:.4f}, RMSE={ped_rmse:.4f}, R2={ped_r2:.4f}")
    print(f"Vehicles:    MAE={veh_mae:.4f}, RMSE={veh_rmse:.4f}, R2={veh_r2:.4f}")
    
    # Save STGCN metrics
    gnn_metrics_df = pd.DataFrame([
        {'Target': 'Pedestrians', 'MAE': ped_mae, 'RMSE': ped_rmse, 'R2': ped_r2},
        {'Target': 'Vehicles', 'MAE': veh_mae, 'RMSE': veh_rmse, 'R2': veh_r2}
    ])
    gnn_metrics_df.to_excel(os.path.join(output_dir, "STGCN_Accuracy_Metrics.xlsx"), index=False)
    
    # 5. Run Policy Energy Simulator
    print("\n[Step 5] Running Policy Energy Optimization Simulator...")
    summary_df, detailed_df = run_energy_optimization(df_test)
    
    # Display comparison
    print("\nSystem Policy Comparison Summary:")
    print(summary_df.to_markdown(index=False))
    
    # 6. Export Reports
    print(f"\n[Step 6] Exporting evaluation reports to {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV Export
    csv_path = os.path.join(output_dir, "System_Evaluation_Metrics.csv")
    summary_df.to_csv(csv_path, index=False)
    
    # Excel Report
    xlsx_path = os.path.join(output_dir, "System_Evaluation_Report.xlsx")
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name="Policy_Comparison", index=False)
        detailed_df.head(1000).to_excel(writer, sheet_name="Hourly_Samples", index=False)
        gnn_metrics_df.to_excel(writer, sheet_name="GNN_Accuracy", index=False)
        
    print(f"Saved CSV to {csv_path}")
    print(f"Saved Excel Report to {xlsx_path}")
    
    # 7. Generate Plotly Comparison Charts
    print("\n[Step 7] Exporting Plotly comparison charts...")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Subplots
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=(
        '<b>Total Energy Consumption (kWh) - Test Period</b>',
        '<b>Safety Violation Rate (%)</b>'
    ))
    
    fig.add_trace(
        go.Bar(
            x=summary_df['Policy'],
            y=summary_df['Total Energy (kWh)'],
            marker=dict(color=['#e11d48', '#d97706', '#059669']),
            text=[f"{v:,.1f} kWh" for v in summary_df['Total Energy (kWh)']],
            textposition='auto',
            name='Energy'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=summary_df['Policy'],
            y=summary_df['Safety Violation Rate (%)'],
            marker=dict(color=['#0f172a', '#ea580c', '#e11d48']),
            text=[f"{v:.3f}%" for v in summary_df['Safety Violation Rate (%)']],
            textposition='auto',
            name='Violations'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(text='<b>Lumina Smart Grid - Municipal Dimming Policy Benchmark</b>', font=dict(size=16)),
        template='plotly_dark',
        paper_bgcolor='#060913',
        plot_bgcolor='rgba(12,21,45,0.4)',
        showlegend=False,
        height=500
    )
    
    fig.update_yaxes(showgrid=True, gridcolor='#1a2c5a')
    
    html_path = os.path.join(plots_dir, "policy_comparison_plotly.html")
    fig.write_html(html_path)
    print(f"Saved Plotly HTML chart to {html_path}")

    png_path = os.path.join(plots_dir, "policy_comparison_plotly.png")
    try:
        fig.write_image(png_path)
        print(f"Saved Plotly PNG chart to {png_path}")
    except Exception as img_err:
        print(f"Warning: could not export PNG chart ({img_err}); ensure kaleido is installed.")
    
    # 8. Update Daily KPI Excel Sheet in-place
    kpi_file = os.path.join(data_dir, "23_Daily_KPI_Summary.xlsx")
    if os.path.exists(kpi_file):
        try:
            print("\n[Step 8] Integrating simulated metrics into Daily KPI Summary...")
            excel_raw = pd.read_excel(kpi_file)
            
            # Identify columns in row 2 (which is index 2)
            cols = excel_raw.iloc[2].tolist()
            
            # Map daily data
            detailed_df['Date_Str'] = detailed_df['Timestamp'].dt.strftime('%Y-%m-%d')
            daily_base = detailed_df.groupby('Date_Str')['kWh_Fixed'].sum().to_dict()
            daily_adaptive = detailed_df.groupby('Date_Str')['kWh_Adaptive'].sum().to_dict()
            daily_brightness = detailed_df.groupby('Date_Str')['Brightness_Adaptive'].mean().to_dict()
            
            updated_count = 0
            # Data starts at row 3 (index 3)
            for idx in range(3, len(excel_raw)):
                date_val = str(excel_raw.iloc[idx, 0]).strip().split(" ")[0] # Date column is index 0
                if date_val in daily_adaptive:
                    kwh_fixed = daily_base[date_val]
                    kwh_adaptive = daily_adaptive[date_val]
                    saved_kwh = kwh_fixed - kwh_adaptive
                    saved_pct = (saved_kwh / kwh_fixed) * 100.0 if kwh_fixed > 0 else 0.0
                    avg_b = daily_brightness[date_val]
                    co2_saved = saved_kwh * 0.533
                    
                    # Columns to update:
                    # 'Total_Energy_kWh' -> Index 3
                    # 'Energy_Saved_vs_Baseline_kWh' -> Index 4
                    # 'Savings_pct' -> Index 5
                    # 'Avg_Brightness_pct' -> Index 6
                    # 'CO2_Saved_kg' -> Index 16
                    # 'STGCN_MAE' -> Index 18
                    # 'STGCN_RMSE' -> Index 19
                    excel_raw.iloc[idx, 3] = round(kwh_adaptive, 2)
                    excel_raw.iloc[idx, 4] = round(saved_kwh, 2)
                    excel_raw.iloc[idx, 5] = round(saved_pct, 2)
                    excel_raw.iloc[idx, 6] = round(avg_b, 1)
                    excel_raw.iloc[idx, 16] = round(co2_saved, 2)
                    excel_raw.iloc[idx, 18] = round(ped_mae, 3)
                    excel_raw.iloc[idx, 19] = round(ped_rmse, 3)
                    updated_count += 1
                    
            excel_raw.to_excel(kpi_file, index=False)
            print(f"Successfully integrated {updated_count} days into {kpi_file}!")
        except Exception as kpi_err:
            print(f"Warning: Could not update KPI Excel: {kpi_err}")
            
    print("=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_system_evaluation(force_retrain=True)
