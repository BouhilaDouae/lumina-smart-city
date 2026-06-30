import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from optimization.simulator import run_dimming_simulation
from optimization.evaluate import evaluate_dimming_policies
from prediction.graph_builder import clean_load_excel

def run_pipeline_and_save_results(data_dir="data", output_dir="scratch/plots"):
    print("=" * 60)
    print("      LUMINA SMART GRID - ADAPTIVE LIGHTING OPTIMIZATION ENGINE")
    print("=" * 60)
    
    # 1. Run the simulation
    df_sim = run_dimming_simulation(data_dir)
    
    # 2. Evaluate performance
    print("\n[Step 2] Evaluating policy benchmarks...")
    metrics_df = evaluate_dimming_policies(df_sim)
    
    print("\nBenchmark Evaluation Summary:")
    print(metrics_df.to_markdown(index=False))
    
    # Save the summary to a log file
    os.makedirs("scratch", exist_ok=True)
    metrics_df.to_excel("scratch/Optimization_Policy_Benchmark.xlsx", index=False)
    print("Saved metrics summary to scratch/Optimization_Policy_Benchmark.xlsx")
    
    # 3. Generate visualization comparison plots
    print("\n[Step 3] Generating comparative charts...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply premium plot style matching app theme
    plt.rcParams['figure.facecolor'] = '#060913'
    plt.rcParams['axes.facecolor'] = '#0c152d'
    plt.rcParams['text.color'] = '#ffffff'
    plt.rcParams['axes.labelcolor'] = '#8b9bb4'
    plt.rcParams['xtick.color'] = '#8b9bb4'
    plt.rcParams['ytick.color'] = '#8b9bb4'
    plt.rcParams['grid.color'] = '#1a2c5a'
    plt.rcParams['grid.alpha'] = 0.4
    plt.rcParams['axes.edgecolor'] = '#1a2c5a'
    
    # Subplots for energy consumption and safety violations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Energy chart
    sns.barplot(
        ax=axes[0],
        x='Policy',
        y='Total Energy (kWh)',
        data=metrics_df,
        hue='Policy',
        palette='mako',
        legend=False
    )
    axes[0].set_title('Total Energy Consumption (kWh) - 30 Days', fontweight='bold', fontsize=12, color='#ffffff')
    axes[0].set_ylabel('Energy (kWh)', color='#8b9bb4')
    axes[0].set_xlabel('', color='#8b9bb4')
    axes[0].grid(True, linestyle='--', alpha=0.3, color='#1a2c5a')
    for p in axes[0].patches:
        h = p.get_height()
        axes[0].annotate(f"{h:.1f} kWh", (p.get_x() + p.get_width() / 2., h * 0.85),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', color='white', fontweight='bold')
        
    # Safety Violation Rate chart
    sns.barplot(
        ax=axes[1],
        x='Policy',
        y='Safety Violation Rate (%)',
        data=metrics_df,
        hue='Policy',
        palette='rocket',
        legend=False
    )
    axes[1].set_title('Safety Violation Rate (%)', fontweight='bold', fontsize=12, color='#ffffff')
    axes[1].set_ylabel('Violation Rate (%)', color='#8b9bb4')
    axes[1].set_xlabel('', color='#8b9bb4')
    axes[1].grid(True, linestyle='--', alpha=0.3, color='#1a2c5a')
    for p in axes[1].patches:
        h = p.get_height()
        # Fallback if height is 0
        y_pos = max(h * 0.5, 0.1)
        axes[1].annotate(f"{h:.2f}%", (p.get_x() + p.get_width() / 2., y_pos),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', color='white', fontweight='bold')
        
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "dimming_policy_comparison.png")
    plt.savefig(plot_path, dpi=300, facecolor='#060913', bbox_inches='tight')
    plt.close()
    print(f"Saved comparative chart to {plot_path}")
    
    # 4. Update the Daily KPI Excel Sheet in-place
    print("\n[Step 4] Integrating simulated metrics into Daily KPI Summary...")
    kpi_file = os.path.join(data_dir, "23_Daily_KPI_Summary.xlsx")
    
    # Load raw excel to preserve header formatting structure
    excel_raw = pd.read_excel(kpi_file)
    
    # Aggregate simulator results by Date/Day
    df_sim['Date_Str'] = df_sim['Timestamp'].dt.strftime('%Y-%m-%d')
    
    # Calculate daily parameters for AIC and Baseline
    daily_base = df_sim.groupby('Date_Str')['kWh_Base'].sum().to_dict()
    daily_aic = df_sim.groupby('Date_Str')['kWh_AIC'].sum().to_dict()
    daily_brightness = df_sim.groupby('Date_Str')['Brightness_AIC'].mean().to_dict()
    
    # Row index 2 contains the actual headers
    # Data is in rows 3 onwards (0-indexed in pandas row 3 corresponds to pandas index 2)
    updated_count = 0
    for idx, row in excel_raw.iterrows():
        if idx < 2:
            continue
        date_val = str(row.iloc[0]).strip().split(" ")[0]  # Get Date part
        if date_val in daily_aic:
            kwh_base = daily_base[date_val]
            kwh_aic = daily_aic[date_val]
            saved_kwh = kwh_base - kwh_aic
            saved_pct = (saved_kwh / kwh_base) * 100.0 if kwh_base > 0 else 0.0
            avg_b = daily_brightness[date_val]
            co2_saved = saved_kwh * 0.533 # 533g/kWh = 0.533kg/kWh
            
            # Update cells:
            # Column indices:
            # 'Total_Energy_kWh' -> Index 3
            # 'Energy_Saved_vs_Baseline_kWh' -> Index 4
            # 'Savings_pct' -> Index 5
            # 'Avg_Brightness_pct' -> Index 6
            # 'CO2_Saved_kg' -> Index 16
            excel_raw.iloc[idx, 3] = round(kwh_aic, 2)
            excel_raw.iloc[idx, 4] = round(saved_kwh, 2)
            excel_raw.iloc[idx, 5] = round(saved_pct, 2)
            excel_raw.iloc[idx, 6] = round(avg_b, 1)
            excel_raw.iloc[idx, 16] = round(co2_saved, 2)
            updated_count += 1
            
    # Save the updated sheet back
    excel_raw.to_excel(kpi_file, index=False)
    print(f"Successfully updated {updated_count} daily rows in {kpi_file} with AI dimming performance logs!")
    print("\n" + "=" * 60)
    print("SIMULATION & INTEGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline_and_save_results()
