import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

def plot_performance_metrics(metrics_df, target_name, output_dir="scratch/plots"):
    """Generates comparative bar charts for MAE and RMSE across all 5 models."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sort models by MAE (lower is better)
    df_sorted = metrics_df.sort_values(by='MAE').reset_index(drop=True)
    
    # Palette
    colors = sns.color_palette("viridis", len(df_sorted))
    
    # MAE plot
    sns.barplot(ax=axes[0], x='MAE', y='Model', data=df_sorted, hue='Model', palette=colors, legend=False)
    axes[0].set_title(f'Mean Absolute Error (MAE) - {target_name}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('MAE (Lower is Better)')
    axes[0].set_ylabel('')
    
    # Add values on bars
    for i, v in enumerate(df_sorted['MAE']):
        axes[0].text(v + (v * 0.01), i, f"{v:.2f}", va='center', fontweight='semibold')
        
    # RMSE plot
    df_sorted_rmse = metrics_df.sort_values(by='RMSE').reset_index(drop=True)
    sns.barplot(ax=axes[1], x='RMSE', y='Model', data=df_sorted_rmse, hue='Model', palette=colors, legend=False)
    axes[1].set_title(f'Root Mean Squared Error (RMSE) - {target_name}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('RMSE (Lower is Better)')
    axes[1].set_ylabel('')
    
    # Add values on bars
    for i, v in enumerate(df_sorted_rmse['RMSE']):
        axes[1].text(v + (v * 0.01), i, f"{v:.2f}", va='center', fontweight='semibold')
        
    plt.suptitle(f'Model Comparison Benchmark - {target_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    file_name = f"model_comparison_{target_name.lower()}.png"
    plt.savefig(os.path.join(output_dir, file_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparative metrics plot to {output_dir}/{file_name}")

def plot_predictions_vs_actual(df_test, predictions, target_name, node_id, output_dir="scratch/plots", num_slots=96):
    """Plots the actual vs predicted values for a specific node over a given time window."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter test set for the target node
    node_df = df_test[df_test['Node_ID'] == node_id].copy()
    node_df = node_df.sort_values(by='Timestamp').head(num_slots)
    
    if len(node_df) == 0:
        print(f"Warning: No test records found for {node_id}")
        return
        
    plt.figure(figsize=(14, 7))
    
    # Plot actual
    plt.plot(node_df['Timestamp'], node_df[target_name], label='Actual', color='black', linewidth=2.5, zorder=10)
    
    # Plot predictions
    for model_name, preds in predictions.items():
        # Align predictions with the filtered test rows
        # The predictions array corresponds to the entire test set. Let's filter by the same indices.
        node_indices = node_df.index
        # Predictions list in predictions has the same length as df_test. Let's map it via row index.
        node_preds = np.clip(preds[node_indices - df_test.index[0]], 0, None)
        
        plt.plot(node_df['Timestamp'], node_preds, label=model_name, alpha=0.8, linestyle='--')
        
    plt.title(f'2-Day Prediction Forecast vs Actual - Node: {node_id} ({target_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Timestamp)')
    plt.ylabel(f'{target_name} Count')
    plt.legend(frameon=True, facecolor='white', edgecolor='lightgray')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    file_name = f"forecast_vs_actual_{target_name.lower()}_{node_id}.png"
    plt.savefig(os.path.join(output_dir, file_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction vs actual forecast plot to {output_dir}/{file_name}")
