import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import os

def calculate_mape(y_true, y_pred):
    """Computes Mean Absolute Percentage Error (MAPE)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero by replacing 0 with a very small number or masking
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_predictions(predictions, y_true):
    """Computes all key academic regression metrics for each model."""
    results = []
    for model_name, y_pred in predictions.items():
        # Ensure non-negative predictions
        y_pred_clipped = np.clip(y_pred, 0, None)
        
        mae = mean_absolute_error(y_true, y_pred_clipped)
        rmse = root_mean_squared_error(y_true, y_pred_clipped)
        mape = calculate_mape(y_true, y_pred_clipped)
        r2 = r2_score(y_true, y_pred_clipped)
        
        results.append({
            'Model': model_name,
            'MAE': round(float(mae), 4),
            'RMSE': round(float(rmse), 4),
            'MAPE_pct': round(float(mape), 4),
            'R2': round(float(r2), 4)
        })
        
    return pd.DataFrame(results)

def log_experiment_results(metrics_df, target_name, log_file="data/28_Model_Experiment_Log.xlsx"):
    """Logs the model benchmark results to Excel for experiment tracking."""
    if not os.path.exists(log_file):
        # Create empty log df with standard headers
        log_df = pd.DataFrame(columns=[
            'Experiment_ID', 'Target', 'Model', 'MAE', 'RMSE', 'MAPE_pct', 'R2', 'Timestamp'
        ])
    else:
        try:
            # Let's find the correct row header. If it's a generated summary or has empty rows, we load it safely.
            from prediction.graph_builder import clean_load_excel
            log_df = clean_load_excel(log_file)
        except Exception:
            log_df = pd.DataFrame(columns=[
                'Experiment_ID', 'Target', 'Model', 'MAE', 'RMSE', 'MAPE_pct', 'R2', 'Timestamp'
            ])
            
    # Append new results
    new_rows = []
    run_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    exp_start_id = len(log_df) + 1
    
    for idx, row in metrics_df.iterrows():
        new_rows.append({
            'Experiment_ID': f"EXP_{exp_start_id + idx:03d}",
            'Target': target_name,
            'Model': row['Model'],
            'MAE': row['MAE'],
            'RMSE': row['RMSE'],
            'MAPE_pct': row['MAPE_pct'],
            'R2': row['R2'],
            'Timestamp': run_timestamp
        })
        
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([log_df, df_new], ignore_index=True)
    
    try:
        df_combined.to_excel(log_file, index=False)
        print(f"Logged {len(new_rows)} model results to {log_file}")
    except Exception as e:
        print(f"Warning: Could not log results to Excel: {e}")
        
    return df_combined
