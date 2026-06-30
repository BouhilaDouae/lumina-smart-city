import pandas as pd
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from prediction.data_loader import load_traffic_data
from prediction.graph_builder import build_graphs, get_pyg_graph
from prediction.feature_engineering import build_spatio_temporal_features
from prediction.models import train_and_predict_all, TemporalGNN
from prediction.evaluate import evaluate_predictions, log_experiment_results
from prediction.visualize import plot_performance_metrics, plot_predictions_vs_actual

def train_and_predict_gnn(df_train, df_test, road_g, target_col, epochs=60, lr=0.01):
    """
    Constructs spatio-temporal sequences, trains the TemporalGNN model,
    saves the weights and scaler, and returns predictions aligned with df_test.
    """
    # 1. Get PyG Graph Structure
    edge_index, edge_weight, nodes = get_pyg_graph(road_g)
    num_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # 2. Identify GNN features
    if 'Pedestrians' in target_col:
        feature_cols = [
            'Ped_Lag_1', 'Ped_Lag_2', 'Ped_Lag_3', 'Ped_Lag_4',
            'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
            'Is_Weekend', 'Is_Rush', 'Is_Night', 'Ramadan_Effect',
            'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality'
        ]
    else:
        feature_cols = [
            'Veh_Lag_1', 'Veh_Lag_2', 'Veh_Lag_3', 'Veh_Lag_4',
            'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
            'Is_Weekend', 'Is_Rush', 'Is_Night', 'Ramadan_Effect',
            'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality'
        ]
        
    num_features = len(feature_cols)
    
    # 3. Scale Features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)
    
    all_df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    all_timestamps = sorted(all_df['Timestamp'].unique())
    
    # Map features and targets for each (Timestamp, Node_ID)
    feature_map = {}
    target_map = {}
    
    scaled_features_all = scaler.transform(all_df[feature_cols].values)
    targets_all = all_df[target_col].values
    
    for idx, row in all_df.iterrows():
        ts = row['Timestamp']
        node = row['Node_ID']
        feature_map[(ts, node)] = scaled_features_all[idx]
        target_map[(ts, node)] = targets_all[idx]
        
    # 4. Build Spatio-Temporal Graph Sequences
    look_back = 4
    X_seq = []
    y_target = []
    seq_timestamps = []
    
    for i in range(look_back, len(all_timestamps)):
        target_ts = all_timestamps[i]
        
        # Build look-back sequence of matrices
        seq_features = []
        for l in range(look_back):
            ts_step = all_timestamps[i - look_back + l]
            step_matrix = []
            for node in nodes:
                step_matrix.append(feature_map.get((ts_step, node), np.zeros(num_features)))
            seq_features.append(step_matrix)
            
        target_vals = []
        for node in nodes:
            target_vals.append(target_map.get((target_ts, node), 0.0))
            
        X_seq.append(seq_features)
        y_target.append(target_vals)
        seq_timestamps.append(target_ts)
        
    X_seq = np.array(X_seq)  # [num_samples, look_back, num_nodes, num_features]
    y_target = np.array(y_target)  # [num_samples, num_nodes]
    
    # Split chronologically
    train_indices = []
    test_indices = []
    
    train_timestamps_set = set(df_train['Timestamp'].unique())
    test_timestamps_set = set(df_test['Timestamp'].unique())
    
    for idx, ts in enumerate(seq_timestamps):
        if ts in train_timestamps_set:
            train_indices.append(idx)
        elif ts in test_timestamps_set:
            test_indices.append(idx)
            
    X_train_tensor = torch.tensor(X_seq[train_indices], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_target[train_indices], dtype=torch.float32)
    
    X_test_tensor = torch.tensor(X_seq[test_indices], dtype=torch.float32)
    test_ts_list = [seq_timestamps[i] for i in test_indices]
    
    # Train Loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 5. Training
    model = TemporalGNN(node_features=num_features, hidden_channels=32, num_layers=2, forecast_horizon=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_X, edge_index, edge_weight)  # [batch, num_nodes, 1]
            loss = criterion(out.squeeze(-1), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(train_loader.dataset)
        if (epoch + 1) % 15 == 0:
            print(f"   [GNN Epoch {epoch+1}/{epochs}] MSE Loss: {epoch_loss:.5f}")
            
    # 6. Predict
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_tensor, edge_index, edge_weight)  # [test_samples, num_nodes, 1]
        test_preds = test_out.squeeze(-1).numpy()
        
    gnn_results = {}
    for i, ts in enumerate(test_ts_list):
        for j, node in enumerate(nodes):
            gnn_results[(ts, node)] = test_preds[i, j]
            
    # Align to df_test
    gnn_preds = []
    for _, row in df_test.iterrows():
        ts = row['Timestamp']
        node = row['Node_ID']
        val = gnn_results.get((ts, node), row[target_col])
        gnn_preds.append(max(0.0, val))
        
    # 7. Save weights and scaling variables
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_name = "best_gnn_pedestrians.pt" if 'Pedestrians' in target_col else "best_gnn_vehicles.pt"
    torch.save(model.state_dict(), os.path.join(model_name))  # Save to current workspace root or models/
    # Save to models/ folder as well for dashboard access
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))
    
    scaler_name = "scaler_gnn_pedestrians.pkl" if 'Pedestrians' in target_col else "scaler_gnn_vehicles.pkl"
    import pickle
    with open(os.path.join(model_dir, scaler_name), 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"   Saved GNN model and scaler to {model_dir}/")
    return np.array(gnn_preds)

def run_benchmark():
    print("=" * 60)
    print("      LUMINA SMART GRID - AI PREDICTION PIPELINE BENCHMARK")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[Step 1] Loading raw datasets...")
    df_raw = load_traffic_data()
    road_g, _, _ = build_graphs()
    print(f"Loaded traffic dataset with shape: {df_raw.shape}")
    print(f"Road network graph consists of: {road_g.number_of_nodes()} nodes, {road_g.number_of_edges()} edges")
    
    # 2. Feature Engineering
    print("\n[Step 2] Engineering spatio-temporal and graph features...")
    df_features = build_spatio_temporal_features(df_raw, road_g)
    print(f"Feature matrix successfully constructed. Shape: {df_features.shape}")
    
    # 3. Train/Test Split (Chronological to prevent leakage)
    print("\n[Step 3] Splitting dataset chronologically...")
    unique_timestamps = sorted(df_features['Timestamp'].unique())
    split_idx = int(len(unique_timestamps) * 0.833) # 25 days out of 30
    split_time = unique_timestamps[split_idx]
    
    df_train = df_features[df_features['Timestamp'] < split_time].copy().reset_index(drop=True)
    df_test = df_features[df_features['Timestamp'] >= split_time].copy().reset_index(drop=True)
    
    print(f"Train period: {df_train['Timestamp'].min()} to {df_train['Timestamp'].max()} ({df_train.shape[0]} samples)")
    print(f"Test period: {df_test['Timestamp'].min()} to {df_test['Timestamp'].max()} ({df_test.shape[0]} samples)")
    
    targets = {
        'Pedestrians': 'Target_Pedestrians',
        'Vehicles': 'Target_Vehicles'
    }
    
    all_metrics = {}
    
    # 4. Model Benchmarking
    for label, target_col in targets.items():
        print("\n" + "-"*50)
        print(f" TRAINING BENCHMARK FOR TARGET: {label.upper()}")
        print("-"*50)
        
        # Train and Predict
        predictions, y_true = train_and_predict_all(df_train, df_test, target_col)
        
        # Train and Predict GNN
        print(f"Training and forecasting using Temporal GNN for {label}...")
        gnn_preds = train_and_predict_gnn(df_train, df_test, road_g, target_col)
        predictions['Temporal GNN (Spatio-Temporal)'] = gnn_preds
        
        # Evaluate
        metrics_df = evaluate_predictions(predictions, y_true)
        all_metrics[label] = metrics_df
        
        print(f"\nBenchmark Results for {label}:")
        print(metrics_df.to_markdown(index=False))
        
        # Log to Excel
        log_experiment_results(metrics_df, label)
        
        # 5. Visualizations
        print(f"\n[Step 5] Generating visualization plots for {label}...")
        plot_performance_metrics(metrics_df, label)
        
        # Plot predictions vs actual for top degree node (e.g., Node_001)
        plot_predictions_vs_actual(df_test, predictions, target_col, 'Node_001', num_slots=96) # 48 hours
        
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print("Generated charts and forecasts can be viewed in scratch/plots/")
    print("=" * 60)

def predict_gnn(df_train, df_test, road_g, target_col, fallback_epochs=30):
    """
    Returns GNN predictions for df_test, aligned row-by-row.
    - Loads pre-trained weights from models/ if they exist (fast inference).
    - Otherwise trains from scratch using fallback_epochs.
    """
    import pickle

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    model_name = "best_gnn_pedestrians.pt" if 'Pedestrians' in target_col else "best_gnn_vehicles.pt"
    scaler_name = "scaler_gnn_pedestrians.pkl" if 'Pedestrians' in target_col else "scaler_gnn_vehicles.pkl"
    model_path = os.path.join(model_dir, model_name)
    scaler_path = os.path.join(model_dir, scaler_name)

    if 'Pedestrians' in target_col:
        feature_cols = [
            'Ped_Lag_1', 'Ped_Lag_2', 'Ped_Lag_3', 'Ped_Lag_4',
            'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
            'Is_Weekend', 'Is_Rush', 'Is_Night', 'Ramadan_Effect',
            'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality'
        ]
    else:
        feature_cols = [
            'Veh_Lag_1', 'Veh_Lag_2', 'Veh_Lag_3', 'Veh_Lag_4',
            'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
            'Is_Weekend', 'Is_Rush', 'Is_Night', 'Ramadan_Effect',
            'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality'
        ]

    num_features = len(feature_cols)
    edge_index, edge_weight, nodes = get_pyg_graph(road_g)
    look_back = 4

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"   [GNN] Loading pre-trained weights from {model_path}...")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        model = TemporalGNN(node_features=num_features, hidden_channels=32, num_layers=2, forecast_horizon=1)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
    else:
        print(f"   [GNN] No pre-trained weights found. Training ({fallback_epochs} epochs)...")
        return train_and_predict_gnn(df_train, df_test, road_g, target_col, epochs=fallback_epochs)

    # Build spatio-temporal sequences for inference (using saved scaler)
    all_df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    all_timestamps = sorted(all_df['Timestamp'].unique())

    scaled_features_all = scaler.transform(all_df[feature_cols].values)
    targets_all = all_df[target_col].values

    feature_map = {}
    target_map = {}
    for idx, row in all_df.iterrows():
        ts = row['Timestamp']
        node = row['Node_ID']
        feature_map[(ts, node)] = scaled_features_all[idx]
        target_map[(ts, node)] = targets_all[idx]

    X_seq, y_target_seq, seq_timestamps = [], [], []
    for i in range(look_back, len(all_timestamps)):
        target_ts = all_timestamps[i]
        seq_features = []
        for l in range(look_back):
            ts_step = all_timestamps[i - look_back + l]
            step_matrix = [feature_map.get((ts_step, node), np.zeros(num_features)) for node in nodes]
            seq_features.append(step_matrix)
        target_vals = [target_map.get((target_ts, node), 0.0) for node in nodes]
        X_seq.append(seq_features)
        y_target_seq.append(target_vals)
        seq_timestamps.append(target_ts)

    X_seq = np.array(X_seq)
    test_timestamps_set = set(df_test['Timestamp'].unique())
    test_indices = [i for i, ts in enumerate(seq_timestamps) if ts in test_timestamps_set]
    test_ts_list = [seq_timestamps[i] for i in test_indices]

    X_test_tensor = torch.tensor(X_seq[test_indices], dtype=torch.float32)
    with torch.no_grad():
        test_out = model(X_test_tensor, edge_index, edge_weight)
        test_preds = test_out.squeeze(-1).numpy()

    gnn_results = {}
    for i, ts in enumerate(test_ts_list):
        for j, node in enumerate(nodes):
            gnn_results[(ts, node)] = test_preds[i, j]

    gnn_preds = []
    for _, row in df_test.iterrows():
        val = gnn_results.get((row['Timestamp'], row['Node_ID']), row[target_col])
        gnn_preds.append(max(0.0, val))

    return np.array(gnn_preds)


if __name__ == "__main__":
    run_benchmark()
