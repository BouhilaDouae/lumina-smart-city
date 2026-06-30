import os
import sys
import pickle
import numpy as np
import pandas as pd

TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    import torch_geometric.nn as pyg_nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    torch = None
    nn = None
    pyg_nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = e

from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from prediction.data_loader import load_traffic_data
from prediction.graph_builder import build_graphs, get_pyg_graph

class TorchUnavailableError(RuntimeError):
    pass


def ensure_torch_available():
    if not TORCH_AVAILABLE:
        print(f"WARNING: PyTorch is unavailable ({TORCH_IMPORT_ERROR}). Falling back to Sklearn Random Forest for STGCN Mock.")

if TORCH_AVAILABLE:
    class TemporalConv(nn.Module):
        """
        1D Temporal Convolutional Layer with Gated Linear Units (GLU).
        """
        def __init__(self, in_channels, out_channels, kernel_size=2):
            super(TemporalConv, self).__init__()
            self.kernel_size = kernel_size
            # GLU needs twice the output channels
            self.conv = nn.Conv1d(in_channels, 2 * out_channels, kernel_size=kernel_size)
            self.sigmoid = nn.Sigmoid()
else:
    class TemporalConv:
        def __init__(self, *args, **kwargs):
            ensure_torch_available()

    def forward(self, x):
        # Input shape: [batch_size, seq_len, num_nodes, in_channels]
        batch_size, seq_len, num_nodes, in_channels = x.shape
        
        # Reshape for 1D convolution: [batch_size * num_nodes, in_channels, seq_len]
        x = x.transpose(1, 2).reshape(batch_size * num_nodes, in_channels, seq_len)
        
        # Conv output shape: [batch_size * num_nodes, 2 * out_channels, seq_len - kernel_size + 1]
        out = self.conv(x)
        
        # Split channels for GLU gating: out = P * sigmoid(Q)
        half_channels = out.shape[1] // 2
        p = out[:, :half_channels, :]
        q = out[:, half_channels:, :]
        
        gated = p * self.sigmoid(q)
        
        # Reshape back to: [batch_size, seq_len - kernel_size + 1, num_nodes, out_channels]
        new_seq_len = gated.shape[2]
        out_channels = gated.shape[1]
        
        gated = gated.reshape(batch_size, num_nodes, out_channels, new_seq_len).transpose(2, 3).transpose(1, 2)
        return gated

if TORCH_AVAILABLE:
    class SpatialGCN(nn.Module):
        """
        Spatial Graph Convolutional Layer utilizing PyTorch Geometric GCNConv.
        Uses batched edge indices to compute GCNConv across the entire batch efficiently.
        """
        def __init__(self, in_channels, out_channels):
            super(SpatialGCN, self).__init__()
            self.gcn = pyg_nn.GCNConv(in_channels, out_channels)
            self.relu = nn.ReLU()

        def _batch_graph_structure(self, edge_index, num_graphs, num_nodes, edge_weight=None):
            """Replicates edge_index and edge_weight to batch disjoint graphs."""
            device = edge_index.device
            row, col = edge_index
            
            # Offsets for each graph in the batch
            offsets = torch.arange(num_graphs, device=device).unsqueeze(1) * num_nodes
            
            row_batched = (row.unsqueeze(0) + offsets).view(-1)
            col_batched = (col.unsqueeze(0) + offsets).view(-1)
            
            batched_edge_index = torch.stack([row_batched, col_batched], dim=0)
            
            if edge_weight is not None:
                batched_edge_weight = edge_weight.repeat(num_graphs)
                return batched_edge_index, batched_edge_weight
                
            return batched_edge_index, None

        def forward(self, x, edge_index, edge_weight=None):
            # Input shape: [batch_size, seq_len, num_nodes, in_channels]
            batch_size, seq_len, num_nodes, in_channels = x.shape
            
            # Replicate edge indices for batch_size * seq_len disjoint graphs
            num_graphs = batch_size * seq_len
            batched_edge_index, batched_edge_weight = self._batch_graph_structure(
                edge_index, num_graphs, num_nodes, edge_weight
            )
            
            # Flatten x to [batch_size * seq_len * num_nodes, in_channels]
            x_flat = x.reshape(num_graphs * num_nodes, in_channels)
            
            # Apply GCNConv
            out_flat = self.relu(self.gcn(x_flat, batched_edge_index, batched_edge_weight))
            
            # Reshape back to: [batch_size, seq_len, num_nodes, out_channels]
            out_channels = out_flat.shape[1]
            out = out_flat.reshape(batch_size, seq_len, num_nodes, out_channels)
            return out
else:
    class SpatialGCN:
        def __init__(self, *args, **kwargs):
            ensure_torch_available()

if TORCH_AVAILABLE:
    class STGCNBlock(nn.Module):
        """
        Spatio-Temporal Graph Convolutional Block (STGCN Sandwich Block).
        Consists of Temporal Gated Conv -> Spatial GCN -> Temporal Gated Conv.
        """
        def __init__(self, in_channels, spatial_channels, out_channels, kernel_size=2):
            super(STGCNBlock, self).__init__()
            self.tconv1 = TemporalConv(in_channels, spatial_channels, kernel_size=kernel_size)
            self.sconv = SpatialGCN(spatial_channels, spatial_channels)
            self.tconv2 = TemporalConv(spatial_channels, out_channels, kernel_size=kernel_size)
            self.norm = nn.LayerNorm(out_channels)

        def forward(self, x, edge_index, edge_weight=None):
            # x: [batch, seq_len, num_nodes, in_channels]
            h = self.tconv1(x)       # Reduces seq_len (e.g. 4 -> 3)
            h = self.sconv(h, edge_index, edge_weight) # Keeps seq_len (3)
            h = self.tconv2(h)       # Reduces seq_len (e.g. 3 -> 2)
            
            # Apply Layer Normalization along channels
            h = self.norm(h)
            return h

    class STGCN(nn.Module):
        """
        Complete Spatio-Temporal Graph Convolutional Network (STGCN).
        """
        def __init__(self, num_features, hidden_channels=32, kernel_size=2, forecast_horizon=1):
            super(STGCN, self).__init__()
            # STGCN Block reduces sequence length by 2 * (kernel_size - 1)
            # For sequence length 4 and kernel_size 2, final sequence length will be 4 - 2 * (2-1) = 2.
            self.stgcn_block = STGCNBlock(
                in_channels=num_features, 
                spatial_channels=hidden_channels, 
                out_channels=hidden_channels, 
                kernel_size=kernel_size
            )
            # Output layers: projects flattened temporal channels to forecast horizon
            self.lin = nn.Linear(hidden_channels * 2, forecast_horizon)

        def forward(self, x, edge_index, edge_weight=None):
            # x: [batch, seq_len, num_nodes, num_features]
            # edge_index: [2, num_edges]
            # edge_weight: [num_edges]
            h = self.stgcn_block(x, edge_index, edge_weight) # [batch, 2, num_nodes, hidden_channels]
            
            batch_size, seq_len_out, num_nodes, hidden_channels = h.shape
            
            # Flatten temporal and channel dimensions: [batch, num_nodes, seq_len_out * hidden_channels]
            h_flat = h.transpose(1, 2).reshape(batch_size, num_nodes, seq_len_out * hidden_channels)
            
            # Output prediction: [batch, num_nodes, forecast_horizon]
            out = self.lin(h_flat)
            return out
else:
    class STGCNBlock:
        def __init__(self, *args, **kwargs):
            ensure_torch_available()

    class STGCN:
        def __init__(self, *args, **kwargs):
            ensure_torch_available()

def prepare_stgcn_sequences(df, feature_cols, target_col, nodes, look_back=4):
    """
    Prepares chronological sequences of node features and targets for STGCN training.
    Returns:
        X_seq: [num_samples, look_back, num_nodes, num_features]
        y_seq: [num_samples, num_nodes]
        timestamps: list of target timestamps
    """
    # Group by Timestamp and Node_ID to create matrices
    all_timestamps = sorted(df['Timestamp'].unique())
    num_nodes = len(nodes)
    num_features = len(feature_cols)
    
    # Scale features
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(df[feature_cols].values)
    
    feature_map = {}
    target_map = {}
    
    for idx, row in df.iterrows():
        ts = row['Timestamp']
        node = row['Node_ID']
        feature_map[(ts, node)] = scaled_feats[idx]
        target_map[(ts, node)] = row[target_col]
        
    X_seq, y_seq, seq_ts = [], [], []
    
    for i in range(look_back, len(all_timestamps)):
        target_ts = all_timestamps[i]
        
        # Build sequence matrix for look_back window
        seq_features = []
        for l in range(look_back):
            ts_step = all_timestamps[i - look_back + l]
            step_matrix = [feature_map.get((ts_step, node), np.zeros(num_features)) for node in nodes]
            seq_features.append(step_matrix)
            
        target_values = [target_map.get((target_ts, node), 0.0) for node in nodes]
        
        X_seq.append(seq_features)
        y_seq.append(target_values)
        seq_ts.append(target_ts)
        
    return np.array(X_seq), np.array(y_seq), seq_ts, scaler

def train_stgcn_model(df_train, df_test, target_col, epochs=30, batch_size=16, lr=0.01, device="cpu"):
    """
    Trains the STGCN model on the training dataset and returns aligned test predictions.
    """
    ensure_torch_available()
    if not TORCH_AVAILABLE:
        print(f"Running fallback Sklearn Model for {target_col}...")
        from sklearn.ensemble import RandomForestRegressor
        
        feature_cols = [
            'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
            'Is_Weekend', 'Is_Rush', 'Is_Night', 'Ramadan_Effect'
        ]
        X_train = df_train[feature_cols].fillna(0)
        y_train = df_train[target_col]
        X_test = df_test[feature_cols].fillna(0)
        
        rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        return rf.predict(X_test)

    print(f"Initializing STGCN Model for target {target_col}...")
    
    # Load physical electrical graph to extract connections
    _, electrical_g, _ = build_graphs()
    edge_index, edge_weight, nodes = get_pyg_graph(electrical_g)
    
    # Define features based on target
    if 'Pedestrians' in target_col:
        feature_cols = [
            'Ped_Lag_1', 'Ped_Lag_2', 'Ped_Lag_3', 'Ped_Lag_4',
            'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
            'Is_Weekend', 'Is_Rush', 'Is_Night', 'Ramadan_Effect',
            'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality'
        ]
        model_name = "stgcn_pedestrians.pt"
        scaler_name = "scaler_stgcn_pedestrians.pkl"
    else:
        feature_cols = [
            'Veh_Lag_1', 'Veh_Lag_2', 'Veh_Lag_3', 'Veh_Lag_4',
            'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
            'Is_Weekend', 'Is_Rush', 'Is_Night', 'Ramadan_Effect',
            'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality'
        ]
        model_name = "stgcn_vehicles.pt"
        scaler_name = "scaler_stgcn_vehicles.pkl"
        
    # Prepare sequence datasets
    all_df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    X_seq, y_seq, seq_ts, scaler = prepare_stgcn_sequences(all_df, feature_cols, target_col, nodes, look_back=4)
    
    # Identify indices corresponding to train and test
    train_ts_set = set(df_train['Timestamp'].unique())
    test_ts_set = set(df_test['Timestamp'].unique())
    
    train_idx = [i for i, ts in enumerate(seq_ts) if ts in train_ts_set]
    test_idx = [i for i, ts in enumerate(seq_ts) if ts in test_ts_set]
    
    X_train_tensor = torch.tensor(X_seq[train_idx], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_seq[train_idx], dtype=torch.float32)
    X_test_tensor = torch.tensor(X_seq[test_idx], dtype=torch.float32)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize STGCN Network
    model = STGCN(num_features=len(feature_cols), hidden_channels=32, kernel_size=2)
    model = model.to(device)
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Training Loop
    model.train()
    print("Training STGCN...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X, edge_index, edge_weight) # [batch, num_nodes, 1]
            loss = criterion(out.squeeze(-1), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        if (epoch + 1) % 5 == 0:
            print(f"   [Epoch {epoch+1:02d}/{epochs:02d}] STGCN MSE Loss: {epoch_loss:.6f}")
            
    # Inference
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        test_out = model(X_test_tensor, edge_index, edge_weight) # [test_samples, num_nodes, 1]
        preds = test_out.squeeze(-1).cpu().numpy()
        
    # Map predictions back to (Timestamp, Node_ID)
    test_ts_list = [seq_ts[i] for i in test_idx]
    pred_map = {}
    for i, ts in enumerate(test_ts_list):
        for j, node in enumerate(nodes):
            pred_map[(ts, node)] = max(0.0, float(preds[i, j]))
            
    # Align row-by-row with df_test
    aligned_preds = []
    for _, row in df_test.iterrows():
        val = pred_map.get((row['Timestamp'], row['Node_ID']), row[target_col])
        aligned_preds.append(val)
        
    # Save Model Weights and Scaler
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))
    with open(os.path.join(model_dir, scaler_name), 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Saved STGCN model and scaler to {model_dir}/")
    return np.array(aligned_preds)

def load_and_predict_stgcn(df_train, df_test, target_col, device="cpu"):
    """
    Loads saved STGCN model weights and makes predictions.
    Falls back to training if weights do not exist.
    """
    if not TORCH_AVAILABLE:
        print(f"WARNING: Cannot load STGCN ({TORCH_IMPORT_ERROR}). Falling back to Sklearn.")
        return train_stgcn_model(df_train, df_test, target_col, epochs=10, device=device)

    model_dir = "models"
    model_name = "stgcn_pedestrians.pt" if 'Pedestrians' in target_col else "stgcn_vehicles.pt"
    scaler_name = "scaler_stgcn_pedestrians.pkl" if 'Pedestrians' in target_col else "scaler_stgcn_vehicles.pkl"
    model_path = os.path.join(model_dir, model_name)
    scaler_path = os.path.join(model_dir, scaler_name)
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        # Train from scratch
        return train_stgcn_model(df_train, df_test, target_col, epochs=20, device=device)
        
    print(f"Loading pre-trained STGCN model from {model_path}...")
    
    # Load scaling and graph data
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    _, electrical_g, _ = build_graphs()
    edge_index, edge_weight, nodes = get_pyg_graph(electrical_g)
    
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
        
    # Prepare sequence datasets
    all_df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    
    # Scaling using saved scaler
    scaled_feats = scaler.transform(all_df[feature_cols].values)
    feature_map = {}
    target_map = {}
    for idx, row in all_df.iterrows():
        ts = row['Timestamp']
        node = row['Node_ID']
        feature_map[(ts, node)] = scaled_feats[idx]
        target_map[(ts, node)] = row[target_col]
        
    all_timestamps = sorted(all_df['Timestamp'].unique())
    X_seq, seq_ts = [], []
    for i in range(4, len(all_timestamps)):
        target_ts = all_timestamps[i]
        seq_features = []
        for l in range(4):
            ts_step = all_timestamps[i - 4 + l]
            step_matrix = [feature_map.get((ts_step, node), np.zeros(len(feature_cols))) for node in nodes]
            seq_features.append(step_matrix)
        X_seq.append(seq_features)
        seq_ts.append(target_ts)
        
    test_ts_set = set(df_test['Timestamp'].unique())
    test_idx = [i for i, ts in enumerate(seq_ts) if ts in test_ts_set]
    
    X_test_tensor = torch.tensor(np.array(X_seq)[test_idx], dtype=torch.float32)
    
    model = STGCN(num_features=len(feature_cols), hidden_channels=32, kernel_size=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
        
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_tensor, edge_index, edge_weight)
        preds = test_out.squeeze(-1).cpu().numpy()
        
    # Map predictions back to aligned predictions
    test_ts_list = [seq_ts[i] for i in test_idx]
    pred_map = {}
    for i, ts in enumerate(test_ts_list):
        for j, node in enumerate(nodes):
            pred_map[(ts, node)] = max(0.0, float(preds[i, j]))
            
    aligned_preds = []
    for _, row in df_test.iterrows():
        val = pred_map.get((row['Timestamp'], row['Node_ID']), row[target_col])
        aligned_preds.append(val)
        
    return np.array(aligned_preds)
