import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class TemporalGNN(torch.nn.Module):
    """
    Temporal Graph Neural Network combining GCNConv layers with LSTM cells.
    Used for spatio-temporal predictions of pedestrians and vehicles over nodes.
    """
    def __init__(self, node_features, hidden_channels=64, num_layers=3, forecast_horizon=1):
        super(TemporalGNN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(pyg_nn.GCNConv(node_features, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(pyg_nn.GCNConv(hidden_channels, hidden_channels))
            
        self.convs.append(pyg_nn.GCNConv(hidden_channels, hidden_channels))
        
        # LSTM for capturing temporal trends
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        
        # Output layers
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, forecast_horizon)
        self.relu = nn.ReLU()
        
    def forward(self, data_seq, edge_index, edge_weight=None):
        """
        data_seq: [batch_size, seq_len, num_nodes, node_features]
        edge_index: [2, num_edges]
        edge_weight: [num_edges]
        """
        batch_size, seq_len, num_nodes, node_features = data_seq.shape
        
        # Run GCNConv spatial features for each timestep
        seq_output = []
        for t in range(seq_len):
            x = data_seq[:, t, :, :]  # [batch_size, num_nodes, node_features]
            x = x.reshape(-1, node_features)  # [batch_size*num_nodes, node_features]
            
            h = x
            for conv in self.convs:
                h = self.relu(conv(h, edge_index, edge_weight))
                
            h = h.reshape(batch_size, num_nodes, -1)  # [batch_size, num_nodes, hidden_channels]
            seq_output.append(h)
            
        lstm_input = torch.stack(seq_output, dim=1)  # [batch_size, seq_len, num_nodes, hidden_channels]
        
        # Run temporal processing via LSTM for each node's time series
        lstm_out = []
        for node_idx in range(num_nodes):
            node_seq = lstm_input[:, :, node_idx, :]  # [batch_size, seq_len, hidden_channels]
            out, _ = self.lstm(node_seq)
            lstm_out.append(out[:, -1, :])  # Take the last hidden state [batch_size, hidden_channels]
            
        lstm_out = torch.stack(lstm_out, dim=1)  # [batch_size, num_nodes, hidden_channels]
        
        # Linear projection to forecast
        x = self.relu(self.lin1(lstm_out))
        x = self.lin2(x)  # [batch_size, num_nodes, forecast_horizon]
        return x

class HistoricalAverageModel:
    """Historical Average (HA) model that predicts traffic based on historical mean for specific node, hour, and day-of-week."""
    def __init__(self):
        self.lookup = {}
        
    def fit(self, X_train, y_train):
        # We need the original Node_ID, Hour, and Day_of_Week which are in X_train
        # Let's combine them with y_train to compute the means
        df = X_train.copy()
        df['Target'] = y_train
        
        # Group by Node_ID, Hour, and Day_of_Week
        self.lookup = df.groupby(['Node_ID', 'Hour', 'Day_of_Week'])['Target'].mean().to_dict()
        self.global_mean = y_train.mean()
        
    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            key = (row['Node_ID'], row['Hour'], row['Day_of_Week'])
            predictions.append(self.lookup.get(key, self.global_mean))
        return np.array(predictions)

def get_temporal_features():
    return [
        'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
        'Is_Weekend', 'Is_Rush', 'Is_Night', 'Ramadan_Effect',
        'Ped_Lag_1', 'Veh_Lag_1', 'Ped_Lag_2', 'Veh_Lag_2',
        'Ped_Lag_3', 'Veh_Lag_3', 'Ped_Lag_48', 'Veh_Lag_48'
    ]

def get_spatio_temporal_features():
    return get_temporal_features() + [
        'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality',
        'Spatial_Lag_Ped', 'Spatial_Lag_Veh'
    ]

def train_and_predict_all(df_train, df_test, target_col):
    """Trains and predicts all 5 benchmark models for a given target column (Pedestrians or Vehicles)."""
    y_train = df_train[target_col].values
    y_test = df_test[target_col].values
    
    predictions = {}
    
    # 1. Historical Average (HA)
    ha_model = HistoricalAverageModel()
    ha_model.fit(df_train, y_train)
    predictions['Historical Average'] = ha_model.predict(df_test)
    
    # 2. Linear Ridge Regression (Temporal-Only)
    temp_cols = get_temporal_features()
    ridge_temp = Ridge(alpha=1.0)
    ridge_temp.fit(df_train[temp_cols], y_train)
    predictions['Ridge (Temporal-Only)'] = ridge_temp.predict(df_test[temp_cols])
    
    # 3. Temporal XGBoost (Temporal-Only)
    xgb_temp = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_temp.fit(df_train[temp_cols], y_train)
    predictions['XGBoost (Temporal-Only)'] = xgb_temp.predict(df_test[temp_cols])
    
    # 4. Graph-Feature-Enhanced Ridge Regression (Spatio-Temporal)
    st_cols = get_spatio_temporal_features()
    ridge_st = Ridge(alpha=1.0)
    ridge_st.fit(df_train[st_cols], y_train)
    predictions['Ridge (Spatio-Temporal)'] = ridge_st.predict(df_test[st_cols])
    
    # 5. Graph-Feature-Enhanced XGBoost (Spatio-Temporal)
    xgb_st = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_st.fit(df_train[st_cols], y_train)
    predictions['XGBoost (Spatio-Temporal)'] = xgb_st.predict(df_test[st_cols])
    
    return predictions, y_test
