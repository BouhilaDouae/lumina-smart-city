import numpy as np
import pandas as pd
import networkx as nx
from prediction.graph_builder import build_graphs, compute_graph_metrics

def create_temporal_features(df):
    """Encodes cyclic time features and extracts weekend/rush-hour/holiday flags."""
    df = df.copy()
    
    # Cyclic encoding for Hour (0-23)
    df['sin_hour'] = np.sin(2 * np.pi * df['Hour'].astype(float) / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['Hour'].astype(float) / 24.0)
    
    # Cyclic encoding for Day of Week (0-6)
    dow_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    if df['Day_of_Week'].dtype == object or str(df['Day_of_Week'].dtype).startswith('str'):
        dow_numeric = df['Day_of_Week'].map(dow_map).fillna(0).astype(float)
    else:
        dow_numeric = df['Day_of_Week'].astype(float)
        
    df['sin_dow'] = np.sin(2 * np.pi * dow_numeric / 7.0)
    df['cos_dow'] = np.cos(2 * np.pi * dow_numeric / 7.0)
    
    # Map Yes/No or boolean flags to 1/0
    for col in ['Is_Weekend', 'Is_Rush', 'Is_Night', 'Ramadan_Effect']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)
            
    return df

def build_spatio_temporal_features(df_traffic, road_graph):
    """Generates lag features and spatial graph neighbor traffic features."""
    # Pivot traffic to a wide format [Timestamp x Node_ID] for easy lag and spatial calculations
    pivoted_ped = df_traffic.pivot(index='Timestamp', columns='Node_ID', values='Pedestrians')
    pivoted_veh = df_traffic.pivot(index='Timestamp', columns='Node_ID', values='Vehicles')
    
    # Compute graph metrics
    metrics = compute_graph_metrics(road_graph)
    
    # Prepare dataframes to hold all engineered features
    features_list = []
    
    # Nodes in traffic data
    traffic_nodes = pivoted_ped.columns.tolist()
    
    # Loop over nodes to construct dataset
    for node in traffic_nodes:
        node_df = pd.DataFrame(index=pivoted_ped.index)
        node_df['Timestamp'] = pivoted_ped.index
        node_df['Node_ID'] = node
        
        # Targets
        node_df['Target_Pedestrians'] = pivoted_ped[node]
        node_df['Target_Vehicles'] = pivoted_veh[node]
        
        # Temporal lags (t-1, t-2, t-3, t-48)
        for lag in [1, 2, 3, 48]:
            node_df[f'Ped_Lag_{lag}'] = pivoted_ped[node].shift(lag)
            node_df[f'Veh_Lag_{lag}'] = pivoted_veh[node].shift(lag)
            
        # Topological centralities
        node_metrics = metrics.get(node, {
            'degree_centrality': 0.0,
            'closeness_centrality': 0.0,
            'betweenness_centrality': 0.0,
            'eigenvector_centrality': 0.0
        })
        for metric_name, val in node_metrics.items():
            node_df[metric_name] = val
            
        # Spatial lag features: weighted sum of neighbor traffic at t-1
        neighbors = list(road_graph.neighbors(node)) if node in road_graph else []
        if neighbors:
            weights = []
            for n in neighbors:
                edge_data = road_graph.get_edge_data(node, n)
                # Fallback if weight missing
                w = edge_data.get('weight', 1.0) if edge_data else 1.0
                weights.append(w)
            weights = np.array(weights)
            # Normalize neighborhood weights
            if weights.sum() > 0:
                weights = weights / weights.sum()
                
            # Compute weighted sums over time
            weighted_ped_lag = np.zeros(len(pivoted_ped))
            weighted_veh_lag = np.zeros(len(pivoted_veh))
            for n, w in zip(neighbors, weights):
                if n in pivoted_ped.columns:
                    weighted_ped_lag += pivoted_ped[n].shift(1).fillna(0).values * w
                    weighted_veh_lag += pivoted_veh[n].shift(1).fillna(0).values * w
            
            node_df['Spatial_Lag_Ped'] = weighted_ped_lag
            node_df['Spatial_Lag_Veh'] = weighted_veh_lag
        else:
            node_df['Spatial_Lag_Ped'] = 0.0
            node_df['Spatial_Lag_Veh'] = 0.0
            
        features_list.append(node_df)
        
    df_features = pd.concat(features_list, axis=0).reset_index(drop=True)
    
    # Merge temporal flags (Hour, Minute, Is_Weekend, sin_hour, cos_hour, etc.) back
    temp_df = df_traffic[['Timestamp', 'Node_ID', 'Hour', 'Minute', 'Day_of_Week', 'Is_Weekend', 'Is_Rush', 'Is_Night', 'Ramadan_Effect']].drop_duplicates()
    temp_df = create_temporal_features(temp_df)
    
    df_merged = pd.merge(df_features, temp_df, on=['Timestamp', 'Node_ID'], how='left')
    
    # Drop rows with NaN values resulting from shifts (e.g., first 48 hours)
    df_merged = df_merged.dropna().reset_index(drop=True)
    return df_merged

if __name__ == "__main__":
    from prediction.data_loader import load_traffic_data
    
    print("Loading raw traffic data...")
    df_raw = load_traffic_data()
    print("Building road graph...")
    road_g, _, _ = build_graphs()
    
    print("Engineering spatio-temporal features...")
    df_features = build_spatio_temporal_features(df_raw, road_g)
    print(f"Features created! Shape: {df_features.shape}")
    print("Columns:", df_features.columns.tolist()[:15])
