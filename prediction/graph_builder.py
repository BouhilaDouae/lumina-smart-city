import os
import pandas as pd
import networkx as nx
import numpy as np

def clean_load_excel(file_path):
    """Loads an Excel file and dynamically identifies the correct header row."""
    df = pd.read_excel(file_path)
    # Search for a row containing column markers
    header_idx = None
    for i in range(min(10, len(df))):
        row_vals = [str(x).strip() for x in df.iloc[i].values]
        if any(marker in row_vals for marker in ['Node_ID', 'Node_i', 'Timestamp', 'Route_ID', 'Fault_ID', 'Lamp_ID', 'Report_ID']):
            header_idx = i
            break
    
    if header_idx is not None:
        cols = df.iloc[header_idx].tolist()
        cleaned_df = df.iloc[header_idx+1:].copy()
        cleaned_df.columns = [str(c).strip() for c in cols]
        # Clean unnamed/NaN columns
        cleaned_df = cleaned_df.loc[:, ~cleaned_df.columns.str.startswith('Unnamed')]
        cleaned_df = cleaned_df.loc[:, cleaned_df.columns.notna()]
        return cleaned_df
    return df

def build_graphs(data_dir="data"):
    """Parses node and adjacency matrix data to construct road and electrical graphs."""
    nodes_file = os.path.join(data_dir, "15_Extended_Nodes.xlsx")
    adj_file = os.path.join(data_dir, "16_Adjacency_Matrix.xlsx")
    
    df_nodes = clean_load_excel(nodes_file)
    df_adj = clean_load_excel(adj_file)
    
    # Initialize graphs
    road_graph = nx.Graph()
    electrical_graph = nx.Graph()
    
    # Add nodes with attributes
    for _, row in df_nodes.iterrows():
        node_id = row['Node_ID']
        attrs = row.to_dict()
        # Convert numeric types
        for key in ['Latitude', 'Longitude', 'Elevation_m', 'Height_m']:
            if key in attrs:
                try:
                    attrs[key] = float(attrs[key])
                except (ValueError, TypeError):
                    pass
        road_graph.add_node(node_id, **attrs)
        electrical_graph.add_node(node_id, **attrs)
        
    # Add edges
    for _, row in df_adj.iterrows():
        node_i = row['Node_i']
        node_j = row['Node_j']
        edge_type = str(row['Edge_Type']).strip().lower()
        bidirectional = str(row['Bidirectional']).strip().lower() == 'yes'
        
        weight = float(row['Weight'])
        dist = float(row['Distance_m'])
        
        edge_attrs = {
            'weight': weight,
            'distance_m': dist,
            'edge_type': edge_type
        }
        
        if edge_type == 'road':
            road_graph.add_edge(node_i, node_j, **edge_attrs)
            if bidirectional and not road_graph.has_edge(node_j, node_i):
                road_graph.add_edge(node_j, node_i, **edge_attrs)
        elif edge_type == 'electrical':
            electrical_graph.add_edge(node_i, node_j, **edge_attrs)
            if bidirectional and not electrical_graph.has_edge(node_j, node_i):
                electrical_graph.add_edge(node_j, node_i, **edge_attrs)
                
    return road_graph, electrical_graph, df_nodes

def compute_graph_metrics(graph):
    """Computes academic-grade graph centrality and degree metrics."""
    metrics = {}
    
    # Degree metrics
    degrees = dict(graph.degree())
    degree_centrality = nx.degree_centrality(graph)
    
    # Closeness centrality
    closeness = nx.closeness_centrality(graph)
    
    # Betweenness centrality
    betweenness = nx.betweenness_centrality(graph)
    
    # Eigenvector centrality (with fallback if it doesn't converge)
    try:
        eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {node: 0.0 for node in graph.nodes()}
        
    for node in graph.nodes():
        metrics[node] = {
            'degree': degrees.get(node, 0),
            'degree_centrality': degree_centrality.get(node, 0.0),
            'closeness_centrality': closeness.get(node, 0.0),
            'betweenness_centrality': betweenness.get(node, 0.0),
            'eigenvector_centrality': eigenvector.get(node, 0.0)
        }
        
    return metrics

def plot_graph_topology(graph, title, output_path):
    """Plots the spatial network structure using Matplotlib and saves to disk."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))
    
    # Extract coordinates
    pos = {}
    for node, attrs in graph.nodes(data=True):
        if 'Longitude' in attrs and 'Latitude' in attrs:
            pos[node] = (attrs['Longitude'], attrs['Latitude'])
            
    if len(pos) == 0:
        # Fallback to spring layout if coordinates missing
        pos = nx.spring_layout(graph)
        
    # Draw nodes based on degree
    degrees = dict(graph.degree())
    node_sizes = [v * 30 + 50 for v in degrees.values()]
    
    # Draw edges with opacity based on weight
    edges = graph.edges(data=True)
    if edges:
        weights = [e[2].get('weight', 1.0) for e in edges]
        max_weight = max(weights) if weights else 1.0
        edge_widths = [1.5 + (w / max_weight) * 3 for w in weights]
    else:
        edge_widths = 1
        
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='skyblue', edgecolors='navy')
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.5, edge_color='gray')
    
    # Label top nodes by degree
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    labels = {node: node for node, _ in top_nodes}
    nx.draw_networkx_labels(graph, pos, labels, font_size=8, font_weight='bold')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_pyg_graph(nx_graph):
    """
    Converts a NetworkX graph into PyTorch Geometric tensors.
    Returns:
        edge_index: torch.LongTensor of shape [2, 2 * num_edges]
        edge_weight: torch.FloatTensor of shape [2 * num_edges]
        nodes: Sorted list of node IDs
    """
    import torch
    nodes = sorted(nx_graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    edge_index_list = []
    edge_weight_list = []
    
    for u, v, d in nx_graph.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        
        # Bidirectional edges for undirected graph
        edge_index_list.append([u_idx, v_idx])
        edge_index_list.append([v_idx, u_idx])
        
        # Weight inversely proportional to distance (closer = stronger)
        dist = float(d.get('distance_m', 100.0))
        weight = 1.0 / (dist + 1e-5)
        
        edge_weight_list.append(weight)
        edge_weight_list.append(weight)
        
    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
        # Normalize weights
        edge_weight = edge_weight / (edge_weight.max() + 1e-8)
        
    return edge_index, edge_weight, nodes

if __name__ == "__main__":
    print("Building graphs from datasets...")
    road_g, elect_g, df_nodes = build_graphs()
    print(f"Road graph built: {road_g.number_of_nodes()} nodes, {road_g.number_of_edges()} edges")
    print(f"Electrical graph built: {elect_g.number_of_nodes()} nodes, {elect_g.number_of_edges()} edges")
    
    # Compute metrics
    metrics = compute_graph_metrics(road_g)
    print("Graph metrics computed successfully for road network.")
    
    # Plot topology
    plot_graph_topology(road_g, "Road Network Topology - Lumina Smart Grid", "scratch/plots/road_network.png")
    plot_graph_topology(elect_g, "Electrical Grid Topology - Lumina Smart Grid", "scratch/plots/electrical_grid.png")
    print("Saved network topology plots to scratch/plots/")
