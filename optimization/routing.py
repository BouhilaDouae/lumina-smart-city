import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from prediction.graph_builder import build_graphs

def bellman_ford_paths(graph, source):
    """
    Academic-grade implementation of Bellman-Ford algorithm on a graph.
    Computes single-source shortest path distances and predecessors.
    Detects negative weight cycles.
    """
    nodes = list(graph.nodes())
    edges = []
    for u, v, d in graph.edges(data=True):
        weight = float(d.get('distance_m', 100.0))
        edges.append((u, v, weight))
        # Since it's a road graph, if it's bidirectional we add the reverse edge
        if d.get('edge_type') == 'road' or d.get('bidirectional') == 'Yes':
            edges.append((v, u, weight))
            
    # Step 1: Initialize distances and predecessors
    dist = {node: float('inf') for node in nodes}
    pred = {node: None for node in nodes}
    dist[source] = 0.0
    
    # Step 2: Relax edges |V| - 1 times
    for _ in range(len(nodes) - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                
    # Step 3: Check for negative weight cycles
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break
            
    return dist, pred, has_negative_cycle

def get_path_from_predecessors(pred, target):
    """Reconstructs the path from source to target using the predecessor map."""
    path = []
    curr = target
    while curr is not None:
        path.append(curr)
        curr = pred[curr]
    path.reverse()
    return path

def plan_maintenance_route(source_node, target_node, data_dir="data"):
    """
    Plans the optimal route from a maintenance source depot to a faulty lamp node
    using the Bellman-Ford algorithm on the road network.
    """
    # Load road graph
    road_g, _, _ = build_graphs(data_dir)
    
    if source_node not in road_g:
        raise ValueError(f"Source node {source_node} not found in road network.")
    if target_node not in road_g:
        raise ValueError(f"Target node {target_node} not found in road network.")
        
    dist, pred, has_neg_cycle = bellman_ford_paths(road_g, source_node)
    
    if has_neg_cycle:
        print("Warning: Negative cycle detected in road network!")
        
    total_dist = dist[target_node]
    
    if total_dist == float('inf'):
        # No path found
        return {
            'path': [],
            'distance_m': 0.0,
            'hops': 0,
            'time_min': 0.0,
            'has_negative_cycle': has_neg_cycle,
            'status': 'No Path Connected'
        }
        
    path = get_path_from_predecessors(pred, target_node)
    hops = len(path) - 1
    
    # Estimate travel time: assume average municipal maintenance vehicle speed of 25 km/h
    # 25 km/h = 25000 m / 60 min = 416.7 m / min
    time_min = total_dist / 416.7
    
    return {
        'path': path,
        'distance_m': round(total_dist, 1),
        'hops': hops,
        'time_min': round(time_min, 1),
        'has_negative_cycle': has_neg_cycle,
        'status': 'Optimal Path Found'
    }
