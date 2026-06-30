import os
import sys
import pandas as pd
import networkx as nx

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from prediction.graph_builder import build_graphs

class DisjointSetUnion:
    """Disjoint Set Union (DSU) / Union-Find data structure for Kruskal's algorithm."""
    def __init__(self, elements):
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}

    def find(self, i):
        if self.parent[i] == i:
            return i
        # Path compression
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            # Union by rank
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False

def compute_electrical_mst(data_dir="data"):
    """
    Computes the Minimum Spanning Forest/Tree of the electrical grid using Kruskal's Algorithm.
    Returns:
        mst_edges: list of edges in the MST/MSF, with attributes
        metrics: dict of cabling savings metrics
    """
    # 1. Load electrical graph
    _, electrical_g, _ = build_graphs(data_dir)
    nodes = list(electrical_g.nodes())
    
    # 2. Extract and sort electrical edges by Distance_m
    edges = []
    for u, v, d in electrical_g.edges(data=True):
        dist = float(d.get('distance_m', 100.0))
        edges.append((dist, u, v, d))
        
    # Sort edges by weight/distance (Kruskal's Step 1)
    edges.sort(key=lambda x: x[0])
    
    # 3. Apply Kruskal's Algorithm using DSU
    dsu = DisjointSetUnion(nodes)
    mst_edges = []
    
    total_original_length = sum([e[0] for e in edges])
    total_optimized_length = 0.0
    
    for dist, u, v, d in edges:
        # Check if u and v are in different components
        if dsu.find(u) != dsu.find(v):
            dsu.union(u, v)
            mst_edges.append((u, v, d))
            total_optimized_length += dist
            
    # Calculate savings
    length_saved = total_original_length - total_optimized_length
    pct_saved = (length_saved / total_original_length) * 100.0 if total_original_length > 0 else 0.0
    
    # Line losses estimation: line losses are proportional to cabling length
    # Let's assume original cabling had 5.0% line losses (relative to total power).
    # Cabling reduction directly reduces the line resistance and thus line losses.
    original_losses_kWh = total_original_length * 0.05 # arbitrary scale for baseline losses
    optimized_losses_kWh = total_optimized_length * 0.05
    losses_saved_kWh = original_losses_kWh - optimized_losses_kWh
    
    metrics = {
        'original_edges': electrical_g.number_of_edges(),
        'mst_edges': len(mst_edges),
        'original_length_m': round(total_original_length, 1),
        'optimized_length_m': round(total_optimized_length, 1),
        'length_saved_m': round(length_saved, 1),
        'savings_pct': round(pct_saved, 2),
        'estimated_line_loss_saved_kWh': round(losses_saved_kWh, 2)
    }
    
    return mst_edges, metrics
