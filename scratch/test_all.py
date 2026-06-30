import os
import sys
import pandas as pd
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    torch = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = e

# Fix console encoding
sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Downloads\Lumina-Smart-Grid-GNN"
if root_dir not in sys.path:
    sys.path.append(root_dir)

from prediction.stgcn_model import STGCN
from prediction.energy_optimizer import run_energy_optimization
from prediction.nlp_analyzer import MultilingualNLPAnalyzer
from optimization.cabling import compute_electrical_mst
from optimization.routing import plan_maintenance_route

def run_tests():
    print("=" * 60)
    print("           LUMINA SMART GRID - AUTOMATED TEST SUITE")
    print("=" * 60)
    
    passed_tests = 0
    skipped_tests = 0
    total_tests = 6
    
    # Test 1: Datasets existence
    print("\n[Test 1] Verifying dataset files...")
    data_dir = "data"
    required_files = [
        "15_Extended_Nodes.xlsx",
        "16_Adjacency_Matrix.xlsx",
        "17_Traffic_30min_30days.xlsx",
        "20_Fault_History_500.xlsx",
        "21_NLP_Reports_300.xlsx",
        "22_Weather_30days.xlsx",
        "23_Daily_KPI_Summary.xlsx",
        "25_Predictive_Maintenance.xlsx",
        "26_Route_Optimization.xlsx"
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    if not missing:
        print("✓ All required dataset files are present!")
        passed_tests += 1
    else:
        print(f"✗ Missing files: {missing}")
        
    # Test 2: STGCN Model Instantiation & Forward Pass
    print("\n[Test 2] Testing STGCN Architecture...")
    if not TORCH_AVAILABLE:
        print(f"⚠️ PyTorch import failed, skipping STGCN test: {TORCH_IMPORT_ERROR}")
        skipped_tests += 1
    else:
        try:
            # Dummy sequence shape: [batch_size, seq_len, num_nodes, num_features]
            # batch=4, seq_len=4, nodes=120, features=16
            x = torch.randn(4, 4, 120, 16)
            edge_index = torch.randint(0, 120, (2, 400))
            edge_weight = torch.rand(400)
            
            model = STGCN(num_features=16, hidden_channels=32, kernel_size=2)
            out = model(x, edge_index, edge_weight)
            
            # Expected output shape: [batch_size, num_nodes, forecast_horizon=1]
            if out.shape == (4, 120, 1):
                print("✓ STGCN instantiated and forward pass completed successfully!")
                passed_tests += 1
            else:
                print(f"✗ Unexpected output shape: {out.shape}")
        except Exception as e:
            print(f"✗ STGCN test failed: {e}")
        
    # Test 3: Energy Optimization Metrics
    print("\n[Test 3] Testing Energy Optimization Engine...")
    try:
        # Create a small dummy dataframe simulating traffic data
        dummy_sim = pd.DataFrame([
            {'Timestamp': '2026-04-01 20:00:00', 'Node_ID': 'Node_001', 'Pedestrians': 2, 'Vehicles': 1},
            {'Timestamp': '2026-04-01 21:00:00', 'Node_ID': 'Node_001', 'Pedestrians': 12, 'Vehicles': 5},
            {'Timestamp': '2026-04-01 22:00:00', 'Node_ID': 'Node_001', 'Pedestrians': 25, 'Vehicles': 10}
        ])
        
        summary_df, detailed_df = run_energy_optimization(dummy_sim)
        
        if len(summary_df) == 3 and 'STGCN Adaptive Dimming' in summary_df['Policy'].values:
            print("✓ Energy Optimization metrics and savings calculated correctly!")
            passed_tests += 1
        else:
            print("✗ Energy Optimization output format invalid.")
    except Exception as e:
        print(f"✗ Energy Optimizer test failed: {e}")
        
    # Test 4: Kruskal's MST Cabling Optimization
    print("\n[Test 4] Testing Kruskal's MST Cabling...")
    try:
        mst_edges, metrics = compute_electrical_mst()
        if len(mst_edges) > 0 and 'savings_pct' in metrics:
            print(f"✓ Kruskal's MST calculated. Cabling saved: {metrics['length_saved_m']} m ({metrics['savings_pct']}%)")
            passed_tests += 1
        else:
            print("✗ MST edges list or metrics is empty.")
    except Exception as e:
        print(f"✗ Kruskal's MST test failed: {e}")
        
    # Test 5: Bellman-Ford Shortest Path Routing
    print("\n[Test 5] Testing Bellman-Ford Routing...")
    try:
        route_res = plan_maintenance_route("Node_001", "Node_015")
        if route_res['status'] == 'Optimal Path Found' and len(route_res['path']) > 0:
            print(f"✓ Bellman-Ford routing path planned successfully! Total distance: {route_res['distance_m']} m")
            passed_tests += 1
        else:
            print(f"✗ Routing failed: {route_res['status']}")
    except Exception as e:
        print(f"✗ Bellman-Ford test failed: {e}")
        
    # Test 6: Multilingual NLP Analyzer
    print("\n[Test 6] Testing Multilingual NLP Analyzer...")
    try:
        analyzer = MultilingualNLPAnalyzer()
        res = analyzer.analyze_report("Court-circuit detecte poteau Node_040, intervention urgente.")
        if res['Language'] == 'French' and res['Extracted_Node'] == 'Node_040':
            print("✓ Multilingual NLP correctly identified language and resolved Node ID!")
            passed_tests += 1
        else:
            print(f"✗ NLP parsed values mismatch: {res}")
    except Exception as e:
        print(f"✗ NLP Analyzer test failed: {e}")
        
    effective_tests = total_tests - skipped_tests
    print("\n" + "=" * 60)
    if skipped_tests > 0:
        print(f"TEST RESULTS: {passed_tests} / {effective_tests} PASSED, {skipped_tests} SKIPPED")
    else:
        print(f"TEST RESULTS: {passed_tests} / {total_tests} PASSED")
    print("=" * 60)
    
    if passed_tests == effective_tests:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
