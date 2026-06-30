import os
import sys

def main():
    print("Testing Lumina Smart Grid GNN Project...")

    # 1. Test Data Generation
    print("\n--- Testing Data Generation ---")
    try:
        from data.generate_datasets import main as generate_data
        generate_data()
        print("Data generation passed.")
    except Exception as e:
        print(f"Data generation failed: {e}")

    # 2. Test STGCN Training
    print("\n--- Testing STGCN Training ---")
    try:
        from prediction.stgcn_model import train_stgcn_model
        # Simple test mock call if needed, or rely on run_pipeline
        print("STGCN code is loaded.")
    except Exception as e:
        print(f"STGCN Training module failed to load: {e}")

    # 3. Test Routing (Bellman-Ford)
    print("\n--- Testing Routing ---")
    try:
        from optimization.routing import run_routing_example
        run_routing_example()
        print("Routing passed.")
    except Exception as e:
        print(f"Routing failed: {e}")

    # 4. Test Cabling (Kruskal)
    print("\n--- Testing Cabling ---")
    try:
        from optimization.cabling import run_cabling_example
        run_cabling_example()
        print("Cabling passed.")
    except Exception as e:
        print(f"Cabling failed: {e}")

    # 5. Test NLP
    print("\n--- Testing NLP ---")
    try:
        from prediction.nlp_analyzer import analyze_reports
        # Call it with dummy data if needed
        print("NLP Module is loaded.")
    except Exception as e:
        print(f"NLP failed: {e}")

    # 6. Test Evaluation
    print("\n--- Testing Evaluation ---")
    try:
        from evaluation.evaluate_system import main as eval_main
        eval_main()
        print("System Evaluation passed.")
    except Exception as e:
        print(f"System Evaluation failed: {e}")

if __name__ == "__main__":
    main()
