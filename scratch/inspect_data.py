import os
import sys
import pandas as pd

# Fix encoding
sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Downloads\Lumina-Smart-Grid-GNN"
if root_dir not in sys.path:
    sys.path.append(root_dir)

from prediction.graph_builder import clean_load_excel

data_dir = r"c:\Users\HP\Downloads\Lumina-Smart-Grid-GNN\data"
files = [
    "15_Extended_Nodes.xlsx",
    "16_Adjacency_Matrix.xlsx",
    "20_Fault_History_500.xlsx",
    "21_NLP_Reports_300.xlsx",
    "25_Predictive_Maintenance.xlsx",
    "26_Route_Optimization.xlsx"
]

for file in files:
    path = os.path.join(data_dir, file)
    if os.path.exists(path):
        print(f"=== {file} ===")
        try:
            df = clean_load_excel(path)
            print(f"Columns: {df.columns.tolist()}")
            print(f"Shape: {df.shape}")
            # Print row values by replacing any non-ascii characters to avoid print errors
            first_rows = df.head(2).to_string(index=False)
            print(first_rows.encode('ascii', errors='replace').decode('ascii'))
            print()
        except Exception as e:
            print(f"Error reading {file}: {e}\n")
    else:
        print(f"{file} does not exist\n")
