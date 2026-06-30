import os
import sys
import pandas as pd

root_dir = r"c:\Users\HP\Downloads\Lumina-Smart-Grid-GNN"
if root_dir not in sys.path:
    sys.path.append(root_dir)

from prediction.graph_builder import clean_load_excel

data_dir = r"c:\Users\HP\Downloads\Lumina-Smart-Grid-GNN\data"
files = [
    "15_Extended_Nodes.xlsx",
    "16_Adjacency_Matrix.xlsx",
    "20_Fault_History_500.xlsx",
    "25_Predictive_Maintenance.xlsx",
    "26_Route_Optimization.xlsx"
]

search_val = "Node_088"
search_val_2 = "L_033"

for file in files:
    path = os.path.join(data_dir, file)
    if os.path.exists(path):
        df = clean_load_excel(path)
        for col in df.columns:
            matches = df[df[col].astype(str).str.contains(search_val, na=False)]
            if len(matches) > 0:
                print(f"File {file}, Col {col} has matches for {search_val}: {len(matches)} rows")
            matches_2 = df[df[col].astype(str).str.contains(search_val_2, na=False)]
            if len(matches_2) > 0:
                print(f"File {file}, Col {col} has matches for {search_val_2}: {len(matches_2)} rows")
