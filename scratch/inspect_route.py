import os
import pandas as pd

path = r"c:\Users\HP\Downloads\Lumina-Smart-Grid-GNN\data\26_Route_Optimization.xlsx"
df = pd.read_excel(path)
for i in range(10):
    print(f"Row {i}: {df.iloc[i].values.tolist()}")
