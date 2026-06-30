import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Downloads\Lumina-Smart-Grid-GNN"
if root_dir not in sys.path:
    sys.path.append(root_dir)

from prediction.graph_builder import clean_load_excel

path = r"c:\Users\HP\Downloads\Lumina-Smart-Grid-GNN\data\21_NLP_Reports_300.xlsx"
df = clean_load_excel(path)

print(f"Total reports: {len(df)}")
print(f"Unique Extracted_Node: {df['Extracted_Node'].nunique()}")
print(f"Unique Extracted_Lamp: {df['Extracted_Lamp'].nunique()}")
print(f"Unique Extracted_Fault: {df['Extracted_Fault'].nunique()}")
print(f"Unique Sentiment (Urgency Level): {df['Sentiment'].value_counts()}")
print(f"Unique Extracted_Fault values: {df['Extracted_Fault'].value_counts()}")
