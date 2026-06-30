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

print("Language counts:")
print(df['Language'].value_counts())
print("\nSample reports:")
for idx, row in df.head(15).iterrows():
    print(f"[{row['Language']}] {row['Raw_Report']}")
    print(f"  -> Node: {row['Extracted_Node']}, Lamp: {row['Extracted_Lamp']}, Fault: {row['Extracted_Fault']}, Urgency: {row['Urgency_Score']}, Sentiment: {row['Sentiment']}\n")
