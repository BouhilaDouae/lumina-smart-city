import os
import sys
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Downloads\Lumina-Smart-Grid-GNN"
if root_dir not in sys.path:
    sys.path.append(root_dir)

from prediction.graph_builder import clean_load_excel

path = r"c:\Users\HP\Downloads\Lumina-Smart-Grid-GNN\data\21_NLP_Reports_300.xlsx"
df = clean_load_excel(path)

# Let's build a matcher
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
tfidf_matrix = vectorizer.fit_transform(df['Raw_Report'].astype(str))

def parse_report(text):
    # 1. Check exact/fuzzy match in training data
    vec = vectorizer.transform([text])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    best_idx = sims.argmax()
    best_sim = sims[best_idx]
    
    # 2. Extract language
    # If contains Arabic chars
    if re.search(r'[\u0600-\u06ff]', text):
        lang = 'Arabic'
    elif any(kw in text.lower() for kw in ['daro', 'khayb', 'khsso', 'mherres', 'dar', 'chi', 'had', 'machhi', 'khellik', 'dwi']):
        lang = 'Darija'
    else:
        lang = 'French'
        
    # Regex extractors
    node_match = re.search(r'Node_(\d+)', text)
    lamp_match = re.search(r'L_(\d+)', text)
    
    extracted_node = f"Node_{int(node_match.group(1)):03d}" if node_match else None
    extracted_lamp = f"L_{int(lamp_match.group(1)):03d}" if lamp_match else None
    
    # If similarity is high, retrieve labels from best match
    if best_sim > 0.6:
        match_row = df.iloc[best_idx]
        pred_node = extracted_node or match_row['Extracted_Node']
        pred_lamp = extracted_lamp or match_row['Extracted_Lamp']
        pred_fault = match_row['Extracted_Fault']
        pred_sentiment = match_row['Sentiment']
        urgency = float(match_row['Urgency_Score'])
    else:
        # Fallback to rules or model
        pred_node = extracted_node or "Node_001"
        pred_lamp = extracted_lamp or "L_001"
        pred_fault = "lamp_failure"
        pred_sentiment = "urgent"
        urgency = 0.5
        
    return {
        'Language': lang,
        'Node': pred_node,
        'Lamp': pred_lamp,
        'Fault': pred_fault,
        'Urgency': urgency,
        'Sentiment': pred_sentiment,
        'Similarity': best_sim
    }

# Let's test on test split
train_df = df.iloc[:240]
test_df = df.iloc[240:]

# Rebuild vectorizer on train
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
tfidf_matrix = vectorizer.fit_transform(train_df['Raw_Report'].astype(str))

correct = 0
for idx, row in test_df.iterrows():
    res = parse_report(row['Raw_Report'])
    is_correct = (res['Node'] == row['Extracted_Node'])
    if is_correct:
        correct += 1

print(f"Test Node Extraction Accuracy: {correct}/{len(test_df)} ({correct/len(test_df):.3f})")
