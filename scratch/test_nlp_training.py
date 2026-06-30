import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.stdout.reconfigure(encoding='utf-8')

root_dir = r"c:\Users\HP\Downloads\Lumina-Smart-Grid-GNN"
if root_dir not in sys.path:
    sys.path.append(root_dir)

from prediction.graph_builder import clean_load_excel

path = r"c:\Users\HP\Downloads\Lumina-Smart-Grid-GNN\data\21_NLP_Reports_300.xlsx"
df = clean_load_excel(path)

X = df['Raw_Report'].astype(str)

targets = ['Language', 'Extracted_Fault', 'Sentiment', 'Extracted_Node', 'Extracted_Lamp']

for target in targets:
    y = df[target].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train_vec))
    test_acc = accuracy_score(y_test, clf.predict(X_test_vec))
    
    print(f"Target: {target:18s} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")
