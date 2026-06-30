import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from prediction.graph_builder import clean_load_excel, compute_graph_metrics, build_graphs

class MultilingualNLPAnalyzer:
    """
    Multilingual NLP classifier to analyze unstructured reports (French, Arabic, Darija).
    Extracts language, fault category, urgency level, Node ID, and Lamp ID.
    """
    def __init__(self, data_dir="data", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.vectorizer = None
        self.lang_clf = None
        self.fault_clf = None
        self.sent_clf = None
        
        # Load node registry for fallback matching
        self.nodes_df = clean_load_excel(os.path.join(data_dir, "15_Extended_Nodes.xlsx"))
        road_g, _, _ = build_graphs(data_dir)
        self.node_metrics = compute_graph_metrics(road_g)
        
        # Cache models
        self.load_or_train_models()
        
    def train_models(self):
        print("Training NLP models from data/21_NLP_Reports_300.xlsx...")
        path = os.path.join(self.data_dir, "21_NLP_Reports_300.xlsx")
        df = clean_load_excel(path)
        
        X = df['Raw_Report'].astype(str).tolist()
        
        # Vectorizer
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=2500)
        X_vec = self.vectorizer.fit_transform(X)
        
        # Language Classifier
        self.lang_clf = LogisticRegression(max_iter=1000)
        self.lang_clf.fit(X_vec, df['Language'].astype(str))
        
        # Fault Classifier
        self.fault_clf = LogisticRegression(max_iter=1000)
        self.fault_clf.fit(X_vec, df['Extracted_Fault'].astype(str))
        
        # Sentiment/Urgency Classifier
        self.sent_clf = LogisticRegression(max_iter=1000)
        self.sent_clf.fit(X_vec, df['Sentiment'].astype(str))
        
        # Save models
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, "nlp_vectorizer.pkl"), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(self.model_dir, "nlp_lang_clf.pkl"), 'wb') as f:
            pickle.dump(self.lang_clf, f)
        with open(os.path.join(self.model_dir, "nlp_fault_clf.pkl"), 'wb') as f:
            pickle.dump(self.fault_clf, f)
        with open(os.path.join(self.model_dir, "nlp_sent_clf.pkl"), 'wb') as f:
            pickle.dump(self.sent_clf, f)
        print("NLP models successfully trained and cached.")

    def load_or_train_models(self):
        vec_path = os.path.join(self.model_dir, "nlp_vectorizer.pkl")
        lang_path = os.path.join(self.model_dir, "nlp_lang_clf.pkl")
        fault_path = os.path.join(self.model_dir, "nlp_fault_clf.pkl")
        sent_path = os.path.join(self.model_dir, "nlp_sent_clf.pkl")
        
        if (os.path.exists(vec_path) and os.path.exists(lang_path) and 
            os.path.exists(fault_path) and os.path.exists(sent_path)):
            print("Loading pre-trained NLP models...")
            with open(vec_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(lang_path, 'rb') as f:
                self.lang_clf = pickle.load(f)
            with open(fault_path, 'rb') as f:
                self.fault_clf = pickle.load(f)
            with open(sent_path, 'rb') as f:
                self.sent_clf = pickle.load(f)
        else:
            self.train_models()

    def analyze_report(self, text):
        """
        Analyzes a single report text and returns classified entities and scores.
        """
        # Vectorize text
        vec = self.vectorizer.transform([text])
        
        # Predict categorical fields
        pred_lang = self.lang_clf.predict(vec)[0]
        pred_fault = self.fault_clf.predict(vec)[0]
        pred_sent = self.sent_clf.predict(vec)[0]
        
        # Urgency scoring based on predicted sentiment
        sent_urgency_map = {
            'critical': 0.95,
            'urgent': 0.75,
            'moderate': 0.45,
            'low': 0.15
        }
        urgency_score = sent_urgency_map.get(pred_sent, 0.5)
        
        # Regex Entity Extraction
        # Look for explicit Node and Lamp IDs (e.g. Node_012, L_045)
        node_match = re.search(r'(?:Node_?|poteau_?|poste_?)\s*(\d+)', text, re.IGNORECASE)
        lamp_match = re.search(r'(?:L_?|lampadaire_?)\s*(\d+)', text, re.IGNORECASE)
        
        extracted_node = f"Node_{int(node_match.group(1)):03d}" if node_match else None
        extracted_lamp = f"L_{int(lamp_match.group(1)):02d}" if lamp_match else None
        
        # Fallback to Street or Sector matching in text if Node is not found
        if not extracted_node:
            matched_streets = []
            for _, row in self.nodes_df.iterrows():
                street = str(row['Street'])
                if street.lower() in text.lower():
                    matched_streets.append(row['Node_ID'])
            
            if matched_streets:
                # Resolve tie: choose matching node with highest degree centrality
                matched_streets.sort(key=lambda nid: self.node_metrics.get(nid, {}).get('degree_centrality', 0.0), reverse=True)
                extracted_node = matched_streets[0]
                
        # If still no node match, check Sector matching
        if not extracted_node:
            for s in ['Sector_A', 'Sector_B', 'Sector_C', 'Sector_D', 'Sector_E', 'Sector_F', 'Sector_G']:
                if s.lower() in text.lower() or s.replace('_', ' ').lower() in text.lower():
                    # Find first node in that sector
                    nodes_in_sec = self.nodes_df[self.nodes_df['Sector'] == s]['Node_ID'].tolist()
                    if nodes_in_sec:
                        extracted_node = nodes_in_sec[0]
                        break
                        
        # Ultimate fallback to Node_001 if nothing matches
        if not extracted_node:
            extracted_node = "Node_001"
            
        # Ultimate fallback to Lamp ID from predictive maintenance if Node is resolved
        if not extracted_lamp and extracted_node:
            # Check if a lamp is connected to this node in node registry / pm
            pm_file = os.path.join(self.data_dir, "25_Predictive_Maintenance.xlsx")
            if os.path.exists(pm_file):
                try:
                    df_pm = clean_load_excel(pm_file)
                    match_lamp = df_pm[df_pm['Node_ID'] == extracted_node]['Lamp_ID'].tolist()
                    if match_lamp:
                        extracted_lamp = match_lamp[0]
                except:
                    pass
            if not extracted_lamp:
                extracted_lamp = "L_01"
                
        return {
            'Language': pred_lang,
            'Extracted_Fault': pred_fault,
            'Sentiment': pred_sent,
            'Urgency_Score': urgency_score,
            'Extracted_Node': extracted_node,
            'Extracted_Lamp': extracted_lamp,
            'Confidence_Score': float(np.max(self.fault_clf.predict_proba(vec)))
        }

if __name__ == "__main__":
    analyzer = MultilingualNLPAnalyzer()
    
    # Test cases
    test_1 = "Court-circuit detecte poteau Node_040, intervention urgente."
    test_2 = "العمود الكهربائي Node_112 مائل بشكل خطير، يرجى التدخل."
    test_3 = "Câble d Al Fida mherres, khesshom yji ysalho bsraa."
    
    print("\nTest 1 Output:")
    print(analyzer.analyze_report(test_1))
    print("\nTest 2 Output:")
    print(analyzer.analyze_report(test_2))
    print("\nTest 3 Output:")
    print(analyzer.analyze_report(test_3))
