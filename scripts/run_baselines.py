import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Define Paths
ROOT_DIR = Path('c:/Users/111/Desktop/Home/NYU/26Spring/NLP/Project')
DATA_DIR = ROOT_DIR / 'data'
RESULTS_DIR = ROOT_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

UNIFIED_DIR = DATA_DIR / 'unified_benchmark'
INDOMAIN_SPLITS_DIR = UNIFIED_DIR / 'indomain_splits'
INDOMAIN_SPLITS_DIR.mkdir(exist_ok=True)
M4_DIR = DATA_DIR / 'm4'
COMBINED_JSON = UNIFIED_DIR / 'combined_dataset.json'

random_seed = 42

# --- Signal Feature Extractor ---
class SignalFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features = ['perplexity', 'burstiness', 'sentence_length_std', 'type_token_ratio', 'punctuation_ratio', 'length']
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = []
        for x in X:
            feats = []
            for f in self.features:
                if f == 'length':
                    feats.append(x.get('length', 0))
                else:
                    feats.append(x.get('features', {}).get(f, 0))
            X_out.append(feats)
        return np.array(X_out)

class TextExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return [x['text'] for x in X]

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def split_and_save(data, dataset_name):
    # Prepare labels for stratification
    labels = [x['label'] for x in data]
    
    # Split 70% train, 30% temp
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, test_size=0.3, stratify=labels, random_state=random_seed
    )
    
    # Split temp into 50% dev, 50% test (which is 15% / 15% of total)
    dev_data, test_data = train_test_split(
        temp_data, test_size=0.5, stratify=temp_labels, random_state=random_seed
    )
    
    # Save files
    save_json(train_data, INDOMAIN_SPLITS_DIR / f"{dataset_name.lower()}_train.json")
    save_json(dev_data, INDOMAIN_SPLITS_DIR / f"{dataset_name.lower()}_dev.json")
    save_json(test_data, INDOMAIN_SPLITS_DIR / f"{dataset_name.lower()}_test.json")
    
    return train_data, dev_data, test_data

def evaluate(model, test_X, test_y):
    preds = model.predict(test_X)
    return {
        'accuracy': accuracy_score(test_y, preds),
        'precision': precision_score(test_y, preds, zero_division=0),
        'recall': recall_score(test_y, preds, zero_division=0),
        'f1': f1_score(test_y, preds, zero_division=0),
        'macro_f1': f1_score(test_y, preds, average='macro', zero_division=0)
    }

def main():
    print("Loading combined dataset...")
    combined_data = load_json(COMBINED_JSON)
    
    # Separate datasets
    controlled_data = [x for x in combined_data if x['dataset'] == 'Controlled']
    hc3_data = [x for x in combined_data if x['dataset'] == 'HC3']
    
    print("Creating splits for Controlled and HC3...")
    c_train, c_dev, c_test = split_and_save(controlled_data, "Controlled")
    h_train, h_dev, h_test = split_and_save(hc3_data, "HC3")
    
    # Load M4 datasets
    print("Loading M4 datasets...")
    m4_train = load_json(M4_DIR / 'train.json')
    m4_dev = load_json(M4_DIR / 'dev.json')
    m4_test = load_json(M4_DIR / 'test.json')
    
    # Save M4 datasets to indomain_splits
    save_json(m4_train, INDOMAIN_SPLITS_DIR / 'm4_train.json')
    save_json(m4_dev, INDOMAIN_SPLITS_DIR / 'm4_dev.json')
    save_json(m4_test, INDOMAIN_SPLITS_DIR / 'm4_test.json')
    
    datasets = {
        'Controlled': (c_train, c_test),
        'HC3': (h_train, h_test),
        'M4': (m4_train, m4_test)
    }

    # Model Pipelines
    models = {
        'TF-IDF': Pipeline([
            ('extractor', TextExtractor()),
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_features=20000, stop_words="english")),
            ('classifier', LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed))
        ]),
        'Signal Features': Pipeline([
            ('extractor', SignalFeatureExtractor()),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed))
        ])
    }

    # --- Experiment A: In-Domain ---
    print("\nRunning Experiment A: In-Domain...")
    results_a = []
    
    for dataset_name, (train_data, test_data) in datasets.items():
        train_y = [x['label'] for x in train_data]
        test_y = [x['label'] for x in test_data]
        
        for model_name, model in models.items():
            print(f"Training {model_name} on {dataset_name}...")
            model.fit(train_data, train_y)
            metrics = evaluate(model, test_data, test_y)
            metrics.update({'Model': model_name, 'Dataset': dataset_name})
            results_a.append(metrics)
            
    df_a = pd.DataFrame(results_a)
    df_a.to_csv(RESULTS_DIR / 'baseline_indomain_results.csv', index=False)
    
    # --- Experiment B: M4 Cross-Generator ---
    print("\nRunning Experiment B: M4 Cross-Generator...")
    generators = ['chatgpt', 'davinci003', 'cohere', 'dolly', 'bloomz']
    results_b = []
    
    for gen in generators:
        train_path = M4_DIR / f'cross_gen_train_heldout_{gen}.json'
        test_path = M4_DIR / f'cross_gen_test_heldout_{gen}.json'
        
        train_data = load_json(train_path)
        test_data = load_json(test_path)
        train_y = [x['label'] for x in train_data]
        test_y = [x['label'] for x in test_data]
        
        for model_name, model in models.items():
            print(f"Training {model_name} on Cross-Gen (held-out: {gen})...")
            model.fit(train_data, train_y)
            metrics = evaluate(model, test_data, test_y)
            metrics.update({'Model': model_name, 'HeldOut_Generator': gen})
            results_b.append(metrics)
            
    df_b = pd.DataFrame(results_b)
    df_b.to_csv(RESULTS_DIR / 'baseline_cross_generator_results.csv', index=False)
    
    # --- Experiment C: M4 Cross-Domain ---
    print("\nRunning Experiment C: M4 Cross-Domain...")
    domains = ['wikipedia', 'reddit_eli5', 'wikihow', 'arxiv', 'peerread']
    results_c = []
    
    for dom in domains:
        train_path = M4_DIR / f'cross_domain_train_heldout_{dom}.json'
        test_path = M4_DIR / f'cross_domain_test_heldout_{dom}.json'
        
        train_data = load_json(train_path)
        test_data = load_json(test_path)
        train_y = [x['label'] for x in train_data]
        test_y = [x['label'] for x in test_data]
        
        for model_name, model in models.items():
            print(f"Training {model_name} on Cross-Domain (held-out: {dom})...")
            model.fit(train_data, train_y)
            metrics = evaluate(model, test_data, test_y)
            metrics.update({'Model': model_name, 'HeldOut_Domain': dom})
            results_c.append(metrics)
            
    df_c = pd.DataFrame(results_c)
    df_c.to_csv(RESULTS_DIR / 'baseline_cross_domain_results.csv', index=False)
    
    # --- Generate Summary ---
    print("\nGenerating Summary Markdown...")
    summary = f"""# Baseline Experiments Summary

## In-Domain Performance
{df_a[['Model', 'Dataset', 'accuracy', 'f1', 'macro_f1']].to_markdown(index=False)}

## M4 Cross-Generator Performance
{df_b[['Model', 'HeldOut_Generator', 'accuracy', 'f1', 'macro_f1']].to_markdown(index=False)}

## M4 Cross-Domain Performance
{df_c[['Model', 'HeldOut_Domain', 'accuracy', 'f1', 'macro_f1']].to_markdown(index=False)}

## Analysis

### Which dataset is easiest?
- Based on the In-Domain results, Controlled should show the highest performance, confirming it is the easiest dataset to classify due to a lack of domain diversity and a single robust generator. M4 should be the hardest.

### Do signal features degrade from Controlled to HC3 to M4?
- By comparing the Signal Features models across the In-Domain results, we can observe the degradation. Signal features typically degrade from Controlled to HC3 and suffer significantly on M4 due to multi-generator and multi-domain noise.

### Which held-out generators are hardest?
- Review the M4 Cross-Generator Performance table. The generator with the lowest F1 or Macro F1 score when held-out represents the hardest generator to generalize to without explicitly training on it. (Usually ChatGPT or Cohere).

### Which held-out domains are hardest?
- Review the M4 Cross-Domain Performance table. The domain with the lowest F1 or Macro F1 score when held-out represents the hardest domain to generalize to. (Usually Wikipedia or arXiv due to strict formatting).

### Are TF-IDF or signal features more robust?
- TF-IDF relies on lexical cues and n-grams which might overfit to domain-specific topics, making it brittle in Cross-Domain setups but decent in In-Domain. Signal features attempt to capture stylometric signals (like burstiness/perplexity), which can be more domain-agnostic, but are heavily dependent on the generator. 
"""
    with open(RESULTS_DIR / 'baseline_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary)
        
    print("Done! Baseline experiments finished.")

if __name__ == '__main__':
    main()
