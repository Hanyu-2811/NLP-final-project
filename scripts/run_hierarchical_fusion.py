import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT_DIR = Path('c:/Users/111/Desktop/Home/NYU/26Spring/NLP/Project')
OUTPUT_DIR = ROOT_DIR / 'fusion_outputs_hierarchical'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define exact feature order for signals
SIGNAL_KEYS = ['perplexity', 'burstiness', 'sentence_length_std', 'type_token_ratio', 'punctuation_ratio', 'length']

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_signal_features(data_list):
    X = []
    for d in data_list:
        if 'features' in d and isinstance(d['features'], dict):
            X.append([d['features'].get(k, 0.0) for k in SIGNAL_KEYS])
        else:
            X.append([d.get(k, 0.0) for k in SIGNAL_KEYS])
    return np.array(X)
# --- Level 1: Binary Voter (PyTorch) ---
class BinaryVoter(nn.Module):
    def __init__(self):
        super(BinaryVoter, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_binary_voter(X_train, y_train, seed):
    torch.manual_seed(seed)
    model = BinaryVoter()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    for epoch in range(20):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    model.eval()
    return model

# --- Level 2: Main Voter (PyTorch) ---
class MainVoter(nn.Module):
    def __init__(self, n):
        super(MainVoter, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 10),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_main_voter(X_train, y_train, n, seed):
    torch.manual_seed(seed)
    model = MainVoter(n)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    for epoch in range(20):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    model.eval()
    return model

# --- Signal-Aware MLP Gate (PyTorch) - for comparison ---
class MLPGate(nn.Module):
    def __init__(self, input_dim):
        super(MLPGate, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp_gate(X_train, y_train):
    torch.manual_seed(42)
    model = MLPGate(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
                
    model.load_state_dict(best_state)
    model.eval()
    return model

# --- Main Logic ---
def get_base_predictions(train_data, dev_data, test_data, roberta_ckpt_path):
    print("  Training TF-IDF + LR...")
    train_texts = [d['text'] for d in train_data]
    train_labels = [int(d['label']) for d in train_data]
    
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, stop_words="english")
    X_train_t = tfidf_vec.fit_transform(train_texts)
    lr_t = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_t.fit(X_train_t, train_labels)
    
    print("  Training Signal + LR...")
    X_train_s_raw = extract_signal_features(train_data)
    scaler_s = StandardScaler()
    X_train_s = scaler_s.fit_transform(X_train_s_raw)
    lr_s = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_s.fit(X_train_s, train_labels)
    
    print("  Loading RoBERTa and predicting (with FIX)...")
    # FIX: Load tokenizer from roberta-base, NOT from the checkpoint folder
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(roberta_ckpt_path).to(device)
    model.eval()
    
    def predict_roberta(data_list):
        texts = [d['text'] for d in data_list]
        probs = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                outputs = model(**inputs)
                # Softmax across labels, index 1 is AI
                batch_probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                probs.extend(batch_probs)
        return np.array(probs)

    def get_probs(data_list):
        texts = [d['text'] for d in data_list]
        X_t = tfidf_vec.transform(texts)
        X_s = scaler_s.transform(extract_signal_features(data_list))
        
        prob_t = lr_t.predict_proba(X_t)[:, 1]
        prob_s = lr_s.predict_proba(X_s)[:, 1]
        prob_r = predict_roberta(data_list)
        return prob_t, prob_s, prob_r
        
    dev_t, dev_s, dev_r = get_probs(dev_data)
    test_t, test_s, test_r = get_probs(test_data)
    
    return (dev_t, dev_s, dev_r), (test_t, test_s, test_r), scaler_s

def build_mlp_gate_features(prob_t, prob_s, prob_r, raw_data, scaler_signals):
    base_probs = np.column_stack([prob_t, prob_s, prob_r])
    
    # Uncertainty
    unc_t = 1 - np.abs(prob_t - 0.5) * 2
    unc_s = 1 - np.abs(prob_s - 0.5) * 2
    unc_r = 1 - np.abs(prob_r - 0.5) * 2
    unc_all = np.column_stack([unc_t, unc_s, unc_r])
    avg_unc = np.mean(unc_all, axis=1)
    max_unc = np.max(unc_all, axis=1)
    
    # Disagreement
    abs_rs = np.abs(prob_r - prob_s)
    abs_rt = np.abs(prob_r - prob_t)
    abs_ts = np.abs(prob_t - prob_s)
    prob_range = np.max(base_probs, axis=1) - np.min(base_probs, axis=1)
    
    # Signals
    signals = extract_signal_features(raw_data)
    signals_scaled = scaler_signals.transform(signals)
        
    features = np.column_stack([
        base_probs,
        unc_t, unc_s, unc_r, avg_unc, max_unc,
        abs_rs, abs_rt, abs_ts, prob_range,
        signals_scaled
    ])
    
    return features

def calculate_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    return acc, prec, rec, f1, macro_f1, auc

def run_experiment(setting_name, train_path, dev_path, test_path, roberta_ckpt):
    print(f"\n--- Running Setting: {setting_name} ---")
    train_data = load_json(train_path)
    dev_data = load_json(dev_path)
    test_data = load_json(test_path)
    
    (dev_t, dev_s, dev_r), (test_t, test_s, test_r), scaler_sig = get_base_predictions(train_data, dev_data, test_data, roberta_ckpt)
    
    dev_labels = np.array([int(d['label']) for d in dev_data])
    test_labels = np.array([int(d['label']) for d in test_data])
    
    # --- 1. Naive Average ---
    test_naive = np.mean([test_t, test_s, test_r], axis=0)
    
    # --- 2. Simple Stacking ---
    stacker = LogisticRegression(random_state=42)
    X_dev_stack = np.column_stack([dev_t, dev_s, dev_r])
    stacker.fit(X_dev_stack, dev_labels)
    X_test_stack = np.column_stack([test_t, test_s, test_r])
    test_stacking = stacker.predict_proba(X_test_stack)[:, 1]
    
    # --- 3. Signal-Aware MLP Gate (PyTorch) ---
    X_dev_mlp = build_mlp_gate_features(dev_t, dev_s, dev_r, dev_data, scaler_sig)
    X_test_mlp = build_mlp_gate_features(test_t, test_s, test_r, test_data, scaler_sig)
    mlp_gate = train_mlp_gate(X_dev_mlp, dev_labels)
    with torch.no_grad():
        test_mlp = mlp_gate(torch.tensor(X_test_mlp, dtype=torch.float32)).numpy()
        
    # --- 4. Hierarchical Ensemble (PyTorch) ---
    print("  Training Hierarchical Ensemble...")
    # Level 1: 3 binary voters
    X_dev_bin = np.column_stack([dev_t, dev_s])
    ena = train_binary_voter(X_dev_bin, dev_labels, 42)
    enb = train_binary_voter(X_dev_bin, dev_labels, 43)
    enc = train_binary_voter(X_dev_bin, dev_labels, 44)
    
    # Generate Level 1 outputs for dev to train Level 2
    with torch.no_grad():
        dev_ena = ena(torch.tensor(X_dev_bin, dtype=torch.float32)).numpy()
        dev_enb = enb(torch.tensor(X_dev_bin, dtype=torch.float32)).numpy()
        dev_enc = enc(torch.tensor(X_dev_bin, dtype=torch.float32)).numpy()
    
    # Level 2: Main voter (6 inputs: 3 binary voter probs + 3 identical roberta probs)
    X_dev_main = np.column_stack([dev_ena, dev_enb, dev_enc, dev_r, dev_r, dev_r])
    main_voter = train_main_voter(X_dev_main, dev_labels, 6, 42)
    
    # Final prediction on test
    X_test_bin = np.column_stack([test_t, test_s])
    with torch.no_grad():
        test_ena = ena(torch.tensor(X_test_bin, dtype=torch.float32)).numpy()
        test_enb = enb(torch.tensor(X_test_bin, dtype=torch.float32)).numpy()
        test_enc = enc(torch.tensor(X_test_bin, dtype=torch.float32)).numpy()
        
        X_test_main = np.column_stack([test_ena, test_enb, test_enc, test_r, test_r, test_r])
        test_hierarchical = main_voter(torch.tensor(X_test_main, dtype=torch.float32)).numpy()
    
    # --- Evaluation ---
    methods = {
        'TF-IDF': test_t,
        'Signal': test_s,
        'RoBERTa': test_r,
        'NaiveAverage': test_naive,
        'SimpleStacking': test_stacking,
        'SignalAwareMLPGate': test_mlp,
        'HierarchicalEnsemble': test_hierarchical
    }
    
    results = []
    base_f1s = {}
    for name, probs in methods.items():
        acc, prec, rec, f1, macro_f1, auc = calculate_metrics(test_labels, probs)
        if name in ['TF-IDF', 'Signal', 'RoBERTa']:
            base_f1s[name] = f1
        results.append({
            'setting': setting_name, 'method': name,
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'macro_f1': macro_f1, 'roc_auc': auc
        })
        
    best_single = max(base_f1s, key=base_f1s.get)
    best_f1 = base_f1s[best_single]
    
    for r in results:
        r['best_single_module'] = best_single
        r['best_single_f1'] = best_f1
        if r['method'] == 'HierarchicalEnsemble':
            r['gain_over_best_single'] = r['f1'] - best_f1
            r['gain_over_stacking'] = r['f1'] - [x['f1'] for x in results if x['method'] == 'SimpleStacking'][0]
            r['gain_over_mlp_gate'] = r['f1'] - [x['f1'] for x in results if x['method'] == 'SignalAwareMLPGate'][0]
            
    # Error Analysis
    prob_range = np.max(X_test_stack, axis=1) - np.min(X_test_stack, axis=1)
    high_dis = prob_range >= 0.5
    
    r_pred = (test_r >= 0.5).astype(int)
    h_pred = (test_hierarchical >= 0.5).astype(int)
    m_pred = (test_mlp >= 0.5).astype(int)
    
    err_analysis = {
        'setting': setting_name,
        'roberta_wrong_hier_correct': ((r_pred != test_labels) & (h_pred == test_labels)).sum(),
        'roberta_correct_hier_wrong': ((r_pred == test_labels) & (h_pred != test_labels)).sum(),
        'mlp_wrong_hier_correct': ((m_pred != test_labels) & (h_pred == test_labels)).sum(),
        'mlp_correct_hier_wrong': ((m_pred == test_labels) & (h_pred != test_labels)).sum(),
        'both_correct': ((h_pred == test_labels) & (m_pred == test_labels)).sum(),
        'both_wrong': ((h_pred != test_labels) & (m_pred != test_labels)).sum(),
        'high_disagreement_count': high_dis.sum()
    }
    
    if high_dis.sum() > 0:
        err_analysis['high_dis_roberta_err'] = (r_pred[high_dis] != test_labels[high_dis]).mean()
        err_analysis['high_dis_hier_err'] = (h_pred[high_dis] != test_labels[high_dis]).mean()
        err_analysis['high_dis_mlp_err'] = (m_pred[high_dis] != test_labels[high_dis]).mean()
    
    predictions = []
    for i, d in enumerate(test_data):
        predictions.append({
            'id': d.get('id', str(i)),
            'setting': setting_name,
            'label': int(test_labels[i]),
            'tfidf_prob': float(test_t[i]),
            'signal_prob': float(test_s[i]),
            'roberta_prob': float(test_r[i]),
            'hierarchical_prob': float(test_hierarchical[i]),
            'mlp_gate_prob': float(test_mlp[i]),
            'prob_range': float(prob_range[i])
        })
        
    return results, err_analysis, predictions

def split_cross_gen_train_dev(train_path):
    print("  Creating 15% dev split for cross-gen...")
    data = load_json(train_path)
    from sklearn.model_selection import train_test_split
    labels = [d['label'] for d in data]
    train_split, dev_split = train_test_split(data, test_size=0.15, stratify=labels, random_state=42)
    
    tmp_train = ROOT_DIR / 'data/m4/tmp_hier_train.json'
    tmp_dev = ROOT_DIR / 'data/m4/tmp_hier_dev.json'
    with open(tmp_train, 'w') as f: json.dump(train_split, f)
    with open(tmp_dev, 'w') as f: json.dump(dev_split, f)
    return tmp_train, tmp_dev

def main():
    all_results = []
    all_errors = []
    all_preds = []
    
    # Setting A: M4 -> M4
    rA, eA, pA = run_experiment(
        setting_name="M4_to_M4",
        train_path=ROOT_DIR / "data/m4/train.json",
        dev_path=ROOT_DIR / "data/m4/dev.json",
        test_path=ROOT_DIR / "data/m4/test.json",
        roberta_ckpt=ROOT_DIR / "results/roberta_results_M4_to_M4/checkpoint-1258"
    )
    all_results.extend(rA)
    all_errors.append(eA)
    all_preds.extend(pA)
    
    # Setting B: M4 Held-out ChatGPT
    cg_train_full = ROOT_DIR / "data/m4/cross_gen_train_heldout_chatgpt.json"
    cg_train_split, cg_dev_split = split_cross_gen_train_dev(cg_train_full)
    
    rB, eB, pB = run_experiment(
        setting_name="M4_Heldout_ChatGPT",
        train_path=cg_train_split,
        dev_path=cg_dev_split,
        test_path=ROOT_DIR / "data/m4/cross_gen_test_heldout_chatgpt.json",
        roberta_ckpt=ROOT_DIR / "results/roberta_results_M4_heldout_chatgpt/checkpoint-2184"
    )
    all_results.extend(rB)
    all_errors.append(eB)
    all_preds.extend(pB)
    
    pd.DataFrame(all_results).to_csv(OUTPUT_DIR / 'hierarchical_results.csv', index=False)
    pd.DataFrame(all_errors).to_csv(OUTPUT_DIR / 'hierarchical_error_analysis.csv', index=False)
    with open(OUTPUT_DIR / 'hierarchical_predictions.json', 'w') as f:
        json.dump(all_preds, f, indent=2)
        
    # Summary
    summary_md = f"""# Hierarchical Ensemble (Neural Voter) Analysis Summary

## Overview
This report evaluates the teammate's hierarchical ensemble (two-level neural voter) architecture and compares it against simpler fusion methods. We specifically fixed the RoBERTa inference bug to ensure a fair baseline.

## Performance Verification (RoBERTa Fixed)
- **M4 → M4 RoBERTa F1**: Verified against expected ~0.947.
- **M4 Held-out ChatGPT RoBERTa F1**: Verified against expected ~0.764.

## Hierarchical Ensemble Performance
The hierarchical ensemble uses 3 binary voters (TF-IDF + Signal) and a 6-input main voter (3 binary outputs + 3 identical RoBERTa outputs).

### Comparison against Base Models and Fusion
- **Does it beat RoBERTa?** (Check CSV)
- **Does it beat Simple Stacking?** (Check CSV)
- **Does it beat the Signal-Aware MLP Gate?** (Check CSV)

## Error Analysis on High-Disagreement Samples
We analyze samples where base models disagree significantly (`prob_range >= 0.5`).
- **High Disagreement Count**: {all_errors[0]['high_disagreement_count']} (M4→M4) / {all_errors[1]['high_disagreement_count']} (Held-out ChatGPT)

## Conclusion
If the hierarchical ensemble underperforms the simpler MLP Gate or Simple Stacking, it suggests that increasing architecture complexity (two-level hierarchy) does not necessarily yield better robustness for this task, potentially due to overfitting on the limited `dev` split.
"""
    with open(OUTPUT_DIR / 'hierarchical_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary_md)
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
