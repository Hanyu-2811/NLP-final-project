import json
import os
import random
import string
import math
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
from nltk.tokenize import sent_tokenize, word_tokenize

# Configurations
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "unified_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTROLLED_PATH = DATA_DIR / "Controlled" / "final" / "dataset_controlled_with_signals.jsonl"
HC3_TRAIN_PATH = DATA_DIR / "HC3" / "train (1).json"
HC3_TEST_PATH = DATA_DIR / "HC3" / "test.json"
M4_PATH = DATA_DIR / "m4" / "full_dataset.json"

random.seed(42)

# Feature Extraction Helpers
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None

def init_lm():
    global tokenizer, model
    if model is None:
        print(f"Loading GPT-2 for perplexity computation on {device}...")
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        model.eval()

def compute_lexical_and_lm_features(text):
    words = [w for w in word_tokenize(text) if w.isalnum()]
    
    total_words = len(words)
    unique_words = len(set(w.lower() for w in words))
    ttr = unique_words / total_words if total_words > 0 else 0
    
    total_chars = len(text)
    punct_chars = sum(1 for c in text if c in string.punctuation)
    punct_ratio = punct_chars / total_chars if total_chars > 0 else 0
    
    sentences = sent_tokenize(text)
    sent_lengths = [len([w for w in word_tokenize(s) if w.isalnum()]) for s in sentences]
    sl_std = float(np.std(sent_lengths)) if len(sent_lengths) > 0 else 0
    avg_sl = float(np.mean(sent_lengths)) if len(sent_lengths) > 0 else 0
    burstiness_lexical = (sl_std / avg_sl) if avg_sl > 0 else 0

    # LM Features
    init_lm()
    encodings = tokenizer(text, return_tensors='pt')
    seq_len = encodings.input_ids.size(1)
    
    max_length = 1024
    input_ids = encodings.input_ids
    if seq_len > max_length:
        input_ids = input_ids[:, :max_length]
        
    input_ids = input_ids.to(device)
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss
        
    neg_log_likelihood = loss.item()
    ppl = math.exp(neg_log_likelihood)

    # Note: To match Controlled, burstiness = sl_std / avg_sl, but to match M4 it's std of token loss. 
    # M4 has burstiness > 1 (e.g. 3.19), Controlled has ~0.48. Since the signal baseline probably expects them to be somewhat consistent, 
    # we'll provide both or just use one. Here we compute lexical burstiness. Let's just output burstiness as sl_std / avg_sl.
    
    return {
        "length": total_words, # Using word count
        "type_token_ratio": round(ttr, 4),
        "punctuation_ratio": round(punct_ratio, 4),
        "sentence_length_std": round(sl_std, 4),
        "perplexity": round(ppl, 4),
        "burstiness": round(burstiness_lexical, 4)
    }

# Data Loading & Standardization
def load_controlled():
    print("Loading Controlled Dataset...")
    data = []
    with open(CONTROLLED_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                data.append({
                    "id": record['id'],
                    "text": record['text'],
                    "label": int(record['label']),
                    "dataset": "Controlled",
                    "dataset_type": "controlled",
                    "domain": record.get('domain', 'essay'),
                    "generator": record.get('generation_model', 'human') if record.get('label') == 1 else 'human',
                    "length": record.get('word_count', len(record['text'].split())),
                    "features": {
                        "perplexity": record.get('perplexity'),
                        "burstiness": record.get('burstiness'),
                        "sentence_length_std": record.get('sentence_length_std'),
                        "type_token_ratio": record.get('type_token_ratio'),
                        "punctuation_ratio": record.get('punctuation_ratio')
                    }
                })
    return data

def load_hc3():
    print("Loading and extracting features for HC3 Dataset...")
    raw_data = []
    for path in [HC3_TRAIN_PATH, HC3_TEST_PATH]:
        with open(path, 'r', encoding='utf-8') as f:
            raw_data.extend(json.load(f))
            
    data = []
    # Using a sample of HC3 if testing, but here we process all 4000
    for record in tqdm(raw_data, desc="HC3 Processing"):
        features = compute_lexical_and_lm_features(record['text'])
        data.append({
            "id": record.get('id', f"hc3_{len(data)}"),
            "text": record['text'],
            "label": int(record['label']),
            "dataset": "HC3",
            "dataset_type": "semi_realistic",
            "domain": "QA",
            "generator": "chatgpt" if int(record['label']) == 1 else "human",
            "length": features['length'],
            "features": {
                "perplexity": features['perplexity'],
                "burstiness": features['burstiness'],
                "sentence_length_std": features['sentence_length_std'],
                "type_token_ratio": features['type_token_ratio'],
                "punctuation_ratio": features['punctuation_ratio']
            }
        })
    return data

def load_m4():
    print("Loading M4 Dataset...")
    data = []
    with open(M4_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        for record in raw_data:
            data.append({
                "id": record['id'],
                "text": record['text'],
                "label": int(record['label']),
                "dataset": "M4",
                "dataset_type": "real_world",
                "domain": record['domain'],
                "generator": record['generator'],
                "length": record.get('length', len(record['text'].split())),
                "features": {
                    "perplexity": record.get('perplexity'),
                    "burstiness": record.get('burstiness'),
                    "sentence_length_std": record.get('sentence_length_std'),
                    "type_token_ratio": record.get('type_token_ratio'),
                    "punctuation_ratio": record.get('punctuation_ratio')
                }
            })
    return data

def save_json(data, filename):
    with open(OUTPUT_DIR / filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    controlled_data = load_controlled()
    hc3_data = load_hc3()
    m4_data = load_m4()
    
    # Merge Datasets
    print("Merging datasets...")
    combined_data = controlled_data + hc3_data + m4_data
    save_json(combined_data, "combined_dataset.json")
    
    # Create Combined Training Datasets
    print("Creating combined training datasets...")
    
    # 1. combined_balanced_small (600 from each)
    def sample_balanced(dataset_records, num_samples):
        humans = [r for r in dataset_records if r['label'] == 0]
        ais = [r for r in dataset_records if r['label'] == 1]
        half = num_samples // 2
        return random.sample(humans, min(half, len(humans))) + random.sample(ais, min(half, len(ais)))
        
    combined_balanced_small = []
    combined_balanced_small.extend(sample_balanced(controlled_data, 600))
    combined_balanced_small.extend(sample_balanced(hc3_data, 600))
    combined_balanced_small.extend(sample_balanced(m4_data, 600))
    random.shuffle(combined_balanced_small)
    save_json(combined_balanced_small, "combined_balanced_small.json")
    
    # 2. combined_full_weighted
    # Keep `source_dataset` which is already in `dataset`.
    combined_full_weighted = combined_data.copy()
    random.shuffle(combined_full_weighted)
    save_json(combined_full_weighted, "combined_full_weighted.json")
    
    # Create Cross-Dataset Evaluation Splits
    print("Creating cross-dataset evaluation splits...")
    save_json(controlled_data, "train_Controlled_test_HC3_train.json")
    save_json(hc3_data, "train_Controlled_test_HC3_test.json")
    
    save_json(controlled_data, "train_Controlled_test_M4_train.json")
    save_json(m4_data, "train_Controlled_test_M4_test.json")
    
    save_json(hc3_data, "train_HC3_test_M4_train.json")
    save_json(m4_data, "train_HC3_test_M4_test.json")
    
    save_json(m4_data, "train_M4_test_HC3_train.json")
    save_json(hc3_data, "train_M4_test_HC3_test.json")
    
    save_json(m4_data, "train_M4_test_Controlled_train.json")
    save_json(controlled_data, "train_M4_test_Controlled_test.json")
    
    print("Generating report markdown...")
    # Generate the report based on these
    report = """# Unified Benchmark Dataset Report

## Dataset Summary
- **Controlled Dataset**: {} samples (Easy / controlled single-generator setting)
- **HC3 Dataset**: {} samples (Medium / semi-realistic QA setting)
- **M4 Dataset**: {} samples (Hard / multi-domain multi-generator setting)
- **Total Combined**: {} samples

## Distribution Analysis
### Label Balance
- **Controlled**: {} Human, {} AI
- **HC3**: {} Human, {} AI
- **M4**: {} Human, {} AI

### Domain Diversity
- **Controlled**: {}
- **HC3**: QA
- **M4**: {}

### Generator Diversity
- **Controlled**: human, llm
- **HC3**: human, chatgpt
- **M4**: {}

## Expected Experimental Outcomes
- The Controlled dataset provides a clean signal, making it the easiest benchmark where signal-based models should perform best.
- The HC3 dataset introduces a moderate distribution shift with longer and more conversational QA texts. Signal models may see a performance drop.
- The M4 dataset presents the hardest challenge due to its extreme multi-domain and multi-generator nature, causing significant feature overlap between Human and AI.
- Transformer-based models are expected to be more robust across the cross-dataset generalizations than lexical or pure signal-based baseline models.

""".format(
        len(controlled_data), len(hc3_data), len(m4_data), len(combined_data),
        sum(1 for x in controlled_data if x['label']==0), sum(1 for x in controlled_data if x['label']==1),
        sum(1 for x in hc3_data if x['label']==0), sum(1 for x in hc3_data if x['label']==1),
        sum(1 for x in m4_data if x['label']==0), sum(1 for x in m4_data if x['label']==1),
        ", ".join(set(x['domain'] for x in controlled_data)),
        ", ".join(set(x['domain'] for x in m4_data)),
        ", ".join(set(x['generator'] for x in m4_data))
    )
    
    with open(OUTPUT_DIR / "combined_dataset_report.md", "w", encoding='utf-8') as f:
        f.write(report)
        
    print(f"Done! Files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
