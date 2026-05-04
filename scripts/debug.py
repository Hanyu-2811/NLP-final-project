import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Check paths
m4_test_path1 = "data/m4/test.json"
m4_test_path2 = "data/unified_benchmark/indomain_splits/m4_test.json"
try:
    d1 = load_json(m4_test_path1)
    d2 = load_json(m4_test_path2)
    print(f"data/m4/test.json has {len(d1)} samples")
    print(f"indomain_splits/m4_test.json has {len(d2)} samples")
    
    match = True
    for i in range(min(10, len(d1))):
        if d1[i].get('id') != d2[i].get('id'):
            match = False
    print(f"First 10 IDs match: {match}")
    
    # Let's check features in d1
    print("Features schema in data/m4/test.json:")
    sample = d1[0]
    print(f"Has 'features' key: {'features' in sample}")
    if 'features' in sample:
        print(f"Keys in features: {sample['features'].keys()}")
    print(f"Top-level keys: {sample.keys()}")
    
    zero_count = 0
    for d in d1:
        feats = d.get('features', {})
        if not feats:
            zero_count += 1
    print(f"Rows with missing 'features' dict: {zero_count} out of {len(d1)}")
    
except Exception as e:
    print(f"Error loading datasets: {e}")

# Check RoBERTa inference
ckpt = "results/roberta_results_M4_to_M4/checkpoint-1258"
print(f"\nChecking checkpoint: {ckpt}")
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(ckpt)
model.eval()

# Check label mapping
print(f"model.config.id2label: {model.config.id2label}")
print(f"model.config.label2id: {model.config.label2id}")

# Run inference on first 5 samples
test_samples = d1[:5]
texts = [d['text'] for d in test_samples]
labels = [int(d['label']) for d in test_samples]

inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Input IDs for sample 0: {inputs['input_ids'][0][:20]}")
print(f"Input IDs for sample 1: {inputs['input_ids'][1][:20]}")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

for i in range(5):
    print(f"\nSample {i}:")
    print(f"ID: {test_samples[i].get('id')}")
    print(f"True label: {labels[i]}")
    print(f"Logits: {logits[i].tolist()}")
    print(f"Softmax: {probs[i].tolist()}")
    print(f"Predicted class (argmax): {torch.argmax(probs[i]).item()}")

# Compare with results/roberta_predictions.json
try:
    old_preds = load_json("results/roberta_predictions.json")
    old_m4 = [p for p in old_preds if p['experiment'] == 'M4_to_M4']
    print(f"\nOld M4_to_M4 predictions count: {len(old_m4)}")
    
    if old_m4:
        for i in range(5):
            print(f"Old Pred {i} ID: {old_m4[i]['id']}, Prob: {old_m4[i]['roberta_prob']}, Label: {old_m4[i]['label']}")
            
        probs_array = [p['roberta_prob'] for p in old_m4]
        print(f"Old Probs - Mean: {np.mean(probs_array):.4f}, Min: {np.min(probs_array):.4f}, Max: {np.max(probs_array):.4f}, Std: {np.std(probs_array):.4f}")
        pred_pos_rate = sum(p >= 0.5 for p in probs_array) / len(probs_array)
        print(f"Old Predicted positive rate: {pred_pos_rate:.4f}")
except Exception as e:
    print(f"Error loading old predictions: {e}")
