import json
import os
import random
import string
import hashlib
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

# Configuration
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "m4"
OUTPUT_DIR = ROOT_DIR / "data" / "m4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ['wikipedia', 'reddit', 'wikihow', 'peerread', 'arxiv']
DOMAIN_MAPPING = {
    'wikipedia': 'wikipedia',
    'reddit': 'reddit_eli5',
    'wikihow': 'wikihow',
    'peerread': 'peerread',
    'arxiv': 'arxiv'
}
GENERATORS = ['davinci', 'chatgpt', 'cohere', 'dolly', 'bloomz']
GEN_MAPPING = {
    'davinci': 'davinci003',
    'chatgpt': 'chatgpt',
    'cohere': 'cohere',
    'dolly': 'dolly',
    'bloomz': 'bloomz',
    'human': 'human'
}

MIN_WORDS = 80
MAX_WORDS = 1500
TARGET_AI_PER_GEN = 350 # For abundant domains
BATCH_SIZE = 32

random.seed(42)

# --- Feature Extraction Setup ---
print("Loading model for perplexity computation...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
model.eval()

def get_word_count(text):
    return len(text.split())

def calculate_batch_perplexity(texts):
    if not texts:
        return []
    # Truncate at 1024 tokens to speed up and fit in GPT-2 context
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size(0), shift_labels.size(1))
        
        valid_lengths = (shift_labels != -100).sum(dim=1).float()
        # Avoid division by zero
        valid_lengths[valid_lengths == 0] = 1.0
        
        valid_loss = loss.sum(dim=1) / valid_lengths
        ppl = torch.exp(valid_loss)
        
        results = []
        for i in range(loss.size(0)):
            v_len = int(valid_lengths[i].item())
            if v_len > 0:
                v_losses = loss[i, :v_len]
                burstiness = float(torch.std(v_losses).item()) if v_len > 1 else 0.0
                results.append((float(ppl[i].item()), burstiness))
            else:
                results.append((0.0, 0.0))
        
    return results

# --- Data Loading ---
print("Loading and filtering data...")
raw_data = {dom: {"human": [], "machine": {gen: [] for gen in GENERATORS}} for dom in DOMAINS}
seen_hashes = set()

def get_hash(t):
    return hashlib.md5(t.encode('utf-8')).hexdigest()

for domain in DOMAINS:
    for gen in GENERATORS:
        file_path = None
        for f in os.listdir(DATA_DIR):
            if f.endswith('.jsonl') and domain.lower() in f.lower() and gen.lower() in f.lower():
                file_path = DATA_DIR / f
                break
        
        if not file_path:
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Human
                    htext = data.get('human_text') or data.get('human_reviews') or ''
                    if isinstance(htext, list): htext = ' '.join(htext)
                    if isinstance(htext, str) and htext.strip():
                        htext = ' '.join(htext.split()) # normalize whitespace
                        h_hash = get_hash(htext)
                        if h_hash not in seen_hashes:
                            hw = len(htext.split())
                            if MIN_WORDS <= hw <= MAX_WORDS:
                                raw_data[domain]['human'].append(htext)
                                seen_hashes.add(h_hash)
                    
                    # Machine
                    mtext = data.get('machine_text')
                    if not mtext:
                        for k in data.keys():
                            if (gen.lower() in k.lower() or gen.lower().replace('z', '') in k.lower()) and 'reviews' in k.lower():
                                mtext = data[k]
                                break
                    if isinstance(mtext, list): mtext = ' '.join(mtext)
                    if isinstance(mtext, str) and mtext.strip():
                        mtext = ' '.join(mtext.split())
                        m_hash = get_hash(mtext)
                        if m_hash not in seen_hashes:
                            mw = len(mtext.split())
                            if MIN_WORDS <= mw <= MAX_WORDS:
                                raw_data[domain]['machine'][gen].append(mtext)
                                seen_hashes.add(m_hash)
                except Exception as e:
                    pass

# --- Balancing ---
print("Balancing data...")
final_samples = []

for domain in DOMAINS:
    human_texts = raw_data[domain]['human']
    ai_counts = {gen: len(raw_data[domain]['machine'][gen]) for gen in GENERATORS}
    
    if domain == 'peerread':
        # Low resource handling: use max available balanced
        min_ai = min(ai_counts.values()) if ai_counts else 0
        ai_target = min_ai
    else:
        # Abundant handling
        min_ai = min(ai_counts.values()) if ai_counts else 0
        ai_target = min(TARGET_AI_PER_GEN, min_ai)
        
    human_target = ai_target * len(GENERATORS)
    
    # Adjust if not enough human texts
    if len(human_texts) < human_target:
        human_target = len(human_texts)
        ai_target = human_target // len(GENERATORS)
        human_target = ai_target * len(GENERATORS) # keep balance exact
        
    print(f"Domain: {domain} | Target AI per gen: {ai_target} | Target Human: {human_target}")
    
    # Sample and add to final
    random.shuffle(human_texts)
    for i, t in enumerate(human_texts[:human_target]):
        final_samples.append({
            "id": f"{DOMAIN_MAPPING[domain]}_human_{i}",
            "text": t,
            "label": 0,
            "dataset": "M4",
            "domain": DOMAIN_MAPPING[domain],
            "generator": "human"
        })
        
    for gen in GENERATORS:
        gen_texts = raw_data[domain]['machine'][gen]
        random.shuffle(gen_texts)
        for i, t in enumerate(gen_texts[:ai_target]):
            final_samples.append({
                "id": f"{DOMAIN_MAPPING[domain]}_{GEN_MAPPING[gen]}_{i}",
                "text": t,
                "label": 1,
                "dataset": "M4",
                "domain": DOMAIN_MAPPING[domain],
                "generator": GEN_MAPPING[gen]
            })

print(f"Total samples before feature extraction: {len(final_samples)}")

# --- Feature Extraction ---
print("Computing features in batches...")

for i in tqdm(range(0, len(final_samples), BATCH_SIZE)):
    batch = final_samples[i:i+BATCH_SIZE]
    texts = [s['text'] for s in batch]
    
    # Text-level perplexity and burstiness
    ppl_burst_results = calculate_batch_perplexity(texts)
    
    for j, sample in enumerate(batch):
        text = sample['text']
        words = text.split()
        total_words = len(words)
        unique_words = len(set(words))
        ttr = unique_words / total_words if total_words > 0 else 0
        
        total_chars = len(text)
        punct_chars = sum(1 for c in text if c in string.punctuation)
        punct_ratio = punct_chars / total_chars if total_chars > 0 else 0
        
        sentences = sent_tokenize(text)
        sent_lengths = [len(s.split()) for s in sentences]
        sl_std = np.std(sent_lengths) if len(sent_lengths) > 0 else 0
        
        ppl, burstiness = ppl_burst_results[j]
            
        sample.update({
            "length": total_words,
            "type_token_ratio": ttr,
            "punctuation_ratio": punct_ratio,
            "sentence_length_std": float(sl_std),
            "perplexity": ppl,
            "burstiness": burstiness
        })

# --- Splits ---
print("Creating splits...")
def save_json(data, name):
    with open(OUTPUT_DIR / f"{name}.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

random.shuffle(final_samples)

train_idx = int(0.7 * len(final_samples))
dev_idx = int(0.85 * len(final_samples))

train_data = final_samples[:train_idx]
dev_data = final_samples[train_idx:dev_idx]
test_data = final_samples[dev_idx:]

save_json(train_data, "train")
save_json(dev_data, "dev")
save_json(test_data, "test")
save_json(final_samples, "full_dataset")

# Cross-generator splits
for gen in GENERATORS:
    gen_name = GEN_MAPPING[gen]
    train_split = [s for s in final_samples if s['generator'] != gen_name]
    test_split = [s for s in final_samples if s['generator'] in [gen_name, 'human']]
    save_json(train_split, f"cross_gen_train_heldout_{gen_name}")
    save_json(test_split, f"cross_gen_test_heldout_{gen_name}")

# Cross-domain splits
for domain in DOMAINS:
    dom_name = DOMAIN_MAPPING[domain]
    train_split = [s for s in final_samples if s['domain'] != dom_name]
    test_split = [s for s in final_samples if s['domain'] == dom_name]
    save_json(train_split, f"cross_domain_train_heldout_{dom_name}")
    save_json(test_split, f"cross_domain_test_heldout_{dom_name}")

# --- Summary Report ---
print("Generating summary report...")
summary = f"# M4 Dataset Subset Summary\n\n"
summary += f"**Total Samples**: {len(final_samples)}\n\n"

# Domain Dist
summary += "## Samples by Domain\n"
for dom in DOMAIN_MAPPING.values():
    count = sum(1 for s in final_samples if s['domain'] == dom)
    note = " *(Stress-test Domain)*" if dom == 'peerread' else ""
    summary += f"- **{dom}**: {count}{note}\n"

# Gen Dist
summary += "\n## Samples by Generator\n"
for gen in GEN_MAPPING.values():
    count = sum(1 for s in final_samples if s['generator'] == gen)
    summary += f"- **{gen}**: {count}\n"

# Length stats
lengths = [s['length'] for s in final_samples]
summary += "\n## Length Statistics\n"
summary += f"- **Average Length**: {np.mean(lengths):.2f} words\n"
summary += f"- **Minimum Length**: {np.min(lengths)} words\n"
summary += f"- **Maximum Length**: {np.max(lengths)} words\n"

# Warnings
summary += "\n## Imbalance Warnings & Notes\n"
summary += "- `peerread` is a low-resource domain in this filtering scheme and serves as a stress test.\n"
summary += "- Data was strictly balanced (Human ≈ AI) within each domain to preserve representation.\n"

with open(OUTPUT_DIR / "dataset_summary.md", "w", encoding='utf-8') as f:
    f.write(summary)

print("Done!")
