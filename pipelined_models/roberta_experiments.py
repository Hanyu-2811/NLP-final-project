import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import os

os.environ["WANDB_DISABLED"] = "true"  # Disable wandb to prevent login prompts

# Paths
ROOT_DIR = Path('c:/Users/111/Desktop/Home/NYU/26Spring/NLP/Project')
DATA_DIR = ROOT_DIR / 'data'
RESULTS_DIR = ROOT_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
INDOMAIN_DIR = DATA_DIR / 'unified_benchmark/indomain_splits'
M4_SUBSET_DIR = DATA_DIR / 'm4'

random_seed = 42

# Dataset Class
class AIDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, ids):
        self.encodings = encodings
        self.labels = labels
        self.ids = ids

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Helpers
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
        'macro_f1': f1_score(labels, predictions, average='macro', zero_division=0)
    }

"""These below functions are contributing to the 3 experiments:
1) prepare_dataset(data, tokenizer)
2) *roberta model creation
2) run_experiment(exp_name, train_data, dev_data, test_data, results_list, predictions_list)

"""
def prepare_dataset(data, tokenizer):
    texts = [x['text'] for x in data]
    labels = [x['label'] for x in data]
    ids = [x['id'] for x in data]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    return AIDetectionDataset(encodings, labels, ids)

# create the roberta model:
# return the tokernizer for the model and the model itself
def roberta_model():
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    return tokenizer, model

def run_experiment(exp_name, train_data, dev_data, test_data, results_list, predictions_list):
    print(f"\n{'='*50}\nStarting Experiment: {exp_name}\n{'='*50}")
    
    # tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    # model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    
    tokenizer, model = roberta_model()
    
    train_dataset = prepare_dataset(train_data, tokenizer)
    dev_dataset = prepare_dataset(dev_data, tokenizer)
    test_dataset = prepare_dataset(test_data, tokenizer)
    
    output_dir = RESULTS_DIR / f'roberta_results_{exp_name}'
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=RESULTS_DIR / 'logs',
        logging_steps=50,
        seed=random_seed
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    print("Training model...")
    trainer.train()
    
    # this is preparation for ensemble on development sets
    dev_output = trainer.predict(dev_dataset)
    dev_logits = dev_output.predictions
    dev_prob = softmax(dev_logits)[:,1]
    dev_labels = np.array(dev_dataset.labels)
    
    # this is preparation for ensemble on test sets
    test_output = trainer.predict(test_dataset)
    test_logits = test_output.predictions
    test_prob = softmax(test_logits)[:,1]
    test_labels = np.array(test_dataset.labels)
    
    print("Evaluating on test set...")
    metrics = test_output.metrics
    
    # Store metrics
    metrics_formatted = {
        'experiment': exp_name,
        'accuracy': metrics['test_accuracy'],
        'precision': metrics['test_precision'],
        'recall': metrics['test_recall'],
        'f1': metrics['test_f1'],
        'macro_f1': metrics['test_macro_f1']
    }
    results_list.append(metrics_formatted)
    print(f"Test Results for {exp_name}: {metrics_formatted}")
    
    # Process predictions
    pred_labels = np.argmax(test_logits, axis=-1).tolist()
    
    for i, _id in enumerate(test_dataset.ids):
        predictions_list.append({
            'experiment': exp_name,
            'id': _id,
            'true_label': test_dataset.labels[i],
            'pred_label': pred_labels[i],
            'roberta_prob': float(test_prob[i])
        })
    
    return {
        "trainer": trainer,
        "tokenizer": tokenizer,
        "model": model,
        "dev_probs": dev_prob,
        "dev_labels": dev_labels,
        "test_probs": test_prob,
        "test_labels": test_labels,
        "dev_ids": dev_dataset.ids,
        "test_ids": test_dataset.ids
    }

# prepare roberta datasets
def roberta_datasets():
    results = []
    predictions = []
    
    # Load required data
    print("Loading data...")
    hc3_train = load_json(INDOMAIN_DIR / 'hc3_train.json')
    hc3_dev = load_json(INDOMAIN_DIR / 'hc3_dev.json')
    
    m4_train = load_json(INDOMAIN_DIR / 'm4_train.json')
    m4_dev = load_json(INDOMAIN_DIR / 'm4_dev.json')
    m4_test = load_json(INDOMAIN_DIR / 'm4_test.json')
    
    cg_chatgpt_train = load_json(M4_SUBSET_DIR / 'cross_gen_train_heldout_chatgpt.json')
    cg_chatgpt_test = load_json(M4_SUBSET_DIR / 'cross_gen_test_heldout_chatgpt.json')
    
    # Experiment 3 requires a 90/10 split from the cg_chatgpt_train
    print("Creating dev split for ChatGPT held-out experiment...")
    labels = [x['label'] for x in cg_chatgpt_train]
    cg_train_split, cg_dev_split = train_test_split(
        cg_chatgpt_train, test_size=0.1, stratify=labels, random_state=random_seed
    )
    
    return hc3_dev, hc3_train, m4_test, results, predictions, m4_train, m4_dev, cg_dev_split, cg_train_split, cg_chatgpt_test
    
# all experiments run for roberta
def run_roberta_experiments(hc3_dev, hc3_train, m4_test, results, predictions, m4_train, m4_dev, cg_dev_split, cg_train_split, cg_chatgpt_test):
    # 1. HC3 to M4
    exp1 =run_experiment(
        exp_name='HC3_to_M4',
        train_data=hc3_train,
        dev_data=hc3_dev,
        test_data=m4_test,
        results_list=results,
        predictions_list=predictions
    )
    
    # 2. M4 to M4
    exp2 = run_experiment(
        exp_name='M4_to_M4',
        train_data=m4_train,
        dev_data=m4_dev,
        test_data=m4_test,
        results_list=results,
        predictions_list=predictions
    )
    
    # 3. M4 Cross-Generator (Held-out ChatGPT)
    exp3 = run_experiment(
        exp_name='M4_heldout_chatgpt',
        train_data=cg_train_split,
        dev_data=cg_dev_split,
        test_data=cg_chatgpt_test,
        results_list=results,
        predictions_list=predictions
    )
    
    # Output results
    print("Saving results to CSV...")
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / 'roberta_results.csv', index=False)
    
    print("Saving predictions to JSON...")
    with open(RESULTS_DIR / 'roberta_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
        
    print("All done!")

    return exp1, exp2, exp3
# if __name__ == "__main__":
#     main()
