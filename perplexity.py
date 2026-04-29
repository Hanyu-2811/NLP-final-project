import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from ..evaluation import evaluate_predictions

class ModelB:
    """
    Perplexity + Burstiness + Logistic Regression
    Uses GPT-2 to calculate statistical features of the text.
    """
    def __init__(self, model_id="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_id} for Baseline B on {self.device}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        # Logistic Regression will use the extracted features
        self.classifier = LogisticRegression(random_state=42)

    def calculate_ppl(self, text):
        """
        Calculates the perplexity of a given text using GPT-2.
        """
        if not text.strip():
            return 0.0
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        
        # If text is too short, GPT-2 might struggle; return a neutral value
        if inputs["input_ids"].size(1) <= 1:
            return 0.0

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            ppl = torch.exp(loss).item()
        return ppl

    def extract_features(self, texts):
        """
        Extracts real Perplexity and Burstiness features.
        """
        features = []
        ppl = [] # probability
        for text in tqdm(texts, desc="Extracting PPL features"):
            # 1. Total Perplexity
            total_ppl = self.calculate_ppl(text)
            
            # 2. Burstiness (Standard Deviation of sentence-level PPL)
            # Basic sentence splitting by common punctuation
            sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if len(s.strip()) > 10]
            
            if len(sentences) > 1:
                sentence_ppls = [self.calculate_ppl(s) for s in sentences]
                burstiness = np.std(sentence_ppls)
            else:
                burstiness = 0.0
                
            features.append([total_ppl, burstiness])
            ppl.append(total_ppl) # add to ppl
            
        return ppl, np.array(features)

    def train(self, train_df):
        ppl, X_train = self.extract_features(train_df['text'])
        y_train = train_df['label']
        self.classifier.fit(X_train, y_train)
        return ppl

    def predict(self, df):
        X = self.extract_features(df['text'])
        return self.classifier.predict(X)

    def run(self, train_df, predict_df):
        print("Running Baseline B (GPT-2 Perplexity + Burstiness + LR)...")
        ppl = self.train(train_df) # also get probabilities
        preds = self.predict(predict_df)
        # metrics = evaluate_predictions(test_df['label'], preds)
        # return metrics
        
        return 1/ppl, preds