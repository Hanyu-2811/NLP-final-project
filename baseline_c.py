import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import numpy as np
from ..evaluation import evaluate_predictions

class BaselineC:
    """
    Baseline C: Mean-pooled RoBERTa embeddings + Logistic Regression
    RoBERTa is kept frozen (no fine-tuning).
    """
    def __init__(self, model_name='roberta-base', device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} for Baseline C on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval() 
        
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)

    def extract_embeddings(self, texts, batch_size=8):
        """
        Extracts mean-pooled embeddings from frozen RoBERTa.
        """
        all_embeddings = []
        
        with torch.no_grad(): 
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting RoBERTa embeddings"):
                # Handle potential list or series input
                batch_texts = texts[i:i+batch_size]
                if hasattr(batch_texts, 'tolist'):
                    batch_texts = batch_texts.tolist()
                
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                outputs = self.model(**inputs)
                
                # Mean pool across the sequence length dimension
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)

    def train(self, train_df):
        X_train = self.extract_embeddings(train_df['text'])
        y_train = train_df['label']
        self.classifier.fit(X_train, y_train)

    def predict(self, df):
        X = self.extract_embeddings(df['text'])
        return self.classifier.predict(X)

    def run(self, train_df, test_df):
        print("Running Baseline C (RoBERTa Frozen Embeddings + LR)...")
        self.train(train_df)
        preds = self.predict(test_df)
        metrics = evaluate_predictions(test_df['label'], preds)
        return metrics
