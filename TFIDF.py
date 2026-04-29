from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from ..evaluation import evaluate_predictions

class ModelA:
    """
    TF-IDF + Logistic Regression
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression(random_state=42)

    def train(self, train_df):
        X_train = self.vectorizer.fit_transform(train_df['text'])
        y_train = train_df['label']
        self.model.fit(X_train, y_train)

    def predict(self, df):
        X = self.vectorizer.transform(df['text'])
        probs = self.model.predict_proba(X)[:,1]
        return probs, self.model.predict(X)

    def run(self, train_df, test_df):
        print("Running Baseline A (TF-IDF + LR)...")
        self.train(train_df)
        probs, preds = self.predict(test_df)
        # metrics = evaluate_predictions(test_df['label'], preds)
        # return metrics
        return probs, preds