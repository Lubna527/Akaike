import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from typing import Tuple

class EmailClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.label_encoder = LabelEncoder()

    def train(self, filepath: str) -> Tuple[float, dict]:
        """Train the classifier on labeled email data"""
        df = pd.read_csv(filepath)
        X = df['email'].tolist()
        y = self.label_encoder.fit_transform(df['type'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return accuracy, report

    def save(self, model_path: str):
        """Save model artifacts"""
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder
        }, model_path)

    @classmethod
    def load(cls, model_path: str) -> 'EmailClassifier':
        """Load trained model"""
        instance = cls()
        artifacts = joblib.load(model_path)
        instance.model = artifacts['model']
        instance.vectorizer = artifacts['vectorizer']
        instance.label_encoder = artifacts['label_encoder']
        return instance

    def predict(self, text: str) -> Tuple[str, dict]:
        """Make prediction on single email"""
        vec = self.vectorizer.transform([text])
        pred = self.model.predict(vec)[0]
        return self.label_encoder.inverse_transform([pred])[0]
