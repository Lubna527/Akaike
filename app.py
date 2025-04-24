import argparse
import pandas as pd
from models import LSTMEmailClassifier
from utils import PIIMasker
from api import app
import uvicorn
from pyngrok import ngrok
import joblib

def train_model():
    df = pd.read_csv('/data/combined_emails_with_natural_pii.csv')
    emails = df['email_text'].tolist()
    categories = df['category'].tolist()
    
    masker = PIIMasker()
    classifier = LSTMEmailClassifier()
    
    masked_emails = [masker.mask_email(email)['masked_email'] for email in emails]
    classifier.train(masked_emails, categories)
    classifier.save('models/lstm_classifier.joblib')

def run_api():
    classifier = LSTMEmailClassifier()
    classifier.load('models/lstm_classifier.joblib')
    
    # Start ngrok tunnel
    ngrok.set_auth_token("YOUR_NGROK_TOKEN")
    ngrok_tunnel = ngrok.connect(8000)
    print(f'Public URL: {ngrok_tunnel.public_url}')
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--serve", action="store_true", help="Run the API server")
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.serve:
        run_api()
    else:
        print("Please specify --train or --serve")
