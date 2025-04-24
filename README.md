# Akaike

# Email Classification System with PII Masking

## Features
- Accurate PII masking (names, emails, phone numbers, etc.)
- LSTM-based email classification
- FastAPI REST endpoint
- Ngrok tunneling for easy testing

## Setup
1. Install dependencies:

pip install -r requirements.txt
python -m spacy download en_core_web_sm

2. Place your dataset at /data/combined_emails_with_natural_pii.csv

## Usage
1. Train the model:
   
python app.py --train

2. Run the API:

python app.py --serve

3. Test with curl:

curl -X POST "http://localhost:8000/classify" \
-H "Content-Type: application/json" \
-d '{"email_body":"Hello, my name is John Doe..."}'

API Response Format
json
{
  "input_email_body": "original text",
  "list_of_masked_entities": [/* detected PII */],
  "masked_email": "masked text",
  "category_of_the_email": "predicted category"
}
