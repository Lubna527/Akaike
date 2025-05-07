## Akaike
## Email PII Masking & Classification

Automatically detect and mask PII in emails, then classify them by category.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Train the model (optional):
```bash
python train_model.py
```

## Usage

### Web Interface
```bash
python app.py
```

### API Server
```bash
python api.py
```

### Endpoints
- POST `/classify` - Classify email text

## Deployment

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/your-username/email-pii-classifier)

## Model Training
Training data should be a CSV with columns:
- `email`: Raw email text
- `type`: Category label

## Gardio Clickable Test Link
[![Gardio_Link]](https://69cb9d6d2593673c13.gradio.live)
