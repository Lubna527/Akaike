from fastapi import FastAPI
from pydantic import BaseModel
from models import EmailClassifier
from utils import PIIMasker
import joblib

app = FastAPI()

# Load models
masker = PIIMasker()
classifier = EmailClassifier.load('email_classifier.joblib')

class EmailRequest(BaseModel):
    text: str

@app.post("/classify")
async def classify_email(request: EmailRequest):
    """API endpoint for email classification"""
    masked = masker.mask(request.text)
    category = classifier.predict(masked['text'])
    return {
        "category": category,
        "masked_text": masked['text'],
        "entities": masked['entities']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
