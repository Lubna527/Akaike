from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from models import LSTMEmailClassifier
from utils import PIIMasker

app = FastAPI()
masker = PIIMasker()
classifier = LSTMEmailClassifier()

class EmailRequest(BaseModel):
    email_body: str

class MaskedEntity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[MaskedEntity]
    masked_email: str
    category_of_the_email: str

@app.post("/classify", response_model=EmailResponse)
async def classify_email(request: EmailRequest):
    try:
        masking_result = masker.mask_email(request.email_body)
        category = classifier.predict(masking_result["masked_email"])
        return {
            "input_email_body": request.email_body,
            "list_of_masked_entities": [
                {"position": e["position"], "classification": e["classification"], "entity": e["entity"]}
                for e in masking_result["entities"]
            ],
            "masked_email": masking_result["masked_email"],
            "category_of_the_email": category
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
