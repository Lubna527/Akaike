import re
import spacy
from typing import Dict, List, Any

class PIIMasker:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.patterns = {
            "full_name": r'\b([A-Z][a-z]+(?: [A-Z][a-z\.]+){1,3})\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_number": r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{3}[-.\s]?\d{4}\b',
            "dob": r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)?\d{2}\b',
            "aadhar_num": r'\b\d{4}[ -]?\d{4}[ -]?\d{4}\b',
            "credit_debit_no": r'\b(?:\d[ -]*?){13,16}\b',
            "cvv_no": r'\b\d{3,4}\b',
            "expiry_no": r'\b(?:0[1-9]|1[0-2])[/-](?:\d{4}|\d{2})\b'
        }

    def mask_email(self, email_body: str) -> Dict[str, Any]:
        doc = self.nlp(email_body)
        masked_text = email_body
        entities = []
        masked_positions = set()
        
        # Masking logic (same as previous implementation)
        # ...
        
        return {"masked_email": masked_text, "entities": sorted(entities, key=lambda x: x["position"][0])}
