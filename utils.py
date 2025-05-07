import re
import spacy
from typing import Dict, List, Any

class PIIMasker:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.placeholders = {
            'full_name': '[full_name]',
            'email': '[email]',
            'phone': '[phone]',
            'dob': '[dob]',
            'aadhar': '[aadhar]',
            'credit_card': '[credit_card]'
        }
        self.patterns = [
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 'full_name'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email'),
            (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', 'phone'),
            (r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(19|20)\d{2}\b', 'dob')
        ]

    def mask(self, text: str) -> Dict[str, Any]:
        """Mask PII in text and return masked text with entity info"""
        protected = {}
        # Step 1: Protect existing placeholders
        for i, match in enumerate(re.finditer(r'\[\w+\]', text)):
            protected[f'__PROTECTED_{i}__'] = match.group()
            text = text.replace(match.group(), f'__PROTECTED_{i}__')
        
        # Step 2: Mask new entities
        entities = []
        for pattern, pii_type in self.patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                if not self._is_overlapping(start, end, entities):
                    entities.append({
                        'start': start,
                        'end': end,
                        'type': pii_type,
                        'value': match.group()
                    })
        
        # Apply masking from end to start
        for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
            text = text[:entity['start']] + self.placeholders[entity['type']] + text[entity['end']:]
        
        # Step 3: Restore protected placeholders
        for placeholder, original in protected.items():
            text = text.replace(placeholder, original)
            
        return {'text': text, 'entities': entities}

    def _is_overlapping(self, start: int, end: int, entities: List[Dict]) -> bool:
        """Check if new entity overlaps with existing ones"""
        for entity in entities:
            if not (end <= entity['start'] or start >= entity['end']):
                return True
        return False
