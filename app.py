import gradio as gr
from models import EmailClassifier
from utils import PIIMasker
import joblib

# Initialize components
masker = PIIMasker()
classifier = EmailClassifier.load('email_classifier.joblib')

def process_email(email: str) -> Dict[str, Any]:
    """Process email through masking and classification"""
    # Mask PII
    masked = masker.mask(email)
    
    # Classify
    category = classifier.predict(masked['text'])
    
    return {
        'masked_text': masked['text'],
        'category': category,
        'entities': masked['entities']
    }

# Gradio interface
demo = gr.Interface(
    fn=process_email,
    inputs=gr.Textbox(label="Email Text", lines=5),
    outputs=[
        gr.Textbox(label="Masked Text"),
        gr.Label(label="Predicted Category"),
        gr.JSON(label="Detected PII Entities")
    ],
    title="Email PII Masking & Classification",
    description="Automatically mask PII and classify email content"
)

if __name__ == "__main__":
    demo.launch()
