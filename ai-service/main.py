from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import os

# LOAIDING THE MODEL USING TRANSFORMERS 
app = FastAPI()

# 1. Load the model from the folder you downloaded from Colab
# Ensure the folder name matches exactly (e.g., "saved_model")
MODEL_PATH = "./toxicity_model_brain" 
TOXICITY_LABELS = [
    "toxicity", "insult", "harassment",
    "threat", "non_toxic", "accusation", "suspicion"]

print(f"Loading model from {MODEL_PATH}...")
try:
    classifier = pipeline(
        "zero-shot-classification", 
        model=MODEL_PATH,    # Looks for pytorch_model.bin here
        tokenizer=MODEL_PATH, # Looks for tokenizer_config.json here
        device=-1                  # -1 means CPU (Safe for DigitalOcean)
    )
    print("Model loaded successfully!")

except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. {e}")
    print("Did you unzip 'toxicity_model.zip' into the same folder as app.py?")
    classifier = None

class CommentRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "Toxicity AI Service is Running", "model_loaded": classifier is not None}
    
@app.post("/predict")
def predict(request: CommentRequest):
    # Defensive: handle missing/empty text without calling the model
    text = (request.text or "").strip()
    if len(text) == 0:
        empty_scores = {label: 0.0 for label in TOXICITY_LABELS}
        empty_scores["non_toxic"] = 1.0
        return {"is_toxic": False, "detailed_scores": empty_scores}

    if not classifier:
        # Keep API stable but avoid exceptions; upstream may choose to treat as toxic
        return {"error": "Model failed to initialize. Check server logs."}

    try:
        # RUN THE AI
        # multi_label=True means a comment can be BOTH an 'insult' and a 'threat'.
        output = classifier(text, candidate_labels=TOXICITY_LABELS, multi_label=True)

        # PROCESS RESULTS
        # The output comes as two lists: ['insult', 'threat'] and [0.99, 0.01].
        # We zip them into a clean dictionary: {'insult': 0.99, 'threat': 0.01 ...}
        scores = {label: round(score, 4) for label, score in zip(output['labels'], output['scores'])}

        # DECISION LOGIC (threshold-based, avoids relying on 'non_toxic')
        harassment = float(scores.get('harassment', 0.0))
        threat = float(scores.get('threat', 0.0))
        toxicity = float(scores.get('toxicity', 0.0))

        har_th = float(os.getenv('TOXIC_HARASSMENT_TH', '0.5'))
        thr_th = float(os.getenv('TOXIC_THREAT_TH', '0.5'))
        tox_th = float(os.getenv('TOXIC_TOXICITY_TH', '0.8'))

        is_toxic = (harassment >= har_th) or (threat >= thr_th) or (toxicity >= tox_th)
        return {"is_toxic": is_toxic, "detailed_scores": scores}
    except Exception as e:
        # Fail-safe: don't crash the service; upstream can handle fallback behavior
        print(f"Toxicity classification error: {e}")
        return {"error": "classification_failed"}
