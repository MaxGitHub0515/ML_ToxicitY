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
    print("CRITICAL ERROR: Could not load model. {e}")
    print("Did you unzip 'toxicity_model.zip' into the same folder as app.py?")
    classifier = None

class CommentRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "Toxicity AI Service is Running", "model_loaded": classifier is not None}
    
@app.post("/predict")
def predict(request: CommentRequest):
    if not classifier:
        return {"error": "Model failed to initialize. Check server logs."}
    # RUN THE AI
    # multi_label=True means a comment can be BOTH an 'insult' and a 'threat'.
    output = classifier(request.text, candidate_labels=TOXICITY_LABELS, multi_label=True)

    # PROCESS RESULTS
    # The output comes as two lists: ['insult', 'threat'] and [0.99, 0.01].
    # We zip them into a clean dictionary: {'insult': 0.99, 'threat': 0.01 ...}
    scores = {label: round(score, 4) for label, score in zip(output['labels'], output['scores'])}

    # DECISION LOGIC
    # 1. It is toxic if the 'non_toxic' score is too low (less than 50% confidence)
    # 2. OR if any dangerous flag (threat/hate_speech) is extremely high (> 90%)
    is_safe = scores['non_toxic'] > 0.5
    
    return {
        "is_toxic": not is_safe,
        "detailed_scores": scores
    }
