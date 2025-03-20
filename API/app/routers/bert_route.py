from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import joblib
import logging
import random
from transformers import pipeline

# ✅ Initialize FastAPI Router
router = APIRouter(prefix="/bot", tags=["bot"])

# ✅ Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nutrition_bot")

# ✅ Detect Device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ✅ Load BERT Model for Macronutrient Prediction
bert_model_path = "C:/Users/Asus/train_bert/bert_regression_whoInsaDataset"
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
bert_model.to(device)
bert_model.eval()

# ✅ Load Scaler
scaler = joblib.load("C:/Users/Asus/Documents/GitHub/CalorAI/API/app/nlp_module/data/who_dataset/pickle/scaler_merged_dataset.pkl")

classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

def zero_shot_meal_detector(text):
    candidate_labels = ["meal description", "general sentence"]
    result = classifier(text, candidate_labels)
    logger.info(f"Zero-shot classification result: {result}")
    return result["labels"][0] == "meal description"


# ✅ Request Model for FastAPI
class ChatInput(BaseModel):
    text: str

# ✅ Predict Macronutrients (BERT)
def predict_macros(text):
    text = text.lower()

    inputs = bert_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)

    predicted_macros = outputs.logits.detach().cpu().numpy()[0]
    predicted_macros = scaler.inverse_transform([predicted_macros])[0]

    return predicted_macros

# ✅ Main logic

def chatbot_response(user_input):
    macros = predict_macros(user_input)

    if zero_shot_meal_detector(user_input):
        fallback_msgs = [
            "Desculpe, não consegui identificar essa refeição.",
            "Hmm, não consegui calcular as macros dessa descrição.",
            "Não reconheço essa refeição no momento. Tente ser mais específico."
        ]
        return {"message": random.choice(fallback_msgs)}

    logger.info(f"Input: {user_input} => Prediction: {macros}")

    return {
        "carbs": round(macros[0], 2),
        "proteins": round(macros[1], 2),
        "fats": round(macros[2], 2),
        "energy": round(macros[3], 2)
    }

# ✅ FastAPI Endpoint
@router.post("/chat/")
async def chat(user_input: ChatInput):
    response = chatbot_response(user_input.text)
    return response