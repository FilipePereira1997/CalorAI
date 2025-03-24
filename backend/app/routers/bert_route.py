import os
import torch
import logging
import joblib
import google.generativeai as genai
from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login, hf_hub_download
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nutrition_bot")

load_dotenv()
logger.info("Loaded environment variables.")

HF_API_KEY = os.getenv("HF_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Hugging Face login
login(HF_API_KEY)
logger.info("Logged in to Hugging Face Hub.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# FastAPI router
router = APIRouter(prefix="/chat", tags=["chat"])

# Load classification model
repo_id = "FilipePereira1997/CalorAI"
logger.info("Starting to load model : Food Detector")
classifier_model_subfolder = "models/bert_food_detector"
bert_classifier_model = AutoModelForSequenceClassification.from_pretrained(repo_id, subfolder=classifier_model_subfolder)
bert_tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=classifier_model_subfolder)
bert_classifier_model.to(device)
bert_classifier_model.eval()

# Load regression model
logger.info("Starting to load model : Macronutrient Regressor Big")
regression_model_subfolder = "models/bert_regression_big"
bert_regression_model = AutoModelForSequenceClassification.from_pretrained(repo_id, subfolder=regression_model_subfolder)
bert_tokenizer_regression = AutoTokenizer.from_pretrained(repo_id, subfolder=regression_model_subfolder)
bert_regression_model.to(device)
bert_regression_model.eval()

# Load scaler
scaler_filename = "pickle/scaler_fsda_insa.pkl"
scaler_path = hf_hub_download(repo_id=repo_id, filename=scaler_filename, use_auth_token=HF_API_KEY)
logger.info("Scaler file downloaded.")
scaler = joblib.load(scaler_path)

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Utils
def is_food_related(text: str) -> bool:
    if(len(text) < 10):
        return False
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = bert_classifier_model(**inputs)

    logits = outputs.logits.detach().cpu().numpy()
    logger.info(f"Raw logits: {logits}")

    probs = torch.softmax(outputs.logits, dim=1)
    logger.info(f"Probs -> Not Food: {probs[0][0].item():.2f} | Food: {probs[0][1].item():.2f}")

    food_prob = probs[0][1].item()
    return food_prob >= 0.75


def predict_macros(text):
    text = text.lower()
    inputs = bert_tokenizer_regression(text, return_tensors="pt", padding="max_length", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_regression_model(**inputs)
    predicted_macros = outputs.logits.detach().cpu().numpy()[0]
    predicted_macros = scaler.inverse_transform([predicted_macros])[0]
    return predicted_macros

def generate_gemini_response(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 512
        }
    )
    return response.text.strip()

def generate_prompt_food_item(user_query: str, macros: list) -> str:
    return f"""
You are a nutrition assistant.

The user provided the following food description: "{user_query}".

Predicted nutritional values per portion:
- Carbohydrates: {macros[3]:.2f} grams
- Proteins: {macros[1]:.2f} grams
- Fats: {macros[2]:.2f} grams
- Energy: {macros[0]:.2f} kilocalories

Your tasks:
1. Confirm the food item mentioned.
2. Present the nutrients in a friendly, engaging style.
3. Use markdown formatting (e.g., bullet points, bold text).
4. End with *3 suggested follow-up questions* in italics.

Respond conversationally, as if you were a helpful AI nutritionist.
"""

def generate_prompt_fallback(user_query: str) -> str:
    return f"""
You are a nutrition assistant.

The user typed: "{user_query}", which does not appear to be food-related.

Politely explain this and kindly suggest the user provide the name of a food or meal. Keep your response short, warm, and helpful.
"""

# Chatbot handler
def chatbot_response(user_input):
    is_food = is_food_related(user_input)
    if is_food:
        macros = predict_macros(user_input)
        prompt = generate_prompt_food_item(user_input, macros)
    else:
        prompt = generate_prompt_fallback(user_input)
    response_text = generate_gemini_response(prompt)
    return {"message": response_text}

# FastAPI Schema
class ChatInput(BaseModel):
    text: str

@router.post("")
async def chat(user_input: ChatInput):
    response = chatbot_response(user_input.text)
    return response