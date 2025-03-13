from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
import spacy
import random

# ✅ Initialize FastAPI Router
router = APIRouter(prefix="/bot", tags=["bot"])

# ✅ Detect Device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Load BERT Model for Macronutrient Prediction
bert_model_path = "C:/Users/Asus/PycharmProjects/CalorAI/API/app/nlp_module/bert_training/bert_nutrition_classifier"
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
bert_model.to(device)
bert_model.eval()

# ✅ Load T5 Model for Conversation
t5_model_name = "t5-small"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, legacy=False)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_model.to(device)
t5_model.eval()

# ✅ Load Spacy NLP Model for Meal Detection
nlp = spacy.load("en_core_web_sm")


# ✅ Request Model for FastAPI
class ChatInput(BaseModel):
    text: str


# ✅ Function to Predict Macronutrients (BERT)
def predict_macros(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)

    predicted_macros = outputs.logits.detach().cpu().numpy()[0]

    return {
        "Carbohydrates (g)": float(predicted_macros[0]),
        "Protein (g)": float(predicted_macros[1]),
        "Fat (g)": float(predicted_macros[2]),
        "Energy (kcal)": float(predicted_macros[3])
    }


# ✅ Function to Generate Responses (T5)
def chat_with_t5(user_input):
    inputs = t5_tokenizer(user_input, return_tensors="pt", max_length=50, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = t5_model.generate(**inputs, max_length=50, pad_token_id=t5_tokenizer.eos_token_id)

    return t5_tokenizer.decode(output[0], skip_special_tokens=True)


def detect_food_entities(text):
    """Detects if the text contains food-related entities using SpaCy and a keyword list."""
    doc = nlp(text)

    # ✅ List of common food-related words
    food_keywords = [
        "eat", "ate", "eating", "food", "dish", "meal", "snack", "breakfast", "lunch", "dinner",
        "toast", "eggs", "chicken", "salad", "rice", "pasta", "sandwich", "burger", "fries",
        "fruit", "cereal", "oatmeal", "pizza", "soup", "steak", "vegetables", "fish", "shrimp",
        "smoothie", "yogurt", "avocado", "cheese", "milk", "bread", "pancakes"
    ]

    # ✅ Look for food-related entities in SpaCy
    entity_found = any(ent.label_ in ["FOOD", "PRODUCT"] for ent in doc.ents)

    # ✅ Look for food keywords in the text
    keyword_found = any(word in text.lower() for word in food_keywords)

    # ✅ Return True if either method detects food-related text
    return entity_found or keyword_found

# ✅ Main Chatbot Logic
def chatbot_response(user_input):
    greetings = ["Hello! 😊 How can I help you today?", "Hi! Ready to analyze your meal?", "Hey there! Want to check your meal's nutrition?"]
    goodbyes = ["Goodbye! If you need me, I'll be here. 👋", "Bye! Take care of your diet. 🥗", "It was great talking to you! See you soon. 😊"]

    # If user says goodbye
    if user_input.lower() in ["bye", "goodbye", "see you", "exit"]:
        return random.choice(goodbyes)

    # If user greets the bot
    elif user_input.lower() in ["hi", "hello", "hey"]:
        return random.choice(greetings)

    # ✅ If Spacy detects a meal description, predict macronutrients
    elif detect_food_entities(user_input):
        macros = predict_macros(user_input)
        return (
            f"That sounds like a great meal! Here’s the estimated nutritional breakdown:\n"
            f"🥗 **Carbohydrates:** {macros['Carbohydrates (g)']:.2f}g\n"
            f"🍗 **Proteins:** {macros['Protein (g)']:.2f}g\n"
            f"🧈 **Fats:** {macros['Fat (g)']:.2f}g\n"
            f"🔥 **Total Energy:** {macros['Energy (kcal)']:.2f} kcal\n\n"
            f"Let me know if you need any advice on balancing your diet! 😊"
        )

    # If it's a general conversation, use T5 to generate a response
    else:
        return chat_with_t5(user_input)


# ✅ FastAPI Endpoint for Chatbot
@router.post("/chat/")
async def chat(user_input: ChatInput):
    response = chatbot_response(user_input.text)
    return {"response": response}
