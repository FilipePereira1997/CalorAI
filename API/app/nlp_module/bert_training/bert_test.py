from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Detectar dispositivo (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Carregar o modelo e tokenizer salvos
model_path = "bert_nutrition_classifier"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Mover modelo para o dispositivo correto
model.to(device)
model.eval()  # Coloca o modelo em modo de avaliação (evita dropout durante inferência)

def predict_macros(text):
    # Tokenizar o texto e mover para o dispositivo correto
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Mover inputs para o mesmo device do modelo

    # Fazer a previsão
    with torch.no_grad():  # Desativa o autograd para inferência (economiza memória)
        outputs = model(**inputs)

    # Extrair os logits e converter para valores numéricos
    predicted_macros = outputs.logits.detach().cpu().numpy()[0]

    # Retornar os valores previstos
    return {
        "Carbohydrates (g)": predicted_macros[0],
        "Protein (g)": predicted_macros[1],
        "Fat (g)": predicted_macros[2],
        "Energy (kcal)": predicted_macros[3]
    }


meal1 = "For breakfast, I had scrambled eggs with toast."
meal2 = "I had a grilled chicken salad with avocado and a side of brown rice."
meal3 = "Dinner was a cheeseburger with fries and soda."

# Fazer previsões
print("Meal 1:", predict_macros(meal1))
print("Meal 2:", predict_macros(meal2))
print("Meal 3:", predict_macros(meal3))
