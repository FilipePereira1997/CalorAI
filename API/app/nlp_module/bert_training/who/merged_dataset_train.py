import torch

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

dataset_name = "normalized_merged_dataset"

# Load the dataset
df = pd.read_csv("api/app/nlp_module/dataset/who/normalized/normalized_merged_dataset.csv")

# Define text and numerical features
text_column = "meal_description"
macro_columns = ["carb", "protein", "fat", "energy"]

# Convert dataframe to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

model_name = "xlm-roberta-base"

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_format(examples):
    tokens = tokenizer(examples[text_column], padding="max_length", truncation=True, return_tensors="pt")

    labels = torch.tensor(
        [[examples[col][i] for col in macro_columns] for i in range(len(examples[text_column]))],
        dtype=torch.float32
    )

    tokens["labels"] = labels
    return {key: tokens[key] for key in tokens}



# Apply tokenization
dataset = dataset.map(tokenize_and_format, batched=True)

# Split into train & test sets
dataset = dataset.train_test_split(test_size=0.2)
train_data = dataset["train"]
test_data = dataset["test"]

# ✅ Ensure the dataset format is correct and move it to the correct device
train_data.set_format(type="torch")
test_data.set_format(type="torch")

from transformers import AutoModelForSequenceClassification

num_labels = len(macro_columns)  # Now predicting 4 continuous values (carb, protein, fat, energy)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="regression"
)

# Move model to the correct device
model.to(device)

from transformers import Trainer, TrainingArguments
import torch.nn as nn

class MSETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(torch.float32).to(device)  # ✅ Move labels to correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}  # ✅ Move all inputs to device

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.MSELoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits
    mae = np.mean(np.abs(preds - labels))
    return {"mae": mae}

torch.cuda.empty_cache()

training_args = TrainingArguments(
    output_dir="./results_who_insa_dataset",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_dir="./logs_who_insa_dataset"
)

trainer = MSETrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics
)

trainer.train()

import joblib

scaler = joblib.load("C:/Users/Asus/Documents/GitHub/CalorAI/API/app/nlp_module/data/who_dataset/pickle/scaler_merged_dataset.pkl")

def predict_macros(text):
    text = full_clean_pipeline(text)  # passa pelo pré-processamento completo

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    predicted_macros = outputs.logits.detach().cpu().numpy()[0]

    predicted_macros = scaler.inverse_transform([predicted_macros])[0]

    return {
        "carbs": predicted_macros[0],
        "protein": predicted_macros[1],
        "fat": predicted_macros[2],
        "energy": predicted_macros[3]
    }

model.save_pretrained("./models/bert_regression_fullDataset")
tokenizer.save_pretrained("./models/bert_regression_fullDataset")