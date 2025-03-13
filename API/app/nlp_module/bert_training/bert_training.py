import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Load the dataset
df = pd.read_csv("../pre_processing/encoded_meals.csv")

# Define the text input and labels
text_column = "meal_description"
label_columns = df.columns[6:]

df[label_columns] = df[label_columns].astype(float)


# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(df)

# Load a BERT tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_data(examples):
    return tokenizer(examples[text_column], padding="max_length", truncation=True)

from sklearn.model_selection import train_test_split

# Convert to Pandas DataFrame for splitting
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df[text_column].tolist(),
    df[label_columns].values.tolist(),
    test_size=0.2,
    random_state=42
)

train_labels = [[float(label) for label in labels] for labels in train_labels]
test_labels = [[float(label) for label in labels] for labels in test_labels]

# Convert back to Hugging Face Dataset
train_data = Dataset.from_dict({"meal_description": train_texts, "labels": train_labels})
test_data = Dataset.from_dict({"meal_description": test_texts, "labels": test_labels})

train_data = train_data.map(tokenize_data, batched=True)
test_data = test_data.map(tokenize_data, batched=True)


from transformers import AutoModelForSequenceClassification

num_labels = len(label_columns)

# Load a BERT model for multi-label classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification")

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

from transformers import Trainer

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# Train the model
trainer.train()

model.save_pretrained("../models/bert_multi_label_classification")
tokenizer.save_pretrained("../models/bert_multi_label_classification")

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    logits, labels = pred
    predictions = (logits > 0.5).astype(int)  # Convert logits to binary values
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

trainer.compute_metrics = compute_metrics

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)

