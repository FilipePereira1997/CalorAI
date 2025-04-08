##############################################################################
# 1) Install libraries if needed:
#    pip install sentence-transformers nltk
##############################################################################
import json
import random
import torch
import numpy as np
import csv
import time
import logging
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import SentenceEvaluator

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Setup logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

##############################################################################
# 1.1) Define text preprocessing functions using NLTK
##############################################################################
import re

def clean_text(text):
    """
    Lowercase, remove extra spaces, and standardize placeholders.
    Replace any variant of [food] or {food} with {FOOD}.
    """
    text = text.lower().strip()
    text = " ".join(text.split())
    text = re.sub(r'\[food\]|\{food\}', '{FOOD}', text, flags=re.IGNORECASE)
    return text

def lemmatize_text(text):
    """
    Tokenize and lemmatize text using NLTK.
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)

def preprocess_text(text):
    """
    Apply cleaning and lemmatization.
    """
    text = clean_text(text)
    text = lemmatize_text(text)
    return text

##############################################################################
# 2) Define metric functions
##############################################################################
def compute_mrr(model, data):
    queries = [ex.texts[0] for ex in data]
    answers = [ex.texts[1] for ex in data]
    emb_queries = model.encode(queries, convert_to_tensor=True)
    emb_answers = model.encode(answers, convert_to_tensor=True)
    similarities = util.cos_sim(emb_queries, emb_answers)  # shape (N, N)
    ranks = []
    n = len(data)
    for i in range(n):
        sims = similarities[i]
        sorted_indices = torch.argsort(sims, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(1.0 / rank)
    return float(sum(ranks) / n)

def compute_ndcg(model, data):
    """
    Since each query has one correct answer, DCG is 1/log2(rank+1)
    and the ideal DCG (IDCG) is 1. Thus, nDCG = 1/log2(rank+1).
    """
    queries = [ex.texts[0] for ex in data]
    answers = [ex.texts[1] for ex in data]
    emb_queries = model.encode(queries, convert_to_tensor=True)
    emb_answers = model.encode(answers, convert_to_tensor=True)
    similarities = util.cos_sim(emb_queries, emb_answers)
    ndcg_scores = []
    for i in range(len(data)):
        sims = similarities[i]
        sorted_indices = torch.argsort(sims, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ndcg = 1.0 / (torch.log2(torch.tensor(rank + 1, dtype=torch.float)) + 1e-8)
        ndcg_scores.append(ndcg.item())
    return float(sum(ndcg_scores) / len(ndcg_scores))

def compute_precision_at_k(model, data, k=1):
    queries = [ex.texts[0] for ex in data]
    answers = [ex.texts[1] for ex in data]
    emb_queries = model.encode(queries, convert_to_tensor=True)
    emb_answers = model.encode(answers, convert_to_tensor=True)
    similarities = util.cos_sim(emb_queries, emb_answers)
    correct_count = 0
    for i in range(len(data)):
        sims = similarities[i]
        sorted_indices = torch.argsort(sims, descending=True)
        top_k = sorted_indices[:k]
        if i in top_k:
            correct_count += 1
    return correct_count / len(data)

##############################################################################
# 2.1) Evaluator class to compute and log metrics at the end of each epoch
##############################################################################
class ComprehensiveEvaluator(SentenceEvaluator):
    def __init__(self, dev_data, k=1):
        super().__init__()
        self.dev_data = dev_data
        self.k = k

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        mrr_value = compute_mrr(model, self.dev_data)
        ndcg_value = compute_ndcg(model, self.dev_data)
        precision_value = compute_precision_at_k(model, self.dev_data, k=self.k)
        print(f"\n**** Evaluation on dev set - Epoch: {epoch}, Steps: {steps} ****")
        print(f"MRR: {mrr_value:.4f}")
        print(f"nDCG: {ndcg_value:.4f}")
        print(f"Precision@{self.k}: {precision_value:.4f}\n")
        logging.info(f"Epoch: {epoch}, Steps: {steps}, MRR: {mrr_value:.4f}, nDCG: {ndcg_value:.4f}, Precision@{self.k}: {precision_value:.4f}")
        return mrr_value  # You can choose which metric to use for early stopping if desired

##############################################################################
# 3) Load your JSON and create the list of examples (English only)
##############################################################################
with open("../../datasets/QA/qa_en_bmr.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Preprocess questions and answers using NLTK and add a default label=1.0 for each example
examples = []
for qa in qa_data:
    question = preprocess_text(qa["question"])
    answer = preprocess_text(qa["answer"])
    examples.append(InputExample(texts=[question, answer], label=1.0))
print(f"Total examples (EN only): {len(examples)}")

##############################################################################
# 4) Split into 80% train, 20% validation
##############################################################################
random.shuffle(examples)
split_index = int(0.8 * len(examples))
train_data = examples[:split_index]
dev_data   = examples[split_index:]
print(f"Training examples: {len(train_data)}")
print(f"Validation examples: {len(dev_data)}")

##############################################################################
# 5) Check GPU availability
##############################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

##############################################################################
# 6) Create hyperparameter grid with additional parameters
##############################################################################
epoch_values = [3, 4, 5]
batch_values = [8, 16]
lr_values = [1e-5, 2e-5, 5e-5]
warmup_values = [50, 100, 200]
loss_functions = ["MultipleNegativesRankingLoss", "CosineSimilarityLoss"]

param_grid = []
for ep in epoch_values:
    for bs in batch_values:
        for lr in lr_values:
            for wu in warmup_values:
                for loss_fn in loss_functions:
                    param_grid.append({
                        "epochs": ep,
                        "batch_size": bs,
                        "learning_rate": lr,
                        "warmup_steps": wu,
                        "loss_function": loss_fn
                    })

print(f"Total hyperparameter combinations: {len(param_grid)}")

##############################################################################
# 7) Setup CSV logging for experiment results
##############################################################################
results_file = "experiment_results_v2.csv"
with open(results_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "Epochs", "Batch Size", "Learning Rate", "Warmup Steps", "Loss Function", "MRR", "nDCG", "Precision@1", "Time(s)"])

##############################################################################
# 8) Grid search loop for experiments with additional parameters and logging
##############################################################################
best_mrr = -1
best_config = None
best_model = None
# Use the model "all-mpnet-base-v2"
model_name_base = "sentence-transformers/all-mpnet-base-v2"

for i, params in enumerate(param_grid):
    start_time = time.time()
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    lr = params["learning_rate"]
    wu = params["warmup_steps"]
    loss_fn_name = params["loss_function"]

    print(f"\n=== Experiment {i+1}/{len(param_grid)} ===")
    print(f"Epochs={epochs}, Batch Size={batch_size}, LR={lr}, Warmup Steps={wu}, Loss Function={loss_fn_name}\n")
    logging.info(f"Experiment {i+1}: Epochs={epochs}, Batch Size={batch_size}, LR={lr}, Warmup Steps={wu}, Loss Function={loss_fn_name}")

    # Load the base model using all-mpnet-base-v2
    model = SentenceTransformer(model_name_base, device=device)

    # Create DataLoader
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    # Create loss function based on parameter
    if loss_fn_name == "MultipleNegativesRankingLoss":
        train_loss = losses.MultipleNegativesRankingLoss(model)
    elif loss_fn_name == "CosineSimilarityLoss":
        train_loss = losses.CosineSimilarityLoss(model=model)
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model)

    # Create evaluator on the dev set with k=1 for Precision@1
    evaluator = ComprehensiveEvaluator(dev_data, k=1)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=wu,
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=0,  # evaluate at end of each epoch
        optimizer_params={'lr': lr}
    )

    # Evaluate final metrics on dev set
    final_mrr = compute_mrr(model, dev_data)
    final_ndcg = compute_ndcg(model, dev_data)
    final_precision = compute_precision_at_k(model, dev_data, k=1)
    elapsed_time = time.time() - start_time
    print(f"Final metrics (Dev set) -> MRR: {final_mrr:.4f}, nDCG: {final_ndcg:.4f}, Precision@1: {final_precision:.4f}, Time: {elapsed_time:.1f}s")

    # Log results to CSV
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([i+1, epochs, batch_size, lr, wu, loss_fn_name, final_mrr, final_ndcg, final_precision, elapsed_time])

    # Save the best model based on MRR (or choose another metric if desired)
    if final_mrr > best_mrr:
        best_mrr = final_mrr
        best_config = params
        best_model = model

##############################################################################
# 9) Display and save the best model
##############################################################################
print("\n======= FINAL RESULTS =======")
print(f"Best dev set MRR: {best_mrr:.4f}")
print("Best hyperparameter configuration:", best_config)
logging.info(f"Best dev set MRR: {best_mrr:.4f}, Best config: {best_config}")

if best_model is not None:
    best_model.save("modelo_QA_all-mpnet-base-v2_best_v2")
