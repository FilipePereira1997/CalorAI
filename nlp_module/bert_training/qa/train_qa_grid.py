##############################################################################
# 1) Instalar bibliotecas se necessário:
#    pip install sentence-transformers
##############################################################################
import json
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import SentenceEvaluator

##############################################################################
# 2) Definir a função de cálculo do MRR (Mean Reciprocal Rank)
##############################################################################
def compute_mrr(model, data):
    perguntas = [ex.texts[0] for ex in data]
    respostas = [ex.texts[1] for ex in data]

    emb_perguntas = model.encode(perguntas, convert_to_tensor=True)
    emb_respostas = model.encode(respostas, convert_to_tensor=True)

    similarities = util.cos_sim(emb_perguntas, emb_respostas)  # shape (N, N)
    ranks = []
    n = len(data)
    for i in range(n):
        sims_i = similarities[i]
        sorted_indices = torch.argsort(sims_i, descending=True)
        rank_i = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(1.0 / rank_i)

    return float(sum(ranks) / n)

##############################################################################
# 2.1) Classe para avaliar MRR ao final de cada epoch
##############################################################################
class MRREvaluator(SentenceEvaluator):
    def __init__(self, dev_data):
        super().__init__()
        self.dev_data = dev_data

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        mrr_value = compute_mrr(model, self.dev_data)
        print(f"\n**** Avaliação no dev set - Epoch: {epoch}, Steps: {steps}, MRR: {mrr_value:.4f}\n")
        return mrr_value

##############################################################################
# 3) Carregar seu JSON e criar a lista total de exemplos (PT + EN)
##############################################################################
with open("../../datasets/QA/qa_pt_en.json", "r", encoding="utf-8") as f:
    data = json.load(f)

data_pt = data["pt"]
data_en = data["en"]

examples = []
for qa in data_pt:
    examples.append(InputExample(texts=[qa["pergunta"], qa["resposta"]]))
for qa in data_en:
    examples.append(InputExample(texts=[qa["question"], qa["answer"]]))

print(f"Total de exemplos (PT+EN): {len(examples)}")

##############################################################################
# 4) Separar em 80% treino, 20% validação
##############################################################################
random.shuffle(examples)
split_index = int(0.8 * len(examples))
train_data = examples[:split_index]
dev_data   = examples[split_index:]

print(f"Treino: {len(train_data)} exemplos")
print(f"Validação: {len(dev_data)} exemplos")

##############################################################################
# 5) Verificar GPU
##############################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

##############################################################################
# 6) Criação do grid de hiperparâmetros
##############################################################################
epoch_values = [3, 4, 5]
batch_values = [8, 16]
lr_values = [1e-5, 2e-5, 5e-5]
warmup_values = [50, 100, 200]

param_grid = []
for ep in epoch_values:
    for bs in batch_values:
        for lr in lr_values:
            for wu in warmup_values:
                param_grid.append({
                    "epochs": ep,
                    "batch_size": bs,
                    "learning_rate": lr,
                    "warmup_steps": wu
                })

print(f"Total de combinações de hiperparâmetros: {len(param_grid)}")

##############################################################################
# 7) Loop de experimentos (grid search)
##############################################################################
best_mrr = -1
best_config = None
best_model = None

model_name_base = "sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1"

for i, params in enumerate(param_grid):
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    lr = params["learning_rate"]
    wu = params["warmup_steps"]

    print(f"\n=== Experimento {i+1}/{len(param_grid)} ===")
    print(f"Epochs={epochs}, Batch={batch_size}, LR={lr}, Warmup={wu}\n")

    # Carrega o modelo base
    model = SentenceTransformer(model_name_base, device=device)

    # Cria DataLoader e define a loss
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Cria evaluator no dev set
    evaluator = MRREvaluator(dev_data)

    # Treina
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=wu,
        show_progress_bar=True,   # Pode deixar True se quiser ver a barra
        evaluator=evaluator,
        evaluation_steps=0,       # avalia só ao final de cada epoch
        optimizer_params={'lr': lr}
    )

    # Avalia no final
    final_mrr = compute_mrr(model, dev_data)
    print(f"Resultado final (MRR) = {final_mrr:.4f}")

    # Guarda se for o melhor
    if final_mrr > best_mrr:
        best_mrr = final_mrr
        best_config = params
        best_model = model

##############################################################################
# 8) Exibir e salvar o melhor modelo
##############################################################################
print("\n======= RESULTADOS FINAIS =======")
print(f"Melhor MRR no dev set: {best_mrr:.4f}")
print("Melhor configuração de hiperparâmetros:", best_config)

if best_model is not None:
    best_model.save("modelo_QA_xlm-roberta_best")

##############################################################################
# 9) Exemplo de uso do melhor modelo
##############################################################################
nova_pergunta = "Posso comer pão se já consumi 1800 kcal hoje?"
emb_pergunta = best_model.encode(nova_pergunta, convert_to_tensor=True)

respostas_dataset = [ex.texts[1] for ex in train_data] + [ex.texts[1] for ex in dev_data]
emb_respostas = best_model.encode(respostas_dataset, convert_to_tensor=True)

similaridades = util.cos_sim(emb_pergunta, emb_respostas)
idx_melhor = torch.argmax(similaridades, dim=1).item()
resposta_escolhida = respostas_dataset[idx_melhor]

print("\nPergunta do usuário:", nova_pergunta)
print("Resposta mais similar do dataset:", resposta_escolhida)
