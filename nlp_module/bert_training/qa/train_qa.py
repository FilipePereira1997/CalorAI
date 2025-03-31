##############################################################################
# 1) Instalar bibliotecas se necessário:
#    pip install sentence-transformers
##############################################################################
import json
import random
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import SentenceEvaluator

##############################################################################
# Função para cálculo do MRR
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
# Classe para avaliar MRR ao final de cada epoch
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
# Carregar o JSON e construir a lista de exemplos
##############################################################################
with open("../../datasets/QA/qa_pt_en.json", "r", encoding="utf-8") as f:
    data = json.load(f)

data_pt = data["pt"]
data_en = data["en"]

train_examples = []
for qa in data_pt:
    train_examples.append(InputExample(texts=[qa["pergunta"], qa["resposta"]]))
for qa in data_en:
    train_examples.append(InputExample(texts=[qa["question"], qa["answer"]]))

print(f"Total de exemplos: {len(train_examples)}")

##############################################################################
# Separar em 80% treino, 20% validação
##############################################################################
random.shuffle(train_examples)
split_index = int(0.8 * len(train_examples))
train_data = train_examples[:split_index]
dev_data   = train_examples[split_index:]

print(f"Treino: {len(train_data)} exemplos")
print(f"Validação: {len(dev_data)} exemplos")

##############################################################################
# Verificar GPU
##############################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

##############################################################################
# Carregar modelo e criar DataLoader
##############################################################################
model_name = "sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1"
model = SentenceTransformer(model_name, device=device)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Cria o evaluator com base no dev set
mrr_evaluator = MRREvaluator(dev_data)

##############################################################################
# Ajustar parâmetros de treino e rodar a avaliação ao final de cada epoch
##############################################################################
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=50,
    show_progress_bar=True,
    evaluator=mrr_evaluator,
    evaluation_steps=0  # 0 => avaliação apenas no final de cada epoch
)

##############################################################################
# Salvar o modelo
##############################################################################
model.save("modelo_QA_xlm-roberta")

##############################################################################
# Exemplo de uso: buscar a resposta mais adequada a uma pergunta nova
##############################################################################
nova_pergunta = "Posso comer pão se já consumi 1800 kcal hoje?"
emb_pergunta = model.encode(nova_pergunta, convert_to_tensor=True)

# Você pode usar as respostas do dataset inteiro ou só parte dele
respostas_dataset = [ex.texts[1] for ex in train_data] + [ex.texts[1] for ex in dev_data]

emb_respostas = model.encode(respostas_dataset, convert_to_tensor=True)
similaridades = util.cos_sim(emb_pergunta, emb_respostas)
idx_melhor = torch.argmax(similaridades, dim=1).item()
resposta_escolhida = respostas_dataset[idx_melhor]

print("Pergunta do usuário:", nova_pergunta)
print("Resposta mais similar do dataset:", resposta_escolhida)
