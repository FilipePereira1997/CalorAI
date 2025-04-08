# app/services/qa_template.py
import json
import torch
from sentence_transformers import SentenceTransformer, util
import logging

logger = logging.getLogger(__name__)

QA_DATA_PATH = "app/data/qa_pt_en.json"
MODEL_PATH = "app/data/models/modelo_QA_all-mpnet-base-v2_best_v2"

logger.info(f"Carregando modelo de QA: {MODEL_PATH}")
qa_model = SentenceTransformer(MODEL_PATH)

logger.info(f"Carregando dataset de QA: {QA_DATA_PATH}")
with open(QA_DATA_PATH, "r", encoding="utf-8") as f:
    qa_json = json.load(f)
    qa_data = qa_json["pt"] + qa_json["en"]

qa_questions = [ex.get("pergunta") or ex.get("question") for ex in qa_data]
qa_embeddings = qa_model.encode(qa_questions, convert_to_tensor=True, normalize_embeddings=True)
logger.info(f"Total de perguntas QA carregadas: {len(qa_questions)}")

def preencher_template_resposta(template: str, alimento_info: dict, contexto: dict = {}) -> str:
    # substituicoes = {
    #     "[ALIMENTO]": alimento_info.get("food_name", ""),
    #     "[CALORIAS]": f"{alimento_info.get('kcal', 0):.0f}",
    #     "[PROTEINAS]": f"{alimento_info.get('protein', 0):.1f}",
    #     "[LIPIDOS]": f"{alimento_info.get('fat', 0):.1f}",
    #     "[HIDRATOS]": f"{alimento_info.get('carb', 0):.1f}",
    #     "[INGREDIENTES]": alimento_info.get("ingredients", ""),
    #     "[META_KCAL]" : 2000,
    #     "[KCAL_CONSUMIDAS]" : 500
    # }
    # substituicoes.update(contexto)
    #for chave, valor in substituicoes.items():
    #    template = template.replace(chave, str(valor))
    return template

def responder_com_template(user_query: str, alimento_info: dict, contexto: dict = {}):
    logger.info(f"Selecionando resposta para pergunta: {user_query}")
    query_emb = qa_model.encode(user_query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(query_emb, qa_embeddings)[0]
    top_idx = torch.argmax(scores).item()
    resposta_template = qa_data[top_idx].get("resposta") or qa_data[top_idx].get("answer")
    logger.info(f"Template selecionado: {resposta_template}")
    return preencher_template_resposta(resposta_template, alimento_info, contexto)