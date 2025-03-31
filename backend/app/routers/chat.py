# app/router/chat.py
from fastapi import APIRouter
from pydantic import BaseModel
from services.food_search import search_food
from services.qa_template import responder_com_template
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatInput(BaseModel):
    text: str

@router.post("")
async def chat(user_input: ChatInput):
    logger.info(f"Recebida pergunta do usuário: {user_input.text}")
    resultados = search_food(user_input.text, top_k=3)
    if not resultados or resultados[0]["score"] < 0.3:
        sugestoes = [r["food_name"] for r in resultados]
        sugestoes_str = ", ".join(sugestoes)
        logger.info(f"Pontuação baixa. Sugestões oferecidas: {sugestoes_str}")
        return {"message": f"Não encontrei correspondência clara. Queres dizer: {sugestoes_str}?"}

    alimento = resultados[0]
    logger.info(f"Melhor alimento encontrado: {alimento['food_name']} com score {alimento['score']:.2f}")
    resposta = responder_com_template(user_input.text, alimento)
    logger.info(f"Resposta gerada: {resposta}")
    return {"message": resposta}

