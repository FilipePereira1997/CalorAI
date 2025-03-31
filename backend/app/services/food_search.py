# app/services/food_search.py
import numpy as np
import logging
from .model_loader import load_model, load_dataset, load_or_build_embeddings

logger = logging.getLogger(__name__)

model = load_model()
df = load_dataset()
embeddings, index = load_or_build_embeddings(model, df)

def search_food(user_query, top_k=3):
    logger.info(f"Buscando alimentos semelhantes para: '{user_query}'")
    query_emb = model.encode(user_query, convert_to_numpy=True)
    query_emb = (query_emb / np.linalg.norm(query_emb)).astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for i in range(top_k):
        row_idx = indices[0][i]
        score = distances[0][i]
        food_name = df["FOOD_NAME"].iloc[row_idx]
        logger.info(f"{i+1}) {food_name} (score={score:.3f})")
        results.append({
            "rank": i + 1,
            "score": float(score),
            "food_name": food_name,
            "ingredients": df["FOOD_INGREDIENTS"].iloc[row_idx],
            "kcal": df["ENERGY (KCAL)"].iloc[row_idx],
            "protein": df["PROTEIN (G)"].iloc[row_idx],
            "fat": df["TOTAL LIPID (FAT) (G)"].iloc[row_idx],
            "carb": df["CARBOHYDRATE, BY DIFFERENCE (G)"].iloc[row_idx],
        })
    return results
