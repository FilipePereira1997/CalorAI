# app/services/model_loader.py
import os
import joblib
import numpy as np
import pandas as pd
import faiss
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = "app/data/fsda_insa_dataset.parquet"
EMBEDDINGS_PATH = "app/data/food_embeddings.npy"
INDEX_PATH = "app/data/food_index.faiss"
MODEL_PATH = "sentence-transformers/all-mpnet-base-v2"

def load_model():
    logger.info(f"Carregando modelo de embeddings: {MODEL_PATH}")
    return SentenceTransformer(MODEL_PATH)

def load_dataset():
    logger.info(f"Carregando dataset de alimentos: {DATASET_PATH}")
    return pd.read_parquet(DATASET_PATH)

def load_or_build_embeddings(model, df):
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(INDEX_PATH):
        logger.info("Carregando embeddings e índice FAISS do cache...")
        embeddings = np.load(EMBEDDINGS_PATH)
        index = faiss.read_index(INDEX_PATH)
        logger.info(f"Embeddings carregados com shape: {embeddings.shape}")
        logger.info(f"FAISS index carregado com {index.ntotal} vetores")
    else:
        logger.info("Gerando novos embeddings e índice FAISS...")
        food_texts = (
            df["FOOD_NAME"].fillna("").astype(str) + " " +
            df["FOOD_INGREDIENTS"].fillna("").astype(str)
        ).tolist()
        embeddings = model.encode(food_texts, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        np.save(EMBEDDINGS_PATH, embeddings)
        logger.info(f"Embeddings salvos em: {EMBEDDINGS_PATH}")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, INDEX_PATH)
        logger.info(f"Índice FAISS salvo em: {INDEX_PATH}")

    return embeddings, index