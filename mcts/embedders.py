"""
Sentence transformer wrappers for embedding
"""

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch

from mcts.mcts_node import ConversationState

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_SEM_MODEL = "all-MiniLM-L6-v2"


class SemanticWrapper:
    def __init__(self, 
            model_name: str = "all-MiniLM-L6-v2", 
            device: str = "cuda",
    ):
        self.model = SentenceTransformer(model_name, device=device)
    
    def get_embed(self, text: str):
        v = self.model.encode([text], normalize_embeddings=True)[0]  # L2-normalized
        return v.astype(np.float32)