# build_dataset.py
# 
# Builds embedded dataset from dialogue texts

import os, json, random
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

from mcts.mcts_node import ConversationState

# ---------------------------
# Dataset → (z_t, z_{t+1}) pairs
# ---------------------------

def _s2t(state):
    # Naive serialization of a ConversationState to a single string.
    return "\n\n".join(state.messages)

def build_pairs_from_dataset(
        embedder,
        max_dialogues=2000,
        max_pairs=200_000,
        split="train",
        seed=0,
        streaming=True,
        save_path=None,
):
    """
    Produce two tensors X, Y where each row is a (z_t, z_{t+1}) pair:
      z_t     = Embedding( all messages up to turn t   )
      z_{t+1} = Embedding( all messages up to turn t+1 )
    """
    
    ds = load_dataset("lmsys/lmsys-chat-1m", split=split, streaming=streaming)

    X, Y = [], []
    dcount, pcount = 0, 0   # dialogue / pair counts

    for row in ds:
        # texts = _extract_text_messages(row)
        texts = [d["content"] for d in row["conversation"]]
        if len(texts) < 2:
            continue

        # Build cumulative states and create pairs (state_t → state_{t+1})
        state = ConversationState(messages=[])
        states = []
        for t in range(len(texts)):
            state = state.add_message(texts[t])
            states.append(state)

        # For t from 0..N-1, we form (S_t -> S_{t+1})
        for t in range(len(texts)):
            s_t = states[t]
            s_tp1 = states[t+1]
            z_t = np.asarray(embedder.encode(_s2t(s_t)), dtype=np.float32)
            z_tp1 = np.asarray(embedder.encode(_s2t(s_tp1)), dtype=np.float32)
            X.append(z_t)
            Y.append(z_tp1)
            pcount += 1
            if pcount >= max_pairs:
                break

        dcount += 1
        if dcount >= max_dialogues:
            break

    X = torch.tensor(np.stack(X), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)
    return X, Y


if __name__ == "__main__":
    pass