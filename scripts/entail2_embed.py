"""
Loads MCTS outputs, embeds+scores with NLI, saves
TODO: terrible indexing system here
"""

import json
import pickle
import os
import sys

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from mcts.mcts_node import LeverNode, ConversationState


scenario_name = "fender"
experiment_name = "fenderh"
# v0s = [-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
v0s = [0.4]
v1 = 1

if not experiment_name.startswith(scenario_name):
    print(f"WARNING: SCENARIO AND EXPERIMENT DO NOT MATCH.")
    sys.exit(1)

PROJ_PATH = "/sciclone/proj-ds/geograd/stmorse/mdp/"

# ------------------------------------------------------------------------

class NLIWrapper:
    def __init__(self):
        model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move model to GPU
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_embed_and_prob(self, premise, hypothesis):
        inputs = self.tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")

        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # get embedding and logits
        with torch.no_grad():
            enc = self.model.deberta(**inputs)                    # base encoder
            pooled = self.model.pooler(enc.last_hidden_state)     # [1, hidden_size]
            features = self.model.dropout(pooled)                 # pre-classifier embedding
            logits = self.model.classifier(features)              # NLI logits (E/C/N)

        # features is the pair-specific embedding used for the decision
        embedding = features.squeeze(0).cpu()  # shape [hidden_size], here 1024
        probs  = logits.softmax(dim=-1).squeeze(0).cpu()

        return embedding, probs
    

# --- LOAD RECORDS ---

for v0 in v0s:

    base_path = os.path.join(PROJ_PATH, experiment_name)
    exp_path = os.path.join(base_path, f"v0_{v0:.2f}_v1_{v1:.2f}")
    save_path = os.path.join(exp_path, "embed.pkl")

    print(f"\n{"="*30}\n")
    print(f"Loading scenario details ...")
    print(f" Scenario: {scenario_name}")
    print(f" Experiment: {experiment_name}")
    print(f" Valence: {v0:.2f} --- {v1:.2f}")

    with open(f"scenarios/{scenario_name}.json", "r") as f:
        scenario = json.load(f)
    hypothesis = scenario["base"]
    levers = scenario["levers"]

    # various metadata about what we're doing
    META = {}
    
    # data
    texts = []      # raw texts
    labels = []     # (turn, layer, child, lever, generation (-1 is selected))
    embeddings = []
    probs = []
    
    # --- GRAB FINAL CONVO TEXT ---

    # get all records
    # TODO: we aren't really using a lot of this in this version
    records = []
    for fname in os.listdir(exp_path):
        if fname.startswith("turn") and fname.endswith("json"):
            with open(os.path.join(exp_path, fname), "rb") as f:
                # records.append(pickle.load(f))
                records.append(json.load(f))

    # get the actual conversation outcome from the last record
    texts.extend(records[-1]["state"])
    META["convo_length"] = len(texts)

    # --- GRAB AND LABEL TEXTS FROM MCTS ---

    # now build all trees with embeddings
    print("Loading model ...")
    entailer = NLIWrapper()

    def _embed_children(node, turn, layer):
        for k, child in enumerate(node.children):
            lever = levers.index(child.lever)

            # embed generations
            gi = 0
            for gen in child.generations:
                for msg in gen:
                    texts.append(msg)
                    labels.append((turn, layer, k, lever, gi))
                gi += 1

            # embed state
            for msg in child.state.messages[-2:]:
                texts.append(msg)
                labels.append((turn, layer, k, lever, gi))

            _embed_children(child, turn, layer+1)

    # iterate thru each turn's root node and build embeddings
    for fname in os.listdir(exp_path):
        if fname.startswith("turn") and fname.endswith("pkl"):
            turn = int(fname[5:6])
            with open(os.path.join(exp_path, fname), "rb") as f:
                root = pickle.load(f)

            _embed_children(root, turn, 1)

    META["trees_length"] = len(texts) - META["convo_length"]

    # --- STORE PERSONAS ---

    persona0 = scenario["personas"]["stance"][f"{v0:.2f}"]
    persona1 = scenario["personas"]["stance"][f"{v1:.2f}"]
    texts.append(persona0)
    texts.append(persona1)

    META["persona_length"] = 2


    # --- EMBED AND SCORE TEXTS ---

    print("Loading model ...")
    entailer = NLIWrapper()

    print("Embedding and scoring ...")
    for i, text in enumerate(texts):
        if i % 100 == 0: print(f" > {i}")
        emb, prob = entailer.get_embed_and_prob(text, hypothesis)
        embeddings.append(emb)
        probs.append(prob)

    # convert to numpy arrays
    embeddings = torch.stack(embeddings).numpy()
    probs = torch.stack(probs).numpy()


    # --- SAVE ---

    # save
    print(f"Saving to {save_path} ... ")
    with open(save_path, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "probs": probs,
            "texts": texts,
            "labels": labels,
            "meta": META,
        }, f)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Probs shape: {probs.shape}")
    print(f"Raw texts: {len(texts)}")
    print(f"Labels (MCTS): {len(labels)}")
    
print("\nCOMPLETE\n\n")