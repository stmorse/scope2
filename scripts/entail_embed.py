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


scenario_name = "fender"
experiment_name = "fender2"
v0s = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
v1 = 1

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

    with open(os.path.join(base_path, "init.json"), "r") as f:
        init = json.load(f)
    with open(f"scenarios/{scenario_name}.json", "r") as f:
        scenario = json.load(f)
    hypothesis = scenario["base"]

    print(f"Loading records ({exp_path})... ")
    records = []
    for fname in os.listdir(exp_path):
        if fname.startswith("turn"):
            with open(os.path.join(exp_path, fname), "rb") as f:
                records.append(pickle.load(f))

    num_turns = len(records)
    num_candidates = len(records[0]["records"])
    num_simulations = len(records[0]["records"]["candidate_0"]) - 1  # exclude "initial_state"

    print(f"Turns: {num_turns}, Candidates: {num_candidates}, Sims: {num_simulations}")


    # --- PULL ROLLOUT TEXTS ---

    # TODO: we have an absolutely hackjob approach to indexing these for later
    # plotting that is terrible and could be fixed

    print("Grabbing rollout texts ... ")

    texts = []
    idxs = {}
    k = 0
    no_errors = True
    for turn in range(num_turns):  # ex: 4 turns: 0, 1, 2, 3
        idxs[turn] = {}

        # append Agent 0 for this turn
        idxs[turn]["Agent_0"] = k
        try:
            texts.append(records[turn]["state"][turn * 2])
        except Exception as e:
            print(f"[DEBUG] turn={turn} {e}")
            print(records[turn]["state"])
            # sys.exit(1)
            no_errors = False
        k += 1
        
        # now store each candidate and *subsequent* rollout
        idxs[turn]["Agent_1"] = {}
        for cand in range(num_candidates):

            # append candidate text
            idxs[turn]["Agent_1"][f"candidate_{cand}"] = {}
            idxs[turn]["Agent_1"][f"candidate_{cand}"]["response"] = k
            cand_text = records[turn]["records"][f"candidate_{cand}"][0]["initial_state"][-1]
            texts.append(cand_text)
            k += 1
            
            for sim in range(num_simulations):
                idxs[turn]["Agent_1"][f"candidate_{cand}"][f"sim_{sim}"] = []
                
                # grab the full rollout
                rollout = records[turn]["records"][f"candidate_{cand}"][sim + 1]["rollout"]
                
                # we can skip past previous turns, and the prompt + candidate
                start = ((turn + 1) * 2)
                end = len(rollout)
                for roll in range(start, end):  
                    idxs[turn]["Agent_1"][f"candidate_{cand}"][f"sim_{sim}"].append(k)
                    t = rollout[roll]
                    texts.append(t)
                    k += 1

    if not no_errors:
        print(f"Encountered error, skipping to next valence...\n")
        continue
    
    print(len(texts))


    # --- EMBED AND SCORE TEXTS ---

    embeddings = []
    probs = []

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


    # --- EMBED AND SCORE PERSONAS ---

    persona0 = scenario["personas"]["stance"][f"{v0:.2f}"]
    persona1 = scenario["personas"]["stance"][f"{v1:.2f}"]
    persona_embeddings = []
    persona_probs = []

    for persona in [persona0, persona1]:
        emb, prob = entailer.get_embed_and_prob(persona, hypothesis)
        persona_embeddings.append(emb)
        persona_probs.append(prob)

    persona_embeddings = torch.stack(persona_embeddings).numpy()
    persona_probs = torch.stack(persona_probs).numpy()


    # --- SAVE ---

    # save
    print(f"Saving to {save_path} ... ")
    with open(save_path, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "probs": probs,
            "texts": texts,
            "idxs": idxs,
            "persona_embeddings": persona_embeddings,
            "persona_probs": persona_probs,
        }, f)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Probs shape: {probs.shape}")
    print("COMPLETE")