# generate persuader utterance, many user replies
# embed with semantic and entail

import configparser
import json
import pickle
import os
import time

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch

from agent.agent import Agent
from mcts.mcts_node import ConversationState


# TEMPERATURE = 0.1
v0 = 0.2
v1 = 1
scenario_name = "fender"
experiment_name = f"fender/spreads_big_v0_0.20"

N_TRIALS_PER_LEVER = 20      # Persuader responses per lever
N_RESPONSE_PER_TRIAL = 20    # Target responses per persuader response


NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
SEM_MODEL = "all-MiniLM-L6-v2"

config = configparser.ConfigParser()
config.read('config.ini')
PROJ_PATH = config.get("path", "PROJ_PATH")

t0 = time.time()

# ------------------------------------------------------------------------

class NLIWrapper:
    def __init__(self):
        model_name = NLI_MODEL
        
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


class SemanticWrapper:
    def __init__(self, 
            model_name: str = "all-MiniLM-L6-v2", 
            device: str = "cuda",
    ):
        self.model = SentenceTransformer(model_name, device=device)
    
    def get_embed(self, text: str):
        v = self.model.encode([text], normalize_embeddings=True)[0]  # L2-normalized
        return v.astype(np.float32)

# ------------------------------------------------------------------------

def _log(msg):
    print(f"{msg} \033[90m({time.time()-t0:.3f})\033[0m")

# copped from main.py
def build_persona(scenario, valence, order):
    stance = scenario["personas"]["stance"][f"{valence:.2f}"].strip()
    background = scenario["personas"]["background"][order].strip()
    # personality = scenario["personas"]["personality"].strip()

    persona = f"{background} {stance}"
    return persona


base_path = os.path.join(PROJ_PATH, experiment_name)
save_name = os.path.join(base_path, "results.pkl")

os.makedirs(base_path, exist_ok=True)

_log(f"\n{"="*30}\n")
_log(f"Loading scenario details ...")
_log(f" Scenario: {scenario_name}")
_log(f" Experiment: {experiment_name}")
_log(f" Valence: {v0:.2f} --- {v1:.2f}")

with open(f"scenarios/{scenario_name}.json", "r") as f:
    scenario = json.load(f)
hypothesis = scenario["base"]
levers = scenario["levers"]

_log(f"\nInitializing agents ...")
valences = [v0, v1]
agents = {i: Agent(
    name=scenario["personas"]["names"][i],
    order=i,
    provider="ollama", 
    model="llama3.2:latest", 
    persona=build_persona(scenario, valences[i], i),
    forcing=False
) for i in range(2)}

# initialize conversation state
state = ConversationState(
    messages=[scenario["prompt"]],
    agents=scenario["personas"]["names"],
)

# initialize embedders
_log("Initializing embedders...")
nli_model = NLIWrapper()
sem_model = SemanticWrapper()

# {"lever": [[(a1_resp, emb), (a0_resp, emb), ...], ...]}
results = {}

_log("Generating responses ...")
# for temp in np.arange(0, 0.21, 0.05):
for temp in [0.8]:
    _log(f"\nTEMP: {temp:.2f}")

    results[f"{temp:.2f}"] = {}

    for lever in levers:
        _log(f" > {lever}")

        # results[lever] = []
        results[f"{temp:.2f}"][lever] = []

        for i in range(N_TRIALS_PER_LEVER):
            _log(f"   {i}")
            a1_response = agents[1].get_response(
                state, lever=lever, temperature=temp)

            ent_result = nli_model.get_embed_and_prob(
                premise=a1_response, hypothesis=hypothesis)
            sem_embed = sem_model.get_embed(a1_response)

            # results[lever].append([{
            results[f"{temp:.2f}"][lever].append([{
                "agent": 1,
                "response": a1_response,
                "entail": ent_result,
                "semantic": sem_embed
            }])

            state_ = state.add_message(a1_response)
            
            for j in range(N_RESPONSE_PER_TRIAL):
                a0_response = agents[0].get_response(
                    state_, lever=None, temperature=temp)

                ent_result = nli_model.get_embed_and_prob(
                    premise=a0_response, hypothesis=hypothesis)
                sem_embed = sem_model.get_embed(a0_response)

                # results[lever][i].append({
                results[f"{temp:.2f}"][lever][i].append({
                    "agent": 0,
                    "response": a0_response,
                    "entail": ent_result,
                    "semantic": sem_embed
                })

_log("Saving ...")
with open(save_name, "wb") as f:
    pickle.dump(results, f)

_log("COMPLETE\n")