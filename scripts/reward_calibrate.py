# generate a full multi-turn conversation (no MCTS)
# score with entailment and interview for comparison

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


TEMPERATURE = 0.1
LEVER = 0
v0 = 0.2
v1 = 1
N_CONVERSATIONS = 50
N_TURNS = 4


scenario_name = "fender"
experiment_folder = f"{scenario_name}/rewards"

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

config = configparser.ConfigParser()
config.read('config.ini')
PROJ_PATH = config.get("path", "PROJ_PATH")

base_path = os.path.join(PROJ_PATH, experiment_folder)
save_name = os.path.join(base_path, f"results_v0_{v0:.2f}_t_{TEMPERATURE:.2f}_l_{LEVER}.pkl")

os.makedirs(base_path, exist_ok=True)

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

# ------------------------------------------------------------------------

def _log(msg, showtime=True):
    if showtime:
        newlines = ""
        while msg.endswith("\n"):
            newlines += "\n"
            msg = msg[:-1]
        print(f"{msg} \033[90m({time.time()-t0:.3f})\033[0m {newlines}")
    else:
        print(f"{msg}")

# ------------------------------------------------------------------------


_log(f"\n{"="*30}\n", showtime=False)
_log(f"Loading scenario details ...")
_log(f" Scenario: {scenario_name}")
_log(f" Save path: {experiment_folder}")
_log(f" File: {save_name}")
_log(f" Valence: {v0:.2f} --- {v1:.2f}")

with open(f"scenarios/{scenario_name}.json", "r") as f:
    scenario = json.load(f)
hypothesis = scenario["base"]
levers = scenario["levers"]

_log(f"Using lever: {levers[LEVER]}")

_log(f"\nInitializing agents ...")
valences = [v0, v1]
agents = {i: Agent(
    name=scenario["personas"]["names"][i],
    order=i,
    provider="ollama", 
    model="llama3.2:latest", 
    persona=Agent.build_persona(scenario, valences[i], i),
    forcing=False
) for i in range(2)}

# initialize embedders
_log("Initializing embedder...")
nli_model = NLIWrapper()

results = []

for i in range(N_CONVERSATIONS):
    _log(f"\nCONVERSATION {i+1} of {N_CONVERSATIONS}\n")

    results.append([])

    # initialize conversation state
    state = ConversationState(
        messages=[scenario["prompt"]],
        agents=scenario["personas"]["names"],
    )

    ent_reward = nli_model.get_embed_and_prob(
        premise=scenario["prompt"], hypothesis=hypothesis) 

    rating_with_persona = agents[0].interview(
        state, hypothesis=hypothesis, use_persona=True)
    
    rating_no_persona = agents[0].interview(
        state, hypothesis=hypothesis, use_persona=False)

    results[i].append({
        "agent": 0,
        "response": scenario["prompt"],
        "entail": ent_reward,
        "entail_full": ent_reward,  # same as entail for first message
        "rating_with_persona": rating_with_persona,
        "rating_no_persona": rating_no_persona,
    })

    er = ent_reward[1].numpy()

    _log(
        f"\033[93m{agents[0].name}:\033[0m {scenario['prompt']}\n"
        f"\033[90m({er[0]:.2f}, {er[1]:.2f}, {er[2]:.2f}) "
        f"({rating_with_persona}) ({rating_no_persona})\033[0m",
        showtime=False
    )

    # run full conversation
    for turn in range(N_TURNS):

        for a in [1,0]:
        
            response = agents[a].get_response(
                state, 
                lever=levers[LEVER] if a==1 else None,
                temperature=TEMPERATURE,
            )

            state = state.add_message(response)

            ent_reward = nli_model.get_embed_and_prob(
                premise=response, hypothesis=hypothesis) 
            
            ent_reward2 = nli_model.get_embed_and_prob(
                premise="\n".join(state.get_messages_from_agent(agent=a)), 
                hypothesis=hypothesis)

            rating_with_persona = agents[a].interview(
                state, hypothesis=hypothesis, use_persona=True)
            
            rating_no_persona = agents[a].interview(
                state, hypothesis=hypothesis, use_persona=False)

            results[i].append({
                "agent": a,
                "response": response,
                "entail": ent_reward,
                "entail_full": ent_reward2,
                "rating_with_persona": rating_with_persona,
                "rating_no_persona": rating_no_persona,
            })

            er = ent_reward[1].numpy()
            er2 = ent_reward2[1].numpy()

            _log(
                f"\033[93m{agents[a].name}:\033[0m {response}\n"
                f"\033[90m({er[0]:.2f}, {er[1]:.2f}, {er[2]:.2f}) "
                f"({er2[0]:.2f}, {er2[1]:.2f}, {er2[2]:.2f}) "
                f"({rating_with_persona}) ({rating_no_persona})\033[0m",
                showtime=False
            )

_log("\n\nSaving ...")
with open(save_name, "wb") as f:
    pickle.dump(results, f)

_log("COMPLETE\n")

