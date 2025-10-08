import json
import os
import pickle
from typing import List, Tuple

from agent.agent import Agent
from mcts.mcts_node import LeverNode, ConversationState
from mcts.reward_functions import NLIReward


# load root node
# iterate through children and compare:
# entailment score <--> interview score

scenario_name = "fender"
experiment_name = "fenderh2"
v0 = 0.1
v1 = 1


PROJ_PATH = "/sciclone/proj-ds/geograd/stmorse/mdp/"

with open(f"scenarios/{scenario_name}.json", "r") as f:
    scenario = json.load(f)
hypothesis = scenario["base"]



# ---- INIT AGENT ----

# this is copped from main.py
def build_persona(scenario, valence):
    stance = scenario["personas"]["stance"][f"{valence:.2f}"].strip()
    background = scenario["personas"]["background"].strip()
    personality = scenario["personas"]["personality"].strip()

    persona = f"{background} {personality} {stance}"
    return persona

agent0 = Agent(
    name=scenario["names"],
    order=0,
    provider="ollama",
    model="llama3.2:latest",
    persona=build_persona(scenario, v0),
    forcing=False, 
)

# ---- DOUBLE SCORE ALL NODES ----

# we have to recalc rewards because the node only stores total reward
# and will include child inputs unless its a leaf
reward_function = NLIReward(hypothesis=hypothesis)

entail_scores = []
interview_scores = []
texts = []
reasons = []

def _record_children(node):
    for k, child in enumerate(node.children):
        # print(f"[DEBUG] {child.state.messages}")

        # get interview response
        interview_res = agent0.interview(
            child.state, hypothesis=hypothesis)
        
        # print(interview_res)
        # print(reason)
        # print()

        interview_scores.append(interview_res)
        reasons.append("")

        # get entailment score
        entail_score = reward_function.calculate_reward(child.state)
        entail_scores.append(entail_score)

        texts.append(child.state.get_last_message(agent=0))

        _record_children(child)

path = f"{experiment_name}/v0_{v0:.2f}_v1_{v1:.2f}"
exp_path = os.path.join(PROJ_PATH, path)
for fname in os.listdir(exp_path):
    if fname.startswith("turn") and fname.endswith("pkl"):
        turn = int(fname[5:6])  # turn_X_root.pkl
        with open(os.path.join(exp_path, fname), "rb") as f:
            root = pickle.load(f)
        
        print(f"\nScoring {turn}")
        _record_children(root)

output = {
    "entail_scores": entail_scores,
    "interview_scores": interview_scores,
    "reasons": reasons,
    "texts": texts,
}
with open(os.path.join(PROJ_PATH, f"{path}/scores.pkl"), "wb") as f:   
    pickle.dump(output, f)

print("COMPLETE")