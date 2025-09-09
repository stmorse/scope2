"""
Entrypoint for running MCTS
"""

import argparse
import json
import logging
import os
import pickle
import time

from agent.agent import Agent
from mcts.conversation_planner import ConversationPlanner
from mcts.mcts_node import ConversationState
from mcts.reward_functions import *


BASE_PATH = "/sciclone/proj-ds/geograd/stmorse/mdp"

def build_persona(scenario, valence):
    stance = scenario["personas"]["stance"][f"{valence:.2f}"].strip()
    background = scenario["personas"]["background"].strip()
    personality = scenario["personas"]["personality"].strip()

    persona = f"{background} {personality} {stance}"
    return persona

def main():
    """Main function to run conversation planning."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=f"test")
    parser.add_argument("--valence0", type=float, default=-1.0)
    parser.add_argument("--valence1", type=float, default=1.0)
    args = parser.parse_args()

    # --- make directory for outputs and logs ---

    # Create directory for experiment results
    path = os.path.join(
        BASE_PATH, args.path, 
        f"v0_{args.valence0:.2f}_v1_{args.valence1:.2f}")
    os.makedirs(path, exist_ok=True)

    
    # --- create logger ---

    # Initialize logging
    logger = logging.getLogger('MCTS_ConversationPlanner')
    if not logger.handlers:
        # console handler (default: stderr)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)

        # file handler
        log_file = os.path.join(path, "output.log")
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        logger.addHandler(handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    def _log(message):
        try:
            logger.info(message)
        except Exception as e:
            print(f"(Exception during logging) (Message: {message}) Exception: {e}")

    _log(f"Logging initiated. Results and logs at: {path}")


    # --- get inputs ---

    # base experiment params
    with open(os.path.join(BASE_PATH, args.path, "init.json"), "r") as f:
        init = json.load(f)

    # scenario params
    with open(os.path.join("scenarios", f"{init["scenario"]}.json"), "r") as f:
        scenario = json.load(f)

    methods = scenario["personas"]["methods"]
    

    # --- initialize agents, reward ---

    # initialize agents
    valences = [args.valence0, args.valence1]
    _log(
        f"\nInitializing agents:\n "
        f"(Provider: {init.get("provider")}, Model: {init.get("model")})\n"
        f"(Valences: Agent 0 ({valences[0]}), Agent 1 ({valences[1]}))\n"
    )
    agents = {i: Agent(
        name=scenario["names"][i],
        order=i,
        provider=init.get("provider"), 
        model=init.get("model"), 
        personality=build_persona(scenario, valences[i]),
        forcing=(int(init.get("forcing"))==1),
        base_method=methods[0],
    ) for i in range(2)}

    # Initialize reward function
    _log(f"\nLoading reward model ... ")
    reward = scenario["reward"]
    if reward == "words":
        reward_function = WordCountReward(agent=0)
    elif reward == "harm":
        reward_function = SafetyReward(harm=True)
    elif reward == "safe":
        reward_function = SafetyReward(harm=False)
    elif reward == "topic":
        reward_function = TopicReward(topic_sentence=scenario["base"])
    elif reward == "combo":
        reward_function = CombinedReward(
            TopicReward(topic_sentence=scenario["base"]), 
            SentimentReward(), 
            tradeoff=0.8
        )
    elif reward == "entail":
        reward_function = NLIReward(hypothesis=scenario["base"])
    _log(f"Reward model: {type(reward_function).__name__}\n")

    _log(f"Configuration: {init}\n")
    _log(f"Initial prompt: \n\"{scenario["prompt"]}\"\n")
    
    
    # --- run experiment ---

    t0 = time.time()

    # initialize conversation
    state = ConversationState(
        messages=[scenario["prompt"]], 
        agents=scenario["names"]
    )

    # iterate through args.turns of dialogue
    for turn in range(init["turns"]):
        _log(f"\n{"=" * 20} TURN {turn+1}/{init["turns"]} {"=" * 20}\n")

        # Initialize conversation planner
        _log(f"Initializing planner ... ({time.time()-t0:.3f})\n")
        planner = ConversationPlanner(
            agents=agents,
            reward_function=reward_function,
            max_depth=init["depth"],
            rollout_depth=init["rollout_depth"],
            num_simulations=init["simulations"],
            exploration_constant=1.414,  # sqrt(2)
            logger=logger
        )

        results = planner.plan_conversation(state, init["candidates"], methods)
        records = planner.get_records()
        
        _log(f"\nResults (ranked by score):")
        results.sort(key=lambda x: x[1], reverse=True)
        for i, (candidate, score) in enumerate(results, 1):
            _log(f"\nRank {i} (Score: {score:.4f})\nResponse: {candidate}\n")
            
        best_cand, score = results[0]
        _log(f"\n{"="*60}\nBest candidate (Score: {score}):\n{best_cand}\n")
        
        # get Agent 0 response
        state = state.add_message(best_cand)
        response = agents[0].get_response(state)
        _log(f"Agent 0 response:\n{response}\n")

        # setup new state for next turn
        state = state.add_message(response)

        _log(f"\n\n[DEBUG] [TURN {turn}]\n{state}")

        save_path = os.path.join(path, f"turn_{turn}.json")
        _log(f"Saving records to path ({save_path}) ... ({time.time()-t0:.3f})")
        full_record = {
            "records": records,
            "results": results,
            "response": response,
            "state": state.messages.copy()
        }
        with open(save_path, "w") as f:
            # pickle.dump(full_record, f)
            json.dump(full_record, f)

    _log(f"\n\n{"="*60}\n\nTranscript of entire conversation:\n")
    _log("\n\n".join(state.get_annotated_messages()))
    _log("\n\nCOMPLETE\n\n")

if __name__ == "__main__":
    main()
