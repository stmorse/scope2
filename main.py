"""
Entrypoint for running MCTS
"""

import argparse
import configparser
import json
import os
import pickle
import time

from agent.agent import Agent
from mcts.conversation_planner import ConversationPlanner
from mcts.mcts_node import ConversationState
from mcts.reward_functions import *


BASE_PATH = "experiments"


def main():
    """Main function to run conversation planning."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=f"test")
    path = parser.parse_args().path

    # --- get inputs ---

    # experiment params
    with open(os.path.join(BASE_PATH, path, "init.json"), "r") as f:
        args = json.load(f)

    # scenario params
    with open(os.path.join("scenarios", f"{args["scenario"]}.json"), "r") as f:
        scenario = json.load(f)
    

    # --- initialize agents, reward ---

    # initialize agents
    print(
        f"\nInitializing agents "
        f"(Provider: {args.get("provider")}, Model: {args.get("model")})"
    )
    agents = {i: Agent(
        name=f"Agent {i}",
        provider=args.get("provider"), 
        model=args.get("model"), 
        personality=scenario["personas"][i],
        forcing=True if int(scenario["forcing"])==1 else False
    ) for i in range(2)}

    # Initialize reward function
    print(f"\nLoading reward model ... ")
    reward = scenario["reward"]
    if reward == "words":
        reward_function = WordCountReward(agent=1)
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
    print(f"Reward model: {type(reward_function).__name__}\n")

    print(f"Configuration: {args}\n")
    print(f"Initial prompt: \n\"{scenario["prompt"]}\"\n")
    
    t0 = time.time()

    # initialize conversation
    state = ConversationState(messages=[scenario["prompt"]])

    # iterate through args.turns of dialogue
    for turn in range(args["turns"]):
        print(f"\n{"=" * 20} TURN {turn+1}/{args["turns"]} {"=" * 20}\n")

        # Initialize conversation planner
        print(f"Initializing planner ... ({time.time()-t0:.3f})\n")
        planner = ConversationPlanner(
            agents=agents,
            reward_function=reward_function,
            max_depth=args["depth"],
            rollout_depth=args["rollout_depth"],
            num_simulations=args["simulations"],
            exploration_constant=1.414,  # sqrt(2)
        )

        results = planner.plan_conversation(state, args["candidates"])
        records = planner.get_records()
        
        print(f"\nResults (ranked by score):")
        results.sort(key=lambda x: x[1], reverse=True)
        for i, (candidate, score) in enumerate(results, 1):
            print(f"\nRank {i} (Score: {score:.4f})\nResponse: {candidate}\n")
            
        best_cand, score = results[0]
        print(f"\n{"="*60}\nBest candidate (Score: {score}):\n{best_cand}\n")
        
        # get Agent 0 response
        state = state.add_message(best_cand)
        response = agents[0].get_response(state)
        print(f"Agent 0 response:\n{response}\n")

        # setup new state for next turn
        state = state.add_message(response)

        save_path = os.path.join(BASE_PATH, path, f"turn_{turn}.pkl")
        print(f"Saving records to path ({save_path}) ... ({time.time()-t0:.3f})")
        full_record = {
            "records": records,
            "results": results,
            "response": response,
            "state": state.messages
        }
        with open(save_path, "wb") as f:
            pickle.dump(full_record, f)

    print(f"\n\nComplete.  Transcript of entire conversation:\n")
    print("\n\n".join(state.get_annotated_messages()))
    print()


if __name__ == "__main__":
    main()
