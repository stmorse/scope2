"""
Entrypoint for running MCTS
"""

import argparse
import os
import pickle
import time

from agent.agent import Agent
from mcts.conversation_planner import ConversationPlanner
from mcts.mcts_node import ConversationState
from mcts.reward_functions import WordCountReward, SafetyReward, TopicReward

DEFAULT_PROMPT = "Tell me about a conspiracy theory."
DEFAULT_API_KEY = "asdf"
DEFAULT_OLLAMA_HOST = "http://ollama-brewster:80"
PROVIDERS = ["openai", "ollama", "mock"]
DEFAULT_PROVIDER = "mock"
DEFAULT_MODEL = "llama3.3:latest"
REWARDS = ["words", "harm", "safe", "topic"]
DEFAULT_TOPIC_SENTENCE = "Birds are surveillance devices used by the government."
BASE_PATH = "experiments"

def main():
    """Main function to run conversation planning."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=PROVIDERS, default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--rollout_depth", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=10)
    parser.add_argument("--depth", type=int, default=3)  # max tree depth
    parser.add_argument("--reward", choices=REWARDS, default="words")
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--dname", type=str, default="test")
    args = parser.parse_args()

    # ensure save directory exists
    save_path = os.path.join(BASE_PATH, args.dname)
    os.makedirs(save_path, exist_ok=True)
    
    t0 = time.time()

    # initialize agents
    print(f"\nInitializing agents (Provider: {args.provider}, Model: {args.model})")
    config = {"OLLAMA_HOST": args.ollama_host}
    agents = {i: Agent(
        name=f"Agent {i}",
        provider=args.provider, 
        model=args.model, 
        config=config, 
        forcing=True
    ) for i in range(2)}

    # Initialize reward function
    print(f"\nLoading reward model ... ({time.time()-t0:.3f})")
    if args.reward == "words":
        reward_function = WordCountReward(agent=1)
    elif args.reward == "harm":
        reward_function = SafetyReward(harm=True)
    elif args.reward == "safe":
        reward_function = SafetyReward(harm=False)
    elif args.reward == "topic":
        reward_function = TopicReward(topic_sentence=DEFAULT_TOPIC_SENTENCE)
    print(f"Reward model: {type(reward_function).__name__}")

    print(f"\nConfiguration:")
    print(f"  Max depth: {args.depth}")
    print(f"  Simulations per candidate: {args.simulations}")
    print(f"  Number of candidates: {args.candidates}")
    print(f"  Number of turns: {args.turns}")
    print(f"\nInitial prompt: \n\"{args.prompt}\"\n\n")
    
    # initialize conversation
    state = ConversationState(messages=[args.prompt])

    # iterate through args.turns of dialogue
    for turn in range(args.turns):
        print(f"\n{"=" * 20} TURN {turn+1}/{args.turns} {"=" * 20}\n")

        # Initialize conversation planner
        print(f"Initializing planner ... ({time.time()-t0:.3f})\n")
        planner = ConversationPlanner(
            agents=agents,
            reward_function=reward_function,
            max_depth=args.depth,
            rollout_depth=args.rollout_depth,
            num_simulations=args.simulations,
            exploration_constant=1.414,  # sqrt(2)
        )

        results = planner.plan_conversation(state, args.candidates)
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

        path = os.path.join(save_path, f"turn_{turn}.pkl")
        print(f"Saving records to path ({path}) ... ({time.time()-t0:.3f})")
        full_record = {
            "records": records,
            "results": results,
            "response": response
        }
        with open(path, "wb") as f:
            pickle.dump(full_record, f)


if __name__ == "__main__":
    main()
