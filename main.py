"""
Entrypoint for running MCTS
"""

import argparse
import os
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
    
    # initialize agents
    config = {"OLLAMA_HOST": args.ollama_host}
    agents = {i+1: Agent(
        name=f"Agent {i+1}",
        provider=args.provider, 
        model=args.model, 
        config=config, 
        forcing=True
    ) for i in range(2)}

    # Initialize reward function
    if args.reward == "words":
        reward_function = WordCountReward(agent=1)
    elif args.reward == "harm":
        reward_function = SafetyReward(harm=True)
    elif args.reward == "safe":
        reward_function = SafetyReward(harm=False)
    elif args.reward == "topic":
        reward_function = TopicReward(topic_sentence=DEFAULT_TOPIC_SENTENCE)
    print(f"Reward function: {type(reward_function).__name__}")
    
    t0 = time.time()
    print(f"Initializing planner ... ({t0})")

    # Initialize conversation planner
    planner = ConversationPlanner(
        agents=agents,
        reward_function=reward_function,
        max_depth=args.depth,
        rollout_depth=args.rollout_depth,
        num_simulations=args.simulations,
        exploration_constant=1.414,  # sqrt(2)
        dname=args.dname,
    )
    
    print(f"\nConfiguration:")
    print(f"  Max depth: {args.depth}")
    print(f"  Simulations per candidate: {args.simulations}")
    print(f"  Number of candidates: {args.candidates}")
    print(f"  Number of turns: {args.turns}")
    print(f"\nInitial prompt: \"{args.prompt}\"\n{"=" * 60}\n")
    
    # prompt = args.prompt
    state = ConversationState(
        messages=[args.prompt],
        current_turn=1,  # awaiting Agent 2's response
        depth=1,
    )
    for turn in range(args.turns):
        print(f"\n{"=" * 60} TURN {turn+1}/{args.turns} {"=" * 60}\n\n")

        # Run conversation planning
        try:
            print(f"Starting planning ... ({time.time()-t0:.3f})\n")
            results = planner.plan_conversation(prompt, args.candidates, turn)
            
            print(f"\nResults (ranked by score):")
            results.sort(key=lambda x: x[1], reverse=True)
            
            for i, (candidate, score) in enumerate(results, 1):
                print(f"\nRank {i} (Score: {score:.4f})")
                print("-" * 40)
                print(f"Response: {candidate}\n")
            
            print("\n" + "=" * 60)
            print(f"Best candidate (Score: {results[0][1]:.4f}):")
            print(f"\"{results[0][0]}\"\n\n")
            print(f"COMPLETE ... ({time.time()-t0:.3f})")
        
        except Exception as e:
            print(f"Error during planning: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # get Agent 1 response
        prompt = agents[0].get_response(results[0][0], forcing=True)

if __name__ == "__main__":
    main()
