"""
Entrypoint for running MCTS
"""

import argparse
import os

from conversation_planner import ConversationPlanner
from llm_providers import OpenAIProvider, OllamaProvider, MockProvider
from reward_functions import WordCountReward

DEFAULT_PROMPT = "What are your thoughts on artificial intelligence?"
DEFAULT_OLLAMA_HOST = "http://ollama-brewster:80"
PROVIDERS = ["openai", "ollama", "mock"]
DEFAULT_PROVIDER = "mock"
DEFAULT_MODEL = "llama3.3:latest"

def main():
    """Main function to run conversation planning."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=PROVIDERS, default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=10)
    parser.add_argument("--depth", type=int, default=3)  # max tree depth
    parser.add_argument("--reward", choices=["words"], default="words")
    args = parser.parse_args()
    
    print(f"{'=' * 60}\nMCTS Conversation Planning System\n{'=' * 60}")
    
    # Initialize LLM providers
    try:
        if args.provider == "openai":
            client = OpenAIProvider(args.model)
            print(f"Using OpenAI model: {args.model}")
    
        elif args.provider == "ollama":
            client = OllamaProvider(args.model, args.ollama_host)
            print(f"Using Ollama model: {args.model} (host: {args.ollama_host})")

        else:
            client = MockProvider()
            print("Using mock provider for demonstration.")

    except Exception as e:
        print(f"Error connecting to {args.provider}: {e}")
        print("Using mock provider instead.")
        client = MockProvider()

    # Initialize reward function
    if args.reward == "words":
        reward_function = WordCountReward(agent=1)
    print(f"Reward function: {type(reward_function).__name__}")
    
    # Initialize conversation planner
    planner = ConversationPlanner(
        client=client,
        reward_function=reward_function,
        max_depth=args.depth,
        num_simulations=args.simulations,
        exploration_constant=1.414,  # sqrt(2)
        temperature=0.7,
    )
    
    print(f"\nConfiguration:")
    print(f"  Max depth: {args.depth}")
    print(f"  Simulations per candidate: {args.simulations}")
    print(f"  Number of candidates: {args.candidates}")
    
    print(f"\nInitial prompt: \"{args.prompt}\"")
    print("\n" + "=" * 60)
    
    # Run conversation planning
    try:
        results = planner.plan_conversation(args.prompt, args.candidates)
        
        print(f"\nResults (ranked by score):")
        results.sort(key=lambda x: x[1], reverse=True)
        
        for i, (candidate, score) in enumerate(results, 1):
            print(f"\nRank {i} (Score: {score:.4f})")
            print("-" * 40)
            print(f"Response: {candidate}\n")
        
        print("\n" + "=" * 60)
        print(f"Best candidate (Score: {results[0][1]:.4f}):")
        print(f"\"{results[0][0]}\"")
        
        # Show statistics
        stats = planner.get_statistics()
        print(f"\nPlanner Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error during planning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
