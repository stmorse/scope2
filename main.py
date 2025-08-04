"""
Main script demonstrating conversation planning with MCTS.
Shows how to use the system with different LLM providers.
"""

import os
import argparse
from conversation_planner import ConversationPlanner
from llm_providers import OpenAIProvider, OllamaProvider, MockProvider
from reward_functions import WordCountReward

DEFAULT_PROMPT = "What are your thoughts on artificial intelligence?"
DEFAULT_OLLAMA_HOST = "http://ollama-brewster:80"

def main():
    """Main function to run conversation planning."""
    parser = argparse.ArgumentParser(description="MCTS Conversation Planning")
    parser.add_argument("--provider", choices=["openai", "ollama", "mock"], 
                       default="mock", help="LLM provider to use")
    parser.add_argument("--agent1-model", default="llama3.3:latest", 
                       help="Model for Agent 1")
    parser.add_argument("--agent2-model", default="llama3.3:latest", 
                       help="Model for Agent 2")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST,
                       help="Ollama server host")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                       help="Initial prompt from Agent 1")
    parser.add_argument("--candidates", type=int, default=3,
                       help="Number of candidate responses to evaluate")
    parser.add_argument("--simulations", type=int, default=10,
                       help="Number of MCTS simulations per candidate")
    parser.add_argument("--depth", type=int, default=3,
                       help="Maximum conversation depth")
    parser.add_argument("--reward", choices=["words"],
                       default="words", help="Reward function to use")

    args = parser.parse_args()
    
    print("=" * 60)
    print("MCTS Conversation Planning System")
    print("=" * 60)
    
    # Initialize LLM providers
    try:
        if args.provider == "openai":
            agent1_provider = OpenAIProvider(args.agent1_model)
            agent2_provider = OpenAIProvider(args.agent2_model)
            print(
                f"Using OpenAI models: {args.agent1_model} (Agent1), "
                f"{args.agent2_model} (Agent2)")
    
        elif args.provider == "ollama":
            agent1_provider = OllamaProvider(args.agent1_model, args.ollama_host)
            agent2_provider = OllamaProvider(args.agent2_model, args.ollama_host)
            print(
                f"Using Ollama models: {args.agent1_model} (Agent1), "
                f"{args.agent2_model} (Agent2)")
            print(f"Ollama host: {args.ollama_host}")

        else:
            agent1_provider = MockProvider()
            agent2_provider = MockProvider()
            print("Using mock providers for demonstration.")

    except Exception as e:
        print(f"Error connecting to {args.provider}: {e}")
        print("Using mock provider instead.")
        agent1_provider = MockProvider()
        agent2_provider = MockProvider()

    # Initialize reward function
    if args.reward == "words":
        reward_function = WordCountReward(agent=1)
    print(f"Reward function: {type(reward_function).__name__}")
    
    # Initialize conversation planner
    planner = ConversationPlanner(
        agent1_provider=agent1_provider,
        agent2_provider=agent2_provider,
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
