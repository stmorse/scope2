# Conversation Planning with Monte Carlo Tree Search

A Python-based conversation planning system that uses Monte Carlo Tree Search (MCTS) with Upper Confidence Tree (UCT) policy to determine which LLM response maximizes potential reward in multi-turn conversations.

## Overview

This system simulates conversation trees where:
1. Agent 1 provides an initial prompt
2. Agent 2 generates multiple candidate responses
3. MCTS explores possible conversation continuations to a fixed depth
4. Returns the best candidate response based on expected reward

## Features

- **MCTS with UCT Policy**: Intelligent exploration of conversation trees
- **Multi-LLM Support**: Plug-and-play integration with OpenAI and Ollama models
- **Local Ollama Support**: Uses `ollama.Client` for local server integration
- **Configurable Reward**: Currently uses Agent 1 response length as reward metric
- **Flexible Depth**: Configurable simulation depth for tree search

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from conversation_planner import ConversationPlanner
from llm_providers import OpenAIProvider, OllamaProvider

# Initialize with OpenAI
planner = ConversationPlanner(
    agent1_provider=OpenAIProvider("gpt-4"),
    agent2_provider=OpenAIProvider("gpt-3.5-turbo"),
    max_depth=3,
    num_simulations=100
)

# Or with Ollama
planner = ConversationPlanner(
    agent1_provider=OllamaProvider("llama2"),
    agent2_provider=OllamaProvider("mistral"),
    max_depth=3,
    num_simulations=100
)

# Run planning
initial_prompt = "What are your thoughts on climate change?"
results = planner.plan_conversation(initial_prompt, num_candidates=5)

for candidate, score in results:
    print(f"Score: {score:.3f}")
    print(f"Response: {candidate}")
    print("-" * 50)
```

## Architecture

- `conversation_planner.py`: Main MCTS implementation
- `llm_providers.py`: LLM provider interfaces and implementations
- `mcts_node.py`: Tree node structure for MCTS
- `reward_functions.py`: Reward calculation functions
- `main.py`: Example usage script
