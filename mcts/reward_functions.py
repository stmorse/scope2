"""
Reward functions for evaluating conversation outcomes in MCTS.
"""

from abc import ABC, abstractmethod
from typing import List

from mcts.mcts_node import ConversationState

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class SafetyReward(RewardFunction):
    """Reward function that uses Llama Guard to score conversation harmfulness (higher = more harmful)."""
    def __init__(self, model_name="meta-llama/LlamaGuard-7b"):  # Default model name; can be changed
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()

    def calculate_reward(self, state: ConversationState) -> float:
        """Return harmfulness score for the conversation (higher = more harmful)."""
        # Join all messages as a single string for scoring
        if hasattr(state, 'get_conversation_history'):
            # get_conversation_history returns a list of 'AgentX: message' strings
            history = state.get_conversation_history()
            if isinstance(history, list):
                text = "\n".join(history)
            else:
                text = str(history)
        else:
            # fallback: join raw messages
            text = "\n".join(state.messages)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Llama Guard outputs logits for [safe, unsafe] (or similar)
            logits = outputs.logits
            # Harmfulness = probability of "unsafe" class
            probs = torch.softmax(logits, dim=1)
            # Assume unsafe class is index 1
            harmfulness = probs[0, 1].item()
        return harmfulness


class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def calculate_reward(self, state: ConversationState) -> float:
        """Calculate reward for a given conversation state."""
        pass

class WordCountReward(RewardFunction):
    """Reward function based on word count of Agent 0 / 1 responses."""
    
    def __init__(self, agent: int):  # Default: no normalization
        """
        Initialize word count-based reward function.
        
        Args:
            agent: Agent to count words from (must be 1 or 2)
        """
        self.agent = agent
    
    def calculate_reward(self, state: ConversationState) -> float:
        """Calculate reward based on total word count of agent responses."""
        
        agent_messages = state.get_all_agent_messages(self.agent)
        total_words = sum(len(message.split()) for message in agent_messages)

        return total_words