"""
Reward functions for evaluating conversation outcomes in MCTS.
"""

from abc import ABC, abstractmethod
from typing import List
from mcts_node import ConversationState


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