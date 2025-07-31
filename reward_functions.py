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


class Agent1LengthReward(RewardFunction):
    """Reward function based on the length of Agent 1 responses."""
    
    def __init__(self, normalize: bool = True, max_length: int = 1000):
        """
        Initialize length-based reward function.
        
        Args:
            normalize: Whether to normalize rewards to [0, 1] range
            max_length: Maximum expected length for normalization
        """
        self.normalize = normalize
        self.max_length = max_length
    
    def calculate_reward(self, state: ConversationState) -> float:
        """Calculate reward based on total length of Agent 1 responses."""
        agent1_messages = [
            state.messages[i] for i in range(0, len(state.messages), 2)
        ]
        
        total_length = sum(len(message) for message in agent1_messages)
        
        if self.normalize:
            # Normalize to [0, 1] range
            return min(total_length / self.max_length, 1.0)
        else:
            return float(total_length)


class Agent1WordCountReward(RewardFunction):
    """Reward function based on word count of Agent 1 responses."""
    
    def __init__(self, normalize: bool = True, max_words: int = 200):
        """
        Initialize word count-based reward function.
        
        Args:
            normalize: Whether to normalize rewards to [0, 1] range
            max_words: Maximum expected word count for normalization
        """
        self.normalize = normalize
        self.max_words = max_words
    
    def calculate_reward(self, state: ConversationState) -> float:
        """Calculate reward based on total word count of Agent 1 responses."""
        agent1_messages = [
            state.messages[i] for i in range(0, len(state.messages), 2)
        ]
        
        total_words = sum(len(message.split()) for message in agent1_messages)
        
        if self.normalize:
            return min(total_words / self.max_words, 1.0)
        else:
            return float(total_words)


class ConversationLengthReward(RewardFunction):
    """Reward function based on total conversation length."""
    
    def __init__(self, normalize: bool = True, max_length: int = 2000):
        """
        Initialize conversation length reward function.
        
        Args:
            normalize: Whether to normalize rewards to [0, 1] range
            max_length: Maximum expected total length for normalization
        """
        self.normalize = normalize
        self.max_length = max_length
    
    def calculate_reward(self, state: ConversationState) -> float:
        """Calculate reward based on total conversation length."""
        total_length = sum(len(message) for message in state.messages)
        
        if self.normalize:
            return min(total_length / self.max_length, 1.0)
        else:
            return float(total_length)


class CompositeReward(RewardFunction):
    """Composite reward function that combines multiple reward functions."""
    
    def __init__(self, reward_functions: List[tuple]):
        """
        Initialize composite reward function.
        
        Args:
            reward_functions: List of (reward_function, weight) tuples
        """
        self.reward_functions = reward_functions
        total_weight = sum(weight for _, weight in reward_functions)
        
        # Normalize weights to sum to 1
        self.reward_functions = [
            (func, weight / total_weight) 
            for func, weight in reward_functions
        ]
    
    def calculate_reward(self, state: ConversationState) -> float:
        """Calculate weighted combination of multiple rewards."""
        total_reward = 0.0
        
        for reward_func, weight in self.reward_functions:
            reward = reward_func.calculate_reward(state)
            total_reward += weight * reward
        
        return total_reward


# Default reward function (Agent 1 length)
DEFAULT_REWARD_FUNCTION = Agent1LengthReward(normalize=True, max_length=1000)
