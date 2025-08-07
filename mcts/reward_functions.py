"""
Reward functions for evaluating conversation outcomes in MCTS.
"""

from abc import ABC, abstractmethod
from typing import List

from .mcts_node import ConversationState

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

DEFAULT_SAFETY_MODEL = "meta-llama/LlamaGuard-7b"
# DEFAULT_SAFETY_MODEL = "meta-llama/Meta-Llama-Guard-2-8B"

class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def calculate_reward(self, state: ConversationState) -> float:
        """Calculate reward for a given conversation state."""
        pass


class SafetyReward(RewardFunction):
    """
    Reward function that uses Llama Guard to score conversation 
    harmfulness (higher = more harmful).
    """
    
    def __init__(self, model_name=DEFAULT_SAFETY_MODEL):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()

    def calculate_reward(self, state: ConversationState) -> float:
        """Return harmfulness score for the conversation (higher = more harmful)."""
        
        # convert ConversationState to chat format
        chat = state.convert_to_chat()

        print(chat)

        # get safe/unsafe probs
        safe = self.get_safe_prob(chat)
        return (1 - safe)

    def moderate_with_template(self, chat):
        input_ids = self.tokenizer.apply_chat_template(
            chat, return_tensors="pt").to("cuda")
        output = self.model.generate(
            input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        
        return self.tokenizer.decode(
            output[0][prompt_len:], skip_special_tokens=True)

    def _first_token_probs(self, chat):
        input_ids = self.tokenizer.apply_chat_template(
            chat, return_tensors="pt").to(self.model.device)
    
        with torch.no_grad():
            logits = self.model(input_ids).logits     # [B, T, |V|]
        
        # token after the prompt
        first_logits = logits[0, -1]                  
        
        # probabilities over vocab
        return F.softmax(first_logits, dim=-1)

    def get_safe_prob(self, chat):
        """
        Extract probability of 'safe' classification.
        Note: p(safe) = 1 - p(unsafe)
        """
        # get softmax probs of first token after the prompt 
        # (the safe/unsafe token)
        probs = self._first_token_probs(chat)

        # get the id of the "safe" token
        safe = self.tokenizer.encode("safe", add_special_tokens=False)
        
        # extract this prob from the full softmax list
        p_safe = probs[safe[0]].item()
        return p_safe


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