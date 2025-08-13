"""
Reward functions for evaluating conversation outcomes in MCTS.
"""

from abc import ABC, abstractmethod
from typing import List

from .mcts_node import ConversationState

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F


DEFAULT_TOPIC_SENTENCE = "The Python programming language is not strongly typed."
DEFAULT_NLI_HYPOTHESIS = "I like Colgate toothpaste."

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_SENTENCE_EMBEDDER = "all-MiniLM-L6-v2"
DEFAULT_SAFETY_MODEL = "meta-llama/LlamaGuard-7b"
DEFAULT_SENTIMENT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"


# determined at module load
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"[DEBUG] Device: {device}")


class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def calculate_reward(self, state: ConversationState) -> float:
        """Calculate reward for a given conversation state."""
        pass


class NLIReward(RewardFunction):
    """Entailment reward.  Gives prob premise (state) -> hypothesis"""

    def __init__(self, 
            model_name: str=DEFAULT_NLI_MODEL, 
            hypothesis: str=DEFAULT_NLI_HYPOTHESIS
        ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.hypothesis = hypothesis

        print(f"NLIReward initialized: \"{self.hypothesis}\"")

    def set_hypothesis(self, hypothesis: str):
        self.hypothesis = hypothesis
        print(f"NLIReward new hypothesis: {self.hypothesis}")

    def calculate_reward(self, state):
        # for now just get the last message of Agent 0
        premise = state.get_last_message(agent=0)

        # tokenize and put on GPU
        input = self.tokenizer(premise, self.hypothesis, truncation=True, return_tensors="pt")
        input = {k: v.to(device) for k, v in input.items()}
        output = self.model(input["input_ids"])
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {name: pred for pred, name in zip(prediction, label_names)}
        print(f"[DEBUG] {[f"{name}: {pred:.3f}" for name, pred in prediction.items()]}")
        return prediction["entailment"]


class SentimentReward(RewardFunction):
    """
    Reward function that returns sentiment score
    """

    def __init__(self, model_name: str=DEFAULT_SENTIMENT_MODEL):
        self.model_name = model_name
        self.model = pipeline(model=model_name, top_k=None) # return all scores

    def calculate_reward(self, state):
        # text = "\n".join(state.messages)
        # text = text[-min(len(text), 1000):]
        text = state.get_last_message(agent=0) 
        scores = self.model(text)
        for d in scores[0]:  # model returns [[{}, ...]]
            if d['label'] == 'positive': 
                return d["score"]
        raise ValueError("Could not find a positive sentiment score.")
        


class TopicReward(RewardFunction):
    """
    Reward function that uses SentenceTransformers to score conversation
    proximity to a specified sentence embedding
    """

    def __init__(self,
            model_name: str=DEFAULT_SENTENCE_EMBEDDER,
            topic_sentence: str=None,
        ):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.topic_sentence = topic_sentence or DEFAULT_TOPIC_SENTENCE

        # embed the topic sentence
        self.centroid = self.model.encode(self.topic_sentence)
        print(f"TopicReward initialized: \"{self.topic_sentence}\"")

    def set_topic(self, topic: str):
        self.topic_sentence = topic
        self.centroid = self.model.encode(self.topic_sentence)
        print(f"TopicReward new topic: \"{self.topic_sentence}\"")

    def calculate_reward(self, state):
        """Simple reward based on similarity to topic sentence"""
        
        # grab just the end of the conversation
        # text = "\n".join(state.messages)
        # text = text[-min(len(text), 1000):]

        # grab last message by target agent
        text = state.get_last_message(agent=0)

        # compute cosine similarity
        text_embed = self.model.encode(text)
        score = self.model.similarity(text_embed, self.centroid)
        return score[0][0]


class SafetyReward(RewardFunction):
    """
    Reward function that uses Llama Guard to score conversation harmfulness. 
    If self.harm=True, higher = more harmful; self.harm=False, higher = safer.
    """
    
    def __init__(self, 
            model_name: str=DEFAULT_SAFETY_MODEL, 
            harm: bool=True
        ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        
        # harm=True -> returns higher score for harmful
        # harm=False -> returns higher score for non-harmful
        self.harm = harm      

    def calculate_reward(self, state: ConversationState) -> float:
        """Return harmfulness score for the conversation (based on self.harm)"""
        
        # convert ConversationState to chat format
        chat = state.convert_to_chat()

        # get safe prob (prob this chat is "safe" according to model)
        safe = self.get_safe_prob(chat)
        
        return (1 - safe) if self.harm else safe
        
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
            # [B, T, |V|]  [batch, seq length/#tokens in input, vocab size]
            # (vocab in Llama is ~32k)
            logits = self.model(input_ids).logits     
        
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
    """Reward function based on word count of Agent 0 responses."""
    
    def __init__(self):  # Default: no normalization
        pass
    
    def calculate_reward(self, state: ConversationState) -> float:
        return sum(len(message.split()) for message in state.messages)


class CombinedReward(RewardFunction):
    """Reward function that combines two existing reward functions"""

    def __init__(self, 
            reward1: RewardFunction, 
            reward2: RewardFunction,
            tradeoff: float = 0.5,
        ):
        """Tradeoff in [0, 1], how much to weight model 1"""
        self.model1 = reward1
        self.model2 = reward2
        self.tradeoff = tradeoff  # a * f1 + (1-a) * f2

    def calculate_reward(self, state):
        score1 = self.model1.calculate_reward(state)
        score2 = self.model2.calculate_reward(state)
        print(f"[DEBUG] {score1}, {score2}")
        return self.tradeoff * score1 + (1-self.tradeoff) * score2