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


DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_SENTENCE_EMBEDDER = "all-MiniLM-L6-v2"
DEFAULT_SAFETY_MODEL = "meta-llama/LlamaGuard-7b"
DEFAULT_SENTIMENT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"


# determined at module load
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


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
            hypothesis: str=None
        ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.hypothesis = hypothesis

        if hypothesis is None:
            print("Warning: no hypothesis specified.")

    def set_hypothesis(self, hypothesis: str):
        self.hypothesis = hypothesis

    def embed_and_score_single(self, premise):
        """
        Compute embedding and score for a single agent utterance
        Score = logit_entail - logit_contradict
        """
        
        inputs = self.tokenizer(
            premise, self.hypothesis, 
            truncation=True, return_tensors="pt"
        )

        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # get embedding and logits
        with torch.no_grad():
            enc = self.model.deberta(**inputs)                    # base encoder
            pooled = self.model.pooler(enc.last_hidden_state)     # [1, hidden_size]
            features = self.model.dropout(pooled)                 # pre-classifier embedding
            logits = self.model.classifier(features)              # NLI logits (E/C/N)

        # features is the pair-specific embedding used for the decision
        embedding = features.squeeze(0).cpu()  # shape [hidden_size], here 1024
        # probs  = logits.softmax(dim=-1).squeeze(0).cpu()
        logits = logits.squeeze(0).cpu()
        score = logits[2] - logits[1]

        return embedding, score
    
    # def embed_and_score(self, state, agent=0):
    #     """
    #     Compute embedding (for final agent utterance) and score (aggregated)
    #     """

    #     # TODO: include \ell_0 (persona)

    #     total_score = 0
    #     for utterance in state.get_messages_from_agent(agent):
    #         embedding, score = self.embed_and_score_single(utterance)
    #         total_score += score

    #     # returns final embedding
    #     return embedding, total_score

    def embed_and_score(self, state, agent=0):
        """
        Computes embedding and score for full conversation, 
        includes valence (`valence` should be on -1 to 1)
        """

        premise = " ".join(state.get_messages_from_agent(agent=agent))
        embedding, score = self.embed_and_score_single(premise)
        return embedding, score

    def calculate_reward(self, state, ):
        # for now just get the last message of Agent 0
        # premise = state.get_last_message(agent=0)
        premise = " ".join(state.get_messages_from_agent(agent=0))
        # print(f"[DEBUG] {premise}")

        # tokenize and put on GPU
        input = self.tokenizer(premise, self.hypothesis, truncation=True, return_tensors="pt")
        input = {k: v.to(device) for k, v in input.items()}
        output = self.model(input["input_ids"])
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {name: pred for pred, name in zip(prediction, label_names)}
        print(f"[DEBUG] {[f"{name}: {pred:.3f}" for name, pred in prediction.items()]}")
        # return prediction["entailment"]
        return prediction["entailment"] - prediction["contradiction"]



class TopicReward(RewardFunction):
    def __init__(self,
            model_name: str=DEFAULT_SENTENCE_EMBEDDER,
            topic_sentence: str=None,
        ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.topic_sentence = topic_sentence

        if topic_sentence is None:
            print("Warning: no topic sentence specified, using placeholder.")
            topic_sentence = "(None specified)"

        # embed the topic sentence
        self.centroid = self.model.encode(self.topic_sentence)

    def set_topic(self, topic: str):
        self.topic_sentence = topic
        self.centroid = self.model.encode(self.topic_sentence)
        print(f"TopicReward new topic: \"{self.topic_sentence}\"")

    def embed_and_score(self, state, agent=1):
        """Currently just embeds last utterance from agent"""
        text = state.get_last_message(agent=agent)
        embedding = self.model.encode(text)
        score = self.model.similarity(embedding, self.centroid)
        return embedding, score[0][0]

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
    
    def __init__(self, agent: int = 0):  # Default: no normalization
        self.agent = agent
    
    def calculate_reward(self, state: ConversationState) -> float:
        return sum(
            len(message.split()) 
            for message in state.get_messages_from_agent(agent=self.agent)
        )


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