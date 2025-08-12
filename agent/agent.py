"""
Agent wrapping an LLM backend
TODO: persona
"""

from .llm_client import LLMClient
from . import prompts
from mcts.mcts_node import ConversationState

class Agent:
    def __init__(self, 
            name: str, 
            provider: str, 
            model: str, 
            config: dict,
            forcing: bool = False,
        ):
        self.name = name
        self.client = LLMClient(provider, model, config)
        self.forcing = forcing

    def get_response(self, state: ConversationState) -> str:
        """Given a conversation state, generate response using agent's client"""
        
        # NOTE: ConversationState is in dialogue framing, we now convert it
        # to LLM framing (all dialogue inside of role-playing user prompt)

        # TODO persona stuff

        prompt = prompts.DIALOGUE.format(
            agent_name=self.name,
            history="\n".join(state.get_annotated_messages()),
        )

        return self.client.get_response(prompt, forcing=self.forcing)