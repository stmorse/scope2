"""
Agent wrapping an LLM backend
"""

from .llm_client import LLMClient
from . import prompts
from mcts.mcts_node import ConversationState

class Agent:
    def __init__(self, 
            name: str, 
            provider: str, 
            model: str, 
            personality: str = None,
            forcing: bool = False,
        ):
        self.name = name
        self.client = LLMClient(
            provider, 
            model, 
            forcing=forcing, 
        )
        self.personality = personality or "(None specified)"

    def get_response(self, state: ConversationState, **kwargs) -> str:
        """Given a conversation state, generate response using agent's client"""
        
        # NOTE: ConversationState is in dialogue framing, we now convert it
        # to LLM framing (all dialogue inside of role-playing user prompt)

        persona = prompts.PERSONA.format(
            agent_name=self.name,
            personality=self.personality,
        )

        # TODO: this is hacky for naming
        anum = int(self.name[-1])
        prompt = prompts.DIALOGUE.format(
            counterpart=f"Agent {0 if anum==1 else 1}",
            agent_name=self.name,
            history="\n".join(state.get_annotated_messages2(whoami=anum)),
        )

        response = self.client.get_response(
            prompt, 
            system_message=persona,
            **kwargs
        )

        return response