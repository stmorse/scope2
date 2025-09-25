"""
Agent wrapping an LLM backend
"""
from typing import Optional

from .llm_client import LLMClient
from . import prompts
from mcts.mcts_node import ConversationState

class Agent:
    def __init__(self, 
            name: str, 
            order: int,
            provider: str, 
            model: str, 
            persona: str = None,
            forcing: bool = False
        ):
        self.name = name
        self.order = order   # 0 (target) or 1 (persuader)
        self.client = LLMClient(
            provider, 
            model, 
            forcing=forcing, 
        )
        self.persona = persona or "(None specified)"

    def get_response(self, 
            state: ConversationState, 
            lever: Optional[str] = None,
            **kwargs,
    ) -> str:
        """Given a conversation state, generate response using agent's client"""
        
        # NOTE: ConversationState is in dialogue framing, we now convert it
        # to LLM framing (all dialogue inside of role-playing user prompt)

        persona = prompts.PERSONA.format(
            agent_name=self.name,
            persona=self.persona,
        )

        # TODO: this is a bit hacky
        other_agent = state.agents[0] if state.agents[1] == self.name else state.agents[1]
        if lever:
            prompt = prompts.DIALOGUE_WITH_PREAMBLE.format(
                counterpart=other_agent,
                agent_name=self.name,
                history="\n".join(state.get_annotated_messages2(whoami=self.order)),
                lever=lever
            )
        else:
            prompt = prompts.DIALOGUE.format(
                counterpart=other_agent,
                agent_name=self.name,
                history="\n".join(state.get_annotated_messages2(whoami=self.order)),
            )

        response = self.client.get_response(
            prompt, 
            system_message=persona,
            **kwargs
        )

        return response