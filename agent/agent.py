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

    @classmethod
    def build_persona(cls, scenario, valence, order):
        stance = scenario["personas"]["stance"][f"{valence:.2f}"].strip()
        background = scenario["personas"]["background"][order].strip()

        persona = f"{background} {stance}"
        return persona

    def get_response(self, 
            state: ConversationState, 
            lever: Optional[str] = None,
            conditioning: Optional[str] = None,
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

        # TODO: implement `conditioning`

        response = self.client.get_response(
            prompt, 
            system_message=persona,
            **kwargs
        )

        return response

    def interview(self,
            state: ConversationState,
            hypothesis: str,
            # provide_rationale: bool = False,
            use_persona: bool = True,
            **kwargs,
    ) -> int:

        persona = prompts.PERSONA.format(
            agent_name=self.name,
            persona=self.persona,
        )

        # TODO: this is a bit hacky
        other_agent = state.agents[0] if state.agents[1] == self.name else state.agents[1]

        prompt = prompts.INTERVIEW.format(
            counterpart=other_agent,
            history="\n".join(state.get_annotated_messages2(whoami=self.order)),
            hypothesis=hypothesis,
        )

        if use_persona:
            response = self.client.get_response(
                prompt,
                system_message=persona,
                **kwargs
            )
        else:
            response = self.client.get_response(prompt, **kwargs)

        res = -1
        try:
            res = int(response)
        except Exception as e:
            print(f"Error parsing interview response: {response} ({e})")
        
        return res

