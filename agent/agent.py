from .llm_client import LLMClient

class Agent:
    def __init__(self, provider: str, model: str):
        self.client = LLMClient(provider, model)

    def get_response(self, prompt: str) -> str:
        """Generate response using agent's client"""
        
        # TODO persona stuff

        return self.client.get_response(prompt)