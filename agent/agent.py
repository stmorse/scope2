from .llm_client import LLMClient

class Agent:
    def __init__(self, provider: str, model: str, config: dict):
        self.client = LLMClient(provider, model, config)

    def get_response(self, prompt: str, forcing: bool = False) -> str:
        """Generate response using agent's client"""
        
        # TODO persona stuff

        return self.client.get_response(prompt, forcing=forcing)