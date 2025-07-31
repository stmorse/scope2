"""
LLM Provider interfaces and implementations for conversation planning.
Supports OpenAI and Ollama models with plug-and-play architecture.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

# Optional imports for different LLM providers
try:
    import openai
except ImportError:
    openai = None
    
try:
    import ollama
except ImportError:
    ollama = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate a single response to the given prompt."""
        pass
    
    @abstractmethod
    def generate_multiple_responses(self, prompt: str, num_responses: int, temperature: float = 0.7) -> List[str]:
        """Generate multiple responses to the given prompt."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize OpenAI provider.
        
        Args:
            model: OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (if None, uses environment variable)
        """
        if openai is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        self.model = model
        if api_key:
            openai.api_key = api_key
        self.client = openai.OpenAI()
    
    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate a single response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating OpenAI response: {e}")
            return ""
    
    def generate_multiple_responses(self, prompt: str, num_responses: int, temperature: float = 0.7) -> List[str]:
        """Generate multiple responses using OpenAI API."""
        responses = []
        for _ in range(num_responses):
            response = self.generate_response(prompt, temperature)
            if response:
                responses.append(response)
        return responses


class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation for local models."""
    
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        """
        Initialize Ollama provider.
        
        Args:
            model: Ollama model name (e.g., "llama2", "mistral", "codellama")
            host: Ollama server host URL
        """
        if ollama is None:
            raise ImportError("Ollama package not installed. Run: pip install ollama")
        
        self.model = model
        self.client = ollama.Client(host=host)
    
    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate a single response using Ollama API."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": 1000
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Error generating Ollama response: {e}")
            return ""
    
    def generate_multiple_responses(self, prompt: str, num_responses: int, temperature: float = 0.7) -> List[str]:
        """Generate multiple responses using Ollama API."""
        responses = []
        for _ in range(num_responses):
            response = self.generate_response(prompt, temperature)
            if response:
                responses.append(response)
        return responses


class MockProvider(LLMProvider):
    """Mock LLM provider for testing purposes."""
    
    def __init__(self, responses: List[str] = None):
        """Initialize with predefined responses for testing."""
        self.responses = responses or [
            "This is a mock response.",
            "Another mock response for testing.",
            "A third mock response with different content."
        ]
        self.call_count = 0
    
    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Return a mock response."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
    
    def generate_multiple_responses(self, prompt: str, num_responses: int, temperature: float = 0.7) -> List[str]:
        """Generate multiple mock responses."""
        responses = []
        for _ in range(num_responses):
            responses.append(self.generate_response(prompt, temperature))
        return responses
