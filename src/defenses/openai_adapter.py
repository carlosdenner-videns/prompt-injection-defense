"""
OpenAI GPT-4 adapter for prompt injection detection and response analysis.
"""

import os
import time
from typing import Optional
from dataclasses import dataclass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


@dataclass
class ModelResponse:
    """Response from OpenAI model with metadata."""
    content: str
    latency_ms: float
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIAdapter:
    """
    Adapter for OpenAI GPT-4 API calls with timing and response capture.
    
    Usage:
        adapter = OpenAIAdapter(api_key="your-api-key")
        response = adapter.call_model("Hello, how are you?")
        print(f"Response: {response}")
        print(f"Latency: {adapter.last_latency_ms:.2f}ms")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 30
    ):
        """
        Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (default: "gpt-4")
            temperature: Sampling temperature 0-2 (default: 0.7)
            max_tokens: Maximum tokens in response (default: None = no limit)
            timeout: Request timeout in seconds (default: 30)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
        
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Track last call metrics
        self.last_latency_ms: float = 0.0
        self.last_response: Optional[ModelResponse] = None
    
    def call_model(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call OpenAI model and return response text.
        
        Args:
            prompt: User prompt to send to the model
            system_prompt: Optional system prompt (default: None)
        
        Returns:
            Response text from the model
            
        Raises:
            Exception: If API call fails
        """
        t0 = time.perf_counter_ns()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response
            content = response.choices[0].message.content or ""
            
            # Calculate latency
            latency_ms = (time.perf_counter_ns() - t0) / 1e6
            
            # Store metadata
            self.last_latency_ms = latency_ms
            self.last_response = ModelResponse(
                content=content,
                latency_ms=latency_ms,
                model=response.model,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0
            )
            
            return content
            
        except Exception as e:
            latency_ms = (time.perf_counter_ns() - t0) / 1e6
            self.last_latency_ms = latency_ms
            raise Exception(f"OpenAI API call failed after {latency_ms:.2f}ms: {str(e)}")
    
    def call_with_metadata(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> ModelResponse:
        """
        Call model and return full response with metadata.
        
        Args:
            prompt: User prompt to send to the model
            system_prompt: Optional system prompt
        
        Returns:
            ModelResponse with content, latency, and token usage
        """
        content = self.call_model(prompt, system_prompt)
        return self.last_response
    
    def get_stats(self) -> dict:
        """
        Get statistics from the last API call.
        
        Returns:
            Dictionary with latency and token usage stats
        """
        if not self.last_response:
            return {
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        
        return {
            "latency_ms": self.last_response.latency_ms,
            "prompt_tokens": self.last_response.prompt_tokens,
            "completion_tokens": self.last_response.completion_tokens,
            "total_tokens": self.last_response.total_tokens,
            "model": self.last_response.model
        }


# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize adapter (reads OPENAI_API_KEY from environment)
    adapter = OpenAIAdapter(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4",
        temperature=0.7
    )
    
    # Simple call
    prompt = "What is the capital of France?"
    response = adapter.call_model(prompt)
    print(f"Response: {response}")
    print(f"Latency: {adapter.last_latency_ms:.2f}ms")
    
    # Call with metadata
    system_prompt = "You are a helpful assistant."
    full_response = adapter.call_with_metadata(prompt, system_prompt)
    print(f"\nFull response metadata:")
    print(f"  Content: {full_response.content}")
    print(f"  Latency: {full_response.latency_ms:.2f}ms")
    print(f"  Model: {full_response.model}")
    print(f"  Tokens: {full_response.total_tokens} (prompt: {full_response.prompt_tokens}, completion: {full_response.completion_tokens})")
    
    # Get stats
    stats = adapter.get_stats()
    print(f"\nStats: {stats}")
