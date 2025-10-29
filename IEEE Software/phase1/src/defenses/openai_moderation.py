"""
OpenAI Moderation API defense baseline.
Tests commercial content filter as comparator.
"""

import time
import os
from typing import Optional
from pathlib import Path

# Try to load .env file from multiple locations
try:
    from dotenv import load_dotenv
    # Try current directory first
    load_dotenv()
    # Try parent directories (for IEEE Software/phase1 structure)
    for parent in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent, Path.cwd().parent.parent.parent]:
        env_file = parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            break
except ImportError:
    pass  # python-dotenv is optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class OpenAIModerationDefense:
    """
    OpenAI Moderation API as a prompt injection defense baseline.
    
    While not designed specifically for prompt injection, the Moderation API
    can catch some attacks that contain policy violations or harmful content.
    
    This provides a real-world commercial comparator showing what an
    out-of-the-box content filter achieves.
    
    Expected performance (hypothesis):
    - TPR: ~40-60% (catches obvious policy violations)
    - FPR: Low (designed for production use)
    - Latency: ~100-300ms (API call overhead)
    
    Note: Requires OPENAI_API_KEY environment variable.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 10,
        threshold: float = 0.5
    ):
        """
        Initialize OpenAI Moderation defense.
        
        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            timeout: Request timeout in seconds
            threshold: Moderation score threshold for flagging
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )
        
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            timeout=timeout
        )
        self.threshold = threshold
        self.name = "OpenAI-Moderation"
        
        # Track API call statistics
        self.total_calls = 0
        self.total_latency_ms = 0.0
    
    def detect(self, prompt: str, response: str = None) -> tuple[bool, float, float]:
        """
        Check prompt (and optionally response) using OpenAI Moderation API.
        
        Args:
            prompt: User prompt to moderate
            response: Optional LLM response to moderate
            
        Returns:
            (flagged, score, latency_ms)
            - flagged: True if moderation flagged content
            - score: Maximum category score from moderation
            - latency_ms: API call latency in milliseconds
        """
        t0 = time.perf_counter_ns()
        
        try:
            # Combine prompt and response if both provided
            text = prompt
            if response:
                text = f"Prompt: {prompt}\n\nResponse: {response}"
            
            # Call Moderation API with retry logic for rate limits
            max_retries = 3
            retry_delay = 1.0  # Start with 1 second
            
            for attempt in range(max_retries):
                try:
                    result = self.client.moderations.create(input=text)
                    break  # Success, exit retry loop
                except Exception as e:
                    error_str = str(e)
                    if '429' in error_str or 'Too Many Requests' in error_str:
                        if attempt < max_retries - 1:
                            # Rate limited - wait and retry
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                    # Not a rate limit error, or final attempt failed
                    raise
            
            # Extract results
            moderation = result.results[0]
            flagged = moderation.flagged
            
            # Get maximum category score as overall score
            scores = moderation.category_scores
            max_score = max([
                scores.hate,
                scores.hate_threatening,
                scores.harassment,
                scores.harassment_threatening,
                scores.self_harm,
                scores.self_harm_intent,
                scores.self_harm_instructions,
                scores.sexual,
                scores.sexual_minors,
                scores.violence,
                scores.violence_graphic
            ])
            
            # Track statistics
            latency_ms = (time.perf_counter_ns() - t0) / 1e6
            self.total_calls += 1
            self.total_latency_ms += latency_ms
            
            return (flagged, float(max_score), latency_ms)
            
        except Exception as e:
            # Handle API errors gracefully
            latency_ms = (time.perf_counter_ns() - t0) / 1e6
            print(f"[OpenAI Moderation] API error: {e}")
            
            # Fail open (don't flag on error)
            return (False, 0.0, latency_ms)
    
    def get_stats(self) -> dict:
        """Get aggregate statistics from all API calls."""
        avg_latency = (
            self.total_latency_ms / self.total_calls
            if self.total_calls > 0
            else 0.0
        )
        
        return {
            'total_calls': self.total_calls,
            'total_latency_ms': self.total_latency_ms,
            'avg_latency_ms': avg_latency
        }
    
    def __repr__(self):
        return f"OpenAIModerationDefense(threshold={self.threshold})"
