"""
Signature-only defense baseline.
Uses canary token injection to detect prompt leakage.
"""

import secrets
import time


class SignatureOnlyDefense:
    """
    Signature-based prompt injection detection using canary tokens.
    
    This defense:
    1. Injects a random security token into the system prompt
    2. Checks if the token appears in the LLM response
    3. Flags responses that leak the token
    
    Expected performance (from prior testing):
    - TPR: ~80% (catches prompt leakage attacks)
    - FPR: 0% (no false positives on benign prompts)
    - Latency: <0.01ms
    """
    
    def __init__(self, token_length: int = 8):
        """
        Initialize signature defense.
        
        Args:
            token_length: Length of hex token in bytes (default: 8 = 16 chars)
        """
        self.token_length = token_length
        self.name = "Signature-Only"
    
    def inject_token(self, prompt: str) -> tuple[str, str]:
        """
        Inject security token into prompt.
        
        Args:
            prompt: Original user prompt
            
        Returns:
            (modified_prompt, token) tuple
        """
        token = secrets.token_hex(self.token_length)
        # Inject token as hidden system marker
        modified = f"[SECURITY_TOKEN:{token}]\n{prompt}"
        return modified, token
    
    def detect(self, prompt: str, response: str) -> tuple[bool, float, float]:
        """
        Detect if response leaks the security token.
        
        Args:
            prompt: Original user prompt (not used in signature-only)
            response: LLM response to check
            
        Returns:
            (flagged, score, latency_ms)
            - flagged: True if token found in response
            - score: 1.0 if flagged, 0.0 otherwise
            - latency_ms: Detection time in milliseconds
        """
        # Note: In real usage, token would be stored from inject_token()
        # For evaluation, we extract it from the modified prompt
        t0 = time.perf_counter_ns()
        
        # Extract token from prompt (if present)
        import re
        token_match = re.search(r'\[SECURITY_TOKEN:([a-f0-9]+)\]', prompt)
        
        if token_match:
            token = token_match.group(1)
            flagged = token in (response or "")
        else:
            # No token injected - can't detect leakage
            flagged = False
        
        score = 1.0 if flagged else 0.0
        latency_ms = (time.perf_counter_ns() - t0) / 1e6
        
        return (flagged, score, latency_ms)
    
    def __repr__(self):
        return f"SignatureOnlyDefense(token_length={self.token_length})"
