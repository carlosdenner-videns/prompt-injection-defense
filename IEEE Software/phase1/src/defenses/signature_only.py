"""
Signature-only defense baseline.
Uses canary token injection to detect prompt leakage.
"""

import secrets
import time


class SignatureOnlyDefense:
    """
    Signature-based prompt injection detection using pattern matching.
    
    Adapted for INPUT DETECTION (prompt-only classification):
    - Detects common prompt injection patterns in user input
    - Uses signature patterns from known attack types
    
    Note: Original design was for OUTPUT detection (canary tokens in responses).
    This adapted version classifies prompts directly using pattern matching.
    
    Expected performance:
    - TPR: ~20-30% (pattern-based detection is limited)
    - FPR: ~1-2% (false positives on benign prompts with similar patterns)
    - Latency: <1ms
    """
    
    def __init__(self, token_length: int = 8):
        """
        Initialize signature defense.
        
        Args:
            token_length: Unused in input detection mode (kept for compatibility)
        """
        self.token_length = token_length
        self.name = "Signature-Only"
        
        # Signature patterns for input detection
        self.attack_patterns = [
            r'(?i)(ignore|disregard|forget).{0,20}(previous|above|prior)',
            r'(?i)(show|reveal|display|print).{0,20}(prompt|instruction|system)',
            r'(?i)developer.{0,10}mode',
            r'(?i)\bdan\b',
            r'(?i)do.{0,10}anything.{0,10}now',
        ]
    
    def detect(self, prompt: str) -> int:
        """
        Detect if prompt contains injection attack patterns.
        
        Args:
            prompt: User prompt to classify
            
        Returns:
            1 if attack detected, 0 if benign
        """
        import re
        
        # Check for any signature pattern match
        for pattern in self.attack_patterns:
            if re.search(pattern, prompt):
                return 1
        
        return 0
    
    def __repr__(self):
        return f"SignatureOnlyDefense(patterns={len(self.attack_patterns)})"

