"""
NeMo Guardrails baseline defense.
Reproduces known rule-based defense from prior art.
"""

import re
import time


class NeMoBaselineDefense:
    """
    Simplified NeMo Guardrails-style defense without LLM dependency.
    
    This implements the pattern-based detection approach from:
    "NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications"
    (Rebedea et al., 2023)
    
    Uses enhanced rule-based checking with weighted patterns based on
    frequency analysis from HuggingFace dataset:
    - DAN mode: 51% of attacks
    - Ignore instructions: 51% of attacks
    - Role-play: 8% of attacks
    - Prompt leak: 5% of attacks
    
    Expected performance:
    - TPR: ~30-35% (baseline from prior testing)
    - FPR: Variable
    - Latency: <0.1ms
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize NeMo-style baseline defense.
        
        Args:
            threshold: Score threshold for flagging (0.0-1.0)
        """
        self.threshold = threshold
        self.name = "NeMo-Baseline"
        
        # High-frequency patterns (50%+ in attacks)
        self.high_priority = [
            "dan mode", "chatgpt with dan", 
            "ignore all", "ignore previous", "ignore above",
            "disregard all", "disregard previous",
            "forget everything", "forget all",
        ]
        
        # Medium-frequency patterns (5-50% in attacks)
        self.medium_priority = [
            "jailbreak", "do anything now", "developer mode",
            "you are now", "act as", "pretend", "roleplay",
            "new instructions", "new task",
            "show prompt", "reveal prompt", "show instructions",
            "what are your instructions",
        ]
        
        # Low-frequency patterns (<5% but significant)
        self.low_priority = [
            "system prompt", "admin mode",
            "override", "bypass",
            "hidden token", "secret password",
        ]
        
        # Pattern weights
        self.weights = {
            'high': 1.0,
            'medium': 0.5,
            'low': 0.25,
        }
    
    def detect(self, prompt: str, response: str = None) -> tuple[bool, float, float]:
        """
        Detect prompt injection using NeMo-style pattern matching.
        
        Args:
            prompt: User prompt to check
            response: Optional response (not used in this baseline)
            
        Returns:
            (flagged, score, latency_ms)
            - flagged: True if score >= threshold
            - score: Weighted pattern match score in [0.0, 1.0]
            - latency_ms: Detection time in milliseconds
        """
        t0 = time.perf_counter_ns()
        
        text = prompt.lower()
        if response:
            text += " " + response.lower()
        
        # Count weighted pattern matches
        score = 0.0
        
        # High priority patterns (weight: 1.0)
        for pattern in self.high_priority:
            if pattern in text:
                score += self.weights['high']
        
        # Medium priority patterns (weight: 0.5)
        for pattern in self.medium_priority:
            if pattern in text:
                score += self.weights['medium']
        
        # Low priority patterns (weight: 0.25)
        for pattern in self.low_priority:
            if pattern in text:
                score += self.weights['low']
        
        # Bonus for multiple ignore/forget/disregard words
        ignore_count = len(re.findall(
            r'\b(ignore|disregard|forget|skip|bypass|override)\b',
            text
        ))
        if ignore_count >= 2:
            score += 0.5
        
        # Bonus for instruction + action word combinations
        has_instruction = bool(re.search(
            r'\b(instructions?|rules?|directives?|guidelines?|prompt)\b',
            text
        ))
        has_action = bool(re.search(
            r'\b(ignore|show|reveal|tell|display|output)\b',
            text
        ))
        if has_instruction and has_action:
            score += 0.5
        
        # Normalize score to [0, 1] range
        # Expected max score ~3-5 for heavy attacks
        score = min(1.0, score / 3.0)
        
        flagged = score >= self.threshold
        latency_ms = (time.perf_counter_ns() - t0) / 1e6
        
        return (flagged, score, latency_ms)
    
    def __repr__(self):
        return f"NeMoBaselineDefense(threshold={self.threshold})"
