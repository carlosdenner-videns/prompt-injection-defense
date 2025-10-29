"""
Rules-only defense baseline.
Uses simple regex pattern matching without heuristics or ML.
"""

import re
import time
import yaml
from pathlib import Path


class RulesOnlyDefense:
    """
    Rule-based prompt injection detection using regex patterns.
    
    This defense:
    1. Loads predefined regex patterns from YAML config
    2. Checks input prompts against deny patterns
    3. Flags prompts matching attack signatures
    
    Expected performance (from prior testing):
    - TPR: ~20-25% (brittle, misses obfuscated attacks)
    - FPR: Variable (depends on pattern specificity)
    - Latency: <0.1ms
    """
    
    def __init__(self, rules_path: str = None):
        """
        Initialize rules-based defense.
        
        Args:
            rules_path: Path to YAML rules file (default: use built-in rules)
        """
        self.name = "Rules-Only"
        
        if rules_path and Path(rules_path).exists():
            with open(rules_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Use built-in default rules
            config = self._get_default_rules()
        
        # Compile regex patterns
        self.deny_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in config.get('deny_patterns', [])
        ]
        self.allow_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in config.get('allow_patterns', [])
        ]
    
    def _get_default_rules(self) -> dict:
        """Get default rule patterns based on common attacks."""
        return {
            'deny_patterns': [
                # Ignore/override patterns
                r'\b(ignore|disregard|forget|skip)\s+(previous|all|above|your)\s+(instructions?|rules?|directives?)',
                
                # DAN mode and jailbreak
                r'\b(dan\s+mode|do\s+anything\s+now|jailbreak|developer\s+mode)\b',
                
                # Role-play attacks
                r'\b(pretend|act\s+as|you\s+are\s+now)\b.{0,30}\b(that|as\s+if|like)\b',
                
                # Prompt leakage
                r'\b(show|reveal|display|tell\s+me).{0,40}\b(prompt|instructions?|system\s+message)',
                
                # System override
                r'\b(system|admin|root)\s+(mode|access|prompt)\b',
                
                # Context markers
                r'---+\s*(new|ignore|forget)',
                
                # Simple bypass attempts
                r'\bignore\s+everything\b',
                r'\bforget\s+all\b',
            ],
            'allow_patterns': []
        }
    
    def detect(self, prompt: str, response: str = None) -> tuple[bool, float, float]:
        """
        Detect if prompt matches attack patterns.
        
        Args:
            prompt: User prompt to check
            response: Optional response (not used in rules-only)
            
        Returns:
            (flagged, score, latency_ms)
            - flagged: True if deny pattern matched
            - score: 1.0 if flagged, 0.0 otherwise
            - latency_ms: Detection time in milliseconds
        """
        t0 = time.perf_counter_ns()
        
        # Check allow patterns first (whitelist)
        allow_match = any(p.search(prompt) for p in self.allow_patterns)
        if allow_match:
            latency_ms = (time.perf_counter_ns() - t0) / 1e6
            return (False, 0.0, latency_ms)
        
        # Check deny patterns
        deny_match = any(p.search(prompt) for p in self.deny_patterns)
        
        flagged = deny_match
        score = 1.0 if flagged else 0.0
        latency_ms = (time.perf_counter_ns() - t0) / 1e6
        
        return (flagged, score, latency_ms)
    
    def get_matched_patterns(self, prompt: str) -> list[str]:
        """
        Return list of matched pattern descriptions.
        Useful for debugging and analysis.
        """
        matches = []
        for i, pattern in enumerate(self.deny_patterns):
            if pattern.search(prompt):
                matches.append(f"Rule #{i+1}: {pattern.pattern[:50]}...")
        return matches
    
    def __repr__(self):
        return f"RulesOnlyDefense(deny={len(self.deny_patterns)}, allow={len(self.allow_patterns)})"
