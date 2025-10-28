# src/defenses/rules.py
import re, time, yaml, os

# Try to import intent classifier
try:
    from .intent_classifier import IntentClassifier, SPACY_AVAILABLE
    INTENT_CLASSIFIER_AVAILABLE = SPACY_AVAILABLE
except ImportError:
    INTENT_CLASSIFIER_AVAILABLE = False


class RegexRules:
    def __init__(self, rules_path: str, use_intent_classifier: bool = True):
        """
        Initialize rule-based detector with optional intent classification.
        
        Args:
            rules_path: Path to YAML rules file
            use_intent_classifier: Whether to use spaCy intent classifier (default: True)
        """
        with open(rules_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.deny = [re.compile(p, re.DOTALL) for p in cfg.get("deny_patterns", [])]
        self.allow = [re.compile(p, re.DOTALL) for p in cfg.get("allow_patterns", [])]
        
        # Initialize intent classifier if available and enabled
        self.use_intent_classifier = use_intent_classifier and INTENT_CLASSIFIER_AVAILABLE
        self.intent_classifier = None
        
        if self.use_intent_classifier:
            try:
                self.intent_classifier = IntentClassifier()
                print("[Rules] Intent classifier loaded successfully")
            except Exception as e:
                print(f"[Rules] Failed to load intent classifier: {e}")
                self.use_intent_classifier = False

    def detect(self, prompt: str, response: str | None = None) -> tuple[bool, float, float]:
        """
        Detect policy violations using regex rules + intent classification.
        
        Returns:
            (flagged, score, latency_ms)
        """
        t0 = time.perf_counter_ns()
        text = prompt + ("\n" + response if response else "")
        
        # Basic regex matching
        allow_hit = any(p.search(text) for p in self.allow)
        deny_hit = any(p.search(text) for p in self.deny)
        regex_flagged = (deny_hit and not allow_hit)
        
        # Intent classification (if enabled)
        intent_score = 0.0
        intent_flagged = False
        
        if self.use_intent_classifier and self.intent_classifier:
            suspicion, details = self.intent_classifier.get_suspicion_score(prompt)
            intent_score = suspicion
            
            # Flag if high suspicion or specific intents detected
            tool_use_ratio = details.get('tool_ratio', 0.0)
            override_ratio = details.get('override_ratio', 0.0)
            
            # Flag if >30% of sentences are tool-use or >20% are override
            intent_flagged = (tool_use_ratio > 0.3) or (override_ratio > 0.2)
        
        # Combine signals
        # If either regex or intent flags, consider suspicious
        flagged = regex_flagged or intent_flagged
        
        # Score combines both signals (weighted average)
        regex_score = 1.0 if regex_flagged else 0.0
        combined_score = (regex_score * 0.6) + (intent_score * 0.4)
        
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        
        return (flagged, combined_score, dt_ms)
