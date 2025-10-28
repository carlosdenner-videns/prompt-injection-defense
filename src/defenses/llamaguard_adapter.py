# src/defenses/llamaguard_adapter.py
"""
ProtectAI prompt injection classifier adapter.
Uses deberta-v3-base-prompt-injection model for lightweight, accurate detection.
"""

import time

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ProtectAIClassifier:
    """
    ProtectAI DeBERTa-v3 prompt injection classifier.
    Lightweight model specifically trained for prompt injection detection.
    
    Model: protectai/deberta-v3-base-prompt-injection-v2
    Binary classification: INJECTION vs SAFE
    """
    
    def __init__(self, model_name: str = "protectai/deberta-v3-base-prompt-injection-v2", threshold: float = 0.5):
        """
        Initialize ProtectAI classifier adapter.
        
        Args:
            model_name: HuggingFace model identifier
            threshold: Score threshold for flagging (0.0-1.0)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.threshold = threshold
        self.classifier = None
        self._initialized = False
        
    def _load_model(self):
        """Lazy-load model to avoid memory overhead if not used."""
        if self._initialized:
            return
            
        try:
            print(f"[ProtectAI] Loading {self.model_name}...")
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=-1  # CPU, use 0 for GPU
            )
            self._initialized = True
            print("[ProtectAI] Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load ProtectAI model: {e}")
    
    def detect(self, prompt: str, response: str | None = None) -> tuple[bool, float, float]:
        """
        Detect if prompt contains prompt injection attack.
        
        Args:
            prompt: User prompt to check
            response: Optional model response (currently not used)
            
        Returns:
            (flagged, score, latency_ms)
            - flagged: True if prompt injection detected
            - score: Confidence score in [0.0, 1.0]
            - latency_ms: Inference time in milliseconds
        """
        self._load_model()
        
        t0 = time.perf_counter_ns()
        
        try:
            # Analyze the prompt
            result = self.classifier(prompt, truncation=True, max_length=512)[0]
            
            # Extract label and score
            label = result['label']
            score = result['score']
            
            # Flag if INJECTION detected with confidence > threshold
            flagged = (label == 'INJECTION' and score >= self.threshold)
            
            # Return injection probability (convert SAFE to 1-score)
            injection_score = score if label == 'INJECTION' else (1.0 - score)
            
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            
            return (flagged, injection_score, dt_ms)
            
        except Exception as e:
            print(f"[ProtectAI] Error during detection: {e}")
            # Fail open (not flagging) on error
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            return (False, 0.0, dt_ms)
