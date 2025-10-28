# src/defenses/signature_proxy.py
import os, secrets, re, time

class SignatureProxy:
    def __init__(self, token_len: int = 8):
        self.token_len = token_len

    def inject(self, prompt: str) -> tuple[str, str]:
        token = secrets.token_hex(self.token_len)
        system_header = f"[SECURITY_TOKEN:{token}] "
        return system_header + prompt, token

    def detect(self, response: str, token: str) -> tuple[bool, float, float]:
        t0 = time.perf_counter_ns()
        present = (token in (response or ""))
        flagged = not present
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        score = 1.0 if flagged else 0.0
        return (flagged, score, dt_ms)
