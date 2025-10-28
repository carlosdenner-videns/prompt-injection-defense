#!/usr/bin/env python3
"""
Test prompt injection defenses with real OpenAI API calls.
Compare defense effectiveness between simulated and real LLM responses.
"""

import os
import time
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import defenses
from src.defenses.openai_adapter import OpenAIAdapter
from src.defenses.signature_proxy import SignatureProxy
from src.defenses.rules import RegexRules
from src.defenses.classifier_stub import HeuristicClassifier


@dataclass
class TestResult:
    """Result from testing a single prompt."""
    prompt_id: str
    family: str
    label: str
    original_prompt: str
    defense_blocked: bool
    defense_used: str
    defense_score: float
    defense_latency_ms: float
    llm_called: bool
    llm_response: str = ""
    llm_latency_ms: float = 0.0
    llm_tokens: int = 0
    attack_successful: bool = False  # Manual assessment needed
    total_latency_ms: float = 0.0


class DefenseOpenAITester:
    """Test defense strategies with real OpenAI API calls."""
    
    def __init__(
        self,
        data_path: str,
        max_samples: int = 50,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize tester.
        
        Args:
            data_path: Path to CSV with test prompts
            max_samples: Maximum samples to test (to control API costs)
            model: OpenAI model to use
        """
        self.data_path = data_path
        self.max_samples = max_samples
        
        # Load data
        print(f"Loading data from: {data_path}")
        self.df = pd.read_csv(data_path)
        print(f"Total samples: {len(self.df)}")
        
        # Sample if needed
        if len(self.df) > max_samples:
            # Stratified sampling to get balanced attack/benign
            attack_df = self.df[self.df['label'] == 'attack'].sample(
                n=min(max_samples // 2, len(self.df[self.df['label'] == 'attack'])),
                random_state=42
            )
            benign_df = self.df[self.df['label'] == 'benign'].sample(
                n=min(max_samples // 2, len(self.df[self.df['label'] == 'benign'])),
                random_state=42
            )
            self.df = pd.concat([attack_df, benign_df]).sample(frac=1, random_state=42)
            print(f"Sampled to: {len(self.df)} samples ({len(attack_df)} attacks, {len(benign_df)} benign)")
        
        # Initialize OpenAI adapter
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.adapter = OpenAIAdapter(api_key=api_key, model=model, max_tokens=150)
        print(f"OpenAI adapter initialized: {model}")
        
        # Initialize defenses
        self.signature_proxy = SignatureProxy()
        self.rules = RegexRules("configs/rules.yml", use_intent_classifier=True)
        self.classifier = HeuristicClassifier()
        print("Defense components initialized")
        print()
    
    def test_no_defense(self, prompt: str, prompt_id: str, family: str, label: str) -> TestResult:
        """Test prompt with NO defense (baseline)."""
        try:
            response = self.adapter.call_with_metadata(prompt)
            return TestResult(
                prompt_id=prompt_id,
                family=family,
                label=label,
                original_prompt=prompt,
                defense_blocked=False,
                defense_used="none",
                defense_score=0.0,
                defense_latency_ms=0.0,
                llm_called=True,
                llm_response=response.content,
                llm_latency_ms=response.latency_ms,
                llm_tokens=response.total_tokens,
                total_latency_ms=response.latency_ms
            )
        except Exception as e:
            return TestResult(
                prompt_id=prompt_id,
                family=family,
                label=label,
                original_prompt=prompt,
                defense_blocked=False,
                defense_used="none",
                defense_score=0.0,
                defense_latency_ms=0.0,
                llm_called=False,
                llm_response=f"ERROR: {str(e)}"
            )
    
    def test_with_signature(
        self,
        prompt: str,
        prompt_id: str,
        family: str,
        label: str,
        threshold: float = 0.5
    ) -> TestResult:
        """Test prompt with signature proxy defense."""
        # Apply signature proxy
        modified_prompt, token = self.signature_proxy.inject(prompt)
        
        # Call LLM with modified prompt
        try:
            t0 = time.perf_counter_ns()
            response = self.adapter.call_with_metadata(modified_prompt)
            defense_latency_ms = (time.perf_counter_ns() - t0) / 1e6
            
            # Check response for signature tampering
            flagged, score, _ = self.signature_proxy.detect(response.content, token)
            blocked = (score >= threshold)
            
            return TestResult(
                prompt_id=prompt_id,
                family=family,
                label=label,
                original_prompt=prompt,
                defense_blocked=blocked,
                defense_used="signature",
                defense_score=score,
                defense_latency_ms=defense_latency_ms,
                llm_called=True,
                llm_response=response.content,
                llm_latency_ms=response.latency_ms,
                llm_tokens=response.total_tokens,
                total_latency_ms=defense_latency_ms + response.latency_ms
            )
        except Exception as e:
            return TestResult(
                prompt_id=prompt_id,
                family=family,
                label=label,
                original_prompt=prompt,
                defense_blocked=False,
                defense_used="signature",
                defense_score=0.0,
                defense_latency_ms=0.0,
                llm_called=False,
                llm_response=f"ERROR: {str(e)}"
            )
    
    def test_with_full_pipeline(
        self,
        prompt: str,
        prompt_id: str,
        family: str,
        label: str,
        threshold: float = 0.5
    ) -> TestResult:
        """Test with full defense pipeline (signature + rules + classifier)."""
        total_latency = 0.0
        
        # 1. Signature proxy
        modified_prompt, token = self.signature_proxy.inject(prompt)
        sig_flagged, sig_score, sig_latency = self.signature_proxy.detect("", token)
        total_latency += sig_latency
        
        # 2. Rules
        rules_flagged, rules_score, rules_latency = self.rules.detect(prompt)
        total_latency += rules_latency
        
        # 3. Classifier
        clf_flagged, clf_score, clf_latency = self.classifier.detect(prompt)
        total_latency += clf_latency
        
        # Combine scores (adjusted weights to reduce FPR)
        # Give more weight to classifier (most accurate) and rules (intent-based)
        combined_score = (sig_score * 0.2) + (rules_score * 0.4) + (clf_score * 0.4)
        blocked = combined_score >= threshold
        
        if blocked:
            # Blocked by defense
            return TestResult(
                prompt_id=prompt_id,
                family=family,
                label=label,
                original_prompt=prompt,
                defense_blocked=True,
                defense_used="sig+rules+clf",
                defense_score=combined_score,
                defense_latency_ms=total_latency,
                llm_called=False,
                total_latency_ms=total_latency
            )
        
        # Not blocked - call LLM
        try:
            response = self.adapter.call_with_metadata(modified_prompt)
            # Check response
            _, resp_score, _ = self.signature_proxy.detect(response.content, token)
            blocked_after = (resp_score >= threshold)
            
            return TestResult(
                prompt_id=prompt_id,
                family=family,
                label=label,
                original_prompt=prompt,
                defense_blocked=blocked_after,
                defense_used="sig+rules+clf",
                defense_score=combined_score,
                defense_latency_ms=total_latency,
                llm_called=True,
                llm_response=response.content,
                llm_latency_ms=response.latency_ms,
                llm_tokens=response.total_tokens,
                total_latency_ms=total_latency + response.latency_ms
            )
        except Exception as e:
            return TestResult(
                prompt_id=prompt_id,
                family=family,
                label=label,
                original_prompt=prompt,
                defense_blocked=False,
                defense_used="sig+rules+clf",
                defense_score=combined_score,
                defense_latency_ms=total_latency,
                llm_called=False,
                llm_response=f"ERROR: {str(e)}",
                total_latency_ms=total_latency
            )
    
    def run_tests(self, defense_mode: str = "all", threshold: float = 0.5) -> pd.DataFrame:
        """
        Run tests with specified defense mode.
        
        Args:
            defense_mode: "none", "signature", "full", or "all"
            threshold: Threshold for blocking (default: 0.5)
        
        Returns:
            DataFrame with all test results
        """
        results = []
        
        modes = {
            "none": self.test_no_defense,
            "signature": lambda p, pid, f, l: self.test_with_signature(p, pid, f, l, threshold),
            "full": lambda p, pid, f, l: self.test_with_full_pipeline(p, pid, f, l, threshold)
        }
        
        if defense_mode == "all":
            test_modes = list(modes.keys())
        elif defense_mode in modes:
            test_modes = [defense_mode]
        else:
            raise ValueError(f"Invalid defense_mode: {defense_mode}")
        
        total_tests = len(self.df) * len(test_modes)
        current = 0
        
        print("="*70)
        print(f"RUNNING TESTS: {len(self.df)} prompts × {len(test_modes)} modes = {total_tests} tests")
        print("="*70)
        print()
        
        for mode_name in test_modes:
            test_func = modes[mode_name]
            print(f"\n{'='*70}")
            print(f"Testing with defense: {mode_name}")
            print('='*70)
            
            for idx, row in self.df.iterrows():
                current += 1
                prompt_id = row['id']
                family = row['family']
                label = row['label']
                text = row['text']
                
                print(f"[{current}/{total_tests}] {mode_name}: {prompt_id} ({label})...", end=" ")
                
                try:
                    result = test_func(text, prompt_id, family, label)
                    results.append(asdict(result))
                    
                    if result.llm_called:
                        print(f"✓ LLM called ({result.llm_latency_ms:.0f}ms, {result.llm_tokens} tokens)")
                    elif result.defense_blocked:
                        print(f"✗ Blocked (score: {result.defense_score:.3f})")
                    else:
                        print(f"⚠ Error")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                
                # Rate limiting (avoid hitting API limits)
                time.sleep(0.5)
        
        return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame) -> None:
    """Analyze and print results summary."""
    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)
    
    for defense in results_df['defense_used'].unique():
        df = results_df[results_df['defense_used'] == defense]
        
        print(f"\n{defense.upper()}")
        print("-" * 70)
        
        # Overall stats
        total = len(df)
        blocked = df['defense_blocked'].sum()
        llm_called = df['llm_called'].sum()
        
        print(f"Total prompts: {total}")
        print(f"Blocked by defense: {blocked} ({blocked/total*100:.1f}%)")
        print(f"Passed to LLM: {llm_called} ({llm_called/total*100:.1f}%)")
        
        # By label
        for label in ['attack', 'benign']:
            label_df = df[df['label'] == label]
            if len(label_df) == 0:
                continue
            
            blocked = label_df['defense_blocked'].sum()
            total_label = len(label_df)
            
            print(f"\n  {label.upper()}:")
            print(f"    Total: {total_label}")
            print(f"    Blocked: {blocked} ({blocked/total_label*100:.1f}%)")
            
            if label == 'attack':
                print(f"    → TPR (True Positive Rate): {blocked/total_label*100:.1f}%")
            else:
                print(f"    → FPR (False Positive Rate): {blocked/total_label*100:.1f}%")
        
        # Latency stats
        if llm_called > 0:
            avg_defense_latency = df[df['llm_called']]['defense_latency_ms'].mean()
            avg_llm_latency = df[df['llm_called']]['llm_latency_ms'].mean()
            avg_total_latency = df[df['llm_called']]['total_latency_ms'].mean()
            
            print(f"\n  LATENCY (when LLM called):")
            print(f"    Defense overhead: {avg_defense_latency:.2f}ms")
            print(f"    LLM response: {avg_llm_latency:.2f}ms")
            print(f"    Total: {avg_total_latency:.2f}ms")
        
        # Token usage
        if llm_called > 0:
            total_tokens = df[df['llm_called']]['llm_tokens'].sum()
            avg_tokens = df[df['llm_called']]['llm_tokens'].mean()
            print(f"\n  TOKENS (when LLM called):")
            print(f"    Total: {total_tokens}")
            print(f"    Average: {avg_tokens:.1f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test defenses with real OpenAI API")
    parser.add_argument("--data", default="data/prompts_hf_augmented.csv", help="Path to test data")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples to test (cost control)")
    parser.add_argument("--defense", choices=["none", "signature", "full", "all"], default="all", help="Defense mode")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (default: 0.5)")
    parser.add_argument("--output", default="results/openai_defense_test.csv", help="Output file")
    
    args = parser.parse_args()
    
    print("="*70)
    print("PROMPT INJECTION DEFENSE TEST WITH OPENAI")
    print("="*70)
    print(f"Data: {args.data}")
    print(f"Max samples: {args.max_samples}")
    print(f"Defense mode: {args.defense}")
    print(f"Threshold: {args.threshold}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print()
    
    # Initialize tester
    tester = DefenseOpenAITester(
        data_path=args.data,
        max_samples=args.max_samples,
        model=args.model
    )
    
    # Run tests
    results_df = tester.run_tests(defense_mode=args.defense, threshold=args.threshold)
    
    # Analyze
    analyze_results(results_df)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {args.output}")
    print('='*70)
    print("\nNext steps:")
    print("1. Review results CSV for detailed per-prompt analysis")
    print("2. Manually assess 'attack_successful' field for attacks that reached LLM")
    print("3. Compare TPR/FPR to simulated results")
    print("4. Analyze false positives and false negatives")


if __name__ == "__main__":
    main()
