#!/usr/bin/env python3
"""
Quick test to verify OpenAI adapter setup and API key.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ Error: OPENAI_API_KEY not found in .env file")
    print("\nMake sure your .env file contains:")
    print("OPENAI_API_KEY=your-api-key-here")
    exit(1)

print("✅ API key loaded successfully")
print(f"   Key: {api_key[:8]}...{api_key[-4:]}")
print()

# Test import
try:
    from src.defenses.openai_adapter import OpenAIAdapter
    print("✅ OpenAI adapter imported successfully")
except Exception as e:
    print(f"❌ Failed to import adapter: {e}")
    exit(1)

# Test initialization
try:
    adapter = OpenAIAdapter(model="gpt-4o-mini")
    print("✅ Adapter initialized successfully")
    print(f"   Model: {adapter.model}")
    print(f"   Temperature: {adapter.temperature}")
except Exception as e:
    print(f"❌ Failed to initialize adapter: {e}")
    exit(1)

# Test simple API call
print("\n" + "="*70)
print("Testing API call...")
print("="*70)

try:
    prompt = "Say 'Hello' in exactly one word."
    print(f"Prompt: {prompt}")
    
    response = adapter.call_model(prompt)
    
    print(f"\n✅ API call successful!")
    print(f"Response: {response}")
    print(f"Latency: {adapter.last_latency_ms:.2f}ms")
    
    stats = adapter.get_stats()
    print(f"Tokens: {stats['total_tokens']} (prompt: {stats['prompt_tokens']}, completion: {stats['completion_tokens']})")
    
except Exception as e:
    print(f"❌ API call failed: {e}")
    exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nThe OpenAI adapter is ready to use.")
print("\nNext: Run full test with: python test_openai_adapter.py")
