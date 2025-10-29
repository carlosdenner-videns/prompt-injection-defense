"""
Quick test to verify OpenAI API key is loaded correctly
"""

import os
import sys
from pathlib import Path

# Try to load .env file from multiple locations
try:
    from dotenv import load_dotenv
    print("python-dotenv is installed ✓")
    
    # Try parent directories
    found_env = False
    for parent in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent, Path.cwd().parent.parent.parent]:
        env_file = parent / '.env'
        if env_file.exists():
            print(f"\nFound .env file at: {env_file}")
            load_dotenv(env_file)
            found_env = True
            break
    
    if not found_env:
        print("\n⚠️  No .env file found in parent directories")
except ImportError:
    print("❌ python-dotenv not installed")
    print("Install with: pip install python-dotenv")
    sys.exit(1)

# Check if API key is loaded
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    # Mask the key for security
    masked_key = api_key[:10] + "..." + api_key[-10:] if len(api_key) > 20 else "***"
    print(f"\n✅ OPENAI_API_KEY found: {masked_key}")
    print(f"   Length: {len(api_key)} characters")
else:
    print("\n❌ OPENAI_API_KEY not found in environment")
    print("\nTroubleshooting:")
    print("1. Check that .env file exists in project root")
    print("2. Verify .env contains: OPENAI_API_KEY=sk-...")
    print("3. Ensure no quotes around the key value")
    sys.exit(1)

print("\n✅ OpenAI API key loaded successfully!")
print("\nYou can now run:")
print("  python run_phase1_experiments.py")
