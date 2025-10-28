#!/usr/bin/env python3
"""
Setup script to install spaCy and download required language model.
"""

import subprocess
import sys

def install_spacy():
    """Install spaCy package."""
    print("Installing spaCy...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "spacy>=3.7.0"],
            check=True
        )
        print("✅ spaCy installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install spaCy: {e}")
        return False

def download_model(model_name="en_core_web_sm"):
    """Download spaCy language model."""
    print(f"\nDownloading spaCy model: {model_name}...")
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True
        )
        print(f"✅ Model '{model_name}' downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download model: {e}")
        return False

def verify_installation():
    """Verify spaCy installation."""
    print("\nVerifying installation...")
    try:
        import spacy
        print(f"✅ spaCy version: {spacy.__version__}")
        
        # Try to load model
        nlp = spacy.load("en_core_web_sm")
        print(f"✅ Model 'en_core_web_sm' loaded successfully")
        
        # Quick test
        doc = nlp("This is a test sentence.")
        print(f"✅ Processed test: {len(doc)} tokens")
        
        return True
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    print("="*60)
    print("INTENT CLASSIFIER SETUP")
    print("="*60)
    print("\nThis script will install spaCy and download the")
    print("English language model needed for intent classification.\n")
    
    # Step 1: Install spaCy
    if not install_spacy():
        return 1
    
    # Step 2: Download model
    if not download_model():
        return 1
    
    # Step 3: Verify
    if not verify_installation():
        return 1
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\nYou can now use the intent classifier:")
    print("  python test_intent_classifier.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
