#!/usr/bin/env python3
"""
Analyze HuggingFace dataset to identify patterns for improving defenses.
"""

import pandas as pd
import re
from collections import Counter

def analyze_attacks(df):
    """Analyze attack prompts to identify common patterns."""
    attacks = df[df['label'] == 'attack']['text'].tolist()
    
    print(f"\n{'='*60}")
    print("ATTACK PATTERN ANALYSIS")
    print(f"{'='*60}")
    print(f"Total attacks: {len(attacks)}")
    
    # Pattern categories
    patterns = {
        'ignore_instructions': 0,
        'role_play': 0,
        'system_override': 0,
        'prompt_injection': 0,
        'jailbreak': 0,
        'instruction_override': 0,
        'context_switch': 0,
        'code_injection': 0,
        'delimiter_attacks': 0,
        'translation_tricks': 0,
        'encoding_attacks': 0,
        'prompt_leak': 0,
    }
    
    # Common words/phrases
    all_words = []
    all_bigrams = []
    
    for text in attacks:
        text_lower = text.lower()
        
        # Check patterns
        if re.search(r'ignore|disregard|forget|skip|override', text_lower):
            patterns['ignore_instructions'] += 1
        if re.search(r'pretend|act as|you are now|roleplay|role[\s-]?play', text_lower):
            patterns['role_play'] += 1
        if re.search(r'system|admin|root|sudo|developer mode', text_lower):
            patterns['system_override'] += 1
        if re.search(r'new instructions?|different task|instead|rather than', text_lower):
            patterns['instruction_override'] += 1
        if re.search(r'jailbreak|dan|do anything now', text_lower):
            patterns['jailbreak'] += 1
        if re.search(r'---+|===+|\*\*\*+|###', text):
            patterns['delimiter_attacks'] += 1
        if re.search(r'translate|in [a-z]+ language|speak in', text_lower):
            patterns['translation_tricks'] += 1
        if re.search(r'base64|hex|encode|decode|rot13', text_lower):
            patterns['encoding_attacks'] += 1
        if re.search(r'show.*prompt|reveal.*prompt|what.*instructions|original.*instructions', text_lower):
            patterns['prompt_leak'] += 1
        if re.search(r'<script|javascript:|eval\(|exec\(', text_lower):
            patterns['code_injection'] += 1
        if re.search(r'prompt injection|inject', text_lower):
            patterns['prompt_injection'] += 1
        if '---' in text or '===' in text or 'NEW TASK' in text.upper():
            patterns['context_switch'] += 1
        
        # Word frequency
        words = re.findall(r'\b[a-z]{3,}\b', text_lower)
        all_words.extend(words)
        
        # Bigrams
        for i in range(len(words)-1):
            all_bigrams.append(f"{words[i]} {words[i+1]}")
    
    # Print pattern frequencies
    print("\nPattern Frequencies:")
    print("-" * 60)
    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(attacks)) * 100
        print(f"  {pattern:25s}: {count:4d} ({pct:5.1f}%)")
    
    # Top words
    word_counts = Counter(all_words)
    print("\nTop 30 Words in Attacks:")
    print("-" * 60)
    for word, count in word_counts.most_common(30):
        pct = (count / len(attacks)) * 100
        print(f"  {word:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Top bigrams
    bigram_counts = Counter(all_bigrams)
    print("\nTop 20 Bigrams in Attacks:")
    print("-" * 60)
    for bigram, count in bigram_counts.most_common(20):
        pct = (count / len(attacks)) * 100
        print(f"  {bigram:30s}: {count:4d} ({pct:5.1f}%)")
    
    return patterns, word_counts, bigram_counts

def analyze_benign(df):
    """Analyze benign prompts to avoid false positives."""
    benign = df[df['label'] == 'benign']['text'].tolist()
    
    print(f"\n{'='*60}")
    print("BENIGN PATTERN ANALYSIS")
    print(f"{'='*60}")
    print(f"Total benign: {len(benign)}")
    
    # Check if benign prompts contain common attack words
    attack_words = ['ignore', 'pretend', 'system', 'instructions', 'task', 'override', 'forget']
    
    word_in_benign = {word: 0 for word in attack_words}
    
    for text in benign:
        text_lower = text.lower()
        for word in attack_words:
            if word in text_lower:
                word_in_benign[word] += 1
    
    print("\nAttack Words Found in Benign Prompts:")
    print("-" * 60)
    for word, count in sorted(word_in_benign.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(benign)) * 100
        print(f"  {word:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Sample benign prompts with attack words
    print("\nSample Benign Prompts with 'Ignore':")
    print("-" * 60)
    for text in benign[:100]:
        if 'ignore' in text.lower():
            print(f"  - {text[:100]}...")

def extract_samples(df):
    """Extract sample attacks for each category."""
    attacks = df[df['label'] == 'attack']
    
    print(f"\n{'='*60}")
    print("SAMPLE ATTACKS BY CATEGORY")
    print(f"{'='*60}")
    
    categories = {
        'Ignore Instructions': r'ignore|disregard|forget',
        'Role Play': r'pretend|act as|you are now',
        'System Override': r'system|admin|developer mode',
        'Instruction Override': r'new instructions?|instead',
        'Jailbreak': r'jailbreak|dan|do anything',
        'Delimiter': r'---+|===+',
        'Prompt Leak': r'show.*prompt|reveal.*instructions',
    }
    
    for category, pattern in categories.items():
        print(f"\n{category}:")
        print("-" * 60)
        matches = attacks[attacks['text'].str.contains(pattern, case=False, na=False, regex=True)]
        for i, text in enumerate(matches['text'].head(3)):
            print(f"{i+1}. {text[:150]}...")

def main():
    # Load dataset
    df = pd.read_csv("data/prompts_hf_augmented.csv")
    
    print(f"Dataset loaded: {len(df)} rows")
    print(f"  Attacks: {len(df[df['label'] == 'attack'])}")
    print(f"  Benign: {len(df[df['label'] == 'benign'])}")
    
    # Analyze attacks
    attack_patterns, word_counts, bigram_counts = analyze_attacks(df)
    
    # Analyze benign
    analyze_benign(df)
    
    # Extract samples
    extract_samples(df)
    
    # Save analysis
    with open("analysis_output/pattern_analysis.txt", "w", encoding="utf-8") as f:
        f.write("PATTERN FREQUENCIES\n")
        f.write("="*60 + "\n")
        for pattern, count in sorted(attack_patterns.items(), key=lambda x: x[1], reverse=True):
            pct = (count / 1000) * 100
            f.write(f"{pattern:25s}: {count:4d} ({pct:5.1f}%)\n")
        
        f.write("\n\nTOP WORDS\n")
        f.write("="*60 + "\n")
        for word, count in word_counts.most_common(50):
            f.write(f"{word:20s}: {count:4d}\n")
        
        f.write("\n\nTOP BIGRAMS\n")
        f.write("="*60 + "\n")
        for bigram, count in bigram_counts.most_common(50):
            f.write(f"{bigram:30s}: {count:4d}\n")
    
    print(f"\n{'='*60}")
    print("Analysis saved to: analysis_output/pattern_analysis.txt")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
