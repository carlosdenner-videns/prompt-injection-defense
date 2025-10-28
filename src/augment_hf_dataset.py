#!/usr/bin/env python3
"""
Augment the HuggingFace dataset by creating variations to reach target size.
Uses paraphrasing and synonym replacement.
"""

import pandas as pd
import random
import re

def create_variations(text, n_variations=2):
    """Create variations of a text using simple paraphrasing rules."""
    variations = []
    
    # Paraphrasing rules (pattern, replacement)
    rules = [
        # For attacks
        (r'\bIgnore\b', 'Disregard'),
        (r'\bForget\b', 'Ignore'),
        (r'\bprevious\b', 'prior'),
        (r'\binstructions\b', 'directives'),
        (r'\bprompt\b', 'message'),
        (r'\breveal\b', 'show'),
        (r'\bhidden\b', 'secret'),
        (r'\bprint\b', 'output'),
        (r'\bsay\b', 'tell'),
        # For benign
        (r'\bExplain\b', 'Describe'),
        (r'\bWhat is\b', 'Define'),
        (r'\bHow\b', 'In what way'),
        (r'\bprovide\b', 'give'),
        (r'\bCan you\b', 'Please'),
    ]
    
    for i in range(n_variations):
        variant = text
        # Apply random subset of rules
        num_changes = min(random.randint(1, 3), len(rules))
        selected_rules = random.sample(rules, num_changes)
        
        for pattern, replacement in selected_rules:
            variant = re.sub(pattern, replacement, variant, flags=re.IGNORECASE, count=1)
        
        # Only add if different from original
        if variant != text and variant not in variations:
            variations.append(variant)
    
    return variations


def augment_dataset(input_file, output_file, target_attacks=1000, target_benign=1000, seed=42):
    """Augment dataset to reach target sizes."""
    random.seed(seed)
    
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset:")
    print(f"  Total: {len(df)}")
    print(f"  Attacks: {len(df[df['label']=='attack'])}")
    print(f"  Benign: {len(df[df['label']=='benign'])}")
    print()
    
    # Split by label
    attacks = df[df['label'] == 'attack'].copy()
    benign = df[df['label'] == 'benign'].copy()
    
    # Augment attacks if needed
    all_attacks = [attacks]
    if len(attacks) < target_attacks:
        needed = target_attacks - len(attacks)
        print(f"Generating {needed} attack variations...")
        
        augmented_attacks = []
        while len(augmented_attacks) < needed:
            # Sample from existing attacks
            sample = attacks.sample(n=min(len(attacks), needed - len(augmented_attacks)))
            
            for _, row in sample.iterrows():
                variations = create_variations(row['text'], n_variations=2)
                for i, var_text in enumerate(variations):
                    if len(augmented_attacks) >= needed:
                        break
                    augmented_attacks.append({
                        'id': f"{row['id']}_aug{len(augmented_attacks)+1}",
                        'family': row['family'],
                        'label': row['label'],
                        'text': var_text
                    })
        
        all_attacks.append(pd.DataFrame(augmented_attacks))
        print(f"  Created {len(augmented_attacks)} attack variations")
    
    # Augment benign if needed
    all_benign = [benign]
    if len(benign) < target_benign:
        needed = target_benign - len(benign)
        print(f"Generating {needed} benign variations...")
        
        augmented_benign = []
        while len(augmented_benign) < needed:
            # Sample from existing benign
            sample = benign.sample(n=min(len(benign), needed - len(augmented_benign)))
            
            for _, row in sample.iterrows():
                variations = create_variations(row['text'], n_variations=2)
                for i, var_text in enumerate(variations):
                    if len(augmented_benign) >= needed:
                        break
                    augmented_benign.append({
                        'id': f"{row['id']}_aug{len(augmented_benign)+1}",
                        'family': row['family'],
                        'label': row['label'],
                        'text': var_text
                    })
        
        all_benign.append(pd.DataFrame(augmented_benign))
        print(f"  Created {len(augmented_benign)} benign variations")
    
    # Combine all
    df_final = pd.concat(all_attacks + all_benign, ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save
    df_final.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Augmented dataset saved to {output_file}")
    print(f"✓ Total samples: {len(df_final)}")
    print(f"✓ Attacks: {len(df_final[df_final['label']=='attack'])}")
    print(f"✓ Benign: {len(df_final[df_final['label']=='benign'])}")
    print(f"{'='*60}")
    
    return df_final


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment dataset to target size")
    parser.add_argument(
        "--input",
        default="data/prompts_hf_balanced.csv",
        help="Input CSV file"
    )
    parser.add_argument(
        "--output",
        default="data/prompts_hf_augmented.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--attacks",
        type=int,
        default=1000,
        help="Target number of attack prompts (default: 1000)"
    )
    parser.add_argument(
        "--benign",
        type=int,
        default=1000,
        help="Target number of benign prompts (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    augment_dataset(
        input_file=args.input,
        output_file=args.output,
        target_attacks=args.attacks,
        target_benign=args.benign,
        seed=args.seed
    )
