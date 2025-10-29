"""
Data utilities for Phase 1: train/dev/test splits and attack family labeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split


# Attack family detection patterns
ATTACK_FAMILY_PATTERNS = {
    'exfiltration': [
        r'(?i)(show|reveal|display|tell|output|print).{0,40}(prompt|instruction|system|secret|token|password)',
        r'(?i)what (are|is|was).{0,30}(your|the).{0,20}(prompt|instruction|system)',
    ],
    'instruction_override': [
        r'(?i)(ignore|disregard|forget|skip|override|bypass).{0,30}(previous|all|above|prior|your).{0,30}(instruction|rule|directive|guideline|context)',
        r'(?i)(new|different|instead).{0,20}(instruction|task|directive|job)',
    ],
    'jailbreak': [
        r'(?i)\b(dan|do anything now|developer mode|jailbreak)\b',
        r'(?i)(chatgpt|you).{0,20}(with|in).{0,20}dan',
    ],
    'role_play': [
        r'(?i)(pretend|act as|you are now|roleplay|role.play).{0,30}(that|as if|like)',
        r'(?i)from now on.{0,20}(you|act)',
    ],
    'context_injection': [
        r'(?i)^---+\s*(new|ignore|forget)',
        r'(?i)^===+\s*(new|ignore|forget)',
    ],
    'encoding_bypass': [
        r'(?i)(translate|speak|answer|encode).{0,30}(base64|hex|cipher|rot13|code)',
    ],
}


def label_attack_family(text: str) -> str:
    """
    Label an attack with its primary family.
    
    Args:
        text: Prompt text to classify
        
    Returns:
        Attack family label or 'other_attack' if no match
    """
    import re
    
    text_lower = text.lower()
    
    # Check each family
    for family, patterns in ATTACK_FAMILY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return family
    
    return 'other_attack'


def add_attack_families(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add attack family labels to dataset.
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        
    Returns:
        DataFrame with added 'family' column
    """
    df = df.copy()
    
    # Label attacks by family
    df['family'] = df.apply(
        lambda row: label_attack_family(row['text']) if row['label'] == 1 else 'benign',
        axis=1
    )
    
    return df


def create_train_dev_test_split(
    df: pd.DataFrame,
    train_size: float = 0.5,
    dev_size: float = 0.2,
    test_size: float = 0.2,
    ood_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/dev/test/OOD splits.
    
    Args:
        df: Dataset with 'label' and 'family' columns
        train_size: Fraction for training (pattern discovery)
        dev_size: Fraction for development (threshold tuning)
        test_size: Fraction for final evaluation
        ood_size: Fraction for out-of-distribution testing
        random_state: Random seed for reproducibility
        
    Returns:
        (train_df, dev_df, test_df, ood_df) tuple
    """
    assert abs(train_size + dev_size + test_size + ood_size - 1.0) < 1e-6
    
    # Ensure family column exists
    if 'family' not in df.columns:
        df = add_attack_families(df)
    
    # First split: separate OOD from the rest
    # Stratify by both label and family
    df['strata'] = df['label'].astype(str) + '_' + df['family']
    
    remaining, ood = train_test_split(
        df,
        test_size=ood_size,
        stratify=df['strata'],
        random_state=random_state
    )
    
    # Split remaining into train/dev/test
    # Adjust proportions (since we already removed OOD)
    remaining_total = train_size + dev_size + test_size
    train_frac = train_size / remaining_total
    dev_frac = dev_size / remaining_total
    test_frac = test_size / remaining_total
    
    # Second split: train vs (dev + test)
    train, dev_test = train_test_split(
        remaining,
        train_size=train_frac,
        stratify=remaining['strata'],
        random_state=random_state
    )
    
    # Third split: dev vs test
    dev_test_total = dev_frac + test_frac
    dev_proportion = dev_frac / dev_test_total
    
    dev, test = train_test_split(
        dev_test,
        train_size=dev_proportion,
        stratify=dev_test['strata'],
        random_state=random_state
    )
    
    # Clean up temporary strata column
    for split_df in [train, dev, test, ood]:
        split_df.drop(columns=['strata'], inplace=True)
    
    print(f"Created splits:")
    print(f"  Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"  Dev:   {len(dev)} ({len(dev)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")
    print(f"  OOD:   {len(ood)} ({len(ood)/len(df)*100:.1f}%)")
    
    return train, dev, test, ood


def save_splits(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    ood: pd.DataFrame,
    output_dir: str = "data/splits"
) -> None:
    """Save train/dev/test/OOD splits to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrames
    train.to_csv(output_path / "train.csv", index=False)
    dev.to_csv(output_path / "dev.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)
    ood.to_csv(output_path / "ood.csv", index=False)
    
    # Save split metadata
    metadata = {
        'random_state': 42,
        'train_size': len(train),
        'dev_size': len(dev),
        'test_size': len(test),
        'ood_size': len(ood),
        'total_size': len(train) + len(dev) + len(test) + len(ood),
        'attack_families': list(ATTACK_FAMILY_PATTERNS.keys()),
        'split_date': pd.Timestamp.now().isoformat()
    }
    
    with open(output_path / "split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved splits to: {output_dir}")
    print(f"  - train.csv ({len(train)} samples)")
    print(f"  - dev.csv ({len(dev)} samples)")
    print(f"  - test.csv ({len(test)} samples)")
    print(f"  - ood.csv ({len(ood)} samples)")
    print(f"  - split_metadata.json")


def load_splits(input_dir: str = "data/splits") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/dev/test/OOD splits from CSV files."""
    input_path = Path(input_dir)
    
    train = pd.read_csv(input_path / "train.csv")
    dev = pd.read_csv(input_path / "dev.csv")
    test = pd.read_csv(input_path / "test.csv")
    ood = pd.read_csv(input_path / "ood.csv")
    
    return train, dev, test, ood


def analyze_split_distribution(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    ood: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze distribution of labels and families across splits.
    
    Returns:
        DataFrame with distribution statistics
    """
    stats = []
    
    for split_name, split_df in [('Train', train), ('Dev', dev), ('Test', test), ('OOD', ood)]:
        # Overall label distribution
        attacks = len(split_df[split_df['label'] == 1])
        benign = len(split_df[split_df['label'] == 0])
        
        stat_dict = {
            'Split': split_name,
            'Total': len(split_df),
            'Attacks': attacks,
            'Benign': benign,
            'Attack %': f"{attacks/len(split_df)*100:.1f}%"
        }
        
        # Family distribution (for attacks only)
        attack_df = split_df[split_df['label'] == 1]
        for family in ATTACK_FAMILY_PATTERNS.keys():
            family_count = len(attack_df[attack_df['family'] == family])
            stat_dict[f'{family}'] = family_count
        
        # Other attacks
        other_count = len(attack_df[attack_df['family'] == 'other_attack'])
        stat_dict['other_attack'] = other_count
        
        stats.append(stat_dict)
    
    return pd.DataFrame(stats)


if __name__ == '__main__':
    # Example usage
    print("Loading dataset...")
    df = pd.read_csv('data/prompts_hf_augmented.csv')
    
    # Convert labels if needed
    if df['label'].dtype == 'object':
        label_map = {'attack': 1, 'benign': 0}
        df['label'] = df['label'].map(label_map)
    
    # Add attack families
    print("\nLabeling attack families...")
    df = add_attack_families(df)
    
    # Show family distribution
    print("\nAttack family distribution:")
    attack_df = df[df['label'] == 1]
    family_counts = attack_df['family'].value_counts()
    print(family_counts)
    
    # Create splits
    print("\n" + "="*60)
    print("Creating train/dev/test/OOD splits...")
    print("="*60)
    train, dev, test, ood = create_train_dev_test_split(df)
    
    # Save splits
    save_splits(train, dev, test, ood)
    
    # Analyze distribution
    print("\n" + "="*60)
    print("Split Distribution Analysis")
    print("="*60)
    dist_df = analyze_split_distribution(train, dev, test, ood)
    print(dist_df.to_string(index=False))
    
    print("\n✅ Data splits created successfully!")
    print("\nNext steps:")
    print("  1. Use train split for pattern discovery (Phase 3)")
    print("  2. Use dev split for threshold tuning (Phase 4)")
    print("  3. Use test split for final evaluation")
    print("  4. Use OOD split to test generalization")
