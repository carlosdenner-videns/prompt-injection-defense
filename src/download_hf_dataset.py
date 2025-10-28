#!/usr/bin/env python3
"""
Download and prepare the qualifire/prompt-injections-benchmark dataset from HuggingFace.
Creates a balanced, clean dataset for prompt injection defense evaluation.
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("ERROR: datasets library not installed")
    print("Install with: pip install datasets")


def clean_text(text):
    """
    Clean text by removing control characters and normalizing whitespace.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove control characters (except newline, tab, carriage return)
    # Control characters are in Unicode category 'Cc'
    text = ''.join(char for char in text if unicodedata.category(char) != 'Cc' or char in '\n\r\t')
    
    # Normalize unicode to NFC form (canonical composition)
    text = unicodedata.normalize('NFC', text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def load_single_dataset(dataset_name):
    """Load a single HuggingFace dataset and return as DataFrame."""
    print(f"  Loading: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"  ✗ Error loading {dataset_name}: {e}")
        return None
    
    # Get the train split (or first available split)
    if 'train' in dataset:
        df = pd.DataFrame(dataset['train'])
    elif 'test' in dataset:
        df = pd.DataFrame(dataset['test'])
    else:
        # Use first available split
        split_name = list(dataset.keys())[0]
        df = pd.DataFrame(dataset[split_name])
    
    print(f"  ✓ Loaded {len(df)} examples from {dataset_name}")
    return df


def download_and_prepare_dataset(
    dataset_names=None,
    output_dir="data",
    n_samples=2000,
    seed=42
):
    """
    Download and combine multiple HuggingFace datasets, clean, and create balanced subset.
    
    Args:
        dataset_names: List of HuggingFace dataset identifiers (or single string)
        output_dir: Directory to save output files
        n_samples: Total number of samples to extract (balanced between attack/benign)
        seed: Random seed for reproducibility
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library required. Install with: pip install datasets")
    
    # Default datasets if none provided
    if dataset_names is None:
        dataset_names = ["deepset/prompt-injections", "JasperLS/prompt-injections"]
    elif isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    print(f"Downloading and combining {len(dataset_names)} dataset(s)")
    print("=" * 60)
    
    # Load all datasets
    all_dfs = []
    for dataset_name in dataset_names:
        df = load_single_dataset(dataset_name)
        if df is not None:
            df['source'] = dataset_name  # Track source
            all_dfs.append(df)
    
    if len(all_dfs) == 0:
        raise ValueError("No datasets could be loaded successfully")
    
    print(f"\nSuccessfully loaded {len(all_dfs)} dataset(s)")
    print()
    
    print(f"\nSuccessfully loaded {len(all_dfs)} dataset(s)")
    print()
    
    # Process each dataset to standardize columns
    processed_dfs = []
    for df in all_dfs:
        source = df['source'].iloc[0] if 'source' in df.columns else 'unknown'
        print(f"Processing {source}...")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Identify text and label columns
        text_col = None
        for col in ['text', 'prompt', 'input', 'question', 'sentence']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            # Use first non-source column as text
            text_col = [c for c in df.columns if c != 'source'][0]
            print(f"  Warning: Using column '{text_col}' as text column")
        
        # Identify label column
        label_col = None
        for col in ['label', 'labels', 'type', 'category', 'is_injection']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            print(f"  Warning: No clear label column found. Using first non-text column.")
            label_col = [c for c in df.columns if c not in [text_col, 'source']][0]
        
        print(f"  Text column: {text_col}, Label column: {label_col}")
        
        # Clean the text
        df['text_clean'] = df[text_col].apply(clean_text)
        
        # Remove empty texts
        df = df[df['text_clean'].str.len() > 0]
        
        # Map labels to attack/benign
        unique_labels = df[label_col].unique()
        
        if len(unique_labels) == 2:
            # Binary classification
            if set(unique_labels) == {0, 1}:
                df['label_mapped'] = df[label_col].apply(lambda x: 'attack' if x == 1 else 'benign')
            elif set(map(str, unique_labels)) == {'injection', 'legitimate'}:
                df['label_mapped'] = df[label_col].apply(lambda x: 'attack' if x == 'injection' else 'benign')
            elif set(map(str, unique_labels)) == {'attack', 'benign'}:
                df['label_mapped'] = df[label_col].astype(str)
            else:
                # Heuristic mapping
                attack_keywords = ['inject', 'attack', 'jail', 'hack', 'malicious']
                def map_label(lbl):
                    lbl_str = str(lbl).lower()
                    return 'attack' if any(kw in lbl_str for kw in attack_keywords) else 'benign'
                df['label_mapped'] = df[label_col].apply(map_label)
        else:
            # Multi-class: map to attack/benign
            def map_multiclass(lbl):
                lbl_str = str(lbl).lower()
                return 'attack' if any(kw in lbl_str for kw in ['inject', 'attack', 'jail', 'hack', 'malicious', 'prompt_injection']) else 'benign'
            df['label_mapped'] = df[label_col].apply(map_multiclass)
        
        print(f"  Label distribution: {df['label_mapped'].value_counts().to_dict()}")
        processed_dfs.append(df)
    
    # Combine all datasets
    print("\nCombining all datasets...")
    df_combined = pd.concat(processed_dfs, ignore_index=True)
    print(f"Total combined examples: {len(df_combined)}")
    print(f"Combined label distribution:")
    print(df_combined['label_mapped'].value_counts())
    
    print(f"Combined label distribution:")
    print(df_combined['label_mapped'].value_counts())
    print()
    
    # Split into attack and benign
    attacks = df_combined[df_combined['label_mapped'] == 'attack'].copy()
    benign = df_combined[df_combined['label_mapped'] == 'benign'].copy()
    
    print(f"Total attacks: {len(attacks)}")
    print(f"Total benign: {len(benign)}")
    print()
    
    # Sample balanced subset
    np.random.seed(seed)
    n_per_class = n_samples // 2
    
    if len(attacks) < n_per_class:
        print(f"Warning: Only {len(attacks)} attacks available (requested {n_per_class}), sampling all")
        attacks_sample = attacks
    else:
        attacks_sample = attacks.sample(n=n_per_class, random_state=seed)
        print(f"Sampled {n_per_class} attacks from {len(attacks)} available")
    
    if len(benign) < n_per_class:
        print(f"Warning: Only {len(benign)} benign available (requested {n_per_class}), sampling all")
        benign_sample = benign
    else:
        benign_sample = benign.sample(n=n_per_class, random_state=seed)
        print(f"Sampled {n_per_class} benign from {len(benign)} available")
    
    # Combine
    df_balanced = pd.concat([attacks_sample, benign_sample], ignore_index=True)
    
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Create final dataset
    df_final = pd.DataFrame({
        'id': [f'HF{i+1}' for i in range(len(df_balanced))],
        'family': df_balanced['label_mapped'].apply(lambda x: x if x == 'benign' else 'attack'),
        'label': df_balanced['label_mapped'],
        'text': df_balanced['text_clean']
    })
    
    # Save to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "prompts_hf_balanced.csv"
    df_final.to_csv(output_file, index=False)
    
    print("=" * 60)
    print(f"✓ Dataset prepared successfully!")
    print(f"✓ Total samples: {len(df_final)}")
    print(f"✓ Attacks: {len(df_final[df_final['label']=='attack'])}")
    print(f"✓ Benign: {len(df_final[df_final['label']=='benign'])}")
    print(f"✓ Saved to: {output_file}")
    print("=" * 60)
    
    # Print sample rows
    print("\nSample attack prompts:")
    print("-" * 60)
    for text in df_final[df_final['label']=='attack']['text'].head(3):
        print(f"  • {text[:100]}...")
    
    print("\nSample benign prompts:")
    print("-" * 60)
    for text in df_final[df_final['label']=='benign']['text'].head(3):
        print(f"  • {text[:100]}...")
    print()
    
    # Statistics
    print("\nText length statistics:")
    print(df_final.groupby('label')['text'].apply(lambda x: x.str.len().describe()))
    
    return df_final


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and prepare HuggingFace prompt injection dataset"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["deepset/prompt-injections", "JasperLS/prompt-injections"],
        help="HuggingFace dataset names (default: deepset/prompt-injections JasperLS/prompt-injections)"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory (default: data)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Total number of samples (balanced) (default: 2000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    try:
        df = download_and_prepare_dataset(
            dataset_names=args.datasets,
            output_dir=args.output_dir,
            n_samples=args.samples,
            seed=args.seed
        )
        
        print("\n✓ Success! You can now run experiments with:")
        print(f"  python src/run_experiment.py --data {args.output_dir}/prompts_hf_balanced.csv --pipeline signature,rules,classifier --out results/hf_baseline")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
