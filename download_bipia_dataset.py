"""
Additional Prompt Injection Dataset Downloader and Merger

This script downloads additional prompt injection datasets from public sources
and merges them with our existing HuggingFace combined dataset.

Since BIPIA requires their Python package installation and programmatic access,
this script instead uses publicly available datasets that can be directly downloaded.

Sources:
- deepset/prompt-injections (HuggingFace)
- Additional curated attack patterns

Author: Carlo (with GitHub Copilot)
Date: October 28, 2025
"""

import os
import json
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


class AdditionalDatasetDownloader:
    """
    Download and process additional prompt injection datasets.
    
    Uses publicly available datasets from HuggingFace and other sources.
    """
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize dataset downloader.
        
        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def download_from_huggingface(self) -> pd.DataFrame:
        """
        Download additional datasets from HuggingFace.
        
        Returns:
            DataFrame with downloaded data
        """
        print("\n" + "=" * 70)
        print("DOWNLOADING FROM HUGGINGFACE")
        print("=" * 70)
        
        try:
            from datasets import load_dataset
        except ImportError:
            print("  ❌ Error: datasets library not installed")
            print("  Run: pip install datasets")
            return pd.DataFrame()
        
        all_data = []
        
        # Try to download deepset/prompt-injections
        try:
            print("\n📥 Downloading deepset/prompt-injections...")
            dataset = load_dataset("deepset/prompt-injections", split="train")
            print(f"  ✅ Loaded {len(dataset)} examples")
            
            for example in dataset:
                all_data.append({
                    "text": example.get("text", ""),
                    "label": "attack" if example.get("label", 0) == 1 else "benign",
                    "source": "deepset",
                    "category": "prompt_injection"
                })
        except Exception as e:
            print(f"  ⚠️  Could not load deepset/prompt-injections: {e}")
        
        # Try other datasets if available
        try:
            print("\n📥 Downloading fka/awesome-chatgpt-prompts...")
            dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train")
            print(f"  ✅ Loaded {len(dataset)} examples (benign prompts)")
            
            # These are benign creative prompts
            for example in dataset[:500]:  # Limit to 500
                all_data.append({
                    "text": example.get("prompt", ""),
                    "label": "benign",
                    "source": "awesome-chatgpt",
                    "category": "creative_prompt"
                })
        except Exception as e:
            print(f"  ⚠️  Could not load awesome-chatgpt-prompts: {e}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            print(f"\n  📊 Total examples downloaded: {len(df)}")
            return df
        else:
            print(f"\n  ⚠️  No data downloaded")
            return pd.DataFrame()
    
    def merge_with_existing(
        self,
        bipia_df: pd.DataFrame,
        existing_path: str = "data/prompts_hf_augmented.csv"
    ) -> pd.DataFrame:
        """
        Merge BIPIA dataset with existing HuggingFace dataset.
        
        Args:
            bipia_df: BIPIA DataFrame
            existing_path: Path to existing dataset CSV
            
        Returns:
            Combined DataFrame
        """
        print("\n" + "=" * 70)
        print("MERGING WITH EXISTING DATASET")
        print("=" * 70)
        
        existing_path = Path(existing_path)
        
        if not existing_path.exists():
            print(f"  ⚠️  Existing dataset not found: {existing_path}")
            print(f"  📝 Using BIPIA data only")
            combined_df = bipia_df.copy()
        else:
            print(f"  📂 Loading existing dataset: {existing_path}")
            existing_df = pd.read_csv(existing_path)
            print(f"  📊 Existing dataset: {len(existing_df)} examples")
            
            # Standardize columns
            if 'source' not in existing_df.columns:
                existing_df['source'] = 'HuggingFace'
            if 'category' not in existing_df.columns:
                existing_df['category'] = existing_df['label']
            
            # Combine datasets
            print(f"\n  🔗 Merging datasets...")
            combined_df = pd.concat([existing_df, bipia_df], ignore_index=True)
            print(f"  ✅ Combined dataset: {len(combined_df)} examples")
        
        # Remove duplicates based on text
        print(f"\n  🔍 Checking for duplicates...")
        before = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
        after = len(combined_df)
        duplicates_removed = before - after
        
        if duplicates_removed > 0:
            print(f"  🗑️  Removed {duplicates_removed} duplicate examples")
        else:
            print(f"  ✅ No duplicates found")
        
        # Final statistics
        print(f"\n📊 Final Dataset Statistics:")
        print(f"  Total examples: {len(combined_df)}")
        print(f"\n  By label:")
        print(combined_df['label'].value_counts().to_string())
        print(f"\n  By source:")
        print(combined_df['source'].value_counts().to_string())
        
        return combined_df
    
    def save_combined_dataset(
        self,
        df: pd.DataFrame,
        output_path: str = "data/prompts_bipia_combined.csv"
    ):
        """
        Save combined dataset to CSV.
        
        Args:
            df: Combined DataFrame
            output_path: Output file path
        """
        output_path = Path(output_path)
        
        print("\n" + "=" * 70)
        print("SAVING COMBINED DATASET")
        print("=" * 70)
        
        # Save full dataset
        df.to_csv(output_path, index=False)
        print(f"\n  ✅ Saved to: {output_path}")
        print(f"  📊 Total examples: {len(df)}")
        
        # Create balanced sample (equal attacks and benign)
        print(f"\n  📊 Creating balanced sample...")
        attacks = df[df['label'] == 'attack']
        benign = df[df['label'] == 'benign']
        
        # Sample to match the smaller group
        n_samples = min(len(attacks), len(benign))
        
        if n_samples > 0:
            balanced_df = pd.concat([
                attacks.sample(n=n_samples, random_state=42),
                benign.sample(n=n_samples, random_state=42)
            ], ignore_index=True)
            
            # Shuffle
            balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            balanced_path = output_path.parent / f"{output_path.stem}_balanced.csv"
            balanced_df.to_csv(balanced_path, index=False)
            print(f"  ✅ Balanced dataset saved to: {balanced_path}")
            print(f"  📊 Total examples: {len(balanced_df)} ({n_samples} attacks + {n_samples} benign)")
        else:
            print(f"  ⚠️  Cannot create balanced sample (insufficient data)")
    
    def generate_statistics_report(
        self,
        df: pd.DataFrame,
        output_path: str = "data/bipia_statistics.txt"
    ):
        """
        Generate detailed statistics report.
        
        Args:
            df: Combined DataFrame
            output_path: Output file path for report
        """
        output_path = Path(output_path)
        
        print("\n" + "=" * 70)
        print("GENERATING STATISTICS REPORT")
        print("=" * 70)
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("BIPIA DATASET STATISTICS REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("OVERALL STATISTICS")
        report_lines.append("-" * 70)
        report_lines.append(f"Total examples: {len(df)}")
        report_lines.append("")
        
        # Label distribution
        report_lines.append("LABEL DISTRIBUTION")
        report_lines.append("-" * 70)
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            report_lines.append(f"  {label}: {count} ({pct:.1f}%)")
        report_lines.append("")
        
        # Source distribution
        report_lines.append("SOURCE DISTRIBUTION")
        report_lines.append("-" * 70)
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            pct = count / len(df) * 100
            report_lines.append(f"  {source}: {count} ({pct:.1f}%)")
        report_lines.append("")
        
        # Category distribution (if available)
        if 'category' in df.columns:
            report_lines.append("CATEGORY DISTRIBUTION")
            report_lines.append("-" * 70)
            category_counts = df['category'].value_counts()
            for category, count in category_counts.items():
                pct = count / len(df) * 100
                report_lines.append(f"  {category}: {count} ({pct:.1f}%)")
            report_lines.append("")
        
        # Attack type distribution (if available)
        if 'attack_type' in df.columns:
            attack_df = df[df['label'] == 'attack']
            if len(attack_df) > 0:
                report_lines.append("ATTACK TYPE DISTRIBUTION")
                report_lines.append("-" * 70)
                attack_type_counts = attack_df['attack_type'].value_counts()
                for attack_type, count in attack_type_counts.items():
                    pct = count / len(attack_df) * 100
                    report_lines.append(f"  {attack_type}: {count} ({pct:.1f}%)")
                report_lines.append("")
        
        # Text length statistics
        report_lines.append("TEXT LENGTH STATISTICS")
        report_lines.append("-" * 70)
        df['text_length'] = df['text'].str.len()
        report_lines.append(f"  Min: {df['text_length'].min()} characters")
        report_lines.append(f"  Max: {df['text_length'].max()} characters")
        report_lines.append(f"  Mean: {df['text_length'].mean():.1f} characters")
        report_lines.append(f"  Median: {df['text_length'].median():.1f} characters")
        report_lines.append("")
        
        # Sample examples
        report_lines.append("SAMPLE EXAMPLES")
        report_lines.append("-" * 70)
        
        # Sample attack
        attack_sample = df[df['label'] == 'attack'].sample(n=1, random_state=42).iloc[0]
        report_lines.append("Attack Example:")
        report_lines.append(f"  Text: {attack_sample['text'][:100]}...")
        if 'attack_type' in df.columns:
            report_lines.append(f"  Type: {attack_sample['attack_type']}")
        report_lines.append("")
        
        # Sample benign
        benign_sample = df[df['label'] == 'benign'].sample(n=1, random_state=42).iloc[0]
        report_lines.append("Benign Example:")
        report_lines.append(f"  Text: {benign_sample['text'][:100]}...")
        report_lines.append("")
        
        report_lines.append("=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)
        
        # Write to file
        report_text = "\n".join(report_lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n  ✅ Statistics report saved to: {output_path}")
        
        # Also print to console
        print("\n" + report_text)


def main():
    """
    Main execution function.
    
    Downloads additional prompt injection datasets, merges with existing 
    HuggingFace data, and generates statistics reports.
    """
    print("\n" + "=" * 70)
    print("ADDITIONAL DATASET DOWNLOADER AND MERGER")
    print("=" * 70)
    
    # Initialize downloader
    downloader = AdditionalDatasetDownloader(output_dir="data")
    
    # Download from HuggingFace
    additional_df = downloader.download_from_huggingface()
    
    if additional_df.empty:
        print("\n⚠️  No additional data downloaded.")
        print("Make sure you have the datasets library installed:")
        print("  pip install datasets")
        return
    
    # Merge with existing dataset
    combined_df = downloader.merge_with_existing(
        additional_df,
        existing_path="data/prompts_hf_augmented.csv"
    )
    
    # Save combined dataset
    downloader.save_combined_dataset(
        combined_df,
        output_path="data/prompts_extended.csv"
    )
    
    # Generate statistics report
    downloader.generate_statistics_report(
        combined_df,
        output_path="data/extended_statistics.txt"
    )
    
    print("\n" + "=" * 70)
    print("✅ DATASET DOWNLOAD AND MERGE COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    print("  - data/prompts_extended.csv (full dataset)")
    print("  - data/prompts_extended_balanced.csv (balanced sample)")
    print("  - data/extended_statistics.txt (statistics report)")
    print("\nNext steps:")
    print("  1. Review the statistics report")
    print("  2. Run experiments on the combined dataset:")
    print("     python src/run_experiment.py --data data/prompts_extended_balanced.csv")
    print("  3. Test with OpenAI:")
    print("     python test_defenses_with_openai.py --data data/prompts_extended_balanced.csv")


if __name__ == "__main__":
    main()
