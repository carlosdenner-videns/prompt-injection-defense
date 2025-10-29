"""
Phase 1 Validation Script

Quick validation to ensure Phase 1 is set up correctly before running experiments.
"""

import sys
from pathlib import Path
import pandas as pd


def validate_directory_structure():
    """Check that all required directories exist."""
    print("Checking directory structure...")
    
    required_dirs = [
        "configs",
        "data", 
        "src",
        "src/defenses",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"  ❌ Missing directory: {dir_path}")
            all_exist = False
        else:
            print(f"  ✅ Found: {dir_path}")
    
    return all_exist


def validate_config_files():
    """Check that configuration files exist and are valid."""
    print("\nChecking configuration files...")
    
    # Check experiment.json
    experiment_config = Path("configs/experiment.json")
    if not experiment_config.exists():
        print("  ❌ Missing: configs/experiment.json")
        return False
    print("  ✅ Found: configs/experiment.json")
    
    # Check rules.yml
    rules_config = Path("configs/rules.yml")
    if not rules_config.exists():
        print("  ❌ Missing: configs/rules.yml")
        return False
    print("  ✅ Found: configs/rules.yml")
    
    return True


def validate_dataset():
    """Validate dataset file exists and has correct format."""
    print("\nChecking dataset...")
    
    dataset_path = Path("data/prompts_hf_augmented.csv")
    
    if not dataset_path.exists():
        print("  ❌ Missing: data/prompts_hf_augmented.csv")
        return False
    
    print("  ✅ Found: data/prompts_hf_augmented.csv")
    
    # Load and validate
    try:
        df = pd.read_csv(dataset_path)
        
        # Check columns
        if 'text' not in df.columns or 'label' not in df.columns:
            print("  ❌ Dataset missing required columns (text, label)")
            return False
        
        print(f"  ✅ Dataset has required columns")
        
        # Convert string labels to numeric if needed
        if df['label'].dtype == 'object':
            label_map = {'attack': 1, 'benign': 0}
            df['label'] = df['label'].map(label_map)
        
        # Check size
        total = len(df)
        attacks = len(df[df['label'] == 1])
        benign = len(df[df['label'] == 0])
        
        print(f"  ℹ️  Total samples: {total}")
        print(f"  ℹ️  Attacks: {attacks} ({attacks/total*100:.1f}%)")
        print(f"  ℹ️  Benign: {benign} ({benign/total*100:.1f}%)")
        
        # Warn if not balanced
        if total != 2000:
            print(f"  ⚠️  Expected 2,000 samples, found {total}")
        
        if abs(attacks - 1000) > 50 or abs(benign - 1000) > 50:
            print(f"  ⚠️  Dataset not balanced (expected ~1000 attacks, ~1000 benign)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading dataset: {e}")
        return False


def validate_source_files():
    """Check that defense implementation files exist."""
    print("\nChecking source files...")
    
    required_files = [
        "src/defenses/__init__.py",
        "src/defenses/signature_only.py",
        "src/defenses/rules_only.py",
        "src/defenses/nemo_baseline.py",
        "src/defenses/openai_moderation.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"  ❌ Missing: {file_path}")
            all_exist = False
        else:
            print(f"  ✅ Found: {file_path}")
    
    return all_exist


def validate_dependencies():
    """Check that required Python packages are installed."""
    print("\nChecking Python dependencies...")
    
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('yaml', 'pyyaml'),
        ('scipy', 'scipy'),
        ('sklearn', 'scikit-learn'),
    ]
    
    all_installed = True
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} not installed")
            all_installed = False
    
    # Optional packages
    print("\nChecking optional dependencies...")
    optional_packages = [
        ('openai', 'openai (for OpenAI Moderation API)'),
        ('matplotlib', 'matplotlib (for plots)'),
        ('seaborn', 'seaborn (for plots)'),
    ]
    
    for module_name, description in optional_packages:
        try:
            __import__(module_name)
            print(f"  ✅ {description}")
        except ImportError:
            print(f"  ⚠️  {description} - optional, install if needed")
    
    return all_installed


def main():
    print("="*70)
    print("PHASE 1: VALIDATION CHECK")
    print("="*70)
    print()
    
    checks = [
        ("Directory Structure", validate_directory_structure),
        ("Configuration Files", validate_config_files),
        ("Dataset", validate_dataset),
        ("Source Files", validate_source_files),
        ("Dependencies", validate_dependencies),
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nYou can now run Phase 1 experiments:")
        print("  python run_phase1_experiments.py --skip-moderation")
        print("\nOr with OpenAI Moderation API:")
        print("  $env:OPENAI_API_KEY = 'your-key'")
        print("  python run_phase1_experiments.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running experiments.")
        print("\nCommon fixes:")
        print("  - Install missing dependencies: pip install -r requirements.txt")
        print("  - Check dataset file exists in data/ directory")
        print("  - Verify all source files are present")
        return 1


if __name__ == '__main__':
    sys.exit(main())
