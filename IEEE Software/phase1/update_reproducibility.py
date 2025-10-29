"""
Populate REPRODUCIBILITY.md with actual system and package information.
"""

import platform
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import psutil


def get_system_info():
    """Collect system information."""
    info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(),
        'ram': round(psutil.virtual_memory().total / (1024**3), 1),  # GB
        'os_info': f"{platform.system()} {platform.release()} ({platform.version()})",
        'python_version': sys.version.split()[0],
        'platform': platform.platform()
    }
    return info


def get_package_versions():
    """Get installed package versions."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            capture_output=True,
            text=True,
            check=True
        )
        all_packages = result.stdout.strip()
        
        # Extract core packages
        core = [
            'pandas', 'numpy', 'scipy', 'scikit-learn',
            'matplotlib', 'seaborn', 'openai', 'python-dotenv'
        ]
        
        core_packages = []
        for line in all_packages.split('\n'):
            package_name = line.split('==')[0].lower()
            if package_name in core or package_name.replace('-', '_') in core:
                core_packages.append(line)
        
        return '\n'.join(core_packages), all_packages
        
    except Exception as e:
        return f"Error: {e}", f"Error: {e}"


def update_reproducibility_doc():
    """Update REPRODUCIBILITY.md with system information."""
    
    print("Collecting system information...")
    sys_info = get_system_info()
    
    print("Collecting package versions...")
    core_packages, all_packages = get_package_versions()
    
    # Read template
    repro_file = Path('REPRODUCIBILITY.md')
    with open(repro_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace placeholders
    replacements = {
        '{timestamp}': sys_info['timestamp'],
        '{processor}': sys_info['processor'],
        '{cpu_count}': str(sys_info['cpu_count']),
        '{ram}': str(sys_info['ram']),
        '{os_info}': sys_info['os_info'],
        '{python_version}': sys_info['python_version'],
        '{platform}': sys_info['platform'],
        '{core_packages}': core_packages,
        '{all_packages}': all_packages
    }
    
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    
    # Write updated content
    with open(repro_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Updated REPRODUCIBILITY.md")
    print("\nSystem Information:")
    print(f"  OS: {sys_info['os_info']}")
    print(f"  Python: {sys_info['python_version']}")
    print(f"  CPU: {sys_info['processor']} ({sys_info['cpu_count']} cores)")
    print(f"  RAM: {sys_info['ram']} GB")
    print(f"\nCore Packages:")
    for pkg in core_packages.split('\n'):
        print(f"  {pkg}")


if __name__ == '__main__':
    update_reproducibility_doc()
