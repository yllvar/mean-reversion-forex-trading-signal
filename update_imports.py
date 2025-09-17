"""Script to update imports after restructuring the project."""
import os
from pathlib import Path

def update_file_imports(file_path):
    """Update imports in a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update imports
    replacements = {
        'from data_fetcher': 'from metasync_dashboard.data.data_fetcher',
        'from signals': 'from metasync_dashboard.strategies.signals',
        'from executor': 'from metasync_dashboard.execution.executor',
        'from utils': 'from metasync_dashboard.utils.utils',
        'from config': 'from metasync_dashboard.config',
    }
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Update imports in all Python files."""
    src_dir = Path('src/metasync_dashboard')
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                print(f"Updating imports in {file_path}")
                update_file_imports(file_path)

if __name__ == "__main__":
    main()
