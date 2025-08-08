import os
from pathlib import Path

# Define the project structure
folders = [
    "src",
    "docs",
    "lambda",
    "screenshots"
]

files = [
    ".env",
    "README.md",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "src/__init__.py",
    "src/pipeline.py",
    "src/model.py",
    "src/strategies.py",
    "src/train_utils.py",
    "src/main.py",
    "docs/setup_guide.md",
    "lambda/handler.py",
    "lambda/requirements.txt",
]

# Create folders
for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"✔️ Created folder: {folder}")

# Touch files
for file_path in files:
    path = Path(file_path)
    if not path.exists():
        path.touch()
        print(f"🆕 Touched file: {file_path}")
    else:
        print(f"✅ Already exists: {file_path}")

print("\n📁 Project structure is now set up.")
