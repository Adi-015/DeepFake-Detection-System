import os
import shutil
from pathlib import Path

source_base = Path(r"G:\deepfake-project\data\real_vs_fake\real-vs-fake")
target_base = Path(r"G:\deepfake-project\data\test_subset")

print(f"Source base exists: {source_base.exists()}")
if not source_base.exists():
    # Let's check G:\deepfake-project\data
    parent = Path(r"G:\deepfake-project\data")
    print(f"Listing {parent}:")
    if parent.exists():
        for p in parent.iterdir():
            print(" ", p)

splits = ['train', 'valid', 'test']
classes = ['real', 'fake']
N_SAMPLES = 50

for split in splits:
    for cls in classes:
        src_dir = source_base / split / cls
        dst_dir = target_base / split / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        existing_files = list(dst_dir.glob('*'))
        if len(existing_files) >= N_SAMPLES:
            print(f"{dst_dir} already has {len(existing_files)} files. Skipping copy.")
            continue

        files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp'):
            files.extend(list(src_dir.glob(ext)))

        print(f"Found {len(files)} source files in {src_dir}")
        selected = files[:N_SAMPLES]
        for f in selected:
            shutil.copy2(f, dst_dir / f.name)
        print(f"Copied {len(selected)} files to {dst_dir}")

print("Test subset creation complete!")
for split in splits:
    for cls in classes:
        dst_dir = target_base / split / cls
        count = len(list(dst_dir.glob('*')))
        print(f"  {split}/{cls}: {count} files")
