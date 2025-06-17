"""
X-ray Dataset Preparation Script
Organizes X-ray images into proper training/validation structure.
"""

import os
import shutil
import argparse
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare X-ray Dataset')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Path to source dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output organized dataset')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Training split ratio (default: 0.8)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--copy_files', action='store_true',
                        help='Copy files instead of moving them')

    return parser.parse_args()


def scan_source_directory(source_dir):
    """
    Scan source directory and organize files by class.

    Expected structure:
    source_dir/
    ├── Normal/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── Pneumonia/
    │   ├── image3.jpg
    │   └── image4.jpg
    └── COVID-19/
        ├── image5.jpg
        └── image6.jpg
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    # Find all class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]

    if not class_dirs:
        raise ValueError(f"No class directories found in {source_dir}")

    # Organize files by class
    files_by_class = defaultdict(list)
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    for class_dir in class_dirs:
        class_name = class_dir.name

        # Find all image files in this class directory
        image_files = []
        for ext in supported_extensions:
            image_files.extend(class_dir.glob(f'*{ext}'))
            image_files.extend(class_dir.glob(f'*{ext.upper()}'))

        if image_files:
            files_by_class[class_name] = [str(f) for f in image_files]
            print(f"Found {len(image_files)} images in class '{class_name}'")
        else:
            print(f"Warning: No images found in class directory '{class_name}'")

    return dict(files_by_class)


def create_splits(files_by_class, train_split, val_split, seed):
    """Create train/validation splits for each class."""

    if abs(train_split + val_split - 1.0) > 1e-6:
        raise ValueError("Train and validation splits must sum to 1.0")

    random.seed(seed)

    splits = {
        'train': defaultdict(list),
        'val': defaultdict(list)
    }

    for class_name, files in files_by_class.items():
        # Shuffle files
        files_shuffled = files.copy()
        random.shuffle(files_shuffled)

        # Calculate split indices
        n_files = len(files_shuffled)
        n_train = int(n_files * train_split)

        # Split files
        train_files = files_shuffled[:n_train]
        val_files = files_shuffled[n_train:]

        splits['train'][class_name] = train_files
        splits['val'][class_name] = val_files

        print(f"Class '{class_name}': {len(train_files)} train, {len(val_files)} val")

    return splits


def create_directory_structure(output_dir, splits):
    """Create the organized directory structure."""
    output_path = Path(output_dir)

    # Create main directories
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Create class subdirectories
    for split_name, classes in splits.items():
        split_dir = output_path / split_name

        for class_name in classes.keys():
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created directory structure in: {output_dir}")


def copy_or_move_files(splits, output_dir, copy_files=True):
    """Copy or move files to the organized structure."""
    output_path = Path(output_dir)

    operation = shutil.copy2 if copy_files else shutil.move
    operation_name = "Copying" if copy_files else "Moving"

    total_files = 0
    for split_name, classes in splits.items():
        for class_name, files in classes.items():
            total_files += len(files)

    print(f"{operation_name} {total_files} files...")

    processed = 0
    for split_name, classes in splits.items():
        for class_name, files in classes.items():
            target_dir = output_path / split_name / class_name

            for file_path in files:
                source_file = Path(file_path)
                target_file = target_dir / source_file.name

                # Handle duplicate names
                counter = 1
                original_target = target_file
                while target_file.exists():
                    stem = original_target.stem
                    suffix = original_target.suffix
                    target_file = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

                try:
                    operation(source_file, target_file)
                    processed += 1

                    if processed % 100 == 0:
                        print(f"Processed {processed}/{total_files} files...")

                except Exception as e:
                    print(f"Error processing {source_file}: {e}")

    print(f"Successfully processed {processed}/{total_files} files")


def save_dataset_info(output_dir, splits, args):
    """Save dataset information and metadata."""

    # Calculate statistics
    stats = {}
    total_files = 0

    for split_name, classes in splits.items():
        stats[split_name] = {}
        split_total = 0

        for class_name, files in classes.items():
            count = len(files)
            stats[split_name][class_name] = count
            split_total += count
            total_files += count

        stats[split_name]['total'] = split_total

    # Create dataset info
    dataset_info = {
        'dataset_name': 'X-ray Classification Dataset',
        'created_at': str(Path().cwd()),
        'source_directory': args.source_dir,
        'output_directory': args.output_dir,
        'train_split': args.train_split,
        'val_split': args.val_split,
        'random_seed': args.seed,
        'total_files': total_files,
        'statistics': stats,
        'classes': list(next(iter(splits.values())).keys()),
        'splits': list(splits.keys())
    }

    # Save to JSON
    info_file = Path(args.output_dir) / 'dataset_info.json'
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Dataset info saved to: {info_file}")

    # Print summary
    print("\n" + "="*50)
    print("DATASET PREPARATION SUMMARY")
    print("="*50)
    print(f"Total files: {total_files}")
    print(f"Classes: {', '.join(dataset_info['classes'])}")
    print("\nSplit distribution:")
    for split_name, split_stats in stats.items():
        print(f"  {split_name.upper()}:")
        for class_name, count in split_stats.items():
            if class_name != 'total':
                print(f"    {class_name}: {count}")
        print(f"    Total: {split_stats['total']}")


def main():
    """Main function."""
    args = parse_arguments()

    print("X-ray Dataset Preparation")
    print("=" * 30)
    print(f"Source: {args.source_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Train split: {args.train_split}")
    print(f"Val split: {args.val_split}")
    print(f"Random seed: {args.seed}")
    print(f"Operation: {'Copy' if args.copy_files else 'Move'}")
    print()

    try:
        # Step 1: Scan source directory
        print("Step 1: Scanning source directory...")
        files_by_class = scan_source_directory(args.source_dir)

        if not files_by_class:
            print("No image files found in source directory!")
            return

        # Step 2: Create splits
        print("\nStep 2: Creating train/validation splits...")
        splits = create_splits(files_by_class, args.train_split, args.val_split, args.seed)

        # Step 3: Create directory structure
        print("\nStep 3: Creating directory structure...")
        create_directory_structure(args.output_dir, splits)

        # Step 4: Copy/move files
        print("\nStep 4: Organizing files...")
        copy_or_move_files(splits, args.output_dir, args.copy_files)

        # Step 5: Save dataset info
        print("\nStep 5: Saving dataset information...")
        save_dataset_info(args.output_dir, splits, args)

        print("\n✅ Dataset preparation completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during dataset preparation: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
