"""
X-ray Dataset Class for PyTorch
Custom dataset class for loading and preprocessing X-ray images.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path
from typing import Optional, Callable, List, Tuple


class XRayDataset(Dataset):
    """
    Custom PyTorch Dataset for X-ray images.

    Expected directory structure:
    data_dir/
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

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    ):
        """
        Initialize the X-ray dataset.

        Args:
            data_dir: Path to the dataset directory
            transform: Optional transform to be applied to images
            target_transform: Optional transform to be applied to targets
            extensions: Tuple of supported file extensions
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.extensions = extensions

        # Validate directory
        if not self.data_dir.exists():
            raise ValueError(f"Dataset directory does not exist: {data_dir}")

        # Find all class directories
        self.class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        if not self.class_dirs:
            raise ValueError(f"No class directories found in {data_dir}")

        # Create class to index mapping
        self.classes = sorted([d.name for d in self.class_dirs])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Load all image paths and labels
        self.samples = self._load_samples()

        if not self.samples:
            raise ValueError(f"No valid images found in {data_dir}")

        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
        print(f"Classes: {self.classes}")

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their corresponding labels."""
        samples = []

        for class_dir in self.class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]

            # Find all image files in this class directory
            image_files = []
            for ext in self.extensions:
                image_files.extend(class_dir.glob(f'*{ext}'))
                image_files.extend(class_dir.glob(f'*{ext.upper()}'))

            # Add to samples list
            for image_path in image_files:
                samples.append((str(image_path), class_idx))

        return samples

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (image, label)
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")

        image_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(image_path)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def get_class_distribution(self) -> dict:
        """Get the distribution of samples across classes."""
        distribution = {cls: 0 for cls in self.classes}

        for _, label in self.samples:
            class_name = self.classes[label]
            distribution[class_name] += 1

        return distribution

    def get_sample_info(self, idx: int) -> dict:
        """Get detailed information about a specific sample."""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")

        image_path, label = self.samples[idx]
        class_name = self.classes[label]

        # Get image info
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
        except Exception as e:
            width = height = mode = format_name = None

        return {
            'index': idx,
            'image_path': image_path,
            'class_name': class_name,
            'class_index': label,
            'image_width': width,
            'image_height': height,
            'image_mode': mode,
            'image_format': format_name
        }

    def save_dataset_info(self, output_path: str):
        """Save dataset information to a JSON file."""
        info = {
            'dataset_path': str(self.data_dir),
            'num_samples': len(self.samples),
            'num_classes': len(self.classes),
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'class_distribution': self.get_class_distribution(),
            'sample_paths': [path for path, _ in self.samples]
        }

        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"Dataset info saved to {output_path}")


def create_dataset_from_info(info_path: str, transform: Optional[Callable] = None) -> XRayDataset:
    """
    Create a dataset from a saved dataset info file.

    Args:
        info_path: Path to the dataset info JSON file
        transform: Optional transform to apply to images

    Returns:
        XRayDataset instance
    """
    with open(info_path, 'r') as f:
        info = json.load(f)

    return XRayDataset(
        data_dir=info['dataset_path'],
        transform=transform
    )


# Utility functions for dataset analysis
def analyze_dataset(dataset: XRayDataset) -> dict:
    """Analyze dataset and return statistics."""

    distribution = dataset.get_class_distribution()

    # Calculate statistics
    total_samples = len(dataset)
    num_classes = len(dataset.classes)

    # Class balance
    class_percentages = {
        cls: (count / total_samples) * 100
        for cls, count in distribution.items()
    }

    # Image size analysis (sample first 100 images)
    sample_indices = list(range(min(100, total_samples)))
    image_sizes = []

    for idx in sample_indices:
        info = dataset.get_sample_info(idx)
        if info['image_width'] and info['image_height']:
            image_sizes.append((info['image_width'], info['image_height']))

    # Calculate average image size
    if image_sizes:
        avg_width = sum(w for w, h in image_sizes) / len(image_sizes)
        avg_height = sum(h for w, h in image_sizes) / len(image_sizes)
    else:
        avg_width = avg_height = None

    analysis = {
        'total_samples': total_samples,
        'num_classes': num_classes,
        'classes': dataset.classes,
        'class_distribution': distribution,
        'class_percentages': class_percentages,
        'average_image_size': (avg_width, avg_height) if avg_width else None,
        'sample_image_sizes': image_sizes[:10]  # First 10 as examples
    }

    return analysis


def print_dataset_summary(dataset: XRayDataset):
    """Print a summary of the dataset."""
    analysis = analyze_dataset(dataset)

    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Number of classes: {analysis['num_classes']}")
    print(f"Classes: {', '.join(analysis['classes'])}")
    print()

    print("Class Distribution:")
    for cls, count in analysis['class_distribution'].items():
        percentage = analysis['class_percentages'][cls]
        print(f"  {cls}: {count} samples ({percentage:.1f}%)")
    print()

    if analysis['average_image_size']:
        avg_w, avg_h = analysis['average_image_size']
        print(f"Average image size: {avg_w:.0f} x {avg_h:.0f} pixels")

    print("=" * 50)


# Example usage and testing
if __name__ == "__main__":
    import torchvision.transforms as transforms

    # Example transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Test dataset creation (this would fail without actual data)
    try:
        dataset = XRayDataset(
            data_dir="./test_data",  # This directory doesn't exist
            transform=transform
        )
        print_dataset_summary(dataset)

    except ValueError as e:
        print(f"Expected error (no test data): {e}")
        print("To test with real data, provide a valid data directory path.")
