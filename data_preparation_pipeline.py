#!/usr/bin/env python3
"""
Medical Image Data Preparation Pipeline
======================================

This script prepares MRI, CT, and X-ray datasets for ML training by:
1. Organizing datasets into a unified structure
2. Cleaning images by removing watermarks, labels, and text
3. Standardizing image formats and sizes
4. Creating proper train/validation/test splits

Author: Health AI Team
Date: 2024
"""

import os
import sys
import shutil
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MedicalImageCleaner:
    """Cleans medical images by removing watermarks, labels, and text."""

    def __init__(self):
        self.text_detection_model = None
        self.initialize_text_detection()

    def initialize_text_detection(self):
        """Initialize text detection model (if available)."""
        try:
            import easyocr
            self.text_detection_model = easyocr.Reader(['en'])
            logger.info("EasyOCR initialized for text detection")
        except ImportError:
            logger.warning("EasyOCR not available. Using basic text removal methods.")
            self.text_detection_model = None

    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions in the image."""
        text_regions = []

        if self.text_detection_model:
            # Use EasyOCR for text detection
            results = self.text_detection_model.readtext(image)
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Confidence threshold
                    x1, y1 = bbox[0]
                    x3, y3 = bbox[2]
                    text_regions.append((int(x1), int(y1), int(x3), int(y3)))
        else:
            # Basic text detection using morphological operations
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply morphological operations to detect text-like regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            # Find contours that might be text
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Filter for text-like regions (rectangular, small aspect ratio)
                if 0.1 < aspect_ratio < 10 and 10 < w < 200 and 5 < h < 50:
                    text_regions.append((x, y, x + w, y + h))

        return text_regions

    def detect_watermarks(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect watermark regions in the image."""
        watermark_regions = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect semi-transparent watermarks
        # Watermarks often have lower contrast
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blur)

        # Threshold to find low-contrast regions
        _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter for watermark-like regions
            if 100 < area < 10000:  # Reasonable watermark size
                watermark_regions.append((x, y, x + w, y + h))

        return watermark_regions

    def detect_labels_and_letters(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect labels and letter regions in the image."""
        label_regions = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # Filter for label-like regions
            if 0.5 < aspect_ratio < 5 and 20 < w < 300 and 10 < h < 100:
                label_regions.append((x, y, x + w, y + h))

        return label_regions

    def remove_regions(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Remove detected regions using inpainting."""
        if not regions:
            return image

        # Create mask for regions to remove
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for x1, y1, x2, y2 in regions:
            # Add padding around region
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)

            mask[y1:y2, x1:x2] = 255

        # Use inpainting to fill removed regions
        cleaned_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

        return cleaned_image

    def clean_image(self, image_path: str, output_path: str) -> bool:
        """Clean a single image by removing watermarks, labels, and text."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return False

            # Detect regions to remove
            text_regions = self.detect_text_regions(image)
            watermark_regions = self.detect_watermarks(image)
            label_regions = self.detect_labels_and_letters(image)

            # Combine all regions
            all_regions = text_regions + watermark_regions + label_regions

            # Remove regions if any detected
            if all_regions:
                cleaned_image = self.remove_regions(image, all_regions)
                logger.info(f"Cleaned {len(all_regions)} regions from {image_path}")
            else:
                cleaned_image = image
                logger.debug(f"No regions to clean in {image_path}")

            # Save cleaned image
            cv2.imwrite(output_path, cleaned_image)
            return True

        except Exception as e:
            logger.error(f"Error cleaning image {image_path}: {str(e)}")
            return False

class DatasetOrganizer:
    """Organizes medical image datasets into a unified structure."""

    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.cleaner = MedicalImageCleaner()

        # Create output directory structure
        self.create_directory_structure()

    def create_directory_structure(self):
        """Create the organized directory structure."""
        structure = {
            'train': ['mri', 'ct', 'xray'],
            'validation': ['mri', 'ct', 'xray'],
            'test': ['mri', 'ct', 'xray'],
            'metadata': [],
            'cleaned_images': ['mri', 'ct', 'xray']
        }

        for main_dir, subdirs in structure.items():
            main_path = self.output_dir / main_dir
            main_path.mkdir(parents=True, exist_ok=True)

            for subdir in subdirs:
                sub_path = main_path / subdir
                sub_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created directory structure in {self.output_dir}")

    def organize_mri_dataset(self):
        """Organize MRI brain tumor dataset."""
        mri_source = self.source_dir / "MRI_Scans_organized_dataset"
        if not mri_source.exists():
            logger.warning(f"MRI dataset not found at {mri_source}")
            return

        logger.info("Organizing MRI dataset...")

        # Process train and test directories
        for split in ['train', 'test']:
            split_source = mri_source / split
            if not split_source.exists():
                continue

            # Create split directory
            split_output = self.output_dir / split / 'mri'
            split_output.mkdir(parents=True, exist_ok=True)

            # Process each class
            for class_dir in split_source.iterdir():
                if class_dir.is_dir():
                    class_name = self.clean_class_name(class_dir.name)
                    class_output = split_output / class_name
                    class_output.mkdir(parents=True, exist_ok=True)

                    # Process images in class
                    for img_path in class_dir.glob("*.jpg"):
                        output_path = class_output / img_path.name

                        # Clean and copy image
                        cleaned_path = self.output_dir / 'cleaned_images' / 'mri' / f"{class_name}_{img_path.stem}.jpg"
                        if self.cleaner.clean_image(str(img_path), str(cleaned_path)):
                            shutil.copy2(str(cleaned_path), str(output_path))
                            os.remove(str(cleaned_path))  # Remove temporary cleaned file
                        else:
                            shutil.copy2(str(img_path), str(output_path))

    def organize_ct_datasets(self):
        """Organize CT datasets (brain, lung, kidney, stroke)."""
        logger.info("Organizing CT datasets...")

        # Brain CT
        brain_ct_source = self.source_dir / "Computed Tomography (CT) of the Brain"
        if brain_ct_source.exists():
            self.process_ct_dataset(brain_ct_source, "brain_ct")

        # Lung CT Cancer
        lung_ct_source = self.source_dir / "Lung_CT_Cancer"
        if lung_ct_source.exists():
            self.process_ct_dataset(lung_ct_source, "lung_ct_cancer")

        # Kidney CT
        kidney_ct_source = self.source_dir / "CT KIDNEY DATASET- Normal-Cyst-Tumor"
        if kidney_ct_source.exists():
            self.process_kidney_ct_dataset(kidney_ct_source)

        # Brain Stroke CT
        stroke_ct_source = self.source_dir / "Brain_Stroke_CT_Dataset"
        if stroke_ct_source.exists():
            self.process_ct_dataset(stroke_ct_source, "brain_stroke_ct")

        # COVID-19 CT
        covid_ct_source = self.source_dir / "COVID-19_Lung_CT_Scans"
        if covid_ct_source.exists():
            self.process_covid_ct_dataset(covid_ct_source)

    def organize_xray_dataset(self):
        """Organize X-ray chest dataset."""
        xray_source = self.source_dir / "Chest_X_ray_17_Diseases"
        if not xray_source.exists():
            logger.warning(f"X-ray dataset not found at {xray_source}")
            return

        logger.info("Organizing X-ray dataset...")

        # Read CSV file
        csv_path = xray_source / "xray_chest.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)

            # Create train/validation/test splits
            train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['type'])
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['type'])

            # Process each split
            for split_name, split_df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
                split_output = self.output_dir / split_name / 'xray'
                split_output.mkdir(parents=True, exist_ok=True)

                for _, row in split_df.iterrows():
                    class_name = self.clean_class_name(row['type'])
                    class_output = split_output / class_name
                    class_output.mkdir(parents=True, exist_ok=True)

                    # Process image path
                    img_path = xray_source / "files" / row['jpg'].lstrip('/')
                    if img_path.exists():
                        output_path = class_output / f"{class_name}_{img_path.stem}.jpg"

                        # Clean and copy image
                        cleaned_path = self.output_dir / 'cleaned_images' / 'xray' / f"{class_name}_{img_path.stem}.jpg"
                        if self.cleaner.clean_image(str(img_path), str(cleaned_path)):
                            shutil.copy2(str(cleaned_path), str(output_path))
                            os.remove(str(cleaned_path))
                        else:
                            shutil.copy2(str(img_path), str(output_path))

    def process_ct_dataset(self, source_dir: Path, dataset_name: str):
        """Process a generic CT dataset."""
        logger.info(f"Processing {dataset_name}...")

        # Find all image files
        image_files = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.png"))

        if not image_files:
            logger.warning(f"No images found in {source_dir}")
            return

        # Create train/validation/test splits
        train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

        # Process each split
        for split_name, split_files in [('train', train_files), ('validation', val_files), ('test', test_files)]:
            split_output = self.output_dir / split_name / 'ct'
            split_output.mkdir(parents=True, exist_ok=True)

            for img_path in split_files:
                # Determine class from path
                class_name = self.determine_ct_class(img_path, dataset_name)
                class_output = split_output / class_name
                class_output.mkdir(parents=True, exist_ok=True)

                output_path = class_output / f"{class_name}_{img_path.stem}.jpg"

                # Clean and copy image
                cleaned_path = self.output_dir / 'cleaned_images' / 'ct' / f"{class_name}_{img_path.stem}.jpg"
                if self.cleaner.clean_image(str(img_path), str(cleaned_path)):
                    shutil.copy2(str(cleaned_path), str(output_path))
                    os.remove(str(cleaned_path))
                else:
                    shutil.copy2(str(img_path), str(output_path))

    def process_kidney_ct_dataset(self, source_dir: Path):
        """Process kidney CT dataset with CSV metadata."""
        logger.info("Processing kidney CT dataset...")

        # Read CSV file
        csv_path = source_dir / "kidneyData.csv"
        if not csv_path.exists():
            logger.warning(f"Kidney CSV not found at {csv_path}")
            return

        df = pd.read_csv(csv_path)

        # Create train/validation/test splits
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Class'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['Class'])

        # Process each split
        for split_name, split_df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
            split_output = self.output_dir / split_name / 'ct'
            split_output.mkdir(parents=True, exist_ok=True)

            for _, row in split_df.iterrows():
                class_name = self.clean_class_name(row['Class'])
                class_output = split_output / class_name
                class_output.mkdir(parents=True, exist_ok=True)

                # Process image path
                img_path = source_dir / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone" / row['path'].split('/')[-1]
                if img_path.exists():
                    output_path = class_output / f"{class_name}_{img_path.stem}.jpg"

                    # Clean and copy image
                    cleaned_path = self.output_dir / 'cleaned_images' / 'ct' / f"{class_name}_{img_path.stem}.jpg"
                    if self.cleaner.clean_image(str(img_path), str(cleaned_path)):
                        shutil.copy2(str(cleaned_path), str(output_path))
                        os.remove(str(cleaned_path))
                    else:
                        shutil.copy2(str(img_path), str(output_path))

    def process_covid_ct_dataset(self, source_dir: Path):
        """Process COVID-19 CT dataset."""
        logger.info("Processing COVID-19 CT dataset...")

        # Process COVID-19 and Non-COVID-19 directories
        for class_name in ['COVID-19', 'Non-COVID-19']:
            class_dir = source_dir / class_name
            if not class_dir.exists():
                continue

            # Find all images
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

            if not image_files:
                continue

            # Create splits
            train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
            val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

            # Process each split
            for split_name, split_files in [('train', train_files), ('validation', val_files), ('test', test_files)]:
                split_output = self.output_dir / split_name / 'ct'
                split_output.mkdir(parents=True, exist_ok=True)

                clean_class_name = self.clean_class_name(class_name)
                class_output = split_output / clean_class_name
                class_output.mkdir(parents=True, exist_ok=True)

                for img_path in split_files:
                    output_path = class_output / f"{clean_class_name}_{img_path.stem}.jpg"

                    # Clean and copy image
                    cleaned_path = self.output_dir / 'cleaned_images' / 'ct' / f"{clean_class_name}_{img_path.stem}.jpg"
                    if self.cleaner.clean_image(str(img_path), str(cleaned_path)):
                        shutil.copy2(str(cleaned_path), str(output_path))
                        os.remove(str(cleaned_path))
                    else:
                        shutil.copy2(str(img_path), str(output_path))

    def clean_class_name(self, class_name: str) -> str:
        """Clean class name for file system compatibility."""
        # Remove special characters and spaces
        cleaned = re.sub(r'[^\w\s-]', '', class_name)
        cleaned = re.sub(r'[-\s]+', '_', cleaned)
        cleaned = cleaned.lower().strip('_')
        return cleaned

    def determine_ct_class(self, img_path: Path, dataset_name: str) -> str:
        """Determine class name from image path for CT datasets."""
        path_str = str(img_path).lower()

        if 'normal' in path_str:
            return 'normal'
        elif 'tumor' in path_str or 'cancer' in path_str:
            return 'tumor'
        elif 'cyst' in path_str:
            return 'cyst'
        elif 'stone' in path_str:
            return 'stone'
        elif 'bleeding' in path_str:
            return 'bleeding'
        elif 'ischemia' in path_str:
            return 'ischemia'
        elif 'covid' in path_str:
            return 'covid_19'
        else:
            return dataset_name

    def create_metadata(self):
        """Create metadata files for the organized dataset."""
        logger.info("Creating metadata...")

        metadata = {
            'dataset_info': {
                'name': 'Medical Imaging Trio Dataset',
                'description': 'Organized and cleaned MRI, CT, and X-ray datasets',
                'total_images': 0,
                'modalities': ['mri', 'ct', 'xray'],
                'splits': ['train', 'validation', 'test']
            },
            'class_distribution': {},
            'modality_distribution': {},
            'cleaning_stats': {
                'images_cleaned': 0,
                'regions_removed': 0
            }
        }

        # Count images and classes
        for split in ['train', 'validation', 'test']:
            for modality in ['mri', 'ct', 'xray']:
                modality_dir = self.output_dir / split / modality
                if modality_dir.exists():
                    for class_dir in modality_dir.iterdir():
                        if class_dir.is_dir():
                            class_name = class_dir.name
                            image_count = len(list(class_dir.glob("*.jpg")))

                            # Update metadata
                            metadata['total_images'] += image_count

                            if class_name not in metadata['class_distribution']:
                                metadata['class_distribution'][class_name] = {}
                            metadata['class_distribution'][class_name][split] = image_count

                            if modality not in metadata['modality_distribution']:
                                metadata['modality_distribution'][modality] = {}
                            if split not in metadata['modality_distribution'][modality]:
                                metadata['modality_distribution'][modality][split] = 0
                            metadata['modality_distribution'][modality][split] += image_count

        # Save metadata
        metadata_path = self.output_dir / 'metadata' / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")
        return metadata

    def organize_all_datasets(self):
        """Organize all datasets."""
        logger.info("Starting dataset organization...")

        # Organize each modality
        self.organize_mri_dataset()
        self.organize_ct_datasets()
        self.organize_xray_dataset()

        # Create metadata
        metadata = self.create_metadata()

        logger.info("Dataset organization completed!")
        logger.info(f"Total images organized: {metadata['total_images']}")

        return metadata

def main():
    """Main function to run the data preparation pipeline."""
    parser = argparse.ArgumentParser(description='Medical Image Data Preparation Pipeline')
    parser.add_argument('--source_dir', type=str, required=True,
                       help='Path to source dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output organized dataset directory')
    parser.add_argument('--skip_cleaning', action='store_true',
                       help='Skip image cleaning step')

    args = parser.parse_args()

    # Validate paths
    source_path = Path(args.source_dir)
    if not source_path.exists():
        logger.error(f"Source directory does not exist: {source_path}")
        sys.exit(1)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create organizer
    organizer = DatasetOrganizer(args.source_dir, args.output_dir)

    # Organize datasets
    metadata = organizer.organize_all_datasets()

    # Print summary
    print("\n" + "="*50)
    print("DATASET ORGANIZATION COMPLETED")
    print("="*50)
    print(f"Total images: {metadata['total_images']}")
    print(f"Output directory: {args.output_dir}")
    print("\nClass distribution:")
    for class_name, splits in metadata['class_distribution'].items():
        total = sum(splits.values())
        print(f"  {class_name}: {total} images")
    print("\nModality distribution:")
    for modality, splits in metadata['modality_distribution'].items():
        total = sum(splits.values())
        print(f"  {modality}: {total} images")
    print("="*50)

if __name__ == "__main__":
    main()
