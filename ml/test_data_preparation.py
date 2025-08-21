#!/usr/bin/env python3
"""
Test script for Medical Image Data Preparation Pipeline
======================================================

This script tests the setup and verifies that all dependencies are available.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing package imports...")

    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'matplotlib'),
        ('sklearn', 'scikit-learn'),
        ('tqdm', 'tqdm'),
        ('pathlib', 'pathlib2')
    ]

    missing_packages = []

    for module, package in required_packages:
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            missing_packages.append(package)

    # Test EasyOCR separately (optional)
    try:
        import easyocr
        print("  ✅ easyocr (optional)")
    except ImportError:
        print("  ⚠️  easyocr - NOT INSTALLED (will use basic text detection)")

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip3 install -r data_preparation_requirements.txt")
        return False

    print("✅ All required packages are available!")
    return True

def test_source_directory():
    """Test if the source directory exists and has the expected structure."""
    print("\n📁 Testing source directory...")

    source_dir = Path("/Users/francis/Downloads/MRI_CT_XRAY_Datset")

    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False

    print(f"✅ Source directory exists: {source_dir}")

    # Check for expected subdirectories
    expected_dirs = [
        "MRI_Scans_organized_dataset",
        "Computed Tomography (CT) of the Brain",
        "Lung_CT_Cancer",
        "COVID-19_Lung_CT_Scans",
        "Chest_X_ray_17_Diseases",
        "Brain_Stroke_CT_Dataset",
        "CT KIDNEY DATASET- Normal-Cyst-Tumor"
    ]

    found_dirs = []
    for expected_dir in expected_dirs:
        if (source_dir / expected_dir).exists():
            print(f"  ✅ {expected_dir}")
            found_dirs.append(expected_dir)
        else:
            print(f"  ⚠️  {expected_dir} - NOT FOUND")

    if len(found_dirs) >= 3:
        print(f"✅ Found {len(found_dirs)} dataset directories")
        return True
    else:
        print("❌ Insufficient dataset directories found")
        return False

def test_output_directory():
    """Test if output directory can be created."""
    print("\n📁 Testing output directory...")

    output_dir = Path("./organized_medical_dataset")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory ready: {output_dir}")
        return True
    except Exception as e:
        print(f"❌ Cannot create output directory: {e}")
        return False

def test_image_processing():
    """Test basic image processing capabilities."""
    print("\n🖼️  Testing image processing...")

    try:
        import cv2
        import numpy as np

        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 255, 255]  # White square

        # Test basic operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        print("  ✅ OpenCV image processing")

        # Test inpainting
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        inpainted = cv2.inpaint(test_image, mask, 3, cv2.INPAINT_TELEA)

        print("  ✅ Inpainting functionality")

        return True

    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False

def test_data_preparation_pipeline():
    """Test if the main pipeline script can be imported."""
    print("\n🔧 Testing pipeline script...")

    try:
        # Test if the script exists
        script_path = Path("data_preparation_pipeline.py")
        if not script_path.exists():
            print("❌ data_preparation_pipeline.py not found")
            return False

        print("✅ Pipeline script found")

        # Test basic import (without running)
        import importlib.util
        spec = importlib.util.spec_from_file_location("pipeline", script_path)
        pipeline_module = importlib.util.module_from_spec(spec)

        # Test if key classes exist
        if hasattr(pipeline_module, 'MedicalImageCleaner'):
            print("  ✅ MedicalImageCleaner class")
        else:
            print("  ❌ MedicalImageCleaner class not found")
            return False

        if hasattr(pipeline_module, 'DatasetOrganizer'):
            print("  ✅ DatasetOrganizer class")
        else:
            print("  ❌ DatasetOrganizer class not found")
            return False

        return True

    except Exception as e:
        print(f"❌ Pipeline script test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🏥 Medical Image Data Preparation - Setup Test")
    print("=" * 50)

    tests = [
        ("Package Imports", test_imports),
        ("Source Directory", test_source_directory),
        ("Output Directory", test_output_directory),
        ("Image Processing", test_image_processing),
        ("Pipeline Script", test_data_preparation_pipeline)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! You're ready to run the data preparation pipeline.")
        print("\n🚀 Next steps:")
        print("   1. Run: ./run_data_preparation.sh")
        print("   2. Or run: python3 data_preparation_pipeline.py --source_dir '/Users/francis/Downloads/MRI_CT_XRAY_Datset' --output_dir './organized_medical_dataset'")
    else:
        print("❌ Some tests failed. Please fix the issues before running the pipeline.")
        print("\n🔧 Common fixes:")
        print("   1. Install missing packages: pip3 install -r data_preparation_requirements.txt")
        print("   2. Check source directory path")
        print("   3. Ensure you have write permissions")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
