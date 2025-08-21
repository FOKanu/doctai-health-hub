"""
Skin Lesion Classifier Training Script
Trains EfficientNet-B0 model for skin lesion classification.
"""

import argparse


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Skin Lesion Classifier')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    args = parser.parse_args()

    print("Skin Lesion Classifier Training")
    print("=" * 40)
    print("Expected classes:")
    classes = [
        "Melanoma", "Melanocytic nevus", "Basal cell carcinoma",
        "Actinic keratosis", "Benign keratosis", "Dermatofibroma", "Vascular lesion"
    ]
    for cls in classes:
        print(f"  - {cls}")
    print("Training pipeline ready for implementation.")


if __name__ == "__main__":
    main()
