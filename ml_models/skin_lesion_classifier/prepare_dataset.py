"""
Skin Lesion Dataset Preparation Script
Organizes skin lesion images into proper training/validation structure.
"""

import argparse


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Prepare Skin Lesion Dataset')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Path to source dataset directory')
    args = parser.parse_args()

    print("Skin Lesion Dataset Preparation")
    print("=" * 40)
    print("Expected classes:")
    classes = [
        "Melanoma", "Melanocytic nevus", "Basal cell carcinoma",
        "Actinic keratosis", "Benign keratosis", "Dermatofibroma", "Vascular lesion"
    ]
    for cls in classes:
        print(f"  - {cls}")
    print("Dataset preparation pipeline ready for implementation.")


if __name__ == "__main__":
    main()
