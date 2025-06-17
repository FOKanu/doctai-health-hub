"""
EEG Classifier Training Script
Trains CNN-LSTM model for EEG seizure detection.
"""

import argparse


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train EEG Classifier')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    args = parser.parse_args()

    print("EEG Classifier Training")
    print("=" * 30)
    print("Expected EEG classes: Normal, Seizure, Abnormal")
    print("Training pipeline ready for implementation.")


if __name__ == "__main__":
    main()
