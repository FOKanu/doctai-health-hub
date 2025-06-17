#!/usr/bin/env python3
"""
Local Prediction CLI Script
Run medical image predictions locally from the command line.

Usage:
    python run_local_prediction.py --image path/to/image.jpg --type xray
    python run_local_prediction.py --image path/to/ct_scan.png --type ct_scan --id my_scan_001
    python run_local_prediction.py --image path/to/mri.jpg --type mri --verbose
"""

import argparse
import json
import sys
import os
from pathlib import Path
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from medical_image_manager import get_manager, get_supported_image_types, ModelLoadingStrategy
from prediction_schema import PredictionResult


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run medical image predictions locally',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --image chest_xray.jpg --type xray
  %(prog)s --image brain_ct.png --type ct_scan --id patient_001
  %(prog)s --image brain_mri.jpg --type mri --verbose --output results.json
  %(prog)s --image eeg_spectrogram.png --type eeg
  %(prog)s --image skin_lesion.jpg --type skin_lesion

Supported image types:
  """ + ", ".join(get_supported_image_types())
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to the medical image file'
    )

    parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=get_supported_image_types(),
        help='Type of medical image'
    )

    parser.add_argument(
        '--id',
        type=str,
        default=None,
        help='Optional unique identifier for the image'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional output file path to save results as JSON'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output with detailed information'
    )

    parser.add_argument(
        '--loading-strategy',
        type=str,
        choices=['lazy', 'eager', 'on_demand'],
        default='lazy',
        help='Model loading strategy (default: lazy)'
    )

    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print JSON output'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show prediction statistics after processing'
    )

    return parser.parse_args()


def load_image_bytes(image_path: str) -> bytes:
    """
    Load image file as bytes.

    Args:
        image_path: Path to the image file

    Returns:
        bytes: Image file contents as bytes

    Raises:
        FileNotFoundError: If image file doesn't exist
        IOError: If unable to read image file
    """
    image_file = Path(image_path)

    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not image_file.is_file():
        raise IOError(f"Path is not a file: {image_path}")

    try:
        with open(image_file, 'rb') as f:
            return f.read()
    except Exception as e:
        raise IOError(f"Unable to read image file {image_path}: {e}")


def format_prediction_result(result: PredictionResult, verbose: bool = False) -> dict:
    """
    Format prediction result for output.

    Args:
        result: Prediction result from the model
        verbose: Whether to include verbose information

    Returns:
        dict: Formatted result dictionary
    """
    # Basic result information
    formatted_result = {
        'prediction': {
            'top_class': result.top_class,
            'confidence': round(result.confidence, 4),
            'risk_level': result.risk_level
        },
        'recommendations': result.recommendations,
        'metadata': {
            'image_id': result.image_id,
            'image_type': result.image_type,
            'model_name': result.model_name,
            'model_version': result.model_version,
            'processing_time_ms': round(result.processing_time_ms or 0, 2),
            'created_at': result.created_at.isoformat() if result.created_at else None
        }
    }

    # Add verbose information if requested
    if verbose:
        formatted_result.update({
            'class_probabilities': {
                k: round(v, 4) for k, v in result.class_probabilities.items()
            },
            'class_labels': result.class_labels,
            'additional_info': result.additional_info
        })

    return formatted_result


def print_summary(result: PredictionResult, verbose: bool = False):
    """Print a human-readable summary of the prediction."""
    print("\n" + "=" * 60)
    print("MEDICAL IMAGE PREDICTION SUMMARY")
    print("=" * 60)

    print(f"Image ID: {result.image_id}")
    print(f"Image Type: {result.image_type}")
    print(f"Model: {result.model_name} v{result.model_version}")
    print()

    print("PREDICTION RESULTS:")
    print(f"  Top Class: {result.top_class}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Risk Level: {result.risk_level.upper()}")

    if result.processing_time_ms:
        print(f"  Processing Time: {result.processing_time_ms:.1f}ms")
    print()

    if verbose and result.class_probabilities:
        print("CLASS PROBABILITIES:")
        sorted_probs = sorted(
            result.class_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for class_name, prob in sorted_probs:
            print(f"  {class_name}: {prob:.1%}")
        print()

    if result.recommendations:
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
        print()

    if verbose and result.additional_info:
        print("ADDITIONAL INFO:")
        for key, value in result.additional_info.items():
            print(f"  {key}: {value}")
        print()

    print("=" * 60)


def save_results(result_dict: dict, output_path: str, pretty: bool = False):
    """Save results to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            if pretty:
                json.dump(result_dict, f, indent=2, default=str)
            else:
                json.dump(result_dict, f, default=str)

        print(f"Results saved to: {output_path}")

    except Exception as e:
        print(f"Error saving results to {output_path}: {e}", file=sys.stderr)


def main():
    """Main function."""
    try:
        # Parse arguments
        args = parse_arguments()

        if args.verbose:
            print("Local Medical Image Prediction CLI")
            print("=" * 40)
            print(f"Image: {args.image}")
            print(f"Type: {args.type}")
            print(f"ID: {args.id or 'auto-generated'}")
            print(f"Loading Strategy: {args.loading_strategy}")
            print()

        # Load image
        if args.verbose:
            print("Loading image...")

        try:
            image_bytes = load_image_bytes(args.image)
            image_size_kb = len(image_bytes) / 1024

            if args.verbose:
                print(f"Image loaded successfully ({image_size_kb:.1f} KB)")

        except (FileNotFoundError, IOError) as e:
            print(f"Error loading image: {e}", file=sys.stderr)
            return 1

        # Initialize manager
        if args.verbose:
            print("Initializing medical image manager...")

        loading_strategy = ModelLoadingStrategy(args.loading_strategy)
        manager = get_manager(loading_strategy)

        # Make prediction
        if args.verbose:
            print(f"Running {args.type} prediction...")

        start_time = time.time()
        result = manager.predict(image_bytes, args.type, args.id)
        total_time = time.time() - start_time

        if args.verbose:
            print(f"Prediction completed in {total_time:.2f} seconds")

        # Format results
        result_dict = format_prediction_result(result, args.verbose)

        # Print results
        if args.verbose:
            print_summary(result, verbose=True)
        else:
            # Print JSON output
            if args.pretty:
                print(json.dumps(result_dict, indent=2, default=str))
            else:
                print(json.dumps(result_dict, default=str))

        # Save results if output file specified
        if args.output:
            save_results(result_dict, args.output, args.pretty)

        # Show statistics if requested
        if args.stats:
            stats = manager.get_prediction_stats()
            print("\nPREDICTION STATISTICS:")
            print(json.dumps(stats, indent=2, default=str))

        # Return appropriate exit code
        if result.top_class == "Error":
            return 1
        else:
            return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
