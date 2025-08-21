"""
Medical Image Manager - Centralized Model Router
Manages all medical image classification models and provides a unified interface.
"""

import os
import sys
import importlib
import time
from typing import Dict, Optional, Any
from enum import Enum
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction_schema import PredictionResult, ImageType, RiskLevel, create_prediction_result


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoadingStrategy(str, Enum):
    """Model loading strategies"""
    LAZY = "lazy"  # Load models on first use
    EAGER = "eager"  # Load all models at startup
    ON_DEMAND = "on_demand"  # Load and unload models as needed


class MedicalImageManager:
    """
    Centralized manager for all medical image classification models.
    Provides a unified interface for prediction across different image types.
    """

    def __init__(self, loading_strategy: ModelLoadingStrategy = ModelLoadingStrategy.LAZY):
        """
        Initialize the medical image manager.

        Args:
            loading_strategy: Strategy for loading models (lazy, eager, on_demand)
        """
        self.loading_strategy = loading_strategy
        self.loaded_models: Dict[ImageType, Any] = {}
        self.model_configs = self._get_model_configurations()

        # Track model performance
        self.prediction_stats = {
            image_type: {
                'total_predictions': 0,
                'total_time_ms': 0.0,
                'avg_time_ms': 0.0,
                'errors': 0
            }
            for image_type in ImageType
        }

        logger.info(f"Medical Image Manager initialized with {loading_strategy.value} loading strategy")

        # Load all models if eager loading is enabled
        if self.loading_strategy == ModelLoadingStrategy.EAGER:
            self._load_all_models()

    def _get_model_configurations(self) -> Dict[ImageType, Dict[str, str]]:
        """Get configuration for all available models."""
        return {
            ImageType.XRAY: {
                'module_path': 'xray_classifier.model',
                'model_dir': 'xray_classifier',
                'description': 'Chest X-ray classifier for pneumonia and COVID-19 detection'
            },
            ImageType.CT_SCAN: {
                'module_path': 'ct_scan_classifier.model',
                'model_dir': 'ct_scan_classifier',
                'description': 'CT scan classifier for brain abnormalities'
            },
            ImageType.MRI: {
                'module_path': 'mri_classifier.model',
                'model_dir': 'mri_classifier',
                'description': 'MRI classifier for brain tumor detection'
            },
            ImageType.EEG: {
                'module_path': 'eeg_classifier.model',
                'model_dir': 'eeg_classifier',
                'description': 'EEG classifier for seizure detection'
            },
            ImageType.SKIN_LESION: {
                'module_path': 'skin_lesion_classifier.model',
                'model_dir': 'skin_lesion_classifier',
                'description': 'Skin lesion classifier for melanoma detection'
            }
        }

    def _load_model(self, image_type: ImageType) -> bool:
        """
        Load a specific model.

        Args:
            image_type: Type of medical image model to load

        Returns:
            bool: True if model loaded successfully
        """
        if image_type in self.loaded_models:
            logger.info(f"{image_type.value} model already loaded")
            return True

        try:
            config = self.model_configs[image_type]
            module_path = config['module_path']

            logger.info(f"Loading {image_type.value} model from {module_path}")

            # Import the model module
            model_module = importlib.import_module(module_path)

            # Load the model
            success = model_module.load_model()

            if success:
                self.loaded_models[image_type] = model_module
                logger.info(f"✅ {image_type.value} model loaded successfully")
                return True
            else:
                logger.error(f"❌ Failed to load {image_type.value} model")
                return False

        except ImportError as e:
            logger.error(f"❌ Failed to import {image_type.value} model module: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error loading {image_type.value} model: {e}")
            return False

    def _load_all_models(self):
        """Load all available models."""
        logger.info("Loading all models...")

        for image_type in ImageType:
            self._load_model(image_type)

        loaded_count = len(self.loaded_models)
        total_count = len(ImageType)

        logger.info(f"Loaded {loaded_count}/{total_count} models successfully")

    def _unload_model(self, image_type: ImageType):
        """
        Unload a specific model to free memory.

        Args:
            image_type: Type of medical image model to unload
        """
        if image_type in self.loaded_models:
            del self.loaded_models[image_type]
            logger.info(f"Unloaded {image_type.value} model")

    def predict(self, image_bytes: bytes, image_type: str, image_id: str = None) -> PredictionResult:
        """
        Predict medical image classification.

        Args:
            image_bytes: Raw image bytes
            image_type: Type of medical image (xray, ct_scan, mri, eeg, skin_lesion)
            image_id: Optional unique identifier for the image

        Returns:
            PredictionResult: Standardized prediction result
        """
        # Validate and convert image type
        try:
            img_type = ImageType(image_type.lower())
        except ValueError:
            supported_types = [t.value for t in ImageType]
            return self._create_error_result(
                image_id or "unknown",
                f"Unsupported image type '{image_type}'. Supported types: {supported_types}",
                image_type
            )

        # Generate image ID if not provided
        if image_id is None:
            image_id = f"{img_type.value}_{int(time.time() * 1000)}"

        # Load model if not already loaded
        if img_type not in self.loaded_models:
            if not self._load_model(img_type):
                return self._create_error_result(
                    image_id,
                    f"Failed to load {img_type.value} model",
                    image_type
                )

        # Get the model module
        model_module = self.loaded_models[img_type]

        # Track prediction timing
        start_time = time.time()

        try:
            # Make prediction
            result = model_module.predict(image_bytes, image_id)

            # Update statistics
            prediction_time_ms = (time.time() - start_time) * 1000
            self._update_stats(img_type, prediction_time_ms, success=True)

            logger.info(f"Prediction completed for {img_type.value}: {result.top_class} "
                       f"(confidence: {result.confidence:.3f}, time: {prediction_time_ms:.1f}ms)")

            return result

        except Exception as e:
            # Update error statistics
            prediction_time_ms = (time.time() - start_time) * 1000
            self._update_stats(img_type, prediction_time_ms, success=False)

            logger.error(f"Prediction failed for {img_type.value}: {e}")

            return self._create_error_result(
                image_id,
                f"Prediction failed: {str(e)}",
                image_type
            )
        finally:
            # Unload model if using on-demand strategy
            if self.loading_strategy == ModelLoadingStrategy.ON_DEMAND:
                self._unload_model(img_type)

    def _create_error_result(self, image_id: str, error_message: str, image_type: str) -> PredictionResult:
        """Create a standardized error result."""
        try:
            img_type = ImageType(image_type.lower())
        except ValueError:
            img_type = ImageType.XRAY  # Default fallback

        return create_prediction_result(
            image_id=image_id,
            image_type=img_type,
            model_name="medical_image_manager",
            model_version="1.0.0",
            top_class="Error",
            top_class_index=-1,
            confidence=0.0,
            class_probabilities={"Error": 1.0},
            class_labels=["Error"],
            risk_level=RiskLevel.HIGH,
            recommendations=["Unable to process image. Please consult a medical professional."],
            additional_info={"error": error_message}
        )

    def _update_stats(self, image_type: ImageType, prediction_time_ms: float, success: bool):
        """Update prediction statistics."""
        stats = self.prediction_stats[image_type]

        if success:
            stats['total_predictions'] += 1
            stats['total_time_ms'] += prediction_time_ms
            stats['avg_time_ms'] = stats['total_time_ms'] / stats['total_predictions']
        else:
            stats['errors'] += 1

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all available models."""
        info = {
            'loading_strategy': self.loading_strategy.value,
            'loaded_models': list(self.loaded_models.keys()),
            'available_models': {},
            'prediction_stats': self.prediction_stats
        }

        for image_type, config in self.model_configs.items():
            info['available_models'][image_type.value] = {
                'description': config['description'],
                'loaded': image_type in self.loaded_models,
                'module_path': config['module_path']
            }

        return info

    def get_supported_image_types(self) -> list:
        """Get list of supported image types."""
        return [image_type.value for image_type in ImageType]

    def preload_model(self, image_type: str) -> bool:
        """
        Preload a specific model.

        Args:
            image_type: Type of medical image model to preload

        Returns:
            bool: True if model loaded successfully
        """
        try:
            img_type = ImageType(image_type.lower())
            return self._load_model(img_type)
        except ValueError:
            logger.error(f"Invalid image type: {image_type}")
            return False

    def unload_model(self, image_type: str) -> bool:
        """
        Unload a specific model.

        Args:
            image_type: Type of medical image model to unload

        Returns:
            bool: True if model unloaded successfully
        """
        try:
            img_type = ImageType(image_type.lower())
            if img_type in self.loaded_models:
                self._unload_model(img_type)
                return True
            else:
                logger.warning(f"{image_type} model is not loaded")
                return False
        except ValueError:
            logger.error(f"Invalid image type: {image_type}")
            return False

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics for all models."""
        return {
            'loading_strategy': self.loading_strategy.value,
            'loaded_models_count': len(self.loaded_models),
            'stats_by_model': self.prediction_stats
        }


# Global manager instance
_manager: Optional[MedicalImageManager] = None


def get_manager(loading_strategy: ModelLoadingStrategy = ModelLoadingStrategy.LAZY) -> MedicalImageManager:
    """
    Get the global medical image manager instance.

    Args:
        loading_strategy: Strategy for loading models

    Returns:
        MedicalImageManager: Global manager instance
    """
    global _manager
    if _manager is None:
        _manager = MedicalImageManager(loading_strategy)
    return _manager


def predict(image_bytes: bytes, image_type: str, image_id: str = None) -> PredictionResult:
    """
    Convenience function for making predictions.

    Args:
        image_bytes: Raw image bytes
        image_type: Type of medical image
        image_id: Optional unique identifier for the image

    Returns:
        PredictionResult: Standardized prediction result
    """
    manager = get_manager()
    return manager.predict(image_bytes, image_type, image_id)


def get_supported_image_types() -> list:
    """Get list of supported image types."""
    manager = get_manager()
    return manager.get_supported_image_types()


def get_model_info() -> Dict[str, Any]:
    """Get information about all available models."""
    manager = get_manager()
    return manager.get_model_info()


# Example usage and testing
if __name__ == "__main__":
    import json

    print("Medical Image Manager - Testing")
    print("=" * 50)

    # Initialize manager
    manager = get_manager(ModelLoadingStrategy.LAZY)

    # Get model info
    info = manager.get_model_info()
    print("Available models:")
    print(json.dumps(info['available_models'], indent=2))
    print()

    # Test with mock data
    mock_image_bytes = b"mock_medical_image_data"

    # Test each image type
    for image_type in manager.get_supported_image_types():
        print(f"Testing {image_type} classifier...")

        try:
            result = manager.predict(mock_image_bytes, image_type, f"test_{image_type}_001")
            print(f"  ✅ Prediction: {result.top_class}")
            print(f"  ✅ Confidence: {result.confidence:.3f}")
            print(f"  ✅ Risk Level: {result.risk_level}")
            print(f"  ✅ Processing Time: {result.processing_time_ms:.1f}ms")
        except Exception as e:
            print(f"  ❌ Error: {e}")

        print()

    # Get final statistics
    stats = manager.get_prediction_stats()
    print("Prediction Statistics:")
    print(json.dumps(stats, indent=2, default=str))

    print("\n✅ Medical Image Manager testing completed!")
