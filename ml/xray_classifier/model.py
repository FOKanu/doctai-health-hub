"""
X-ray Classifier Model
Detects pneumonia, COVID-19, and normal findings in chest X-rays.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import time
from typing import Union
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction_schema import (
    PredictionResult,
    ImageType,
    RiskLevel,
    create_prediction_result,
    assess_risk_level,
    generate_recommendations
)


class XRayClassifier(nn.Module):
    """
    EfficientNet-B4 based X-ray classifier for detecting pneumonia and COVID-19.
    """

    def __init__(self, num_classes=3):
        super(XRayClassifier, self).__init__()
        # In a real implementation, this would load EfficientNet-B4
        # For now, we'll create a simple mock architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class XRayModelManager:
    """Manages X-ray classification model loading and prediction."""

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = ["Normal", "Pneumonia", "COVID-19"]
        self.model_name = "xray_classifier_efficientnet_b4"
        self.model_version = "1.0.0"

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path: str = None) -> bool:
        """
        Load the X-ray classification model.

        Args:
            model_path: Path to the model file. If None, loads default model.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        try:
            # In a real implementation, this would load the actual trained model
            # For now, we'll create a mock model
            self.model = XRayClassifier(num_classes=len(self.class_labels))

            # Mock loading pretrained weights
            # In reality: self.model.load_state_dict(torch.load(model_path))

            self.model.to(self.device)
            self.model.eval()

            print(f"✅ X-ray model loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"❌ Error loading X-ray model: {e}")
            return False

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocess image bytes for model input.

        Args:
            image_bytes: Raw image bytes

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms
        image_tensor = self.transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    def predict(self, image_bytes: bytes, image_id: str) -> PredictionResult:
        """
        Predict X-ray classification.

        Args:
            image_bytes: Raw image bytes
            image_id: Unique identifier for the image

        Returns:
            PredictionResult: Standardized prediction result
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_bytes)

            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                probabilities = probabilities.cpu().numpy()[0]

            # Get top prediction
            top_class_index = int(np.argmax(probabilities))
            top_class = self.class_labels[top_class_index]
            confidence = float(probabilities[top_class_index])

            # Create class probabilities dictionary
            class_probabilities = {
                label: float(prob)
                for label, prob in zip(self.class_labels, probabilities)
            }

            # Assess risk level
            risk_level = self._assess_xray_risk(top_class, confidence)

            # Generate recommendations
            recommendations = self._generate_xray_recommendations(top_class, risk_level)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Create standardized result
            result = create_prediction_result(
                image_id=image_id,
                image_type=ImageType.XRAY,
                model_name=self.model_name,
                model_version=self.model_version,
                top_class=top_class,
                top_class_index=top_class_index,
                confidence=confidence,
                class_probabilities=class_probabilities,
                class_labels=self.class_labels,
                risk_level=risk_level,
                recommendations=recommendations,
                processing_time_ms=processing_time_ms,
                additional_info={
                    "image_size": f"{image_tensor.shape[2]}x{image_tensor.shape[3]}",
                    "device_used": str(self.device),
                    "preprocessing_applied": ["resize", "normalize", "rgb_conversion"]
                }
            )

            return result

        except Exception as e:
            # Return error result
            processing_time_ms = (time.time() - start_time) * 1000
            return create_prediction_result(
                image_id=image_id,
                image_type=ImageType.XRAY,
                model_name=self.model_name,
                model_version=self.model_version,
                top_class="Error",
                top_class_index=-1,
                confidence=0.0,
                class_probabilities={"Error": 1.0},
                class_labels=["Error"],
                risk_level=RiskLevel.HIGH,
                recommendations=["Unable to process image. Please try again or consult a medical professional."],
                processing_time_ms=processing_time_ms,
                additional_info={"error": str(e)}
            )

    def _assess_xray_risk(self, top_class: str, confidence: float) -> RiskLevel:
        """Assess risk level for X-ray predictions."""
        if top_class == "Normal":
            return RiskLevel.LOW
        elif top_class in ["Pneumonia", "COVID-19"]:
            if confidence > 0.8:
                return RiskLevel.HIGH
            else:
                return RiskLevel.MEDIUM
        else:
            return RiskLevel.MEDIUM

    def _generate_xray_recommendations(self, top_class: str, risk_level: RiskLevel) -> list:
        """Generate X-ray specific recommendations."""
        base_recommendations = generate_recommendations(top_class, risk_level, ImageType.XRAY)

        # Add X-ray specific recommendations
        if top_class == "Normal":
            base_recommendations.extend([
                "No abnormalities detected",
                "Continue regular health check-ups"
            ])
        elif top_class == "Pneumonia":
            base_recommendations.extend([
                "Antibiotic treatment may be required",
                "Monitor respiratory symptoms",
                "Consider hospitalization if severe"
            ])
        elif top_class == "COVID-19":
            base_recommendations.extend([
                "Isolate immediately",
                "Contact health authorities",
                "Monitor oxygen saturation",
                "Follow COVID-19 protocols"
            ])

        return base_recommendations


# Global model manager instance
_model_manager = None


def load_model(model_path: str = None) -> bool:
    """
    Load the X-ray classification model.

    Args:
        model_path: Path to the model file

    Returns:
        bool: True if loaded successfully
    """
    global _model_manager
    _model_manager = XRayModelManager()
    return _model_manager.load_model(model_path)


def predict(image_bytes: bytes, image_id: str) -> PredictionResult:
    """
    Predict X-ray classification.

    Args:
        image_bytes: Raw image bytes
        image_id: Unique identifier for the image

    Returns:
        PredictionResult: Standardized prediction result
    """
    global _model_manager
    if _model_manager is None:
        raise ValueError("Model not loaded. Call load_model() first.")

    return _model_manager.predict(image_bytes, image_id)


# For testing purposes
if __name__ == "__main__":
    # Mock test
    print("Testing X-ray classifier...")

    # Load model
    success = load_model()
    if success:
        print("Model loaded successfully!")

        # Create mock image bytes (in reality, this would be actual image data)
        mock_image_bytes = b"mock_xray_image_data"

        try:
            result = predict(mock_image_bytes, "test_xray_001")
            print(f"Prediction: {result.top_class}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Risk Level: {result.risk_level}")
        except Exception as e:
            print(f"Prediction failed: {e}")
    else:
        print("Failed to load model!")
