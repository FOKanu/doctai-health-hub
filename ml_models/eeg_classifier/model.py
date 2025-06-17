"""
EEG Classifier Model
Detects seizures and abnormalities in EEG signals.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import time
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


class EEGClassifier(nn.Module):
    """
    CNN-LSTM based EEG classifier for seizure detection.
    """

    def __init__(self, num_classes=3):
        super(EEGClassifier, self).__init__()
        # Mock architecture for EEG spectrograms
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class EEGModelManager:
    """Manages EEG classification model loading and prediction."""

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = ["Normal", "Seizure", "Abnormal"]
        self.model_name = "eeg_classifier_cnn_lstm"
        self.model_version = "1.0.0"

        # Image preprocessing pipeline for EEG spectrograms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path: str = None) -> bool:
        """Load the EEG classification model."""
        try:
            self.model = EEGClassifier(num_classes=len(self.class_labels))
            self.model.to(self.device)
            self.model.eval()

            print(f"✅ EEG model loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"❌ Error loading EEG model: {e}")
            return False

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess EEG spectrogram image bytes for model input."""
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    def predict(self, image_bytes: bytes, image_id: str) -> PredictionResult:
        """Predict EEG classification."""
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
            risk_level = self._assess_eeg_risk(top_class, confidence)

            # Generate recommendations
            recommendations = self._generate_eeg_recommendations(top_class, risk_level)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Create standardized result
            result = create_prediction_result(
                image_id=image_id,
                image_type=ImageType.EEG,
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
                    "preprocessing_applied": ["resize", "normalize"]
                }
            )

            return result

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            return create_prediction_result(
                image_id=image_id,
                image_type=ImageType.EEG,
                model_name=self.model_name,
                model_version=self.model_version,
                top_class="Error",
                top_class_index=-1,
                confidence=0.0,
                class_probabilities={"Error": 1.0},
                class_labels=["Error"],
                risk_level=RiskLevel.HIGH,
                recommendations=["Unable to process EEG data. Please consult a medical professional."],
                processing_time_ms=processing_time_ms,
                additional_info={"error": str(e)}
            )

    def _assess_eeg_risk(self, top_class: str, confidence: float) -> RiskLevel:
        """Assess risk level for EEG predictions."""
        if top_class == "Normal":
            return RiskLevel.LOW
        elif top_class == "Seizure":
            if confidence > 0.8:
                return RiskLevel.HIGH
            else:
                return RiskLevel.MEDIUM
        elif top_class == "Abnormal":
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.MEDIUM

    def _generate_eeg_recommendations(self, top_class: str, risk_level: RiskLevel) -> list:
        """Generate EEG specific recommendations."""
        base_recommendations = generate_recommendations(top_class, risk_level, ImageType.EEG)

        # Add EEG-specific recommendations
        if top_class == "Normal":
            base_recommendations.extend([
                "Normal EEG pattern detected",
                "Continue routine monitoring if indicated"
            ])
        elif top_class == "Seizure":
            base_recommendations.extend([
                "Seizure activity detected",
                "Immediate neurological evaluation required",
                "Consider anti-seizure medication",
                "Monitor for additional seizure activity"
            ])
        elif top_class == "Abnormal":
            base_recommendations.extend([
                "Abnormal EEG pattern detected",
                "Neurological consultation recommended",
                "Consider follow-up EEG monitoring",
                "Correlate with clinical symptoms"
            ])

        return base_recommendations


# Global model manager instance
_model_manager = None


def load_model(model_path: str = None) -> bool:
    """Load the EEG classification model."""
    global _model_manager
    _model_manager = EEGModelManager()
    return _model_manager.load_model(model_path)


def predict(image_bytes: bytes, image_id: str) -> PredictionResult:
    """Predict EEG classification."""
    global _model_manager
    if _model_manager is None:
        raise ValueError("Model not loaded. Call load_model() first.")

    return _model_manager.predict(image_bytes, image_id)


if __name__ == "__main__":
    print("Testing EEG classifier...")

    success = load_model()
    if success:
        print("Model loaded successfully!")

        mock_image_bytes = b"mock_eeg_data"

        try:
            result = predict(mock_image_bytes, "test_eeg_001")
            print(f"Prediction: {result.top_class}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Risk Level: {result.risk_level}")
        except Exception as e:
            print(f"Prediction failed: {e}")
    else:
        print("Failed to load model!")
