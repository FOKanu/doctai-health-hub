"""
MRI Classifier Model
Detects brain tumors and abnormalities in MRI scans.
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


class MRIClassifier(nn.Module):
    """
    DenseNet-121 based MRI classifier for brain tumor detection.
    """

    def __init__(self, num_classes=4):
        super(MRIClassifier, self).__init__()
        # Mock architecture - in reality would use DenseNet-121
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


class MRIModelManager:
    """Manages MRI classification model loading and prediction."""

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = ["Normal", "Glioma", "Meningioma", "Pituitary"]
        self.model_name = "mri_classifier_densenet121"
        self.model_version = "1.0.0"

        # Image preprocessing pipeline for MRI scans
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path: str = None) -> bool:
        """Load the MRI classification model."""
        try:
            self.model = MRIClassifier(num_classes=len(self.class_labels))
            self.model.to(self.device)
            self.model.eval()

            print(f"✅ MRI model loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"❌ Error loading MRI model: {e}")
            return False

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess MRI image bytes for model input."""
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    def predict(self, image_bytes: bytes, image_id: str) -> PredictionResult:
        """Predict MRI classification."""
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
            risk_level = self._assess_mri_risk(top_class, confidence)

            # Generate recommendations
            recommendations = self._generate_mri_recommendations(top_class, risk_level)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Create standardized result
            result = create_prediction_result(
                image_id=image_id,
                image_type=ImageType.MRI,
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
                    "preprocessing_applied": ["resize", "center_crop", "normalize"]
                }
            )

            return result

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            return create_prediction_result(
                image_id=image_id,
                image_type=ImageType.MRI,
                model_name=self.model_name,
                model_version=self.model_version,
                top_class="Error",
                top_class_index=-1,
                confidence=0.0,
                class_probabilities={"Error": 1.0},
                class_labels=["Error"],
                risk_level=RiskLevel.HIGH,
                recommendations=["Unable to process MRI scan. Please consult a medical professional."],
                processing_time_ms=processing_time_ms,
                additional_info={"error": str(e)}
            )

    def _assess_mri_risk(self, top_class: str, confidence: float) -> RiskLevel:
        """Assess risk level for MRI predictions."""
        if top_class == "Normal":
            return RiskLevel.LOW
        elif top_class in ["Glioma", "Meningioma", "Pituitary"]:
            if confidence > 0.85:
                return RiskLevel.HIGH
            else:
                return RiskLevel.MEDIUM
        else:
            return RiskLevel.MEDIUM

    def _generate_mri_recommendations(self, top_class: str, risk_level: RiskLevel) -> list:
        """Generate MRI specific recommendations."""
        base_recommendations = generate_recommendations(top_class, risk_level, ImageType.MRI)

        # Add MRI-specific recommendations
        if top_class == "Normal":
            base_recommendations.extend([
                "No brain abnormalities detected",
                "Continue routine monitoring"
            ])
        elif top_class == "Glioma":
            base_recommendations.extend([
                "Glioma detected - requires immediate attention",
                "Urgent neurosurgical consultation",
                "Consider biopsy for grading",
                "Multidisciplinary team approach recommended"
            ])
        elif top_class == "Meningioma":
            base_recommendations.extend([
                "Meningioma detected",
                "Neurosurgical evaluation required",
                "Monitor growth with serial imaging",
                "Treatment depends on size and location"
            ])
        elif top_class == "Pituitary":
            base_recommendations.extend([
                "Pituitary abnormality detected",
                "Endocrinology consultation recommended",
                "Hormone level testing required",
                "Consider pituitary-specific MRI protocol"
            ])

        return base_recommendations


# Global model manager instance
_model_manager = None


def load_model(model_path: str = None) -> bool:
    """Load the MRI classification model."""
    global _model_manager
    _model_manager = MRIModelManager()
    return _model_manager.load_model(model_path)


def predict(image_bytes: bytes, image_id: str) -> PredictionResult:
    """Predict MRI classification."""
    global _model_manager
    if _model_manager is None:
        raise ValueError("Model not loaded. Call load_model() first.")

    return _model_manager.predict(image_bytes, image_id)


if __name__ == "__main__":
    print("Testing MRI classifier...")

    success = load_model()
    if success:
        print("Model loaded successfully!")

        mock_image_bytes = b"mock_mri_data"

        try:
            result = predict(mock_image_bytes, "test_mri_001")
            print(f"Prediction: {result.top_class}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Risk Level: {result.risk_level}")
        except Exception as e:
            print(f"Prediction failed: {e}")
    else:
        print("Failed to load model!")
