"""
Skin Lesion Classifier Model
Detects melanoma and other skin lesions.
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


class SkinLesionClassifier(nn.Module):
    """
    EfficientNet-B0 based skin lesion classifier for melanoma detection.
    """

    def __init__(self, num_classes=7):
        super(SkinLesionClassifier, self).__init__()
        # Mock architecture - in reality would use EfficientNet-B0
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


class SkinLesionModelManager:
    """Manages skin lesion classification model loading and prediction."""

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = [
            "Melanoma", "Melanocytic nevus", "Basal cell carcinoma",
            "Actinic keratosis", "Benign keratosis", "Dermatofibroma", "Vascular lesion"
        ]
        self.model_name = "skin_lesion_classifier_efficientnet_b0"
        self.model_version = "1.0.0"

        # Image preprocessing pipeline for skin lesions
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path: str = None) -> bool:
        """Load the skin lesion classification model."""
        try:
            self.model = SkinLesionClassifier(num_classes=len(self.class_labels))
            self.model.to(self.device)
            self.model.eval()

            print(f"✅ Skin lesion model loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"❌ Error loading skin lesion model: {e}")
            return False

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess skin lesion image bytes for model input."""
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    def predict(self, image_bytes: bytes, image_id: str) -> PredictionResult:
        """Predict skin lesion classification."""
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
            risk_level = self._assess_skin_risk(top_class, confidence)

            # Generate recommendations
            recommendations = self._generate_skin_recommendations(top_class, risk_level)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Create standardized result
            result = create_prediction_result(
                image_id=image_id,
                image_type=ImageType.SKIN_LESION,
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
                image_type=ImageType.SKIN_LESION,
                model_name=self.model_name,
                model_version=self.model_version,
                top_class="Error",
                top_class_index=-1,
                confidence=0.0,
                class_probabilities={"Error": 1.0},
                class_labels=["Error"],
                risk_level=RiskLevel.HIGH,
                recommendations=["Unable to process skin lesion image. Please consult a dermatologist."],
                processing_time_ms=processing_time_ms,
                additional_info={"error": str(e)}
            )

    def _assess_skin_risk(self, top_class: str, confidence: float) -> RiskLevel:
        """Assess risk level for skin lesion predictions."""
        high_risk_classes = ["Melanoma", "Basal cell carcinoma"]
        medium_risk_classes = ["Actinic keratosis"]

        if top_class in high_risk_classes:
            if confidence > 0.7:
                return RiskLevel.HIGH
            else:
                return RiskLevel.MEDIUM
        elif top_class in medium_risk_classes:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _generate_skin_recommendations(self, top_class: str, risk_level: RiskLevel) -> list:
        """Generate skin lesion specific recommendations."""
        base_recommendations = generate_recommendations(top_class, risk_level, ImageType.SKIN_LESION)

        # Add skin-specific recommendations
        if top_class == "Melanoma":
            base_recommendations.extend([
                "Possible melanoma detected - urgent dermatology referral",
                "Biopsy required for confirmation",
                "Avoid sun exposure",
                "Monitor for changes in other moles"
            ])
        elif top_class == "Basal cell carcinoma":
            base_recommendations.extend([
                "Possible basal cell carcinoma",
                "Dermatology consultation recommended",
                "Early treatment is highly effective",
                "Use sun protection"
            ])
        elif top_class == "Actinic keratosis":
            base_recommendations.extend([
                "Actinic keratosis detected",
                "Dermatology evaluation recommended",
                "Potential for malignant transformation",
                "Strict sun protection advised"
            ])
        elif top_class in ["Melanocytic nevus", "Benign keratosis", "Dermatofibroma"]:
            base_recommendations.extend([
                "Benign lesion detected",
                "Monitor for changes",
                "Routine dermatology check-up recommended"
            ])
        elif top_class == "Vascular lesion":
            base_recommendations.extend([
                "Vascular lesion detected",
                "Usually benign",
                "Consult dermatologist if concerned"
            ])

        return base_recommendations


# Global model manager instance
_model_manager = None


def load_model(model_path: str = None) -> bool:
    """Load the skin lesion classification model."""
    global _model_manager
    _model_manager = SkinLesionModelManager()
    return _model_manager.load_model(model_path)


def predict(image_bytes: bytes, image_id: str) -> PredictionResult:
    """Predict skin lesion classification."""
    global _model_manager
    if _model_manager is None:
        raise ValueError("Model not loaded. Call load_model() first.")

    return _model_manager.predict(image_bytes, image_id)


if __name__ == "__main__":
    print("Testing skin lesion classifier...")

    success = load_model()
    if success:
        print("Model loaded successfully!")

        mock_image_bytes = b"mock_skin_lesion_data"

        try:
            result = predict(mock_image_bytes, "test_skin_001")
            print(f"Prediction: {result.top_class}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Risk Level: {result.risk_level}")
        except Exception as e:
            print(f"Prediction failed: {e}")
    else:
        print("Failed to load model!")
