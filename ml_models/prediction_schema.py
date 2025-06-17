"""
Unified prediction schema for all medical image classifiers.
This ensures consistency across all model outputs.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
import uuid
from datetime import datetime


class ImageType(str, Enum):
    """Supported medical image types"""
    XRAY = "xray"
    CT_SCAN = "ct_scan"
    MRI = "mri"
    EEG = "eeg"
    SKIN_LESION = "skin_lesion"


class RiskLevel(str, Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionResult(BaseModel):
    """
    Unified prediction result schema for all medical image classifiers.
    All models must output predictions in this format.
    """

    # Unique identifier for this prediction
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Image metadata
    image_id: str = Field(..., description="Unique identifier for the analyzed image")
    image_type: ImageType = Field(..., description="Type of medical image analyzed")

    # Model information
    model_name: str = Field(..., description="Name/version of the model used")
    model_version: Optional[str] = Field(None, description="Version of the model")

    # Prediction results
    top_class: str = Field(..., description="The predicted class with highest probability")
    top_class_index: int = Field(..., description="Index of the top predicted class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for top prediction")

    # Detailed probabilities
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability scores for each class"
    )
    class_labels: List[str] = Field(..., description="List of all possible class labels")

    # Risk assessment
    risk_level: RiskLevel = Field(..., description="Overall risk assessment")

    # Medical recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Medical recommendations based on the prediction"
    )

    # Additional metadata
    processing_time_ms: Optional[float] = Field(None, description="Time taken for prediction in milliseconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Optional additional data
    additional_info: Optional[Dict] = Field(default_factory=dict, description="Any additional model-specific information")

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "image_id": "img_12345",
                "image_type": "xray",
                "model_name": "xray_classifier_v1",
                "model_version": "1.0.0",
                "top_class": "Normal",
                "top_class_index": 0,
                "confidence": 0.92,
                "class_probabilities": {
                    "Normal": 0.92,
                    "Pneumonia": 0.06,
                    "COVID-19": 0.02
                },
                "class_labels": ["Normal", "Pneumonia", "COVID-19"],
                "risk_level": "low",
                "recommendations": [
                    "Continue routine monitoring",
                    "Maintain healthy lifestyle"
                ],
                "processing_time_ms": 245.7,
                "created_at": "2024-01-16T10:30:00Z",
                "additional_info": {
                    "image_quality": "good",
                    "preprocessing_applied": ["resize", "normalize"]
                }
            }
        }


def create_prediction_result(
    image_id: str,
    image_type: ImageType,
    model_name: str,
    top_class: str,
    top_class_index: int,
    confidence: float,
    class_probabilities: Dict[str, float],
    class_labels: List[str],
    risk_level: RiskLevel,
    recommendations: List[str] = None,
    model_version: str = None,
    processing_time_ms: float = None,
    additional_info: Dict = None
) -> PredictionResult:
    """
    Helper function to create a standardized PredictionResult.

    Args:
        image_id: Unique identifier for the analyzed image
        image_type: Type of medical image
        model_name: Name of the model used
        top_class: Predicted class with highest probability
        top_class_index: Index of the top predicted class
        confidence: Confidence score for top prediction
        class_probabilities: Dictionary of class probabilities
        class_labels: List of all possible class labels
        risk_level: Risk assessment level
        recommendations: List of medical recommendations
        model_version: Version of the model
        processing_time_ms: Processing time in milliseconds
        additional_info: Additional model-specific information

    Returns:
        PredictionResult: Standardized prediction result
    """
    return PredictionResult(
        image_id=image_id,
        image_type=image_type,
        model_name=model_name,
        model_version=model_version,
        top_class=top_class,
        top_class_index=top_class_index,
        confidence=confidence,
        class_probabilities=class_probabilities,
        class_labels=class_labels,
        risk_level=risk_level,
        recommendations=recommendations or [],
        processing_time_ms=processing_time_ms,
        additional_info=additional_info or {}
    )


# Risk assessment helper functions
def assess_risk_level(top_class: str, confidence: float, image_type: ImageType) -> RiskLevel:
    """
    Assess risk level based on prediction results.
    This is a general function that can be overridden by specific models.
    """
    # General risk assessment logic
    if top_class.lower() in ['normal', 'benign', 'healthy']:
        return RiskLevel.LOW
    elif confidence > 0.8:
        return RiskLevel.HIGH
    else:
        return RiskLevel.MEDIUM


def generate_recommendations(
    top_class: str,
    risk_level: RiskLevel,
    image_type: ImageType
) -> List[str]:
    """
    Generate medical recommendations based on prediction results.
    This provides default recommendations that can be customized by specific models.
    """
    base_recommendations = []

    if risk_level == RiskLevel.LOW:
        base_recommendations = [
            "Continue routine monitoring",
            "Maintain healthy lifestyle",
            "Follow up as scheduled"
        ]
    elif risk_level == RiskLevel.MEDIUM:
        base_recommendations = [
            "Consult with a medical professional",
            "Monitor for changes",
            "Follow up in 2-4 weeks"
        ]
    else:  # HIGH risk
        base_recommendations = [
            "Seek immediate medical attention",
            "Consult with a specialist",
            "Additional testing may be required"
        ]

    # Add image-type specific recommendations
    if image_type == ImageType.XRAY and top_class.lower() in ['pneumonia', 'covid-19']:
        base_recommendations.append("Monitor oxygen levels")
        base_recommendations.append("Consider isolation if infectious")
    elif image_type == ImageType.SKIN_LESION and risk_level == RiskLevel.HIGH:
        base_recommendations.append("Dermatology consultation recommended")
        base_recommendations.append("Avoid sun exposure")

    return base_recommendations
