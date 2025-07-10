"""
Composite Health Score Aggregator
Aggregates multiple health metrics into a single health score (0-100).
"""

from typing import Optional, Dict

def normalize(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize a value to a 0-100 scale.
    """
    if max_value == min_value:
        return 0.0
    return max(0.0, min(100.0, 100 * (value - min_value) / (max_value - min_value)))

def aggregate_health_score(
    skin_health: Optional[float] = None,  # 0-100
    fitness: Optional[float] = None,      # 0-100
    diet: Optional[float] = None,         # 0-100
    medication_adherence: Optional[float] = None,  # 0-100
    appointment_adherence: Optional[float] = None, # 0-100
    custom_metrics: Optional[Dict[str, float]] = None, # 0-100
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Aggregate multiple health metrics into a composite health score (0-100).
    All inputs should be normalized to 0-100. Missing metrics are ignored.
    Weights should sum to 1.0. If not provided, defaults are used.
    """
    # Default weights (should sum to 1.0)
    default_weights = {
        'skin_health': 0.2,
        'fitness': 0.2,
        'diet': 0.2,
        'medication_adherence': 0.2,
        'appointment_adherence': 0.2
    }
    if weights:
        default_weights.update(weights)

    # Collect available metrics
    metrics = {
        'skin_health': skin_health,
        'fitness': fitness,
        'diet': diet,
        'medication_adherence': medication_adherence,
        'appointment_adherence': appointment_adherence
    }
    if custom_metrics:
        metrics.update(custom_metrics)
        for k in custom_metrics:
            if k not in default_weights:
                default_weights[k] = 0.0  # Add custom metrics with zero weight unless specified

    # Filter out missing metrics
    valid_metrics = {k: v for k, v in metrics.items() if v is not None}
    if not valid_metrics:
        return 0.0

    # Normalize weights for present metrics
    present_weights = {k: default_weights.get(k, 0.0) for k in valid_metrics}
    total_weight = sum(present_weights.values())
    if total_weight == 0:
        # If all weights are zero, assign equal weight
        n = len(valid_metrics)
        present_weights = {k: 1.0 / n for k in valid_metrics}
        total_weight = 1.0
    else:
        present_weights = {k: w / total_weight for k, w in present_weights.items()}

    # Weighted sum
    score = sum(valid_metrics[k] * present_weights[k] for k in valid_metrics)
    return round(score, 2)

# Example usage (for testing):
if __name__ == "__main__":
    example = aggregate_health_score(
        skin_health=90,
        fitness=80,
        diet=70,
        medication_adherence=95,
        appointment_adherence=85
    )
    print(f"Example health score: {example}")
