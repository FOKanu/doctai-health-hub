"""
Transformer-based Vital Signs Analyzer
Analyzes continuous health metrics for trend detection and anomaly identification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VitalSignsTransformer(nn.Module):
    """
    Transformer model for analyzing vital signs time series data
    Detects trends, anomalies, and predicts future values
    """

    def __init__(
        self,
        input_dim: int = 8,  # Number of vital signs
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_length: int = 1000,
        num_classes: int = 3  # normal, warning, critical
    ):
        super(VitalSignsTransformer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_length, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.trend_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # improving, stable, declining
        )

        self.anomaly_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self.value_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_dim)
        )

        # Health score regressor
        self.health_score_regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the vital signs transformer

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Attention mask for padding [batch_size, seq_len]

        Returns:
            Dictionary containing all model outputs
        """
        batch_size, seq_len, _ = x.shape

        # Project input to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)

        # Create attention mask if not provided
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)

        # Apply transformer
        transformer_out = self.transformer(x, src_key_padding_mask=~mask)

        # Global average pooling (masked)
        masked_out = transformer_out * mask.unsqueeze(-1).float()
        pooled = masked_out.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()

        # Output predictions
        trend_logits = self.trend_classifier(pooled)
        anomaly_logits = self.anomaly_detector(pooled)
        predicted_values = self.value_predictor(pooled)
        health_score = self.health_score_regressor(pooled)

        return {
            'trend_logits': trend_logits,
            'trend_probs': F.softmax(trend_logits, dim=1),
            'anomaly_logits': anomaly_logits,
            'anomaly_probs': F.softmax(anomaly_logits, dim=1),
            'predicted_values': predicted_values,
            'health_score': health_score,
            'hidden_states': transformer_out
        }


class VitalSignsDataset(Dataset):
    """
    Dataset for vital signs time series data
    """

    def __init__(
        self,
        sequences: List[Dict],
        max_sequence_length: int = 1000,
        input_dim: int = 8
    ):
        self.sequences = sequences
        self.max_sequence_length = max_sequence_length
        self.input_dim = input_dim

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # Extract vital signs data
        vital_signs = sequence['vital_signs']  # [seq_len, input_dim]
        labels = sequence.get('labels', None)  # Optional labels
        timestamps = sequence.get('timestamps', None)

        seq_len = len(vital_signs)
        if seq_len > self.max_sequence_length:
            vital_signs = vital_signs[:self.max_sequence_length]
            if labels is not None:
                labels = labels[:self.max_sequence_length]
            if timestamps is not None:
                timestamps = timestamps[:self.max_sequence_length]
            seq_len = self.max_sequence_length

        # Pad sequences
        padded_vitals = np.zeros((self.max_sequence_length, self.input_dim))
        padded_vitals[:seq_len] = np.array(vital_signs)

        # Create mask
        mask = np.zeros(self.max_sequence_length, dtype=np.bool_)
        mask[:seq_len] = True

        if labels is not None:
            padded_labels = np.zeros(self.max_sequence_length, dtype=np.int64)
            padded_labels[:seq_len] = np.array(labels)
        else:
            padded_labels = None

        return {
            'vital_signs': torch.FloatTensor(padded_vitals),
            'mask': torch.BoolTensor(mask),
            'labels': torch.LongTensor(padded_labels) if padded_labels is not None else None,
            'length': seq_len
        }


class VitalSignsAnalyzer:
    """
    Main class for vital signs analysis functionality
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model = None
        self.trend_names = ['improving', 'stable', 'declining']
        self.anomaly_names = ['normal', 'warning', 'critical']
        self.vital_sign_names = [
            'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'temperature', 'oxygen_saturation', 'respiratory_rate',
            'blood_glucose', 'weight'
        ]

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning(f"Model not found at {model_path}, initializing new model")
            self.model = VitalSignsTransformer().to(device)

    def load_model(self, model_path: str):
        """Load a trained vital signs analyzer model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = VitalSignsTransformer().to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info(f"Loaded vital signs analyzer model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def save_model(self, model_path: str, optimizer_state: Optional[Dict] = None):
        """Save the trained model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'trend_names': self.trend_names,
            'anomaly_names': self.anomaly_names,
            'vital_sign_names': self.vital_sign_names,
            'timestamp': datetime.now().isoformat()
        }
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state

        torch.save(checkpoint, model_path)
        logger.info(f"Saved vital signs analyzer model to {model_path}")

    def analyze_vital_signs(
        self,
        vital_signs_data: List[List[float]],
        timestamps: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze vital signs time series data

        Args:
            vital_signs_data: List of vital signs measurements [seq_len, num_vitals]
            timestamps: Optional timestamps for each measurement

        Returns:
            Dictionary with analysis results
        """
        if not self.model:
            raise ValueError("Model not loaded")

        self.model.eval()

        # Prepare input
        vital_signs = torch.FloatTensor(vital_signs_data).unsqueeze(0).to(self.device)
        mask = torch.BoolTensor([[True] * len(vital_signs_data)]).to(self.device)

        with torch.no_grad():
            outputs = self.model(vital_signs, mask)

        # Process outputs
        trend_probs = outputs['trend_probs'].cpu().numpy()[0]
        anomaly_probs = outputs['anomaly_probs'].cpu().numpy()[0]
        predicted_values = outputs['predicted_values'].cpu().numpy()[0]
        health_score = outputs['health_score'].cpu().numpy()[0]

        predicted_trend = self.trend_names[np.argmax(trend_probs)]
        predicted_anomaly = self.anomaly_names[np.argmax(anomaly_probs)]

        return {
            'trend': {
                'predicted': predicted_trend,
                'probabilities': {
                    name: float(prob) for name, prob in zip(self.trend_names, trend_probs)
                }
            },
            'anomaly': {
                'predicted': predicted_anomaly,
                'probabilities': {
                    name: float(prob) for name, prob in zip(self.anomaly_names, anomaly_probs)
                }
            },
            'predicted_values': {
                name: float(value) for name, value in zip(self.vital_sign_names, predicted_values)
            },
            'health_score': float(health_score),
            'confidence': float(max(trend_probs)),
            'timestamps': timestamps
        }

    def detect_anomalies(
        self,
        vital_signs_data: List[List[float]],
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Detect anomalies in vital signs data

        Args:
            vital_signs_data: Vital signs measurements
            threshold: Confidence threshold for anomaly detection

        Returns:
            List of detected anomalies
        """
        analysis = self.analyze_vital_signs(vital_signs_data)
        anomalies = []

        # Check for critical anomalies
        if analysis['anomaly']['probabilities']['critical'] > threshold:
            anomalies.append({
                'type': 'critical',
                'confidence': analysis['anomaly']['probabilities']['critical'],
                'description': 'Critical vital signs detected',
                'timestamp': datetime.now().isoformat()
            })

        # Check for warnings
        elif analysis['anomaly']['probabilities']['warning'] > threshold:
            anomalies.append({
                'type': 'warning',
                'confidence': analysis['anomaly']['probabilities']['warning'],
                'description': 'Warning signs detected',
                'timestamp': datetime.now().isoformat()
            })

        return anomalies

    def predict_trends(
        self,
        vital_signs_data: List[List[float]],
        forecast_steps: int = 7
    ) -> Dict:
        """
        Predict future trends in vital signs

        Args:
            vital_signs_data: Historical vital signs data
            forecast_steps: Number of steps to forecast

        Returns:
            Dictionary with trend predictions
        """
        analysis = self.analyze_vital_signs(vital_signs_data)

        # Simple trend prediction based on current analysis
        # In a real implementation, this would use the transformer to generate future values
        current_trend = analysis['trend']['predicted']

        if current_trend == 'improving':
            trend_prediction = 'continued_improvement'
        elif current_trend == 'declining':
            trend_prediction = 'continued_decline'
        else:
            trend_prediction = 'stable'

        return {
            'current_trend': current_trend,
            'predicted_trend': trend_prediction,
            'forecast_steps': forecast_steps,
            'confidence': analysis['confidence'],
            'health_score_projection': analysis['health_score']
        }

    def train(
        self,
        train_sequences: List[Dict],
        val_sequences: List[Dict],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        save_path: str = 'vital_signs_analyzer_model.pth'
    ):
        """Train the vital signs analyzer model"""
        logger.info("Starting vital signs analyzer training...")

        # Create datasets and dataloaders
        train_dataset = VitalSignsDataset(train_sequences)
        val_dataset = VitalSignsDataset(val_sequences)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        # Loss functions
        trend_loss = nn.CrossEntropyLoss()
        anomaly_loss = nn.CrossEntropyLoss()
        prediction_loss = nn.MSELoss()
        health_loss = nn.MSELoss()

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                vital_signs = batch['vital_signs'].to(self.device)
                mask = batch['mask'].to(self.device)
                labels = batch['labels']

                optimizer.zero_grad()

                outputs = self.model(vital_signs, mask)

                # Calculate losses (using mock labels for now)
                batch_size = vital_signs.shape[0]
                mock_trend_labels = torch.randint(0, 3, (batch_size,)).to(self.device)
                mock_anomaly_labels = torch.randint(0, 3, (batch_size,)).to(self.device)
                mock_target_values = vital_signs[:, -1, :]  # Use last timestep as target
                mock_health_scores = torch.rand(batch_size, 1).to(self.device)

                loss = (
                    trend_loss(outputs['trend_logits'], mock_trend_labels) +
                    anomaly_loss(outputs['anomaly_logits'], mock_anomaly_labels) +
                    prediction_loss(outputs['predicted_values'], mock_target_values) +
                    health_loss(outputs['health_score'], mock_health_scores)
                )

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    vital_signs = batch['vital_signs'].to(self.device)
                    mask = batch['mask'].to(self.device)

                    outputs = self.model(vital_signs, mask)

                    # Calculate validation loss
                    batch_size = vital_signs.shape[0]
                    mock_trend_labels = torch.randint(0, 3, (batch_size,)).to(self.device)
                    mock_anomaly_labels = torch.randint(0, 3, (batch_size,)).to(self.device)
                    mock_target_values = vital_signs[:, -1, :]
                    mock_health_scores = torch.rand(batch_size, 1).to(self.device)

                    loss = (
                        trend_loss(outputs['trend_logits'], mock_trend_labels) +
                        anomaly_loss(outputs['anomaly_logits'], mock_anomaly_labels) +
                        prediction_loss(outputs['predicted_values'], mock_target_values) +
                        health_loss(outputs['health_score'], mock_health_scores)
                    )

                    val_loss += loss.item()

            # Update learning rate
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(save_path, optimizer.state_dict())

            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

        logger.info("Training completed!")


def create_mock_vital_signs_data(num_sequences: int = 100) -> List[Dict]:
    """Create mock vital signs data for testing"""
    sequences = []

    for i in range(num_sequences):
        seq_len = np.random.randint(50, 200)
        vital_signs = []
        timestamps = []

        # Generate realistic vital signs data
        base_heart_rate = np.random.normal(70, 10)
        base_temp = np.random.normal(98.6, 0.5)

        for t in range(seq_len):
            # Add some temporal correlation and noise
            heart_rate = base_heart_rate + np.random.normal(0, 5) + np.sin(t * 0.1) * 3
            temp = base_temp + np.random.normal(0, 0.3)
            bp_systolic = np.random.normal(120, 15)
            bp_diastolic = np.random.normal(80, 10)
            oxygen = np.random.normal(98, 1)
            respiratory = np.random.normal(16, 2)
            glucose = np.random.normal(100, 15)
            weight = np.random.normal(70, 5)

            vital_signs.append([
                heart_rate, bp_systolic, bp_diastolic, temp,
                oxygen, respiratory, glucose, weight
            ])

            timestamps.append((datetime.now() - timedelta(days=seq_len-t)).isoformat())

        sequences.append({
            'vital_signs': vital_signs,
            'timestamps': timestamps,
            'labels': np.random.randint(0, 3, seq_len)  # Mock labels
        })

    return sequences


if __name__ == "__main__":
    # Example usage
    analyzer = VitalSignsAnalyzer()

    # Create mock data
    train_data = create_mock_vital_signs_data(80)
    val_data = create_mock_vital_signs_data(20)

    # Train model
    analyzer.train(train_data, val_data, epochs=5, batch_size=16)

    # Test analysis
    test_data = create_mock_vital_signs_data(1)[0]['vital_signs']
    result = analyzer.analyze_vital_signs(test_data)
    print("Vital signs analysis:", result)

    # Test anomaly detection
    anomalies = analyzer.detect_anomalies(test_data)
    print("Detected anomalies:", anomalies)
