"""
LSTM-based Progression Tracker for Medical Image Sequences
Analyzes temporal changes in medical images to track disease progression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressionLSTM(nn.Module):
    """
    LSTM-based model for tracking medical image progression over time
    Combines CNN features with LSTM for temporal analysis
    """

    def __init__(
        self,
        input_size: int = 2048,  # CNN feature size
        hidden_size: int = 512,
        num_layers: int = 2,
        num_classes: int = 3,  # improving, stable, worsening
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(ProgressionLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer for temporal analysis
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Attention mechanism for focusing on important time steps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * self.num_directions,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # Progression score regression
        self.progression_regressor = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

        # Time interval embedding
        self.time_embedding = nn.Embedding(100, 64)  # Support up to 100 time steps

    def forward(
        self,
        features: torch.Tensor,
        time_intervals: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the progression tracker

        Args:
            features: CNN features [batch_size, seq_len, input_size]
            time_intervals: Time intervals between images [batch_size, seq_len]
            lengths: Actual sequence lengths for padding [batch_size]

        Returns:
            Dictionary containing classification and regression outputs
        """
        batch_size, seq_len, _ = features.shape

        # Pack padded sequences if lengths are provided
        if lengths is not None:
            packed_features = nn.utils.rnn.pack_padded_sequence(
                features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed_features)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(features)

        # Apply attention mechanism
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )

        # Global average pooling over time dimension
        if lengths is not None:
            # Mask padded positions
            mask = torch.arange(seq_len, device=features.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).expand_as(attn_out)
            attn_out = attn_out * mask.float()
            pooled = attn_out.sum(dim=1) / lengths.unsqueeze(1).float()
        else:
            pooled = attn_out.mean(dim=1)

        # Classification output
        class_logits = self.classifier(pooled)
        class_probs = F.softmax(class_logits, dim=1)

        # Progression score
        progression_score = self.progression_regressor(pooled).squeeze(-1)

        return {
            'class_logits': class_logits,
            'class_probs': class_probs,
            'progression_score': progression_score,
            'attention_weights': attn_weights,
            'hidden_states': lstm_out
        }


class ProgressionDataset(Dataset):
    """
    Dataset for medical image progression sequences
    """

    def __init__(
        self,
        sequences: List[Dict],
        max_sequence_length: int = 10,
        feature_dim: int = 2048
    ):
        self.sequences = sequences
        self.max_sequence_length = max_sequence_length
        self.feature_dim = feature_dim

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # Extract features and labels
        features = sequence['features']  # List of CNN features
        labels = sequence['labels']  # List of progression labels
        time_intervals = sequence.get('time_intervals', None)

        # Pad sequences to max length
        seq_len = len(features)
        if seq_len > self.max_sequence_length:
            features = features[:self.max_sequence_length]
            labels = labels[:self.max_sequence_length]
            if time_intervals:
                time_intervals = time_intervals[:self.max_sequence_length]
            seq_len = self.max_sequence_length

        # Pad with zeros
        padded_features = np.zeros((self.max_sequence_length, self.feature_dim))
        padded_features[:seq_len] = np.array(features)

        padded_labels = np.zeros(self.max_sequence_length, dtype=np.int64)
        padded_labels[:seq_len] = np.array(labels)

        if time_intervals:
            padded_intervals = np.zeros(self.max_sequence_length, dtype=np.int64)
            padded_intervals[:seq_len] = np.array(time_intervals)
        else:
            padded_intervals = np.arange(self.max_sequence_length)

        return {
            'features': torch.FloatTensor(padded_features),
            'labels': torch.LongTensor(padded_labels),
            'time_intervals': torch.LongTensor(padded_intervals),
            'length': seq_len
        }


class ProgressionTracker:
    """
    Main class for progression tracking functionality
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model = None
        self.class_names = ['improving', 'stable', 'worsening']

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning(f"Model not found at {model_path}, initializing new model")
            self.model = ProgressionLSTM().to(device)

    def load_model(self, model_path: str):
        """Load a trained progression tracker model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = ProgressionLSTM().to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info(f"Loaded progression tracker model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def save_model(self, model_path: str, optimizer_state: Optional[Dict] = None):
        """Save the trained model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'timestamp': datetime.now().isoformat()
        }
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state

        torch.save(checkpoint, model_path)
        logger.info(f"Saved progression tracker model to {model_path}")

    def predict_progression(
        self,
        image_features: List[np.ndarray],
        time_intervals: Optional[List[int]] = None
    ) -> Dict:
        """
        Predict progression from a sequence of image features

        Args:
            image_features: List of CNN features from images
            time_intervals: Days between consecutive images

        Returns:
            Dictionary with progression prediction results
        """
        if not self.model:
            raise ValueError("Model not loaded")

        self.model.eval()

        # Prepare input
        features = torch.FloatTensor(image_features).unsqueeze(0).to(self.device)
        lengths = torch.LongTensor([len(image_features)]).to(self.device)

        if time_intervals:
            intervals = torch.LongTensor(time_intervals).unsqueeze(0).to(self.device)
        else:
            intervals = None

        with torch.no_grad():
            outputs = self.model(features, intervals, lengths)

        # Process outputs
        class_probs = outputs['class_probs'].cpu().numpy()[0]
        progression_score = outputs['progression_score'].cpu().numpy()[0]
        predicted_class = np.argmax(class_probs)

        return {
            'predicted_class': self.class_names[predicted_class],
            'class_probabilities': {
                name: float(prob) for name, prob in zip(self.class_names, class_probs)
            },
            'progression_score': float(progression_score),
            'confidence': float(class_probs[predicted_class]),
            'attention_weights': outputs['attention_weights'].cpu().numpy() if outputs['attention_weights'] is not None else None
        }

    def train(
        self,
        train_sequences: List[Dict],
        val_sequences: List[Dict],
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        save_path: str = 'progression_tracker_model.pth'
    ):
        """Train the progression tracker model"""
        logger.info("Starting progression tracker training...")

        # Create datasets and dataloaders
        train_dataset = ProgressionDataset(train_sequences)
        val_dataset = ProgressionDataset(val_sequences)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        # Loss functions
        classification_loss = nn.CrossEntropyLoss()
        regression_loss = nn.MSELoss()

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['length'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(features, lengths=lengths)

                # Calculate losses
                class_loss = classification_loss(outputs['class_logits'], labels)
                reg_loss = regression_loss(outputs['progression_score'], torch.rand_like(outputs['progression_score']))  # Placeholder

                total_loss = class_loss + 0.1 * reg_loss
                total_loss.backward()

                optimizer.step()
                train_loss += total_loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    lengths = batch['length'].to(self.device)

                    outputs = self.model(features, lengths=lengths)

                    class_loss = classification_loss(outputs['class_logits'], labels)
                    reg_loss = regression_loss(outputs['progression_score'], torch.rand_like(outputs['progression_score']))

                    total_loss = class_loss + 0.1 * reg_loss
                    val_loss += total_loss.item()

            # Update learning rate
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(save_path, optimizer.state_dict())

            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

        logger.info("Training completed!")


def create_mock_sequence_data(num_sequences: int = 100) -> List[Dict]:
    """Create mock training data for testing"""
    sequences = []

    for i in range(num_sequences):
        seq_len = np.random.randint(3, 8)
        features = [np.random.randn(2048) for _ in range(seq_len)]
        labels = np.random.randint(0, 3, seq_len)
        time_intervals = np.random.randint(1, 30, seq_len)

        sequences.append({
            'features': features,
            'labels': labels,
            'time_intervals': time_intervals
        })

    return sequences


if __name__ == "__main__":
    # Example usage
    tracker = ProgressionTracker()

    # Create mock data
    train_data = create_mock_sequence_data(80)
    val_data = create_mock_sequence_data(20)

    # Train model
    tracker.train(train_data, val_data, epochs=5, batch_size=8)

    # Test prediction
    test_features = [np.random.randn(2048) for _ in range(5)]
    result = tracker.predict_progression(test_features)
    print("Progression prediction:", result)
