import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyFusionModel(nn.Module):
    """
    Early fusion model for multimodal sentiment analysis.
    Concatenates features from text, audio, and visual modalities,
    then applies fully connected layers for sentiment prediction.
    Args:
        text_dim (int): Dimensionality of text features.
        audio_dim (int): Dimensionality of audio features.
        visual_dim (int): Dimensionality of visual features.
        hidden_dim (int): Hidden layer size for fusion MLP.
        dropout_rate (float): Dropout rate for regularization.
    """
    def __init__(
        self, 
        text_dim, 
        audio_dim, 
        visual_dim, 
        hidden_dim=256, 
        dropout_rate=0.3
    ):
        super(EarlyFusionModel, self).__init__()
        
        # Total input dimension after concatenation
        total_dim = text_dim + audio_dim + visual_dim
        
        # Fully connected layers for fusion and prediction
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, text_features, audio_features, visual_features):
        """
        Forward pass for early fusion model.

        Args:
            text_features (Tensor): Tensor of shape [batch_size, text_dim].
            audio_features (Tensor): Tensor of shape [batch_size, audio_dim].
            visual_features (Tensor): Tensor of shape [batch_size, visual_dim].

        Returns:
            Tensor: Predicted sentiment score of shape [batch_size, 1].
        """
        # Concatenate modality features along the feature dimension
        concat_features = torch.cat([text_features, audio_features, visual_features], dim=1)
        
        # Apply fusion layers
        sentiment = self.fusion_layers(concat_features)
        
        return sentiment

