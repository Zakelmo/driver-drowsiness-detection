"""
Modules de modèles Deep Learning pour la détection de somnolence.
"""

from .cnn import EyeCNN, YawnCNN
from .mlp import FatigueMLP
from .transfer_learning import TransferLearningModel

__all__ = [
    'EyeCNN',
    'YawnCNN', 
    'FatigueMLP',
    'TransferLearningModel'
]
