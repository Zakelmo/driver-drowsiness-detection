"""
Modules de modèles Deep Learning pour la détection de somnolence.
"""

# Import conditionnel pour éviter les erreurs si TensorFlow n'est pas installé
try:
    from .cnn import EyeCNN, YawnCNN
    from .mlp import FatigueMLP
    from .transfer_learning import TransferLearningModel
    __all__ = [
        'EyeCNN',
        'YawnCNN', 
        'FatigueMLP',
        'TransferLearningModel'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Models not available: {e}")
    __all__ = []
