"""
Driver Drowsiness Detection System
Système de Détection de Somnolence du Conducteur

Ce package contient l'ensemble des modules pour la détection 
de la fatigue et de la somnolence par Deep Learning.
"""

__version__ = "1.0.0"
__author__ = "SDIA Student"

from .utils.preprocessing import ImagePreprocessor
from .utils.metrics import FatigueMetrics, ModelMetrics
from .utils.alerts import AlertSystem

__all__ = [
    'ImagePreprocessor',
    'FatigueMetrics', 
    'ModelMetrics',
    'AlertSystem'
]
