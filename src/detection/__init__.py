"""
Modules de détection pour l'extraction des features faciales.
"""

from .face_detector import FaceDetector
from .landmark_extractor import LandmarkExtractor

__all__ = [
    'FaceDetector',
    'LandmarkExtractor'
]
