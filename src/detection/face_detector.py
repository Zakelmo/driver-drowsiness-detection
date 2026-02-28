"""
Module de détection de visage.

Utilise OpenCV Haar Cascades ou MediaPipe pour détecter
et localiser le visage dans une image.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class FaceDetector:
    """
    Détecteur de visage utilisant Haar Cascades ou DNN.
    """
    
    def __init__(self, method: str = 'haar'):
        """
        Initialise le détecteur de visage.
        
        Args:
            method: Méthode de détection ('haar' ou 'dnn')
        """
        self.method = method
        self.face_cascade = None
        self.eye_cascade = None
        
        if method == 'haar':
            self._init_haar_cascades()
    
    def _init_haar_cascades(self):
        """Initialise les classificateurs Haar."""
        # Chargement des classificateurs pré-entraînés
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Détecte le visage dans l'image.
        
        Args:
            image: Image d'entrée (BGR)
            
        Returns:
            Bounding box (x, y, w, h) ou None si pas de visage
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Détection des visages
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) == 0:
            return None
        
        # Retourner le plus grand visage (probablement le conducteur)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest_face)
    
    def detect_eyes(self, 
                   face_roi: np.ndarray) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """
        Détecte les yeux dans la région du visage.
        
        Args:
            face_roi: Région d'intérêt du visage
            
        Returns:
            Tuple (oeil_gauche, oeil_droit) avec bounding boxes
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        if len(eyes) < 2:
            return None, None
        
        # Trier par position x (gauche à droite)
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # Le plus à gauche est l'œil gauche, le plus à droite l'œil droit
        left_eye = tuple(eyes[0])
        right_eye = tuple(eyes[-1])
        
        return left_eye, right_eye
    
    def draw_detections(self, 
                       image: np.ndarray,
                       face_bbox: Tuple[int, ...],
                       left_eye: Optional[Tuple] = None,
                       right_eye: Optional[Tuple] = None) -> np.ndarray:
        """
        Dessine les détections sur l'image.
        
        Args:
            image: Image originale
            face_bbox: Bounding box du visage (x, y, w, h)
            left_eye: Bounding box œil gauche
            right_eye: Bounding box œil droit
            
        Returns:
            Image avec annotations
        """
        result = image.copy()
        
        # Dessiner le visage
        x, y, w, h = face_bbox
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(result, 'Visage', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Dessiner les yeux
        if left_eye:
            ex, ey, ew, eh = left_eye
            cv2.rectangle(result, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        
        if right_eye:
            ex, ey, ew, eh = right_eye
            cv2.rectangle(result, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        
        return result


if __name__ == "__main__":
    print("FaceDetector initialisé!")
    detector = FaceDetector()
    print("✓ Détecteur Haar Cascade chargé")
