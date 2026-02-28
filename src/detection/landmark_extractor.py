"""
Extracteur de landmarks faciaux utilisant MediaPipe.

MediaPipe Face Mesh fournit 468 points de repère 3D du visage,
permettant une détection précise des yeux, bouche, etc.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
from collections import namedtuple


class LandmarkExtractor:
    """
    Extracteur de landmarks faciaux avec MediaPipe Face Mesh.
    """
    
    # Indices des landmarks importants
    # OEIL GAUCHE (de la perspective de la personne)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    # OEIL DROIT
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    # BOUCHE (extérieur + intérieur)
    MOUTH_OUTER_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    MOUTH_INNER_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialise l'extracteur de landmarks.
        
        Args:
            static_image_mode: True pour images, False pour vidéo
            max_num_faces: Nombre maximum de visages à détecter
            min_detection_confidence: Seuil de confiance pour la détection
            min_tracking_confidence: Seuil de confiance pour le tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.results = None
    
    def process(self, image: np.ndarray) -> bool:
        """
        Traite une image pour extraire les landmarks.
        
        Args:
            image: Image BGR (OpenCV)
            
        Returns:
            True si un visage est détecté
        """
        # Conversion BGR -> RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Traitement
        self.results = self.face_mesh.process(rgb_image)
        
        return self.results.multi_face_landmarks is not None
    
    def get_face_landmarks(self, face_idx: int = 0) -> Optional[List]:
        """
        Récupère tous les landmarks d'un visage.
        
        Args:
            face_idx: Index du visage (si plusieurs visages)
            
        Returns:
            Liste des landmarks ou None
        """
        if self.results is None or self.results.multi_face_landmarks is None:
            return None
        
        if face_idx >= len(self.results.multi_face_landmarks):
            return None
        
        return self.results.multi_face_landmarks[face_idx].landmark
    
    def get_eye_landmarks(self, 
                         image_shape: Tuple[int, int],
                         face_idx: int = 0) -> Tuple[Optional[List], Optional[List]]:
        """
        Récupère les landmarks des yeux.
        
        Args:
            image_shape: Dimensions de l'image (h, w)
            face_idx: Index du visage
            
        Returns:
            Tuple (landmarks_oeil_gauche, landmarks_oeil_droit)
        """
        landmarks = self.get_face_landmarks(face_idx)
        if landmarks is None:
            return None, None
        
        h, w = image_shape[:2]
        
        # Extraire les points des yeux
        left_eye = []
        for idx in self.LEFT_EYE_INDICES:
            lm = landmarks[idx]
            left_eye.append((int(lm.x * w), int(lm.y * h)))
        
        right_eye = []
        for idx in self.RIGHT_EYE_INDICES:
            lm = landmarks[idx]
            right_eye.append((int(lm.x * w), int(lm.y * h)))
        
        return left_eye, right_eye
    
    def get_mouth_landmarks(self,
                           image_shape: Tuple[int, int],
                           face_idx: int = 0) -> Optional[List]:
        """
        Récupère les landmarks de la bouche.
        
        Args:
            image_shape: Dimensions de l'image
            face_idx: Index du visage
            
        Returns:
            Liste des points de la bouche
        """
        landmarks = self.get_face_landmarks(face_idx)
        if landmarks is None:
            return None
        
        h, w = image_shape[:2]
        
        # Points clés de la bouche pour le MAR
        # 4 points: coins gauche/droite, lèvres sup/inf
        mouth_indices = [61, 291, 0, 17]  # gauche, droite, haut, bas
        
        mouth_points = []
        for idx in mouth_indices:
            lm = landmarks[idx]
            mouth_points.append((int(lm.x * w), int(lm.y * h)))
        
        return mouth_points
    
    def get_eye_regions(self,
                       image: np.ndarray,
                       face_idx: int = 0,
                       padding: int = 5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extrait les régions des yeux de l'image.
        
        Args:
            image: Image complète
            face_idx: Index du visage
            padding: Marge autour des yeux
            
        Returns:
            Tuple (region_oeil_gauche, region_oeil_droit)
        """
        left_eye_pts, right_eye_pts = self.get_eye_landmarks(image.shape, face_idx)
        
        if left_eye_pts is None or right_eye_pts is None:
            return None, None
        
        # Calculer les bounding boxes
        left_eye = self._points_to_bbox(left_eye_pts, image.shape, padding)
        right_eye = self._points_to_bbox(right_eye_pts, image.shape, padding)
        
        if left_eye is None or right_eye is None:
            return None, None
        
        # Extraire les régions
        lx, ly, lw, lh = left_eye
        rx, ry, rw, rh = right_eye
        
        left_roi = image[ly:ly+lh, lx:lx+lw]
        right_roi = image[ry:ry+rh, rx:rx+rw]
        
        return left_roi, right_roi
    
    def _points_to_bbox(self,
                       points: List[Tuple[int, int]],
                       image_shape: Tuple[int, int],
                       padding: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Convertit une liste de points en bounding box.
        
        Args:
            points: Liste de points (x, y)
            image_shape: Dimensions de l'image
            padding: Marge à ajouter
            
        Returns:
            Bounding box (x, y, w, h)
        """
        if not points:
            return None
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        x_min = max(0, min(xs) - padding)
        y_min = max(0, min(ys) - padding)
        x_max = min(image_shape[1], max(xs) + padding)
        y_max = min(image_shape[0], max(ys) + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def draw_landmarks(self,
                      image: np.ndarray,
                      draw_eyes: bool = True,
                      draw_mouth: bool = True,
                      face_idx: int = 0) -> np.ndarray:
        """
        Dessine les landmarks sur l'image.
        
        Args:
            image: Image d'entrée
            draw_eyes: Dessiner les yeux
            draw_mouth: Dessiner la bouche
            face_idx: Index du visage
            
        Returns:
            Image avec landmarks
        """
        if self.results is None or self.results.multi_face_landmarks is None:
            return image
        
        annotated_image = image.copy()
        
        face_landmarks = self.results.multi_face_landmarks[face_idx]
        
        # Dessiner tous les landmarks
        self.mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        # Dessiner les contours des yeux
        if draw_eyes:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        return annotated_image
    
    def close(self):
        """Libère les ressources."""
        self.face_mesh.close()


if __name__ == "__main__":
    print("LandmarkExtractor initialisé avec MediaPipe!")
    extractor = LandmarkExtractor()
    print(f"✓ MediaPipe Face Mesh chargé")
    print(f"  - Points par œil: 6")
    print(f"  - Total landmarks: 468")
