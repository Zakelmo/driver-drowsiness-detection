"""
Extracteur de landmarks faciaux utilisant MediaPipe.

MediaPipe Face Mesh fournit 468 points de repère 3D du visage,
permettant une détection précise des yeux, bouche, etc.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

# Import MediaPipe
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    mp = None


class LandmarkExtractor:
    """
    Extracteur de landmarks faciaux avec MediaPipe Face Mesh.
    """
    
    # Indices des landmarks importants
    # OEIL GAUCHE (de la perspective de la personne)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    # OEIL DROIT
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    # Points bouche pour MAR
    MOUTH_INDICES = [61, 291, 0, 17]  # gauche, droite, haut, bas
    
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
        if not MP_AVAILABLE or mp is None:
            raise ImportError(
                "MediaPipe n'est pas installé. "
                "Installez avec: pip install mediapipe"
            )
        
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Essayer d'accéder à solutions de différentes façons
        try:
            # Méthode 1: mp.solutions (ancienne API)
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        except AttributeError:
            try:
                # Méthode 2: mp.python.solutions (nouvelle API)
                from mediapipe.python import solutions
                self.mp_face_mesh = solutions.face_mesh
                self.mp_drawing = solutions.drawing_utils
                self.mp_drawing_styles = solutions.drawing_styles
            except ImportError:
                raise ImportError(
                    "Impossible d'accéder à MediaPipe solutions. "
                    "Essayez: pip install mediapipe==0.10.0"
                )
        
        # Créer le détecteur FaceMesh
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=self.static_image_mode,
                max_num_faces=self.max_num_faces,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'initialisation de FaceMesh: {e}")
        
        self.results = None
    
    def process(self, image: np.ndarray) -> bool:
        """
        Traite une image pour extraire les landmarks.
        
        Args:
            image: Image BGR (OpenCV)
            
        Returns:
            True si un visage est détecté
        """
        if self.face_mesh is None:
            return False
        
        # Conversion BGR -> RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Traitement
        self.results = self.face_mesh.process(rgb_image)
        
        return (
            self.results is not None and 
            hasattr(self.results, 'multi_face_landmarks') and 
            self.results.multi_face_landmarks is not None
        )
    
    def get_face_landmarks(self, face_idx: int = 0) -> Optional[List]:
        """
        Récupère tous les landmarks d'un visage.
        
        Args:
            face_idx: Index du visage (si plusieurs visages)
            
        Returns:
            Liste des landmarks ou None
        """
        if self.results is None or not hasattr(self.results, 'multi_face_landmarks'):
            return None
        
        if self.results.multi_face_landmarks is None:
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
            if idx < len(landmarks):
                lm = landmarks[idx]
                left_eye.append((int(lm.x * w), int(lm.y * h)))
        
        right_eye = []
        for idx in self.RIGHT_EYE_INDICES:
            if idx < len(landmarks):
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
        
        mouth_points = []
        for idx in self.MOUTH_INDICES:
            if idx < len(landmarks):
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
        
        # Vérifier les limites
        h, w = image.shape[:2]
        lx, ly = max(0, lx), max(0, ly)
        rx, ry = max(0, rx), max(0, ry)
        
        left_roi = image[ly:min(ly+lh, h), lx:min(lx+lw, w)]
        right_roi = image[ry:min(ry+rh, h), rx:min(rx+rw, w)]
        
        return left_roi, right_roi
    
    def _points_to_bbox(self,
                       points: List[Tuple[int, int]],
                       image_shape: Tuple[int, int],
                       padding: int) -> Optional[Tuple[int, int, int, int]]:
        """Convertit une liste de points en bounding box."""
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
        """Dessine les landmarks sur l'image."""
        if self.face_mesh is None or self.results is None:
            return image
        
        if not hasattr(self.results, 'multi_face_landmarks') or self.results.multi_face_landmarks is None:
            return image
        
        annotated_image = image.copy()
        
        face_landmarks = self.results.multi_face_landmarks[face_idx]
        
        # Dessiner les landmarks
        try:
            # Tous les landmarks
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Yeux
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
        except Exception as e:
            # Fallback: dessiner les points clés manuellement
            h, w = image.shape[:2]
            for idx in [33, 133, 362, 263]:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)
        
        return annotated_image
    
    def close(self):
        """Libère les ressources."""
        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except:
                pass


if __name__ == "__main__":
    print("Test de LandmarkExtractor")
    print("="*50)
    
    try:
        extractor = LandmarkExtractor()
        print("✓ LandmarkExtractor initialisé avec succès!")
    except Exception as e:
        print(f"✗ Erreur: {e}")
