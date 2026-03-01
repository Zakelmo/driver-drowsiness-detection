"""
Extracteur de landmarks faciaux utilisant MediaPipe.

MediaPipe Face Mesh fournit 468 points de repère 3D du visage,
permettant une détection précise des yeux, bouche, etc.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

# Import MediaPipe avec gestion multi-version
try:
    import mediapipe as mp
    
    # Essayer différents chemins d'accès pour solutions
    try:
        # Ancienne API: mp.solutions
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    except AttributeError:
        try:
            # Nouvelle API: mediapipe.python.solutions
            from mediapipe.python import solutions
            mp_face_mesh = solutions.face_mesh
            mp_drawing = solutions.drawing_utils
            mp_drawing_styles = solutions.drawing_styles
        except (ImportError, AttributeError):
            # Dernier essai: importer directement
            try:
                from mediapipe.python.solutions import face_mesh as mp_face_mesh
                from mediapipe.python.solutions import drawing_utils as mp_drawing
                from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
            except ImportError:
                mp_face_mesh = None
                mp_drawing = None
                mp_drawing_styles = None
    
    MP_AVAILABLE = mp_face_mesh is not None
    
except ImportError:
    mp = None
    mp_face_mesh = None
    mp_drawing = None
    mp_drawing_styles = None
    MP_AVAILABLE = False


class LandmarkExtractor:
    """
    Extracteur de landmarks faciaux avec MediaPipe Face Mesh.
    """
    
    # Indices des landmarks importants
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Oeil gauche
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Oeil droit
    MOUTH_INDICES = [61, 291, 0, 17]  # Bouche: gauche, droite, haut, bas
    
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
        if not MP_AVAILABLE or mp_face_mesh is None:
            raise ImportError(
                "MediaPipe Face Mesh n'est pas disponible. "
                "Essayez: pip install mediapipe==0.10.13"
            )
        
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_face_mesh = mp_face_mesh
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        
        # Créer le détecteur FaceMesh
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=self.static_image_mode,
                max_num_faces=self.max_num_faces,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
        except Exception as e:
            raise RuntimeError(f"Erreur initialisation FaceMesh: {e}")
        
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
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(rgb_image)
        
        return (
            self.results is not None and 
            hasattr(self.results, 'multi_face_landmarks') and 
            self.results.multi_face_landmarks is not None
        )
    
    def get_face_landmarks(self, face_idx: int = 0) -> Optional[List]:
        """Récupère tous les landmarks d'un visage."""
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
        """Récupère les landmarks des yeux."""
        landmarks = self.get_face_landmarks(face_idx)
        if landmarks is None:
            return None, None
        
        h, w = image_shape[:2]
        
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
        """Récupère les landmarks de la bouche."""
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
        if self.mp_drawing is not None and self.mp_drawing_styles is not None:
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
            except Exception:
                pass
        
        return annotated_image
    
    def close(self):
        """Libère les ressources."""
        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except:
                pass


# Test
if __name__ == "__main__":
    print("Test de LandmarkExtractor")
    print("="*50)
    print(f"MediaPipe disponible: {MP_AVAILABLE}")
    
    try:
        extractor = LandmarkExtractor()
        print("✓ LandmarkExtractor initialisé avec succès!")
    except Exception as e:
        print(f"✗ Erreur: {e}")
