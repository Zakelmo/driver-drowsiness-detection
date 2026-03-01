"""
Script de détection de somnolence en temps réel.

Utilise la webcam pour capturer le visage du conducteur,
détecte la fatigue en temps réel et génère des alertes.
"""

import os
import sys
import cv2
import numpy as np
import argparse
from datetime import datetime

# Ajout du path source
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from detection.landmark_extractor import LandmarkExtractor
from utils.preprocessing import ImagePreprocessor
from utils.metrics import FatigueMetrics, calculate_ear, calculate_mar
from utils.alerts import AlertSystem
from models.cnn import EyeCNN


class DrowsinessDetector:
    """
    Détecteur de somnolence en temps réel.
    """
    
    def __init__(self, 
                 use_cnn: bool = False,
                 cnn_model_path: str = None,
                 camera_id: int = 0,
                 display_size: tuple = (800, 600)):
        """
        Initialise le détecteur.
        
        Args:
            use_cnn: Utiliser le modèle CNN pour la classification
            cnn_model_path: Chemin vers le modèle CNN sauvegardé
            camera_id: ID de la caméra
            display_size: Taille d'affichage
        """
        self.use_cnn = use_cnn
        self.camera_id = camera_id
        self.display_size = display_size
        
        # Initialisation des composants
        print("Initialisation du système de détection...")
        
        self.landmark_extractor = LandmarkExtractor(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.preprocessor = ImagePreprocessor()
        self.fatigue_metrics = FatigueMetrics()
        self.alert_system = AlertSystem()
        
        # Chargement du modèle CNN si demandé
        self.eye_cnn = None
        if use_cnn and cnn_model_path and os.path.exists(cnn_model_path):
            print(f"Chargement du modèle CNN: {cnn_model_path}")
            self.eye_cnn = EyeCNN()
            self.eye_cnn.load(cnn_model_path)
        
        # Initialisation de la caméra
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Variables de statistiques
        self.frame_count = 0
        self.detection_count = 0
        self.alert_count = 0
        self.start_time = datetime.now()
        
        print("✓ Système initialisé et prêt!")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Traite une frame vidéo.
        
        Args:
            frame: Frame BGR d'OpenCV
            
        Returns:
            Frame annotée
        """
        self.frame_count += 1
        result_frame = frame.copy()
        
        # Détection des landmarks
        face_detected = self.landmark_extractor.process(frame)
        
        if face_detected:
            self.detection_count += 1
            
            # Extraction des landmarks des yeux
            left_eye_pts, right_eye_pts = self.landmark_extractor.get_eye_landmarks(
                frame.shape
            )
            
            # Extraction des landmarks de la bouche
            mouth_pts = self.landmark_extractor.get_mouth_landmarks(frame.shape)
            
            # Calcul des EAR
            ear_left = calculate_ear(left_eye_pts) if left_eye_pts else 0.3
            ear_right = calculate_ear(right_eye_pts) if right_eye_pts else 0.3
            ear_avg = (ear_left + ear_right) / 2
            
            # Calcul du MAR
            mar = calculate_mar(mouth_pts) if mouth_pts else 0.2
            
            # Mise à jour des métriques de fatigue
            alerts = self.fatigue_metrics.update(ear_avg, mar)
            
            # Gestion des alertes
            if alerts['drowsiness_alert'] or alerts['eye_closure']:
                self.alert_count += 1
                self.alert_system.trigger_alert('drowsiness')
                result_frame = self.alert_system.get_visual_alert_frame(
                    result_frame, 'drowsiness'
                )
            elif alerts['yawn']:
                self.alert_system.trigger_alert('yawn')
            
            # Dessiner les landmarks
            result_frame = self.landmark_extractor.draw_landmarks(result_frame)
            
            # Affichage des métriques
            result_frame = self._draw_metrics(result_frame, ear_avg, mar, alerts)
        
        else:
            # Aucun visage détecté
            cv2.putText(result_frame, "Aucun visage detecte", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Affichage des statistiques globales
        result_frame = self._draw_stats(result_frame)
        
        return result_frame
    
    def _draw_metrics(self, frame: np.ndarray, ear: float, mar: float, 
                     alerts: dict) -> np.ndarray:
        """
        Dessine les métriques sur la frame.
        
        Args:
            frame: Frame d'entrée
            ear: Eye Aspect Ratio
            mar: Mouth Aspect Ratio
            alerts: Dictionnaire d'alertes
            
        Returns:
            Frame avec métriques
        """
        h, w = frame.shape[:2]
        
        # Panel d'informations (fond noir semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Couleurs selon les seuils
        ear_color = (0, 255, 0) if ear > 0.25 else (0, 0, 255)
        mar_color = (0, 255, 0) if mar < 0.6 else (0, 165, 255)
        
        # EAR
        cv2.putText(frame, f"EAR: {ear:.3f}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
        
        # MAR
        cv2.putText(frame, f"MAR: {mar:.3f}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mar_color, 2)
        
        # Statut yeux
        eye_status = "OUVERTS" if ear > 0.25 else "FERMES"
        eye_status_color = (0, 255, 0) if ear > 0.25 else (0, 0, 255)
        cv2.putText(frame, f"Yeux: {eye_status}", (20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_status_color, 2)
        
        # Alertes
        alert_text = ""
        alert_color = (0, 255, 0)
        if alerts['drowsiness_alert']:
            alert_text = "⚠️ FATIGUE DETECTEE!"
            alert_color = (0, 0, 255)
        elif alerts['eye_closure']:
            alert_text = "Yeux fermes prolonges"
            alert_color = (0, 165, 255)
        elif alerts['yawn']:
            alert_text = "Baillement detecte"
            alert_color = (0, 255, 255)
        else:
            alert_text = "✅ Normal"
        
        cv2.putText(frame, alert_text, (20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
        
        return frame
    
    def _draw_stats(self, frame: np.ndarray) -> np.ndarray:
        """
        Dessine les statistiques globales.
        
        Args:
            frame: Frame d'entrée
            
        Returns:
            Frame avec statistiques
        """
        h, w = frame.shape[:2]
        
        # Panel en bas à droite
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-250, h-100), (w-10, h-10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS
        elapsed = (datetime.now() - self.start_time).total_seconds()
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-240, h-70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Détections
        detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
        cv2.putText(frame, f"Detection: {detection_rate:.0f}%", (w-240, h-45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Alertes
        cv2.putText(frame, f"Alertes: {self.alert_count}", (w-240, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """
        Lance la détection en temps réel.
        """
        print("\n" + "="*60)
        print("DÉTECTION DE SOMNOLENCE - TEMPS RÉEL")
        print("="*60)
        print("Commandes:")
        print("  - 'q' : Quitter")
        print("  - 's' : Sauvegarder une capture")
        print("  - 'r' : Réinitialiser les compteurs")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Erreur: Impossible de lire la caméra")
                    break
                
                # Traitement de la frame
                result_frame = self.process_frame(frame)
                
                # Redimensionnement pour affichage
                display_frame = cv2.resize(result_frame, self.display_size)
                
                # Affichage
                cv2.imshow('Détection de Somnolence', display_frame)
                
                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nArrêt demandé par l'utilisateur")
                    break
                
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"reports/capture_{timestamp}.png"
                    cv2.imwrite(filename, result_frame)
                    print(f"✓ Capture sauvegardée: {filename}")
                
                elif key == ord('r'):
                    self.frame_count = 0
                    self.detection_count = 0
                    self.alert_count = 0
                    self.start_time = datetime.now()
                    self.fatigue_metrics.reset()
                    print("✓ Compteurs réinitialisés")
        
        except KeyboardInterrupt:
            print("\n\nInterruption clavier")
        
        finally:
            self.stop()
    
    def stop(self):
        """
        Arrête proprement le système.
        """
        print("\n" + "="*60)
        print("STATISTIQUES FINALES")
        print("="*60)
        print(f"Frames traitées: {self.frame_count}")
        print(f"Visages détectés: {self.detection_count}")
        print(f"Alertes générées: {self.alert_count}")
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"Durée: {elapsed:.1f} secondes")
        
        stats = self.fatigue_metrics.get_statistics()
        print(f"\nMétriques de fatigue:")
        print(f"  - Clignements: {stats['blink_count']}")
        print(f"  - Bâillements: {stats['yawn_count']}")
        print(f"  - PERCLOS final: {stats['perclos']*100:.1f}%")
        
        print("\nArrêt du système...")
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.landmark_extractor.close()
        
        print("✓ Système arrêté proprement")


def main():
    """
    Point d'entrée principal.
    """
    parser = argparse.ArgumentParser(
        description='Détection de somnolence en temps réel'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='ID de la caméra (défaut: 0)'
    )
    parser.add_argument(
        '--cnn', '-m',
        type=str,
        default=None,
        help='Chemin vers le modèle CNN (optionnel)'
    )
    parser.add_argument(
        '--width', '-w',
        type=int,
        default=800,
        help='Largeur d\'affichage (défaut: 800)'
    )
    parser.add_argument(
        '--height', '-H',
        type=int,
        default=600,
        help='Hauteur d\'affichage (défaut: 600)'
    )
    
    args = parser.parse_args()
    
    # Création du détecteur
    detector = DrowsinessDetector(
        use_cnn=args.cnn is not None,
        cnn_model_path=args.cnn,
        camera_id=args.camera,
        display_size=(args.width, args.height)
    )
    
    # Lancement
    detector.run()


if __name__ == "__main__":
    main()
