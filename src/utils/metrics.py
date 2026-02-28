"""
Module de métriques pour la détection de somnolence.

Implémente :
- EAR (Eye Aspect Ratio) : Détection fermeture des yeux
- MAR (Mouth Aspect Ratio) : Détection bâillements
- PERCLOS : Pourcentage de fermeture des paupières
- Métriques d'évaluation des modèles (Chapitre 1, 2)
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional
from scipy.spatial import distance as dist
import cv2


def calculate_ear(eye_landmarks: List[Tuple[int, int]]) -> float:
    """
    Calcule l'Eye Aspect Ratio (EAR) pour détecter la fermeture des yeux.
    
    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    
    Indices des landmarks de l'œil (MediaPipe Face Mesh):
    P1 (0) ---- P4 (3)
       |   Eye   |
    P2 (1) P3 (2)
       |         |
    P6 (5) P5 (4)
    
    Args:
        eye_landmarks: Liste de 6 points (x, y) de l'œil
        
    Returns:
        Valeur EAR (diminue quand l'œil se ferme)
    """
    if len(eye_landmarks) != 6:
        raise ValueError("6 landmarks requis pour calculer l'EAR")
    
    # Calcul des distances euclidiennes
    # Distance verticale
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])  # P2-P6
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])  # P3-P5
    
    # Distance horizontale
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])  # P1-P4
    
    # Calcul EAR
    ear = (A + B) / (2.0 * C) if C > 0 else 0
    
    return ear


def calculate_mar(mouth_landmarks: List[Tuple[int, int]]) -> float:
    """
    Calcule le Mouth Aspect Ratio (MAR) pour détecter les bâillements.
    
    MAR = (||P2-P8|| + ||P3-P7|| + ||P4-P6||) / (2 * ||P1-P5||)
    
    Indices des landmarks de la bouche (MediaPipe):
    P1 (0) - Coin gauche
    P2 (1), P3 (2), P4 (3) - Lèvre supérieure
    P5 (6) - Coin droit
    P6 (4), P7 (5), P8 (6) - Lèvre inférieure
    
    Args:
        mouth_landmarks: Liste de points (x, y) de la bouche
        
    Returns:
        Valeur MAR (augmente pendant un bâillement)
    """
    # Pour la bouche, on utilise typiquement 12 points (6 supérieurs, 6 inférieurs)
    # Version simplifiée avec 4 points principaux
    if len(mouth_landmarks) < 4:
        raise ValueError("Au moins 4 landmarks requis pour calculer le MAR")
    
    # Points principaux de la bouche
    left_corner = mouth_landmarks[0]    # Coin gauche
    upper_lip = mouth_landmarks[1]      # Lèvre supérieure centrale
    right_corner = mouth_landmarks[2]   # Coin droit
    lower_lip = mouth_landmarks[3]      # Lèvre inférieure centrale
    
    # Distance verticale (ouverture de la bouche)
    vertical = dist.euclidean(upper_lip, lower_lip)
    
    # Distance horizontale (largeur de la bouche)
    horizontal = dist.euclidean(left_corner, right_corner)
    
    # Calcul MAR
    mar = vertical / horizontal if horizontal > 0 else 0
    
    return mar


def calculate_perclos(eye_status_history: List[int], 
                      window_size: int = 30,
                      fps: int = 30) -> float:
    """
    Calcule le PERCLOS (PERcentage of eye CLOSure).
    
    PERCLOS = (nombre de frames yeux fermés / total frames) * 100
    
    Args:
        eye_status_history: Historique des états des yeux (0=ouvert, 1=fermé)
        window_size: Taille de la fenêtre temporelle en secondes
        fps: Images par seconde
        
    Returns:
        Valeur PERCLOS entre 0 et 1
    """
    num_frames = window_size * fps
    
    if len(eye_status_history) < num_frames:
        # Pas assez d'historique, utiliser ce qui est disponible
        num_frames = len(eye_status_history)
    
    if num_frames == 0:
        return 0.0
    
    # Récupérer les dernières frames
    recent_frames = eye_status_history[-num_frames:]
    
    # Compter les frames avec yeux fermés
    closed_frames = sum(recent_frames)
    
    # Calcul PERCLOS
    perclos = closed_frames / num_frames
    
    return perclos


class FatigueMetrics:
    """
    Classe pour suivre et calculer les métriques de fatigue en temps réel.
    
    Implémente les indicateurs clés de la littérature sur la détection
    de somnolence au volant.
    """
    
    def __init__(self, 
                 ear_threshold: float = 0.25,
                 mar_threshold: float = 0.6,
                 perclos_threshold: float = 0.15,
                 eye_closure_frames: int = 20,
                 yawn_frames: int = 30,
                 fps: int = 30):
        """
        Initialise le tracker de métriques de fatigue.
        
        Args:
            ear_threshold: Seuil EAR pour considérer l'œil fermé
            mar_threshold: Seuil MAR pour considérer un bâillement
            perclos_threshold: Seuil PERCLOS pour alerter (15%)
            eye_closure_frames: Nombre de frames consécutifs pour alerter
            yawn_frames: Nombre de frames consécutifs pour confirmer bâillement
            fps: Images par seconde de la vidéo
        """
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.perclos_threshold = perclos_threshold
        self.eye_closure_frames = eye_closure_frames
        self.yawn_frames = yawn_frames
        self.fps = fps
        
        # Historiques pour analyse temporelle
        self.ear_history = deque(maxlen=fps * 60)  # 60 secondes d'historique
        self.mar_history = deque(maxlen=fps * 60)
        self.eye_status_history = deque(maxlen=fps * 60)  # 0=ouvert, 1=fermé
        self.yawn_status_history = deque(maxlen=fps * 60)  # 0=non, 1=bâillement
        
        # Compteurs
        self.eye_closure_counter = 0
        self.yawn_counter = 0
        self.blink_counter = 0
        self.yawn_count = 0
        
        # États
        self.eyes_closed = False
        self.yawning = False
        
    def update(self, ear: float, mar: float = None):
        """
        Met à jour les métriques avec les nouvelles valeurs.
        
        Args:
            ear: Eye Aspect Ratio actuel
            mar: Mouth Aspect Ratio actuel (optionnel)
            
        Returns:
            Dict avec les alertes déclenchées
        """
        alerts = {
            'eye_closure': False,
            'yawn': False,
            'perclos_warning': False,
            'drowsiness_alert': False
        }
        
        # Historique EAR
        self.ear_history.append(ear)
        
        # Détection fermeture des yeux
        if ear < self.ear_threshold:
            self.eye_closure_counter += 1
            self.eye_status_history.append(1)  # Fermé
            
            # Détecter si les yeux restent fermés trop longtemps
            if self.eye_closure_counter >= self.eye_closure_frames:
                if not self.eyes_closed:
                    alerts['eye_closure'] = True
                    self.eyes_closed = True
        else:
            if self.eye_closure_counter > 0 and self.eye_closure_counter < self.eye_closure_frames:
                # C'était un clignement
                self.blink_counter += 1
            
            self.eye_closure_counter = 0
            self.eye_status_history.append(0)  # Ouvert
            self.eyes_closed = False
        
        # Détection bâillement
        if mar is not None:
            self.mar_history.append(mar)
            
            if mar > self.mar_threshold:
                self.yawn_counter += 1
                self.yawn_status_history.append(1)
                
                if self.yawn_counter >= self.yawn_frames:
                    if not self.yawning:
                        alerts['yawn'] = True
                        self.yawning = True
                        self.yawn_count += 1
            else:
                self.yawn_counter = 0
                self.yawn_status_history.append(0)
                self.yawning = False
        
        # Calcul PERCLOS
        perclos = calculate_perclos(list(self.eye_status_history), 
                                     window_size=60, fps=self.fps)
        
        if perclos > self.perclos_threshold:
            alerts['perclos_warning'] = True
        
        # Alert global de fatigue
        if (alerts['eye_closure'] or 
            (alerts['yawn'] and perclos > self.perclos_threshold / 2)):
            alerts['drowsiness_alert'] = True
        
        return alerts
    
    def get_statistics(self) -> Dict:
        """
        Retourne les statistiques de fatigue calculées.
        
        Returns:
            Dict avec les statistiques
        """
        perclos = calculate_perclos(list(self.eye_status_history), 
                                     window_size=60, fps=self.fps)
        
        stats = {
            'current_ear': self.ear_history[-1] if self.ear_history else 0,
            'current_mar': self.mar_history[-1] if self.mar_history else 0,
            'perclos': perclos,
            'blink_count': self.blink_counter,
            'yawn_count': self.yawn_count,
            'eyes_closed_frames': self.eye_closure_counter,
            'is_eyes_closed': self.eyes_closed,
            'is_yawning': self.yawning
        }
        
        return stats
    
    def get_ear_trend(self, window: int = 30) -> float:
        """
        Calcule la tendance de l'EAR sur une fenêtre.
        
        Args:
            window: Nombre de frames pour la tendance
            
        Returns:
            Moyenne de l'EAR sur la fenêtre
        """
        if len(self.ear_history) < window:
            window = len(self.ear_history)
        
        if window == 0:
            return 0.0
        
        return np.mean(list(self.ear_history)[-window:])
    
    def reset(self):
        """Réinitialise tous les compteurs et historiques."""
        self.ear_history.clear()
        self.mar_history.clear()
        self.eye_status_history.clear()
        self.yawn_status_history.clear()
        
        self.eye_closure_counter = 0
        self.yawn_counter = 0
        self.blink_counter = 0
        self.yawn_count = 0
        
        self.eyes_closed = False
        self.yawning = False


class ModelMetrics:
    """
    Classe pour calculer les métriques d'évaluation des modèles.
    
    Implémente les métriques vues dans le cours :
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix
    - ROC-AUC
    """
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule l'accuracy (précision globale)."""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule la précision (TP / (TP + FP))."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule le recall/sensitivity (TP / (TP + FN))."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule le F1-Score (moyenne harmonique de precision et recall)."""
        prec = ModelMetrics.precision(y_true, y_pred)
        rec = ModelMetrics.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calcule la matrice de confusion.
        
        Returns:
            Matrice 2x2 : [[TN, FP], [FN, TP]]
        """
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        return np.array([[tn, fp], [fn, tp]])
    
    @staticmethod
    def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule la spécificité (TN / (TN + FP))."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    @staticmethod
    def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcule toutes les métriques d'évaluation.
        
        Returns:
            Dict avec toutes les métriques
        """
        return {
            'accuracy': ModelMetrics.accuracy(y_true, y_pred),
            'precision': ModelMetrics.precision(y_true, y_pred),
            'recall': ModelMetrics.recall(y_true, y_pred),
            'f1_score': ModelMetrics.f1_score(y_true, y_pred),
            'specificity': ModelMetrics.specificity(y_true, y_pred)
        }


def visualize_metrics(metrics_tracker: FatigueMetrics, save_path: str = None):
    """
    Visualise les métriques de fatigue en temps réel.
    
    Args:
        metrics_tracker: Instance de FatigueMetrics
        save_path: Chemin pour sauvegarder la figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Historique EAR
    if metrics_tracker.ear_history:
        axes[0, 0].plot(list(metrics_tracker.ear_history), label='EAR', color='blue')
        axes[0, 0].axhline(y=metrics_tracker.ear_threshold, 
                          color='r', linestyle='--', label='Seuil')
        axes[0, 0].set_title('Historique Eye Aspect Ratio (EAR)')
        axes[0, 0].set_xlabel('Frames')
        axes[0, 0].set_ylabel('EAR')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Historique MAR
    if metrics_tracker.mar_history:
        axes[0, 1].plot(list(metrics_tracker.mar_history), label='MAR', color='green')
        axes[0, 1].axhline(y=metrics_tracker.mar_threshold, 
                          color='r', linestyle='--', label='Seuil')
        axes[0, 1].set_title('Historique Mouth Aspect Ratio (MAR)')
        axes[0, 1].set_xlabel('Frames')
        axes[0, 1].set_ylabel('MAR')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Statistiques globales
    stats = metrics_tracker.get_statistics()
    stat_names = ['Blink Count', 'Yawn Count', 'PERCLOS (%)']
    stat_values = [
        stats['blink_count'],
        stats['yawn_count'],
        stats['perclos'] * 100
    ]
    
    bars = axes[1, 0].bar(stat_names, stat_values, color=['skyblue', 'lightgreen', 'salmon'])
    axes[1, 0].set_title('Statistiques de Fatigue')
    axes[1, 0].set_ylabel('Valeur')
    
    # Ajouter les valeurs sur les barres
    for bar, val in zip(bars, stat_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom')
    
    # État actuel
    status_text = f"""
    ÉTAT ACTUEL:
    
    Yeux: {'FERMÉS' if stats['is_eyes_closed'] else 'OUVERTS'}
    Bâillement: {'OUI' if stats['is_yawning'] else 'NON'}
    
    EAR: {stats['current_ear']:.3f}
    MAR: {stats['current_mar']:.3f}
    PERCLOS: {stats['perclos']*100:.1f}%
    
    ALERTE: {'⚠️ FATIGUE DÉTECTÉE' if stats['perclos'] > 0.15 else '✅ NORMAL'}
    """
    
    axes[1, 1].text(0.1, 0.5, status_text, fontsize=12, 
                   verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Statut Actuel')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualisation sauvegardée: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test des métriques
    print("Test des métriques de fatigue...")
    
    # Test EAR
    eye_open = [(30, 30), (28, 25), (32, 25), (30, 30), (28, 35), (32, 35)]
    eye_closed = [(30, 30), (29, 28), (31, 28), (30, 30), (29, 32), (31, 32)]
    
    ear_open = calculate_ear(eye_open)
    ear_closed = calculate_ear(eye_closed)
    
    print(f"EAR œil ouvert: {ear_open:.3f}")
    print(f"EAR œil fermé: {ear_closed:.3f}")
    
    # Test FatigueMetrics
    tracker = FatigueMetrics()
    
    for i in range(100):
        if i < 50:
            ear = 0.3  # Yeux ouverts
        else:
            ear = 0.15  # Yeux fermés
        
        alerts = tracker.update(ear, mar=0.3)
        if any(alerts.values()):
            print(f"Frame {i}: Alertes = {alerts}")
    
    print("\nStatistiques finales:")
    print(tracker.get_statistics())
    
    print("\nMétriques de modèle testées avec succès!")
