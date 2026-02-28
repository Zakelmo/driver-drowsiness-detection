"""
Module de gestion des alertes pour le système de détection de somnolence.

Fournit des mécanismes d'alerte sonore et visuelle lorsque 
la fatigue du conducteur est détectée.
"""

import os
import time
import threading
from typing import Optional
from datetime import datetime


class AlertSystem:
    """
    Système d'alertes pour la détection de somnolence.
    
    Gère les alertes sonores (bips, messages vocaux) et visuelles
    (clignotement, notifications) pour alerter le conducteur.
    """
    
    def __init__(self, 
                 sound_enabled: bool = True,
                 visual_enabled: bool = True,
                 cooldown_period: int = 5,
                 alert_sound_file: str = None):
        """
        Initialise le système d'alertes.
        
        Args:
            sound_enabled: Activer les alertes sonores
            visual_enabled: Activer les alertes visuelles
            cooldown_period: Période minimale entre deux alertes (secondes)
            alert_sound_file: Fichier audio personnalisé pour l'alerte
        """
        self.sound_enabled = sound_enabled
        self.visual_enabled = visual_enabled
        self.cooldown_period = cooldown_period
        self.alert_sound_file = alert_sound_file
        
        self.last_alert_time = 0
        self.is_alerting = False
        self.alert_thread = None
        
        # Initialisation audio
        if self.sound_enabled:
            try:
                import pygame
                pygame.mixer.init()
                self.pygame = pygame
                self.audio_initialized = True
            except Exception as e:
                print(f"Warning: Impossible d'initialiser l'audio: {e}")
                self.audio_initialized = False
                self.sound_enabled = False
    
    def can_alert(self) -> bool:
        """
        Vérifie si une alerte peut être déclenchée (respect du cooldown).
        
        Returns:
            True si une alerte peut être déclenchée
        """
        current_time = time.time()
        return (current_time - self.last_alert_time) >= self.cooldown_period
    
    def trigger_alert(self, alert_type: str = "drowsiness", 
                     message: str = None) -> bool:
        """
        Déclenche une alerte si les conditions sont remplies.
        
        Args:
            alert_type: Type d'alerte ('drowsiness', 'eye_closure', 'yawn')
            message: Message personnalisé pour l'alerte
            
        Returns:
            True si l'alerte a été déclenchée
        """
        if not self.can_alert():
            return False
        
        self.last_alert_time = time.time()
        self.is_alerting = True
        
        # Messages par défaut
        default_messages = {
            'drowsiness': 'ALERTE FATIGUE ! Reposez-vous immédiatement !',
            'eye_closure': 'Yeux fermés détectés !',
            'yawn': 'Bâillement détecté ! Attention à la fatigue.',
            'perclos': 'Niveau de fermeture des yeux élevé !'
        }
        
        message = message or default_messages.get(alert_type, 'Alerte de fatigue !')
        
        # Déclenchement des alertes
        if self.sound_enabled and self.audio_initialized:
            self._play_sound_alert(alert_type)
        
        print(f"\n{'='*60}")
        print(f"⚠️  {message}")
        print(f"{'='*60}\n")
        
        return True
    
    def _play_sound_alert(self, alert_type: str):
        """
        Joue une alerte sonore.
        
        Args:
            alert_type: Type d'alerte
        """
        try:
            if self.alert_sound_file and os.path.exists(self.alert_sound_file):
                # Utiliser le fichier personnalisé
                self.pygame.mixer.music.load(self.alert_sound_file)
                self.pygame.mixer.music.play()
            else:
                # Générer un bip d'alerte
                self._generate_beep()
        except Exception as e:
            print(f"Erreur lors de la lecture du son: {e}")
    
    def _generate_beep(self):
        """
        Génère une série de bips d'alerte.
        """
        def beep_sequence():
            try:
                import winsound
                for _ in range(3):
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                    time.sleep(0.5)
            except:
                # Fallback: imprimer des caractères de cloche
                print('\a')  # Caractère de cloche système
        
        # Lancer dans un thread séparé pour ne pas bloquer
        threading.Thread(target=beep_sequence, daemon=True).start()
    
    def stop_alert(self):
        """Arrête l'alerte en cours."""
        self.is_alerting = False
        
        if self.sound_enabled and self.audio_initialized:
            try:
                self.pygame.mixer.music.stop()
            except:
                pass
    
    def get_visual_alert_frame(self, frame, alert_type: str = "drowsiness"):
        """
        Ajoute un overlay visuel d'alerte sur une frame vidéo.
        
        Args:
            frame: Frame vidéo (numpy array)
            alert_type: Type d'alerte
            
        Returns:
            Frame avec overlay d'alerte
        """
        import cv2
        
        if not self.visual_enabled or not self.is_alerting:
            return frame
        
        h, w = frame.shape[:2]
        
        # Couleurs selon le type d'alerte
        colors = {
            'drowsiness': (0, 0, 255),    # Rouge
            'eye_closure': (0, 165, 255),  # Orange
            'yawn': (0, 255, 255),         # Jaune
            'perclos': (255, 0, 0)         # Bleu
        }
        
        color = colors.get(alert_type, (0, 0, 255))
        
        # Bordure clignotante
        border_thickness = 20
        cv2.rectangle(frame, (0, 0), (w, h), color, border_thickness)
        
        # Message d'alerte
        message = "ALERTE FATIGUE!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Position centrée en haut
        text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 60
        
        # Fond du texte
        cv2.rectangle(frame, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     color, -1)
        
        # Texte blanc
        cv2.putText(frame, message, (text_x, text_y),
                   font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def log_alert(self, alert_type: str, details: dict = None):
        """
        Enregistre une alerte dans un fichier de log.
        
        Args:
            alert_type: Type d'alerte
            details: Détails supplémentaires
        """
        log_file = "reports/alerts_log.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] ALERTE: {alert_type}\n")
            if details:
                for key, value in details.items():
                    f.write(f"  {key}: {value}\n")
            f.write("-" * 40 + "\n")


def test_alert_system():
    """Teste le système d'alertes."""
    print("Test du système d'alertes...")
    
    alert_system = AlertSystem()
    
    print("Test alerte fatigue...")
    alert_system.trigger_alert('drowsiness')
    time.sleep(2)
    
    print("Test alerte yeux fermés...")
    alert_system.trigger_alert('eye_closure')
    time.sleep(2)
    
    print("Test alerte bâillement...")
    alert_system.trigger_alert('yawn')
    
    print("Test terminé!")


if __name__ == "__main__":
    test_alert_system()
