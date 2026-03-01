"""
Application Streamlit pour la Détection de Somnolence.

Interface web interactive permettant:
- Upload d'images/vidéos pour analyse
- Détection en temps réel via webcam avec alertes audio/visuelles
- Visualisation des métriques
- Historique des alertes
"""

import os
import sys
import io
import base64
import tempfile
import time
from datetime import datetime
from collections import deque

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Ajout du path source
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from detection.landmark_extractor import LandmarkExtractor
from utils.preprocessing import ImagePreprocessor
from utils.metrics import FatigueMetrics, calculate_ear, calculate_mar

# Import conditionnel des modèles (évite erreurs si TensorFlow non installé)
try:
    from models.cnn import EyeCNN
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    EyeCNN = None

# Import pour les alertes audio
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# Configuration de la page
st.set_page_config(
    page_title="Détection de Somnolence | Driver Drowsiness",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .alert-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(244, 67, 54, 0); }
        100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0); }
    }
    .stProgress > div > div > div > div {
        background-color: #FF6B6B;
    }
    .alert-badge {
        background-color: #f44336;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


def get_image_download_link(img, filename, text):
    """Génère un lien de téléchargement pour une image."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def process_image(image: np.ndarray, 
                  landmark_extractor: LandmarkExtractor,
                  fatigue_metrics: FatigueMetrics) -> tuple:
    """
    Traite une image et retourne les résultats.
    
    Returns:
        tuple: (image_annotée, ear, mar, alerts, face_detected)
    """
    result_image = image.copy()
    
    # Détection
    face_detected = landmark_extractor.process(image)
    
    ear = 0.0
    mar = 0.0
    alerts = {}
    
    if face_detected:
        # Extraction des landmarks
        left_eye_pts, right_eye_pts = landmark_extractor.get_eye_landmarks(image.shape)
        mouth_pts = landmark_extractor.get_mouth_landmarks(image.shape)
        
        # Calcul EAR
        if left_eye_pts and right_eye_pts:
            ear_left = calculate_ear(left_eye_pts)
            ear_right = calculate_ear(right_eye_pts)
            ear = (ear_left + ear_right) / 2
        
        # Calcul MAR
        if mouth_pts:
            mar = calculate_mar(mouth_pts)
        
        # Mise à jour des métriques
        alerts = fatigue_metrics.update(ear, mar)
        
        # Dessin des landmarks
        result_image = landmark_extractor.draw_landmarks(image)
    
    return result_image, ear, mar, alerts, face_detected


def add_alert_overlay(frame: np.ndarray, alert_type: str, alert_count: int) -> np.ndarray:
    """
    Ajoute un overlay visuel d'alerte sur la frame.
    
    Args:
        frame: Frame vidéo
        alert_type: Type d'alerte ('drowsiness', 'eye_closure', 'yawn')
        alert_count: Numéro de l'alerte
        
    Returns:
        Frame avec overlay
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Couleur selon le type
    if alert_type == 'drowsiness':
        color = (0, 0, 255)  # Rouge
        message = "ALERTE FATIGUE!"
        sub_message = f"Alerte #{alert_count} - Reposez-vous!"
    elif alert_type == 'eye_closure':
        color = (0, 165, 255)  # Orange
        message = "YEUX FERMES!"
        sub_message = f"Alerte #{alert_count}"
    elif alert_type == 'yawn':
        color = (0, 255, 255)  # Jaune
        message = "BAILLEMENT DETECTE!"
        sub_message = f"Alerte #{alert_count}"
    else:
        color = (255, 0, 0)
        message = "ALERTE!"
        sub_message = f"Alerte #{alert_count}"
    
    # Bordure clignotante (épaisseur variable selon le temps)
    border_thickness = 20 + (int(time.time() * 5) % 10)
    cv2.rectangle(overlay, (0, 0), (w, h), color, border_thickness)
    
    # Bandeau en haut
    cv2.rectangle(overlay, (0, 0), (w, 80), color, -1)
    
    # Texte principal
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 1.2, 3)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(overlay, message, (text_x, 50), font, 1.2, (255, 255, 255), 3)
    
    # Texte secondaire
    text_size2 = cv2.getTextSize(sub_message, font, 0.7, 2)[0]
    text_x2 = (w - text_size2[0]) // 2
    cv2.putText(overlay, sub_message, (text_x2, h - 30), font, 0.7, color, 2)
    
    # Icône/Emoji simulation (cercle avec point d'exclamation)
    center = (w - 60, 40)
    cv2.circle(overlay, center, 25, (255, 255, 255), -1)
    cv2.circle(overlay, center, 25, color, 3)
    cv2.putText(overlay, "!", (w - 70, 50), font, 1.5, color, 3)
    
    # Effet de transparence pour le bandeau
    alpha = 0.9
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return overlay


def play_alert_sound():
    """Joue un son d'alerte si pygame est disponible."""
    if PYGAME_AVAILABLE:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            # Créer un bip sonore simple
            frequency = 800  # Hz
            duration = 500   # ms
            
            # Générer un son de bip
            sample_rate = 44100
            t = np.linspace(0, duration/1000, int(sample_rate * duration/1000))
            wave = np.sin(2 * np.pi * frequency * t) * 0.5
            
            # Convertir en format pygame
            sound = pygame.sndarray.make_sound((wave * 32767).astype(np.int16))
            sound.play()
            
        except Exception as e:
            pass  # Silencieux en cas d'erreur audio


def render_header():
    """Affiche l'en-tête de l'application."""
    st.markdown('<p class="main-header">🚗 Détection de Somnolence</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Système de surveillance du conducteur par Deep Learning</p>', 
                unsafe_allow_html=True)


def render_sidebar():
    """Affiche la barre latérale."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Seuils
        st.subheader("Seuils de détection")
        ear_threshold = st.slider("Seuil EAR", 0.1, 0.4, 0.25, 0.01,
                                  help="Eye Aspect Ratio threshold")
        mar_threshold = st.slider("Seuil MAR", 0.3, 1.0, 0.6, 0.05,
                                  help="Mouth Aspect Ratio threshold")
        
        # Alertes
        st.subheader("🔔 Configuration des Alertes")
        audio_alerts = st.checkbox("Activer les alertes audio", value=True,
                                   help="Émettre un bip sonore lors des alertes")
        visual_alerts = st.checkbox("Activer les alertes visuelles", value=True,
                                    help="Afficher des bordures et messages d'alerte")
        
        # Mode d'analyse
        st.markdown("---")
        st.subheader("Mode d'analyse")
        analysis_mode = st.radio(
            "Sélectionnez le mode:",
            ["📷 Image", "🎥 Vidéo", "📹 Webcam"]
        )
        
        # Informations
        st.markdown("---")
        st.subheader("ℹ️ Informations")
        st.markdown("""
        **Métriques:**
        - **EAR**: Eye Aspect Ratio
        - **MAR**: Mouth Aspect Ratio
        - **PERCLOS**: % fermeture des yeux
        
        **Auteur:** SDIA Student
        **Version:** 1.0.0
        """)
        
        return ear_threshold, mar_threshold, analysis_mode, audio_alerts, visual_alerts


def render_image_analysis(ear_threshold: float, mar_threshold: float):
    """Affiche l'analyse d'image."""
    st.header("📷 Analyse d'Image")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image",
        type=['png', 'jpg', 'jpeg'],
        help="Téléchargez une image contenant un visage"
    )
    
    if uploaded_file is not None:
        # Chargement de l'image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Initialisation
        landmark_extractor = LandmarkExtractor(static_image_mode=True)
        fatigue_metrics = FatigueMetrics(
            ear_threshold=ear_threshold,
            mar_threshold=mar_threshold
        )
        
        # Traitement
        with st.spinner("Analyse en cours..."):
            result_image, ear, mar, alerts, face_detected = process_image(
                image, landmark_extractor, fatigue_metrics
            )
        
        # Affichage des résultats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image Originale")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col2:
            st.subheader("Résultat de l'Analyse")
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Métriques
        if face_detected:
            st.markdown("---")
            st.subheader("📊 Métriques de Fatigue")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                ear_color = "normal" if ear > ear_threshold else "off"
                st.metric(
                    "Eye Aspect Ratio (EAR)",
                    f"{ear:.3f}",
                    delta="Ouvert" if ear > ear_threshold else "Fermé",
                    delta_color=ear_color
                )
            
            with metrics_col2:
                mar_color = "off" if mar > mar_threshold else "normal"
                st.metric(
                    "Mouth Aspect Ratio (MAR)",
                    f"{mar:.3f}",
                    delta="Bâillement" if mar > mar_threshold else "Normal",
                    delta_color=mar_color
                )
            
            with metrics_col3:
                stats = fatigue_metrics.get_statistics()
                perclos = stats['perclos']
                st.metric(
                    "PERCLOS",
                    f"{perclos*100:.1f}%",
                    delta="Alerte" if perclos > 0.15 else "OK",
                    delta_color="off" if perclos > 0.15 else "normal"
                )
            
            # Alertes
            if any(alerts.values()):
                st.markdown('<div class="alert-box">', unsafe_allow_html=True)
                st.error("⚠️ **ALERTE: Fatigue détectée!**")
                if alerts['eye_closure']:
                    st.write("- 🚫 Yeux fermés prolongés")
                if alerts['yawn']:
                    st.write("- 🥱 Bâillement détecté")
                if alerts['drowsiness_alert']:
                    st.write("- ⚡ Risque de somnolence élevé")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("✅ État normal - Aucune fatigue détectée")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Aucun visage détecté dans l'image")


def render_webcam_analysis(ear_threshold: float, mar_threshold: float, 
                          audio_alerts: bool, visual_alerts: bool):
    """Affiche l'analyse webcam en temps réel avec alertes."""
    st.header("📹 Détection en Temps Réel")
    
    st.info("""
    📸 **Mode Webcam avec Alertes**
    
    - Les **alertes visuelles** apparaissent comme des bordures rouges clignotantes
    - Les **alertes audio** émettent un bip sonore
    - Les alertes se déclenchent automatiquement en cas de:
      - Yeux fermés > 2 secondes
      - Bâillements détectés
      - Niveau de fatigue élevé (PERCLOS)
    """)
    
    # Initialisation des variables de session pour les alertes
    if 'alert_count' not in st.session_state:
        st.session_state['alert_count'] = 0
    if 'alert_history' not in st.session_state:
        st.session_state['alert_history'] = deque(maxlen=10)
    if 'last_alert_time' not in st.session_state:
        st.session_state['last_alert_time'] = 0
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bouton pour démarrer
        run_camera = st.checkbox("▶️ Démarrer la caméra")
        
        if run_camera:
            # Placeholders pour la vidéo et les alertes
            frame_placeholder = st.empty()
            alert_placeholder = st.empty()
            
            # Initialisation
            landmark_extractor = LandmarkExtractor(static_image_mode=False)
            fatigue_metrics = FatigueMetrics(
                ear_threshold=ear_threshold,
                mar_threshold=mar_threshold
            )
            
            # Ouverture de la caméra
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Zone d'alerte persistante
            alert_status = st.empty()
            
            try:
                while run_camera:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("❌ Erreur: Impossible d'accéder à la webcam")
                        break
                    
                    # Traitement
                    result_frame, ear, mar, alerts, face_detected = process_image(
                        frame, landmark_extractor, fatigue_metrics
                    )
                    
                    # Gestion des alertes
                    current_time = time.time()
                    alert_triggered = False
                    alert_type = None
                    
                    # Vérifier si une alerte doit être déclenchée
                    if alerts.get('drowsiness_alert') or alerts.get('eye_closure'):
                        # Éviter les alertes trop fréquentes (cooldown 3 secondes)
                        if current_time - st.session_state['last_alert_time'] > 3:
                            st.session_state['alert_count'] += 1
                            st.session_state['last_alert_time'] = current_time
                            alert_triggered = True
                            
                            if alerts.get('drowsiness_alert'):
                                alert_type = 'drowsiness'
                            else:
                                alert_type = 'eye_closure'
                            
                            # Ajouter à l'historique
                            st.session_state['alert_history'].append({
                                'time': datetime.now().strftime("%H:%M:%S"),
                                'type': alert_type,
                                'ear': ear,
                                'mar': mar
                            })
                            
                            # Jouer le son d'alerte
                            if audio_alerts:
                                play_alert_sound()
                    
                    elif alerts.get('yawn'):
                        if current_time - st.session_state['last_alert_time'] > 3:
                            st.session_state['alert_count'] += 1
                            st.session_state['last_alert_time'] = current_time
                            alert_triggered = True
                            alert_type = 'yawn'
                            
                            st.session_state['alert_history'].append({
                                'time': datetime.now().strftime("%H:%M:%S"),
                                'type': 'yawn',
                                'ear': ear,
                                'mar': mar
                            })
                            
                            if audio_alerts:
                                play_alert_sound()
                    
                    # Ajouter l'overlay visuel si alerte active
                    if alert_triggered and visual_alerts:
                        result_frame = add_alert_overlay(
                            result_frame, 
                            alert_type, 
                            st.session_state['alert_count']
                        )
                    
                    # Affichage de la frame
                    frame_placeholder.image(
                        cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )
                    
                    # Mise à jour du statut d'alerte
                    if alert_triggered:
                        alert_status.markdown(f"""
                        <div class="alert-box">
                            <h3>🚨 ALERTE ACTIVE 🚨</h3>
                            <p>Type: {alert_type.upper()}</p>
                            <p>Numéro d'alerte: #{st.session_state['alert_count']}</p>
                            <p>Heure: {datetime.now().strftime("%H:%M:%S")}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif any(alerts.values()):
                        alert_status.warning("⚠️ Signes de fatigue détectés - Surveillance active")
                    else:
                        alert_status.success("✅ État normal")
                    
                    # Mise à jour des métriques dans session_state
                    if face_detected:
                        st.session_state['current_ear'] = ear
                        st.session_state['current_mar'] = mar
                        st.session_state['current_alerts'] = alerts
                    
                    # Petit délai pour ne pas surcharger le CPU
                    time.sleep(0.03)  # ~30 FPS
            
            finally:
                cap.release()
                landmark_extractor.close()
                alert_status.empty()
    
    with col2:
        st.subheader("📊 Métriques en Direct")
        
        # Métriques actuelles
        if 'current_ear' in st.session_state:
            ear = st.session_state['current_ear']
            mar = st.session_state['current_mar']
            alerts = st.session_state['current_alerts']
            
            # EAR
            st.write("**Eye Aspect Ratio (EAR)**")
            ear_color = "#4caf50" if ear > ear_threshold else "#f44336"
            st.progress(min(ear / 0.5, 1.0))
            st.markdown(f"<p style='color: {ear_color}; font-weight: bold;'>{ear:.3f}</p>", 
                       unsafe_allow_html=True)
            
            # MAR
            st.write("**Mouth Aspect Ratio (MAR)**")
            mar_color = "#f44336" if mar > mar_threshold else "#4caf50"
            st.progress(min(mar / 1.0, 1.0))
            st.markdown(f"<p style='color: {mar_color}; font-weight: bold;'>{mar:.3f}</p>", 
                       unsafe_allow_html=True)
            
            # État
            if any(alerts.values()):
                st.error("⚠️ **ALERTE!**")
            else:
                st.success("✅ **Normal**")
        else:
            st.info("*En attente de données...*")
        
        st.markdown("---")
        
        # Compteur d'alertes
        st.subheader("🔔 Statistiques d'Alertes")
        alert_count = st.session_state.get('alert_count', 0)
        st.markdown(f"<p class='metric-value'>Total Alertes: <span class='alert-badge'>{alert_count}</span></p>", 
                   unsafe_allow_html=True)
        
        # Historique des alertes
        if st.session_state.get('alert_history'):
            st.write("**Historique récent:**")
            for alert in list(st.session_state['alert_history'])[-5:]:
                alert_icon = "🚨" if alert['type'] == 'drowsiness' else "👁️" if alert['type'] == 'eye_closure' else "🥱"
                st.write(f"{alert_icon} {alert['time']} - {alert['type'].upper()}")
        
        st.markdown("---")
        
        # Instructions
        st.subheader("🎮 Contrôles")
        st.write("- Décochez pour arrêter")
        st.write("- Positionnez-vous face à la caméra")
        st.write("- Assurez un éclairage adéquat")


def render_about():
    """Affiche la page à propos."""
    st.header("📚 À Propos du Projet")
    
    st.markdown("""
    ## Vision par Ordinateur et Deep Learning
    
    Ce projet implémente un système complet de **Détection de Somnolence du Conducteur (DDS)**
    utilisant des techniques avancées de vision par ordinateur et de deep learning.
    
    ### 🧠 Concepts du Cours Appliqués
    
    **Chapitre 1 - Fondamentaux:**
    - Perceptron et classification binaire
    - Fonction sigmoïde et activation
    - Descente de gradient
    - Fonction de perte (Binary Cross-Entropy)
    
    **Chapitre 2 - PMC:**
    - Réseaux multi-couches
    - Forward et Backward propagation
    - Régularisation (Dropout)
    
    **Chapitre 3-4 - CNN:**
    - Convolution et extraction de features
    - Max Pooling
    - Transfer Learning (MobileNetV2)
    - Data Augmentation
    
    ### 🔔 Système d'Alertes
    
    Le système détecte plusieurs signes de fatigue:
    - **Yeux fermés**: EAR (Eye Aspect Ratio) < 0.25 pendant > 2s
    - **Bâillements**: MAR (Mouth Aspect Ratio) > 0.6
    - **PERCLOS**: % de fermeture des yeux sur une fenêtre temporelle
    
    ### 🛠️ Technologies
    
    - **TensorFlow/Keras**: Deep Learning
    - **OpenCV**: Vision par ordinateur
    - **MediaPipe**: Détection faciale
    - **Streamlit**: Interface web
    - **NumPy/Pandas**: Traitement de données
    """)


def main():
    """Point d'entrée principal."""
    # En-tête
    render_header()
    
    # Barre latérale
    ear_threshold, mar_threshold, analysis_mode, audio_alerts, visual_alerts = render_sidebar()
    
    # Contenu principal
    if analysis_mode == "📷 Image":
        render_image_analysis(ear_threshold, mar_threshold)
    
    elif analysis_mode == "🎥 Vidéo":
        st.header("🎥 Analyse Vidéo")
        st.info("Fonctionnalité en cours de développement. Utilisez le mode Webcam pour le moment.")
    
    elif analysis_mode == "📹 Webcam":
        render_webcam_analysis(ear_threshold, mar_threshold, audio_alerts, visual_alerts)
    
    # Pied de page
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>"
        "Développé pour le cours de Deep Learning - SDIA | 2024"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
