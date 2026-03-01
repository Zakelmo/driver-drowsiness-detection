"""
Application Streamlit pour la Détection de Somnolence.

Interface web interactive permettant:
- Upload d'images/vidéos pour analyse
- Détection en temps réel via webcam
- Visualisation des métriques
- Historique des alertes
"""

import os
import sys
import io
import base64
import tempfile
from datetime import datetime

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
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #FF6B6B;
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
        
        # Mode d'analyse
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
        
        return ear_threshold, mar_threshold, analysis_mode


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
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        with col2:
            st.subheader("Résultat de l'Analyse")
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
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


def render_video_analysis(ear_threshold: float, mar_threshold: float):
    """Affiche l'analyse vidéo."""
    st.header("🎥 Analyse Vidéo")
    
    uploaded_file = st.file_uploader(
        "Choisissez une vidéo",
        type=['mp4', 'avi', 'mov'],
        help="Téléchargez une vidéo pour analyse"
    )
    
    if uploaded_file is not None:
        st.info("🎬 Traitement de la vidéo...")
        
        # Sauvegarde temporaire
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        # Affichage de la vidéo originale
        st.video(tfile.name)
        
        st.warning("""
        💡 **Note:** Le traitement vidéo complet peut prendre du temps.
        Pour une démonstration en temps réel, utilisez le mode Webcam.
        """)


def render_webcam_analysis(ear_threshold: float, mar_threshold: float):
    """Affiche l'analyse webcam en temps réel."""
    st.header("📹 Détection en Temps Réel")
    
    st.info("""
    📸 **Mode Webcam**
    
    Cliquez sur "Démarrer la caméra" pour lancer la détection en temps réel.
    Assurez-vous que votre webcam est connectée.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bouton pour démarrer
        run_camera = st.checkbox("▶️ Démarrer la caméra")
        
        if run_camera:
            # Placeholder pour la vidéo
            frame_placeholder = st.empty()
            
            # Initialisation
            landmark_extractor = LandmarkExtractor(static_image_mode=False)
            fatigue_metrics = FatigueMetrics(
                ear_threshold=ear_threshold,
                mar_threshold=mar_threshold
            )
            
            # Ouverture de la caméra
            cap = cv2.VideoCapture(0)
            
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
                    
                    # Affichage
                    frame_placeholder.image(
                        cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_column_width=True
                    )
                    
                    # Mise à jour des métriques
                    if face_detected:
                        st.session_state['current_ear'] = ear
                        st.session_state['current_mar'] = mar
                        st.session_state['current_alerts'] = alerts
            
            finally:
                cap.release()
                landmark_extractor.close()
    
    with col2:
        st.subheader("📊 Métriques en Direct")
        
        # Métriques actuelles
        if 'current_ear' in st.session_state:
            ear = st.session_state['current_ear']
            mar = st.session_state['current_mar']
            alerts = st.session_state['current_alerts']
            
            # Jauges
            st.write("**Eye Aspect Ratio (EAR)**")
            st.progress(min(ear / 0.5, 1.0))
            st.write(f"Valeur: {ear:.3f}")
            
            st.write("**Mouth Aspect Ratio (MAR)**")
            st.progress(min(mar / 1.0, 1.0))
            st.write(f"Valeur: {mar:.3f}")
            
            # État
            if any(alerts.values()):
                st.error("⚠️ **ALERTE!**")
            else:
                st.success("✅ **Normal**")
        else:
            st.write("*En attente de données...*")
        
        # Instructions
        st.markdown("---")
        st.subheader("🎮 Contrôles")
        st.write("- Décochez pour arrêter")
        st.write("- Positionnez-vous face à la caméra")
        st.write("- Assurez un éclairage adéquat")


def render_about():
    """Affiche la page à propos."""
    st.header("📚 À Propos du Projet")
    
    st.markdown("""
    ## Vision par Ordinateur et Deep Learning
    
    Ce projet implémente un système de **détection de somnolence du conducteur**
    utilisant des techniques de Deep Learning et de vision par ordinateur.
    
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
    
    ### 🔧 Architecture du Système
    
    ```
    Webcam → Détection Visage → Landmarks (MediaPipe)
                                     ↓
    ┌─────────────────────────────────────────────────────┐
    │  EAR (Eye Aspect Ratio)                             │
    │  MAR (Mouth Aspect Ratio)                           │
    │  PERCLOS (% fermeture des yeux)                     │
    └─────────────────────────────────────────────────────┘
                     ↓
            Classification Fatigue
                     ↓
               Alerte Conducteur
    ```
    
    ### 📊 Métriques Utilisées
    
    | Métrique | Description | Seuil |
    |----------|-------------|-------|
    | EAR | Ratio d'aspect de l'œil | < 0.25 |
    | MAR | Ratio d'aspect de la bouche | > 0.6 |
    | PERCLOS | % temps yeux fermés | > 15% |
    
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
    ear_threshold, mar_threshold, analysis_mode = render_sidebar()
    
    # Contenu principal
    if analysis_mode == "📷 Image":
        render_image_analysis(ear_threshold, mar_threshold)
    
    elif analysis_mode == "🎥 Vidéo":
        render_video_analysis(ear_threshold, mar_threshold)
    
    elif analysis_mode == "📹 Webcam":
        render_webcam_analysis(ear_threshold, mar_threshold)
    
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
