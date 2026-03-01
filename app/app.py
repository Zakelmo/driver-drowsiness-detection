"""
Application Streamlit pour la Détection de Somnolence.

Interface web interactive permettant:
- Upload d'images/vidéos pour analyse
- Détection en temps réel via webcam avec alertes audio/visuelles modernes
- Visualisation des métriques en temps réel
- Système d'alertes dynamique et réactif
"""

import os
import sys
import io
import base64
import tempfile
import time
import threading
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

# Import conditionnel des modèles
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

# Styles CSS modernes avec animations avancées
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Alert Container */
    .alert-container {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Pulsing Ring Animation */
    @keyframes pulse-ring {
        0% { transform: scale(0.8); opacity: 1; }
        100% { transform: scale(2); opacity: 0; }
    }
    
    .pulse-ring {
        position: absolute;
        border-radius: 50%;
        animation: pulse-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    }
    
    /* Critical Alert */
    .alert-critical {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(255, 65, 108, 0.4);
        animation: slide-in 0.5s ease-out, shake 0.5s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .alert-critical::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent 30%,
            rgba(255,255,255,0.1) 50%,
            transparent 70%
        );
        animation: shimmer 2s infinite;
    }
    
    /* Warning Alert */
    .alert-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(245, 87, 108, 0.3);
        animation: slide-in 0.3s ease-out;
    }
    
    /* Success State */
    .status-safe {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.3);
    }
    
    @keyframes slide-in {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    
    .blinking {
        animation: blink 1s infinite;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* Live Indicator */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #ff416c;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: white;
        border-radius: 50%;
        animation: pulse-live 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse-live {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    /* Alert History Item */
    .alert-history-item {
        background: white;
        border-left: 4px solid #ff416c;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        font-size: 0.9rem;
    }
    
    /* Progress Bar Custom */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 10px;
    }
    
    /* Frame Overlay */
    .frame-overlay {
        position: relative;
    }
    
    .frame-alert-border {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border: 8px solid #ff416c;
        border-radius: 12px;
        animation: border-pulse 0.5s ease-in-out infinite;
        pointer-events: none;
        z-index: 10;
    }
    
    @keyframes border-pulse {
        0%, 100% { border-width: 8px; opacity: 1; }
        50% { border-width: 12px; opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)


def get_audio_html():
    """
    Retourne le HTML/JS pour les alertes audio du navigateur.
    Utilise l'API Web Audio pour générer des sons sans fichiers externes.
    """
    return """
    <script>
    // Système d'alerte audio utilisant Web Audio API
    window.audioContext = null;
    window.lastAlertTime = 0;
    
    function initAudio() {
        if (!window.audioContext) {
            window.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
    }
    
    function playAlertBeep() {
        initAudio();
        const now = Date.now();
        if (now - window.lastAlertTime < 3000) return; // Cooldown 3s
        window.lastAlertTime = now;
        
        const ctx = window.audioContext;
        
        // Oscillateur principal (fréquence décroissante pour effet alarme)
        const osc1 = ctx.createOscillator();
        const gain1 = ctx.createGain();
        
        osc1.type = 'square';
        osc1.frequency.setValueAtTime(880, ctx.currentTime);
        osc1.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.3);
        
        gain1.gain.setValueAtTime(0.3, ctx.currentTime);
        gain1.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
        
        osc1.connect(gain1);
        gain1.connect(ctx.destination);
        
        osc1.start(ctx.currentTime);
        osc1.stop(ctx.currentTime + 0.3);
        
        // Deuxième bip (plus aigu)
        setTimeout(() => {
            const osc2 = ctx.createOscillator();
            const gain2 = ctx.createGain();
            
            osc2.type = 'square';
            osc2.frequency.setValueAtTime(1200, ctx.currentTime);
            osc2.frequency.exponentialRampToValueAtTime(600, ctx.currentTime + 0.3);
            
            gain2.gain.setValueAtTime(0.3, ctx.currentTime);
            gain2.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
            
            osc2.connect(gain2);
            gain2.connect(ctx.destination);
            
            osc2.start(ctx.currentTime);
            osc2.stop(ctx.currentTime + 0.3);
        }, 150);
        
        // Troisième bip (grave)
        setTimeout(() => {
            const osc3 = ctx.createOscillator();
            const gain3 = ctx.createGain();
            
            osc3.type = 'sawtooth';
            osc3.frequency.setValueAtTime(600, ctx.currentTime);
            osc3.frequency.exponentialRampToValueAtTime(300, ctx.currentTime + 0.5);
            
            gain3.gain.setValueAtTime(0.4, ctx.currentTime);
            gain3.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
            
            osc3.connect(gain3);
            gain3.connect(ctx.destination);
            
            osc3.start(ctx.currentTime);
            osc3.stop(ctx.currentTime + 0.5);
        }, 300);
    }
    
    // Initialiser l'audio au premier clic de l'utilisateur
    document.addEventListener('click', initAudio, { once: true });
    </script>
    """


def trigger_browser_alert():
    """Déclenche l'alerte audio dans le navigateur."""
    st.components.v1.html(
        """
        <script>
        if (window.playAlertBeep) {
            window.playAlertBeep();
        }
        </script>
        """,
        height=0
    )


def process_image(image: np.ndarray, 
                  landmark_extractor: LandmarkExtractor,
                  fatigue_metrics: FatigueMetrics) -> tuple:
    """Traite une image et retourne les résultats."""
    result_image = image.copy()
    
    face_detected = landmark_extractor.process(image)
    
    ear = 0.0
    mar = 0.0
    alerts = {}
    
    if face_detected:
        left_eye_pts, right_eye_pts = landmark_extractor.get_eye_landmarks(image.shape)
        mouth_pts = landmark_extractor.get_mouth_landmarks(image.shape)
        
        if left_eye_pts and right_eye_pts:
            ear_left = calculate_ear(left_eye_pts)
            ear_right = calculate_ear(right_eye_pts)
            ear = (ear_left + ear_right) / 2
        
        if mouth_pts:
            mar = calculate_mar(mouth_pts)
        
        alerts = fatigue_metrics.update(ear, mar)
        result_image = landmark_extractor.draw_landmarks(image)
    
    return result_image, ear, mar, alerts, face_detected


def add_modern_alert_overlay(frame: np.ndarray, alert_type: str, 
                             alert_count: int, intensity: float = 1.0) -> np.ndarray:
    """
    Ajoute un overlay d'alerte moderne et dynamique.
    
    Args:
        frame: Frame vidéo
        alert_type: Type d'alerte
        alert_count: Numéro de l'alerte
        intensity: Intensité de l'alerte (0-1)
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Couleurs dynamiques selon le type
    if alert_type == 'drowsiness':
        color = (0, 0, 255)  # Rouge
        title = "DANGER"
        subtitle = "Somnolence Détectée"
        icon = "⚠️"
    elif alert_type == 'eye_closure':
        color = (0, 165, 255)  # Orange
        title = "ATTENTION"
        subtitle = "Yeux Fermés"
        icon = "👁️"
    elif alert_type == 'yawn':
        color = (0, 255, 255)  # Jaune
        title = "ALERTE"
        subtitle = "Bâillement Détecté"
        icon = "🥱"
    else:
        color = (255, 0, 0)
        title = "ALERTE"
        subtitle = "Fatigue"
        icon = "⚡"
    
    # Bordure pulsante animée
    border_thickness = int(8 + 4 * np.sin(time.time() * 10))
    alpha = 0.5 + 0.3 * np.sin(time.time() * 8)
    
    # Dessiner plusieurs bordures pour effet de profondeur
    for i in range(3):
        thickness = border_thickness + i * 4
        color_faded = tuple(int(c * (0.6 - i * 0.15)) for c in color)
        cv2.rectangle(overlay, (thickness, thickness), 
                     (w - thickness, h - thickness), color_faded, 2)
    
    # Bordure principale
    cv2.rectangle(overlay, (0, 0), (w, h), color, border_thickness)
    
    # Bandeau supérieur avec dégradé
    header_height = 70
    for i in range(header_height):
        alpha_header = 1 - (i / header_height) * 0.3
        color_header = tuple(int(c * alpha_header) for c in color)
        cv2.line(overlay, (0, i), (w, i), color_header, 1)
    
    # Texte principal avec ombre
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Ombre du texte
    cv2.putText(overlay, icon, (20, 55), font, 1.5, (0, 0, 0), 4)
    cv2.putText(overlay, title, (70, 50), font, 1.2, (0, 0, 0), 4)
    
    # Texte principal
    cv2.putText(overlay, icon, (20, 55), font, 1.5, (255, 255, 255), 2)
    cv2.putText(overlay, title, (70, 50), font, 1.2, (255, 255, 255), 2)
    
    # Badge de compteur
    badge_text = f"#{alert_count}"
    text_size = cv2.getTextSize(badge_text, font, 0.8, 2)[0]
    badge_x = w - text_size[0] - 30
    cv2.circle(overlay, (badge_x + text_size[0]//2, 35), 25, (255, 255, 255), -1)
    cv2.circle(overlay, (badge_x + text_size[0]//2, 35), 25, color, 3)
    cv2.putText(overlay, badge_text, (badge_x, 45), font, 0.8, color, 2)
    
    # Bandeau inférieur avec détails
    footer_height = 50
    overlay_footer = overlay.copy()
    cv2.rectangle(overlay_footer, (0, h - footer_height), (w, h), color, -1)
    cv2.addWeighted(overlay_footer, 0.85, overlay, 0.15, 0, overlay)
    
    # Texte du footer
    time_str = datetime.now().strftime("%H:%M:%S")
    footer_text = f"{subtitle} | {time_str}"
    text_size = cv2.getTextSize(footer_text, font, 0.7, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(overlay, footer_text, (text_x, h - 18), font, 0.7, (255, 255, 255), 2)
    
    # Indicateurs visuels sur les côtés (LED style)
    led_count = 5
    led_spacing = h // (led_count + 1)
    for i in range(led_count):
        y_pos = (i + 1) * led_spacing
        blink = np.sin(time.time() * 15 + i) > 0
        led_color = color if blink else tuple(int(c * 0.3) for c in color)
        cv2.circle(overlay, (15, y_pos), 6, led_color, -1)
        cv2.circle(overlay, (w - 15, y_pos), 6, led_color, -1)
    
    return overlay


def render_sidebar():
    """Affiche la barre latérale."""
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>⚙️ Configuration</h2>", 
                   unsafe_allow_html=True)
        
        # Seuils
        st.subheader("Seuils de Détection")
        ear_threshold = st.slider("Seuil EAR", 0.1, 0.4, 0.25, 0.01,
                                  help="Eye Aspect Ratio - Yeux considérés comme fermés si EAR < seuil")
        mar_threshold = st.slider("Seuil MAR", 0.3, 1.0, 0.6, 0.05,
                                  help="Mouth Aspect Ratio - Bâillement si MAR > seuil")
        
        # Alertes
        st.markdown("---")
        st.subheader("🔔 Système d'Alertes")
        
        col1, col2 = st.columns(2)
        with col1:
            audio_alerts = st.toggle("🔊 Audio", value=True,
                                    help="Alertes sonores dans le navigateur")
        with col2:
            visual_alerts = st.toggle("👁️ Visuel", value=True,
                                     help="Effets visuels sur la vidéo")
        
        alert_cooldown = st.slider("Cooldown (sec)", 1, 10, 3,
                                   help="Temps minimum entre deux alertes")
        
        # Mode
        st.markdown("---")
        st.subheader("📱 Mode d'Analyse")
        analysis_mode = st.radio(
            "",
            ["📷 Image", "📹 Webcam"],
            label_visibility="collapsed"
        )
        
        # Info
        st.markdown("---")
        st.info("""
        **Version 2.0** 🚀
        
        Nouveau système d'alertes:
        - 🔊 Audio navigateur (Web Audio API)
        - ✨ Effets visuels modernes
        - 📊 Métriques temps réel
        """)
        
        return ear_threshold, mar_threshold, analysis_mode, audio_alerts, visual_alerts, alert_cooldown


def render_webcam_analysis(ear_threshold: float, mar_threshold: float, 
                          audio_alerts: bool, visual_alerts: bool, alert_cooldown: int):
    """Affiche l'analyse webcam en temps réel avec système d'alertes moderne."""
    
    # Injecter le code audio
    st.components.v1.html(get_audio_html(), height=0)
    
    st.markdown("<h2>📹 Détection en Temps Réel</h2>", unsafe_allow_html=True)
    
    # Zone d'information
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                padding: 1rem; border-radius: 12px; margin-bottom: 1rem; border: 1px solid #667eea30;'>
        <h4 style='margin: 0; color: #667eea;'>🎯 Comment ça marche ?</h4>
        <ul style='margin: 0.5rem 0 0 0; padding-left: 1.2rem;'>
            <li>Activez la caméra et autorisez l'accès</li>
            <li>Positionnez-vous face à la caméra</li>
            <li>Le système détecte automatiquement les signes de fatigue</li>
            <li>Les alertes se déclenchent avec sons et effets visuels</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialisation session state
    if 'alert_count' not in st.session_state:
        st.session_state['alert_count'] = 0
    if 'alert_history' not in st.session_state:
        st.session_state['alert_history'] = deque(maxlen=20)
    if 'last_alert_time' not in st.session_state:
        st.session_state['last_alert_time'] = 0
    if 'frame_count' not in st.session_state:
        st.session_state['frame_count'] = 0
    
    # Layout
    col_main, col_metrics = st.columns([3, 2])
    
    with col_main:
        # Bouton de démarrage
        run_camera = st.checkbox("▶️ Démarrer la Détection", key="run_cam")
        
        if run_camera:
            # Placeholders
            frame_placeholder = st.empty()
            alert_banner = st.empty()
            
            # Initialisation
            landmark_extractor = LandmarkExtractor(static_image_mode=False)
            fatigue_metrics = FatigueMetrics(
                ear_threshold=ear_threshold,
                mar_threshold=mar_threshold
            )
            
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            try:
                while run_camera:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Impossible d'accéder à la webcam")
                        break
                    
                    st.session_state['frame_count'] += 1
                    
                    # Traitement
                    result_frame, ear, mar, alerts, face_detected = process_image(
                        frame, landmark_extractor, fatigue_metrics
                    )
                    
                    # Gestion des alertes
                    current_time = time.time()
                    alert_active = False
                    alert_type = None
                    
                    # Vérifier conditions d'alerte
                    if alerts.get('drowsiness_alert') or alerts.get('eye_closure'):
                        if current_time - st.session_state['last_alert_time'] > alert_cooldown:
                            st.session_state['alert_count'] += 1
                            st.session_state['last_alert_time'] = current_time
                            alert_active = True
                            alert_type = 'drowsiness' if alerts.get('drowsiness_alert') else 'eye_closure'
                            
                            # Historique
                            st.session_state['alert_history'].append({
                                'time': datetime.now().strftime("%H:%M:%S"),
                                'type': alert_type,
                                'ear': round(ear, 3),
                                'mar': round(mar, 3)
                            })
                            
                            # Déclencher son
                            if audio_alerts:
                                trigger_browser_alert()
                    
                    elif alerts.get('yawn'):
                        if current_time - st.session_state['last_alert_time'] > alert_cooldown:
                            st.session_state['alert_count'] += 1
                            st.session_state['last_alert_time'] = current_time
                            alert_active = True
                            alert_type = 'yawn'
                            
                            st.session_state['alert_history'].append({
                                'time': datetime.now().strftime("%H:%M:%S"),
                                'type': alert_type,
                                'ear': round(ear, 3),
                                'mar': round(mar, 3)
                            })
                            
                            if audio_alerts:
                                trigger_browser_alert()
                    
                    # Appliquer overlay si alerte active
                    if alert_active and visual_alerts:
                        result_frame = add_modern_alert_overlay(
                            result_frame, alert_type, 
                            st.session_state['alert_count']
                        )
                    
                    # Affichage frame
                    frame_placeholder.image(
                        cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )
                    
                    # Mise à jour alert banner
                    if alert_active:
                        if alert_type == 'drowsiness':
                            alert_banner.markdown("""
                            <div class="alert-critical">
                                <h3 style='margin: 0; font-size: 1.5rem;'>🚨 DANGER - SOMNOLENCE 🚨</h3>
                                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Arrêtez-vous immédiatement et reposez-vous!</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif alert_type == 'eye_closure':
                            alert_banner.markdown("""
                            <div class="alert-warning">
                                <h4 style='margin: 0;'>⚠️ Yeux Fermés Prolongés</h4>
                            </div>
                            """, unsafe_allow_html=True)
                        elif alert_type == 'yawn':
                            alert_banner.markdown("""
                            <div class="alert-warning">
                                <h4 style='margin: 0;'>🥱 Bâillement Détecté</h4>
                            </div>
                            """, unsafe_allow_html=True)
                    elif any(alerts.values()):
                        alert_banner.warning("⚠️ Signes de fatigue détectés - Surveillance active")
                    else:
                        alert_banner.markdown("""
                        <div class="status-safe">
                            ✅ État Normal - Surveillance Active
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Mise à jour métriques
                    if face_detected:
                        st.session_state['current_ear'] = ear
                        st.session_state['current_mar'] = mar
                        st.session_state['current_alerts'] = alerts
                    
                    time.sleep(0.033)  # ~30 FPS
                    
            finally:
                cap.release()
                landmark_extractor.close()
    
    with col_metrics:
        st.markdown("<h3>📊 Métriques en Direct</h3>", unsafe_allow_html=True)
        
        # Indicateur LIVE
        if run_camera:
            st.markdown("<div class='live-indicator'><div class='live-dot'></div> EN DIRECT</div>", 
                       unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Métriques actuelles
        if 'current_ear' in st.session_state and run_camera:
            ear = st.session_state['current_ear']
            mar = st.session_state['current_mar']
            alerts = st.session_state['current_alerts']
            
            # Cartes de métriques
            col1, col2 = st.columns(2)
            
            with col1:
                ear_status = "🟢" if ear > ear_threshold else "🔴"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">EAR</div>
                    <div class="metric-value">{ear:.3f}</div>
                    <div style='font-size: 1.5rem; margin-top: 0.5rem;'>{ear_status}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                mar_status = "🔴" if mar > mar_threshold else "🟢"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MAR</div>
                    <div class="metric-value">{mar:.3f}</div>
                    <div style='font-size: 1.5rem; margin-top: 0.5rem;'>{mar_status}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Barres de progression
            st.write("**Niveau EAR**")
            st.progress(min(ear / 0.5, 1.0))
            
            st.write("**Niveau MAR**")
            st.progress(min(mar / 1.0, 1.0))
            
        else:
            st.info("🎥 Démarrez la caméra pour voir les métriques")
        
        st.markdown("---")
        
        # Statistiques d'alertes
        st.markdown("<h4>🔔 Statistiques</h4>", unsafe_allow_html=True)
        
        alert_count = st.session_state.get('alert_count', 0)
        
        # Grande carte du compteur
        st.markdown(f"""
        <div class="metric-card" style='text-align: center; border: 2px solid {"#ff416c" if alert_count > 0 else "#ddd"};'>
            <div class="metric-label">Alertes Totales</div>
            <div class="metric-value" style='font-size: 3rem; color: {"#ff416c" if alert_count > 0 else "#667eea"};'>
                {alert_count}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Historique
        if st.session_state.get('alert_history'):
            st.write("**Historique récent:**")
            for alert in list(st.session_state['alert_history'])[-5:]:
                icon = {"drowsiness": "🚨", "eye_closure": "👁️", "yawn": "🥱"}.get(alert['type'], "⚠️")
                st.markdown(f"""
                <div class="alert-history-item">
                    <strong>{icon} {alert['time']}</strong><br>
                    <small>Type: {alert['type'].upper()} | EAR: {alert['ear']} | MAR: {alert['mar']}</small>
                </div>
                """, unsafe_allow_html=True)


def render_image_analysis(ear_threshold: float, mar_threshold: float):
    """Affiche l'analyse d'image."""
    st.markdown("<h2>📷 Analyse d'Image</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choisissez une image",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        landmark_extractor = LandmarkExtractor(static_image_mode=True)
        fatigue_metrics = FatigueMetrics(
            ear_threshold=ear_threshold,
            mar_threshold=mar_threshold
        )
        
        with st.spinner("Analyse en cours..."):
            result_image, ear, mar, alerts, face_detected = process_image(
                image, landmark_extractor, fatigue_metrics
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4>Originale</h4>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col2:
            st.markdown("<h4>Analyse</h4>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        if face_detected:
            st.markdown("---")
            
            cols = st.columns(3)
            with cols[0]:
                st.metric("EAR", f"{ear:.3f}", 
                         delta="Ouvert" if ear > ear_threshold else "Fermé",
                         delta_color="normal" if ear > ear_threshold else "inverse")
            with cols[1]:
                st.metric("MAR", f"{mar:.3f}",
                         delta="Normal" if mar < mar_threshold else "Bâillement",
                         delta_color="normal" if mar < mar_threshold else "inverse")
            with cols[2]:
                stats = fatigue_metrics.get_statistics()
                st.metric("PERCLOS", f"{stats['perclos']*100:.1f}%")
            
            if any(alerts.values()):
                st.error("🚨 **Alerte: Signes de fatigue détectés!**")
            else:
                st.success("✅ État normal")
        else:
            st.warning("Aucun visage détecté")


def main():
    """Point d'entrée principal."""
    st.markdown('<p class="main-header">🚗 Détection de Somnolence</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Système intelligent de surveillance du conducteur avec alertes temps réel</p>', 
                unsafe_allow_html=True)
    
    ear_threshold, mar_threshold, analysis_mode, audio_alerts, visual_alerts, alert_cooldown = render_sidebar()
    
    if analysis_mode == "📷 Image":
        render_image_analysis(ear_threshold, mar_threshold)
    else:
        render_webcam_analysis(ear_threshold, mar_threshold, audio_alerts, visual_alerts, alert_cooldown)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888; font-size: 0.9rem;'>"
        "🎓 Projet Deep Learning - SDIA | 2024 | Version 2.0"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
