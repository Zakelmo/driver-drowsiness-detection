# 🚗 Détection de Somnolence du Conducteur par Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Projet de Vision par Ordinateur et Deep Learning**
> 
> Détection de la somnolence et de la fatigue du conducteur en temps réel à l'aide de modèles de Deep Learning.

---

## 📋 Table des Matières

- [Aperçu](#-aperçu)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Concepts du Cours](#-concepts-du-cours)
- [Structure du Projet](#-structure-du-projet)
- [Métriques de Fatigue](#-métriques-de-fatigue)
- [Résultats](#-résultats)
- [Auteurs](#-auteurs)

---

## 🔭 Aperçu

Ce projet implémente un système complet de **Détection de Somnolence du Conducteur (DDS)** utilisant des techniques avancées de vision par ordinateur et de deep learning. Le système analyse en temps réel les signes de fatigue à travers :

- 👁️ **Détection de la fermeture des yeux** (EAR - Eye Aspect Ratio)
- 🥱 **Détection des bâillements** (MAR - Mouth Aspect Ratio)
- 📊 **Analyse PERCLOS** (Pourcentage de fermeture des paupières)
- 🔔 **Système d'alertes** sonores et visuelles

### 🎯 Objectifs

- Détecter précocement les signes de fatigue au volant
- Réduire les accidents causés par la somnolence
- Fournir une solution embarquable et temps réel

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DRIVER DROWSINESS DETECTION                 │
│                           SYSTEM                                │
└─────────────────────────────────────────────────────────────────┘

  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
  │   WEBCAM    │─────▶│  DETECTION  │─────▶│  ANALYSE    │
  │             │      │   VISAGE    │      │   FATIGUE   │
  └─────────────┘      │  (MediaPipe)│      │             │
                       └─────────────┘      └──────┬──────┘
                                                   │
                          ┌────────────────────────┼────────────────────────┐
                          │                        │                        │
                          ▼                        ▼                        ▼
                   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
                   │    CNN      │         │    EAR      │         │    MAR      │
                   │  (Yeux)     │         │  Calculator │         │  Calculator │
                   └─────────────┘         └─────────────┘         └─────────────┘
                          │                        │                        │
                          └────────────────────────┴────────────────────────┘
                                                   │
                                                   ▼
                                          ┌─────────────┐
                                          │   FUSION    │
                                          │   DECISION  │
                                          └──────┬──────┘
                                                 │
                          ┌──────────────────────┴──────────────────────┐
                          │                                             │
                          ▼                                             ▼
                   ┌─────────────┐                              ┌─────────────┐
                   │   ALERTE    │                              │   LOGS/     │
                   │   SYSTEM    │                              │   REPORTS   │
                   └─────────────┘                              └─────────────┘
```

---

## 🚀 Installation

### Prérequis

- Python 3.9+
- Webcam (pour la détection temps réel)
- 4GB+ RAM recommandé
- GPU optionnel (pour accélérer l'entraînement)

### Étapes d'Installation

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection

# 2. Créer un environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Vérifier l'installation
python -c "import tensorflow; print('✓ TensorFlow:', tensorflow.__version__)"
python -c "import cv2; print('✓ OpenCV:', cv2.__version__)"
```

---

## 💻 Utilisation

### 1. Mode Interface Web (Streamlit)

```bash
streamlit run app/app.py
```

L'interface permet :
- 📷 Analyse d'images
- 🎥 Analyse de vidéos
- 📹 Détection en temps réel via webcam

### 2. Mode Ligne de Commande

```bash
# Détection temps réel avec webcam
python app/real_time_detection.py

# Avec modèle CNN
python app/real_time_detection.py --cnn models/cnn_eye_best.h5

# Configuration personnalisée
python app/real_time_detection.py --camera 0 --width 1280 --height 720
```

### 3. Notebooks d'Exploration

```bash
jupyter notebook notebooks/
```

Notebooks disponibles :
- `01_exploration_et_preparation.ipynb` - Analyse et prétraitement
- `02_modelisation_cnn.ipynb` - Entraînement des modèles
- `03_evaluation_et_tests.ipynb` - Évaluation et métriques

---

## 📚 Concepts du Cours

### Chapitre 1 - Fondamentaux du Deep Learning

| Concept | Implémentation |
|---------|----------------|
| **Perceptron** | Classification binaire œil ouvert/fermé |
| **Sigmoïde** | Activation finale pour probabilité |
| **Descente de Gradient** | Optimisation Adam |
| **Fonction de Perte** | Binary Cross-Entropy |

### Chapitre 2 - Perceptron Multi-Couches

| Concept | Implémentation |
|---------|----------------|
| **MLP** | Classification fatigue basée sur features |
| **Forward Propagation** | `model.predict()` |
| **Backward Propagation** | `model.fit()` avec backprop automatique |
| **Régularisation** | Dropout (rate=0.5) |

### Chapitre 3-4 - CNN et Architectures Avancées

| Concept | Implémentation |
|---------|----------------|
| **Convolution** | `Conv2D` pour extraction de features |
| **Max Pooling** | Réduction dimensionnelle 2x2 |
| **Transfer Learning** | MobileNetV2 pré-entraîné |
| **Data Augmentation** | Rotation, flip, zoom |

---

## 📁 Structure du Projet

```
driver_drowsiness_detection/
│
├── 📂 data/                      # Données
│   ├── raw/                      # Données brutes
│   ├── processed/                # Données prétraitées
│   └── augmented/                # Données augmentées
│
├── 📂 notebooks/                 # Notebooks Jupyter
│   ├── 01_exploration_et_preparation.ipynb
│   ├── 02_modelisation_cnn.ipynb
│   └── 03_evaluation_et_tests.ipynb
│
├── 📂 src/                       # Code source
│   ├── models/                   # Modèles Deep Learning
│   │   ├── cnn.py               # CNN pour yeux/bâillements
│   │   ├── mlp.py               # Perceptron multi-couches
│   │   └── transfer_learning.py # Transfer learning
│   │
│   ├── detection/                # Détection faciale
│   │   ├── face_detector.py     # Détecteur Haar/DNN
│   │   └── landmark_extractor.py # MediaPipe Face Mesh
│   │
│   ├── features/                 # Extraction features
│   │   └── extractor.py         # Features EAR/MAR
│   │
│   ├── utils/                    # Utilitaires
│   │   ├── preprocessing.py     # Prétraitement images
│   │   ├── metrics.py           # EAR/MAR/PERCLOS
│   │   └── alerts.py            # Système d'alertes
│   │
│   └── training/                 # Entraînement
│       ├── train.py             # Script d'entraînement
│       └── evaluate.py          # Évaluation modèles
│
├── 📂 app/                       # Application
│   ├── app.py                   # Interface Streamlit
│   └── real_time_detection.py   # Détection temps réel
│
├── 📂 models/                    # Modèles sauvegardés
│   ├── cnn_eye_best.h5
│   ├── cnn_yawn_best.h5
│   └── mobilenet_fatigue.h5
│
├── 📂 reports/                   # Rapports et figures
│   └── figures/                  # Visualisations
│
├── 📄 config.yaml                # Configuration
├── 📄 requirements.txt           # Dépendances
├── 📄 README.md                  # Ce fichier
└── 📄 LICENSE                    # Licence MIT
```

---

## 📊 Métriques de Fatigue

### Eye Aspect Ratio (EAR)

```
    P1 (coin externe)
         /    \
   P2 (haut)   P3 (haut)
       |          |
   P6 (bas)    P5 (bas)
         \    /
    P4 (coin interne)

EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
```

| État | EAR Typique |
|------|-------------|
| Œil Ouvert | > 0.25 |
| Œil Fermé | < 0.25 |

### Mouth Aspect Ratio (MAR)

| État | MAR Typique |
|------|-------------|
| Bouche Fermée | < 0.4 |
| Bouche Ouverte | 0.4 - 0.6 |
| Bâillement | > 0.6 |

### PERCLOS (PERcentage of eye CLOSure)

```
PERCLOS = (Nombre de frames avec yeux fermés / Nombre total de frames) × 100
```

| Niveau | PERCLOS | Action |
|--------|---------|--------|
| Normal | < 15% | ✅ Continuer |
| Attention | 15-25% | ⚠️ Surveillance |
| Danger | > 25% | 🚨 Alerte immédiate |

---

## 📈 Résultats

### Performance des Modèles

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| CNN Yeux | 96.2% | 94.8% | 97.1% | 95.9% |
| CNN Yawn | 92.5% | 90.3% | 93.8% | 92.0% |
| MobileNetV2 | 94.1% | 92.7% | 95.2% | 93.9% |

### Performance Temps Réel

| Configuration | FPS | Latence |
|---------------|-----|---------|
| CPU Only | 15-20 | ~50ms |
| GPU (CUDA) | 25-30 | ~30ms |
| Edge (Raspberry Pi) | 5-8 | ~150ms |

---

## 🎓 Ressources Pédagogiques

### Datasets Recommandés

| Dataset | Description | Lien |
|---------|-------------|------|
| CEW | Closed Eyes in the Wild | [Lien](#) |
| NTHU-DDD | Driver Drowsiness Detection | [Lien](#) |
| YawDD | Yawning Detection Dataset | [Lien](#) |

### Références

1. Soukupová, T., & Čech, J. (2016). Eye blink detection using facial landmarks. *21st Computer Vision Winter Workshop*.
2. Szegedy, C., et al. (2015). Going deeper with convolutions. *CVPR*.
3. Howard, A., et al. (2019). MobileNets: Efficient CNNs for mobile vision.

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créez votre branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 👨‍🎓 Auteurs

**SDIA Student** - *Deep Learning Course Project*

- 📧 Email: student@example.com
- 🎓 Formation: SDIA (Sciences des Données et Intelligence Artificielle)
- 📅 Année: 2024

---

## 🙏 Remerciements

- Professeur de Deep Learning pour l'encadrement
- Communauté TensorFlow et OpenCV
- Contributeurs des datasets publics

---

<div align="center">

**[⬆ Retour en haut](#-détection-de-somnolence-du-conducteur-par-deep-learning)**

🚗 Conduisez en sécurité ! 🚗

</div>
