"""
Module de prétraitement des images pour la détection de somnolence.

Ce module implémente les techniques de prétraitement vues dans le cours :
- Redimensionnement et normalisation
- Data Augmentation
- Extraction des régions d'intérêt (ROI)
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Union
import yaml


class ImagePreprocessor:
    """
    Classe pour le prétraitement des images faciales.
    Implémente les concepts du Chapitre 1 et 2 (normalisation, augmentation).
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le préprocesseur avec la configuration.
        
        Args:
            config_path: Chemin vers le fichier de configuration YAML
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.eye_size = tuple(self.config['preprocessing']['image_size_eye'])
        self.yawn_size = tuple(self.config['preprocessing']['image_size_yawn'])
        self.transfer_size = tuple(self.config['preprocessing']['image_size_transfer'])
        self.normalize_method = self.config['preprocessing']['normalize_method']
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Redimensionne une image aux dimensions cibles.
        
        Args:
            image: Image d'entrée (H, W, C)
            target_size: Dimensions cibles (width, height)
            
        Returns:
            Image redimensionnée
        """
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray, method: str = None) -> np.ndarray:
        """
        Normalise les valeurs des pixels.
        
        Args:
            image: Image d'entrée
            method: Méthode de normalisation ('rescale' ou 'standardize')
            
        Returns:
            Image normalisée
        """
        method = method or self.normalize_method
        image = image.astype(np.float32)
        
        if method == "rescale":
            # Normalisation simple [0, 255] -> [0, 1]
            return image / 255.0
        elif method == "standardize":
            # Standardisation (moyenne 0, écart-type 1)
            mean = np.mean(image)
            std = np.std(image)
            return (image - mean) / (std + 1e-7)
        else:
            raise ValueError(f"Méthode de normalisation inconnue: {method}")
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convertit une image en niveaux de gris.
        
        Args:
            image: Image en couleur (H, W, 3)
            
        Returns:
            Image en niveaux de gris (H, W, 1)
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return np.expand_dims(gray, axis=-1)
        return image
    
    def preprocess_eye(self, eye_image: np.ndarray) -> np.ndarray:
        """
        Prétraite une image d'œil pour le modèle CNN.
        
        Pipeline:
        1. Conversion niveaux de gris
        2. Redimensionnement 48x48
        3. Normalisation [0, 1]
        
        Args:
            eye_image: Image de l'œil
            
        Returns:
            Image prétraitée (48, 48, 1)
        """
        # Conversion en niveaux de gris
        if len(eye_image.shape) == 3:
            eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        
        # Redimensionnement
        eye_image = self.resize_image(eye_image, self.eye_size)
        
        # Normalisation
        eye_image = self.normalize_image(eye_image)
        
        # Ajout de la dimension canal
        eye_image = np.expand_dims(eye_image, axis=-1)
        
        return eye_image
    
    def preprocess_mouth(self, mouth_image: np.ndarray) -> np.ndarray:
        """
        Prétraite une image de bouche pour le modèle CNN.
        
        Args:
            mouth_image: Image de la bouche
            
        Returns:
            Image prétraitée (96, 96, 3)
        """
        # Redimensionnement
        mouth_image = self.resize_image(mouth_image, self.yawn_size)
        
        # Normalisation
        mouth_image = self.normalize_image(mouth_image)
        
        return mouth_image
    
    def preprocess_for_transfer_learning(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraite une image pour le transfer learning.
        
        Args:
            image: Image d'entrée
            
        Returns:
            Image prétraitée (224, 224, 3)
        """
        # Conversion BGR -> RGB si nécessaire
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionnement
        image = self.resize_image(image, self.transfer_size)
        
        # Prétraitement spécifique à MobileNetV2/EfficientNet
        image = image.astype(np.float32)
        image = image / 127.5 - 1.0  # Normalisation [-1, 1]
        
        return image
    
    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Applique des augmentations de données aléatoires.
        
        Implémente les techniques du cours pour améliorer la généralisation.
        
        Args:
            image: Image d'entrée
            
        Returns:
            Image augmentée
        """
        aug_config = self.config['preprocessing']['augmentation']
        
        # Rotation aléatoire
        if np.random.random() < 0.5:
            angle = np.random.uniform(-aug_config['rotation_range'], 
                                       aug_config['rotation_range'])
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), 
                                   borderMode=cv2.BORDER_REFLECT)
        
        # Flip horizontal
        if aug_config['horizontal_flip'] and np.random.random() < 0.5:
            image = cv2.flip(image, 1)
        
        # Zoom
        if np.random.random() < 0.3:
            zoom = np.random.uniform(1 - aug_config['zoom_range'], 
                                     1 + aug_config['zoom_range'])
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom), int(w * zoom)
            image = cv2.resize(image, (new_w, new_h))
            
            # Recadrage au centre
            if zoom > 1:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = image[start_h:start_h + h, start_w:start_w + w]
            else:
                # Padding
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                image = cv2.copyMakeBorder(image, pad_h, h - new_h - pad_h,
                                           pad_w, w - new_w - pad_w,
                                           cv2.BORDER_REFLECT)
        
        # Ajustement de luminosité
        if np.random.random() < 0.3:
            brightness_range = aug_config['brightness_range']
            factor = np.random.uniform(brightness_range[0], brightness_range[1])
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        return image
    
    def extract_roi(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extrait une région d'intérêt (ROI) d'une image.
        
        Args:
            image: Image complète
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            ROI extraite
        """
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        return image[y:y + h, x:x + w]
    
    def preprocess_batch(self, images: List[np.ndarray], 
                         target_type: str = 'eye') -> np.ndarray:
        """
        Prétraite un lot d'images.
        
        Args:
            images: Liste d'images
            target_type: Type de cible ('eye', 'mouth', 'transfer')
            
        Returns:
            Batch d'images prétraitées (N, H, W, C)
        """
        processed = []
        
        for img in images:
            if target_type == 'eye':
                processed.append(self.preprocess_eye(img))
            elif target_type == 'mouth':
                processed.append(self.preprocess_mouth(img))
            elif target_type == 'transfer':
                processed.append(self.preprocess_for_transfer_learning(img))
            else:
                raise ValueError(f"Type de cible inconnu: {target_type}")
        
        return np.array(processed)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Charge la configuration depuis un fichier YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dictionnaire de configuration
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# Fonction utilitaire pour visualiser le prétraitement
def visualize_preprocessing(image_path: str, preprocessor: ImagePreprocessor = None):
    """
    Visualise les différentes étapes du prétraitement.
    
    Args:
        image_path: Chemin vers l'image
        preprocessor: Instance de ImagePreprocessor
    """
    import matplotlib.pyplot as plt
    
    if preprocessor is None:
        preprocessor = ImagePreprocessor()
    
    # Chargement de l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Création des visualisations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Image Originale')
    axes[0, 0].axis('off')
    
    # Niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Niveaux de Gris')
    axes[0, 1].axis('off')
    
    # Redimensionnée pour œil
    eye_resized = preprocessor.resize_image(gray, (48, 48))
    axes[0, 2].imshow(eye_resized, cmap='gray')
    axes[0, 2].set_title('Redimensionnée (48x48)')
    axes[0, 2].axis('off')
    
    # Normalisée
    eye_normalized = preprocessor.preprocess_eye(image)
    axes[1, 0].imshow(eye_normalized.squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Normalisée [0, 1]')
    axes[1, 0].axis('off')
    
    # Augmentée
    augmented = preprocessor.apply_augmentation(image)
    axes[1, 1].imshow(augmented)
    axes[1, 1].set_title('Augmentée')
    axes[1, 1].axis('off')
    
    # Pour transfer learning
    transfer = preprocessor.preprocess_for_transfer_learning(image)
    # Remap [-1, 1] -> [0, 1] pour visualisation
    transfer_vis = (transfer + 1) / 2
    axes[1, 2].imshow(transfer_vis)
    axes[1, 2].set_title('Transfer Learning (224x224)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('reports/figures/preprocessing_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualisation sauvegardée dans 'reports/figures/preprocessing_demo.png'")


if __name__ == "__main__":
    # Test du préprocesseur
    preprocessor = ImagePreprocessor()
    print("Préprocesseur initialisé avec succès!")
    print(f"Configuration chargée: {preprocessor.config['project']['name']}")
