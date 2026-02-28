"""
Modèles CNN (Convolutional Neural Networks) pour la détection de somnolence.

Implémente les architectures CNN vues dans les chapitres 3-4 :
- Conv2D pour extraction de features spatiales
- MaxPooling pour réduction dimensionnelle
- Dropout pour régularisation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from typing import Tuple, Optional, Dict, List
import yaml


class EyeCNN:
    """
    CNN pour la classification œil ouvert / œil fermé.
    
    Architecture inspirée de LeNet-5 et des CNN modernes.
    Input: 48x48x1 (grayscale)
    Output: 1 (sigmoid) - probabilité œil fermé
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le modèle CNN pour les yeux.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['cnn_eye']
        self.input_shape = tuple(self.model_config['input_shape'])
        self.num_classes = self.model_config['num_classes']
        self.learning_rate = self.model_config['learning_rate']
        self.dropout_rate = self.model_config['dropout_rate']
        
        self.model = None
        self.history = None
    
    def build_model(self) -> keras.Model:
        """
        Construit l'architecture du modèle CNN.
        
        Architecture:
        - Conv2D + ReLU + MaxPool (feature extraction bas niveau)
        - Conv2D + ReLU + MaxPool (features moyen niveau)
        - Flatten (vectorisation)
        - Dense + Dropout (classification)
        - Dense (1) + Sigmoid (sortie binaire)
        
        Returns:
            Modèle Keras compilé
        """
        model = models.Sequential(name="EyeCNN")
        
        # ========== BLOC 1: Extraction features bas niveau ==========
        # Conv2D: 32 filtres 3x3, activation ReLU
        model.add(layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            input_shape=self.input_shape,
            name='conv1'
        ))
        # Output: 48x48x32
        
        # MaxPooling: réduction 2x2
        model.add(layers.MaxPooling2D(
            pool_size=(2, 2),
            name='pool1'
        ))
        # Output: 24x24x32
        
        # ========== BLOC 2: Extraction features moyen niveau ==========
        model.add(layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv2'
        ))
        # Output: 24x24x64
        
        model.add(layers.MaxPooling2D(
            pool_size=(2, 2),
            name='pool2'
        ))
        # Output: 12x12x64
        
        # ========== BLOC 3: Features haut niveau ==========
        model.add(layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            name='conv3'
        ))
        # Output: 12x12x128
        
        model.add(layers.MaxPooling2D(
            pool_size=(2, 2),
            name='pool3'
        ))
        # Output: 6x6x128 = 4608 valeurs
        
        # ========== CLASSIFICATION ==========
        # Flatten: conversion en vecteur
        model.add(layers.Flatten(name='flatten'))
        # Output: 4608
        
        # Couche Dense avec Dropout (régularisation)
        model.add(layers.Dense(
            units=128,
            activation='relu',
            name='dense1'
        ))
        
        # Dropout: désactive aléatoirement 50% des neurones pendant l'entraînement
        model.add(layers.Dropout(
            rate=self.dropout_rate,
            name='dropout'
        ))
        
        # Couche de sortie: 1 neurone avec sigmoid (classification binaire)
        model.add(layers.Dense(
            units=1,
            activation='sigmoid',
            name='output'
        ))
        
        # Compilation du modèle
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',  # Fonction de perte pour classification binaire
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        self.model = model
        return model
    
    def summary(self):
        """Affiche le résumé de l'architecture du modèle."""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              batch_size: int = 32,
              epochs: int = 50) -> keras.callbacks.History:
        """
        Entraîne le modèle CNN.
        
        Args:
            X_train: Données d'entraînement (N, 48, 48, 1)
            y_train: Labels d'entraînement (N,)
            X_val: Données de validation (optionnel)
            y_val: Labels de validation (optionnel)
            batch_size: Taille des batchs
            epochs: Nombre d'époques
            
        Returns:
            Historique de l'entraînement
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks pour améliorer l'entraînement
        callbacks = [
            # Early stopping: arrête si pas d'amélioration après 10 époques
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Sauvegarde du meilleur modèle
            ModelCheckpoint(
                'models/cnn_eye_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            # Réduction du learning rate si plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Entraînement
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit la classe des images.
        
        Args:
            X: Images d'entrée (N, 48, 48, 1)
            
        Returns:
            Probabilités de la classe positive (N,)
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas entraîné!")
        return self.model.predict(X)
    
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        if self.model is None:
            raise ValueError("Le modèle n'est pas entraîné!")
        self.model.save(filepath)
        print(f"Modèle sauvegardé: {filepath}")
    
    def load(self, filepath: str):
        """Charge un modèle sauvegardé."""
        self.model = keras.models.load_model(filepath)
        print(f"Modèle chargé: {filepath}")


class YawnCNN:
    """
    CNN pour la détection des bâillements.
    
    Input: 96x96x3 (RGB)
    Output: 1 (sigmoid) - probabilité de bâillement
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['cnn_yawn']
        self.input_shape = tuple(self.model_config['input_shape'])
        self.learning_rate = self.model_config['learning_rate']
        self.dropout_rate = self.model_config['dropout_rate']
        
        self.model = None
    
    def build_model(self) -> keras.Model:
        """Construit le modèle CNN pour les bâillements."""
        model = models.Sequential(name="YawnCNN")
        
        # Bloc 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                               padding='same', input_shape=self.input_shape))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Bloc 2
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Bloc 3
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        
        # Classification
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compilation
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall', 'AUC']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              batch_size=16, epochs=30):
        """Entraîne le modèle de détection des bâillements."""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ModelCheckpoint('models/cnn_yawn_best.h5', monitor='val_accuracy', 
                          save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        return self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks
        )
    
    def predict(self, X):
        """Prédit la probabilité de bâillement."""
        return self.model.predict(X)
    
    def save(self, filepath):
        """Sauvegarde le modèle."""
        self.model.save(filepath)


def create_cnn_demo():
    """
    Crée une démonstration du modèle CNN avec des données synthétiques.
    """
    print("=" * 60)
    print("DÉMONSTRATION DU MODÈLE CNN POUR DÉTECTION DES YEUX")
    print("=" * 60)
    
    # Création du modèle
    cnn = EyeCNN()
    model = cnn.build_model()
    
    # Affichage du résumé
    print("\nArchitecture du modèle:")
    print("-" * 60)
    model.summary()
    
    print("\n" + "=" * 60)
    print("Visualisation de l'architecture")
    print("=" * 60)
    
    # Génération de données synthétiques pour test
    print("\nGénération de données synthétiques...")
    np.random.seed(42)
    X_dummy = np.random.rand(100, 48, 48, 1).astype(np.float32)
    y_dummy = np.random.randint(0, 2, size=(100,))
    
    print(f"Forme des données: {X_dummy.shape}")
    print(f"Distribution des classes: {np.bincount(y_dummy)}")
    
    # Test de prédiction (avant entraînement)
    print("\nTest de prédiction (modèle non entraîné):")
    predictions = cnn.predict(X_dummy[:5])
    print(f"Prédictions (5 premiers): {predictions.flatten()}")
    
    print("\n" + "=" * 60)
    print("✓ Modèle CNN créé avec succès!")
    print("=" * 60)
    
    return cnn


if __name__ == "__main__":
    create_cnn_demo()
