"""
Transfer Learning pour la détection de somnolence.

Utilise des modèles pré-entraînés (MobileNetV2, EfficientNetB0) 
sur ImageNet et les fine-tune pour notre tâche.

Concepts du Chapitre 3-4:
- Feature extraction
- Fine-tuning
- Architectures CNN modernes
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, applications
from typing import Tuple, Optional
import numpy as np
import yaml


class TransferLearningModel:
    """
    Modèle de Transfer Learning pour la classification de fatigue.
    
    Utilise MobileNetV2 ou EfficientNetB0 comme backbone.
    """
    
    def __init__(self,
                 base_model_name: str = "MobileNetV2",
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.0001,
                 config_path: str = "config.yaml"):
        """
        Initialise le modèle de transfer learning.
        
        Args:
            base_model_name: Nom du modèle de base ('MobileNetV2', 'EfficientNetB0')
            input_shape: Dimensions d'entrée
            dropout_rate: Taux de dropout
            learning_rate: Taux d'apprentissage
            config_path: Chemin de configuration
        """
        self.base_model_name = base_model_name
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.base_model = None
        self.model = None
        self.history = None
    
    def _get_base_model(self, weights: str = 'imagenet') -> keras.Model:
        """
        Charge le modèle de base pré-entraîné.
        
        Args:
            weights: Poids à charger ('imagenet' ou None)
            
        Returns:
            Modèle de base
        """
        if self.base_model_name == "MobileNetV2":
            base_model = applications.MobileNetV2(
                weights=weights,
                include_top=False,  # Sans la couche de classification finale
                input_shape=self.input_shape
            )
        elif self.base_model_name == "EfficientNetB0":
            base_model = applications.EfficientNetB0(
                weights=weights,
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == "ResNet50":
            base_model = applications.ResNet50(
                weights=weights,
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Modèle non supporté: {self.base_model_name}")
        
        return base_model
    
    def build_feature_extractor(self, trainable: bool = False) -> keras.Model:
        """
        Construit le modèle pour l'extraction de features.
        
        Dans cette phase, le modèle de base est gelé (non entraînable).
        Seules les couches de classification sont entraînées.
        
        Args:
            trainable: Si True, le modèle de base est entraînable
            
        Returns:
            Modèle complet
        """
        # Chargement du modèle de base
        self.base_model = self._get_base_model(weights='imagenet')
        self.base_model.trainable = trainable
        
        # Construction du modèle
        inputs = keras.Input(shape=self.input_shape)
        
        # Normalisation spécifique au modèle
        if self.base_model_name == "MobileNetV2":
            preprocess_input = applications.mobilenet_v2.preprocess_input
        elif self.base_model_name == "EfficientNetB0":
            preprocess_input = applications.efficientnet.preprocess_input
        elif self.base_model_name == "ResNet50":
            preprocess_input = applications.resnet50.preprocess_input
        
        x = preprocess_input(inputs)
        
        # Feature extraction
        x = self.base_model(x, training=False)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Couches de classification
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)
        
        # Sortie
        outputs = layers.Dense(1, activation='sigmoid', name='prediction')(x)
        
        # Création du modèle
        self.model = keras.Model(inputs, outputs, name=f"{self.base_model_name}_Fatigue")
        
        # Compilation
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall', 'AUC']
        )
        
        return self.model
    
    def unfreeze_layers(self, num_layers: int = 20):
        """
        Débloque les dernières couches du modèle de base pour fine-tuning.
        
        Args:
            num_layers: Nombre de couches à débloquer
        """
        self.base_model.trainable = True
        
        # Geler toutes les couches sauf les 'num_layers' dernières
        for layer in self.base_model.layers[:-num_layers]:
            layer.trainable = False
        
        # Recompilation avec un learning rate plus petit pour fine-tuning
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate / 10),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall', 'AUC']
        )
        
        print(f"Fine-tuning: {num_layers} couches débloquées")
        print(f"Learning rate réduit à: {self.learning_rate / 10}")
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              batch_size: int = 32,
              epochs: int = 20,
              fine_tune: bool = False,
              fine_tune_epochs: int = 10) -> keras.callbacks.History:
        """
        Entraîne le modèle.
        
        Args:
            X_train: Données d'entraînement
            y_train: Labels
            X_val: Données de validation
            y_val: Labels de validation
            batch_size: Taille des batchs
            epochs: Nombre d'époques (feature extraction)
            fine_tune: Si True, effectue le fine-tuning
            fine_tune_epochs: Nombre d'époques de fine-tuning
            
        Returns:
            Historique d'entraînement
        """
        if self.model is None:
            self.build_feature_extractor(trainable=False)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                f'models/{self.base_model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Phase 1: Feature extraction
        print("\n" + "="*60)
        print("PHASE 1: FEATURE EXTRACTION")
        print("="*60)
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks
        )
        
        # Phase 2: Fine-tuning (optionnel)
        if fine_tune:
            print("\n" + "="*60)
            print("PHASE 2: FINE-TUNING")
            print("="*60)
            
            self.unfreeze_layers(num_layers=20)
            
            history_fine = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=fine_tune_epochs,
                validation_data=validation_data,
                callbacks=callbacks
            )
            
            # Combiner les historiques
            for key in self.history.history:
                self.history.history[key].extend(history_fine.history[key])
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit la probabilité de fatigue."""
        if self.model is None:
            raise ValueError("Modèle non entraîné!")
        return self.model.predict(X)
    
    def summary(self):
        """Affiche le résumé du modèle."""
        if self.model is None:
            self.build_feature_extractor()
        return self.model.summary()
    
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """Charge un modèle."""
        self.model = keras.models.load_model(filepath)


if __name__ == "__main__":
    # Test du transfer learning
    print("Test du Transfer Learning")
    print("="*60)
    
    model = TransferLearningModel(base_model_name="MobileNetV2")
    model.build_feature_extractor()
    model.summary()
    
    print("\n✓ Modèle de Transfer Learning prêt!")
