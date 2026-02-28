"""
Perceptron Multi-Couches (MLP) pour la classification de fatigue.

Implémente les concepts du Chapitre 2:
- Réseaux fully-connected
- Fonctions d'activation (ReLU, Sigmoid)
- Dropout pour régularisation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from typing import Tuple, Optional
import numpy as np
import yaml


class FatigueMLP:
    """
    Perceptron Multi-Couches pour classifier l'état de fatigue.
    
    Utilise les features extraites (EAR, MAR, PERCLOS, etc.) pour
    prédire si le conducteur est fatigué.
    
    Input: Vecteur de features (EAR, MAR, taux clignements, etc.)
    Output: 1 (sigmoid) - probabilité de fatigue
    """
    
    def __init__(self, 
                 input_dim: int = 10,
                 hidden_layers: list = [64, 32],
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 config_path: str = "config.yaml"):
        """
        Initialise le MLP.
        
        Args:
            input_dim: Dimension d'entrée (nombre de features)
            hidden_layers: Liste des tailles des couches cachées
            dropout_rate: Taux de dropout
            learning_rate: Taux d'apprentissage
            config_path: Chemin de configuration
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Chargement de la config si disponible
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except:
            self.config = None
        
        self.model = None
        self.history = None
    
    def build_model(self) -> keras.Model:
        """
        Construit l'architecture MLP.
        
        Returns:
            Modèle Keras compilé
        """
        model = models.Sequential(name="FatigueMLP")
        
        # Couche d'entrée
        model.add(layers.Input(shape=(self.input_dim,), name='input'))
        
        # Couches cachées
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(
                units=units,
                activation='relu',
                name=f'hidden_{i+1}'
            ))
            model.add(layers.Dropout(
                rate=self.dropout_rate,
                name=f'dropout_{i+1}'
            ))
        
        # Couche de sortie
        model.add(layers.Dense(
            units=1,
            activation='sigmoid',
            name='output'
        ))
        
        # Compilation
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
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
        """Affiche le résumé du modèle."""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              batch_size: int = 32,
              epochs: int = 100) -> keras.callbacks.History:
        """
        Entraîne le MLP.
        
        Args:
            X_train: Features d'entraînement (N, input_dim)
            y_train: Labels (N,)
            X_val: Features de validation
            y_val: Labels de validation
            batch_size: Taille des batchs
            epochs: Nombre d'époques
            
        Returns:
            Historique d'entraînement
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
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
        """Prédit la probabilité de fatigue."""
        if self.model is None:
            raise ValueError("Modèle non entraîné!")
        return self.model.predict(X)
    
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """Charge un modèle."""
        self.model = keras.models.load_model(filepath)


if __name__ == "__main__":
    # Test du MLP
    print("Test du MLP pour classification de fatigue")
    print("=" * 50)
    
    mlp = FatigueMLP(input_dim=5, hidden_layers=[32, 16])
    mlp.summary()
    
    # Données synthétiques
    X = np.random.rand(200, 5).astype(np.float32)
    y = np.random.randint(0, 2, size=(200,))
    
    print(f"\nDonnées: {X.shape}, Labels: {y.shape}")
    print("MLP prêt à l'emploi!")
