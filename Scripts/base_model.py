from abc import ABC, abstractmethod
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time

class BaseModel(ABC):

    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.training_time = None
        self.model_params = {}
        
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    def fit_timed(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Train model and return training time"""
        start_time = time.time()
        self.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        self.is_trained = True
        return self.training_time
    
    def predict_timed(self, X_test: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make predictions and return inference time"""
        start_time = time.time()
        predictions = self.predict(X_test)
        inference_time = time.time() - start_time
        return predictions, inference_time
    
    def save_model(self, filepath: Path) -> None:
        """Save model to disk"""
        model_data = {
            'name': self.name,
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'model_params': self.model_params,
            'model_state': self._get_model_state()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: Path) -> None:
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.name = model_data['name']
        self.is_trained = model_data['is_trained']
        self.training_time = model_data['training_time']
        self.model_params = model_data['model_params']
        self._set_model_state(model_data['model_state'])
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model-specific state for saving"""
        pass
    
    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model-specific state from loading"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'parameters': self.model_params
        }