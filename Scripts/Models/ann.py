import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from base_model import BaseModel
from typing import Dict, Any

class ANNNetwork(nn.Module):
    """PyTorch neural network architecture."""
    
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 64], 
                 activation='relu', dropout_rate=0.1):
        super(ANNNetwork, self).__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Add batch norm
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class ANNModel(BaseModel):
    """Improved ANN model with better training and validation"""
    
    def __init__(self, input_dim=6, output_dim=2, hidden_layers=[128, 64], 
                 activation='relu', learning_rate=0.001, epochs=100, 
                 batch_size=32, dropout_rate=0.1, early_stopping_patience=10):
        super().__init__("ANN")
        
        # Store parameters
        self.model_params = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_layers': hidden_layers,
            'activation': activation,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'early_stopping_patience': early_stopping_patience
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Normalization parameters
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def _normalize_data(self, X, y=None, fit_scalers=False):
        """Normalize input and output data"""
        if fit_scalers:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) + 1e-8
            if y is not None:
                self.y_mean = y.mean(axis=0)
                self.y_std = y.std(axis=0) + 1e-8
        
        X_norm = (X - self.X_mean) / self.X_std
        
        if y is not None:
            y_norm = (y - self.y_mean) / self.y_std
            return X_norm, y_norm
        
        return X_norm
    
    def fit(self, X_train, y_train):
        """Train the ANN model with validation split"""
        
        # Convert to numpy arrays
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        
        # Update output dimension based on data
        if self.model_params['output_dim'] != y_train.shape[1]:
            self.model_params['output_dim'] = y_train.shape[1]
        
        # Create validation split
        val_split = 0.1
        n_val = int(len(X_train) * val_split)
        indices = np.random.permutation(len(X_train))
        
        X_val = X_train[indices[:n_val]]
        y_val = y_train[indices[:n_val]]
        X_train = X_train[indices[n_val:]]
        y_train = y_train[indices[n_val:]]
        
        # Normalize data
        X_train_norm, y_train_norm = self._normalize_data(X_train, y_train, fit_scalers=True)
        X_val_norm, y_val_norm = self._normalize_data(X_val, y_val, fit_scalers=False)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_norm).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_norm).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_norm).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_norm).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.model_params['batch_size'], shuffle=True)
        
        # Initialize model
        self.model = ANNNetwork(
            self.model_params['input_dim'],
            self.model_params['output_dim'],
            self.model_params['hidden_layers'],
            self.model_params['activation'],
            self.model_params['dropout_rate']
        ).to(self.device)
        
        # Loss and optimizer with scheduler
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.model_params['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(self.model_params['epochs']):
            # Training phase
            self.model.train()
            epoch_train_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            avg_train_loss = epoch_train_loss / len(dataloader)
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.model_params['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{self.model_params['epochs']}], "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Restore best model
        self.model.load_state_dict(best_model_state)
        
    def predict(self, X_test):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        self.model.eval()
        
        # Convert and normalize
        X_test = np.array(X_test, dtype=np.float32)
        X_test_norm = self._normalize_data(X_test)
        X_tensor = torch.FloatTensor(X_test_norm).to(self.device)
        
        # Predict
        with torch.no_grad():
            y_pred_norm = self.model(X_tensor).cpu().numpy()
            
        # Denormalize
        y_pred = y_pred_norm * self.y_std + self.y_mean
        
        return y_pred
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for saving"""
        if self.model is None:
            return {}
        
        return {
            'model_state_dict': self.model.state_dict(),
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model state from loading"""
        if not state:
            return
        
        # Recreate model
        self.model = ANNNetwork(
            self.model_params['input_dim'],
            self.model_params['output_dim'],
            self.model_params['hidden_layers'],
            self.model_params['activation'],
            self.model_params['dropout_rate']
        ).to(self.device)
        
        self.model.load_state_dict(state['model_state_dict'])
        self.X_mean = state['X_mean']
        self.X_std = state['X_std']
        self.y_mean = state['y_mean']
        self.y_std = state['y_std']
        self.train_losses = state.get('train_losses', [])
        self.val_losses = state.get('val_losses', [])