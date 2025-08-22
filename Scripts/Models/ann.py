import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ANNNetwork(nn.Module):
    """PyTorch neural network architecture."""
    
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 64], activation='relu'):
        super(ANNNetwork, self).__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class ANNModel:
    """Artificial Neural Network model for inverse kinematics."""
    
    def __init__(self, input_dim=2, output_dim=2, hidden_layers=[64, 64], 
                 activation='relu', learning_rate=0.001, epochs=100, batch_size=32):
        """
        Initialize ANN model.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Normalization parameters
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        
    def fit(self, X_train, y_train):
        """
        Train the ANN model.
        
        Args:
            X_train: Input features (end-effector positions)
            y_train: Target values (joint angles)
        """
        # Convert to numpy arrays if needed
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        
        # Normalize data
        self.X_mean = X_train.mean(axis=0)
        self.X_std = X_train.std(axis=0) + 1e-8
        self.y_mean = y_train.mean(axis=0)
        self.y_std = y_train.std(axis=0) + 1e-8
        
        X_train_norm = (X_train - self.X_mean) / self.X_std
        y_train_norm = (y_train - self.y_mean) / self.y_std
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train_norm).to(self.device)
        y_tensor = torch.FloatTensor(y_train_norm).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = ANNNetwork(
            self.input_dim, 
            self.output_dim, 
            self.hidden_layers, 
            self.activation
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
                
    def predict(self, X_test):
        """
        Make predictions.
        
        Args:
            X_test: Input features for prediction
            
        Returns:
            Predicted joint angles
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        self.model.eval()
        
        # Convert and normalize
        X_test = np.array(X_test, dtype=np.float32)
        X_test_norm = (X_test - self.X_mean) / self.X_std
        X_tensor = torch.FloatTensor(X_test_norm).to(self.device)
        
        # Predict
        with torch.no_grad():
            y_pred_norm = self.model(X_tensor).cpu().numpy()
            
        # Denormalize
        y_pred = y_pred_norm * self.y_std + self.y_mean
        
        return y_pred


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate dummy data (2D end-effector positions -> 2 joint angles)
    n_samples = 1000
    X_train = np.random.randn(n_samples, 2)  # End-effector positions (x, y)
    y_train = np.random.randn(n_samples, 2)  # Joint angles (theta1, theta2)
    
    # Create and train model
    model = ANNModel(
        input_dim=2,
        output_dim=2,
        hidden_layers=[128, 64, 32],
        activation='relu',
        learning_rate=0.001,
        epochs=100,
        batch_size=32
    )
    model.fit(X_train, y_train)
    
    # Test prediction
    X_test = np.random.randn(10, 2)
    predictions = model.predict(X_test)
    
    print("\nANN Model Test")
    print("-" * 40)
    print(f"Training samples: {X_train.shape}")
    print(f"Test samples: {X_test.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"\nSample predictions (first 3):")
    for i in range(3):
        print(f"  Input: {X_test[i]}, Output: {predictions[i]}")