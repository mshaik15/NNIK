import numpy as np

class ELMModel:

    def __init__(self, input_dim=2, hidden_dim=100, output_dim=2, 
                 activation='relu', random_state=42):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.random_state = random_state
        
        # Model parameters
        self.W_input = None  # Input weights
        self.b_hidden = None  # Hidden bias
        self.W_output = None  # Output weights
        
        # Normalization parameters
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        
        np.random.seed(random_state)
        
    def _activate(self, X):
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, X)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        elif self.activation == 'tanh':
            return np.tanh(X)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
            
    def fit(self, X_train, y_train):
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        
        # Normalize data
        self.X_mean = X_train.mean(axis=0)
        self.X_std = X_train.std(axis=0) + 1e-8
        self.y_mean = y_train.mean(axis=0)
        self.y_std = y_train.std(axis=0) + 1e-8
        
        X_train_norm = (X_train - self.X_mean) / self.X_std
        y_train_norm = (y_train - self.y_mean) / self.y_std
        
        # Random input weights and bias
        self.W_input = np.random.randn(self.input_dim, self.hidden_dim)
        self.b_hidden = np.random.randn(self.hidden_dim)
        
        # Calculate hidden layer output
        H = self._activate(np.dot(X_train_norm, self.W_input) + self.b_hidden)

        H_pinv = np.linalg.pinv(H)
        self.W_output = np.dot(H_pinv, y_train_norm)
        
    def predict(self, X_test):
        if self.W_input is None:
            raise ValueError("Model must be trained before prediction")
            
        X_test = np.array(X_test, dtype=np.float32)
        
        # Normalize input
        X_test_norm = (X_test - self.X_mean) / self.X_std
        
        # Forward pass
        H = self._activate(np.dot(X_test_norm, self.W_input) + self.b_hidden)
        y_pred_norm = np.dot(H, self.W_output)
        
        # Denormalize output
        y_pred = y_pred_norm * self.y_std + self.y_mean
        
        return y_pred


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    
    # Generate dummy data (2D end-effector positions -> 2 joint angles)
    n_samples = 1000
    X_train = np.random.randn(n_samples, 2)  # End-effector positions (x, y)
    y_train = np.random.randn(n_samples, 2)  # Joint angles (theta1, theta2)
    
    # Create and train model
    model = ELMModel(
        input_dim=2,
        hidden_dim=200,
        output_dim=2,
        activation='relu',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Test prediction
    X_test = np.random.randn(10, 2)
    predictions = model.predict(X_test)
    
    print("ELM Model Test")
    print("-" * 40)
    print(f"Training samples: {X_train.shape}")
    print(f"Hidden neurons: {model.hidden_dim}")
    print(f"Test samples: {X_test.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"\nSample predictions (first 3):")
    for i in range(3):
        print(f"  Input: {X_test[i]}, Output: {predictions[i]}")