import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


class SVMModel:
    """Support Vector Machine model for inverse kinematics."""
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        # Initialize SVM model.
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def fit(self, X_train, y_train):
        # Train the SVM model.
        # Normalize data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        
        # Create base SVR model
        base_svr = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma
        )
        
        # Use MultiOutputRegressor for multiple outputs
        self.model = MultiOutputRegressor(base_svr)
        self.model.fit(X_train_scaled, y_train_scaled)
        
    def predict(self, X_test):
        # Make predictions.
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        X_test_scaled = self.scaler_X.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    n_samples = 500
    X_train = np.random.randn(n_samples, 2)  # End-effector positions (x, y)
    y_train = np.random.randn(n_samples, 2)  # Joint angles (theta1, theta2)
    
    # Create and train model
    model = SVMModel(kernel='rbf', C=10.0, epsilon=0.01)
    model.fit(X_train, y_train)
    
    # Test prediction
    X_test = np.random.randn(10, 2)
    predictions = model.predict(X_test)
    
    print("SVM Model Test")
    print("-" * 40)
    print(f"Training samples: {X_train.shape}")
    print(f"Test samples: {X_test.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"\nSample predictions (first 3):")
    for i in range(3):
        print(f"  Input: {X_test[i]}, Output: {predictions[i]}")