import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


class KNNModel:
    """K-Nearest Neighbors model for inverse kinematics."""
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        """
        Initialize KNN model.
        
        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function ('uniform' or 'distance')
            metric: Distance metric to use
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def fit(self, X_train, y_train):
        """
        Train the KNN model.
        
        Args:
            X_train: Input features (end-effector positions)
            y_train: Target values (joint angles)
        """
        # Normalize data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        
        # Create and fit model
        self.model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric
        )
        self.model.fit(X_train_scaled, y_train_scaled)
        
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
            
        X_test_scaled = self.scaler_X.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    
    # Generate dummy data (2D end-effector positions -> 2 joint angles)
    n_samples = 1000
    X_train = np.random.randn(n_samples, 2)  # End-effector positions (x, y)
    y_train = np.random.randn(n_samples, 2)  # Joint angles (theta1, theta2)
    
    # Create and train model
    model = KNNModel(n_neighbors=10, weights='distance')
    model.fit(X_train, y_train)
    
    # Test prediction
    X_test = np.random.randn(10, 2)
    predictions = model.predict(X_test)
    
    print("KNN Model Test")
    print("-" * 40)
    print(f"Training samples: {X_train.shape}")
    print(f"Test samples: {X_test.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"\nSample predictions (first 3):")
    for i in range(3):
        print(f"  Input: {X_test[i]}, Output: {predictions[i]}")