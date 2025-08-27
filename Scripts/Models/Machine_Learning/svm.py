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