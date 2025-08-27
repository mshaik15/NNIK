from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class RandomForestModel:
    """Random Forest model for inverse kinematics."""
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        
        # Create and fit model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        self.model.fit(X_train_scaled, y_train_scaled)
        
    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        X_test_scaled = self.scaler_X.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_


