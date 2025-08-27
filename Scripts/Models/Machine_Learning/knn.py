from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


class KNNModel:
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def fit(self, X_train, y_train):
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
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        X_test_scaled = self.scaler_X.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred


