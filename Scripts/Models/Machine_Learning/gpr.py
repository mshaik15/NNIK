import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic
from sklearn.preprocessing import StandardScaler


class GPRModel:
    """Gaussian Process Regression model for inverse kinematics."""
    
    def __init__(self, kernel_type='rbf', length_scale=1.0, alpha=1e-10, 
                 n_restarts_optimizer=10):
        """
        Initialize GPR model.
        
        Args:
            kernel_type: Type of kernel ('rbf', 'matern', 'rational_quadratic')
            length_scale: Length scale for kernel
            alpha: Noise level
            n_restarts_optimizer: Number of restarts for optimizer
        """
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Build kernel
        self.kernel = self._build_kernel()
        
    def _build_kernel(self):
        """Build the kernel based on specified type."""
        if self.kernel_type == 'rbf':
            kernel = RBF(length_scale=self.length_scale, 
                        length_scale_bounds=(1e-5, 1e5))
        elif self.kernel_type == 'matern':
            kernel = Matern(length_scale=self.length_scale, 
                           length_scale_bounds=(1e-5, 1e5))
        elif self.kernel_type == 'rational_quadratic':
            kernel = RationalQuadratic(length_scale=self.length_scale, 
                                      alpha=1.0)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
            
        # Add white noise kernel
        kernel = kernel + WhiteKernel(noise_level=self.alpha, 
                                      noise_level_bounds=(1e-10, 1e5))
        return kernel
        
    def fit(self, X_train, y_train):
        """
        Train the GPR model.
        
        Args:
            X_train: Input features (end-effector positions)
            y_train: Target values (joint angles)
        """
        # Normalize data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        
        # Create and fit model
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=False  # We handle normalization ourselves
        )
        self.model.fit(X_train_scaled, y_train_scaled)
        
    def predict(self, X_test, return_std=False):
        """
        Make predictions.
        
        Args:
            X_test: Input features for prediction
            return_std: Whether to return standard deviation
            
        Returns:
            Predicted joint angles (and optionally standard deviations)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        X_test_scaled = self.scaler_X.transform(X_test)
        
        if return_std:
            y_pred_scaled, y_std_scaled = self.model.predict(X_test_scaled, 
                                                             return_std=True)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            # Scale standard deviation
            y_std = y_std_scaled * self.scaler_y.scale_
            return y_pred, y_std
        else:
            y_pred_scaled = self.model.predict(X_test_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            return y_pred
    
    def sample(self, X_test, n_samples=1):
        """
        Sample from the posterior distribution.
        
        Args:
            X_test: Input features for sampling
            n_samples: Number of samples to draw
            
        Returns:
            Samples from the posterior distribution
        """
        if self.model is None:
            raise ValueError("Model must be trained before sampling")
            
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Sample from the GP
        y_samples_scaled = self.model.sample_y(X_test_scaled, n_samples=n_samples)
        
        # Denormalize samples
        samples = []
        for i in range(n_samples):
            y_sample = self.scaler_y.inverse_transform(y_samples_scaled[:, :, i])
            samples.append(y_sample)
            
        return np.array(samples)


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    
    # Generate dummy data (2D end-effector positions -> 2 joint angles)
    n_samples = 200  # Smaller dataset for GPR (computationally intensive)
    X_train = np.random.randn(n_samples, 2)  # End-effector positions (x, y)
    y_train = np.random.randn(n_samples, 2)  # Joint angles (theta1, theta2)
    
    # Create and train model
    model = GPRModel(kernel_type='rbf', length_scale=1.0, alpha=1e-5)
    model.fit(X_train, y_train)
    
    # Test prediction with uncertainty
    X_test = np.random.randn(10, 2)
    predictions, uncertainties = model.predict(X_test, return_std=True)
    
    print("GPR Model Test")
    print("-" * 40)
    print(f"Training samples: {X_train.shape}")
    print(f"Test samples: {X_test.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"\nSample predictions with uncertainty (first 3):")
    for i in range(3):
        print(f"  Input: {X_test[i]}")
        print(f"  Output: {predictions[i]} ± {uncertainties[i]}")
    
    # Test sampling
    samples = model.sample(X_test[:1], n_samples=5)
    print(f"\n5 samples for first test input:")
    for i, sample in enumerate(samples):
        print(f"  Sample {i+1}: {sample[0]}")