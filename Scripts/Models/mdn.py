import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MDNNetwork(nn.Module):
    """Mixture Density Network architecture."""
    
    def __init__(self, input_dim, output_dim, hidden_layers=[128, 64], 
                 n_mixtures=5):
        super(MDNNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_mixtures = n_mixtures
        
        # Build hidden layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        self.hidden = nn.Sequential(*layers)
        
        # Output layers for mixture parameters
        # Pi: mixture weights
        self.fc_pi = nn.Linear(prev_dim, n_mixtures)
        # Mu: means for each mixture component
        self.fc_mu = nn.Linear(prev_dim, n_mixtures * output_dim)
        # Sigma: standard deviations for each mixture component
        self.fc_sigma = nn.Linear(prev_dim, n_mixtures * output_dim)
        
    def forward(self, x):
        """Forward pass through the network."""
        h = self.hidden(x)
        
        # Mixture weights (apply softmax for normalization)
        pi = torch.softmax(self.fc_pi(h), dim=1)
        
        # Means
        mu = self.fc_mu(h)
        mu = mu.view(-1, self.n_mixtures, self.output_dim)
        
        # Standard deviations (apply exponential to ensure positive)
        sigma = torch.exp(self.fc_sigma(h))
        sigma = sigma.view(-1, self.n_mixtures, self.output_dim)
        
        return pi, mu, sigma


class MDNModel:
    """Mixture Density Network model for inverse kinematics."""
    
    def __init__(self, input_dim=2, output_dim=2, hidden_layers=[128, 64],
                 n_mixtures=5, learning_rate=0.001, epochs=100, batch_size=32):
        """
        Initialize MDN model.
        
        Args:
            input_dim: Input dimension (end-effector positions)
            output_dim: Output dimension (joint angles)
            hidden_layers: List of hidden layer sizes
            n_mixtures: Number of Gaussian mixture components
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.n_mixtures = n_mixtures
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
        
    def mdn_loss(self, pi, mu, sigma, y):
        """
        Negative log-likelihood loss for MDN.
        
        Args:
            pi: Mixture weights
            mu: Mixture means
            sigma: Mixture standard deviations
            y: Target values
        """
        # Expand y to match mixture dimensions
        y = y.unsqueeze(1).expand_as(mu)
        
        # Calculate Gaussian probability for each mixture component
        m = torch.distributions.Normal(mu, sigma)
        log_prob = m.log_prob(y)
        log_prob = torch.sum(log_prob, dim=2)  # Sum over output dimensions
        
        # Weight by mixture coefficients
        weighted_logprob = log_prob + torch.log(pi + 1e-8)
        
        # LogSumExp for numerical stability
        log_sum = torch.logsumexp(weighted_logprob, dim=1)
        
        return -torch.mean(log_sum)
        
    def fit(self, X_train, y_train):
        """
        Train the MDN model.
        
        Args:
            X_train: Input features (end-effector positions)
            y_train: Target values (joint angles)
        """
        # Convert to numpy arrays
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
        self.model = MDNNetwork(
            self.input_dim,
            self.output_dim,
            self.hidden_layers,
            self.n_mixtures
        ).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                pi, mu, sigma = self.model(batch_X)
                
                # Compute loss
                loss = self.mdn_loss(pi, mu, sigma, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
                
    def predict(self, X_test, mode='mean'):
        """
        Make predictions.
        
        Args:
            X_test: Input features for prediction
            mode: 'mean' for weighted mean, 'sample' for sampling, 'mode' for most likely
            
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
        
        with torch.no_grad():
            pi, mu, sigma = self.model(X_tensor)
            
            if mode == 'mean':
                # Weighted mean of all mixture components
                y_pred_norm = torch.sum(pi.unsqueeze(2) * mu, dim=1)
            elif mode == 'mode':
                # Select component with highest weight
                max_pi_idx = torch.argmax(pi, dim=1)
                y_pred_norm = torch.stack([
                    mu[i, max_pi_idx[i], :] for i in range(mu.size(0))
                ])
            elif mode == 'sample':
                # Sample from the mixture
                y_pred_norm = self._sample_from_mixture(pi, mu, sigma)
            else:
                raise ValueError(f"Unknown prediction mode: {mode}")
                
            # Denormalize
            y_pred_norm = y_pred_norm.cpu().numpy()
            y_pred = y_pred_norm * self.y_std + self.y_mean
            
        return y_pred
    
    def _sample_from_mixture(self, pi, mu, sigma):
        """Sample from the Gaussian mixture."""
        batch_size = pi.size(0)
        samples = []
        
        for i in range(batch_size):
            # Sample mixture component
            component = torch.multinomial(pi[i], 1).item()
            
            # Sample from chosen Gaussian
            m = torch.distributions.Normal(mu[i, component], sigma[i, component])
            sample = m.sample()
            samples.append(sample)
            
        return torch.stack(samples)
    
    def sample(self, X_test, n_samples=10):
        """
        Sample multiple joint angle solutions.
        
        Args:
            X_test: Input features (end-effector positions)
            n_samples: Number of samples to generate
            
        Returns:
            Array of sampled joint angles
        """
        samples = []
        for _ in range(n_samples):
            sample = self.predict(X_test, mode='sample')
            samples.append(sample)
            
        return np.array(samples).transpose(1, 0, 2)


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate dummy data (2D end-effector positions -> 2 joint angles)
    n_samples = 1000
    X_train = np.random.randn(n_samples, 2)  # End-effector positions (x, y)
    y_train = np.random.randn(n_samples, 2)  # Joint angles (theta1, theta2)
    
    # Create and train model
    model = MDNModel(
        input_dim=2,
        output_dim=2,
        hidden_layers=[128, 64],
        n_mixtures=5,
        learning_rate=0.001,
        epochs=100,
        batch_size=32
    )
    model.fit(X_train, y_train)
    
    # Test different prediction modes
    X_test = np.random.randn(5, 2)
    
    print("\nMDN Model Test")
    print("-" * 40)
    print(f"Training samples: {X_train.shape}")
    print(f"Number of mixtures: {model.n_mixtures}")
    print(f"Test samples: {X_test.shape}")
    
    # Mean prediction
    pred_mean = model.predict(X_test, mode='mean')
    print(f"\nMean predictions (first 3):")
    for i in range(3):
        print(f"  Input: {X_test[i]}, Output: {pred_mean[i]}")
    
    # Sample multiple solutions
    samples = model.sample(X_test[:1], n_samples=5)
    print(f"\nMultiple sampled solutions for first test input:")
    print(f"Input: {X_test[0]}")
    for i in range(5):
        print(f"  Solution {i+1}: {samples[0, i]}")