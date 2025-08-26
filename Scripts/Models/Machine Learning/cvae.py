import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CVAENetwork(nn.Module):
    """Conditional VAE network architecture."""
    
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dim=128):
        super(CVAENetwork, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Encoder: q(z|x,c)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: p(x|z,c)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x, c):
        """Encode input and condition to latent distribution parameters."""
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        """Decode from latent space conditioned on input."""
        inputs = torch.cat([z, c], dim=1)
        return self.decoder(inputs)
    
    def forward(self, x, c):
        """Forward pass through the network."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar


class CVAEModel:
    """Conditional Variational Autoencoder model for inverse kinematics."""
    
    def __init__(self, input_dim=2, output_dim=2, latent_dim=10, hidden_dim=128,
                 learning_rate=0.001, epochs=100, batch_size=32, beta=1.0):
        """
        Initialize CVAE model.
        
        Args:
            input_dim: Input dimension (end-effector position)
            output_dim: Output dimension (joint angles)
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer size
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            beta: Weight for KL divergence term
        """
        self.input_dim = output_dim  # Joint angles are input to VAE
        self.condition_dim = input_dim  # End-effector positions are conditions
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta = beta
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Normalization parameters
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        
    def loss_function(self, x_recon, x, mu, logvar):
        """VAE loss function with reconstruction and KL divergence."""
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss
        
    def fit(self, X_train, y_train):
        """
        Train the CVAE model.
        
        Args:
            X_train: Input features (end-effector positions) - used as conditions
            y_train: Target values (joint angles) - reconstructed by VAE
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
        dataset = TensorDataset(y_tensor, X_tensor)  # Note: y is input, X is condition
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = CVAENetwork(
            self.input_dim,
            self.condition_dim,
            self.latent_dim,
            self.hidden_dim
        ).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_y, batch_X in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                y_recon, mu, logvar = self.model(batch_y, batch_X)
                
                # Compute loss
                loss = self.loss_function(y_recon, batch_y, mu, logvar)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / (len(dataloader) * self.batch_size)
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
                
    def predict(self, X_test, n_samples=1):
        """
        Make predictions by sampling from the posterior.
        
        Args:
            X_test: Input features (end-effector positions)
            n_samples: Number of samples to generate per input
            
        Returns:
            Mean of predicted joint angles (or all samples if n_samples > 1)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        self.model.eval()
        
        # Convert and normalize
        X_test = np.array(X_test, dtype=np.float32)
        X_test_norm = (X_test - self.X_mean) / self.X_std
        X_tensor = torch.FloatTensor(X_test_norm).to(self.device)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Sample from prior
                z = torch.randn(X_tensor.size(0), self.latent_dim).to(self.device)
                
                # Decode conditioned on input
                y_pred_norm = self.model.decode(z, X_tensor)
                y_pred = y_pred_norm.cpu().numpy() * self.y_std + self.y_mean
                predictions.append(y_pred)
        
        predictions = np.array(predictions)
        
        if n_samples == 1:
            return predictions[0]
        else:
            # Return mean and all samples
            return predictions.mean(axis=0), predictions
    
    def sample(self, X_test, n_samples=10):
        """
        Sample multiple joint angle solutions for given end-effector positions.
        
        Args:
            X_test: Input features (end-effector positions)
            n_samples: Number of samples to generate
            
        Returns:
            Array of sampled joint angles
        """
        _, samples = self.predict(X_test, n_samples=n_samples)
        return samples.transpose(1, 0, 2)  # Shape: (n_test, n_samples, output_dim)


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate dummy data (2D end-effector positions -> 2 joint angles)
    n_samples = 1000
    X_train = np.random.randn(n_samples, 2)  # End-effector positions (x, y)
    y_train = np.random.randn(n_samples, 2)  # Joint angles (theta1, theta2)
    
    # Create and train model
    model = CVAEModel(
        input_dim=2,
        output_dim=2,
        latent_dim=10,
        hidden_dim=128,
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        beta=0.1
    )
    model.fit(X_train, y_train)
    
    # Test prediction
    X_test = np.random.randn(5, 2)
    predictions = model.predict(X_test)
    
    print("\nCVAE Model Test")
    print("-" * 40)
    print(f"Training samples: {X_train.shape}")
    print(f"Test samples: {X_test.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Test sampling for multiple solutions
    samples = model.sample(X_test[:1], n_samples=5)
    print(f"\nMultiple solutions for first test input:")
    print(f"Input: {X_test[0]}")
    for i in range(5):
        print(f"  Solution {i+1}: {samples[0, i]}")