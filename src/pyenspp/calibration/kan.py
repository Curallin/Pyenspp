# calibration/kan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.special as sp
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Union, List
import copy
import os
import torch.nn.init as init
import matplotlib.pyplot as plt

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=8, spline_order=3, scale_base=1.0, scale_spline=1.0, scale_noise=0.1):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        # 1. Base Function's weights
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # 2. Spline Function's weights
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        # grid registraion
        grid = torch.linspace(-1, 1, grid_size + 2 * spline_order + 1)
        self.register_buffer("grid", grid)
        
        self.reset_parameters(scale_noise)

    def reset_parameters(self, scale_noise):
        # initialize weights with uniform and normal distributions
        init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        std = scale_noise / np.sqrt(self.in_features)
        init.normal_(self.spline_weight, mean=0.0, std=std)

    def b_splines(self, x):
        # calculate B-spline basis functions for input x
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[: -(k + 1)]) / (grid[k:-1] - grid[: -(k + 1)]) * bases[:, :, :-1]
                + (grid[k + 1 :] - x) / (grid[k + 1 :] - grid[1:-k]) * bases[:, :, 1:]
            )
        return bases

    def forward(self, x):
        # x shape: (batch, in_features)

        # --- 1. Base Function  ---
        base_output = F.linear(F.silu(x), self.base_weight)

        # --- 2. Spline Function  ---
        spline_basis = self.b_splines(x) 
        spline_output = torch.einsum("big,oig->bo", spline_basis, self.spline_weight)
        
        # --- 3. Combine two functions ---
        return self.scale_base * base_output + self.scale_spline * spline_output

class KANCSGDNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=20, grid_size=10):
        super(KANCSGDNet, self).__init__()
        
        # Input normalization for distribution stability. used in 《A novel hybrid artificial neural network - parametric scheme for postprocessing medium-range precipitation forecasts》
        self.ln_in = nn.LayerNorm(input_dim)
        
        # KAN layers: Base (SiLU) + Spline components
        self.kan1 = KANLayer(input_dim, hidden_dim, grid_size=grid_size)
        self.kan2 = KANLayer(hidden_dim, 3, grid_size=grid_size)
        
        # Learnable bias for CSGD parameters
        self.bias = nn.Parameter(torch.zeros(3))
        with torch.no_grad():
            self.bias[1] = 0.5  # Initial mu
            self.bias[2] = 0.5  # Initial sigma

    def forward(self, x):
        # 1. Pre-processing：not recommended
        # x = self.ln_in(x)

        # 2. Forward pass through KAN layers
        x = self.kan1(x)
        raw_out = self.kan2(x)
        
        # 3. Output layer processing
        raw_out = raw_out + self.bias
        o1, o2, o3 = raw_out[:, 0], raw_out[:, 1], raw_out[:, 2]
        
        # Parameter constraints for CSGD
        delta = -F.softplus(o1)
        mu = F.softplus(o2) + 1e-4
        sigma = F.softplus(o3) + 1e-4
        
        return mu, sigma, delta

# ==========================================
# 2. Core Mathematical Components (Autograd)
# ==========================================
class SafeGammaInc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, x):
        # Prevent zero inputs
        a = torch.clamp(a, min=1e-5)
        x = torch.clamp(x, min=1e-7)
        ctx.save_for_backward(a, x)
        return torch.special.gammainc(a, x)

    @staticmethod
    def backward(ctx, grad_output):
        a, x = ctx.saved_tensors
        eps = 1e-4
        
        # Analytical gradient for x (Regularized Incomplete Gamma)
        log_grad_x = (a - 1) * torch.log(x) - x - torch.lgamma(a)
        grad_x = torch.exp(log_grad_x)
        
        # Numerical gradient for 'a' via central difference
        a_plus = a + eps
        a_minus = torch.clamp(a - eps, min=1e-5)
        grad_a = (torch.special.gammainc(a_plus, x) - torch.special.gammainc(a_minus, x)) / (2 * eps)
        
        # Gradient clipping for stability
        return (grad_a * grad_output).clamp(-1, 1), (grad_x * grad_output).clamp(-1, 1)

def safe_gammainc(a, x):
    return SafeGammaInc.apply(a, x)

class CSGD_CRPS_Loss(nn.Module):
    def forward(self, y_obs, mu, sigma, delta):
        sigma = torch.clamp(sigma, min=1e-6)
        mu = torch.clamp(mu, min=1e-6)
        
        # CSGD to Gamma conversion: k (shape), theta (scale)
        k = (mu / sigma) ** 2
        theta = (sigma ** 2) / mu
        y_tilde = torch.clamp((y_obs - delta) / theta, min=0.0)
        c_tilde = torch.clamp(-delta / theta, min=0.0)
        
        # CDF values for CRPS calculation
        F_k_y = safe_gammainc(k, y_tilde)
        F_k_c = safe_gammainc(k, c_tilde)
        F_k1_y = safe_gammainc(k + 1, y_tilde)
        F_k1_c = safe_gammainc(k + 1, c_tilde)
        F_2k_2c = safe_gammainc(2 * k, 2 * c_tilde)
        
        # Beta function component for CRPS
        lbeta_val = torch.lgamma(torch.tensor(0.5, device=y_obs.device)) + torch.lgamma(k + 0.5) - torch.lgamma(k + 1.0)
        beta_val = torch.exp(lbeta_val)
        
        # Analytical CRPS formula for CSGD
        term1 = theta * y_tilde * (2 * F_k_y - 1)
        term2 = -theta * c_tilde * (F_k_c ** 2)
        term3 = theta * k * (1 + 2 * F_k_c * F_k1_c - F_k_c**2 - 2 * F_k1_y)
        term4 = -(theta * k / torch.pi) * beta_val * (1 - F_2k_2c)
        return torch.mean(term1 + term2 + term3 + term4)

# ==========================================
# 3. Training Utilities
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            if self.verbose: print(f"✅ Val loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose: print(f"⚠️ EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                
    def restore_best_weights(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

# ==========================================
# 4. Main Wrapper Class
# ==========================================
class KANCSGDRegression:
    def __init__(
        self, 
        hidden_dim: int = 8, 
        grid_size: int = 10,
        lr: float = 0.005, 
        n_epochs: int = 500,
        batch_size: int = 512,
        patience: int = 15,
        verbose: bool = True,
        random_state: int = 42,
        save_path: Optional[str] = "best_kan_model.pth"
    ):
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state
        self.save_path = save_path
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_fitted = False
        self.train_losses = []
        self.val_losses = []
        self.input_dim = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KANCSGDRegression":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.input_dim = X.shape[1]
        
        # Train/Validation split (80/20)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split = int(0.8 * len(X))
        train_idx, val_idx = indices[:split], indices[split:]
        
        X_train = torch.tensor(X[train_idx]).to(self.device)
        y_train = torch.tensor(y[train_idx]).to(self.device)
        X_val = torch.tensor(X[val_idx]).to(self.device)
        y_val = torch.tensor(y[val_idx]).to(self.device)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        
        self.model = KANCSGDNet(self.input_dim, self.hidden_dim, self.grid_size).to(self.device)
        # Use AdamW for KAN optimization
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = CSGD_CRPS_Loss()
        early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose)
        
        for epoch in range(self.n_epochs):
            self.model.train()
            train_loss_accum = 0.0
            for bx, by in train_loader:
                optimizer.zero_grad()
                mu, sigma, delta = self.model(bx)
                loss = criterion(by, mu, sigma, delta)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item() * bx.size(0)
            
            avg_train_loss = train_loss_accum / len(train_idx)
            self.train_losses.append(avg_train_loss)
            
            self.model.eval()
            with torch.no_grad():
                mu_v, sigma_v, delta_v = self.model(X_val)
                val_loss = criterion(y_val, mu_v, sigma_v, delta_v).item()
                self.val_losses.append(val_loss)
            
            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train={avg_train_loss:.4f}, Val={val_loss:.4f}")
            
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop: break
                
        early_stopping.restore_best_weights(self.model)
        self.is_fitted = True
        if self.save_path: self.save_model(self.save_path)
        return self

    def correct(
        self, 
        X: np.ndarray, 
        n_members: int = 100,
        quantiles: Union[float, list, np.ndarray] = None,
        mode: str = 'quantile'
    ) -> np.ndarray:
        """
        Post-process input X to generate ensemble members via Inverse Transform Sampling.
        """
        if not self.is_fitted: 
            raise RuntimeError("Model not fitted.")
        
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1: X = X.reshape(1, -1)
        N = X.shape[0]

        # 1. Determine quantile matrix (N, M)
        if quantiles is not None:
            qs = np.atleast_1d(quantiles).astype(np.float64)
            M = len(qs)
            q_2d = np.tile(qs, (N, 1)) 
        else:
            M = n_members or 50
            if mode == 'quantile':
                # Equidistant quantiles [1/(M+1), ..., M/(M+1)]
                qs = np.linspace(1 / (M + 1), M / (M + 1), M)
                q_2d = np.tile(qs, (N, 1))
            elif mode == 'random':
                # Independent random sampling for each forecast
                q_2d = np.random.rand(N, M)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        # Clip quantiles to avoid numerical issues
        q_2d = np.clip(q_2d, 1e-6, 1-1e-6)

        # 2. Predict CSGD parameters via KAN
        self.model.eval()
        with torch.no_grad():
            mu_t, sigma_t, delta_t = self.model(torch.tensor(X).to(self.device))
            mu, sigma, delta = mu_t.cpu().numpy(), sigma_t.cpu().numpy(), delta_t.cpu().numpy()

        # 3. Derive Gamma shape (k) and scale (theta)
        k = (mu / np.maximum(sigma, 1e-6)) ** 2
        theta = (sigma ** 2) / np.maximum(mu, 1e-6)

        # Broadcast parameters for vectorized calculation
        k_2d = k[:, None]
        theta_2d = theta[:, None]
        delta_2d = delta[:, None]

        # 4. Calculate zero-probability threshold (P(Y=0))
        p_zero = sp.gammainc(k_2d, -delta_2d / theta_2d)
        
        # 5. Inverse CDF transformation
        result = np.zeros((N, M))
        # Only compute inverse Gamma where quantile exceeds P(Y=0)
        mask_pos = q_2d > p_zero
        
        if np.any(mask_pos):
            k_broad = np.broadcast_to(k_2d, (N, M))[mask_pos]
            theta_broad = np.broadcast_to(theta_2d, (N, M))[mask_pos]
            delta_broad = np.broadcast_to(delta_2d, (N, M))[mask_pos]
            q_broad = q_2d[mask_pos]
            
            # y = theta * GammaInv(q, k) + delta
            result[mask_pos] = theta_broad * sp.gammaincinv(k_broad, q_broad) + delta_broad
            
        return np.maximum(result, 0.0)
    
    def plot_loss(self):
        if not self.is_fitted: return
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Val')
        plt.legend(); plt.show()

    def save_model(self, path: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'grid_size': self.grid_size
        }
        torch.save(checkpoint, path)

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.input_dim, self.hidden_dim, self.grid_size = ckpt['input_dim'], ckpt['hidden_dim'], ckpt['grid_size']
        self.model = KANCSGDNet(self.input_dim, self.hidden_dim, self.grid_size).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.is_fitted = True
        return self