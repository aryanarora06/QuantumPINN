"""
Quantum PINN (Physics-Informed Neural Network)
Solves the time-independent Schrödinger equation for:
  - Infinite square well
  - Finite square well

Auto-detects environment: Local, Google Colab, or Kaggle.
"""

import sys
import os

# ─────────────────────────────────────────────
# 1. ENVIRONMENT DETECTION
# ─────────────────────────────────────────────

def detect_environment():
    """Return one of: 'colab', 'kaggle', 'local'."""
    
    # 1. Check Kaggle first! KAGGLE_KERNEL_RUN_TYPE is exclusively injected by Kaggle.
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
        
    # 2. Colab check goes second.
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"

    # 3. Default fallback.
    return "local"


ENV = detect_environment()
print(f"[env] Detected environment: {ENV.upper()}")


# ─────────────────────────────────────────────
# 2. ENVIRONMENT-SPECIFIC SETUP
# ─────────────────────────────────────────────

def setup_environment():
    """Configure display backend per environment (called before pyplot import)."""
    if ENV == "colab":
        import matplotlib
        matplotlib.use("module://matplotlib_inline.backend_inline")
        print("[env] Colab: using inline matplotlib backend.")

    elif ENV == "kaggle":
        import matplotlib
        try:
            matplotlib.use("module://matplotlib_inline.backend_inline")
        except Exception:
            matplotlib.use("Agg")
        print("[env] Kaggle: using inline/Agg matplotlib backend.")

    else:  # local
        import matplotlib
        if "DISPLAY" not in os.environ and sys.platform not in ("win32", "darwin"):
            matplotlib.use("Agg")
            print("[env] Local (headless): using Agg backend – plots will be saved to disk.")
        else:
            print("[env] Local: using default matplotlib GUI backend.")

setup_environment()


# ─────────────────────────────────────────────
# 3. IMPORTS 
# ─────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate  # Added SciPy integrate for stable trapezoid math
from scipy.optimize import fsolve


# ─────────────────────────────────────────────
# 4. OUTPUT HELPER
# ─────────────────────────────────────────────

def save_or_show(fig, filename):
    """Show inline or save to disk when headless."""
    import matplotlib
    headless = "agg" in matplotlib.get_backend().lower()

    if headless:
        out_dir = _output_dir()
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[output] Plot saved → {path}")
    else:
        plt.show()

    plt.close(fig)

def _output_dir():
    """Return (and create) the appropriate output directory."""
    if ENV == "kaggle":
        d = "/kaggle/working"
    elif ENV == "colab":
        d = "/content"
    else:
        try:
            base = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base = os.getcwd()
        d = os.path.join(base, "pinn_output")

    os.makedirs(d, exist_ok=True)
    return d


# ─────────────────────────────────────────────
# 5. CONFIGURATION
# ─────────────────────────────────────────────

# Explicit device definition for Cloud/GPU support
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[env] Using computing device: {DEVICE.type.upper()}")

V0_VALUE = 100.0
L        = 1.0
EPOCHS   = 8000
LR       = 1e-3


# ─────────────────────────────────────────────
# 6. MODEL & PHYSICS
# ─────────────────────────────────────────────

class QuantumPINN(nn.Module):
    def __init__(self, system):
        super().__init__()
        self.system = system
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )
        # Ensure the parameter starts on the correct hardware device
        self.E = nn.Parameter(torch.tensor([2.0], device=DEVICE))

    def forward(self, x):
        psi_raw = self.net(x)
        if self.system == "infinite":
            return x * (L - x) * psi_raw
        else:
            # Maintained the Gaussian envelope constraint
            return torch.exp(-x ** 2) * psi_raw


def get_potential(x, system):
    if system == "infinite":
        return torch.zeros_like(x)
    return torch.where(
        torch.abs(x) < 0.5,
        torch.zeros_like(x),
        torch.full_like(x, V0_VALUE),
    )


def get_analytical_energy(system):
    if system == "infinite":
        return (np.pi ** 2) / (2 * L ** 2)

    def func(E):
        E = np.clip(E, 1e-6, V0_VALUE - 1e-6)
        k     = np.sqrt(2 * E)
        alpha = np.sqrt(2 * (V0_VALUE - E))
        return k * np.tan(k * 0.5) - alpha

    return fsolve(func, 4.0)[0]


# ─────────────────────────────────────────────
# 7. TRAIN + EVALUATE
# ─────────────────────────────────────────────

def run_system(system):
    print(f"\n{'='*50}")
    print(f"Training PINN for {system} well...")
    print(f"{'='*50}\n")

    # Move model to target device
    model     = QuantumPINN(system).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if system == "infinite":
        x_range = (0, L)
        n_collo = 500
    else:
        x_range = (-1.5, 1.5)
        n_collo = 1000

    # Ensure collocation points are on the target device
    x_collo = (torch.linspace(x_range[0], x_range[1], n_collo, device=DEVICE)
               .view(-1, 1)
               .requires_grad_(True))

    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad()

        if x_collo.grad is not None:
            x_collo.grad.zero_()

        psi    = model(x_collo)
        psi_x  = torch.autograd.grad(psi,   x_collo, torch.ones_like(psi),   create_graph=True)[0]
        psi_xx = torch.autograd.grad(psi_x, x_collo, torch.ones_like(psi_x), create_graph=True)[0]

        V        = get_potential(x_collo, system)
        residual = -0.5 * psi_xx + V * psi - model.E * psi
        loss_pde = torch.mean(residual ** 2)

        domain_width = x_range[1] - x_range[0]
        loss_norm    = (torch.mean(psi ** 2) * domain_width - 1.0) ** 2
        total_loss   = loss_pde + 100.0 * loss_norm

        total_loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch:5d} | E: {model.E.item():.4f} | Loss: {total_loss.item():.6f}")

    # ── Results ──────────────────────────────────────────────
    pred_e = model.E.item()
    true_e = get_analytical_energy(system)

    with torch.no_grad():
        # Evaluate on the target device, then transfer to CPU for numpy logic
        x_test    = torch.linspace(x_range[0], x_range[1], 500, device=DEVICE).view(-1, 1)
        psi_final = model(x_test).cpu().numpy().flatten()
        x_test_np = x_test.cpu().numpy().flatten()

        # Using SciPy's integrate to avoid NumPy versioning errors
        norm_factor = np.sqrt(integrate.trapezoid(psi_final ** 2, x_test_np))
        psi_final  /= norm_factor
        if np.sum(psi_final) < 0:
            psi_final = -psi_final

    print(f"\nResults for {system} well:")
    print(f"  Predicted E:  {pred_e:.6f}")
    print(f"  Analytical E: {true_e:.6f}")
    print(f"  Error:        {abs(pred_e - true_e) / true_e * 100:.4f}%")

    # ── Plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_test_np, psi_final, label="PINN Wavefunction", lw=2)

    if system == "finite":
        ax.axvline(-0.5, color="k", ls="--", alpha=0.3)
        ax.axvline( 0.5, color="k", ls="--", alpha=0.3)
        ax.fill_between([-1.5, -0.5], -0.2, max(psi_final) + 0.2, alpha=0.1)
        ax.fill_between([ 0.5,  1.5], -0.2, max(psi_final) + 0.2, alpha=0.1)

    ax.set_title(f"Ground State: {system.capitalize()} Well")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\psi(x)$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_or_show(fig, f"pinn_{system}_well.png")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    for system in ("infinite", "finite"):
        run_system(system)
