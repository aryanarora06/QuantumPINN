# Quantum PINN: Solving the Schrödinger Equation

This project is a Physics-Informed Neural Network (PINN) built with PyTorch. It solves the 1D time-independent Schrödinger equation for both infinite and finite square wells to find the ground state energy and wavefunction.

## Features
* **Physics-Informed Loss:** Uses automatic differentiation to compute analytical derivatives, minimizing the residual of the Schrödinger equation.
* **Energy Eigenvalue Discovery:** Treats the ground state energy $E$ as a learnable parameter alongside the neural network weights.
* **Auto-Environment Detection:** Seamlessly adapts its display and output backend whether you run it locally, in Google Colab, or on Kaggle.
* **Stable Integration:** Uses `scipy.integrate.trapezoid` for robust wavefunction normalization during evaluation.

## Implementation Details

### 1. The Physics (Schrödinger Equation)
The model minimizes the residual of the 1D time-independent Schrödinger equation (in atomic units where $\hbar = m = 1$):

$$-\frac{1}{2}\frac{d^2\psi}{dx^2} + V(x)\psi - E\psi = 0$$

### 2. Network Architecture & Ansatz
The core neural network is a Multi-Layer Perceptron (MLP) consisting of:
* **Input Layer:** 1 node ($x$ coordinate)
* **Hidden Layers:** Two dense layers with 128 units each, utilizing `Tanh` activation functions for smooth, continuous second derivatives.
* **Output Layer:** 1 node (raw wavefunction output, $\text{NN}(x)$)

To enforce boundary conditions without relying solely on the loss function, the network uses a hard-constrained ansatz:
* **Infinite Well:** The wavefunction must be exactly zero at the boundaries ($x=0$ and $x=L$). The network outputs: 
  $$\psi(x) = x(L-x)\text{NN}(x)$$
* **Finite Well:** The wavefunction must decay smoothly to zero at infinity. The network outputs a Gaussian envelope:
  $$\psi(x) = e^{-x^2}\text{NN}(x)$$

### 3. Loss Function
The total loss consists of two competing terms:
1. **PDE Loss ($\mathcal{L}_{PDE}$):** Ensures the wavefunction obeys the Schrödinger equation. Computed as the Mean Squared Error (MSE) of the residual across collocation points.
2. **Normalization Loss ($\mathcal{L}_{norm}$):** Prevents the trivial solution ($\psi = 0$). It enforces $\int |\psi|^2 dx = 1$.

$$\mathcal{L}_{total} = \mathcal{L}_{PDE} + 100.0 \times \mathcal{L}_{norm}$$

*Note: A multiplier of 100.0 is applied to the normalization loss to prioritize avoiding the trivial solution early in training.*

### 4. Training Hyperparameters
* **Optimizer:** Adam
* **Learning Rate:** `1e-3`
* **Collocation Points:** 500 for infinite well, 1000 for finite well.
* **Epochs:** 8000

## Requirements
* `torch`
* `numpy`
* `scipy`
* `matplotlib`

## Usage
Run the script on Kaggle, Google Colab or locally on your machine.
