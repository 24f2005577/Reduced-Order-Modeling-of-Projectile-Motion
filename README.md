# Reduced-Order Modeling of Projectile Motion using PCA

## Overview

This project investigates whether classical projectile motion can be accurately modeled in a lower-dimensional latent space using Principal Component Analysis (PCA) followed by polynomial regression.

Rather than rediscovering known physics, the project explores a deeper modeling question:

> Does variance-preserving dimensionality reduction also preserve the information required to predict system dynamics?

---

## Motivation

PCA is widely used for dimensionality reduction and reduced-order modeling. However, PCA preserves variance, not necessarily predictive structure.

Projectile motion is an ideal test system because:
- The governing equations are known
- Data can be generated without noise
- Any modeling error is methodological, not experimental

---

## Physics Background

Projectile motion without air resistance is governed by:

y(t) = u sin(θ) t − (1/2) g t²

where:
- u is the initial velocity  
- θ is the launch angle  
- t is time  
- g = 9.8 m/s²  

---

## Data Generation

Synthetic, noise-free data is generated under physical constraints:

- u ∈ [20, 50]
- θ ∈ [15°, 75°]
- t ∈ [0, t_f], where  
  t_f = 2u sin(θ) / g

Each data sample consists of:

X = [u, θ, t]  
y = vertical displacement  

This ensures all generated trajectories are physically valid.

---

## Methodology

### 1. Principal Component Analysis (PCA)

- Inputs: (u, θ, t)
- Standardization (zero mean, unit variance)
- Covariance matrix eigen-decomposition
- Projection onto the top 2 principal components

Result:

(u, θ, t) → (z₁, z₂)

---

### 2. Polynomial Regression in Latent Space

Polynomial regression is performed to predict y from the PCA-transformed variables (z₁, z₂).

- Quadratic and cubic polynomial models are tested
- Least-squares regression implemented using NumPy
- No machine learning libraries are used

---

### 3. Evaluation Metrics

Model performance is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of determination (R²)

---

## Results

Despite PCA capturing most of the variance in the input space, the regression model achieves:

R² ≈ 0.5

This result is consistent across different sample sizes and polynomial degrees.

---

## Key Insight

> Preserving variance does not guarantee preserving predictive structure.

Although PCA identifies a low-dimensional subspace that represents the data well, this subspace is not sufficient to reconstruct the nonlinear dynamics of projectile motion.

### Why this happens

- Projectile motion evolves on a curved nonlinear manifold
- PCA is a linear transformation
- Compressing from 3D to 2D causes irreversible information loss
- Polynomial regression cannot recover lost structure

---

## Core Conclusion

> PCA identifies a two-dimensional variance-dominated subspace of projectile motion parameters, but this subspace is insufficient for accurate prediction of the system’s dynamics.

This highlights a fundamental limitation of linear dimensionality reduction when applied to nonlinear physical systems.

---

## Project Structure

.
├── dataset_genarator.py      # Physics-consistent data generation  
├── PCA_tranformer.py         # Standardized PCA implementation  
├── polynomial_regression.py  # Polynomial regression and evaluation  
├── README.md  

---

## How to Run

```bash
python polynomial_regression.py
