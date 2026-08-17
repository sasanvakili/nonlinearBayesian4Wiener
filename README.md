# Dual Bayesian Affine Estimators for Parameter Learning

Code accompanying the paper:

> S. Vakili, D. Woonings, P. Paruchuri, and P. Mohajerin Esfahani, *Dual Bayesian Affine Estimators for Parameter Learning: A Fixed-Point Characterization*.

This repository provides a MATLAB library for Bayesian estimation and active input design in Wiener models. It includes:

- An affine minimum mean-squared-error (MMSE) estimator for the unknown observation-model parameters;
- Dual estimators for joint learning of the latent state and unknown parameters;
- An active-learning algorithm for optimal input design, based on:
  > S. Vakili, M. Mazo Jr., and P. Mohajerin Esfahani, *Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model*, arXiv preprint arXiv:2504.05490, 2025.

## Overview

The Wiener model considered here consists of known discrete-time, linear time-varying state dynamics and an observation model with unknown parameters:

$$
\mathrm{x} _{t+1} = \mathrm{A} _{t} \mathrm{x} _{t} + \mathrm{B} _{t} \mathrm{u} _{t} + \mathrm{w} _{t+1},
$$
$$
\mathrm{y} _{t} = \sum\limits _{n=0}^{N} \theta _{n} \phi _{n}( \mathrm{x} _{t}) + \mathrm{v} _{t} = \langle {\phi(\mathrm{x} _{t}), \theta} \rangle + \mathrm{v} _{t},
$$

where
*   $t \in \\{ 0, \\ldots, T \\}$
*   $\mathrm{x} _{t} \in \mathbb{R} ^{n _{\mathrm{x}}}$ is the state vector.
*   $\mathrm{A} _{t} \in \mathbb{R} ^{n _{\mathrm{x}} \times n _{\mathrm{x}} }$ is the state transition matrix.
*   $\mathrm{B} _{t} \in \mathbb{R} ^{n _{\mathrm{x}} \times n _{\mathrm{u}}}$ is the input matrix.
*   $\mathrm{u} _{t} \in \mathbb{R} ^{n _{\mathrm{u}}}$ is the input vector.
*   $\mathrm{w} _{t+1} \in \mathbb{R} ^{n _{\mathrm{x}}}$ is the process noise.
*   $\mathrm{y} _{t} \in \mathbb{R}$ is the scalar output measurement.
*   $\mathrm{v} _{t} \in \mathbb{R}$  is the measurement noise.
*   $\theta = \[ \theta _{0}, \ldots, \theta _{N} \] ^{\mathsf{T}}$ is the vector of all _unknown_ parameters.
*   $\phi(.)$ is the _known_ basis functions.
  
The _known_ basis functions $\phi _{n}( \mathrm{x} _{t})$ are defined as the following Fourier bases:

$$ 
\begin{cases}
\phi_ {0} ( \mathrm{x} _{t} ) = 1 & n = 0 \\
\phi _{n}( \mathrm{x} _{t} ) = \sum\limits _{\ell \in \\{ -1, 1 \\}} \mathrm{exp} ( j \langle { \ell f _{n} , \mathrm{x} _{t} } \rangle ) & n \geq 1,
\end{cases}
$$

with _known_ frequencies $f _{n} \in \mathbb{R} ^{n _{\mathrm{x}}}$, where $n$ denotes the frequency index, and $\phi(\mathrm{x} _{t}) = \[ \phi _{0}(\mathrm{x} _{t}), \ldots, \phi _{N}(\mathrm{x} _{t}) \] ^{\mathsf{T}}$ is the vector of basis functions evaluated at $\mathrm{x} _{t}$.

The models are then represented in _lifted matrix_ form. The output function is

$$
\mathrm{y} = \Phi ^{\mathsf{T}} \theta + \mathrm{v},
$$

where
*   $\mathrm{y} = \[ \mathrm{y} _{0}, \ldots, \mathrm{y} _{T} \] ^{\mathsf{T}}$ is the vector of measurements.
*   $\Phi = \[ \phi(\mathrm{x} _{0}), \ldots, \phi(\mathrm{x} _{T}) \]$ is the basis aggregation matrix.
*   $\theta = \[ \theta _{0}, \ldots, \theta _{N} \] ^{\mathsf{T}}$ is the vector of all _unknown_ parameters.
*   $\mathrm{v} = \[ \mathrm{v} _{0}, \ldots, \\mathrm{v} _{T} \] ^{\mathsf{T}}$ is the measurement noise vector, where $\mathrm{v} \sim \mathcal{N} ( 0, \Sigma _{\mathrm{v}})$ and $\Sigma _{\mathrm{v}} = \mathrm{diag}( \Sigma _{\mathrm{v} _{0}}^{2}, \ldots, \Sigma _{\mathrm{v} _{T}}^{2})$.

The prior information about $\theta$ is characterized by a probability distribution defined by its mean and covariance, $\mathbb{P} ( \mu _{\theta}, \Sigma _{\theta})$.

The process model is represented as

$$
\mathrm{x} = \mathrm{A} (\mathrm{B} \mathrm{u} + \mathrm{w}),
$$

where
*   $\mathrm{x} = \[ \mathrm{x} _{0} ^{\mathsf{T}}, \ldots, \mathrm{x} _{T} ^{\mathsf{T}} \] ^{\mathsf{T}}$ is the system states vector.
*   $\mathrm{u} = \[ \mu _{\mathrm{x} _{0}} ^{\mathsf{T}}, \mathrm{u} _{0} ^{\mathsf{T}}, \ldots, \mathrm{u} _{T-1} ^{\mathsf{T}} \] ^{\mathsf{T}}$ is the input vector, with the mean of the initial state $\mu _{\mathrm{x} _{0}}$ as its first element.
*   $\mathrm{w} = \[ \mathrm{w} _{0} ^{\mathsf{T}}, \mathrm{w} _{1} ^{\mathsf{T}}, \ldots, \mathrm{w} _{T} ^{\mathsf{T}} \] ^{\mathsf{T}}$ is the noise vector, where $\mathrm{w} _{0} \sim \mathcal{N} (0, \Sigma _{\mathrm{x} _{0}} )$ and $\mathrm{w} \sim \mathcal{N} ( 0 , \Sigma _{\mathrm{w}} )$. The covariance matrix is diagonal only if $\mathrm{w} _{t}$ are independent, i.e., $\Sigma _{\mathrm{w}} = \mathrm{diag} ( \Sigma _{\mathrm{x} _{0}}, \Sigma _{\mathrm{w} _{1}}, \ldots, \Sigma _{\mathrm{w} _{T}} )$.

The matrices $\mathrm{A}$ and $\mathrm{B}$ have the following lower triangular and block-diagonal structures, respectively:

$$
\mathrm{A} = \begin{bmatrix} \mathbb{I} & 0 & 0 & \ldots & 0 \\ 
\mathrm{A} _0 & \mathbb{I} & 0 & \ldots & \vdots \\ 
\mathrm{A} _1 \mathrm{A} _0 & \mathrm{A} _1 & \mathbb{I} & \ddots & \vdots \\ 
\vdots & \vdots &  \vdots & \ddots & 0 \\ 
\prod\limits _{i=0} ^{T-1} \mathrm{A} _{i} & \prod\limits _{i=1} ^{T-1} \mathrm{A} _{i} &  \ldots & \mathrm{A} _{T-1} & \mathbb{I}
\end{bmatrix}, \qquad 
\mathrm{B} = \mathrm{diag} ( \mathbb{I}, \mathrm{B} _{0}, \ldots, \mathrm{B} _{T-1}).
$$

This library uses these structures to compute the affine MMSE estimate $`\hat{\theta}_{\mathrm{af}}(\mathrm{y})`$ and the nonlinear Bayesian estimate $`\hat{\theta}_{\mathrm{nl}}(\mathrm{y})`$, obtained using either the **dual basis-parameter** (DB-P) or **dual state-parameter**
(DS-P) estimator. For further details, see the paper [*Nonlinear Bayesian Estimator for Parameter Learning:
A Fixed-Point Characterization*](https://arxiv.org/pdf/2606.10111).

## Requirements

The `nonlinearBayesian4Wiener` library is implemented in MATLAB and has
no third-party dependencies.

- **Core dependency:** MATLAB (R2020b or newer recommended).

- **Conditionally required:** MATLAB Optimization Toolbox.
  - Required only when the active-learning optimization method is set to `fmincon`.
  - The adaptive optimization method does not require this toolbox.
  - To check whether it is installed, run:

    ```matlab
    ver('optim')
    ```

- **Optional:** MATLAB Parallel Computing Toolbox.
  - Enables parallel execution of `parfor` loops for faster computation.
  - To check whether it is installed, run:

    ```matlab
    ver('parallel')
    ```

  - Without this toolbox, `parfor` loops can run serially, so the library
    remains usable but may require substantially longer computation time.

## Modes of Operation

The `nonlinearBayesian4Wiener` library has three modes of operation:

1.  `affineMMSEestimate`: Solves for the parameters of the **optimal Bayesian affine MMSE estimator** and the **theta estimates** according to $\hat{\theta} _{\mathrm{af}}(\mathrm{y}) = \Psi^{\star} _{\theta}\mathrm{y} + \psi^{\star} _{\theta}$, using the measurements of the entire trajectory $\mathrm{y}$. It returns $\Psi^{\star} _{\theta}$, $\psi^{\star} _{\theta}$, $\mu _{\mathrm{\Phi}}$, $\mathcal{M}$, $\mathcal{J} ^{\star} _{\theta}$, the theta estimates $\hat{\theta} _{\mathrm{af}}( \mathrm{y} )$ and its estimate covariance $\Sigma _{\hat{\theta} _{\mathrm{af}}}$ (See Theorem 2.2 of [*Nonlinear Bayesian Estimator for Parameter Learning:
A Fixed-Point Characterization*](https://arxiv.org/pdf/2606.10111) for further details).

2. `dualMMSEestimate`:

3.  `activeLearning`: Finds the **optimal input** sequence for the chosen horizon $T$ using the **active learning** algorithm and computes its corresponding **optimal Bayesian affine MMSE estimator** parameters. It returns optimizer parameters such as $\alpha$, $\beta$, a status flag indicating convergence to a local minimum or reaching the maximum iteration, the total number of iterations for the adaptive gradient descent algorithm, the gradient vector, and the final cost value. It also returns the optimal input sequence $\mathrm{u} ^{\star}$ as well as all the estimator parameters discussed in `estimatorOnly` mode (See Theorem 3.2 and Section 5 of [Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model](https://arxiv.org/abs/2504.05490) for further details).
