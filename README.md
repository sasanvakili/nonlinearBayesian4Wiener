# Dual Bayesian Affine Estimators for Parameter Learning

Code accompanying the paper:

> S. Vakili, D. Woonings, P. Paruchuri, and P. Mohajerin Esfahani, *Dual Bayesian Affine Estimators for Parameter Learning: A Fixed-Point Characterization*.

This repository provides a MATLAB library for Bayesian estimation and active input design in Wiener models. It includes:

- An affine minimum mean-squared-error (MMSE) estimator for the *unknown* observation-model parameters;
- Dual estimators for joint learning of the *unknown* parameters and *latent* variables;
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

This library uses these structures to compute the affine MMSE estimate $`\hat{\theta}_{\mathrm{af}}`$ and the nonlinear Bayesian estimate $`\hat{\theta}_{\mathrm{nl}}`$, obtained using either the **dual basis-parameter** (DB-P) or **dual state-parameter**
(DS-P) estimator. For further details, see the paper [*Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization*](https://arxiv.org/abs/2606.10111).

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

## Library modes

The `nonlinearBayesian4Wiener` library provides three modes of operation:

1. **`affineMMSEestimate`**
   Computes the optimal Bayesian affine MMSE estimator for the *unknown* parameter vector using the measurements over the complete trajectory, $\mathrm{y}$. It evaluates $\hat{\theta} _{\mathrm{af}}(\mathrm{y}) = \Psi^{\star} _{\theta}\mathrm{y} + \psi^{\star} _{\theta}$ and returns the optimal affine coefficients $\Psi^{\star} _{\theta}$, $\psi^{\star} _{\theta}$, the prior dynamic basis statistics mean $\mu _{\mathrm{\Phi}}$, the operator $\mathcal{M}$, the optimal cost $\mathcal{J} ^{\star} _{\theta}$, the parameter estimate $\hat{\theta} _{\mathrm{af}}$, and its estimation-error covariance $\Sigma _{\hat{\theta} _{\mathrm{af}}}$. See Theorem 2.2 of [*Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization*](https://arxiv.org/abs/2606.10111).

2. **`dualMMSEestimate`**
   Computes the nonlinear Bayesian parameter estimate $`\hat{\theta}_{\mathrm{nl}}`$ through a fixed-point iteration that couples the affine parameter estimator with a dynamic-basis-statistics (DBS) estimator. Select either:

   - `DB-P` for the **dual basis-parameter** estimator, which updates DBS estimates directly through an affine basis estimator.
   - `DS-P` for the **dual state-parameter estimator**, which first computes affine state estimates and their covariance, then maps these state statistics to DBS estimates through the Gaussian DBS operator.
   
   The procedure alternates between parameter estimation and DBS estimation or state estimation until the specified convergence criterion is met. It returns the nonlinear parameter estimates $(\hat{\theta} _{\mathrm{nl}}, \Sigma _{\hat{\theta} _{\mathrm{nl}}})$, the DBS estimates $(\hat{\mathrm{\Phi}} _{\mathrm{nl}}, \Sigma _{\hat{\mathrm{\Phi}} _{\mathrm{nl}}})$, or state estimates $(\hat{\mathrm{x}} _{\mathrm{nl}}, \Sigma _{\hat{\mathrm{x}} _{\mathrm{nl}}})$, together with fixed-point iteration information. See Section 3 of [*Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization*](https://arxiv.org/abs/2606.10111) for details.

3. **`activeLearning`**
   Designs an input sequence for the experiment by minimizing anticipated parameter-estimation uncertainty. The optimal input can be determined *a priori* and independently of measurements by solving $\mathrm{u}^{\star} \in \arg\min _{\mathrm{u} \in \mathbb{U}} \mathbb{E} \[ \lVert \theta - \hat{\theta} _{\mathrm{af}}(\mathrm{y}) \rVert ^{2} \]$, where $\mathbb{U}$ is the input space encoding physical constraints on feasible experiment inputs. The method supports two optimizers:

   - `adaptive`, which uses the library’s adaptive gradient-descent routine and does not require MATLAB Optimization Toolbox.
   - `fmincon`, which uses MATLAB's `fmincon` solver and therefore requires MATLAB Optimization Toolbox.
   
   The routine returns the optimized input trajectory and information about the optimization process. See [*Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model*](https://arxiv.org/abs/2504.05490) for the underlying formulation.


## Usage

The `nonlinearBayesian4Wiener` library can be invoked in MATLAB using the following command:

`[estimator, optimizer, optimalUbar] = nonlinearBayesian4Wiener(model, settings, vecYbar);`

Ensure that the the library's `src` folder is added to MATLAB's path (e.g., `addpath('./src')`), and the inputs are configured according to the descriptions below.

### Inputs

The `Bayesian4Wiener` library takes the following three major inputs:

*   **`model`**: This struct contains the following model parameters:
    *   `numState`: The number of states $n _{\mathrm{x}}$.
    *   `numInput`: The number of inputs $n _{\mathrm{u}}$.
    *   `numTheta`: The number of unknown parameters, including the zero frequency, $N+1$.  
    *   `trajectoryT`: The time index of the trajectory ending time $T$.
    *   `matrixAbar`: The state transition matrix for the entire trajectory, $\mathrm{A}$.
    *   `matrixBbar`: The input matrix for the entire trajectory, $\mathrm{B}$.
    *   `vecUbar`: The input vector for the entire trajectory, $\mathrm{u}$.
    *   `allVecFreq`: The row vector collection of all frequencies $f_{n}$, **EXCLUDING FREQUENCY 0**, i.e., $\[ f _{1}, f _{2}, \ldots, f _{N} \]$.
    *   `muTheta`: The mean of the unknown $\theta$ prior distribution, $\mu _{\theta}$.
    *   `sigmaTheta`: The covariance of the unknown $\theta$ prior distribution, $\Sigma _{\theta}$.
    *   `sigmaVbar`: The measurement noise covariance, $\Sigma _{\mathrm{v}}$.
    *   `sigmaWbar`: The process noise covariance, $\Sigma _{\mathrm{w}}$.

    All the above fields are mandatory and should be properly provided in order for the library to run.

*   **`settings`**: This struct contains the following fields:
    *   `mode`: One of the modes discussed in [Library modes](#library-modes), i.e., `affineMMSEestimate`, `dualMMSEestimate`, or `activeLearning`.  This field should be set to one of the available options for the library to run.
    *   `verbose`: One of the verbosity levels, where $0$ means silent execution and $1$ or $2$ provide minor debugging messages.  Allowed values are 0, 1, or 2. The default is set $0$ for silent execution if this field is not set to one of the available options.
    *   `activeLearning`: This struct contains the following fields:
        *   `gradTol`: Stopping criterion for gradient norm threshold, i.e., $\big\lVert \nabla _{\overline{\mathrm{u}}} \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k}) \big\rVert <$ `gradTol`.
        *   `costTol`: Stopping criterion for cost decrease threshold, i.e., $\big\lvert \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k+1}) - \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k}) \big\rvert <$ `costTol`.
        *   `maxIter`: Stopping criterion for the maximum number of iterations allowed. The algorithm will terminate when $k$ reaches `maxIter`.
        *   `alpha`: The initial value $\alpha _{0}$ required for the adaptive stepsize algorithm (recommended value: $\alpha _{0} = 10^{-10}$).
        *   `beta`: The initial value $\beta _{0}$ required for the adaptive stepsize algorithm (recommended value: $\beta _{0} = 10^{100}$).
        *   `applyToInitX`: A boolean variable determining whether or not the optimization of $\overline{\mathrm{u}}$ includes optimizing the initial state $\mu _{\mathrm{x} _{0}}$.
        *   `existConstraint`: A boolean variable determining whether or not there exists a constraint on the input $\mathbb{U}$.
        *   `vecUmax`: A vector with size $n _{\mathrm{u}}$ that contains the maximum values each dimension of $\mathrm{u} _{t}$ can take. This is required **ONLY** if `existConstraint` is set to true; otherwise, it can be left empty (i.e., `settings.activeLearning.vecUmax = []`).
        *   `vecUmin`: A vector with size $n _{\mathrm{u}}$ that contains the minimum values each dimension of $\mathrm{u} _{t}$ can take. This is required **ONLY** if `existConstraint` is set to true; otherwise, it can be left empty (i.e., `settings.activeLearning.vecUmin = []`).
        *   `maxInitState`: A vector with size $n _{\mathrm{x}}$ that contains the maximum values the initial state $\mu _{\mathrm{x} _{0}}$ can take. This is required **ONLY** if both `existConstraint` and `applyToInitX` are set to true; otherwise, it can be left empty (i.e., `settings.activeLearning.maxInitState = []`).
        *   `minInitState`: A vector with size $n _{\mathrm{x}}$ that contains the minimum values the initial state $\mu _{\mathrm{x} _{0}}$ can take. This is required **ONLY** if both `existConstraint` and `applyToInitX` are set to true; otherwise, it can be left empty (i.e., `settings.activeLearning.minInitState = []`).

        All the above fields are required **ONLY** if the mode is set to `activeLearning` (i.e., `settings.mode = 'activeLearning'`); otherwise, `activeLearning` can be left empty (i.e., `settings.activeLearning = []`).

*   **`vecYbar`**: The vector of measurements, i.e., $\overline{\mathrm{y}} = \[ \mathrm{y} _{0}, \ldots, \mathrm{y} _{T} \] ^{\mathsf{T}}$. This is required **ONLY** if the mode is set to `affineMMSEestimate` or `dualMMSEestimate`; otherwise, it can be left empty (i.e., `vecYbar = []`).
