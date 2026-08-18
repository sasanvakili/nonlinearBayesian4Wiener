# Dual Bayesian Affine Estimators for Parameter Learning

Code accompanying the paper:

> S. Vakili, D. Woonings, P. Paruchuri, and P. Mohajerin Esfahani, *Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization*.

This repository provides a MATLAB library for Bayesian estimation and active input design in Wiener models. It includes:

- An affine minimum mean-squared-error (MMSE) estimator for the *unknown* observation-model parameters;
- Dual estimators for joint learning of the *unknown* parameters and *latent* variables;
- An active-learning algorithm for optimal input design, based on:
  > S. Vakili, M. Mazo Jr., and P. Mohajerin Esfahani, *Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model*, arXiv preprint arXiv:2504.05490, 2025.

## Overview

The Wiener model considered here consists of known discrete-time, linear time-varying state dynamics and an observation model with *unknown* parameters:

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
(DS-P) estimator. For further details, see the [accompanying paper](https://arxiv.org/abs/2606.10111).

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
   Computes the optimal Bayesian affine MMSE estimator for the *unknown* parameter vector using the measurements over the complete trajectory, $\mathrm{y}$. It evaluates $\hat{\theta} _{\mathrm{af}}(\mathrm{y}) = \Psi^{\star} _{\theta}\mathrm{y} + \psi^{\star} _{\theta}$ and returns the optimal affine coefficients $\Psi^{\star} _{\theta}$, $\psi^{\star} _{\theta}$, the prior dynamic basis statistics mean $\mu _{\mathrm{\Phi}}$, the operator $\mathcal{M}$, the optimal cost $\mathcal{J} ^{\star} _{\theta}$, the parameter estimate $\hat{\theta} _{\mathrm{af}}$, and its estimation-error covariance $\Sigma _{\hat{\theta} _{\mathrm{af}}}$. See Theorem 2.2 of [accompanying paper](https://arxiv.org/abs/2606.10111).

2. **`dualMMSEestimate`**
   Computes the nonlinear Bayesian parameter estimate $`\hat{\theta}_{\mathrm{nl}}`$ through a fixed-point iteration that couples the affine parameter estimator with a dynamic-basis-statistics (DBS) estimator. Select either:

   - `DB-P` for the **dual basis-parameter** estimator, which updates DBS estimates directly through an affine basis estimator.
   - `DS-P` for the **dual state-parameter estimator**, which first computes affine state estimates and their covariance, then maps these state statistics to DBS estimates through the Gaussian DBS operator.
   
   The procedure alternates between parameter estimation and DBS estimation or state estimation until the specified convergence criterion is met. It returns the nonlinear parameter estimates $(\hat{\theta} _{\mathrm{nl}}, \Sigma _{\hat{\theta} _{\mathrm{nl}}})$, the DBS estimates $(\hat{\mathrm{\Phi}} _{\mathrm{nl}}, \Sigma _{\hat{\mathrm{\Phi}} _{\mathrm{nl}}})$, or state estimates $(\hat{\mathrm{x}} _{\mathrm{nl}}, \Sigma _{\hat{\mathrm{x}} _{\mathrm{nl}}})$, together with fixed-point iteration information. See Section 3 of [accompanying paper](https://arxiv.org/abs/2606.10111) for details.

3. **`activeLearning`**
   Designs an input sequence for the experiment by minimizing anticipated parameter-estimation uncertainty. The optimal input can be determined *a priori* and independently of measurements by solving $\mathrm{u}^{\star} \in \arg\min _{\mathrm{u} \in \mathbb{U}} \mathbb{E} \[ \lVert \theta - \hat{\theta} _{\mathrm{af}}(\mathrm{y}) \rVert ^{2} \]$, where $\mathbb{U}$ is the input space encoding physical constraints on feasible experiment inputs. The method supports two optimizers:

   - `adaptive`, which uses the library’s adaptive gradient-descent routine and does not require MATLAB Optimization Toolbox.
   - `fmincon`, which uses sequential quadratic programming (`sqp`) algorithm via MATLAB's `fmincon` solver and therefore requires MATLAB Optimization Toolbox.
   
   The routine returns the optimized input trajectory and information about the optimization process. See [*Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model*](https://arxiv.org/abs/2504.05490) for the underlying formulation.

## Usage

From the repository root, add the source directory to the MATLAB path:

```matlab
addpath(genpath('src'));
```

Invoke the library as follows:

```matlab
[estimator, optimizer, optimalUbar] = nonlinearBayesian4Wiener(model, settings, vecYbar);
```

Configure the input arguments `model`, `settings`, and `vecYbar` as described below. The library returns the outputs `estimator`, `optimizer`, and `optimalUbar`, which are described after the input arguments.

### Inputs
The input arguments `model`, `settings`, and `vecYbar` are discussed as follows:
#### `model`
This struct contains the following model parameters:
| Field | Description |
|---|---|
| `numState` | Number of state components, $`n_{\mathrm{x}}`$ |
| `numInput` | Number of input components, $`n_{\mathrm{u}}`$ |
| `numTheta` | Number of *unknown* parameters, including the zero-frequency coefficient: $`N+1`$ |
| `trajectoryT` | Final trajectory index, $`T`$ |
| `matrixAbar` | Lifted state-transition matrix, $`\mathrm{A}`$ |
| `matrixBbar` | Lifted input matrix, $`\mathrm{B}`$ |
| `vecUbar` | Lifted input vector, $`\mathrm{u}`$ |
| `allVecFreq` | Row-wise collection of all nonzero frequency vectors: $`[f_1, f_2, \ldots, f_N]`$. Frequency $`f_0`$ is excluded because $`\phi_0(\mathrm{x}) = 1`$ |
| `muTheta` | Prior mean of the *unknown* parameter vector, $`\mu_{\theta}`$ |
| `sigmaTheta` | Prior covariance of the *unknown* parameter vector, $`\Sigma_{\theta}`$ |
| `sigmaVbar` | Lifted measurement-noise covariance, $`\Sigma_{\mathrm{v}}`$ |
| `sigmaWbar` | Lifted process-noise covariance, $`\Sigma_{\mathrm{w}}`$ |

All fields in `model` are required.

#### `settings`
This struct contains the following fields:
| Field | Allowed values / type | Description |
|---|---|---|
| `mode` | `'affineMMSEestimate'`, `'dualMMSEestimate'`, or `'activeLearning'` | Library mode; see [Library modes](#library-modes) |
| `verbose` | `0`, `1`, or `2` | Verbosity level: `0` is silent; `1` and `2` enable progressively more diagnostic output. Default: `0` |
| `dual` | Struct or `[]` | Dual-estimation settings. Required only when `settings.mode = 'dualMMSEestimate'`; otherwise set `settings.dual = []` |
| `activeLearning` | Struct or `[]` | Active-learning settings. Required only when `settings.mode = 'activeLearning'`; otherwise set `settings.activeLearning = []` |

##### `settings.dual`

Required only when:

```matlab
settings.mode = 'dualMMSEestimate';
```

| Field | Allowed values / type | Description |
|---|---|---|
| `tol` | Positive scalar | Convergence tolerance $\epsilon$. The algorithm terminates when $`\left\lvert \mathcal{J}_{\theta}^{k} - \mathcal{J}_{\theta}^{k-1} \right\rvert < \epsilon`$. Example: `settings.dual.tol = 1e-6` |
| `maxIter` | Positive integer | Maximum number of fixed-point iterations, $K$. The algorithm terminates when the iteration count reaches `maxIter`. Example: `settings.dual.maxIter = 10000` |
| `type` | `'DB-P'` or `'DS-P'` | Dual-estimator variant: `'DB-P'` is the **dual basis-parameter** estimator; `'DS-P'` is the **dual state-parameter** estimator; see [Library modes](#library-modes). Default: `'DS-P'` |

##### `settings.activeLearning`

Required only when:

```matlab
settings.mode = 'activeLearning';
```

| Field | Required | Description |
|---|---|---|
| `solver` | Always | Optimization method: `'adaptive'` or `'fmincon'`. The `'fmincon'` applies sequential quadratic programming (`sqp`) algorithm which requires MATLAB Optimization Toolbox; see [Library modes](#library-modes). Default: `'adaptive'`  |
| `maxIter` | Always | Maximum number of optimization iterations. The algorithm terminates when the iteration count reaches `maxIter` |
| `applyToInitX` | Always | Logical flag indicating whether optimization of $`\mathrm{u}`$ includes the initial-state mean $`\mu_{\mathrm{x}_0}`$ |
| `existConstraint` | Always | Logical flag indicating whether the feasible input set $`\mathbb{U}`$ includes input constraints |
| `vecUmax` | When `settings.activeLearning.existConstraint = true`; otherwise set to `[]` (`settings.activeLearning.vecUmax = []`) | Vector of length $`n_{\mathrm{u}}`$ containing the upper bounds on each component of $`\mathrm{u}_t`$ |
| `vecUmin` | When `settings.activeLearning.existConstraint = true`; otherwise set to `[]` (`settings.activeLearning.vecUmin = []`) | Vector of length $`n_{\mathrm{u}}`$ containing the lower bounds on each component of $`\mathrm{u}_t`$ |
| `maxInitState` | When `settings.activeLearning.existConstraint = true` and `settings.activeLearning.applyToInitX = true`; otherwise set to `[]` (`settings.activeLearning.maxInitState = []`) | Vector of length $`n_{\mathrm{x}}`$ containing upper bounds on $`\mu_{\mathrm{x}_0}`$ |
| `minInitState` | When `settings.activeLearning.existConstraint = true` and `settings.activeLearning.applyToInitX = true`; otherwise set to `[]` (`settings.activeLearning.minInitState = []`) | Vector of length $`n_{\mathrm{x}}`$ containing lower bounds on $`\mu_{\mathrm{x}_0}`$ |
| `gradTol` | When `settings.activeLearning.solver = 'adaptive'`; otherwise do **NOT** define | Gradient-norm stopping tolerance. The adaptive optimizer terminates when the norm of the objective gradient, $`\lVert \nabla_{\mathrm{u}} \mathcal{J}^{\star}_{\theta}(\mathrm{u}^{k}) \rVert`$, is below `gradTol` |
| `costTol` | When `settings.activeLearning.solver = 'adaptive'`; otherwise do **NOT** define | Cost-decrease stopping tolerance. The adaptive optimizer terminates when the absolute change in objective value, $`\lvert \mathcal{J}^{\star}_{\theta}(\mathrm{u}^{k+1}) - \mathcal{J}^{\star}_{\theta}(\mathrm{u}^{k}) \rvert`$ is below `costTol` |
| `alpha` | When `settings.activeLearning.solver = 'adaptive'`; otherwise do **NOT** define | Initial adaptive stepsize parameter $`\alpha_0`$. Recommended value: `1e-10` |
| `beta` | When `settings.activeLearning.solver = 'adaptive'`; otherwise do **NOT** define | Initial adaptive stepsize parameter $`\beta_0`$. Recommended value: `1e100` |

#### `vecYbar`

| Argument | Required | Description |
|---|---|---|
| `vecYbar` | When `settings.mode` is `'affineMMSEestimate'` or `'dualMMSEestimate'`; otherwise set to `[]` (`vecYbar = []`) | Measurement vector $`\mathrm{y} = [ \mathrm{y} _{0}, \ldots, \mathrm{y} _{T} ] ^{\mathsf{T}}`$ |

### Outputs

The `nonlinearBayesian4Wiener` library returns `estimator`, `optimizer`, and `optimalUbar`. Their contents depend on `settings.mode` and, for active learning, on `settings.activeLearning.solver`.

| Mode | `estimator` | `optimizer` | `optimalUbar` |
|---|---|---|---|
| `affineMMSEestimate` | Affine MMSE estimator results | `[]` | `[]` |
| `dualMMSEestimate` with `DB-P` variant | Dual basis-parameter estimator results | `[]` | `[]` |
| `dualMMSEestimate` with `DS-P` variant | Dual state-parameter estimator results | `[]` | `[]` |
| `activeLearning` with `adaptive` solver | Affine MMSE estimator quantities at the final input iterate | Adaptive-optimization results | Optimized stacked input vector |
| `activeLearning` with `fmincon` solver | `[]` | `fmincon` results | Optimized stacked input vector |

#### `estimator` for `affineMMSEestimate`

| Field | Description |
|---|---|
| `matrixPsi` | Optimal affine coefficient $`\Psi_{\theta}^{\star}`$ |
| `vecPsi` | Optimal affine offset $`\psi_{\theta}^{\star}`$ |
| `matrixPhibar` | Prior dynamic-basis-statistics mean $`\mu_{\Phi}`$; see Definition 1 of the [accompanying paper](https://arxiv.org/abs/2606.10111) |
| `matrixM` | Matrix representation of the operator $`\mathcal{M}`$ from Theorem 2.2 of the [accompanying paper](https://arxiv.org/abs/2606.10111) |
| `thetaEstimate` | Affine parameter estimate $`\hat{\theta}_{\mathrm{af}}`$ |
| `sigmaThetaEst` | Estimation-error covariance $`\Sigma_{\hat{\theta}_{\mathrm{af}}}`$ |
| `thetaErr` | Optimal estimation cost $`\mathcal{J}_{\theta}^{\star}`$ |

#### `estimator` for `dualMMSEestimate`

The fields below are returned for both dual-estimator variants.

| Field | Description |
|---|---|
| `status` | Termination status: `converged` if the cost-change tolerance is met, or `max iterations` if the fixed-point iteration reaches `settings.dual.maxIter` |
| `totalIter` | Number of fixed-point iterations completed before termination |
| `thetaEstErr` | Cost history from iteration $1$ through termination: $`[\mathcal{J}_{\theta}^{1}, \ldots, \mathcal{J}_{\theta}^{k}]`$ |
| `thetaEstimate` | Nonlinear Bayesian parameter estimate at termination, $`\hat{\theta}_{\mathrm{nl}}`$ |
| `sigmaThetaEstimate` | Estimation-error covariance at termination, $`\Sigma_{\hat{\theta}_{\mathrm{nl}}}`$ |

The fixed-point iteration terminates when the absolute difference between
successive parameter-estimation costs is less than `settings.dual.tol`, or
when the iteration count reaches `settings.dual.maxIter`. See Algorithm 1 of
the [accompanying paper](https://arxiv.org/abs/2606.10111).

##### Additional fields for `DB-P`

| Field | Description |
|---|---|
| `basisEstimate` | DBS mean estimate at termination, $`\hat{\Phi}_{\mathrm{nl}}`$ |
| `sigmaBasisEstimate` | DBS covariance estimate at termination, $`\Sigma_{\hat{\Phi}_{\mathrm{nl}}}`$ |

##### Additional fields for `DS-P`

| Field | Description |
|---|---|
| `stateEstimate` | State estimate at termination, $`\hat{\mathrm{x}}_{\mathrm{nl}}`$ |
| `sigmaStateEstimate` | State-estimation covariance at termination, $`\Sigma_{\hat{\mathrm{x}}_{\mathrm{nl}}}`$ |

#### Outputs for `activeLearning`

`optimalUbar` contains the optimized stacked input vector
$`\mathrm{u}^{\star}`$ for both optimization methods.

##### `optimizer` with `settings.activeLearning.solver = 'adaptive'`

| Field | Description |
|---|---|
| `alpha` | Final adaptive stepsize parameter, $`\alpha_k`$ |
| `beta` | Final adaptive parameter, $`\beta_k`$ |
| `status` | Termination status: `converged` or `max iterations` |
| `totalIter` | Number of adaptive-gradient-descent iterations completed |
| `gradient` | Objective gradient at termination, $`\nabla_{\mathrm{u}}\mathcal{J}_{\theta}^{\star}(\mathrm{u}^{K})`$ |
| `optimalCost` | Final objective value, $`\mathcal{J}_{\theta}^{\star}(\mathrm{u}^{\star})`$ |

##### `estimator` with `settings.activeLearning.solver = 'adaptive'`

| Field | Description |
|---|---|
| `matrixPsi` | Optimal affine coefficient $`\Psi_{\theta}^{\star}`$ at the final input iterate |
| `vecPsi` | Optimal affine offset $`\psi_{\theta}^{\star}`$ at the final input iterate |
| `matrixPhibar` | Prior DBS mean $`\mu_{\Phi}`$ at the final input iterate |
| `matrixM` | Matrix representation of $`\mathcal{M}`$ at the final input iterate |
| `sigmaThetaEst` | Affine-estimation error covariance $`\Sigma_{\hat{\theta}_{\mathrm{af}}}`$ |
| `thetaErr` | Final optimal estimation cost $`\mathcal{J}_{\theta}^{\star}`$ |

##### `optimizer` with `settings.activeLearning.solver = 'fmincon'`

| Field | Description |
|---|---|
| `status` | `exitflag` returned by MATLAB `fmincon` |
| `output` | `output` struct returned by MATLAB `fmincon` |
| `optimalCost` | Final objective value, $`\mathcal{J}_{\theta}^{\star}(\mathrm{u}^{\star})`$ |

When `settings.activeLearning.solver = 'fmincon'`, `estimator` is returned as `[]`.

### Examples

The [`example/example.m`](example/example.m) script demonstrates how to configure the model and settings structs, provide measurements, and run each available mode of `nonlinearBayesian4Wiener`.

The `example/Experiment_setup_1/` and `example/Experiment_setup_2/` directories contain the corresponding `experimentData.mat` files used by the script and in the Numerical Experiments of the [accompanying paper](https://arxiv.org/abs/2606.10111).

## Citing

If you use the `nonlinearBayesian4Wiener` library for research, please cite our accompanying paper:

```bibtex
@article{vakili2026nonlinear,
  title={Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization},
  author={Vakili, Sasan and Woonings, Dani{\"e}l and Paruchuri, Pradyumna and Mohajerin Esfahani, Peyman},
  journal={arXiv preprint arXiv:2606.10111},
  year={2026}
}
```

