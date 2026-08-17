# Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization

This library implements optimal Bayesian affine and nonlinear MMSE parameter estimators, as well as active learning algorithm for a discrete-time linear time-varying dynamical system with an _unknown_ observation model. The system is described as follows:

$$
\mathrm{x} _{t+1} = \mathrm{A} _{t} \mathrm{x} _{t} + \mathrm{B} _{t} \mathrm{u} _{t} + \mathrm{w} _{t+1},
$$
$$
\mathrm{y} _{t} = \sum\limits _{n=0}^{N} \theta _{n} \phi _{n}( \mathrm{x} _{t}) + \mathrm{v} _{t} = \langle {\phi(\mathrm{x} _{t}), \theta} \rangle,
$$

where
*   $t = \\{ 0, \\ldots, T \\}$
*   $\mathrm{x} _{t} \in \mathbb{R} ^{n _{\mathrm{x}}}$ is the state vector.
*   $\mathrm{A} _{t} \in \mathbb{R} ^{n _{\mathrm{x}} \times n _{\mathrm{x}} }$ is the state transition matrix.
*   $\mathrm{B} _{t} \in \mathbb{R} ^{n _{\mathrm{x}} \times n _{\mathrm{u}}}$ is the input matrix.
*   $\mathrm{u} _{t} \in \mathbb{R} ^{n _{\mathrm{u}}}$ is the input vector.
*   $\mathrm{w} _{t+1} \in \mathbb{R} ^{n _{\mathrm{x}}}$ is the process noise.
*   $\mathrm{y} _{t} \in \mathbb{R}$ is the scalar output measurement.
*    $\mathrm{v} _{t} \in \mathbb{R}$  is the measurement noise.

The _known_ basis functions $\phi _{n}( \mathrm{x} _{t})$ are defined as the following Fourier bases:

$$ 
\begin{cases}
\phi_ {0} ( \mathrm{x} _{t} ) = 1 & n = 0 \\
\phi _{n}( \mathrm{x} _{t} ) = \sum\limits _{\ell \in \\{ -1, 1 \\}} \mathrm{exp} ( j \langle { \ell f _{n} , \mathrm{x} _{t} } \rangle ) & n \geq 1,
\end{cases}
$$

with _known_ frequencies $f _{n} \in \mathbb{R} ^{n _{\mathrm{x}}}$, where $n$ denotes the frequency index, and $\phi(\mathrm{x} _{t}) = \[ \phi _{0}(\mathrm{x} _{t}), \ldots, \phi _{N}(\mathrm{x} _{t}) \] ^{\mathsf{T}}$ is the vector of basis functions evaluated at $\mathrm{x} _{t}$.

## Optimal Bayesian Affine Estimator

The **optimal Bayesian affine estimator** has the form

$$
\hat{\theta} _{\mathrm{B}}( \overline{\mathrm{y}} ) = \Psi^{\star} \overline{\mathrm{y}} + \psi^{\star},
$$

This estimator computes the model parameters $\theta = \[ \theta _{0}, \ldots, \theta _{N} \] ^{\mathsf{T}}$ from the measurement data $\mathrm{y} _{t}$ at all time steps, represented as $\overline{\mathrm{y}} = \[ \mathrm{y} _{0}, \ldots, \mathrm{y} _{T} \] ^{\mathsf{T}}$. To compute $\Psi^{\star}$ and $\psi^{\star}$, the models are represented in _lifted matrix_ form. The output function is

$$
\overline{\mathrm{y}} = \Phi ^{\mathsf{T}} \theta + \overline{\mathrm{v}},
$$

where
*   $\overline{\mathrm{y}} = \[ \mathrm{y} _{0}, \ldots, \mathrm{y} _{T} \] ^{\mathsf{T}}$ is the vector of measurements.
*   $\Phi = \[ \phi(\mathrm{x} _{0}), \ldots, \phi(\mathrm{x} _{T}) \]$ is the basis aggregation matrix.
*   $\theta = \[ \theta _{0}, \ldots, \theta _{N} \] ^{\mathsf{T}}$ is the vector of all _unknown_ parameters.
*   $\overline{\mathrm{v}} = \[ \mathrm{v} _{0}, \ldots, \\mathrm{v} _{T} \] ^{\mathsf{T}}$ is the measurement noise vector, where $\overline{\mathrm{v}} \sim \mathcal{N} ( 0, \Sigma _{\mathrm{v}})$ and $\Sigma _{\mathrm{v}} = \mathrm{diag}( \Sigma _{\mathrm{v} _{0}}^{2}, \ldots, \Sigma _{\mathrm{v} _{T}}^{2})$.

The prior information about $\theta$ is characterized by a probability distribution defined by its mean and covariance, $\mathbb{P} ( \mu _{\theta}, \Sigma _{\theta})$.

The process model is represented as

$$
\overline{\mathrm{x}} = \overline{\mathrm{A}} (\overline{\mathrm{B}} \overline{\mathrm{u}} + \overline{\mathrm{w}}),
$$

where
*   $\overline{\mathrm{x}} = \[ \mathrm{x} _{0} ^{\mathsf{T}}, \ldots, \mathrm{x} _{T} ^{\mathsf{T}} \] ^{\mathsf{T}}$ is the system states vector.
*   $\overline{\mathrm{u}} = \[ \mu _{\mathrm{x} _{0}} ^{\mathsf{T}}, \mathrm{u} _{0} ^{\mathsf{T}}, \ldots, \mathrm{u} _{T-1} ^{\mathsf{T}} \] ^{\mathsf{T}}$ is the input vector, with the mean of the initial state $\mu _{\mathrm{x} _{0}}$ as its first element.
*   $\overline{\mathrm{w}} = \[ \mathrm{w} _{0} ^{\mathsf{T}}, \mathrm{w} _{1} ^{\mathsf{T}}, \ldots, \mathrm{w} _{T} ^{\mathsf{T}} \] ^{\mathsf{T}}$ is the noise vector, where $\mathrm{w} _{0} \sim \mathcal{N} (0, \Sigma _{\mathrm{x} _{0}} )$ and $\overline{\mathrm{w}} \sim \mathcal{N} ( 0 , \Sigma _{\overline{\mathrm{w}}} )$. The covariance matrix is diagonal only if $\mathrm{w} _{t}$ are independent, i.e., $\Sigma _{\overline{\mathrm{w}}} = \mathrm{diag} ( \Sigma _{\mathrm{x} _{0}}, \Sigma _{\mathrm{w} _{1}}, \ldots, \Sigma _{\mathrm{w} _{T}} )$.

The matrices $\overline{\mathrm{A}}$ and $\overline{\mathrm{B}}$ have the following lower triangular and block-diagonal structures, respectively:

$$
\overline{\mathrm{A}} = \begin{bmatrix} \mathbb{I} & 0 & 0 & \ldots & 0 \\ 
\mathrm{A} _0 & \mathbb{I} & 0 & \ldots & \vdots \\ 
\mathrm{A} _1 \mathrm{A} _0 & \mathrm{A} _1 & \mathbb{I} & \ddots & \vdots \\ 
\vdots & \vdots &  \vdots & \ddots & 0 \\ 
\prod\limits _{i=0} ^{T-1} \mathrm{A} _{i} & \prod\limits _{i=1} ^{T-1} \mathrm{A} _{i} &  \ldots & \mathrm{A} _{T-1} & \mathbb{I}
\end{bmatrix}, \qquad 
\overline{\mathrm{B}} = \mathrm{diag} ( \mathbb{I}, \mathrm{B} _{0}, \ldots, \mathrm{B} _{T-1}).
$$

This library uses these structures to compute the matrix $\Psi^{\star}$, the vector $\psi^{\star}$, and estimates $\hat{\theta}_{\mathrm{B}}( \overline{\mathrm{y}} )$ if measurements are also provided.
