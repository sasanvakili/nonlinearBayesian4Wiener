# Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization

This library implements a nonlinear Bayesian parameter estimator and active learning algorithm for a discrete-time linear time-varying dynamical system with an _unknown_ observation model. The system is described as follows:

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
