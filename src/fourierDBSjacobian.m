function matrixC = fourierDBSjacobian(numTheta, numState, allVecFreq, ...
    matrixAi, vecBbarUbar, sigmaWbar, muTheta)
% This function computes the expectation of the Jacobian matrix of the
% Fourier basis functions. The Jacobian formulation is defined in Lemma 3.5 of the paper:
%   "Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization".
% 
% The explicit expression for the expectation of the Fourier basis
% functions is given in Lemma 4.1 of the paper:
%   "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model".
%
% This computation is used by the affine MMSE state estimator.
%
% Papers: 
%   https://arxiv.org/abs/2606.10111
%   https://arxiv.org/abs/2504.05490
% Requirements: nonlinearBayesian4Wiener library; see README.md for details.
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: October 2025

jacobianPhi = zeros(numTheta, numState);
jacobianPhi(2:end,:) = (allVecFreq.').*(1i*exp(1i*(allVecFreq')*matrixAi* ...
    vecBbarUbar)-1i*exp(-1i*(allVecFreq')*matrixAi* ...
    vecBbarUbar)).*exp(-(1/2)*sum(((allVecFreq')*matrixAi* ...
    sigmaWbar).*((allVecFreq')*matrixAi), 2));

matrixC = (muTheta.')*jacobianPhi;

if any(abs(imag(matrixC)) >= 1e-6)
    error(['matrixC has non-negligible imaginary parts ' ...
        '(max abs(imag): %.2e)'], max(abs(imag(muPhiState))));
end
matrixC = real(matrixC);

end