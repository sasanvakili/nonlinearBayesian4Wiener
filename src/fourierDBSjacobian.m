function matrixC = fourierDBSjacobian(numTheta, numState, allVecFreq, ...
    matrixAi, vecBbarUbar, sigmaWbar, muTheta)
% TBD
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: July 2025

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