function thetaEstimator = thetaNonlinearEstimator(model, matrixPhibar, matrixSigmaPhi)
% This function computes the nonlinear parameter estimator defined in
% Equation 13 in Section 2 of the paper:
%   "Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization".
%
% It returns the estimator parameters and estimation error as derived in the theorem 2.2.
% 
% Paper: https://arxiv.org/abs/2606.10111
% Requirements: nonlinearBayesian4Wiener library; see README.md for details.
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: October 2025

thetaEstimator = struct;
matrixM = zeros(model.trajectoryT+1, model.trajectoryT+1);
parfor i=0:model.trajectoryT    
    rowsMatrixM = zeros(1, model.trajectoryT+1);
    for j=i:model.trajectoryT
        sigmaPhi = matrixSigmaPhi(i*model.numTheta+1:(i+1)*model.numTheta, ...
            j*model.numTheta+1:(j+1)*model.numTheta);
        rowsMatrixM(:, j+1) = trace(sigmaPhi*(model.sigmaTheta+(model.muTheta)*(model.muTheta')));
    end
    matrixM(i+1, :) = rowsMatrixM;
end
matrixM = matrixM+triu(matrixM,1).';

matrixPsi = (model.sigmaTheta*matrixPhibar)/(((matrixPhibar.')*...
    model.sigmaTheta*matrixPhibar)+matrixM+model.sigmaVbar);
vecPsi = model.muTheta - matrixPsi*(matrixPhibar.')*model.muTheta;
sigmaThetaEst = model.sigmaTheta - (matrixPsi*(matrixPhibar.')*model.sigmaTheta);
thetaErr = trace(sigmaThetaEst);

thetaEstimator.matrixPhibar = matrixPhibar;
thetaEstimator.matrixM = matrixM;
thetaEstimator.matrixPsi = matrixPsi;
thetaEstimator.vecPsi = vecPsi;
thetaEstimator.sigmaThetaEst = sigmaThetaEst;
thetaEstimator.thetaErr = thetaErr;
end