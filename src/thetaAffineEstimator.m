function thetaEstimator = thetaAffineEstimator(model, vecBbarUbar)
% This function computes the Optimal Bayesian MMSE affine estimator defined in Theorem
% 2.2 of the paper:
%   "Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization".
%
% It returns the estimator parameters and optimal cost as derived in the theorem.
% 
% Paper: https://arxiv.org/abs/2606.10111
% Requirements: nonlinearBayesian4Wiener library; see README.md for details.
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: October 2025

thetaEstimator = struct;

matrixPhibar = zeros(model.numTheta, model.trajectoryT+1);
matrixM = zeros(model.trajectoryT+1, model.trajectoryT+1);
parfor i=0:model.trajectoryT
    matrixAi = model.matrixAbar(model.numState*i+1:model.numState*i+model.numState, :);
    rowsMatrixM = zeros(1, model.trajectoryT+1);
    for j=i:model.trajectoryT
        matrixAj = model.matrixAbar(model.numState*j+1:model.numState*j+model.numState, :);
        if (j == i)
            [sigmaPhi, muPhi] = fourierDBS(model.numTheta, model.allVecFreq, matrixAi, ...
                vecBbarUbar, model.sigmaWbar, matrixAj);
            matrixPhibar(:, i+1) = muPhi;
        else
            sigmaPhi = fourierDBS(model.numTheta, model.allVecFreq, matrixAi, ...
                vecBbarUbar, model.sigmaWbar, matrixAj);
        end
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