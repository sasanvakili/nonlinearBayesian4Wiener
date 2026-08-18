function basisEstimator = basisAffineEstimator(model)
% This function computes the DBS estimation via affine MMSE basis estimator
% defined in Lemma 3.1 of the paper:
%   "Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization".
%
% The function returns the estimator parameters and 
% optimal cost as derived in the lemma.
%
% Paper: https://arxiv.org/abs/2606.10111
% Requirements: nonlinearBayesian4Wiener library; see README.md for details.
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: October 2025

basisEstimator = struct;
vecBbarUbar = model.matrixBbar*model.vecUbar;
matrixMuTheta = cell(1, model.trajectoryT+1);
matrixPhibar = zeros(model.numTheta, model.trajectoryT+1);
matrixSigmaPhi = cell(model.numTheta*(model.trajectoryT+1), 1);
matrixM = zeros(model.trajectoryT+1, model.trajectoryT+1);
parfor i=0:model.trajectoryT
    matrixAi = model.matrixAbar(model.numState*i+1:model.numState*i+model.numState, :);
    rowsMatrixM = zeros(1, model.trajectoryT+1);
    rowsSigmaPhi = zeros(model.numTheta, model.numTheta*(model.trajectoryT+1));
    for j=i:model.trajectoryT
        matrixAj = model.matrixAbar(model.numState*j+1:model.numState*j+model.numState, :);
        if (j == i)
            [sigmaPhi, muPhi] = fourierDBS(model.numTheta, model.allVecFreq, matrixAi, ...
                vecBbarUbar, model.sigmaWbar, matrixAj);
            matrixPhibar(:, i+1) = muPhi;
            matrixMuTheta(i+1) = {model.muTheta};
            tempSigmaPhi = triu(sigmaPhi);
        else
            sigmaPhi = fourierDBS(model.numTheta, model.allVecFreq, matrixAi, ...
                vecBbarUbar, model.sigmaWbar, matrixAj);
            tempSigmaPhi = sigmaPhi;
        end
        rowsMatrixM(:, j+1) = trace(sigmaPhi*(model.sigmaTheta+(model.muTheta)*(model.muTheta')));
        rowsSigmaPhi(:, j*model.numTheta+1:(j+1)*model.numTheta) = tempSigmaPhi;
    end
    matrixM(i+1, :) = rowsMatrixM;
    matrixSigmaPhi(i+1) = {rowsSigmaPhi};  
end
matrixSigmaPhi = cell2mat(matrixSigmaPhi);
matrixM = matrixM+triu(matrixM,1).';
matrixSigmaPhi = matrixSigmaPhi+triu(matrixSigmaPhi,1).';
matrixMuTheta = blkdiag(matrixMuTheta{:});

matrixZ = matrixSigmaPhi*(matrixMuTheta/(((matrixPhibar.')*...
    model.sigmaTheta*matrixPhibar)+matrixM+model.sigmaVbar));

vecPhibar = reshape(matrixPhibar, model.numTheta*(model.trajectoryT+1), 1);
vecZeta = vecPhibar - matrixZ*(matrixMuTheta.')*vecPhibar;

sigmaPhiEst = matrixSigmaPhi-(matrixZ*(matrixMuTheta')*matrixSigmaPhi);
basisEstimationErr = trace(sigmaPhiEst);

basisEstimator.matrixPhibar = matrixPhibar;
basisEstimator.matrixM = matrixM;
basisEstimator.matrixZ = matrixZ;
basisEstimator.vecZeta = vecZeta;
basisEstimator.sigmaPhiEst = sigmaPhiEst;
basisEstimator.basisEstimationErr = basisEstimationErr;
end