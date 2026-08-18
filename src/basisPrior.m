function basisInit = basisPrior(model)
% This function computes the DBS prior according to Definition 1 of the paper:
%   "Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization".
%
% The function returns the DBS prior collection.
%
% Paper: https://arxiv.org/abs/2606.10111
% Requirements: nonlinearBayesian4Wiener library; see README.md for details.
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: October 2025

basisInit = struct;
vecBbarUbar = model.matrixBbar*model.vecUbar;
matrixPhibar = zeros(model.numTheta, model.trajectoryT+1);
matrixSigmaPhi = cell(model.numTheta*(model.trajectoryT+1), 1);
parfor i=0:model.trajectoryT
    matrixAi = model.matrixAbar(model.numState*i+1:model.numState*i+model.numState, :);
    rowsSigmaPhi = zeros(model.numTheta, model.numTheta*(model.trajectoryT+1));
    for j=i:model.trajectoryT
        matrixAj = model.matrixAbar(model.numState*j+1:model.numState*j+model.numState, :);
        if (j == i)
            [sigmaPhi, muPhi] = fourierDBS(model.numTheta, model.allVecFreq, matrixAi, ...
                vecBbarUbar, model.sigmaWbar, matrixAj);
            matrixPhibar(:, i+1) = muPhi;
            tempSigmaPhi = triu(sigmaPhi);
        else
            sigmaPhi = fourierDBS(model.numTheta, model.allVecFreq, matrixAi, ...
                vecBbarUbar, model.sigmaWbar, matrixAj);
            tempSigmaPhi = sigmaPhi;
        end
        rowsSigmaPhi(:, j*model.numTheta+1:(j+1)*model.numTheta) = tempSigmaPhi;
    end
    matrixSigmaPhi(i+1) = {rowsSigmaPhi};  
end
matrixSigmaPhi = cell2mat(matrixSigmaPhi);
matrixSigmaPhi = matrixSigmaPhi+triu(matrixSigmaPhi,1).';

basisInit.matrixSigmaPhi = matrixSigmaPhi;
basisInit.matrixPhibar = matrixPhibar;

end