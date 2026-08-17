function stateEstimator = stateAffineEstimator(model)
% TBD
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: July 2025

stateEstimator = struct;
vecBbarUbar = model.matrixBbar*model.vecUbar;
matrixC = cell(1,model.trajectoryT+1);
matrixPhibar = zeros(model.numTheta, model.trajectoryT+1);
matrixM = zeros(model.trajectoryT+1, model.trajectoryT+1);
parfor i=0:model.trajectoryT
% for i=0:model.trajectoryT
    matrixAi = model.matrixAbar(model.numState*i+1:model.numState*i+model.numState, :);
    rowsMatrixM = zeros(1, model.trajectoryT+1);
    for j=i:model.trajectoryT
        matrixAj = model.matrixAbar(model.numState*j+1:model.numState*j+model.numState, :);
        if (j == i)
            [sigmaPhi, muPhi] = fourierDBS(model.numTheta, model.allVecFreq, matrixAi, ...
                vecBbarUbar, model.sigmaWbar, matrixAj);
            matrixPhibar(:, i+1) = muPhi;
            matrixC(i+1) = {fourierDBSjacobian(model.numTheta, model.numState, ...
                model.allVecFreq, matrixAi, vecBbarUbar, model.sigmaWbar, ...
                model.muTheta)};
        else
            sigmaPhi = fourierDBS(model.numTheta, model.allVecFreq, matrixAi, ...
                vecBbarUbar, model.sigmaWbar, matrixAj);
        end
        rowsMatrixM(:, j+1) = trace(sigmaPhi*(model.sigmaTheta+(model.muTheta)*(model.muTheta')));
    end
    matrixM(i+1, :) = rowsMatrixM;
end
matrixM = matrixM+triu(matrixM,1).';
matrixCbar = blkdiag(matrixC{:});

matrixZ = model.sigmaWbar*(model.matrixAbar')*((matrixCbar.')/(((matrixPhibar.')*...
    model.sigmaTheta*matrixPhibar)+matrixM+model.sigmaVbar));
vecZeta = vecBbarUbar - matrixZ*(matrixPhibar.')*model.muTheta;
sigmaWbarEst = model.sigmaWbar-(matrixZ*matrixCbar*(model.matrixAbar)*model.sigmaWbar);
vecWbarErr = trace(sigmaWbarEst);

stateEstimator.matrixPhibar = matrixPhibar;
stateEstimator.matrixM = matrixM;
stateEstimator.matrixCbar = matrixCbar;
stateEstimator.matrixZ = matrixZ;
stateEstimator.vecZeta = vecZeta;
stateEstimator.sigmaWbarEst = sigmaWbarEst;
stateEstimator.noiseErr = vecWbarErr;
end