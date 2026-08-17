function estimator = dualBasisEstimator(model, vecYbar, settings)
% This function computes the theta and basis estimates
% "TBD"
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: October 2025

muThetaInit = model.muTheta;
sigmaThetaInit = model.sigmaTheta;
basisInit = basisPrior(model);
matrixSigmaPhi = basisInit.matrixSigmaPhi;
matrixPhibar = basisInit.matrixPhibar;
muThetaEst = model.muTheta;
sigmaThetaEst = model.sigmaTheta;

numIter = 0;
thetaEstErr = zeros(2,1);
listThetaEstErr = zeros(settings.dual.maxIter, 1);
while true
    numIter = numIter +1;
    % Theta Estimate:
    model.muTheta = muThetaInit;
    model.sigmaTheta = sigmaThetaInit;
    thetaEstimator = thetaNonlinearEstimator(model, matrixPhibar, matrixSigmaPhi);
    % Basis Estimate:
    model.muTheta = muThetaEst;
    model.sigmaTheta = sigmaThetaEst;
    basisEstimator = basisAffineEstimator(model);
    % Update estimates:
    muThetaEst = thetaEstimator.matrixPsi*vecYbar+thetaEstimator.vecPsi;
    sigmaThetaEst = thetaEstimator.sigmaThetaEst;
    vecPhibar = basisEstimator.matrixZ*vecYbar+basisEstimator.vecZeta;
    matrixPhibar = reshape(vecPhibar, model.numTheta, model.trajectoryT+1);
    matrixSigmaPhi = basisEstimator.sigmaPhiEst;
    % check convergence:
    thetaEstErr(1) = thetaEstErr(2);
    thetaEstErr(2) = thetaEstimator.thetaErr;
    listThetaEstErr(numIter) = thetaEstimator.thetaErr;
    if (abs(thetaEstErr(2) - thetaEstErr(1)) < settings.dual.tol)
        status = 'converged';
        if (settings.verbose >= 2)
            msg = ['Converged', newline, ...
                'Total iterations: ', num2str(numIter), newline, ...
                'Final Err: ', num2str(thetaEstErr(2))];
            disp(msg);
        end
        break
    end
    if (numIter >= settings.dual.maxIter)
        status = 'max iterations';
        if (settings.verbose >= 2)
            msg = ['Stopped at maximum iteration.', newline, ...
                'Iterations: ', num2str(numIter), newline, ...
                'Final Err: ', num2str(thetaEstErr(2)), ...
                newline, 'Final Err diff: ', ...
                num2str(abs(thetaEstErr(2) - thetaEstErr(1)))];
            disp(msg);
        end
        break
    end
end
estimator.status = status;
estimator.totalIter = numIter;
estimator.thetaEstErr = listThetaEstErr(1:numIter);
estimator.thetaEstimate = muThetaEst;
estimator.sigmaThetaEstimate = sigmaThetaEst;
estimator.basisEstimate = matrixPhibar;
estimator.sigmaBasisEstimate = matrixSigmaPhi;
end
