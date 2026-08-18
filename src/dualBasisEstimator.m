function estimator = dualBasisEstimator(model, vecYbar, settings)
% This function implements the dual basis-parameter (DB-P) estimator
% defined in subsection 3.1 of the paper:
%   "Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization".
%
% The function returns the dynamic-basis-statistics (DBS) mean and
% covariance estimates, the parameter estimate and its estimation-error
% covariance, and fixed-point iteration information: termination status,
% number of iterations, and cost history from the first iteration through
% termination.
% 
% Paper: https://arxiv.org/abs/2606.10111
% Requirements: nonlinearBayesian4Wiener library; see README.md for details.
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
