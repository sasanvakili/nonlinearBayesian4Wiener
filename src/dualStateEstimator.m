function estimator = dualStateEstimator(model, vecYbar, settings)
% This function implements the dual state-parameter (DS-P) estimator
% defined in subsection 3.2 of the paper:
%   "Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization".
%
% The function returns the state estimate and its estimation-error
% covariance, the parameter estimate and its estimation-error
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
sigmaWbarInit = model.sigmaWbar;
muThetaEst = model.muTheta;
sigmaThetaEst = model.sigmaTheta;
vecBbarUbarEst = model.matrixBbar*model.vecUbar;
sigmaWbarEst = model.sigmaWbar;
numIter = 0;
thetaEstErr = zeros(2,1);
listThetaEstErr = zeros(settings.dual.maxIter, 1);
while true
    numIter = numIter +1;
    % Theta Estimate:
    model.muTheta = muThetaInit;
    model.sigmaTheta = sigmaThetaInit;
    vecBbarUbar = vecBbarUbarEst;
    model.sigmaWbar = sigmaWbarEst;
    thetaEstimator = thetaAffineEstimator(model, vecBbarUbar);
    % State Estimate:
    model.sigmaWbar = sigmaWbarInit;
    model.muTheta = muThetaEst;
    model.sigmaTheta = sigmaThetaEst;
    stateEstimator = stateAffineEstimator(model);
    % Update estimates:
    muThetaEst = thetaEstimator.matrixPsi*vecYbar+thetaEstimator.vecPsi;
    sigmaThetaEst = thetaEstimator.sigmaThetaEst;
    vecBbarUbarEst = stateEstimator.matrixZ*vecYbar+stateEstimator.vecZeta;
    sigmaWbarEst = stateEstimator.sigmaWbarEst;
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
estimator.stateEstimate = model.matrixAbar*vecBbarUbarEst;
estimator.sigmaStateEstimate = model.matrixAbar*sigmaWbarEst*(model.matrixAbar.');
end
