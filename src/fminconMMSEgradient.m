function [optimizationCost, gradientUbar] = fminconMMSEgradient(vecUbar, model, settings)
% This function return the cost and fourier DBS gradient required for
% fmincon algorithm.
% Paper: https://arxiv.org/abs/2504.05490
% Requirements: Bayesian4Wiener library (see README for details)
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: May 2025

%VecUbar Translation:
model.vecUbar = vecUbar;

%Cost:
vecBbarUbar = model.matrixBbar*model.vecUbar;
estimator = thetaAffineEstimator(model, vecBbarUbar);
optimizationCost = estimator.thetaErr;

%Gradient:
if (nargout > 1)
    gradientUbar = thetaMMSEgradient(model, estimator);
    if ~(settings.activeLearning.applyToInitX)
        %InitState is not optimized!
        gradientUbar(1:model.numState) = zeros(model.numState, 1);
    end
end
end