function [optimizationCost, gradientUbar] = fminconMMSEgradient(vecUbar, model, settings)
% This function prepares the Bayesian affine-MMSE estimation cost and its
% gradient with respect to the input trajectory, used by MATLAB's
% fmincon solver in the active-learning mode.
%
% The active-learning formulation is presented in Section 5 of the paper:
%   "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model".
% 
% Paper: https://arxiv.org/abs/2504.05490
% Requirements: nonlinearBayesian4Wiener library; see README.md for details.
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: October 2025

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