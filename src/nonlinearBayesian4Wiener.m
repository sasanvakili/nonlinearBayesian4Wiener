function [estimator, optimizer, optimalUbar] = nonlinearBayesian4Wiener(model, settings, vecYbar)
% This function implements the methods developed in the paper:
%   "Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization"
%
% Depending on the selected operating mode, this function:
%
%   1. Computes the optimal Bayesian affine MMSE estimator for unknown
%      observation-model parameters and the corresponding parameter estimate.
%
%   2. Computes a dual estimator for joint learning of unknown parameters
%      and latent variables, using either the dual basis-parameter (DB-P)
%      or dual state-parameter (DS-P) formulation.
%
%   3. Designs an experiment input using active learning to minimize the
%      anticipated affine parameter-estimation error. This mode is based on the paper:
%           "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model".
%
% Papers: 
%   https://arxiv.org/abs/2606.10111
%   https://arxiv.org/abs/2504.05490
% Requirements: See README.md for installation and usage details.
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: October 2025

settings = validateInputs(model, settings, vecYbar);
estimator = struct;

switch settings.mode
        case 'affineMMSEestimate'
            if (settings.verbose >= 1)
                msg = 'Computing theta estimates using affine estimator:';
                disp(msg)
            end
            vecBbarUbar = model.matrixBbar*model.vecUbar;
            estimator = thetaAffineEstimator(model, vecBbarUbar);
            estimator.thetaEstimate = estimator.matrixPsi*vecYbar+estimator.vecPsi;
            if (settings.verbose >= 2)
                msg = ['Optimal Err: ', num2str(estimator.thetaErr)];
                disp(msg);
            end
            optimizer = [];
            optimalUbar = [];
        case 'dualMMSEestimate'
            switch settings.dual.type
                case 'DS-P'
                    if (settings.verbose >= 1)
                        msg = 'Computing the dual estimates of theta and states:';
                        disp(msg)
                    end
                    estimator = dualStateEstimator(model, vecYbar, settings);
                    optimizer = [];
                    optimalUbar = [];
                case 'DB-P'
                    if (settings.verbose >= 1)
                        msg = 'Computing the dual estimates of theta and basis collection:';
                        disp(msg)
                    end
                    estimator = dualBasisEstimator(model, vecYbar, settings);
                    optimizer = [];
                    optimalUbar = [];
            end
        case 'activeLearning'
            switch settings.activeLearning.solver 
                case 'adaptive'
                    if (settings.verbose >= 1)
                        msg = ['activeLearning - adaptive: ' ...
                            'Computing optimal input and its corresponding estimator parameters:'];
                        disp(msg)
                    end
                    [estimator, optimizer, optimalUbar] = adaptiveGradientDescent(model, settings);
                case 'fmincon'
                    if (settings.verbose >= 1)
                        msg = ['activeLearning - fmincon: ' ...
                            'Computing optimal input:'];
                        disp(msg)
                    end
                    [optimizer, optimalUbar] = fminconSolver(model, settings);
            end
end