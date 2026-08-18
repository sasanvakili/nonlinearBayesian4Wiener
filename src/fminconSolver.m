function [optimizer, optimalUbar] = fminconSolver(model, settings)
% This function finds the optimal Input sequence using SQP algorithm 
% with a given gradient.
%
% The algorithm optimizes input signals to minimize the Bayesian MMSE
% estimation error for Wiener system identification. It returns estimator
% parameters, Optimization metrics (e.g., convergence status), Optimal
% input signal.
%
% The active-learning formulation is presented in Section 5 of the paper:
%   "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model".
% 
% Paper: https://arxiv.org/abs/2504.05490
% Requirements: nonlinearBayesian4Wiener library; see README.md for details.
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: October 2025

% Decision variable: vecUbar
if (settings.verbose >= 1)
    msg = 'Sequential Quadratic Programming (SQP) algorithm via fmincon solver:';
    disp(msg)
end

switch settings.verbose
    case 0
        displayFlag = 'off';
    case 1
        displayFlag = 'iter';
    case 2
        displayFlag = 'iter-detailed';
end

options = optimoptions(@fmincon, 'Algorithm', 'sqp', 'SpecifyObjectiveGradient', true, ...
    'MaxIterations', settings.activeLearning.maxIter, 'Display', displayFlag);
problem.options = options;

% Linear input constraint
if (settings.activeLearning.existConstraint)
    if ~(settings.activeLearning.applyToInitX)
        %InitState is not optimized!
        maxConstraint = [model.vecUbar(1:model.numState); 
            repmat(settings.activeLearning.vecUmax, model.trajectoryT, 1)];
        minConstraint = [model.vecUbar(1:model.numState);
            repmat(settings.activeLearning.vecUmin, model.trajectoryT, 1)];
    else
        maxConstraint = [settings.activeLearning.maxInitState;
            repmat(settings.activeLearning.vecUmax, model.trajectoryT, 1)];
        minConstraint = [settings.activeLearning.minInitState;
            repmat(settings.activeLearning.vecUmin, model.trajectoryT, 1)];
    end
end
problem.ub = maxConstraint;
problem.lb = minConstraint;

problem.x0 = model.vecUbar;

problem.objective = @(vecUbar)fminconMMSEgradient(vecUbar, model, settings);

problem.solver = 'fmincon';

[optVecU, optCost, exitFlag, output] = fmincon(problem);

if (settings.verbose >= 1)
    msg = ['Total iterations: ', num2str(output.iterations), newline, ...
        'Final cost: ', num2str(optCost)];
    disp(msg);
end

optimizer.status = exitFlag;
optimizer.output = output;
optimizer.optimalCost = optCost;
optimalUbar = optVecU;
end
