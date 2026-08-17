function [estimator, optimizer, optimalUbar] = nonlinearBayesian4Wiener(model, settings, vecYbar)
% TBD
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: July 2025

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
                case 'State_Parameter'
                    if (settings.verbose >= 1)
                        msg = 'Computing the dual estimates of theta and states:';
                        disp(msg)
                    end
                    estimator = dualStateEstimator(model, vecYbar, settings);
                    optimizer = [];
                    optimalUbar = [];
                case 'Basis_Parameter'
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