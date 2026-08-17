close all
clear all
clc

% Add library path
addpath(genpath('../src'));

% Process noise in the experiment data includes:
%   1. vecWbar_1: random values generated with sigmaWbar_1 (sigmaW^2 = 0.001)
%   2. vecWbar_2: random values generated with sigmaWbar_2 (sigmaW^2 = 0.01)

% The following code demonstrates the performance of the offline estimators 
% for Experiment setup 2 (N=10, n_x=2), the second realization of vecVbar and vecWbar_1, 
% and the first realization of theta.
load('Experiment_setup_2/experimentData.mat')
% To try Experiment setup 1 (N=2, n_x=10) load Experiment_setup_1/experimentData.mat:
% load('Experiment_setup_1/experimentData.mat')
vecVbar = vecVbar(1:(model.trajectoryT+1), 2);
trueTheta = trueTheta(:, 1);
% Case - 0.001:
vecWbar = vecWbar_1(1:(model.trajectoryT+1)*model.numState, 2);
model.sigmaWbar = model.sigmaWbar_1;
% Case - 0.01:
% vecWbar = vecWbar_2(1:(model.trajectoryT+1)*model.numState, 1);
% model.sigmaWbar = model.sigmaWbar_2;
model = rmfield(model, 'sigmaWbar_1');
model = rmfield(model, 'sigmaWbar_2');

% Generate measurements from random variable data and the input sequence.
% Create system states:
vecXbar = model.matrixAbar*(model.matrixBbar*model.vecUbar + vecWbar);
% Create measurements:
vecYbar = fourierObservation(model.allVecFreq, trueTheta, vecXbar, ...
    model.numState)+vecVbar;
disp('-------------------------------------------------------------------------------')
%% Affine Estimator - Sinusoid input:
disp('Affine Theta Estimate with sinusoid input:')
settings = struct;
settings.mode = 'affineMMSEestimate';
settings.verbose = 2;
tic
[affineEstimator, ~, ~] = nonlinearBayesian4Wiener(model, settings, vecYbar);
toc
affineThetaEst = affineEstimator.thetaEstimate;
% Affine Estimate Error:
affineThetaEstErr = sum((trueTheta-affineThetaEst).^2);
msg = ['Optimal Error:', num2str(affineEstimator.thetaErr)];
disp(msg)
msg = ['Theta Error:', num2str(affineThetaEstErr)];
disp(msg)
disp('-------------------------------------------------------------------------------')
%% Dual State-Parameter Estimator - Sinusoid input:
disp('Theta Estimate from Dual State-Parameter estimator with sinusoid input:')
settings = struct;
settings.mode = 'dualMMSEestimate';
settings.dual.tol = 1e-6;
settings.dual.maxIter = 10000;
settings.dual.type = 'State_Parameter';
settings.verbose = 2;
tic
[dualStateParameterEstimator, ~, ~] = nonlinearBayesian4Wiener(model, settings, vecYbar);
toc
dualThetaEst = dualStateParameterEstimator.thetaEstimate;
% Dual Estimate Error:
dualThetaEstErr = sum((trueTheta-dualThetaEst).^2);
msg = ['Number of Iterations: ' num2str(dualStateParameterEstimator.totalIter)];
disp(msg)
msg = ['Optimal Error:', num2str(trace(dualStateParameterEstimator.sigmaThetaEstimate))];
disp(msg)
msg = ['Theta Error:', num2str(dualThetaEstErr)];
disp(msg)
disp('-------------------------------------------------------------------------------')

%% Affine and Dual Estimators - Active Learning (fmincon solver):
disp('Optimal Input computation (fmincon solver):');
disp('-------------------------------------------------------------------------------')
settings = struct;
settings.mode = 'activeLearning';
settings.verbose = 2;
settings.activeLearning.solver = 'fmincon';
settings.activeLearning.maxIter = 10; %10000; 
settings.activeLearning.applyToInitX = false;
settings.activeLearning.existConstraint = true;
settings.activeLearning.vecUmax = inputConstraint.vecUmax;
settings.activeLearning.vecUmin = inputConstraint.vecUmin;
settings.activeLearning.maxInitState = [];
settings.activeLearning.minInitState = [];
vecYbar = [];
[~, optimizer, optimalUbar] = nonlinearBayesian4Wiener(model, settings, vecYbar);
originalVecUbar = model.vecUbar;
model.vecUbar = optimalUbar;
% Generate measurements from random variable data and the input sequence.
% Create system states:
vecXbar = model.matrixAbar*(model.matrixBbar*model.vecUbar + vecWbar);
% Create measurements:
vecYbar = fourierObservation(model.allVecFreq, trueTheta, vecXbar, ...
    model.numState)+vecVbar;
disp('-------------------------------------------------------------------------------')

disp('Affine Theta Estimate with optimal vecUbar from Active Learning:')
settings = struct;
settings.mode = 'affineMMSEestimate';
settings.verbose = 2;
tic
[affineEstimator, ~, ~] = nonlinearBayesian4Wiener(model, settings, vecYbar);
toc
affineThetaEst = affineEstimator.thetaEstimate;
% Affine Estimate Error:
affineThetaEstErr = sum((trueTheta-affineThetaEst).^2);
msg = ['Optimal Error:', num2str(affineEstimator.thetaErr)];
disp(msg)
msg = ['Theta Error:', num2str(affineThetaEstErr)];
disp(msg)
disp('-------------------------------------------------------------------------------')

disp(['Theta Estimate from Dual State-Parameter estimator with optimal vecUbar ' ...
    'from Active Learning:'])
settings = struct;
settings.mode = 'dualMMSEestimate';
settings.dual.tol = 1e-6;
settings.dual.maxIter = 10000;
settings.dual.type = 'State_Parameter';
settings.verbose = 2;
tic
[dualStateParameterEstimator, ~, ~] = nonlinearBayesian4Wiener(model, settings, vecYbar);
toc
dualThetaEst = dualStateParameterEstimator.thetaEstimate;
% Dual Estimate Error:
dualThetaEstErr = sum((trueTheta-dualThetaEst).^2);
msg = ['Number of Iterations: ' num2str(dualStateParameterEstimator.totalIter)];
disp(msg)
msg = ['Optimal Error:', num2str(trace(dualStateParameterEstimator.sigmaThetaEstimate))];
disp(msg)
msg = ['Theta Error:', num2str(dualThetaEstErr)];
disp(msg)
disp('-------------------------------------------------------------------------------')

disp(['Theta Estimate from Dual Basis-Parameter estimator with optimal vecUbar ' ...
    'from Active Learning:'])
settings = struct;
settings.mode = 'dualMMSEestimate';
settings.dual.tol = 1e-6;
settings.dual.maxIter = 100; %10000;
settings.dual.type = 'Basis_Parameter';
settings.verbose = 2;
tic
[dualBasisParameterEstimator, ~, ~] = nonlinearBayesian4Wiener(model, settings, vecYbar);
toc
dualThetaEst = dualBasisParameterEstimator.thetaEstimate;
% Dual Estimate Error:
dualThetaEstErr = sum((trueTheta-dualThetaEst).^2);
msg = ['Number of Iterations: ' num2str(dualBasisParameterEstimator.totalIter)];
disp(msg)
msg = ['Optimal Error:', num2str(trace(dualBasisParameterEstimator.sigmaThetaEstimate))];
disp(msg)
msg = ['Theta Error:', num2str(dualThetaEstErr)];
disp(msg)
disp('-------------------------------------------------------------------------------')

%% Affine and Dual Estimators - Active Learning (adaptive gradient descent solver):
disp('Optimal Input computation (adaptive gradient descent solver):');
disp('-------------------------------------------------------------------------------')
settings = struct;
settings.mode = 'activeLearning';
settings.verbose = 2;
settings.activeLearning.solver = 'adaptive';
settings.activeLearning.gradTol = 1e-6;
settings.activeLearning.costTol = 1e-6;
settings.activeLearning.maxIter = 10; %10000;
settings.activeLearning.alpha = 1e-10;
settings.activeLearning.beta = 1e100;
settings.activeLearning.applyToInitX = false;
settings.activeLearning.existConstraint = true;
settings.activeLearning.vecUmax = inputConstraint.vecUmax;
settings.activeLearning.vecUmin = inputConstraint.vecUmin;
settings.activeLearning.maxInitState = [];
settings.activeLearning.minInitState = [];
vecYbar = [];
model.vecUbar = originalVecUbar;
[~, optimizer, optimalUbar] = nonlinearBayesian4Wiener(model, settings, vecYbar);
model.vecUbar = optimalUbar;
% Generate measurements from random variable data and the input sequence.
% Create system states:
vecXbar = model.matrixAbar*(model.matrixBbar*model.vecUbar + vecWbar);
% Create measurements:
vecYbar = fourierObservation(model.allVecFreq, trueTheta, vecXbar, ...
    model.numState)+vecVbar;
disp('-------------------------------------------------------------------------------')

disp('Affine Theta Estimate with optimal vecUbar from Active Learning:')
settings = struct;
settings.mode = 'affineMMSEestimate';
settings.verbose = 2;
tic
[affineEstimator, ~, ~] = nonlinearBayesian4Wiener(model, settings, vecYbar);
toc
affineThetaEst = affineEstimator.thetaEstimate;
% Affine Estimate Error:
affineThetaEstErr = sum((trueTheta-affineThetaEst).^2);
msg = ['Optimal Error:', num2str(affineEstimator.thetaErr)];
disp(msg)
msg = ['Theta Error:', num2str(affineThetaEstErr)];
disp(msg)
disp('-------------------------------------------------------------------------------')

disp(['Theta Estimate from Dual State-Parameter estimator with optimal vecUbar ' ...
    'from Active Learning:'])
settings = struct;
settings.mode = 'dualMMSEestimate';
settings.dual.tol = 1e-6;
settings.dual.maxIter = 10000;
settings.dual.type = 'State_Parameter';
settings.verbose = 2;
tic
[dualStateParameterEstimator, ~, ~] = nonlinearBayesian4Wiener(model, settings, vecYbar);
toc
dualThetaEst = dualStateParameterEstimator.thetaEstimate;
% Dual Estimate Error:
dualThetaEstErr = sum((trueTheta-dualThetaEst).^2);
msg = ['Number of Iterations: ' num2str(dualStateParameterEstimator.totalIter)];
disp(msg)
msg = ['Optimal Error:', num2str(trace(dualStateParameterEstimator.sigmaThetaEstimate))];
disp(msg)
msg = ['Theta Error:', num2str(dualThetaEstErr)];
disp(msg)
disp('-------------------------------------------------------------------------------')

disp(['Theta Estimate from Dual Basis-Parameter estimator with optimal vecUbar ' ...
    'from Active Learning:'])
settings = struct;
settings.mode = 'dualMMSEestimate';
settings.dual.tol = 1e-6;
settings.dual.maxIter = 100; %10000;
settings.dual.type = 'Basis_Parameter';
settings.verbose = 2;
tic
[dualBasisParameterEstimator, ~, ~] = nonlinearBayesian4Wiener(model, settings, vecYbar);
toc
dualThetaEst = dualBasisParameterEstimator.thetaEstimate;
% Dual Estimate Error:
dualThetaEstErr = sum((trueTheta-dualThetaEst).^2);
msg = ['Number of Iterations: ' num2str(dualBasisParameterEstimator.totalIter)];
disp(msg)
msg = ['Optimal Error:', num2str(trace(dualBasisParameterEstimator.sigmaThetaEstimate))];
disp(msg)
msg = ['Theta Error:', num2str(dualThetaEstErr)];
disp(msg)
disp('-------------------------------------------------------------------------------')

% -------------------------------------------------------------------------------
% Fourier observation model:
function vecY = fourierObservation(allVecFreq, theta, vecX, numState)
dataLen = length(vecX)/numState;
vecY = zeros(dataLen,1);
theta0 = repmat(theta(1),dataLen,1);
theta(1) = [];
for i=0:dataLen-1
    tempX = vecX(numState*i+1:numState*i+numState);
    vecY(i+1) = (theta')*(exp(1i*(allVecFreq')*tempX)+exp(-1i*(allVecFreq')*tempX));
end
vecY = vecY+theta0;
if ~isempty(vecY(imag(vecY)>=1e-12))
    error('Error: vectorY has imaginary part!')
end
vecY = real(vecY);
end