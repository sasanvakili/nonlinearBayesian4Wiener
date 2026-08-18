function settings = validateInputs(model, settings, vecYbar)
% This function validates the input arguments provided to
% nonlinearBayesian4Wiener, including their required fields, data types,
% dimensions, and admissible values.
%
% The validation ensures that `model`, `settings`, and `vecYbar` conform to
% the input format required by the selected estimation or active-learning
% mode. The checks are consistent with the Wiener-model formulation and
% estimation algorithms described in the papers:
%
%   "Nonlinear Bayesian Estimator for Parameter Learning: A Fixed-Point Characterization".
%
%   "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model".
%
% Papers: 
%   https://arxiv.org/abs/2606.10111
%   https://arxiv.org/abs/2504.05490
% Requirements: See README.md for input specifications and usage details.
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: October 2025

if ~isstruct(model)
    error('First input argument must be a struct');
end

if ~isstruct(settings)
    error('Second input argument must be a struct');
end

if ~isfield(settings, 'verbose')
    settings.verbose = 0;
elseif ~ismember(settings.verbose, [0, 1, 2])
    error('Invalid verbose value: must be 0, 1, or 2');
end

if ~isfield(settings, 'activeLearning')
    settings.activeLearning = [];
end

if ~isfield(settings, 'dual')
    settings.dual = [];
end

allowedFields = {'mode', 'activeLearning', 'verbose', 'dual'};
fields = fieldnames(settings);
if ~(numel(fields) == numel(allowedFields) && ...
        all(ismember(fields, allowedFields)) && ...
        all(ismember(allowedFields, fields)))
    error(['Nonlinear estimator settings: must have only the fields: ' ...
        'mode, activeLearning, verbose, dual']);
end

allowedFields = {'numState', 'numInput', 'numTheta', 'trajectoryT', ...
    'matrixAbar', 'matrixBbar', 'vecUbar', 'allVecFreq', 'muTheta', ... 
    'sigmaTheta', 'sigmaVbar', 'sigmaWbar'};
fields = fieldnames(model);
if ~(numel(fields) == numel(allowedFields) && ...
        all(ismember(fields, allowedFields)) && ...
        all(ismember(allowedFields, fields)))
    error(['model must have only the fields: numState, numInput, ' ...
        'numTheta, trajectoryT, matrixAbar, matrixBbar, vecUbar, allVecFreq,' ...
        ' muTheta, sigmaTheta, sigmaVbar, sigmaWbar']);
end

if ~(ismatrix(model.matrixAbar) && all(size(model.matrixAbar) == ...
        [model.numState*(model.trajectoryT+1), ...
        model.numState*(model.trajectoryT+1)]))
    error(['matrix Abar: dimension mismatch. ' ...
        'Expected size: [numState*(trajectoryT+1) by numState*(trajectoryT+1)].']);
end

if ~(ismatrix(model.matrixBbar) && all(size(model.matrixBbar) == ...
        [model.numState*(model.trajectoryT+1), ...
        model.numState+model.numInput*model.trajectoryT]))
    error(['matrix Bbar: dimension mismatch. ' ...
        'Expected size: [numState*(trajectoryT+1) by numState+numInput*trajectoryT].']);
end

if ~(isvector(model.vecUbar) && all(size(model.vecUbar) == ...
        [model.numState+model.numInput*model.trajectoryT 1]))
    error(['vector Ubar: dimension mismatch. ' ...
        'Expected size: [numState+numInput*trajectoryT by 1].']);
end

if ~(ismatrix(model.sigmaVbar) && all(size(model.sigmaVbar) == ...
        [model.trajectoryT+1 model.trajectoryT+1]))
    error(['matrix sigmaVbar: dimension mismatch. ' ...
        'Expected size: [trajectoryT+1 by trajectoryT+1].']);
end

if ~(ismatrix(model.sigmaWbar) && all(size(model.sigmaWbar) == ...
        [model.numState*(model.trajectoryT+1), ...
        model.numState*(model.trajectoryT+1)]))
    error(['matrix sigmaWbar: dimension mismatch. ' ...
        'Expected size: [numState*(trajectoryT+1) by numState*(trajectoryT+1)].']);
end

validModes = {'activeLearning', 'dualMMSEestimate', 'affineMMSEestimate'};
settings.mode = validatestring(settings.mode, validModes);

if (strcmp(settings.mode, 'affineMMSEestimate'))
    if ~(isvector(vecYbar) && all(size(vecYbar) ...
            == [model.trajectoryT+1, 1]))
        error(['vecYbar: dimension mismatch. ' ...
            'Expected size: [trajectoryT+1 by 1].']);
    end
end

if (strcmp(settings.mode, 'dualMMSEestimate'))
    if ~(isvector(vecYbar) && all(size(vecYbar) ...
            == [model.trajectoryT+1, 1]))
        error(['vecYbar: dimension mismatch. ' ...
            'Expected size: [trajectoryT+1 by 1].']);
    end
    
    if ~isfield(settings.dual, 'type')
        settings.dual.type = 'DS-P';
    else
        validDualTypes = {'DS-P', 'DB-P'};
        settings.dual.type = validatestring(settings.dual.type, ...
            validDualTypes);
    end

    allowedFields = {'tol', 'maxIter', 'type'};
    fields = fieldnames(settings.dual);
    if ~(numel(fields) == numel(allowedFields) && ...
            all(ismember(fields, allowedFields)) && ...
            all(ismember(allowedFields, fields)))
        error('settings.dual: must have only the fields: tol, maxIter, type');
    end

    if ~(isnumeric(settings.dual.tol) && isscalar(settings.dual.tol))
        error(['Invalid tolerance for dual estimation: settings.dual.tol ' ...
            'must be a scalar numerical value (e.g., 1e-6).'])
    end

    if ~(isnumeric(settings.dual.maxIter) && ...
            isscalar(settings.dual.maxIter))
        error(['Invalid maximum iteration tolerance for dual estimation: ' ...
            'settings.dual.maxIter must be a scalar numerical ' ...
            'value (e.g., 10000).'])
    end
end

if (strcmp(settings.mode, 'activeLearning'))
    if ~isfield(settings.activeLearning, 'solver')
        settings.activeLearning.solver = 'adaptive';
    else
        validSolvers = {'adaptive', 'fmincon'};
        settings.activeLearning.solver = validatestring(settings.activeLearning.solver, ...
            validSolvers);
    end
    switch settings.activeLearning.solver
        case 'fmincon'
            allowedFields = {'solver', 'maxIter','applyToInitX', 'existConstraint', ...
                'vecUmax', 'vecUmin', 'maxInitState', 'minInitState'};
            fields = fieldnames(settings.activeLearning);
            if ~(numel(fields) == numel(allowedFields) && ...
                    all(ismember(fields, allowedFields)) && ...
                    all(ismember(allowedFields, fields)))
                error(['settings.activeLearning - fmincon solver: must have only the fields: ' ...
                    'maxIter, applyToInitX, existConstraint, vecUmax, vecUmin, ' ...
                    'maxInitState, minInitState']);
            end
        case 'adaptive'
            allowedFields = {'solver', 'gradTol', 'costTol', 'maxIter', 'alpha', ...
                'beta', 'applyToInitX', 'existConstraint', 'vecUmax', 'vecUmin', ...
                'maxInitState', 'minInitState'};
            fields = fieldnames(settings.activeLearning);
            if ~(numel(fields) == numel(allowedFields) && ...
                    all(ismember(fields, allowedFields)) && ...
                    all(ismember(allowedFields, fields)))
                error(['settings.activeLearning - adaptive gradient descent: ' ...
                    'must have only the fields: gradTol, costTol, maxIter, ' ...
                    'alpha, beta, applyToInitX, existConstraint, vecUmax, ' ...
                    'vecUmin, maxInitState, minInitState']);
            end
            if ~(isnumeric(settings.activeLearning.gradTol) && ...
                    isscalar(settings.activeLearning.gradTol))
                error(['Invalid gradient tolerance: settings.activeLearning.gradTol ' ...
                    'must be a scalar numerical value (e.g., 1e-6).'])
            end
            if ~(isnumeric(settings.activeLearning.costTol) && ...
                    isscalar(settings.activeLearning.costTol))
                error(['Invalid cost tolerance: settings.activeLearning.costTol ' ...
                    'must be a scalar numerical value (e.g., 1e-6).'])
            end
            if ~(isnumeric(settings.activeLearning.alpha) && ...
                    isscalar(settings.activeLearning.alpha))
                error(['Invalid alpha initialization: ' ...
                    'settings.activeLearning.alpha must be a scalar numerical ' ...
                    'value (e.g., 1e-10).'])
            end
            if ~(isnumeric(settings.activeLearning.beta) && ...
                    isscalar(settings.activeLearning.beta))
                error(['Invalid beta initialization: ' ...
                    'settings.activeLearning.beta must be a scalar numerical ' ...
                    'value (e.g., 1e100).'])
            end
    end
    if ~(isnumeric(settings.activeLearning.maxIter) && ...
            isscalar(settings.activeLearning.maxIter))
        error(['Invalid maximum iteration tolerance: ' ...
            'settings.activeLearning.maxIter must be a scalar numerical ' ...
            'value (e.g., 10000).'])
    end
    if ~(islogical(settings.activeLearning.existConstraint) || ...
            (isnumeric(settings.activeLearning.existConstraint) && ...
            ismember(settings.activeLearning.existConstraint, [0 1])))
        error('settings.activeLearning.existConstraint must be logical or 0/1.');
    end
    if (isnumeric(settings.activeLearning.existConstraint) && ...
            ismember(settings.activeLearning.existConstraint, [0 1]))
        settings.activeLearning.existConstraint = logical( ...
            settings.activeLearning.existConstraint);
    end
    if ~(islogical(settings.activeLearning.applyToInitX) || ...
            (isnumeric(settings.activeLearning.applyToInitX) && ...
            ismember(settings.activeLearning.applyToInitX, [0 1])))
        error('settings.activeLearning.applyToInitX must be logical or 0/1.');
    end
    if (isnumeric(settings.activeLearning.applyToInitX) && ...
            ismember(settings.activeLearning.applyToInitX, [0 1]))
        settings.activeLearning.applyToInitX = logical( ...
            settings.activeLearning.applyToInitX);
    end
    if (settings.activeLearning.existConstraint)
        if ~(isvector(settings.activeLearning.vecUmax) && ...
                all(size(settings.activeLearning.vecUmax) == [model.numInput 1]))
            error(['settings.activeLearning.vecUmax: dimension mismatch. ' ...
                'Expected size: [numInput by 1].']);
        end
        if ~(isvector(settings.activeLearning.vecUmin) && ...
                all(size(settings.activeLearning.vecUmin) == [model.numInput 1]))
            error(['settings.activeLearning.vecUmin: dimension mismatch. ' ...
                'Expected size: [numInput by 1].']);
        end
        if (settings.activeLearning.applyToInitX)
            if ~(isvector(settings.activeLearning.maxInitState) && ...
                    all(size(settings.activeLearning.maxInitState) == [model.numState 1]))
                error(['settings.activeLearning.maxInitState: dimension mismatch. ' ...
                    'Expected size: [numState by 1].']);
            end
            if ~(isvector(settings.activeLearning.minInitState) && ...
                    all(size(settings.activeLearning.minInitState) == [model.numState 1]))
                error(['settings.activeLearning.minInitState: dimension mismatch. ' ...
                    'Expected size: [numState by 1].']);
            end
        end
    end
end

if ~(ismatrix(model.allVecFreq) && all(size(model.allVecFreq) == [model.numState model.numTheta-1]))
    error(['allVecFreq: dimension mismatch. ' ...
        'Expected size: [numState by numTheta-1].']);
end

zeroFreq = all(model.allVecFreq == 0, 1);
if any(zeroFreq)
    index = find(zeroFreq);
    error(['allVecFreq has at least one column of all zeros. ', ...
    'Zero frequencies are not expected. All-zero columns: %s'], mat2str(index));
end

if ~(isvector(model.muTheta) && all(size(model.muTheta) == [model.numTheta 1]))
    error(['vector muTheta: dimension mismatch. ' ...
        'Expected size: [numTheta by 1].']);
end

if ~(ismatrix(model.sigmaTheta) && all(size(model.sigmaTheta) == ...
        [model.numTheta model.numTheta]))
    error(['matrix sigmaTheta: dimension mismatch. ' ...
        'Expected size: [numTheta by numTheta].']);
end

end