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