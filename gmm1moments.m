function [m1,m2,m3,m4] = gmm1moments(w,mu,sigma)
%GMM1MOMENTS returns the central moments of a 1-d Gaussian mixture (gmm).
%   M1 = GAUSSMIXPDF(W,MU,SIGMA) returns the mean M1 (1st central moment) of 
%   the 1-dimensional Gaussian mixture model (gmm) with mixing weights W, 
%   means MU and standard deviations SIGMA.
%   W, MU and SIGMA can either be vectors of the same size or scalars. 
%   A scalar input parameter functions as a constant vector of the same 
%   size as the other input parameters. 
%   
%   W and SIGMA must be positive-valued, otherwise a vector of NaNs is 
%   returned. W needs not be normalized.
%   Default values for MU and SIGMA are 0 and 1 respectively.   
%
%   [M1,M2]= GAUSSMIXPDF(...) returns the variance M2 (2nd central moment).
%
%   [M1,M2,M3]= GAUSSMIXPDF(...) returns the skewness M3.
%
%   [M1,M2,M3,M4]= GAUSSMIXPDF(...) returns the excess kurtosis M4.
%
%   The gmm (unnormalized) pdf has shape: 
%      gmm(x) = sum(W.*normpdf(x, MU, SIGMA))
%   The pdf is normalized only if the weight vector W sums to one.
%
%   See also GMM1CDF, GMM1MAX, GMM1PDF, NORMPDF.

%   Copyright (c) by Luigi Acerbi, July 2013

if nargin<1
    error('gmm1pdf:TooFewInputs','Input argument W is undefined.');
end
if nargin<2
    mu = 0;
end
if nargin<3
    sigma = 1;
end
% Return NaN for out of range parameters.
if any(w(:) <= 0) || any(sigma(:) <= 0)
    y = NaN(size(x),class(x));
    return;
end

% Convert scalar input to vectors
if isscalar(w) && ~isscalar(mu);
    w = w*ones(size(mu,1), size(mu,2)); 
elseif isscalar(w) 
    w = w*ones(size(sigma,1), size(sigma,2));        
end
if isscalar(mu); mu = mu*ones(size(w,1), size(w,2)); end
if isscalar(sigma); sigma = sigma*ones(size(w,1), size(w,2)); end

% Number of components
if size(w,1) > size(w,2); n = size(w,1); else n = size(w,2); end

% Normalize W
w = bsxfun(@rdivide, w, sum(w, 2));

% Mean
m1 = sum(w.*mu, 2);

% Variance
if nargout > 1
    m2 = sum(w.*(mu.^2 + sigma.^2), 2) - m1.^2;
end

% Skewness
if nargout > 2
    dmu = bsxfun(@minus, mu, m1);
    m3 = sum(w.*dmu.*(3*sigma.^2 + dmu.^2), 2)./(m2.^1.5);
    % m3 = sum(w.*(mu.^3 + 3*(sigma.^2).*mu - 3*(mu.^2 + sigma.^2).*(m1*T) + 3*mu.*((m1.^2)*T) - (m1.^3)*T), 2)./(m2.^1.5);
end

% Excess kurtosis
if nargout > 3
    m4 = sum(w.*(3*sigma.^4 + 6*dmu.^2.*(sigma.^2) + dmu.^4), 2)./(m2.^2) - 3;
    % m4 = (sum(w.*(mu.^4 + 6*mu.^2.*sigma.^2 + 3*sigma.^4 - 4*(m1*T).*(mu.^3 + 3*sigma.^2.*mu) + 6*(mu.^2 + sigma.^2).*((m1.^2)*T) - 4*((m1.^3)*T).*mu )) + (m1.^4)*T)./(m2.^2) - 3;
end

end