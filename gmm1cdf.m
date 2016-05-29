function p = gmm1cdf(x,w,mu,sigma)
%GMM1CDF 1-d Gaussian mixture (gmm) cumulative distribution function (cdf).
%   P = GAUSSMIXCDF(X,W,MU,SIGMA) returns the cdf of the 1-dimensional
%   Gaussian mixture model (gmm) with mixing weights W, means MU and 
%   standard deviations SIGMA, evaluated at the values in X.
%   The size of Y is the size of X. W, MU and SIGMA can either be vectors 
%   of the same size or scalars. A scalar input parameter functions as a 
%   constant vector of the same size as the other input parameters. 
%   
%   W and SIGMA must be non-negative valued, otherwise a vector of NaNs is 
%   returned.
%   Default values for MU and SIGMA are 0 and 1 respectively.
%
%   See also ERF, ERFC, GMM1MAX, GMM1PDF, NORMCDF.

%   Copyright (c) by Luigi Acerbi, March 2013

if nargin<1
    error('gmm1cdf:TooFewInputs','Input argument X is undefined.');
end
if nargin<2
    error('gmm1cdf:TooFewInputs','Input argument W is undefined.');
end
if nargin<3
    mu = 0;
end
if nargin<4
    sigma = 1;
end

% Return NaN for out of range parameters.
if any(w(:) < 0) || any(sigma(:) < 0)
    p = NaN(size(x),class(x));
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

M = size(w, 2); % Number of components
N = size(w, 1); % Number of gmms

p = zeros(size(x,1),size(x,2));

for m = 1:M
    fsigmazero = (sigma(:, m) == 0);
    
    % Set edge case sigma=0
    if sum(fsigmazero) > 0
        ptemp = p(fsigmazero, :);        
        f = bsxfun(@ge, x(fsigmazero, :), mu(fsigmazero, m));
        pp = bsxfun(@plus, ptemp, w(fsigmazero, m));
        ptemp(f) = pp(f);
        p(fsigmazero, :) = ptemp;
        p(~fsigmazero, :) = p(~fsigmazero, :) + 0.5 * bsxfun(@times, w(~fsigmazero, m), ...
            erfc( bsxfun(@rdivide,bsxfun(@minus,mu(~fsigmazero, m),x(~fsigmazero, :)),sigma(~fsigmazero, m))/sqrt(2) ));        
    else
        % Use the complementary error function, rather than .5*(1+erf(z/sqrt(2))),
        % to produce accurate near-zero results for large negative x.
        p = p + 0.5 * bsxfun(@times, w(:, m), erfc( bsxfun(@rdivide,bsxfun(@minus,mu(:, m),x),sigma(:, m))/sqrt(2) ));        
    end
    
    % Set edge case sigma=0
    %if sigma(m) == 0
    %    p(x>=mu(m)) = p(x>=mu(m)) + w(m);
    %else
        % Use the complementary error function, rather than .5*(1+erf(z/sqrt(2))),
        % to produce accurate near-zero results for large negative x.
    %    p = p + 0.5 * w(m) * erfc( -(x-mu(m))/(sigma(m)*sqrt(2)) );        
    %end
end

end