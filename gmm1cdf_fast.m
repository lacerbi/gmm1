function p = gmm1cdf_fast(x,w,mu,sigma)
%GMM1CDF 1-d Gaussian mixture (gmm) cumulative distribution function (cdf).
%   P = GAUSSMIXCDF_FAST(X,W,MU,SIGMA) returns the cdf of the 1-dimensional
%   Gaussian mixture model (gmm) with mixing weights W, means MU and 
%   standard deviations SIGMA, evaluated at the values in X.
%   The size of Y is the size of X. W, MU and SIGMA need to be row vectors
%   of the same size. 
%   
%   W and SIGMA must be non-negative valued.
%
%   See also ERF, ERFC, GMM1MAX, GMM1CDF, GMM1PDF, NORMCDF.

%   Copyright (c) by Luigi Acerbi, August 2016

if nargin<4
    error('gmm1cdf_fast:TooFewInputs','Input argument X is undefined.');
end

M = size(w, 2); % Number of components
p = zeros(size(x));

for m = 1:M
    
    % Use the complementary error function, rather than .5*(1+erf(z/sqrt(2))),
    % to produce accurate near-zero results for large negative x.
    p = p + 0.5 * bsxfun(@times, w(:, m), erfc( bsxfun(@rdivide,bsxfun(@minus,mu(:, m),x),sigma(:, m))/sqrt(2) ));        
end

end