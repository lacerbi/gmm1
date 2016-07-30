function [y,g,h] = gmm1pdf(x,w,mu,sigma)
%GMM1PDF 1-d Gaussian mixture (gmm) probability density function (pdf).
%   Y = GMM1PDF(X,W,MU,SIGMA) returns the pdf of the 1-dimensional
%   Gaussian mixture model (gmm) with mixing weights W, means MU and 
%   standard deviations SIGMA, evaluated at the values in X.
%   W, MU and SIGMA can either be row vectors or scalars. A scalar input 
%   parameter functions as a constant vector of the same size as the other 
%   input gmm parameters. W, MU and SIGMA can also be matrices, in which
%   case each row of W, MU and SIGMA corresponds to a different gmm.
%   
%   W and SIGMA must be positive-valued, otherwise a vector of NaNs is 
%   returned.
%   Default values for MU and SIGMA are 0 and 1 respectively.
%
%   [Y,G]= GMM1PDF(...) returns the first derivative G of the gmm pdf, 
%   evaluated at the values in X.
%
%   [Y,G,H]= GMM1PDF(...) returns the second derivative H of the gmm 
%   pdf, evaluated at the values in X.
%
%   The gmm (unnormalized) pdf has shape: 
%      gmm(x) = sum(W.*normpdf(x, MU, SIGMA))
%   The pdf is normalized only if the weight vector W sums to one.
%
%   See also GMM1CDF, GMM1MAX, GMM1PROD, NORMPDF.

%   Reference:
%   Acerbi, L., Vijayakumar, S. & Wolpert, D. M. (2014). "On the Origins 
%   of Suboptimality in Human Probabilistic Inference", PLoS Computational 
%   Biology.

%   Copyright (c) by Luigi Acerbi, June 2014

if nargin<1
    error('gmm1pdf:TooFewInputs','Input argument X is undefined.');
end
if nargin<2
    error('gmm1pdf:TooFewInputs','Input argument W is undefined.');
end
if nargin<3
    mu = 0;
end
if nargin<4
    sigma = 1;
end
% Return NaN for out of range parameters.
if any(w(:) < 0) || any(sigma(:) < 0)
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

M = size(w, 2); % Number of components
N = size(w, 1); % Number of gmms

y = zeros(size(x,1), size(x,2));
if nargout > 1
    g = zeros(size(x,1), size(x,2));
end
if nargout > 2
    h = zeros(size(x,1), size(x,2));
end

for m = 1:M
    tau = 1./sigma(:,m).^2; % Precision
    z = bsxfun(@times, w(:, m)./sigma(:, m), exp(-0.5 * bsxfun(@times, (bsxfun(@minus,x,mu(:,m))).^2, tau)))/sqrt(2*pi);
    y = bsxfun(@plus, y, z);
    if nargout > 1
        t = bsxfun(@times, bsxfun(@minus, mu(:,m), x), tau);
        g = g + bsxfun(@times, z, t);
    end
    if nargout > 2
        h = h + bsxfun(@times, z, bsxfun(times, -t-1, tau));
    end
end

end