function [wp,mup,sigmap] = gmm1prod(w,mu,sigma,w2,mu2,sigma2,skip)
%GMM1PROD product of two 1-d Gaussian mixtures (gmm).
%   [WP,MUP,SIGMAP] = GAUSSMIXPDF(W,MU,SIGMA,W2,MU2,SIGMA2) returns the 
%   1-dimensional Gaussian mixture model (gmm) with mixing weights WP, 
%   means MUP and standard deviations SIGMAP that is the normalized 
%   product of two gmms with weights W1, W2, means MU, MU2 and standard 
%   deviation SIGMA, SIGMA2.
%   W, W2, MU, MU2, SIGMA and SIGMA2 can either be row vectors or scalars. 
%   A scalar input parameter functions as a constant vector of the same 
%   size as the other input gmm parameters. W, W2, MU, MU2, SIGMA and 
%   SIGMA2 can also be matrices, in which case each row corresponds to a 
%   different gmm.
%
%   The number of components of the output gmm is the product of the number
%   of components (columns) of the two input gmm's. 
%
%   [WP,MUP,SIGMAP] = GAUSSMIXPDF(...,SKIP) skips all argument-checking.
%   If SKIP is 1, all input vectors for the first gmm are assumed to be of
%   the same size, and analogously for the second gmm.
%
%   GMM1PROD is useful, for example, in Bayesian inference when computing 
%   the posterior distribution from a prior and likelihood terms which are 
%   both gmms. Suppose that W,MU,SIGMA2 identify the prior and W2,MU2,SIGMA2 
%   denote the likelihood. The posterior mean can be computed as:
%     [wp,mup,sigmap] = gmm1prod(w,mu,sigma,w2,mu2,sigma2);
%     postmean = sum(wp.*mup, 2);
%   
%   See also GMM1CDF, GMM1MAX, GMM1PDF, NORMPDF.

%   If you use this, please cite:
%   Acerbi, L., Vijayakumar, S. & Wolpert, D. M. (2014). "On the Origins 
%   of Suboptimality in Human Probabilistic Inference", PLoS Computational 
%   Biology.

%   Copyright (c) by Luigi Acerbi, June 2014

if nargin<1
    error('gmm1prod:TooFewInputs','Input argument W is undefined.');
end
if nargin<4
    error('gmm1prod:TooFewInputs','Input argument W2 is undefined.');
end
if nargin<7; skip=0; end

% Return NaN for out of range parameters.
% if any(w <= 0) || any(sigma <= 0)
%     y = NaN(size(x),class(x));
%     return;
% end

% Input arguments check and formatting
if ~skip
    % Convert scalar input to vectors (first gmm)
    if isscalar(w) && ~isscalar(mu);
        w = w*ones(size(mu,1), size(mu,2)); 
    elseif isscalar(w) 
        w = w*ones(size(sigma,1), size(sigma,2));        
    end
    if isscalar(mu); mu = mu*ones(size(w,1), size(w,2)); end
    if isscalar(sigma); sigma = sigma*ones(size(w,1), size(w,2)); end

    % Convert scalar input to vectors (second gmm)
    if isscalar(w2) && ~isscalar(mu2);
        w2 = w2*ones(size(mu2,1), size(mu2,2)); 
    elseif isscalar(w2) 
        w2 = w2*ones(size(sigma2,1), size(sigma2,2));        
    end
    if isscalar(mu2); mu2 = mu2*ones(size(w2,1), size(w2,2)); end
    if isscalar(sigma2); sigma2 = sigma2*ones(size(w2,1), size(w2,2)); end
end

M = size(w, 2); % Number of components, first gmm
M2 = size(w2, 2); % Number of components, second gmm
Mp = M*M2; % Number of components, product gmm

wp = zeros(size(w, 1), Mp);
mup = zeros(size(w, 1), Mp);
sigmap = zeros(size(w, 1), Mp);

for i = 1:M
    for j = 1:M2
        ii = (i-1)*M2 + j;
        sigmasq = sigma(:, i).^2 + sigma2(:, j).^2;
        s = exp(-0.5*(mu(:, i)-mu2(:, j)).^2./sigmasq)./sqrt(2*pi*sigmasq);
        wp(:, ii) = w(:, i).*w2(:, j).*s;
        mup(:, ii) = (mu(:, i).*sigma2(:, j).^2 + mu2(:, j).*sigma(:, i).^2)./sigmasq;
        sigmap(:, ii) = sigma(:, i).*sigma2(:, j)./sqrt(sigmasq);        
    end
end

% Normalization
wp = bsxfun(@rdivide, wp, sum(wp, 2));

end