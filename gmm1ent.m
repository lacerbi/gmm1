function h = gmm1ent(w,mu,sigma,nx)
%GMM1ENT 1-d Gaussian mixture (gmm) differential entropy.
%   H = GMM1ENT(W,MU,SIGMA) returns the differential entropy of one or more
%   1-dimensional Gaussian mixture models (gmm) with mixing weights W, 
%   means MU and standard deviations SIGMA.
%   W is a NxM matrix of mixing weights (N is the number of gmms and M the
%   maximum number of components per gmm). MU is a NxM matrix of 
%   components' means. SIGMA is a NxM matrix of components' standard 
%   deviations.
%   W, MU and SIGMA can either be matrices of the same size or scalars. 
%   A scalar input parameter functions as a constant matrix of the same 
%   size as the other input parameters. 
%   
%   For the computation of the differential entropy to be meaningful, the 
%   rows of W must sum to one and W must be non-negative. SIGMA must be
%   positive-valued. It is up to the user to meet these requirements.
%   It is possible for a mixture component to be absent (mixing weight set 
%   to 0).
%   Default values for MU and SIGMA are 0 and 1 respectively.
%
%   H = GMM1ENT(W,MU,SIGMA,NX) divides the integral of the differential
%   entropy in a grid of NX steps (default NX=5000). More steps provide
%   a more accurate value of the differential entropy.
%
%   See also GMM1CDF, GMM1MAX, GMM1PDF, GMM1PROD, NORMPDF.

%   Copyright (c) by Luigi Acerbi, April 2014

if nargin<1
    error('gmm1ent:TooFewInputs','Input argument W is undefined.');
end
if nargin<2
    mu = 0;
end
if nargin<3
    sigma = 1;
end
if nargin<4
    nx = 5000;
end

% Convert scalar input to vectors
if isscalar(w) && ~isscalar(mu);
    w = w*ones(size(mu,1), size(mu,2)); 
elseif isscalar(w) 
    w = w*ones(size(sigma,1), size(sigma,2));        
end
if isscalar(mu); mu = mu*ones(size(w,1), size(w,2)); end
if isscalar(sigma); sigma = sigma*ones(size(w,1), size(w,2)); end

% Check for absent components, set sigma to non-zero value to avoid NaNs
sigma(w == 0) = 1;

M = size(w, 2); % Number of components
N = size(w, 1); % Number of gmms

nx = double(nx);
sdmax = log(nx);
h = zeros(N,1);

if 0
    xx = zeros(N,M*nx);
    for m = 1:M
        y = [-sdmax + ((0:nx-2).*(2*sdmax)/(floor(nx)-1)), sdmax];
        xx(:,(1:nx)+(m-1)*nx) = sigma(:, m)*y + mu(:, m)*ones(1, nx);
    end
    xx = sort(xx,2);

    V = ones(1, nx*M);
    for m = 1:M        
        yy = exp(-0.5*((xx - mu(:,m)*V)./(sigma(:,m)*V)).^2)./(sigma(:,m)*V)/sqrt(2*pi).*log(gmm1pdf_private(xx, w, mu, sigma));
        h = h - w(:,m).*sum(0.5*diff(xx, [], 2).*(yy(:, 1:end-1) + yy(:, 2:end)), 2);
    end

else
    
    % V = ones(1, nx);
    for m = 1:M
        y = [-sdmax + ((0:nx-2).*(2*sdmax)/(floor(nx)-1)), sdmax];
        xx = sigma(:, m)*y + mu(:, m)*ones(1, nx);
        dx = 2*sdmax*sigma(:, m)/(floor(nx)-1);
        % yy = exp(-0.5*((xx - mu(:,m)*V)./(sigma(:,m)*V)).^2)./(sigma(:,m)*V)/sqrt(2*pi).*log(gmm1pdf_private(xx, w, mu, sigma));
        yy = bsxfun(@times, bsxfun(@rdivide, exp(-0.5*bsxfun(@power, bsxfun(@rdivide, bsxfun(@minus, xx, mu(:,m)), sigma(:,m)), 2)), ...
            sigma(:,m))/sqrt(2*pi), log(gmm1pdf_private(xx, w, mu, sigma)));
        h = h - 0.5*dx.*w(:, m).*sum(yy(:, 1:end-1) + yy(:, 2:end), 2);
    end
    
end

end


function y = gmm1pdf_private(x,w,mu,sigma)
%GMM1PDF_PRIVATE 1-d Gaussian mixture (gmm) probability density function (pdf).

y = zeros(size(x,1), size(x,2));
for m = 1:size(w, 2)
    y = y + bsxfun(@times, w(:, m)./sigma(:, m), exp(-0.5 * bsxfun(@power, bsxfun(@rdivide, (bsxfun(@minus,x,mu(:,m))), sigma(:, m)), 2)));
end
y = y / sqrt(2*pi);

end
