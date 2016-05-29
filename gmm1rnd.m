function r = gmm1rnd(w,mu,sigma)
%NORMRND Random arrays from 1-d Gaussian mixture models (gmm).
%   R = GMM1RND(W,MU,SIGMA) returns an array of random numbers chosen from 
%   a 1-d Gaussian mixture model (gmm) with mixing weights W, means MU and
%   standard deviations SIGMA. W, MU and SIGMA are supposed to be either
%   row arrays or matrices where each row represents a different gmm. 
%   The size of R is the number of rows of W. If either MU or SIGMA are 
%   scalar, they are assumed to be the same length as W.
%
%   See also GMM1CDF, GMM1PDF, RAND, RANDN.

r = zeros(size(w, 1), 1);

if isscalar(mu); mu = mu*ones(size(w, 1), size(w, 2)); end
if isscalar(sigma); sigma = sigma*ones(size(w, 1), size(w, 2)); end

% Do not assume W is normalized
t = rand(size(w,1), 1).*sum(w, 2);
r = randn(size(w,1),1);

for i = 1:size(w, 1)
    c = find(cumsum(w(i, :)) >= t(i), 1);
    r(i) = r(i)*sigma(i, c) + mu(i, c);
end

end