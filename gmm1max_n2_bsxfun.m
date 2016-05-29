function [x pval] = gmm1max_n2_fast(w,mu,sigma,niter)
%GMM1MAX_N2 finds the global maximum of 1-d Gaussian mixture model with 2 components.
%
%   X = GMM1MAX_N2(W,MU,SIGMA) returns the location of the global maximum
%   of the 1-dimensional gaussian mixture model (gmm) with 2 components
%   described by mixing weights W, means MU and standard deviations SIGMA, 
%   using a fixed-point search.
%   W, MU and SIGMA can either be vectors of length 2 or scalars. A scalar 
%   input parameter is converted to a vector of length 2. W and SIGMA must 
%   be positive-valued. Default values for MU and SIGMA are 0 and 1 respectively.
%
%   X = GMM1MAX_N2(W,MU,SIGMA,TOL) uses a relative error tolerance of TOL
%   instead of the default, which is 1.e-6. The error tolerance represents
%   an approximate upper bound on the scale of the typical error in the
%   location of the returned maximum, relative to the smallest standard 
%   deviation in the mixture.
%
%   X = GMM1MAX_N2(W,MU,SIGMA,TOL,1) skips the function preprocessing and 
%   input arguments error-checking. Here, MU and SIGMA are assumed to be 
%   vectors of the same size as W and both W and SIGMA have only positive 
%   values. This speeds up computation for intensive computing. 
%   GMM1MAX(W,MU,SIGMA,TOL,0) is the same as GMM1MAX(W,MU,SIGMA,TOL).
%
%   [X,PVAL]= GMM1MAX_N2(...) returns the (unnormalized) gmm probability 
%   density function (pdf) evaluated at the values in X. The pdf is 
%   normalized if the vector of component weights W sums to 1.
%
%   Notes:
%   - The maximized gmm (unnormalized) pdf has shape: 
%       gmm(x) = W(1).*normpdf(x, MU(1), SIGMA(1)) + W(2).*normpdf(x, MU(2), SIGMA(2))
%     The weight vector W does not have to sum to one.
%   - GMM1MAX_N2 implements the fixed-point search algorithm described in 
%     [1], with additional tweaks that speed up computations for gmm with 2 
%     components [3].
%   - If the the gmm has a unique global maximum, GMM1MAX_N2 returns it. 
%     If the gmm has two global maxima (i.e. identical in function value), 
%     GMM1MAX_N2 returns one of them.
%   - GMM1MAX_N2 may stop too early if the global maximum is contained in a 
%     plateau (a zone where the derivative is almost null). Changing the
%     value of TOL may help.
%   - GMM1MAX_N2 is faster than GMM1MAX for gmm with 2 components
%     (the algorithm is identical but the code is optimized).
%
%   References:
%   [1] Carreira-Perpiñán, M.Á., "Mode-finding for mixtures of Gaussian
%   distributions," IEEE Trans. on Pattern Anal. and Machine Intel., 
%   Vol. 22, No. 11, pp. 1318-1323, 2000.
%   [2] Behboodian, J., "On the modes of a mixture of two normal 
%   distributions", Technometrics, Vol. 12, No. 1, pp. 131-139, 1970.
%   [3] Acerbi, L., Vijayakumar, S. & Wolpert, D. M., "On the Origins of 
%   Suboptimality in Human Probabilistic Inference", PLoS Computational 
%   Biology 10(6): e1003661, 2014. 
%
%   See also GMM1CDF, GMM1MAX, GMM1PDF, NORMPDF.
%
%   Copyright (c) by Luigi Acerbi, September 2014

% The mixture might be bimodal, check starting from both components    

if nargin < 4 || isempty(niter); niter = 10; end

nrep = 3;
n = size(w, 1);
xhat = zeros(nrep*n, 1);
xmask = zeros(n, nrep);
for kk = 1:nrep; xmask(:, kk) = (1:n) + n*(kk-1); end
wisigma3 = w./(sigma.^3);
isigma2 = -0.5./(sigma.^2);

xzero = [mu(:, 1), mu(:, 2), w(:, 1).*mu(:, 1) + w(:, 2).*mu(:, 2)];

for kk = 1:nrep
    x = xzero(:, kk);                            
    for mm = 1:niter
        z = wisigma3.*exp(isigma2.*(x*ones(1, 2) - mu).^2);
        x = sum(mu.*z, 2)./sum(z, 2);
    end
    xhat(xmask(:, kk)) = x;
end

% Find a global maximum
y = zeros(n, nrep);
y(:, 1) = w(:, 1)./sigma(:, 1).*exp(isigma2(:, 1) .* (xhat(xmask(:, 1))-mu(:, 1)).^2) + ... 
    w(:, 2)./sigma(:, 2).*exp(isigma2(:, 2) .* (xhat(xmask(:, 1))-mu(:, 2)).^2);
y(:, 2) = w(:, 1)./sigma(:, 1).*exp(isigma2(:, 1) .* (xhat(xmask(:, 2))-mu(:, 1)).^2) + ... 
    w(:, 2)./sigma(:, 2).*exp(isigma2(:, 2) .* (xhat(xmask(:, 2))-mu(:, 2)).^2);
y(:, 3) = w(:, 1)./sigma(:, 1).*exp(isigma2(:, 1) .* (xhat(xmask(:, 3))-mu(:, 1)).^2) + ... 
    w(:, 2)./sigma(:, 2).*exp(isigma2(:, 2) .* (xhat(xmask(:, 3))-mu(:, 2)).^2);

x = xhat(xmask(:, 1) + n*(y(:, 2) > y(:, 1) & y(:, 2) > y(:, 3)) + 2*n*(y(:, 3) > y(:, 1) & y(:, 3) > y(:, 2)));
    
%if nargout > 1
%   pval = 1/sqrt(2*pi) * (w(1)/sigma(1)*exp(-0.5 * ((x - mu(1))/sigma(1)).^2) ... 
%       + w(2)/sigma(2)*exp(-0.5 * ((x - mu(2))/sigma(2)).^2));   
%end

end