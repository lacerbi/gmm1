function [x pval] = gmm1max_n2(w,mu,sigma,tol,flag)
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

if nargin < 4; tol = 1e-6; end
if nargin < 5; flag = 0; end

if ~flag
    if nargin < 1
        error('gmm1max_n2:TooFewInputs','Input argument W is undefined.');
    end
    if nargin < 2; mu = 0; end
    if nargin < 3; sigma = 1; end
        
    % Convert scalar input to vectors
    if isscalar(w); w = w*[1 1]; end
    if isscalar(mu); mu = mu*[1 1]; end
    if isscalar(sigma); sigma = sigma*[1 1]; end

    % Return NaN for out of range parameters.
    if (w(1) <= 0) || (w(2) <= 0) || (sigma(1) <= 0) || (sigma(2) <= 0)
        x = NaN;
        return;
    end
end

% Use ad hoc algorithm for two components (much faster)
if sigma(1)<sigma(2); minsigma = sigma(1); else minsigma = sigma(2); end
toldelta = tol*minsigma;
wisigma3 = [w(1)/(sigma(1)^3) w(2)/(sigma(2)^3)];

% The mixture is unimodal (sufficient condition, see [2])
if (abs(mu(2)-mu(1)) <= 2*minsigma)    
    x = (w(1)*mu(1)/sigma(1) + w(2)*mu(2)/sigma(2))/(w(1)/sigma(1) + w(2)/sigma(2));
    xold = Inf;
    while abs(xold - x) > toldelta
        xold = x;
        z = [wisigma3(1)*exp(-0.5*((x - mu(1))/sigma(1))^2) wisigma3(2)*exp(-0.5*((x - mu(2))/sigma(2))^2)];
        x = (mu(1)*z(1) + mu(2)*z(2))/(z(1) + z(2));
    end

% The mixture might be bimodal, check starting from both components    
else
    xhat = [0 0];
    for m = 1:2
        xold = Inf;
        x = mu(m);
        while abs(xold - x) > toldelta
            xold = x;
            z = [wisigma3(1)*exp(-0.5*((x - mu(1))/sigma(1))^2) wisigma3(2)*exp(-0.5*((x - mu(2))/sigma(2))^2)];
            x = (mu(1)*z(1) + mu(2)*z(2))/(z(1) + z(2));
        end
        xhat(m) = x;
    end

    % Find a global maximum
    y = w(1)/sigma(1)*exp(-0.5*((xhat-mu(1))/sigma(1)).^2) + ... 
        w(2)/sigma(2)*exp(-0.5*((xhat-mu(2))/sigma(2)).^2);
    if y(1) > y(2); x = xhat(1); else x = xhat(2); end
end

if nargout > 1
   pval = 1/sqrt(2*pi) * (w(1)/sigma(1)*exp(-0.5 * ((x - mu(1))/sigma(1)).^2) ... 
       + w(2)/sigma(2)*exp(-0.5 * ((x - mu(2))/sigma(2)).^2));   
end

end