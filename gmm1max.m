function [x pval] = gmm1max(w,mu,sigma,tol,flag,x0,step)
%GMM1MAX finds the global maximum of 1-d Gaussian mixture model (gmm).
%
%   X = GMM1MAX(W,MU,SIGMA) returns the location of the global maximum of 
%   the 1-dimensional gaussian mixture model (gmm) with mixing weights W, 
%   means MU and standard deviations SIGMA, using a fixed-point search.
%   W, MU and SIGMA can either be vectors of the same size or scalars. A 
%   scalar input parameter functions as a constant vector of the same size 
%   as the other input parameters. W and SIGMA must be positive-valued.
%   Default values for MU and SIGMA are 0 and 1 respectively.
%
%   X = GMM1MAX(W,MU,SIGMA,TOL) uses a relative error tolerance of TOL
%   instead of the default, which is 1.e-6. The error tolerance represents
%   an approximate upper bound on the scale of the typical error in the
%   location of the returned maximum, relative to the smallest standard 
%   deviation in the mixture. If TOL is negative, the algorithm uses an 
%   absolute (not relative) error tolerante of abs(TOL).
%
%   X = GMM1MAX(W,MU,SIGMA,TOL,1) skips the function preprocessing and 
%   input arguments error-checking. Here, MU and SIGMA are assumed to be 
%   vectors of the same size as W and both W and SIGMA have only positive 
%   values. This speeds up computation by a bit for intensive computing. 
%   GMM1MAX(W,MU,SIGMA,TOL,0) is the same as GMM1MAX(W,MU,SIGMA,TOL).
%
%   X = GMM1MAX(W,MU,SIGMA,TOL,FLAG,X0) uses the points in vector X0 as 
%   starting points for the fixed-point search. The algorithm runs once for 
%   each starting point, and the maximum is returned. If X0 is empty or not 
%   specified, the default starting points correspond to the component 
%   means MU.
%
%   X = GMM1MAX(W,MU,SIGMA,TOL,FLAG,'guess') attempts to find the global 
%   maximum by guessing the starting position of the search. For n > 2 
%   components this is typically faster than the default search and the 
%   performance gain increases for larger number of components. However, 
%   this method is not guaranteed to converge always to the global maximum;
%   it may occasionally report a local maximum which is close in function 
%   value to the global maximum.
%
%   The guess is computed with brute force, estimating the value of the gmm
%   in the region where the mode can be (between the minimum and the
%   maximum of the component means [1]). The candidate region is sampled
%   with a regular grid of step size dx = 0.1*min([SIGMA, max(MU) - min(MU)]).
%
%   X = GMM1MAX(W,MU,SIGMA,TOL,FLAG,'guess',STEP) uses a relative step size 
%   STEP when sampling the candidate region. The default value is 0.1. 
%   Smaller values may reduce the risk of reporting a local maximum at 
%   a cost of longer computation time.
%
%   X = GMM1MAX(W,MU,SIGMA,TOL,FLAG,'all') instead of returning just the
%   global maximum, returns a vector whose values are all distinct modes 
%   (all local maxima) of the probability density function. Modes are 
%   merged if they are closer than 100 times the given precision TOL.
%
%   [X,PVAL]= GMM1MAX(...) returns the (unnormalized) gmm probability 
%   density function (pdf) evaluated at the values in X. The pdf is 
%   normalized if the vector of component weights W sums to 1.
%
%   Notes:
%   - The maximized gmm (unnormalized) pdf has shape: 
%       gmm(x) = sum(W.*normpdf(x, MU, SIGMA))
%     The weight vector W does not have to sum to one.
%   - GMM1MAX implements the fixed-point search algorithm described in [1], 
%     with additional tweaks that speed up computations, in particular for 
%     gmm with two components.
%   - If the the gmm has a unique global maximum, GMM1MAX returns it. 
%     If the gmm has multiple global maxima (i.e. identical in function 
%     value), GMM1MAX returns one of them.
%     (Unless the option 'all' is used, in which case all global and local
%     maxima are returned.)
%   - Extensive testing has shown that the fixed-point search is guaranteed 
%     to find all the modes (hence also the global maximum) if the vector 
%     of starting points, X0, is initialized to the component means vector 
%     MU (the default option). [1]
%   - GMM1MAX may stop too early if the global maximum is contained in a 
%     plateau (a zone where the derivative is almost null). Changing the
%     value of TOL may help.
%   - For gmm with two components, it is faster to call directly GMM1MAX_N2.
%
%   References:
%   [1] Carreira-Perpiñán, M.Á., "Mode-finding for mixtures of Gaussian
%   distributions," IEEE Trans. on Pattern Anal. and Machine Intel., 
%   Vol. 22, No. 11, pp. 1318-1323, 2000.
%   [2] Behboodian, J., "On the modes of a mixture of two normal 
%   distributions", Technometrics, Vol. 12, No. 1, pp. 131-139, 1970.
%
%   See also GMM1CDF, GMM1MAX_N2, GMM1PDF, NORMPDF.
%
%   Copyright (c) by Luigi Acerbi, March 2013

if nargin < 4; tol = 1e-6; end
if nargin < 5; flag = 0; end
if nargin < 6; x0 = []; end

% Use absolute tolerance
if tol < 0; tol = abs(tol); abstol = 1; else abstol = 0; end

if ~flag
    if nargin < 1
        error('gmm1max:TooFewInputs','Input argument W is undefined.');
    end
    if nargin < 2; mu = 0; end
    if nargin < 3; sigma = 1; end

    % Return NaN for out of range parameters.
    if any(w <= 0) || any(sigma <= 0)
        x = NaN;
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
end

% Number of components
if size(w,1) > size(w,2); n = size(w,1); else n = size(w,2); end

% If the mixture has only one component, the mode is the mean
if n == 1
    x = mu;
    if nargout > 1; pval = 1/sqrt(2*pi)/sigma; end
    return; 
end

% Make a guess for the starting position (faster for n > 2 but risky)
if ischar(x0)
    if x0(1) == 'g' || x0(1) == 'G'
        if nargin < 7; step = 0.1; end

        isigma = 1./sigma; wisigma = w./sigma;
        maxmu = max(mu); minmu = min(mu);
        dx = step*min([min(sigma), maxmu-minmu]);
        xx = [(minmu-dx):dx:(maxmu+dx), maxmu+dx];
        y = zeros(1, size(xx, 2));
        for m = 1:n
            y = y + wisigma(m)*exp(-0.5 * ((xx - mu(m))*isigma(m)).^2);
        end
        [~, index] = max(y);    
        x0 = xx(index);
        n0 = 1;
        allflag = 0;
    else
        x0 = []; 
        n0 = 0;
        allflag = 1;
    end
else
    % Number of starting points
    if size(x0,1) > size(x0,2); n0 = size(x0,1); else n0 = size(x0,2); end
    allflag = 0;
end

% If the mixture has two components, use ad hoc algorithm (much faster).
% This part of the code is identical to the algorithm in function GMM1MAX_N2.
% (GMM1MAX_N2 is not called externally to avoid overhead.)
if n == 2 && n0 <= 2 
    if sigma(1)<sigma(2); minsigma = sigma(1); else minsigma = sigma(2); end
    if abstol; toldelta = tol; else toldelta = tol*minsigma; end
    wisigma3 = [w(1)/(sigma(1)^3) w(2)/(sigma(2)^3)];
    
    % The mixture is unimodal (sufficient condition, see [2]) or one
    % starting point has been specified.
    if (abs(mu(2)-mu(1)) <= 2*minsigma) || (n0 == 1)
        if isempty(x0)
            x = (w(1)*mu(1)/sigma(1) + w(2)*mu(2)/sigma(2))/(w(1)/sigma(1) + w(2)/sigma(2));
        else
            x = x0;
        end
        xold = Inf;
        while abs(xold - x) > toldelta
            xold = x;
            z = [wisigma3(1)*exp(-0.5*((x - mu(1))/sigma(1))^2) wisigma3(2)*exp(-0.5*((x - mu(2))/sigma(2))^2)];
            x = (mu(1)*z(1) + mu(2)*z(2))/(z(1) + z(2));
        end
        xhat = x;
        
    % The mixture might be bimodal, or two starting points have been
    % specified.
    else
        if isempty(x0); x0 = mu; n0 = 2; end
        
        xhat = zeros(1, n0);
        for m = 1:n0
            xold = Inf;
            x = x0(m);
            while abs(xold - x) > toldelta
                xold = x;
                z = [wisigma3(1)*exp(-0.5*((x - mu(1))/sigma(1))^2) wisigma3(2)*exp(-0.5*((x - mu(2))/sigma(2))^2)];
                x = (mu(1)*z(1) + mu(2)*z(2))/(z(1) + z(2));
            end
            xhat(m) = x;
        end
    end
    
% If the mixture has n > 2 components, check convergence with derivative 
% (slightly slower but reduces error significantly)
else
    if isempty(x0); x0 = mu; n0 = n; end
    minsigma = min(sigma);
    
    if abstol; toldelta = tol/minsigma; else toldelta = tol/minsigma^2; end
    xhat = zeros(1, n0);
    wisigma3 = w./(sigma.^3);
    
    for m = 1:n0
        g = Inf;
        x = x0(m);
        z = wisigma3.*exp(-0.5*((x - mu)./sigma).^2);
        while abs(g) > toldelta
            x = sum(mu.*z)/sum(z);
            z = wisigma3.*exp(-0.5*((x - mu)./sigma).^2);
            g = sum((mu - x).*z);
        end
        xhat(m) = x;
    end
end

if ~allflag
    % Pick global maximum
    if n0 == 2
        y = w(1)/sigma(1)*exp(-0.5*((xhat-mu(1))/sigma(1)).^2) + ... 
            w(2)/sigma(2)*exp(-0.5*((xhat-mu(2))/sigma(2)).^2);
        if y(1) > y(2); x = xhat(1); else x = xhat(2); end
    elseif n0 > 2
        y = zeros(1, n0);
        for m = 1:n0
            y = y + w(m)/sigma(m)*exp(-0.5*((xhat-mu(m))/sigma(m)).^2);
        end
        [~, index] = max(y);
        x = xhat(index);                   
    end
else
    x = xhat;
end

% Compute pdf at the modes
if nargout > 1 || allflag
    pval = zeros(size(x,1), size(x,2));
    for m = 1:n
        pval = pval + w(m)/sigma(m)*exp(-0.5 * ((x - mu(m)/sigma(m)).^2));
    end
    pval = pval/sqrt(2*pi);
end

% Prune modes set
if allflag && n0 > 1
    % Merge modes if they are closer than mindiff
    if abstol; mindiff = tol*100; else mindiff = 100*tol*minsigma; end
    pold = pval;
    x = []; pval = [];
    for m = 1:(n0-1)
        if ~isnan(xhat(m))
            xset = abs(xhat - xhat(m)) < mindiff;
            xmodes = xhat(xset);
            [pmax, index] = max(pold(xset));
            x = [x, xmodes(index)];
            pval = [pval, pmax];
            xhat(xset) = NaN(size(xhat(xset)));
            pold(xset) = zeros(size(pold(xset)));        
        end
    end
    if ~isnan(xhat(end))
        x = [x xhat(end)];
        pval = [pval pold(end)];
    end
end

end