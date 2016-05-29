function [x pval] = gmm1max_grid(w,mu,sigma,step,tol,flag)
%GMM1MAX_GRID brute-force search of global maximum of 1-d Gaussian mixture model.
%
%   X = GMM1MAX_GRID(W,MU,SIGMA) returns the location of the global maximum 
%   of the 1-dimensional gaussian mixture model (gmm) with mixing weights W, 
%   means MU and standard deviations SIGMA, using a brute-force search. 
%   W, MU and SIGMA can either be vectors of the same size or scalars. A 
%   scalar input parameter functions as a constant vector of the same size 
%   as the other input parameters. W and SIGMA must be positive-valued. 
%   Default values for MU and SIGMA are 0 and 1 respectively.
%   Note that GMM1MAX_GRID may fail to return the global maximum and return
%   only a local maximum instead.
%
%   X = GMM1MAX_GRID(W,MU,SIGMA,STEP) in the brute-force search uses a 
%   relative grid size of size STEP instead of the default, which is 0.05
%   (see below). The value of STEP strongly influences the probability that 
%   the algorithm returns the correct global maximum. Smaller grid sizes 
%   correspond to lower chances of making mistakes, at the expense of (much) 
%   greater computational cost.
%
%   The brute-force search algorithm is initialized by choosing as candidate 
%   region for the location of the maximum the region between the minimum 
%   and the maximum of the component means [1]. The candidate region is then 
%   sampled with a regular grid of initial step size dx = STEP*min(SIGMA).
%   At each iteration of the algorithm, a new candidate region is extracted
%   by taking a window around the location of the maximum of the gmm
%   evaluated at the previous grid. The step size is reduced accordingly 
%   to dx(i+1) = dx(i)*STEP. The process stops when the grid step is below 
%   the desired accuracy (1.e-6 by default)
%
%   X = GMM1MAX_GRID(W,MU,SIGMA,STEP,TOL) uses a relative error tolerance 
%   of TOL instead of the default, which is 1.e-6. The error tolerance 
%   represents the typical error in the location of the returned maximum, 
%   relative to the smallest standard deviation in the mixture, *if* the
%   search algorithm has identified the correct maximum. Otherwise, the
%   error can be arbitrarily large.
%
%   X = GMM1MAX_GRID(W,MU,SIGMA,STEP,TOL,1) skips the function preprocessing 
%   and input arguments error-checking. Here, MU and SIGMA are assumed to be 
%   vectors of the same size as W, both W and SIGMA have only positive 
%   values, and STEP must be lesser than 1. This speeds up computation by a 
%   bit for intensive computing. GMM1MAX(W,MU,SIGMA,STEP,TOL,0) is the same 
%   as GMM1MAX(W,MU,SIGMA,STEP,TOL).
%
%   Notes:
%   - The maximized gmm (unnormalized) pdf has shape: 
%       gmm(x) = sum(W.*normpdf(x, MU, SIGMA))
%     The weight vector W does not have to sum to one.
%   - If GMM1MAX_GRID fails to find the correct maxima in your problem, try
%     and use a smaller value of STEP. (You can double-check the
%     correctness of the results for typical instances of your problem by
%     using GMM1MAX, which is guarranteed to return the global maximum.)
%   - If the the gmm has a unique global maximum, GMM1MAX_GRID typically 
%     returns it. If the gmm has multiple global maxima (i.e. identical in 
%     function value), GMM1MAX_GRID returns one of them.
%   - In general GMM1MAX_GRID is much faster than GMM1MAX for n>2 mixture 
%     components. For n=2, GMM1MAX (or GMM1MAX_N2) is faster.
%     GMM1MAX_GRID may or may not be faster than GMM1MAX run with the
%     'guess' flag on.
%
%   See also GMM1CDF, GMM1MAX, GMM1MAX_N2, GMM1PDF, NORMPDF.
%
%   Copyright (c) by Luigi Acerbi, March 2013

if nargin < 4; step = 0.05; end
if nargin < 5; tol = 1e-6; end
if nargin < 6; flag = 0; end

if ~flag
    if nargin < 1
        error('gmm1max_grid:TooFewInputs','Input argument W is undefined.');
    end
    if nargin < 2; mu = 0; end
    if nargin < 3; sigma = 1; end
    
    % Return NaN for out of range parameters.
    if any(w <= 0) || any(sigma <= 0)
        x = NaN;
        return;
    end
   
    % Check validity of step size
    if (~isscalar(step) || step > 0.5 || step <= 0)
        error('gmm1max_grid:StepSize','Input argument SIZE must be a positive number much smaller than one (maximum accepted value is SIZE=0.5).');
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


isigma = 1./sigma;
wisigma = w./sigma;
minsigma = min(sigma);
maxmu = max(mu);
minmu = min(mu);

dx = step*minsigma;
% dx = step*min([minsigma, maxmu-minmu]);
xx = [minmu:dx:maxmu, maxmu];

y = zeros(1, size(xx, 2));
for m = 1:n
    y = y + wisigma(m)*exp(-0.5 * ((xx - mu(m))*isigma(m)).^2);
end

while dx > tol
    dxold = dx;
    dx = dx*step;
    [~, index] = max(y);
    xx = [(xx(index)-dxold+dx):dx:(xx(index)+dxold-dx), (xx(index)+dxold-dx)];    
    y = zeros(1, size(xx, 2));
    for m = 1:n
        y = y + wisigma(m)*exp(-0.5 * ((xx - mu(m))*isigma(m)).^2);
    end
end

[~, index] = max(y);
x = xx(index);

if nargout > 1
    pval = zeros(size(x,1), size(x,2));
    for m = 1:n
        pval = pval + wisigma(m)*exp(-0.5 * ((x - mu(m))/sigma(m)).^2);
    end
    pval = pval/sqrt(2*pi);
end

end