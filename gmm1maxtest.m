function [x w mu sigma] = gmm1maxtest(n,ndim,tol,scale,step)

if nargin < 1; n = 1000; end
if nargin < 2; ndim = 2; end
if nargin < 3; tol = 1e-6; end
if nargin < 4; scale = []; end
if isempty(scale); scale = [3 3 3]; end
if nargin < 5; step = 0.01; end

% Create first component
w = ones(1, n);
mu = zeros(1, n);
sigma = ones(1, n);

% Create random additional components
w(2:ndim, 1:n) = exp(scale(1)*(2*rand(ndim-1, n) - 1));
mu(2:ndim, 1:n) = exp(scale(2)*(2*rand(ndim-1, n)-1));
sigma(2:ndim, 1:n) = exp(scale(3)*rand(ndim-1, n));

options = optimset('TolX',1e-14,'Display','off');

x = zeros(4, n);

tic;
for i = 1:n
    x(1, i) = gmm1max(w(:, i), mu(:, i), sigma(:, i), tol, 1);
end
toc;

if ndim == 2
    tic;
    for i = 1:n
        x(2, i) = gmm1max_n2(w(:, i), mu(:, i), sigma(:, i), tol, 1);
    end
    toc;
end

tic;
for i = 1:n      
    x(3, i) = gmm1max_grid(w(:, i), mu(:, i), sigma(:, i), step, tol, 1);
end
toc;

tic;
for i = 1:n
   x(4, i) = gmm1max(w(:, i), mu(:, i), sigma(:, i), tol, 1, 'guess', step);
end
toc;


% tic;
% for i = 1:n
%    x(2, i) = gmm1max2(w(:, i), mu(:, i), sigma(:, i), tol, 1);
% end
% toc;

% tic;
% for i = 1:n
%    x(2, i) = fminbnd(@(xx) -gmm1updf(xx, w(:, i), mu(:, i), sigma(:, i)), ...
%         min(mu(:, i))-0.01, max(mu(:, i))+0.01, options);
% end
% toc;

end

function y = gmm1updf(x,w,mu,sigma)
%GMM1UPDF Gaussian mixture unnormalized probability density function (pdf).
    y = zeros(1, length(x));
    for i = 1:length(w)
        y = y + w(i)/sigma(i)*exp(-0.5*(((x - mu(i))/sigma(i)).^2));
    end
end