%ISGMM1 True for 1-D Gaussian mixture models struct.
%   ISGMM1(A) returns true if A is a 1-D Gaussian mixture model struct and 
%   false otherwise. 
%
%   See also GMM1PDF, ISA.

%   Copyright 2015 Luigi Acerbi.
function tf = isgmm1(a)

% Check if A is a struct with all the required fields
tf = isstruct(a) & ...
    isfield(a, 'mu') & ...
    isfield(a, 'sigma') & ...
    isfield(a, 'w') & ...
    isfield(a, 'n');