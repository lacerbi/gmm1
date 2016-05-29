## gmm1 - 1-D Gaussian mixture model toolbox for MATLAB

A toolbox for computing with 1-D Gaussian mixture models (gmm1).
This code is generally fast but there is space for further improvement (e.g., improved vectorization).

#### Contents:

- `gmm1cdf.m`: gmm1 cumulative distribution function (cdf)
- `gmm1ent.m`: gmm1 differential entropy (numerically estimated)
- `gmm1max.m`: Find the global maximum (mode) of gmm1
- `gmm1max_n2.m`: Find the global maximum (mode) of gmm1 with 2 components (faster than  `gmm1max.m`)
- `gmm1moments.m`: Central moments of gmm1 (mean, variance, skewness, excess kurtosis)
- `gmm1pdf.m`: gmm1 probability density function (pdf)
- `gmm1prod.m`: Product of two gmm1
- `gmm1rnd.m`: Random draw from gmm1 (not optimal, needs recoding)
- `isgmm1.m`: Returns true for a gmm1 struct

#### References:

This toolbox was created for and extensively used in the following publications (please consider citing them if you use this toolbox):

- Acerbi, L., Vijayakumar, S. & Wolpert, D. M. (2014). On the Origins of Suboptimality in Human Probabilistic Inference, *PLoS Computational Biology* 10(6): e1003661.
- Acerbi, L., Ma, W. J. & Vijayakumar, S. (2014). A Framework for Testing Identifiability of Bayesian Models of Perception, *Proc. Advances in Neural Information Processing Systems (NIPS ’14)*, Montreal, Canada.
