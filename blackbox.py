# https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html
# https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.CustomDist.html
import pytensor as pt
import numpy as np
import numpy.random as npr
from scipy.stats import norm, multivariate_normal

generator = npr.Generator(npr.MT19937())

def sample(mu, rng=None, size=None):
    sigma = sigma=np.array([[4, -2], [-2, 4]])
    value = rng.multivariate_normal(mean=mu, cov=sigma, size=size)
    return value

def log_likelihood(value, mu):
    sigma=np.array([[4, -2], [-2, 4]])
    noise_sigma=2e-4
    observed=-1e-3
    likelihood = norm.pdf(value - observed, loc=0, scale=noise_sigma)
    # likelihood = norm.pdf(value - observed, loc=0, scale=noise_sigma) * multivariate_normal(mu, sigma).pdf(value)
    return np.log(likelihood)



class LogLike(pt.graph.Op):

    itypes = [pt.tensor.vector]
    otypes = [pt.tensor.scalar]

    def __init__(self, loglike):
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        logl = self.likelihood(theta)
        outputs[0][0] = np.array(logl)
