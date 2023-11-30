# https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html
# https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.CustomDist.html
import numpy as np
import pytensor as pt
from scipy.stats import norm, multivariate_normal

def sample(mu, rng=None, size=None):
    sigma = sigma=np.array([[4, -2], [-2, 4]])
    value = rng.multivariate_normal(mean=mu, cov=sigma, size=size)
    return value

def log_likelihood(value, mu):
    sigma = np.array([[4, -2], [-2, 4]])
    noise_sigma = 2e-4
    observed = -1e-3
    #operator = np.multiply(-1 / 80, np.add(3 / np.exp(value[0]),  1 / np.exp(value[1])))
    operator = -1 / 80 * (3 / np.exp(value[0]) + 1 / np.exp(value[1]))
    v = np.subtract(operator, observed)
    # likelihood = norm.pdf(v, loc=0, scale=noise_sigma)
    likelihood = 1 / (noise_sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * np.power(v / noise_sigma, 2))
    #likelihood = likelihood * multivariate_normal(mu, sigma).pdf(value)
    return np.log(likelihood)

