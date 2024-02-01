import numpy as np
from scipy.stats import multivariate_normal, norm
import tinyDA as tda

# prior setup
mean_prior = np.array([7, 5])
cov_prior = np.array([[4, -2], [-2, 4]])
prior = multivariate_normal(mean=mean_prior, cov=cov_prior)

# likelihood setup
class custom_likelihood():
    noise_mean = 0
    noise_sigma = 2e-4
    noise_distribution = norm(loc=noise_mean, scale=noise_sigma)
    @staticmethod
    def loglike(data):
        return custom_likelihood.noise_distribution.pdf(data)
    
# forward model setup
def forward_model(params):
    return 1 / 80  * (3 / np.exp(params[0] + 1 / np.exp(params[1])))

# combine into posterior
posterior = tda.Posterior(prior, custom_likelihood, forward_model)

# proposal
proposal = tda.GaussianRandomWalk(cov_prior, scaling=0.5, adaptive=False)

# sample distribution
n_tune = 2000
n_samples = 5000
total_samples = n_tune + n_samples
samples = tda.sample(posterior, proposal, iterations=total_samples, n_chains=4, force_sequential=True)

# convert to ArviZ inference data object
idata = tda.to_inference_data(samples, n_tune)