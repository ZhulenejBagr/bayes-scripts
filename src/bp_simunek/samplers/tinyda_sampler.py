import numpy as np
from scipy.stats import multivariate_normal, norm
import tinyDA as tda
import numpy.typing as npt
from arviz import InferenceData, summary
from src.bp_simunek.plotting.conductivity_plots import plot_pair_custom

def sample(
        samples: int = 10000,
        n_chains: int = 4,
        tune: int = 3000,
        prior_mean: npt.NDArray = np.array([5, 3]),
        prior_cov: npt.NDArray = np.array([[4, -2], [-2, 4]])) -> InferenceData:

    print(f"Sampling {samples} samples with {tune} tune per chain, {n_chains} chains")

    # prior setup
    prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

    # likelihood setup
    class CustomLikelihood():
        noise_mean = 0
        noise_sigma = 2e-4
        noise_distribution = norm(loc=noise_mean, scale=noise_sigma)
        observed = -1e-3
        params = [0, 0]
        @staticmethod
        def loglike(data):
            obs_operator = CustomLikelihood.noise_distribution.logpdf(data - CustomLikelihood.observed)
            prior_likelihood = prior.logpdf(CustomLikelihood.params)
            likelihood = obs_operator + prior_likelihood
            return likelihood

        observed = -1e-3
        # forward model setup
        @staticmethod
        def forward_model(params):
            CustomLikelihood.params = params
            return -1 / 80  * (3 / np.exp(params[0]) + 1 / np.exp(params[1]))

    # combine into posterior
    posterior = tda.Posterior(prior, CustomLikelihood, CustomLikelihood.forward_model)

    # proposal
    #proposal = tda.GaussianRandomWalk(np.eye(2, 2), scaling=1, adaptive=False)
    proposal = tda.IndependenceSampler(prior)

    # sample distribution
    total_samples = tune + samples
    samples = tda.sample(posterior, proposal, iterations=total_samples, force_sequential=True, n_chains=n_chains)

    # convert to ArviZ inference data object
    idata = tda.to_inference_data(chain=samples, burnin=tune + 1, parameter_names=["U_0", "U_1"])
    

    return idata
