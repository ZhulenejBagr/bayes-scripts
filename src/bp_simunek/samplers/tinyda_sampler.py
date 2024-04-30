import numpy as np
from scipy.stats import multivariate_normal, norm
import tinyDA as tda
import numpy.typing as npt

from arviz import InferenceData, summary
from bp_simunek.samplers.idata_tools import save_idata_to_file

def sample(
        samples: int = 10000,
        n_chains: int = 4,
        tune: int = 3000,
        prior_mean: npt.NDArray = np.array([5, 3]),
        prior_cov: npt.NDArray = np.array([[4, -2], [-2, 4]])) -> InferenceData:

    print(f"Sampling {samples} samples with {tune} tune per chain, {n_chains} chains\n\n")

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
            likelihood = obs_operator
            return likelihood

        observed = -1e-3
        # forward model setup
        @staticmethod
        def forward_model(params):
            CustomLikelihood.params = params
            return -1 / 80  * (3 / np.exp(params[0]) + 1 / np.exp(params[1]))

    # combine into posterior
    #prior = multivariate_normal(mean=prior_mean, cov=prior_cov)
    #forward_model = lambda params: -1 / 80  * (3 / np.exp(params[0]) + 1 / np.exp(params[1]))
    #likelihood = lambda data: tda.GaussianLogLike(data, covariance=-1e-3)
    posterior = tda.Posterior(prior, CustomLikelihood, CustomLikelihood.forward_model)

    # proposal
    proposal = tda.GaussianRandomWalk(prior.cov, scaling=0.2, adaptive=True)
    #proposal = tda.IndependenceSampler(prior)

    # sample distribution
    total_samples = tune + samples
    samples = tda.sample(posterior, proposal, iterations=total_samples, force_sequential=True, n_chains=n_chains)

    # convert to ArviZ inference data object
    idata = tda.to_inference_data(chain=samples, burnin=tune + 1, parameter_names=["U_0", "U_1"])
    

    return idata

if __name__ == "__main__":
    idata_standard = sample(tune=10000)
    idata_offset = sample(tune=10000, prior_mean=np.array([8, 6]), prior_cov=np.array([[16, -2], [-2, 16]]))
    save_idata_to_file(idata_standard, "tinyda_randomwalk.standard.idata")
    save_idata_to_file(idata_offset, "tinyda_randomwalk.offset.idata")