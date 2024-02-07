import numpy as np
from scipy.stats import multivariate_normal, norm
import tinyDA as tda
import numpy.typing as npt
from arviz import InferenceData, summary
from plotting.conductivity_plots import plot_pair_custom

def sample(
        samples: int = 10000,
        n_chains: int = 4,
        tune: int = 3000,
        prior_mean: npt.NDArray = np.array([5, 3]),
        prior_cov: npt.NDArray = np.array([[4, -2], [-2, 4]])) -> InferenceData:

    # prior setup
    prior = multivariate_normal(mean=prior_mean, cov=prior_cov)

    # likelihood setup
    class CustomLikelihood():
        noise_mean = 0
        noise_sigma = 2e-4
        noise_distribution = norm(loc=noise_mean, scale=noise_sigma)
        observed = -1e-3
        @staticmethod
        def loglike(data):
            return CustomLikelihood.noise_distribution.pdf(data)

    observed = -1e-3
    # forward model setup
    def forward_model(params):
        return -1 / 80  * (3 / np.exp(params[0]) + 1 / np.exp(params[1]))

    # combine into posterior
    posterior = tda.Posterior(prior, CustomLikelihood, forward_model)

    # proposal
    proposal = tda.GaussianRandomWalk(prior_cov, scaling=0.5, adaptive=False)

    # sample distribution
    total_samples = tune + samples
    samples = tda.sample(posterior, proposal, iterations=total_samples, force_sequential=True, n_chains=n_chains)

    # convert to ArviZ inference data object
    idata = tda.to_inference_data(chain=samples, burnin=tune + 1, parameter_names=["U_0", "U_1"])
    

    return idata


if __name__ == "__main__":
    idata = sample()
    print(summary(idata))
    explicit_data = {
        "U_0": idata["posterior"]["U_0"],
        "U_1": idata["posterior"]["U_1"],
        "log_likelihood": idata["sample_stats"]["likelihood"]
    }
    plot_pair_custom(idata, explicit_data=explicit_data, plot_prior=False)
