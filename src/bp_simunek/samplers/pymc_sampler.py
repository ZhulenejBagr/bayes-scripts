import numpy as np
import numpy.typing as npt
import pymc as pm
from scipy.stats import multivariate_normal
import src.bp_simunek.samplers.blackbox
from src.bp_simunek.samplers.idata_tools import save_idata_to_file
from arviz import InferenceData
import arviz as az

generator = np.random.default_rng(222)

def sample_regular(
        samples: int = 10000,
        n_cores: int = 4,
        n_chains: int = 4,
        tune: int = 3000,
        prior_mean: npt.NDArray = np.array([5, 3]),
        prior_cov: npt.NDArray = np.array([[4, -2], [-2, 4]]),
        step=pm.Metropolis) -> InferenceData:
    """
    Sample using a directly defined model with PYMC functions
    """
    observed = -1e-3
    sigma = 2e-4
    with pm.Model() as model:
        # reference: tabulka 3.1 v sekci 3.2
        # "f_U" v zadání, aka "prior pdf"
        U = pm.MvNormal('U', prior_mean, prior_cov)
        # "G" v zadání, aka "observation operator"
        G_mean = pm.Deterministic('G_mean', -1 / 80 * (3 / np.exp(U[0]) + 1 / np.exp(U[1])))
        # noise function
        G = pm.Normal('G', mu=G_mean, sigma=sigma, observed=observed)
        # run sampling algorithm for posterior
        idata = pm.sample(draws=samples, tune=tune, step=step(), chains=n_chains, cores=n_cores, random_seed=generator)
        # add posterior log likelyhood data
        pm.compute_log_likelihood(idata, extend_inferencedata=True)
        # add prior samples
        prior = pm.sample_prior_predictive(samples=samples*n_chains, var_names=["U"], random_seed=generator)
        idata.extend(prior)
        # add prior likelyhood
        prior_np = idata["prior"]["U"].to_numpy().reshape((-1, 2))
        factor = np.sqrt(np.linalg.det(np.multiply(2 * np.pi, prior_cov)))
        prior_likelyhood = multivariate_normal(mean=prior_mean, cov=prior_cov).pdf(prior_np)
        prior_likelyhood = np.divide(prior_likelyhood, factor)
        idata["prior"]["likelihood"] = prior_likelyhood

        return idata

def sample_blackbox(
        samples: int = 10000,
        n_cores: int = 4,
        n_chains: int = 4,
        tune: int = 3000,
        prior_mean: npt.NDArray = np.array([5, 3]),
        step=pm.Metropolis) -> InferenceData:
    """
    Sample via a blackbox function and pytensor wrapper
    """
    observed = -1e-3

    with pm.Model() as model:
        pm.CustomDist("U", prior_mean, logp=blackbox.log_probability, random=blackbox.sample, ndim_supp=1, ndims_params=(1,))
        # run sampling algorithm for posterior
        idata = pm.sample(draws=samples, tune=tune, step=step(), chains=n_chains, cores=n_cores, random_seed=generator, initvals={"U": prior_mean})
        # add posterior log likelyhood data
        pm.compute_log_likelihood(idata, extend_inferencedata=True)
        # add prior samples
        #prior = pm.sample_prior_predictive(samples=samples*n_chains, var_names=["U"], random_seed=generator)
        #idata.extend(prior)
        # add prior likelyhood
        #prior_np = idata["prior"]["U"].to_numpy().reshape((-1, 2))
        #factor = np.sqrt(np.linalg.det(np.multiply(2 * np.pi, prior_cov)))
        #prior_likelyhood = multivariate_normal(mean=prior_mean, cov=prior_cov).pdf(prior_np)
        #prior_likelyhood = np.divide(prior_likelyhood, factor)
        #idata["prior"]["likelihood"] = prior_likelyhood

        return idata

def generate_idata_sets(
        prior_mean: npt.NDArray = np.array([5, 3]),
        prior_cov: npt.NDArray = np.array([[4, -2], [-2, 4]]),
        prefix: str = "regular",
        samples: int = 10000,
        tune: int = 5000) -> None:
    methods = [pm.Metropolis, pm.NUTS, pm.DEMetropolisZ]
    method_acronyms = ["MH", "NUTS", "DEMZ"]
    for method, acronym in zip(methods, method_acronyms):
        idata = sample_regular(step=method, samples=samples, tune=tune, n_cores=4, n_chains=4, prior_mean=prior_mean, prior_cov=prior_cov)
        print(az.summary(idata), "\n\n")
        save_idata_to_file(idata, filename=f"{prefix}.{acronym}.idata")

def generate_regular_idata_sets() -> None:
    print("Generating standard data sets...")
    prior_mean = np.array([5, 3])
    prior_cov = np.array([[4, -2], [-2, 4]])
    return generate_idata_sets(prior_mean=prior_mean, prior_cov=prior_cov)

def generate_offset_idata_sets() -> None:
    prior_mean = np.array([8, 6])
    prior_cov = np.array([[16, -2], [-2, 16]])
    tune = 10000
    return generate_idata_sets(prior_mean=prior_mean, prior_cov=prior_cov, prefix="offset", tune=tune)
