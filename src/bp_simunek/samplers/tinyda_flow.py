from pathlib import Path
import os
import time
import shutil
import tinyDA as tda
import scipy.stats as sps
import numpy as np
import logging
import arviz as az
import ray

from bp_simunek.simulation.measured_data import MeasuredData

from tinyDA.sampler import ray_is_available

@ray.remote
class SharedTextVariable():
    def __init__(self) -> None:
        self.text = []

    def append(self, text):
        self.text.append(text)

    def get_text(self):
        return self.text


class TinyDAFlowWrapper():
    """
    Wrapper combining a flow123 instance into a tinyDA sampler
    """

    def __init__(self, flow_wrapper):
        self.flow_wrapper = flow_wrapper
        self.observed_data = MeasuredData(self.flow_wrapper.sim._config)
        self.observed_data.initialize()
        self.worker_dirs = []
        self.observed_len = -1
        self.observe_times = []
        self.shared_text_objref = -1
        self.parallel = False

    def create_proposal_matrix(self):
        dists = [prior["dist"] for prior in self.priors]
        cov_vector = np.empty(len(dists))
        for idx, prior in enumerate(dists):
            if hasattr(prior, "std"):
                cov_vector[idx] = np.power(prior.std(), 2)
            else:
                # add support for uniform and other dists that dont have std attrib
                raise Exception("Unsupported distribution, no 'std' attribute.")
        return np.multiply(np.eye(len(cov_vector)), cov_vector)

    def sample(self, sample_count = 20, tune = 1, chains = 4) -> az.InferenceData:
        self.parallel = chains > 1 and ray_is_available
        if self.parallel:
            self.shared_text_objref = SharedTextVariable.remote()

        # setup priors from config of flow wrapper
        self.setup_priors(self.flow_wrapper.sim._config)

        # setup likelihood
        md = MeasuredData(self.flow_wrapper.sim._config)
        md.initialize()
        boreholes = ["H1"]
        cond_boreholes = []
        _, values = md.generate_measured_samples(boreholes, cond_boreholes)
        #self.setup_loglike(values, np.eye(len(values)))
        self.setup_loglike(values, np.multiply(1000, np.eye(len(values))))
        self.observed_len = len(values)

        # combine into posterior
        posterior = tda.Posterior(self.prior, self.loglike, self.forward_model)

        # setup proposal covariance matrix (for random gaussian walk & adaptive metropolis)
        proposal_cov = self.create_proposal_matrix()
        logging.info(proposal_cov)
        # setup proposal
        #proposal = tda.IndependenceSampler(self.prior)
        # TODO figure out how to use GaussianRandomWalk and AdaptiveMetropolis
        # problem - both algorithms add a sample from multivariate normal distribution
        # centered around 0 with a covariance matrix to the existing sample
        # -> result can go negative since its not a lognormal distribution, which breaks the simulation
        # ideas how to fix
        # get rid of some params - probably wont work since almost all of them are lognormal
        # parse them to positive in forward model - resulting posterior distribution will be different/wrong
        proposal = tda.GaussianRandomWalk(proposal_cov, adaptive=True, period=3, scaling=0.15)

        # sample from prior to give all chains a different starting point
        # not doing this causes all of the chains to start from the same spot
        # -> messes up the directory naming, simultaneous access to the same files
        # also adds pointless correlation and reduces coverage
        prior_values = self.prior.rvs(chains)
        if self.parallel:
            prior_values = list(prior_values)

        # sampling process
        samples = tda.sample(posterior, proposal, sample_count, chains, prior_values)

        # if parallel sampling - concat results into one list
        if self.parallel:
            job = self.shared_text_objref.get_text.remote()
            ndone = job
            while ndone:
                _, ndone = ray.wait([ndone])
            obs_times = ray.get(job)
        else:
            obs_times = self.observe_times

        # write observe times to file
        observe_times_path = os.path.join(self.flow_wrapper.sim._config["work_dir"], "observe_times.txt")
        with open(observe_times_path, "w", encoding="utf8") as file:
            file.writelines(obs_times)

        # check and save samples
        idata = tda.to_inference_data(chain=samples, parameter_names=[prior["name"] for prior in self.priors], burnin=tune)

        return idata

    def setup_priors(self, config):
        """
        Prior setup for sampling. All dists are interpreted as normal distributions
        and postprocessed into the proper distribution in the forward model.
        Additional info is saved (type of dist, name) so that forward model knows
        how to transform the individual parameters.
        """
        priors = []
        for param in config["parameters"]:
            prior_name = param["name"]
            bounds = param["bounds"]
            prior_type = param["type"]
            match prior_type:
                case "lognorm":
                    #prior = sps.lognorm(s = bounds[1], scale = np.exp(bounds[0]))
                    mu, sigma = bounds
                    prior = sps.norm(loc=mu, scale=sigma)
                    logging.info("Prior lognorm, mu=%s, std=%s", prior.mean(), prior.std())
                # unused as of now
                #case "unif":
                #    prior = sps.uniform(loc = bounds[0], scale = bounds[1] - bounds[0])
                #    logging.info("Prior uniform, a=%s, b=%s", prior.a, prior.b)
                case "truncnorm":
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
                    a_trunc, b_trunc, mu, sigma = bounds
                    a, b = (a_trunc - mu) / sigma, (b_trunc - mu) / sigma
                    prior = sps.truncnorm(a, b, loc=mu, scale=sigma)
                    logging.info("Prior truncated norm, a=%s, b=%s, mean=%s, std=%s", prior.a, prior.b, prior.mean(), prior.std())
            priors.append({
                "name": prior_name,
                "type": prior_type,
                "dist": prior
            })

        self.priors = priors
        self.prior = tda.CompositePrior([prior["dist"] for prior in priors])

    def setup_loglike(self, observed, cov):
        self.loglike = tda.GaussianLogLike(observed, cov)

    def forward_model(self, params):
        print(params)
        # reject automatically if params go negative
        #if np.any(params <= 0):
        #    logging.info("Invalid proposal, skipping...")
        #    logging.info(params)
        #    return np.zeros(self.observed_len)

        # transform parameters via info from priors
        logging.info("Input params:")
        logging.info(params)

        trans_params = []
        for param, prior in zip(params, self.priors):
            match prior["type"]:
                case "lognorm":
                    trans_param = np.exp(param)
                case "truncnorm":
                    trans_param = param

            trans_params.append(trans_param)

        logging.info("Passing params to flow")
        logging.info(trans_params)
        self.flow_wrapper.set_parameters(data_par=trans_params)

        start = time.time()
        res, data = self.flow_wrapper.get_observations()
        end = time.time()
        elapsed = end - start
        string = str(params.tolist()) + " | " + str(elapsed) + "\n"
        if self.parallel:
            job = self.shared_text_objref.append.remote(string)
            while job:
                _, job = ray.wait([job])
        else:
            self.observe_times.append(string)


        if self.flow_wrapper.sim._config["conductivity_observe_points"]:
            num = len(self.flow_wrapper.sim._config["conductivity_observe_points"])
            data = data[:-num]
        if res >= 0:
            return data
