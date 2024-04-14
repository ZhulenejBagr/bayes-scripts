import os
import time
import traceback
import logging

import numpy as np
import scipy.stats as sps
import arviz as az
import ray
import tinyDA as tda
from tinyDA.sampler import ray_is_available

from bp_simunek.simulation.measured_data import MeasuredData


NUMBER_OF_CHAINS_DEFAULT = 1
FORCE_SEQUENTIAL_DEFAULT = False
PROPOSAL_SCALING_DEFAULT = 0.2
GAMMA_DEFAULT = 1.01
ADAPTIVITY_DEFAULT = False
ADAPTIVITY_PERIOD_DEFAULT = 10
NOISE_STD_DEFAULT = 20
IS_PARALLEL_DEFAULT = False
SAMPLE_COUNT_DEFAULT = 10

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
        # flow wrapper
        self.flow_wrapper = flow_wrapper
        # measured data loader
        self.observed_data = MeasuredData(self.flow_wrapper.sim._config)
        self.observed_data.initialize()
        # better reference to config object
        self.config = self.flow_wrapper.sim._config
        # length of measured data
        self.measured_len = -1
        # time and parameters info of simulations
        self.observe_times = []
        # reference to shared object for logging
        self.shared_text_objref = -1
        # check for sampler config or set default params
        sampler_config_key = "sampler_parameters"
        if sampler_config_key not in self.config:
            logging.warning("Missing sampler parameters, using default values for all")
            self.set_default_sampler_params()
        else:
            try:
                self.load_sampler_params(self.config[sampler_config_key])
            except Exception:
                logging.error("Failed to load sampler params from config, using default values for all")
                logging.error(traceback.format_exc())
                self.set_default_sampler_params()

    def set_default_sampler_params(self):
        self.is_parallel = IS_PARALLEL_DEFAULT
        self.number_of_chains = NUMBER_OF_CHAINS_DEFAULT
        self.force_sequential = FORCE_SEQUENTIAL_DEFAULT
        self.proposal_scaling = PROPOSAL_SCALING_DEFAULT
        self.gamma = GAMMA_DEFAULT
        self.adaptive = ADAPTIVITY_DEFAULT
        self.adaptivity_period = ADAPTIVITY_PERIOD_DEFAULT
        self.noise_std = NOISE_STD_DEFAULT
        self.sample_count = SAMPLE_COUNT_DEFAULT

    def load_sampler_params(self, params):
        # specify number of chains
        chains_key = "chain_count"
        if chains_key not in params:
            logging.warning("Missing number of chains, defaulting to %d", NUMBER_OF_CHAINS_DEFAULT)
            self.number_of_chains = NUMBER_OF_CHAINS_DEFAULT
        else:
            # TODO check if param is actually a valid number, not a string etc.
            self.number_of_chains = params[chains_key]

        # specify whether to force sequential sampling
        force_seq_key = "force_sequential"
        if force_seq_key not in params:
            logging.warning("Force sequential not specified, defaulting to %s", str(FORCE_SEQUENTIAL_DEFAULT))
            self.force_sequential = FORCE_SEQUENTIAL_DEFAULT
        else:
            # TODO check if value is boolean
            self.force_sequential = params[force_seq_key]

        # check if proposal params are specified
        proposal_scaling_key = "proposal_scaling"
        if proposal_scaling_key not in params:
            logging.warning("Unspecified proposal scaling, defaulting to %f", PROPOSAL_SCALING_DEFAULT)
            self.scaling = PROPOSAL_SCALING_DEFAULT
        else:
            self.scaling = params[proposal_scaling_key]

        # adaptive proposal params
        proposal_adaptive_key = "proposal_adaptive"
        if proposal_adaptive_key in params:
            self.adaptive = params[proposal_adaptive_key]

            global_scaling_key = "proposal_gamma"
            if global_scaling_key not in params:
                logging.warning("Unknown proposal gamma, defaulting to %f", GAMMA_DEFAULT)
                self.gamma = GAMMA_DEFAULT
            else:
                self.gamma = params[global_scaling_key]
            
            adaptive_period_key = "proposal_adaptivity_period"
            if adaptive_period_key not in params:
                logging.warning("Unknown adaptivity period, defaulting to %d", ADAPTIVITY_PERIOD_DEFAULT)
                self.adaptivity_period = ADAPTIVITY_PERIOD_DEFAULT
            else:
                self.adaptivity_period = params[adaptive_period_key]

        else:
            logging.warning("Unspecified whether to adapt, defaulting to %s", str(ADAPTIVITY_DEFAULT))
            self.adaptive = ADAPTIVITY_DEFAULT

        # check for noise std
        noise_std_key = "noise_std"
        if noise_std_key not in params:
            logging.info("Noise standard deviation unspecified, defaulting to %f", NOISE_STD_DEFAULT)
            self.noise_std = NOISE_STD_DEFAULT
        else:
            self.noise_std = params[noise_std_key]

        # get number of samples to sample
        sample_count_key = "sample_count"
        if sample_count_key not in params[sample_count_key]:
            logging.warning("Number of samples not specified, defaulting to %d", SAMPLE_COUNT_DEFAULT)
            self.sample_count = SAMPLE_COUNT_DEFAULT
        else:
            self.sample_count = params[sample_count_key]


    def create_proposal_matrix(self):
        cov_vector = np.empty(len(self.priors))
        for idx, prior in enumerate(self.priors):
            if hasattr(prior, "std"):
                cov_vector[idx] = np.power(prior.std(), 2)
            else:
                # add support for uniform and other dists that dont have std attrib
                raise Exception("Unsupported distribution, no 'std' attribute.")
        return np.multiply(np.eye(len(cov_vector)), cov_vector)

    def sample(self) -> az.InferenceData:
        # check whether parallel sampling or not
        self.is_parallel = self.number_of_chains > 1 and ray_is_available and not self.force_sequential

        # setup shared text logging buffer if parallel sampling
        if self.is_parallel:
            self.shared_text_objref = SharedTextVariable.remote()

        # setup priors from config of flow wrapper
        self.setup_priors(self.config)

        # get measured data
        md = MeasuredData(self.config)
        md.initialize()
        # choose which boreholes to use
        boreholes = ["H1"]
        # choose which borehole conductivities to use, empty list means none
        cond_boreholes = []
        # get actual values
        _, values = md.generate_measured_samples(boreholes, cond_boreholes)
        logging.info("Loading observed values:")
        logging.info(values)

        # setup loglike
        self.setup_loglike(values, np.multiply(self.noise_std, np.eye(len(values))))
        self.measured_len = len(values)

        # combine into posterior
        posterior = tda.Posterior(self.prior, self.loglike, self.forward_model)

        # setup proposal covariance matrix (for random gaussian walk & adaptive metropolis)
        proposal_cov = self.create_proposal_matrix()
        logging.info(proposal_cov)
        # setup proposal
        #proposal = tda.IndependenceSampler(self.prior)
        proposal = tda.GaussianRandomWalk(proposal_cov, self.scaling, self.adaptive, self.gamma, self.adaptivity_period)

        # sample from prior to give all chains a different starting point
        # not doing this causes all of the chains to start from the same spot
        # -> messes up the directory naming, simultaneous access to the same files
        # also adds pointless correlation and reduces coverage
        prior_values = self.prior.rvs(self.number_of_chains)
        if self.is_parallel:
            prior_values = list(prior_values)

        # sampling process
        samples = tda.sample(posterior, proposal, self.sample_count, self.number_of_chains, prior_values)

        # if parallel sampling - concat results into one list
        if self.is_parallel:
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
        idata = tda.to_inference_data(chain=samples, parameter_names=self.prior_names, burnin=tune)

        return idata

    def setup_priors(self, config):
        priors = []
        prior_names = []
        for param in config["parameters"]:
            prior_name = param["name"]
            bounds = param["bounds"]
            match param["type"]:
                case "lognorm":
                    prior = sps.lognorm(s = bounds[1], scale = np.exp(bounds[0]))
                    logging.info(f"Prior lognorm, mu={prior.mean()}, std={prior.std()}")
                case "unif":
                    prior = sps.uniform(loc = bounds[0], scale = bounds[1] - bounds[0])
                    logging.info(f"Prior uniform, a={prior.a}, b={prior.b}")
                case "truncnorm":
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
                    a_trunc, b_trunc, mu, sigma = bounds
                    a, b = (a_trunc - mu) / sigma, (b_trunc - mu) / sigma
                    prior = sps.truncnorm(a, b, loc=mu, scale=sigma)
                    logging.info(f"Prior truncated norm, a={prior.a}, b={prior.b}, mean={prior.mean()}, std={prior.std()}")
            priors.append(prior)
            prior_names.append(prior_name)

        self.priors = priors
        self.prior_names = prior_names
        self.prior = tda.CompositePrior(priors)

    def setup_loglike(self, observed, cov):
        self.loglike = tda.GaussianLogLike(observed, cov)

    def forward_model(self, params):
        print(params)
        # reject automatically if params go negative
        if np.any(params <= 0):
            logging.info("Invalid proposal, skipping...")
            logging.info(params)
            return np.zeros(self.measured_len)

        logging.info("Passing params to flow")
        logging.info(params)
        self.flow_wrapper.set_parameters(data_par=params)

        start = time.time()
        res, data = self.flow_wrapper.get_observations()
        end = time.time()
        elapsed = end - start
        string = str(params.tolist()) + " | " + str(elapsed) + "\n"
        if self.is_parallel:
            job = self.shared_text_objref.append.remote(string)
            while job:
                _, job = ray.wait([job])
        else:
            self.observe_times.append(string)


        if self.config["conductivity_observe_points"]:
            num = len(self.config["conductivity_observe_points"])
            data = data[:-num]
        if res >= 0:
            return data
