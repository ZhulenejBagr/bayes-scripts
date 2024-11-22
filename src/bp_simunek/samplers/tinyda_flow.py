import os
import time
import traceback
import logging
from functools import partial

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
TUNE_COUNT_DEFAULT = 1
MLDA_DEFAULT = False
MLDA_LEVELS_DEFAULT = 2

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
        self.tune_count = TUNE_COUNT_DEFAULT
        self.mlda = MLDA_DEFAULT
        self.levels = MLDA_LEVELS_DEFAULT

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
        if sample_count_key not in params:
            logging.warning("Number of samples not specified, defaulting to %d", SAMPLE_COUNT_DEFAULT)
            self.sample_count = SAMPLE_COUNT_DEFAULT
        else:
            self.sample_count = params[sample_count_key]

        # get length of tune
        tune_count_key = "tune_count"
        if tune_count_key not in params:
            logging.warning("Length of tune not specified, defaulting to %d", TUNE_COUNT_DEFAULT)
            self.tune_count = TUNE_COUNT_DEFAULT
        else:
            self.tune_count = params[tune_count_key]

        # check if using MLDA
        mlda_key = "mlda"
        if mlda_key not in params:
            logging.warning("MLDA not specified, defaulting to %d", MLDA_DEFAULT)
            self.mlda = MLDA_DEFAULT
        else:
            self.mlda = params[mlda_key]

        if self.mlda:
             # check for number of mlda levels
            mlda_levels_key = "mlda_levels"
            if mlda_levels_key not in params:
                logging.warning("Number of MLDA levels not specified, defaulting to %d", MLDA_LEVELS_DEFAULT)
                self.mlda_levels = MLDA_LEVELS_DEFAULT
            else:
                self.mlda_levels = params[mlda_levels_key]

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
        noise_cov = np.multiply(self.noise_std, np.eye(len(values)))
        logging.info("Using following noise covariance matrix")
        logging.info(noise_cov)
        self.setup_loglike(values, noise_cov)
        self.measured_len = len(values)

        # combine into posterior
        # if using mlda, use one mesh per model
        if not self.mlda:
            posteriors = tda.Posterior(self.prior, self.loglike, self.forward_model)
        else:
            self.flow_wrapper.set_mlda_level(0)
            posteriors = []
            for level in np.arange(self.mlda_levels):
                logging.info(level)
                forward_model = partial(self.forward_model_mlda, level=level)
                posterior_level = tda.Posterior(self.prior, self.loglike, forward_model)
                posteriors.append(posterior_level)
        # setup proposal covariance matrix (for random gaussian walk & adaptive metropolis)
        proposal_cov = self.create_proposal_matrix()
        # setup proposal
        #proposal = tda.IndependenceSampler(self.prior)
        proposal = tda.GaussianRandomWalk(proposal_cov, self.scaling, self.adaptive, self.gamma, self.adaptivity_period)

        # sample from prior to give all chains a different starting point
        # not doing this causes all of the chains to start from the same spot
        # -> messes up the directory naming, simultaneous access to the same files
        # also adds pointless correlation and reduces coverage
        prior_values = self.prior.rvs(self.number_of_chains)
        if self.number_of_chains > 1:
            prior_values = list(prior_values)

        # sampling process
        samples = tda.sample(posteriors, proposal, self.sample_count, self.number_of_chains, prior_values, 1)

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
        idata = tda.to_inference_data(chain=samples, parameter_names=[prior["name"] for prior in self.priors], burnin=self.tune_count)

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
                    #a, b = (a_trunc - mu) / sigma, (b_trunc - mu) / sigma
                    #prior = sps.truncnorm(a, b, loc=mu, scale=sigma)
                    _, _, mu, sigma = bounds
                    prior = sps.norm(loc=mu, scale=sigma)
                    logging.info("Prior truncated norm, a=%s, b=%s, mean=%s, std=%s", prior.a, prior.b, prior.mean(), prior.std())
            priors.append({
                "name": prior_name,
                "type": prior_type,
                "dist": prior,
                "params": bounds
            })

        self.priors = priors
        self.prior = tda.distributions.JointPrior([prior["dist"] for prior in priors])

    def setup_loglike(self, observed, cov):
        logging.info("bruh")
        self.loglike = tda.GaussianLogLike(np.full(len(observed), 1), cov)

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
                    a, b, mu, sigma = prior["params"]
                    lower_bound = (a - mu) / sigma
                    upper_bound = (b - mu) / sigma
                    phi_a = sps.norm.cdf(lower_bound)
                    phi_b = sps.norm.cdf(upper_bound)
                    phi_param = sps.norm.cdf(param, loc=mu, scale=sigma)
                    trans_param = sps.norm.ppf((phi_b - phi_a)*phi_param + phi_a)*sigma + mu

            trans_params.append(trans_param)

        logging.info("Passing params to flow")
        logging.info(trans_params)
        self.flow_wrapper.set_parameters(data_par=trans_params)

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

    def forward_model_mlda(self, params, level):
        self.flow_wrapper.set_mlda_level(level)
        logging.info("Setting sampler to level %i", level)
        return self.forward_model(params)