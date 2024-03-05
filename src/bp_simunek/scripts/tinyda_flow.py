import tinyDA as tda
import scipy.stats as sps
import numpy as np
import logging

from bp_simunek.simulation.measured_data import MeasuredData

class TinyDAFlowWrapper():
    """
    Wrapper combining a flow123 instance into a tinyDA sampler
    """

    def __init__(self, flow_wrapper):
        self.flow_wrapper = flow_wrapper
        self.observed_data = MeasuredData(self.flow_wrapper.sim._config)
        self.observed_data.initialize()
        self.noise_dist = sps.norm(loc = 0, scale = 2e-4)


    def setup_priors(self, config):
        priors = {}
        idx = 0
        for param in config["parameters"]:
            prior = {
                "name": param["name"]
            }
            bounds = param["bounds"]
            match param["type"]:
                case "lognorm":
                    prior["dist"] = sps.lognorm(s = bounds[1], scale = np.exp(bounds[0]))
                case "unif":
                    prior["dist"] = sps.uniform(loc = bounds[0], scale = bounds[1] - bounds[0])
                # edge case, dont know how to interpret truncnorm - hardcoded to return a known good value for now
                case "truncnorm":
                    # prior["dist"] = sps.norm(loc = 50, scale = 1e-4)
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
                    a_trunc, b_trunc, mu, sigma = bounds
                    a, b = (a_trunc - mu) / sigma, (b_trunc - mu) / sigma
                    prior["dist"] = sps.truncnorm(a, b, loc=mu, scale=sigma)
            priors[idx] = prior
            idx += 1

        self.priors = priors

    def loglike(self, data):
        # temporary random likelihood
        return np.log(sps.uniform(0.2, 0.8).rvs(1))

    def forward_model(self, params):
        self.flow_wrapper.set_parameters(data_par=params)
        res, data = self.flow_wrapper.get_observations()
        # TODO add proper data parsing
        # currently it only removes 2 last elements
        data = data[:-2]
        if res >= 0:
            return data

    # not necesary, used for debugging
    def sample_prior(self):
        assert self.priors
        values = []
        for _, prior in dict(sorted(self.priors.items())).items():
            values.append(prior["dist"].rvs(1))
        return values
