import os
from pathlib import Path
import logging
import tinyDA as tda
import arviz as az
import numpy as np

from bp_simunek.simulation.flow_wrapper import Wrapper
from bp_simunek.scripts.tinyda_flow import TinyDAFlowWrapper
from bp_simunek.samplers.idata_tools import save_idata_to_file
from bp_simunek.simulation.measured_data import MeasuredData


script_dir = script_dir = os.path.dirname(os.path.realpath(__file__))

def test_flow_with_tinyda():
    os.chdir(script_dir)
    # flow setup
    workdir = Path("test_workdir3").absolute()
    solver_id = 42
    wrap = Wrapper(solver_id, workdir)
    wrap.sim._config["measured_data_dir"] = os.path.join(script_dir, "../measured_data")

    # tinyda + flow123 wrapper
    tinyda_wrapper = TinyDAFlowWrapper(wrap)

    # setup priors from config of flow wrapper
    tinyda_wrapper.setup_priors(wrap.sim._config)

    # setup likelihood
    md = MeasuredData(wrap.sim._config)
    md.initialize()
    boreholes = ["H1"]
    cond_boreholes = []
    _, values = md.generate_measured_samples(boreholes, cond_boreholes)
    tinyda_wrapper.setup_loglike(values, np.eye(len(values)))

    # combine into posterior
    posterior = tda.Posterior(tinyda_wrapper.prior, tinyda_wrapper.loglike, tinyda_wrapper.forward_model)

    # setup proposal
    proposal = tda.IndependenceSampler(tinyda_wrapper.prior)

    # sampling process
    sample_count = 10
    samples = tda.sample(posterior, proposal, iterations=sample_count, n_chains=1)

    # check and save samples
    # + 1 to account for the first sample (prior)
    assert len(samples["chain_0"]) == sample_count + 1
    # get param names
    params = [param["name"] for param in wrap.sim._config["parameters"]]
    idata = tda.to_inference_data(chain=samples, parameter_names=params)
    logging.info("\n%s", str(az.summary(idata)))
    save_idata_to_file(idata, filename="flow.idata")
