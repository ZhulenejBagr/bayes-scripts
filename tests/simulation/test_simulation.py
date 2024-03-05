import shutil
import os
import numpy as np
from pathlib import Path
import pytest
import logging
import tinyDA as tda
import arviz as az

from bp_simunek import common
from bp_simunek.common.memoize import File
from bp_simunek.simulation.flow_wrapper import Wrapper
from bp_simunek.simulation.flow123d_simulation import generate_time_axis
from bp_simunek.scripts.tinyda_flow import TinyDAFlowWrapper
from bp_simunek.samplers.idata_tools import save_idata_to_file

script_dir = script_dir = os.path.dirname(os.path.realpath(__file__))

def test_call_flow():
    """
    Run inside container:
    ./bin/endorse_fterm
    rel 4.0.3> source ./venv/bin/activate
    (venv) rel 4.0.3> cd tests/common
    (venv) rel 4.0.3> pytest test_common.py
    """
    os.chdir(script_dir)
    flow_executable = [
        "/opt/flow123d/bin/flow123d",
        ### this is for installed Flow123d package or individual build
        #    - /home/domesova/flow123d_3.1.0/bin/fterm.sh
        #    - /home/paulie/local/flow123d/flow123d_3.1.0/bin/fterm.sh
        # "/home/paulie/Workspace/flow123d/bin/fterm",
        ### for flow123d individual build (not docker image install)
        # "--no-term",
        # "rel",
        ### for flow123d (docker image install)
        #    - --version
        #    - "3.1.0"
        #    - --tty
        #    - "false"
        #    - --interactive
        #    - "false"
        ### this is for both installed Flow123d package or individual build
        #"run",
        "--no_profiler"
    ]
    params = dict({
        "bc_pressure": "0",
        "a_tol": "1.0e-07",
        "output_fields": "[piezo_head_p0, pressure_p0, velocity_p0]"
    })
    workdir = Path("sample")
    common.force_mkdir(workdir, force=True)
    shutil.copy("test_workdir/common_files/square_1x1_xy.msh", workdir)  # copy mesh to sample dir
    completed_process, stdout, stderr =\
        common.flow_call(workdir, flow_executable, File("test_workdir/common_files/10_dirichlet_LMH_tmpl.yaml"), params)
    res, fo = common.flow_check(workdir, completed_process)
    assert res == True
    assert fo.process.returncode == 15 # as long as Flow123d ends with segfault
    assert fo.stdout.path == stdout.path
    assert fo.stderr.path == stderr.path
    assert fo.log.path.endswith("flow123.0.log")
    assert fo.hydro.balance_file.path.endswith("water_balance.yaml")
    assert fo.hydro.spatial_file.path.endswith("_fields.pvd")

    shutil.rmtree(workdir)


def test_flow_simulation1():
    os.chdir(script_dir)
    workdir = Path("test_workdir").absolute()
    solver_id = 42
    wrap = Wrapper(solver_id, workdir)

    # params: "bc_pressure": "0", "a_tol": "1.0e-07"
    params_in = np.array([
        [0, 0, 1e-7],
        [1, 100, 1e-8]
    ])

    for pars in params_in:
        idx = int(pars[0])
        wrap.set_parameters(data_par=pars[1:])
        res, sample_data = wrap.get_observations()

        print("Flow123d res: ", res, sample_data)
        print(sample_data)
        assert res > 0


def test_flow_simulation2():
    os.chdir(script_dir)
    workdir = Path("test_workdir2").absolute()
    solver_id = 42
    wrap = Wrapper(solver_id, workdir)

    params_in = np.array([
        [0, 31041937523.11239, 41023533.68961888, 18558265.240542404, 15795614.673716737, 9.420922909068678e-21,
            49.76537298111819, 1.0078720643166661e-16, 6.637653325266047, -0.0625, -0.6875],
        [1, 53289723492.17694, 41023533.68961888, 18558265.240542404, 15795614.673716737, 9.420922909068678e-21,
            49.76537298111819, 1.0078720643166661e-16, 6.637653325266047, -0.0625, -0.6875]
    ])

    for pars in params_in:
        idx = int(pars[0])
        wrap.set_parameters(data_par=pars[1:])
        res, sample_data = wrap.get_observations()
        logging.info(f"Flow123d res: {res} {sample_data}")

        times = generate_time_axis(wrap.sim._config)
        # sample_data shape: (1, n_times, n_elements)
        assert res >= 0
        assert len(times) == sample_data.shape[1]


#def test_flow_simulation3():
#    os.chdir(script_dir)
#    workdir = Path("test_workdir3").absolute()
#    solver_id = 42
#    wrap = Wrapper(solver_id, workdir)
#
#    params_in = np.array([
#        [0, 31041937523.11239, 41023533.68961888, 18558265.240542404, 15795614.673716737, 9.420922909068678e-21,
#            49.76537298111819, 1.0078720643166661e-16, 6.637653325266047, -0.0625, -0.6875],
#        [1, 53289723492.17694, 41023533.68961888, 18558265.240542404, 15795614.673716737, 9.420922909068678e-21,
#            49.76537298111819, 1.0078720643166661e-16, 6.637653325266047, -0.0625, -0.6875]
#    ])
#
#    for pars in params_in:
#        idx = int(pars[0])
#        wrap.set_parameters(data_par=pars[1:])
#        res, sample_data = wrap.get_observations()
#
#        print("Flow123d res: ", res, sample_data)
#
#        times = generate_time_axis(wrap.sim._config)
#        # sample_data shape: (1, n_times, n_elements)
#        logging.info(pars[1:])
#        assert res >= 0
#        # assert len(times) == sample_data.shape[1]

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

    # extract prior dists
    dists = [prior["dist"] for key, prior in dict(sorted(tinyda_wrapper.priors.items())).items()]

    # prior object for sampling
    comp_prior = tda.CompositePrior(dists)

    # combine into posterior
    posterior = tda.Posterior(comp_prior, tinyda_wrapper, tinyda_wrapper.forward_model)

    # setup proposal
    proposal = tda.IndependenceSampler(comp_prior)

    # sampling process
    sample_count = 2
    samples = tda.sample(posterior, proposal, iterations=sample_count, n_chains=1)
    for sample in samples["chain_0"]:
        logging.info(sample.parameters)

    # check and save samples
    # + 1 to account for the first sample (prior)
    assert len(samples["chain_0"]) == sample_count + 1
    # get param names
    params = [param["name"] for param in wrap.sim._config["parameters"]]
    idata = tda.to_inference_data(chain=samples, parameter_names=params)
    logging.info(az.summary(idata))
    save_idata_to_file(idata, filename="flow.idata",)
