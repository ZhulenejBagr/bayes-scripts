import shutil

from bp_simunek import common
import os
import numpy as np
from scipy import stats
from bp_simunek.common.memoize import File
from pathlib import Path
import pytest
import logging

from bp_simunek.simulation.flow_wrapper import Wrapper
from bp_simunek.simulation.flow123d_simulation import generate_time_axis

script_dir = script_dir = os.path.dirname(os.path.realpath(__file__))

def test_workdir():
    pass



def test_substitute_placeholders():
    pass


def test_check_conv_reasons():
    pass


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


def test_flow_simulation3():
    os.chdir(script_dir)
    workdir = Path("test_workdir3").absolute()
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

        print("Flow123d res: ", res, sample_data)

        times = generate_time_axis(wrap.sim._config)
        # sample_data shape: (1, n_times, n_elements)
        assert res >= 0
        # assert len(times) == sample_data.shape[1]


def test_sample_from_population():
    population = np.array([(i, i*i) for i in [1,2,3,4]])
    frequencies = [10, 3, 20, 4]
    i_samples = common.sample_from_population(10000, frequencies)
    samples = population[i_samples, ...]
    sampled_freq = 4 * [0]
    for i, ii in samples:
        sampled_freq[i-1] += 1
    chisq, pval = stats.chisquare(sampled_freq, np.array(frequencies) / np.sum(frequencies) * len(samples))
    print("\nChi square test pval: ", pval)
    assert pval > 0.05
