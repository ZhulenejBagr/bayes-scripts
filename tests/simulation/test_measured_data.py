import shutil
import os
import numpy as np
from pathlib import Path
import pytest
import logging

from bp_simunek.simulation.flow_wrapper import Wrapper
from bp_simunek.simulation.measured_data import MeasuredData
from bp_simunek.simulation.flow123d_simulation import generate_time_axis

script_dir = script_dir = os.path.dirname(os.path.realpath(__file__))


def test_measured_data():
    os.chdir(script_dir)
    workdir = Path("test_workdir3").absolute()
    solver_id = 42
    wrap = Wrapper(solver_id, workdir)

    # pass the measured data directory through config (better way?)
    wrap.sim._config["measured_data_dir"] = os.path.join(script_dir, "../measured_data")
    conf_times = generate_time_axis(wrap.sim._config)

    md = MeasuredData(wrap.sim._config)
    md.initialize()

    md.plot_all_data()
    assert (workdir/"measured_data_TSX.pdf").exists()
    md.plot_interp_data()
    assert (workdir / "interp_data_TSX.pdf").exists()

    boreholes = ["H1"]
    cond_boreholes = []
    times, values = md.generate_measured_samples(boreholes, cond_boreholes)
    logging.info(times)
    assert len(conf_times) == len(times)
    assert np.all(conf_times == times)
    logging.info(values)
    assert len(values) == len(times)

    cond_boreholes = ["H1_cond"]
    times, values = md.generate_measured_samples(boreholes, cond_boreholes)
    logging.info(values)
    assert len(values) == len(times)+1

    boreholes = ["V1", "V2", "H1", "H2"]
    cond_boreholes = ["V1_cond", "V2_cond", "H1_cond", "H2_cond"]
    times, values = md.generate_measured_samples(boreholes, cond_boreholes)
    logging.info(values)
    assert len(values) == 4*(len(times) + 1)

    boreholes = wrap.sim._config["observe_points"]
    cond_boreholes = wrap.sim._config["conductivity_observe_points"]
    times, values = md.generate_measured_samples(boreholes, cond_boreholes)
    logging.info(values)
    assert len(values) == len(times) + 2

    # cleanup
    os.remove((workdir / "measured_data_TSX.pdf"))
    os.remove((workdir / "interp_data_TSX.pdf"))


# def test_flow_simulation3():
#     os.chdir(script_dir)
#     workdir = Path("test_workdir3").absolute()
#     solver_id = 42
#     wrap = Wrapper(solver_id, workdir)
#
#     params_in = np.array([
#         [0, 31041937523.11239, 41023533.68961888, 18558265.240542404, 15795614.673716737, 9.420922909068678e-21,
#             49.76537298111819, 1.0078720643166661e-16, 6.637653325266047, -0.0625, -0.6875],
#         [1, 53289723492.17694, 41023533.68961888, 18558265.240542404, 15795614.673716737, 9.420922909068678e-21,
#             49.76537298111819, 1.0078720643166661e-16, 6.637653325266047, -0.0625, -0.6875]
#     ])
#
#     for pars in params_in:
#         idx = int(pars[0])
#         wrap.set_parameters(data_par=pars[1:])
#         res, sample_data = wrap.get_observations()
#
#         print("Flow123d res: ", res, sample_data)
#
#         times = generate_time_axis(wrap.sim._config)
#         # sample_data shape: (1, n_times, n_elements)
#         assert res >= 0
#         # assert len(times) == sample_data.shape[1]
