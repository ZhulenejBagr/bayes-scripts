import os
from pathlib import Path
import logging
import arviz as az
import pytest

from bp_simunek.simulation.flow_wrapper import Wrapper
from bp_simunek.scripts.tinyda_flow import TinyDAFlowWrapper
from bp_simunek.samplers.idata_tools import save_idata_to_file


script_dir = script_dir = os.path.dirname(os.path.realpath(__file__))

#@pytest.mark.skip
def test_simulation3_with_tinyda():
    os.chdir(script_dir)
    # flow setup
    workdir = Path("test_workdir3").absolute()
    solver_id = 42
    wrap = Wrapper(solver_id, workdir)
    wrap.sim._config["measured_data_dir"] = os.path.join(script_dir, "../measured_data")

    # tinyda + flow123 wrapper
    tinyda_wrapper = TinyDAFlowWrapper(wrap)

    # sampling process
    sample_count = 5
    idata = tinyda_wrapper.sample(sample_count=sample_count)
    assert idata
    assert idata["posterior"].sizes["draw"] == sample_count
    logging.info("\n%s", str(az.summary(idata)))
    save_idata_to_file(idata, filename="flow.sim3.idata")

#@pytest.mark.skip
def test_simulation11_with_tinyda():
    os.chdir(script_dir)
    workdir = Path("test_workdir11").absolute()
    solver_id = 42
    wrap = Wrapper(solver_id, workdir)
    wrap.sim._config["measured_data_dir"] = os.path.join(script_dir, "../measured_data")

    # tinyda + flow123 wrapper
    tinyda_wrapper = TinyDAFlowWrapper(wrap)

    sample_count = 5
    idata = tinyda_wrapper.sample(sample_count=sample_count)
    assert idata
    assert idata["posterior"].sizes["draw"] == sample_count
    logging.info("\n%s", str(az.summary(idata)))
    save_idata_to_file(idata, filename="flow.sim11.idata")