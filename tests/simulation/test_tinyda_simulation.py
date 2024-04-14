import os
from pathlib import Path
import logging

import arviz as az
import pytest
import ray

from definitions import ROOT_DIR
from bp_simunek.simulation.flow_wrapper import Wrapper, setup_config
from bp_simunek.samplers.tinyda_flow import TinyDAFlowWrapper
from bp_simunek.samplers.idata_tools import save_idata_to_file

script_dir = os.path.dirname(os.path.realpath(__file__))

#@pytest.mark.skip
def test_simulation11_with_tinyda():
    #ray.init(runtime_env={"working_dir": ROOT_DIR})
    os.chdir(script_dir)
    output_dir = Path(ROOT_DIR, "output", "test11").absolute()
    work_dir = Path("templates", "test_workdir11").absolute()
    config = setup_config(work_dir)
    solver_id = 42
    wrap = Wrapper(solver_id, output_dir, config)
    wrap.sim._config["measured_data_dir"] = os.path.join(script_dir, "../measured_data")

    # tinyda + flow123 wrapper
    tinyda_wrapper = TinyDAFlowWrapper(wrap)

    idata = tinyda_wrapper.sample()
    assert idata
    assert idata["posterior"].sizes["draw"] == tinyda_wrapper.sample_count
    logging.info("\n%s", str(az.summary(idata)))
    save_idata_to_file(idata, folder_path=output_dir, filename="flow.sim11.idata")

@pytest.mark.skip
def test_simulation11_simplified_with_tinyda():
    #ray.init(runtime_env={"working_dir": ROOT_DIR})
    os.chdir(script_dir)
    workdir = Path("test_workdir11_simplified").absolute()
    solver_id = 42
    wrap = Wrapper(solver_id, workdir)
    wrap.sim._config["measured_data_dir"] = os.path.join(script_dir, "../measured_data")

    # tinyda + flow123 wrapper
    n_chains = 1
    tinyda_wrapper = TinyDAFlowWrapper(wrap)

    sample_count = 5
    idata = tinyda_wrapper.sample()
    assert idata
    assert idata["posterior"].sizes["draw"] == sample_count
    logging.info("\n%s", str(az.summary(idata)))
    save_idata_to_file(idata, filename="flow.sim11.idata")

@pytest.mark.skip
def test_simulation11_with_tinyda_parallel():
    os.chdir(script_dir)
    workdir = Path("test_workdir11").absolute()
    solver_id = 42
    wrap = Wrapper(solver_id, workdir)
    wrap.sim._config["measured_data_dir"] = os.path.join(script_dir, "../measured_data")

    # tinyda + flow123 wrapper
    tinyda_wrapper = TinyDAFlowWrapper(wrap)

    sample_count = 5
    idata = tinyda_wrapper.sample(sample_count, False)
    assert idata
    assert idata["posterior"].sizes["draw"] == sample_count
    logging.info("\n%s", str(az.summary(idata)))
    save_idata_to_file(idata, filename="flow.sim11.idata")

@pytest.mark.skip
def sample11(idata_name="flow_tinyda_1000.idata"):
    # probably not the best solution
    # 107 char limit for socket path
    tmp_dir_symlink = os.path.join(os.path.expanduser("~"), ".r")
    logging.info(tmp_dir_symlink)
    if not os.path.islink(tmp_dir_symlink):
        raise Exception("Missing symlink for Ray temp storage")
    ray.init(_temp_dir=tmp_dir_symlink)
    os.chdir(script_dir)
    workdir = Path("test_workdir11").absolute()
    solver_id = 42
    wrap = Wrapper(solver_id, workdir)
    wrap.sim._config["measured_data_dir"] = os.path.join(script_dir, "../measured_data")

    # tinyda + flow123 wrapper
    n_chains = 10
    tinyda_wrapper = TinyDAFlowWrapper(wrap)

    idata = tinyda_wrapper.sample()
    logging.info("\n%s", str(az.summary(idata)))
    save_idata_to_file(idata, filename=idata_name)


if __name__ == "__main__":#
    sample11()
