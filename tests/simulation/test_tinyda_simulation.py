import os
from pathlib import Path
import logging
import shutil

import arviz as az
import pytest
import ray

from definitions import ROOT_DIR
from bp_simunek.simulation.flow_wrapper import Wrapper
from bp_simunek.samplers.tinyda_flow import TinyDAFlowWrapper
from bp_simunek.samplers.idata_tools import save_idata_to_file
from bp_simunek.plotting.flow_plots import generate_all_flow_plots

script_dir = os.path.dirname(os.path.realpath(__file__))

#@pytest.mark.skip
def test_simulation11_with_tinyda():
    #ray.init(runtime_env={"working_dir": ROOT_DIR})
    os.chdir(script_dir)
    observe_path = Path(script_dir, "../measured_data").absolute()
    template_dir = Path("templates", "test_workdir11").absolute()
    work_dir = Path(ROOT_DIR, "output", "test11").absolute()

    # clean workdir
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)

    # copy template to workdir
    shutil.copytree(template_dir, work_dir)

    # init wrapper - load config
    wrap = Wrapper(work_dir)

    # add observe path to config
    wrap.set_observe_path(observe_path)

    # tinyda + flow123 wrapper
    tinyda_wrapper = TinyDAFlowWrapper(wrap)

    # run sampling process
    idata = tinyda_wrapper.sample()

    # check if sampling was successful
    assert idata
    assert idata["posterior"].sizes["draw"] == tinyda_wrapper.sample_count

    # print samplin summary
    logging.info("\n%s", str(az.summary(idata)))
    # save results
    save_idata_to_file(idata, folder_path=work_dir, filename="flow.sim11.idata")

    # generate plots
    generate_all_flow_plots(idata, folder=work_dir)

@pytest.mark.skip
def test_simulation11_simplified_with_tinyda():
    #ray.init(runtime_env={"working_dir": ROOT_DIR})
    os.chdir(script_dir)
    workdir = Path("test_workdir11_simplified").absolute()
    solver_id = 42
    wrap = Wrapper(solver_id, workdir)
    wrap.sim._config["measured_data_dir"] = os.path.join(script_dir, "../measured_data")

    # tinyda + flow123 wrapper
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
    observe_path = Path(script_dir, "../measured_data").absolute()
    template_dir = Path("templates", "test_workdir11").absolute()
    work_dir = Path(ROOT_DIR, "output", "test11").absolute()

    # clean workdir
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)

    # copy template to workdir
    shutil.copytree(template_dir, work_dir)

    # init wrapper - load config
    wrap = Wrapper(work_dir)

    # add observe path to config
    wrap.set_observe_path(observe_path)

    # tinyda + flow123 wrapper
    tinyda_wrapper = TinyDAFlowWrapper(wrap)

    # run sampling process
    idata = tinyda_wrapper.sample()

    # check if sampling was successful
    assert idata
    assert idata["posterior"].sizes["draw"] == tinyda_wrapper.sample_count

    # print samplin summary
    logging.info("\n%s", str(az.summary(idata)))
    # save results
    idata_name = f"{tinyda_wrapper.number_of_chains}x{tinyda_wrapper.sample_count}_randomwalk_2.idata"
    save_idata_to_file(idata, folder_path=work_dir, filename=idata_name)

    # generate plots
    generate_all_flow_plots(idata, folder=work_dir)


if __name__ == "__main__":#
    sample11()
