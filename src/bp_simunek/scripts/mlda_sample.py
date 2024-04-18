import logging
import os
from pathlib import Path
import shutil

import ray

from bp_simunek.simulation.flow_wrapper import Wrapper
from bp_simunek.samplers.tinyda_flow import TinyDAFlowWrapper
from bp_simunek.samplers.idata_tools import save_idata_to_file
from bp_simunek.plotting.flow_plots import generate_all_flow_plots

from definitions import ROOT_DIR

script_dir = os.path.dirname(os.path.realpath(__file__))

def sample11(idata_name="flow_tinyda_1000.idata"):
    # probably not the best solution
    # 107 char limit for socket path
    tmp_dir_symlink = os.path.join(os.path.expanduser("~"), ".r")
    logging.info(tmp_dir_symlink)
    if not os.path.islink(tmp_dir_symlink):
        raise Exception("Missing symlink for Ray temp storage")
    ray.init(_temp_dir=tmp_dir_symlink)

    os.chdir(script_dir)
    observe_path = Path(ROOT_DIR, "scripts", "sample_template").absolute()
    template_dir = Path(ROOT_DIR, "scripts", "sample_template").absolute()
    workdir = os.environ.get("SCRATCHDIR")
    if workdir is None:
        work_dir = Path(ROOT_DIR, "output", "test12").absolute()
    else:
        work_dir = Path(os.path.join(workdir, "")).absolute()

    logging.info("Using workdir %s", work_dir)

    # copy template to workdir
    shutil.copytree(template_dir, work_dir, dirs_exist_ok=True)

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

    # save results
    idata_name = f"{tinyda_wrapper.number_of_chains}x{tinyda_wrapper.sample_count}_mlda_0.idata"
    save_idata_to_file(idata, folder_path=work_dir, filename=idata_name)

    # generate plots
    generate_all_flow_plots(idata, folder=work_dir)

if __name__ == "__main__":
    sample11()