from src.bp_simunek.samplers.pymc_sampler import sample_regular
from src.bp_simunek.samplers.idata_tools import save_idata_to_file

if __name__ == "__main__":
    idata = sample_regular()
    save_idata_to_file(idata, filename="idata")
    