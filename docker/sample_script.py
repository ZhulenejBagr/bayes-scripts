from pymc_metropolis import sample_regular, save_idata_to_file

if __name__ == "__main__":
    idata = sample_regular()
    save_idata_to_file(idata, filename="idata")
    