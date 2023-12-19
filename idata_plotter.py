from glob import glob
import os
from pymc_metropolis import plot_all, read_idata_from_file

if __name__ == "__main__":
    # idata name structure:
    # <type_of_set>.<sampler_used>.idata
    #
    paths = glob("/mnt/idata/*")
    for path in paths:
        print(f"Plotting data for {path}")
        folder_path, filename = os.path.split(path)
        method_name = filename.split(".")[-2]
        set_type = filename.split(".")[0]
        idata = read_idata_from_file(folder_path=folder_path, filename=filename)
        plot_folder = os.path.join(os.path.split(folder_path)[0], "graphs", set_type, method_name)
        plot_all(idata, folder_path=plot_folder)
