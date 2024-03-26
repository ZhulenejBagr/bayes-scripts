import matplotlib.pyplot as plt
import arviz as az
from bp_simunek.samplers.idata_tools import read_idata_from_file
from bp_simunek.plotting.plotting_tools import save_plot

if __name__ == "__main__":
    idata = read_idata_from_file("10x500_independence_0.idata")
    axes = az.plot_posterior(idata)
    print(az.summary(idata))
    save_plot("10x500_independence_posterior.pdf")