import numpy as np
from arviz import summary

from src.bp_simunek.samplers.tinyda_sampler import sample
from src.bp_simunek.plotting.conductivity_plots import plot_pair_custom
from src.bp_simunek.samplers.idata_tools import save_idata_to_file

def get_explicit_data(idata):
    return {
        "U_0": idata["posterior"]["U_0"],
        "U_1": idata["posterior"]["U_1"],
        "log_likelihood": idata["sample_stats"]["likelihood"]
    }

# R1 prior specs
R1_mean = np.array([5, 3])
R1_cov = np.array([[4, -2], [-2, 4]])

# R2 prior specs
R2_mean = np.array([7, 5])
R2_cov = np.array([[12, -6], [-6, 12]])

# sample with R1
R1_idata = sample(prior_mean=R1_mean, prior_cov=R1_cov)
print(summary(R1_idata))
save_idata_to_file(R1_idata, "R1.idata")

# sample with R2
R2_idata = sample(prior_mean=R2_mean, prior_cov=R2_cov)
print(summary(R2_idata))
save_idata_to_file(R2_idata, "R2.idata")

# change format of sampled data and plot
plot_pair_custom(R1_idata, filename="R1_pair_plot.png", explicit_data=get_explicit_data(R1_idata), plot_prior=False)
plot_pair_custom(R1_idata, filename="R2_pair_plot.png", explicit_data=get_explicit_data(R2_idata), plot_prior=False)