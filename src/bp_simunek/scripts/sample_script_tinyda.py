from arviz import summary

from src.bp_simunek.samplers.tinyda_sampler import sample
from src.bp_simunek.plotting.conductivity_plots import plot_pair_custom

idata = sample()
print(summary(idata))
explicit_data = {
    "U_0": idata["posterior"]["U_0"],
    "U_1": idata["posterior"]["U_1"],
    "log_likelihood": idata["sample_stats"]["likelihood"]
}
plot_pair_custom(idata, explicit_data=explicit_data, plot_prior=False)