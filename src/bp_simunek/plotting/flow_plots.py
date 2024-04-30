import os
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import scipy.stats as sps
from bp_simunek.samplers.idata_tools import read_idata_from_file
from bp_simunek.plotting.plotting_tools import save_plot
from definitions import ROOT_DIR

def plot_pressures(idata, exp, times):
    plt.figure()
    plt.xlabel("Čas [den]")
    plt.ylabel("Tlak [kPa]")
    plt.title("Vývoj tlaku v čase")
    obs_keys = [f"obs_{idx}" for idx in np.arange(0, 26)]
    exp_plot, = plt.plot(times, exp)
    minimums = []
    maximums = []
    for key in obs_keys:
        observed = idata["posterior_predictive"][key]
        mean = observed.mean()
        std = observed.std()
        minimum = mean - 3*std
        minimums.append(minimum)
        maximum = mean + 3*std
        maximums.append(maximum)

    min_plot, = plt.plot(times, minimums)
    max_plot, = plt.plot(times, maximums)

    plt.legend([exp_plot, min_plot, max_plot], ["Naměřená data", "Minimum dat z inverze", "Maximum dat z inverze"])
    return



def generate_all_flow_plots(idata, folder):
    az.plot_pair(idata, kind="kde")
    save_plot("pair_plot.pdf", folder_path=folder)
    az.plot_trace(idata)
    save_plot("trace_plot.pdf", folder_path=folder)

    #az.plot_forest(idata)
    #save_plot("forest_plot.pdf", folder_path=folder)
    #az.plot_density(idata)
    #save_plot("density_plot.pdf", folder_path=folder)


    axes = az.plot_posterior(idata, )

    prior_means = [
        24.8176103991685,
        17.6221730477346,
        16.2134058307626,
        #17.9098551201864,
        -48.8651125766410,
        33,
        -36.8413614879047,
        1.79175946922806
    ]

    prior_stds = [
        0.5,
        0.3,
        0.3,
        #0.3,
        2.0,
        9,
        0.1,
        0.1
    ]

    for x, axrow in enumerate(axes):
        axrow_len = len(axrow)
        for y, ax in enumerate(axrow):
            idx = x * axrow_len + y
            if idx >= len(idata["posterior"]):
                continue
            if idx % axrow_len == 0:
                ax.set_ylabel("Hustota pravděpodobnosti", fontsize=15)
            if idx // axrow_len == 1:
                ax.set_xlabel("Hodnota parametru", fontsize=15)
            mean = prior_means[idx]
            std = prior_stds[idx]
            xvals = np.linspace(mean - 3 * std, mean + 3 * std, 100)
            yvals = sps.norm.pdf(xvals, mean, std)
            posterior = ax.lines[0]
            prior, = ax.plot(xvals, yvals, color="red", linestyle="dashed", label="Původní odhad")
            if idx == 0:
                ax.legend([prior, posterior], ["Původní odhad", "Výsledek inverze"], fontsize=15, loc="upper left")
            plt.tight_layout()
    save_plot("posterior_plot.pdf", folder_path=folder)

    with open(os.path.join(folder, "summary.txt"), "w+") as file:
        summary = az.summary(idata)
        file.writelines(str(summary))


    # temporarily add constant data, eventually load data from config
    exp = [
        262.00970108619634, 62.843249986131575, 48.08262448100288,
        42.681038591463576, 42.93917678143266, 45.08236786944522,
        49.94181194643198, 56.007637102484054, 61.35265637436543,
        60.28968371081408, 59.66429769637641, 60.60653451083565,
        79.19950050268767, 81.48672728810196, 91.97984245675362,
        90.06164513633016, 89.22193608913719, 79.33002552259684,
        79.32797971798708, 82.6842177813621, 77.31306993967296,
        78.0024018399629, 85.3324173501424, 75.85212647719274,
        82.55973059689288, 89.09948153884811]

    times = [0, 10, 17, 27, 37, 47, 57, 67, 77, 87, 97, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 365]


    plot_pressures(idata, exp, times)
    save_plot("pressure_plot.pdf", folder_path=folder)



if __name__ == "__main__":
    idata_name = "10x500_mlda_0.idata"
    folder_path = os.path.join(ROOT_DIR, "data", "job_21453475.meta-pbs.metacentrum.cz")
    #folder_path = os.path.join(ROOT_DIR, "data", idata_name.split(".")[0])
    idata = read_idata_from_file(idata_name, folder_path)
    generate_all_flow_plots(idata, folder_path)
