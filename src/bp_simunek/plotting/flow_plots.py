import os
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpt
import arviz as az
import numpy as np
import scipy.stats as sps
from ..samplers.idata_tools import read_idata_from_file
from ..plotting.plotting_tools import save_plot
from definitions import ROOT_DIR

def plot_pressures(idata: az.InferenceData, exp, times):
    plt.figure()
    plt.xlabel("Čas [den]")
    plt.ylabel("Tlaková výška [m]")
    plt.xlim(-5, 370)
    plt.ylim(0, 300)
    plt.title("Změna tlakové výšky vrtu H1 v čase")
    obs_keys = [f"obs_{idx}" for idx in np.arange(0, 26)]
    exp_plot, = plt.plot(times, exp, color="black", linewidth=1, linestyle="dotted")
    areas = 100
    quantiles_95 = []
    quantiles_75 = []
    quantiles_25 = []
    quantiles_5 = []
    means = []
    medians = []

    norm_constant = 7000

    prev_time = 0
    prev_linspace = np.ones(areas) * 275
    prev_normhist = np.ones(areas) / areas
    for time_idx, key in enumerate(obs_keys):
        observed = idata["posterior_predictive"][key]
        mean = observed.mean()
        std = observed.std()
        means.append(mean)
        medians.append(np.median(observed))
        quantiles_95.append(np.quantile(observed, 0.95))
        quantiles_75.append(np.quantile(observed, 0.75))
        quantiles_25.append(np.quantile(observed, 0.25))
        quantiles_5.append(np.quantile(observed, 0.05))

        minimum = np.min(observed)
        maximum = np.max(observed)

        linspace = np.linspace(minimum, maximum, areas)
        
        interp_linspace = np.divide(np.add(prev_linspace, linspace), 2)
        interp_time = (prev_time + times[time_idx]) / 2

        hist, bins = np.histogram(observed, linspace)

        norm_hist = np.subtract(hist, np.min(hist))
        #norm_hist = np.divide(norm_hist, np.max(norm_hist))
        norm_hist = np.divide(norm_hist, norm_constant)

        cmap = mpl.colormaps["Oranges"].resampled(areas)

        for area in np.arange(1, areas):
            pdf_a = prev_normhist[area-1] * 0.7 + norm_hist[area-1] * 0.3
            pdf_b = prev_normhist[area-1] * 0.3 + norm_hist[area-1] * 0.7
            plt.fill_between([prev_time, interp_time], [prev_linspace[area-1], interp_linspace[area-1]], [prev_linspace[area], interp_linspace[area]], color=cmap(pdf_a))
            plt.fill_between([interp_time, times[time_idx]], [interp_linspace[area-1], linspace[area-1]], [interp_linspace[area], linspace[area]], color=cmap(pdf_b))
            #plt.fill_between([prev_time, times[time_idx]], [prev_linspace[area-1], linspace[area-1]], [prev_linspace[area], linspace[area]], color=cmap(norm_hist[area-1]))

        prev_time = times[time_idx]
        prev_linspace = linspace
        prev_normhist = norm_hist

    quantiles_95_plot, = plt.plot(times, quantiles_95, color="blue", linewidth=1)
    #quantiles_75_plot, = plt.plot(times, quantiles_75)
    #quantiles_25_plot, = plt.plot(times, quantiles_25)
    quantiles_5_plot, = plt.plot(times, quantiles_5, color="darkblue", linewidth=1)
    median_plot, = plt.plot(times, medians, color="indigo", linewidth=1, linestyle="dashed")

    filled_patch = mpt.Patch(color="orange", label="Pravděpodobnostní hustota inverze")

    plt.legend(
        [
            exp_plot, 
            median_plot, 
            quantiles_95_plot, 
            #quantiles_75_plot, 
            #quantiles_25_plot, 
            quantiles_5_plot,
            filled_patch
        ], 
        [
            "Naměřená data",
            "Medián z inverze", 
            "95. kvantil inverze",
            #"75. kvantil inverze",
            #"25. kvantil inverze",
            "5. kvantil inverze",
            "Pravděpodobnostní hustota inverze"
        ])
    #handles, labels = plt.gca().get_legend_handles_labels()
    #handles.extend([filled_patch])
    #plt.legend(handles, labels)
    return

def corr_progression_plot(idata: az.InferenceData, window_size):
    draws = idata.posterior.sizes["draw"]
    starts = np.arange(0, draws - window_size)
    ess_list = {param: [] for param in idata.posterior.data_vars}
    r_hat_list = {param: [] for param in idata.posterior.data_vars}

    for start in starts:
        subset = idata.sel(draw=slice(start, start + window_size))
        ess = az.ess(subset)
        rhat = az.rhat(subset)
        for param in ess.data_vars:
            ess_list[param] += [ess[param].values.tolist()]
            r_hat_list[param] += [rhat[param].values.tolist()]

    fig, axes = plt.subplots(2, 2, width_ratios=[0.75, 0.25])
    fig.set_figwidth(16)
    fig.set_figheight(9)
    axes[0, 0].set_xlabel("Začátek okna [iterace]")
    axes[0, 0].set_ylabel("Effective Sample Size []")
    axes[0, 0].set_title(f"Vývoj ESS s oknem {window_size}")
    refs = []
    for param, ess in ess_list.items():
        refs += axes[0, 0].plot(starts, ess, linewidth=0.5)

    axes[0, 1].legend(refs, list(ess_list.keys()))
    axes[0, 1].axis("off")

    axes[1, 0].set_xlabel("Začátek okna [iterace]")
    axes[1, 0].set_ylabel("r-hat []")
    axes[1, 0].set_title(f"Vývoj r-hat s oknem {window_size}")
    refs = []
    for param, r_hat in r_hat_list.items():
        refs += axes[1, 0].plot(starts, r_hat, linewidth=0.5)

    axes[1, 1].legend(refs, list(r_hat_list.keys()))
    axes[1, 1].axis("off")


def generate_all_flow_plots(idata: az.InferenceData, folder, config=None):
    az.plot_pair(idata, kind="kde")
    save_plot("pair_plot.pdf", folder_path=folder)
    az.plot_trace(idata)
    plt.tight_layout()
    save_plot("trace_plot.pdf", folder_path=folder)

    corr_progression_plot(idata, 100)
    save_plot("corr_progression_plot.pdf", folder_path=folder)

    axes = az.plot_posterior(idata, grid=[4, 2])

    if config is not None:

        priors = config["parameters"]

        prior_means = []
        prior_stds = []


        for prior in priors:
            prior_type = prior["type"]

            match prior_type:
                case "lognorm":
                    mu, sigma = prior["bounds"]
                case "truncnorm":
                    _, _, mu, sigma = prior["bounds"]

            prior_means += [mu]
            prior_stds += [sigma]
            exp = config["observed"]

    else:
        prior_means = [
           -16.4340685618576,
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
            3,
            0.7,
            0.5,
            0.5,
            #0.3,
            3.0,
            14,
            0.15,
            0.15
        ]

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


    for x, axrow in enumerate(axes):
        axrow_len = len(axrow)
        for y, ax in enumerate(axrow):
            idx = x * axrow_len + y
            if idx >= len(idata["posterior"]):
                continue
            if idx % axrow_len == 0:
                ax.set_ylabel("Hustota pravděpodobnosti", fontsize=15)
            if idx // axrow_len == 3:
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
        accepted, rejected = compute_accepted(idata)
        summary = str(az.summary(idata))
        summary += f"\n\n{accepted} accepted\n{rejected} rejected\n{accepted / (accepted + rejected)} acceptance rate"
        file.writelines(summary)

    times = [0, 10, 17, 27, 37, 47, 57, 67, 77, 87, 97, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 365]


    plot_pressures(idata, exp, times)
    save_plot("pressure_plot.pdf", folder_path=folder)


def compute_accepted(idata):
    variables =  list(idata["posterior"])
    accepted = 0
    rejected = 0
    for chain in idata["posterior"][variables[0]]:
        last_sample = chain[0]
        for sample in chain:
            if sample != last_sample:
                last_sample = sample
                accepted += 1
            else:
                rejected += 1
        
    return accepted, rejected



if __name__ == "__main__":
    idata_name = "10x3000_mlda_0.idata"
    folder_path = os.path.join(ROOT_DIR, "data", idata_name.split(".")[0])
    idata = read_idata_from_file(idata_name, folder_path)
    generate_all_flow_plots(idata,folder_path)
