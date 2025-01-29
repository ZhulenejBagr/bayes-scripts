import os
import logging
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpt
import arviz as az
import numpy as np
import scipy.stats as sps
from ..samplers.idata_tools import read_idata_from_file
from ..plotting.plotting_tools import save_plot, save_plots_pdf_pages
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
        #observed_unfiltered = idata["posterior_predictive"][key]
        #observed = observed_unfiltered.where((observed_unfiltered <= 500) & (observed_unfiltered > 0))
        observed = idata["posterior_predictive"][key]
        mean = observed.mean()
        std = observed.std()
        minimum = observed.min()
        maximum = observed.max()
        means.append(mean)
        medians.append(np.median(observed))
        quantiles_95.append(np.quantile(observed, 0.95))
        quantiles_75.append(np.quantile(observed, 0.75))
        quantiles_25.append(np.quantile(observed, 0.25))
        quantiles_5.append(np.quantile(observed, 0.05))


        """  linspace = np.linspace(minimum, maximum, areas)
        
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
        prev_normhist = norm_hist """

    quantiles_95_plot, = plt.plot(times, quantiles_95, color="blue", linewidth=1)
    #quantiles_75_plot, = plt.plot(times, quantiles_75)
    #quantiles_25_plot, = plt.plot(times, quantiles_25)
    quantiles_5_plot, = plt.plot(times, quantiles_5, color="darkblue", linewidth=1)
    median_plot, = plt.plot(times, medians, color="indigo", linewidth=1, linestyle="dashed")

    #filled_patch = mpt.Patch(color="orange", label="Pravděpodobnostní hustota inverze")

    plt.legend(
        [
            exp_plot, 
            median_plot, 
            quantiles_95_plot, 
            #quantiles_75_plot, 
            #quantiles_25_plot, 
            quantiles_5_plot,
            #filled_patch
        ], 
        [
            "Naměřená data",
            "Medián z inverze", 
            "95. kvantil inverze",
            #"75. kvantil inverze",
            #"25. kvantil inverze",
            "5. kvantil inverze",
            #"Pravděpodobnostní hustota inverze"
        ])
    #handles, labels = plt.gca().get_legend_handles_labels()
    #handles.extend([filled_patch])
    #plt.legend(handles, labels)
    return

def data_window_plots(idata: az.InferenceData, window_size):
    draws = idata.posterior.sizes["draw"]
    starts = np.arange(0, draws - window_size)
    ess_list = {param: np.empty(0, dtype=float) for param in idata.posterior.data_vars}
    r_hat_list = {param: np.empty(0, dtype=float) for param in idata.posterior.data_vars}
    mean_list = {param: np.empty(0, dtype=float) for param in idata.posterior.data_vars}
    std_list = {param: np.empty(0, dtype=float) for param in idata.posterior.data_vars}

    # process all metrics for all data windows
    for start in starts:
        subset = idata.isel(draw=slice(start, start + window_size))
        ess = az.ess(subset)
        r_hat = az.rhat(subset)
        posterior = subset.posterior
        for param in ess.data_vars:
            ess_list[param] = np.append(ess_list[param], ess[param].values.tolist())
            r_hat_list[param] = np.append(r_hat_list[param], r_hat[param].values.tolist())
            values = posterior[param]
            mean_list[param] = np.append(mean_list[param], values.mean())
            std_list[param] = np.append(std_list[param], values.std())

    # first 2 plots - ESS and r-hat
    fig_corr, axes_corr = plt.subplots(2, 2, width_ratios=[0.75, 0.25])
    fig_corr.set_figwidth(16)
    fig_corr.set_figheight(9)
    axes_corr[0, 0].set_xlabel("Začátek okna [iterace]")
    axes_corr[0, 0].set_ylabel("Effective Sample Size []")
    axes_corr[0, 0].set_title(f"Vývoj ESS s oknem {window_size}")
    refs = []
    for param, ess in ess_list.items():
        refs += axes_corr[0, 0].plot(starts, ess, linewidth=0.5)

    axes_corr[0, 1].legend(refs, list(ess_list.keys()))
    axes_corr[0, 1].axis("off")

    axes_corr[1, 0].set_xlabel("Začátek okna [iterace]")
    axes_corr[1, 0].set_ylabel("r-hat []")
    axes_corr[1, 0].set_title(f"Vývoj r-hat s oknem {window_size}")
    refs = []
    for param, r_hat in r_hat_list.items():
        refs += axes_corr[1, 0].plot(starts, r_hat, linewidth=0.5)

    axes_corr[1, 1].legend(refs, list(r_hat_list.keys()))
    axes_corr[1, 1].axis("off")

    # second pair - mean and std

    figs_stats = []
    for param in mean_list:
        means = mean_list[param]
        stds = std_list[param]

        fig_param, axes_param = plt.subplots(figsize=(16, 9))
        axes_param.set_title(f"Vývoj střední a rozpylu parametru {param} s oknem {window_size}")
        axes_param.set_xlabel("Začátek okna [iterace]")
        axes_param.set_ylabel("Hodnota parametru")
        axes_param.grid(True)

        axes_param.plot(starts, means, label="střední hodnota")
        for i in range(1, 100):
            alpha = (1 - (i / 100))  ** 2 * 0.5
            axes_param.fill_between(
                starts,
                means - stds * (i / 100),
                means + stds * (i / 100),
                alpha = alpha,
                color = "red"
            )
        figs_stats += [fig_param]

    return fig_corr, figs_stats

def load_csv(folder_path, file):
    path = os.path.join(folder_path, file)
    with open(path, "r") as file:
        lines = file.readlines()
        n_elements = len(lines[0].split(","))
        cols_arr = np.empty((0, n_elements))
        for line in lines:
            cols = [int(i) for i in line.split(",")]
            cols_arr = np.vstack((cols_arr, cols))

    return cols_arr


def chain_delay_plot(data):
    cols_avg = np.mean(data, axis=1)
    cols_mean = np.median(data, axis=1)
    cols_max = np.max(data, axis=1)

    x_axis = np.arange(0, data.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    axes[0].plot(x_axis, cols_avg, label="průměrné zpoždění")
    axes[0].plot(x_axis, cols_mean, label="medián zpoždění")
    axes[0].plot(x_axis, cols_max, label="maximum zpoždění")
    axes[0].set_xlabel("n-tý přístup k archivu")
    axes[0].set_ylabel("zpoždění vůči nejrychlejšímu chainu")
    axes[0].legend()
    axes[0].grid(True)

    for i in range(data.shape[1]):
        axes[1].plot(x_axis, data[:, i], label=f"Chain {i}")
    axes[1].set_xlabel("n-tý přístup k archivu")
    axes[1].set_ylabel("zpoždění vůči nejrychlejšímu chainu")
    axes[1].legend(ncol=2, loc="upper left")
    axes[1].grid(True)


def plot_likelihood(idata: az.InferenceData, cutoff=-100):
    draws = idata.posterior.sizes["draw"]
    chains = idata.posterior.sizes["chain"]
    likelihoods = idata["sample_stats"]["likelihood"]
    likelihoods = np.clip(likelihoods, cutoff, None)
    x_axis = np.arange(0, draws)

    figs = []

    fig_progression, axes_progression = plt.subplots(2, 1, figsize=(16, 9))
    fig_progression.suptitle(f"Vývoj log-likelihood v čase (hodnoty pod {cutoff} oříznuty)")
    axes_progression[0].set_xlabel("Iterace v chainu")
    axes_progression[0].set_ylabel("Log-likelihood")
    for chain in np.arange(0, chains):
        axes_progression[0].plot(x_axis, likelihoods[chain, :], label=f"Chain {chain}")

    likelihood_mean = np.mean(likelihoods, axis=0)
    likelihood_median = np.median(likelihoods, axis=0)
    likelihood_min = np.min(likelihoods, axis=0)
    axes_progression[0].legend(ncol=2, loc="lower right")
    axes_progression[0].grid(True)

    axes_progression[1].set_xlabel("Iterace v chainu")
    axes_progression[1].set_ylabel("Log-likelihood")
    axes_progression[1].plot(x_axis, likelihood_mean, label="Průměrná log-likelihood")
    axes_progression[1].plot(x_axis, likelihood_median, label="Medián log-likelihood")
    axes_progression[1].plot(x_axis, likelihood_min, label="Minimum log-likelihood")
    axes_progression[1].legend(ncol=2, loc="lower right")
    axes_progression[1].grid(True)

    figs += [fig_progression]

    fig_hist, axes_hist = plt.subplots(figsize=(16, 9))
    fig_hist.suptitle(f"Histogram log-likelihood (hodnoty pod {cutoff} oříznuty)")
    axes_hist.set_xlabel("Log-likelihood")
    axes_hist.set_ylabel("Počet")
    axes_hist.hist(likelihoods.values.flatten(), bins=100)

    figs += [fig_hist]

    return figs


def generate_all_flow_plots(idata: az.InferenceData, folder, config=None):
    az.plot_pair(idata, kind="kde")
    save_plot("pair_plot.pdf", folder_path=folder)
    az.plot_trace(idata)
    plt.tight_layout()
    save_plot("trace_plot.pdf", folder_path=folder)
    likelihood_plots = plot_likelihood(idata)
    save_plots_pdf_pages("likelihood_plot.pdf", folder_path=folder, figs=likelihood_plots)

    corr_plot, stats_plots = data_window_plots(idata, 100)
    save_plot("corr_progression_plot.pdf", folder_path=folder, fig=corr_plot)
    save_plots_pdf_pages("stats_progression_plot.pdf", folder_path=folder, figs=stats_plots)


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


    axes = az.plot_posterior(idata, grid=[4, 2])
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
    idata_name = "2x2000.idata"
    folder_path = os.path.join(ROOT_DIR, "data", "dataset3", "2x2000_DREAMZ")
    idata = read_idata_from_file(idata_name, folder_path)
    idata = idata.sel(draw=slice(100, None))
    generate_all_flow_plots(idata,folder_path)
