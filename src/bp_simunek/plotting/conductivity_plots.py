import os
from typing import List

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import numpy.typing as npt
from arviz import InferenceData
import arviz as az
from scipy.stats import multivariate_normal, gaussian_kde, norm

import src.bp_simunek.samplers.idata_tools as tools
from src.bp_simunek.plotting.plotting_tools import graphs_path, save_plot
from src.bp_simunek.samplers.idata_tools import read_idata_from_file
from src.bp_simunek.plotting.ds_plots import density_scatter

def plot_pair_custom(
        idata: InferenceData,
        filename: str = "posterior_prior_pair_plot.png",
        folder_path: str = graphs_path(),
        explicit_data: dict = None,
        plot_prior: bool = True) -> None:

    # get values from inference data
    # if explicit data is available, use it
    if explicit_data is not None:
        x_data = explicit_data["U_0"] if "U_0" in explicit_data else idata["posterior"]["U"][:, :, 0]
        y_data = explicit_data["U_1"] if "U_1" in explicit_data else idata["posterior"]["U"][:, :, 1]
        log_likelihood = explicit_data["log_likelihood"] if "log_likelihood" in explicit_data else idata["log_likelihood"]["G"]
    else:
        x_data = idata["posterior"]["U"][:, :, 0]
        y_data = idata["posterior"]["U"][:, :, 1]
        log_likelihood = idata["log_likelihood"]["G"]

    if plot_prior:
        prior = idata["prior"]["U"]
        prior_likelihood = idata["prior"]["likelihood"]
        x_prior = prior[:, :, 0]
        y_prior = prior[:, :, 1]

    # change to regular likelyhood
    #log_likelihood = np.exp(log_likelihood)

    # init plot figure
    wrl = [1, 14, 1]
    fig, ax = plt.subplots(nrows=1, ncols=3, gridspec_kw={'width_ratios': wrl})
    fig.set_figwidth(16)
    fig.set_figheight(9)

    # posterior colormap
    posterior_colormap = plt.get_cmap('binary')
    posterior_norm = Normalize(vmin=np.min(log_likelihood), vmax=np.max(log_likelihood))
    posterior_sm = ScalarMappable(cmap=posterior_colormap, norm=posterior_norm)
    posterior_sm.set_array([])

    if plot_prior:
        # prior colormap
        prior_colormap = plt.get_cmap('Reds')
        prior_norm = Normalize(vmin=np.min(prior_likelihood), vmax=np.max(prior_likelihood))
        prior_sm = ScalarMappable(cmap=prior_colormap, norm=prior_norm)
        prior_sm.set_array([])

        # plot prior
        ax[1].scatter(
            x_prior,
            y_prior,
            c=prior_likelihood,
            cmap=prior_colormap,
            label="Prior",
            s=6,
            alpha=0.15
        )

    # plot posterior
    ax[1].scatter(
        x_data,
        y_data,
        c=log_likelihood,
        cmap=posterior_colormap,
        label="Posterior",
        s=6,
        alpha=0.05
    )

    # fix axis limits
    ax[1].set_xlim([-5, 15])
    ax[1].set_ylim([-7.5, 12.5])

    # add colorbars and legend
    plt.legend('upper left')
    fig.colorbar(posterior_sm, label="Posterior PDF", cax=ax[0])
    if plot_prior:
        fig.colorbar(prior_sm, label="Prior PDF", cax=ax[2])

    # save plot to file
    save_plot(folder_path=folder_path, filename=filename)

    # close figure
    plt.close()

def plot_acceptance(
        idata: InferenceData,
        target_acceptance: float = 0.8,
        log: bool = False,
        folder_path: str = graphs_path(),
        filename: str = "acceptance_plot.pdf") -> None:
    
    acceptance = idata["sample_stats"]["accept"].to_numpy()
    n_chains = acceptance.shape[0]
    n_samples = acceptance.shape[1]

    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(16)
    fig.set_figheight(9)
    samples = range(1, n_samples + 1)
    if log:
        ax.set_yscale('log')
    ax.axhline(target_acceptance)
    for chain in range(0, n_chains):
        ax.plot(samples, acceptance[chain, :])

    save_plot(folder_path=folder_path, filename=filename)

    # close figure
    plt.close()

def plot_posterior_with_prior(
        idata: InferenceData,
        filename: str = "posterior_prior_plot.pdf",
        folder_path: str = graphs_path(),
        merge_chains: bool = False,
        single_plot: bool = False,
        analytic: bool = True,
        analytic_mean: npt.NDArray = np.array([5, 3]),
        analytic_cov: npt.NDArray = np.array([[4, -2], [-2, 4]])) -> None:
    posterior_data = idata["posterior"]["U"].to_numpy()
    prior_data = idata["prior"]["U"].to_numpy()

    n_chains = posterior_data.shape[0]

    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    colors = ["blue", "red"]
    prior_colors = ["darkblue", "darkred"]

    if single_plot:
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim([-7.5, 15])
    else:
        fig, ax = plt.subplots(1, 2)
        ax[0].set_xlim([-5, 15])
        ax[1].set_xlim([-7.5, 12.5])

    fig.set_figwidth(16)
    fig.set_figheight(9)
    fig.suptitle("Posterior and prior density")

    for i in range(2):
        lower_bound = np.min((np.min(posterior_data[:, :, i], axis=None), np.min(prior_data[:, :, i], axis=None)))
        upper_bound = np.max((np.max(posterior_data[:, :, i], axis=None), np.max(prior_data[:, :, i], axis=None)))
        linspace = np.linspace(lower_bound, upper_bound, 250)

        if not single_plot:
            ax[i].set_title(f"U[{i}]")
        else:
            ax.set_title("U")

        if not merge_chains:
            for chain in range(0, n_chains):
                density = gaussian_kde(posterior_data[chain, :, i])
                if not single_plot:
                    ax[i].plot(linspace, density(linspace), linestyle=linestyles[chain], color="blue", label=f"Posterior: Chain {chain}")
                else:
                    ax.plot(linspace, density(linspace), linestyle=linestyles[chain], color=colors[i], label=f"Posterior[{i}]: Chain {chain}")
        else:
            merged = posterior_data[:, :, i].flatten()
            density = gaussian_kde(merged)
            if not single_plot:
                ax[i].plot(linspace, density(linspace), color="blue", label="Posterior")
            else:
                ax.plot(linspace, density(linspace), color=colors[i], label=f"Posterior[{i}]")

        if not analytic:
            density = gaussian_kde(prior_data[:, :, i])
            if not single_plot:
                ax[i].plot(linspace, density(linspace), color="darkblue", label="Prior")
        else:
            linspace0 = np.linspace(-5, 15, 250)
            linspace1 = np.linspace(-7.5, 12.5, 250)
            linspaces = [linspace0, linspace1]
            x, y = np.meshgrid(linspace0, linspace1)
            grid = np.dstack((x, y))
            pdf_values = multivariate_normal(mean=analytic_mean, cov=analytic_cov).pdf(grid)
            density = np.sum(pdf_values, axis=1-i)
            # normalize
            density = np.divide(density, 12.5)
            if not single_plot:
                ax[i].plot(linspaces[i], density, color="darkblue", label="Prior")
            else:
                ax.plot(linspaces[i], density, color=prior_colors[i], label=f"Prior[{i}]")
        
        if single_plot and not analytic:
            ax.plot(linspace, density(linspace), color=prior_colors[i], label=f"Prior[{i}]")
        
        if not single_plot:
            ax[i].legend()
        else:
            ax.legend()

    # save plot
    save_plot(folder_path=folder_path, filename=filename)

    # close figure
    plt.close()

def plot_autocorr(
        idata: InferenceData,
        filename: str = "autocorr.pdf",
        folder_path: str = graphs_path()) -> None:
    az.plot_autocorr(idata)
    save_plot(filename=filename, folder_path=folder_path)
    # close figure
    plt.close()

def plot_rank(
        idata: InferenceData,
        filename: str = "rank.pdf",
        folder_path: str = graphs_path()) -> None:
    az.plot_rank(idata)
    save_plot(filename=filename, folder_path=folder_path)
    # close figure
    plt.close()

def plot_trace(
        idata: InferenceData,
        filename: str = "trace.pdf",
        folder_path: str = graphs_path()) -> None:
    
    az.plot_trace(idata, figsize=[16, 9])
    save_plot(filename=filename, folder_path=folder_path)
    # close figure
    plt.close()

def plot_all(
        idata: InferenceData,
        folder_path: str = graphs_path()) -> None:

    plot_pair_custom(idata, folder_path=folder_path)
    #try:
    #    plot_acceptance(idata, folder_path=folder_path)
    #    plot_acceptance(idata, log=True, filename="acceptance_plot_log.pdf", folder_path=folder_path)
    #except:
    #    print("Unable to plot acceptance")
    plot_posterior_with_prior(idata, folder_path=folder_path, analytic=False, merge_chains=True)
    plot_posterior_with_prior(idata, folder_path=folder_path, analytic=True, merge_chains=True, filename="posterior_prior_plot_analytic.pdf")
    plot_posterior_with_prior(idata, folder_path=folder_path, analytic=True, merge_chains=True, single_plot=True, filename="posterior_prior_plot_single.pdf")
    plot_autocorr(idata, folder_path=folder_path)
    plot_rank(idata, folder_path=folder_path)
    plot_trace(idata, folder_path=folder_path)

def plot_posterior_with_prior_compare(
        idata_list: List[InferenceData],
        names: List[str],
        prior_data,
        filename: str = "posterior_prior_plot_compare.pdf",
        folder_path: str = graphs_path(),
        merge_chains: bool = False,
        analytic: bool = True,
        analytic_mean: npt.NDArray = np.array([5, 3]),
        analytic_cov: npt.NDArray = np.array([[4, -2], [-2, 4]])) -> None:

    fig, ax = plt.subplots(2, 2)
    fig.set_figwidth(16)
    fig.set_figheight(9)
    for j in range(4):
        idata = idata_list[j]
        current_ax = ax[j // 2][j % 2]
        current_ax.set_xlabel("Hodnota parametru")
        current_ax.set_ylabel("Pravděpodobnostní hustota")
        if "U" in idata["posterior"]:
            posterior_data = idata["posterior"]["U"].to_numpy()
        else:
            xdata = idata["posterior"]["U_0"].to_numpy()
            ydata = idata["posterior"]["U_1"].to_numpy()
            posterior_data = np.stack([xdata, ydata], axis=-1)

        print(posterior_data.shape)
        n_chains = posterior_data.shape[0]

        linestyles = ["solid", "dotted", "dashed", "dashdot"]
        colors = ["blue", "red"]
        prior_colors = ["darkblue", "darkred"]

        current_ax.set_xlim([-7.5, 15])
        current_ax.set_ylim([0, 1])

        fig.suptitle("Porovnání implementací")

        for i in range(2):
            lower_bound = np.min((np.min(posterior_data[:, :, i], axis=None), np.min(prior_data[:, :, i], axis=None)))
            upper_bound = np.max((np.max(posterior_data[:, :, i], axis=None), np.max(prior_data[:, :, i], axis=None)))
            linspace = np.linspace(lower_bound, upper_bound, 250)

            current_ax.set_title(f"{names[j]}")

            if not merge_chains:
                for chain in range(0, n_chains):
                    density = gaussian_kde(posterior_data[chain, :, i])
                    current_ax.plot(linspace, density(linspace), linestyle=linestyles[chain], color=colors[i], label=f"Posterior[{i}]: Chain {chain}")
            else:
                merged = posterior_data[:, :, i].flatten()
                density = gaussian_kde(merged)
                current_ax.plot(linspace, density(linspace), color=colors[i], label=f"Posterior[{i}]")

            if analytic:
                linspace0 = np.linspace(-5, 15, 250)
                linspace1 = np.linspace(-7.5, 12.5, 250)
                linspaces = [linspace0, linspace1]
                densities = [
                    norm(loc=analytic_mean[0], scale=np.sqrt(analytic_cov[0, 0])).pdf(linspaces[0]),
                    norm(loc=analytic_mean[1], scale=np.sqrt(analytic_cov[1, 1])).pdf(linspaces[1])
                ]
                current_ax.plot(linspaces[i], densities[i], color=prior_colors[i], label=f"Prior[{i}]")
            
            if not analytic:
                density = gaussian_kde(prior_data[:, :, i])
                current_ax.plot(linspace, density(linspace), color=prior_colors[i], label=f"Prior[{i}]")
        
            current_ax.legend()

    #fig.legend(ncol=2, loc="upper left")
    save_plot(folder_path=folder_path, filename=filename)

    # close figure
    plt.close()

def pair_plot_compare():
    idata_paths_standard = [
        "standard.custom_MH.idata",
        "regular.MH.idata",
        "tinyda.standard.idata",
        "tinyda_randomwalk.standard.idata"
    ]

    idata_paths_offset = [
        "offset.custom_MH.idata",
        "offset.MH.idata",
        "tinyda.offset.idata",
        "tinyda_randomwalk.offset.idata"
    ]

    names = ["PyMC MH", "Custom MH", "tinyDA IS", "tinyDA RW"]
    idata_sets = [idata_paths_standard, idata_paths_offset]
    filenames = ["regular", "offset"]
    axlims = [[(0, 10), (0, 11)], [(0, 15), (0, 16)]]
    for id, dataset in enumerate(idata_sets):
        lim = axlims[id]
        fig, axes = plt.subplots(2, 2)
        fig.suptitle("Porovnání implementací")
        fig.set_figheight(9)
        fig.set_figwidth(16)
        prior = read_idata_from_file(dataset[1])["prior"]["U"]
        prior_x = prior[:, :, 0].to_numpy().reshape(-1)
        prior_y = prior[:, :, 1].to_numpy().reshape(-1)

        for idx, idata_path in enumerate(dataset):
            ax = axes[idx // 2, idx % 2]
            ax.set_xlim(*lim[0])
            ax.set_ylim(*lim[1])
            idata = read_idata_from_file(idata_path)
            if "U_0" in idata["posterior"]:
                x = idata["posterior"]["U_0"]
                y = idata["posterior"]["U_1"]
            else:
                x = idata["posterior"]["U"][:, :, 0]
                y = idata["posterior"]["U"][:, :, 1] 
            x = x.to_numpy().squeeze()
            y = y.to_numpy().squeeze()
            
            if x.ndim > 1:
                x = x.reshape(-1)
            if y.ndim > 1:
                y = y.reshape(-1)

            density_scatter(prior_x, prior_y, ax, is_prior=True)
            density_scatter(x, y, ax, is_prior=False)

            xc = np.linspace(0, lim[0][1], 100000)
            yc = np.log(25 * np.exp(xc) / (2 * np.exp(xc) - 75))
            contour, = ax.plot(xc, yc, color="red")
            ax.legend([contour], ["Naměřená data"])
            ax.set_title(names[idx])
            ax.set_xlabel("Vodivost U_0")
            ax.set_ylabel("Vodivost U_1")
        
        save_plot(f"plot_pair_compare_{filenames[id]}.pdf", fig=fig)
        save_plot(f"plot_pair_compare_{filenames[id]}.png", fig=fig)
def posterior_plot_compare():
    idata_paths_standard = [
        "standard.custom_MH.idata",
        "regular.MH.idata",
        "tinyda.standard.idata",
        "tinyda_randomwalk.standard.idata"
    ]

    idata_paths_offset = [
        "offset.custom_MH.idata",
        "offset.MH.idata",
        "tinyda.offset.idata",
        "tinyda_randomwalk.offset.idata"
    ]

    idata_sets = [idata_paths_standard, idata_paths_offset]
    filenames = ["regular", "offset"]
    axlims = [(0, 11), (0, 16)]
    names = ["PyMC MH", "Custom MH", "tinyDA IS", "tinyDA RW"]
    analytic_means = [np.array([5, 3]), np.array([8, 6])]
    analyic_covs = [
        np.array([[4, -2], [-2, 4]]),
        np.array([[16, -2], [-2, 16]])
    ]

    for id, dataset in enumerate(idata_sets):
        lim = axlims[id]
        fig, axes = plt.subplots(2, 2)
        fig.set_figheight(9)
        fig.set_figwidth(16)
        prior = read_idata_from_file(dataset[1])["prior"]["U"]
        idatas = [read_idata_from_file(dataset[i]) for i in range(len(dataset))]
        plot_posterior_with_prior_compare(idatas, names, prior, f"posterior_compare_{filenames[id]}.pdf", merge_chains=True, analytic_mean=analytic_means[id], analytic_cov=analyic_covs[id])

if __name__ == "__main__":
    pair_plot_compare()
    posterior_plot_compare()