import pathlib
import os
import pickle
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import multivariate_normal, gaussian_kde
import blackbox

generator = np.random.default_rng(222)

def base_path():
    return pathlib.Path(__file__).parent.resolve()

def graphs_path():
    return os.path.join(base_path(), "graphs")

def idata_path():
    return os.path.join(base_path(), "idata")

def save_plot(filename, folder_path=graphs_path()):
    # if path doesn't exist, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, filename), dpi=300)

def save_idata_to_file(idata, filename, folder_path=idata_path()):
    # if path doesn't exist, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    path = os.path.join(folder_path, filename)

    if os.path.exists(path=path):
        with open(path, "wb") as file:
            pickle.dump(obj=idata, file=file)
    else:
        with open(path, "ab") as file:
            pickle.dump(obj=idata, file=file)
        

def read_idata_from_file(filename, folder_path=idata_path()):
    path = os.path.join(folder_path, filename)
    try:
        with open(path, "rb") as file: 
            idata = pickle.load(file=file)
    except:
        print("Error reading idata file")

    return idata


def sample_regular(
        samples=10000,
        n_cores=4, n_chains=4,
        tune=3000,
        prior_mean=np.array([5, 3]),
        prior_cov=np.array([[4, -2], [-2, 4]]),
        step=pm.Metropolis):
    """
    Sample using a directly defined model with PYMC functions
    """
    observed = -1e-3
    sigma = 2e-4
    with pm.Model() as model:
        # reference: tabulka 3.1 v sekci 3.2
        # "f_U" v zadání, aka "prior pdf"
        U = pm.MvNormal('U', prior_mean, prior_cov)
        # "G" v zadání, aka "observation operator"
        G_mean = pm.Deterministic('G_mean', -1 / 80 * (3 / np.exp(U[0]) + 1 / np.exp(U[1])))
        # noise function
        G = pm.Normal('G', mu=G_mean, sigma=sigma, observed=observed)
        # run sampling algorithm for posterior
        idata = pm.sample(draws=samples, tune=tune, step=step(), chains=n_chains, cores=n_cores, random_seed=generator)
        # add posterior log likelyhood data
        pm.compute_log_likelihood(idata, extend_inferencedata=True)
        # add prior samples
        prior = pm.sample_prior_predictive(samples=samples*n_chains, var_names=["U"], random_seed=generator)
        idata.extend(prior)
        # add prior likelyhood
        prior_np = idata["prior"]["U"].to_numpy().reshape((-1, 2))
        factor = np.sqrt(np.linalg.det(np.multiply(2 * np.pi, prior_cov)))
        prior_likelyhood = multivariate_normal(mean=prior_mean, cov=prior_cov).pdf(prior_np)
        prior_likelyhood = np.divide(prior_likelyhood, factor)
        idata["prior"]["likelihood"] = prior_likelyhood

        return idata

def sample_blackbox(
        samples=10000,
        n_cores=4,
        n_chains=4,
        tune=3000,
        prior_mean=np.array([5, 3]),
        step=pm.Metropolis,
        ):
    """
    Sample via a blackbox function and pytensor wrapper
    """
    observed = -1e-3

    with pm.Model() as model:
        pm.CustomDist("U", prior_mean, logp=blackbox.log_probability, random=blackbox.sample, ndim_supp=1, ndims_params=(1,))
        # run sampling algorithm for posterior
        idata = pm.sample(draws=samples, tune=tune, step=step(), chains=n_chains, cores=n_cores, random_seed=generator, initvals={"U": prior_mean})
        # add posterior log likelyhood data
        pm.compute_log_likelihood(idata, extend_inferencedata=True)
        # add prior samples
        #prior = pm.sample_prior_predictive(samples=samples*n_chains, var_names=["U"], random_seed=generator)
        #idata.extend(prior)
        # add prior likelyhood
        #prior_np = idata["prior"]["U"].to_numpy().reshape((-1, 2))
        #factor = np.sqrt(np.linalg.det(np.multiply(2 * np.pi, prior_cov)))
        #prior_likelyhood = multivariate_normal(mean=prior_mean, cov=prior_cov).pdf(prior_np)
        #prior_likelyhood = np.divide(prior_likelyhood, factor)
        #idata["prior"]["likelihood"] = prior_likelyhood

        return idata




def custom_pair_plot(idata, filename="posterior_prior_pair_plot.png", folder_path=graphs_path()):
    # get values from inference data
    x_data = idata["posterior"]["U"][:, :, 0]
    y_data = idata["posterior"]["U"][:, :, 1]
    log_likelihood = idata["log_likelihood"]["G"]
    prior = idata["prior"]["U"]
    prior_likelihood = idata["prior"]["likelihood"]

    # change to regular likelyhood
    #log_likelihood = np.exp(log_likelihood)

    # prior data
    x_prior = prior[:, :, 0]
    y_prior = prior[:, :, 1]

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
    fig.colorbar(prior_sm, label="Prior PDF", cax=ax[2])

    # save plot to file
    save_plot(folder_path=folder_path, filename=filename)

    # close figure
    plt.close()

def plot_acceptance(idata, target_acceptance=0.8, log=False, folder_path=graphs_path(), filename="acceptance_plot.pdf"):
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
        idata,
        filename="posterior_prior_plot.pdf",
        folder_path=graphs_path(),
        merge_chains=False,
        single_plot=False,
        analytic=True,
        analytic_mean=np.array([5, 3]),
        analytic_cov=np.array([[4, -2], [-2, 4]])):
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

    #fig.legend(ncol=2, loc="upper left")
    save_plot(folder_path=folder_path, filename=filename)

    # close figure
    plt.close()

def plot_autocorr(idata, filename="autocorr.pdf", folder_path=graphs_path()):
    az.plot_autocorr(idata)
    save_plot(filename=filename, folder_path=folder_path)
    # close figure
    plt.close()

def plot_rank(idata, filename="rank.pdf", folder_path=graphs_path()):
    az.plot_rank(idata)
    save_plot(filename=filename, folder_path=folder_path)
    # close figure
    plt.close()

def plot_trace(idata, filename="trace.pdf", folder_path=graphs_path()):
    az.plot_trace(idata, figsize=[16, 9])
    save_plot(filename=filename, folder_path=folder_path)
    # close figure
    plt.close()

def plot_all(idata, folder_path=graphs_path()):
    custom_pair_plot(idata, folder_path=folder_path)
    try:
        plot_acceptance(idata, folder_path=folder_path)
        plot_acceptance(idata, log=True, filename="acceptance_plot_log.pdf", folder_path=folder_path)
    except:
        print("Unable to plot acceptance")
    plot_posterior_with_prior(idata, folder_path=folder_path, analytic=False, merge_chains=True)
    plot_posterior_with_prior(idata, folder_path=folder_path, analytic=True, merge_chains=True, filename="posterior_prior_plot_analytic.pdf")
    plot_posterior_with_prior(idata, folder_path=folder_path, analytic=True, merge_chains=True, single_plot=True, filename="posterior_prior_plot_single.pdf")
    plot_autocorr(idata, folder_path=folder_path)
    plot_rank(idata, folder_path=folder_path)
    plot_trace(idata, folder_path=folder_path)

def generate_idata_sets(
        prior_mean=np.array([5, 3]),
        prior_cov=np.array([[4, -2], [-2, 4]]),
        prefix="regular",
        samples=10000,
        tune=5000):
    methods = [pm.Metropolis, pm.HamiltonianMC, pm.NUTS, pm.DEMetropolisZ]
    method_acronyms = ["MH", "HMC", "NUTS", "DEMZ"]
    for method, acronym in zip(methods, method_acronyms):
        idata = sample_regular(step=method, samples=samples, tune=tune, n_cores=4, n_chains=4, prior_mean=prior_mean, prior_cov=prior_cov)
        save_idata_to_file(idata, filename=f"{prefix}.{acronym}.idata")

def generate_regular_idata_sets():
    prior_mean = np.array([5, 3])
    prior_cov = np.array([[4, -2], [-2, 4]])
    return generate_idata_sets(prior_mean=prior_mean, prior_cov=prior_cov)

def generate_offset_idata_sets():
    prior_mean = np.array([8, 6])
    prior_cov = np.array([[16, -2], [-2, 16]])
    return generate_idata_sets(prior_mean=prior_mean, prior_cov=prior_cov, prefix="offset")

def plot_idata_sets(prefix="regular"):
    methods = ["NUTS", "DEMZ", "HMC", "MH"]
    idata_paths = [f"{prefix}.{method}.idata" for method in methods]
    for index, path in enumerate(idata_paths):
        idata = read_idata_from_file(filename=path)
        plot_all(idata, folder_path=os.path.join(graphs_path(), prefix, methods[index]))

def plot_posterior_with_prior_compare(
        idata_list,
        filename="posterior_prior_plot_compare.pdf",
        folder_path=graphs_path(),
        merge_chains=False,
        analytic=True,
        analytic_mean=np.array([5, 3]),
        analytic_cov=np.array([[4, -2], [-2, 4]])):
    fig, ax = plt.subplots(2, 2)
    names = ["MH", "Custom MH", "DEMZ", "NUTS"]
    fig.set_figwidth(16)
    fig.set_figheight(9)
    for j in range(4):
        idata = idata_list[j]
        current_ax = ax[j // 2][j % 2]
        posterior_data = idata["posterior"]["U"].to_numpy()
        prior_data = idata["prior"]["U"].to_numpy()

        n_chains = posterior_data.shape[0]

        linestyles = ["solid", "dotted", "dashed", "dashdot"]
        colors = ["blue", "red"]
        prior_colors = ["darkblue", "darkred"]

        current_ax.set_xlim([-7.5, 15])

        fig.suptitle("Porovnání metod vzorkování")

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
                x, y = np.meshgrid(linspace0, linspace1)
                grid = np.dstack((x, y))
                pdf_values = multivariate_normal(mean=analytic_mean, cov=analytic_cov).pdf(grid)
                density = np.sum(pdf_values, axis=1-i)
                # normalize
                density = np.divide(density, 12.5)
                current_ax.plot(linspaces[i], density, color=prior_colors[i], label=f"Prior[{i}]")
            
            if not analytic:
                current_ax.plot(linspace, density(linspace), color=prior_colors[i], label=f"Prior[{i}]")
        
            current_ax.legend()

    #fig.legend(ncol=2, loc="upper left")
    save_plot(folder_path=folder_path, filename=filename)

    # close figure
    plt.close()

def compare_posterior_with_prior():
    idata_names = ["MH", "custom_MH", "DEMZ", "NUTS"]
    idata_list = [read_idata_from_file(f"regular.{name}.idata") for name in idata_names]
    plot_posterior_with_prior_compare(idata_list, merge_chains=True, analytic=True)
if __name__ == "__main__":
    #generate_regular_idata_sets()
    #generate_offset_idata_sets()
    idata = sample_blackbox()
    print(az.summary(idata))
    #save_idata_to_file(idata, filename="blackbox.idata")
    #idata = read_idata_from_file("blackbox.idata")
    compare_posterior_with_prior()
    #print(idata)
    #print(idata["posterior"])
    #print(idata["sample_stats"])
    #az.plot_trace(idata, show=True)
    #plot_all(idata)
    #plot_idata_sets()
    #plot_idata_sets(prefix="offset")
