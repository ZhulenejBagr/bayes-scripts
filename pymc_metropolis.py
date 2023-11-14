import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import multivariate_normal
import pathlib
import os
import pickle

generator = np.random.default_rng(222)

def save_plot(folder_path, filename):
    # if path doesn't exist, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, filename), format="pdf", dpi=300)


def base_path():
    return pathlib.Path(__file__).parent.resolve()

def graphs_path():
    return os.path.join(base_path(), "graphs")

def idata_path():
    return os.path.join(base_path(), "idata")

def save_idata_to_file(idata, folder_path, filename):
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
        

def read_idata_from_file(folder_path, filename):
    path = os.path.join(folder_path, filename)
    try:
        with open(path, "rb") as file: 
            idata = pickle.load(file=file)
    except:
        print("Error reading idata file")

    return idata


def metropolis(samples=10000, n_cores=4, n_chains=4, tune=3000, prior_mean=[5, 3], prior_cov=[[4, -2], [-2, 4]]):
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
        idata = pm.sample(draws=samples, tune=tune, step=pm.Metropolis(), chains=n_chains, cores=n_cores, random_seed=generator)
        # add posterior log likelyhood data
        likelihood = pm.compute_log_likelihood(idata, extend_inferencedata=True)
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


def custom_pair_plot(idata, filename="posterior_plot.pdf", folder_path=graphs_path()):
    # get values from inference data
    x_data = idata["posterior"]["U"][:, :, 0]
    y_data = idata["posterior"]["U"][:, :, 1]
    log_likelihood = idata["log_likelihood"]["G"]
    prior = idata["prior"]["U"]
    prior_likelihood = idata["prior"]["likelihood"] 
    
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

    # add colorbars and legend
    plt.legend('upper left')
    fig.colorbar(posterior_sm, label="Posterior PDF", cax=ax[0])
    fig.colorbar(prior_sm, label="Prior PDF", cax=ax[2])

    # save plot to file
    save_plot(folder_path=folder_path, filename=filename)

def plot_acceptance(idata, target_acceptance=0.8, log=False, folder_path=graphs_path(), filename="acceptance_plot.pdf"):
    acceptance = idata["sample_stats"]["accept"].to_numpy()
    n_chains = acceptance.shape[0]
    n_samples = acceptance.shape[1]

    fig, ax = plt.subplots(1, 1)
    samples = range(1, n_samples + 1)
    if log:
        ax.set_yscale('log')
    ax.axhline(target_acceptance)
    for chain in range(0, n_chains):
        ax.plot(samples, acceptance[chain, :])
    
    save_plot(folder_path=folder_path, filename=filename)


if __name__ == "__main__":
    prior_mean = [5, 3]
    #idata = metropolis(samples=10000, tune=5000, n_cores=4, n_chains=4, prior_mean=prior_mean)
    idata = read_idata_from_file(idata_path(), "sample_idata")
    save_idata_to_file(idata=idata, folder_path=idata_path(), filename="sample_idata")
    custom_pair_plot(idata=idata)
    plot_acceptance(idata=idata)
    plot_acceptance(idata=idata, log=True, filename="acceptance_plot_log.pdf")



    gs = 40
    az.plot_pair(
        data=idata, 
        kind="hexbin", 
        #marginals=True, 
        gridsize=(round(gs * 1.73), gs), 
        hexbin_kwargs={"cmap": "Greys"},
        var_names = ["U", "G_mean"]
        )
    save_plot(graphs_path(), "pair_plot.pdf")
    #az.plot_autocorr(data)
    #az.plot_rank(data=data)
    print(az.summary(data=idata))    
    #az.plot_trace(data=idata, compact=True)
    #az.plot_ppc(data=data)

