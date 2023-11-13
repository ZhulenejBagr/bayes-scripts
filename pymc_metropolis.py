import pymc as pm
import numpy as np
import numpy.random as npr
import arviz as az
import matplotlib.pyplot as plt
import timeit as ti
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import multivariate_normal
import pathlib
import os
generator = np.random.default_rng(222)

def save_plot(folder_path, filename):
    # if path doesn't exist, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, filename), format="pdf", dpi=300)


def base_path():
    return pathlib.Path(__file__).parent.resolve()

def graphs_path():
    return os.path.join(base_path(), "Graphs")

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

def prior_samples(samples=10000, mean=[5,3], cov=[[4,-2],[-2,4]]):
    values = generator.multivariate_normal(mean=mean, cov=cov, size=samples, check_valid='warn')
    adj_cov = np.multiply(2 * np.pi, cov)
    factor = np.sqrt(np.linalg.det(adj_cov))
    likelyhoods = multivariate_normal(mean=mean, cov=cov).pdf(values)
    likelyhoods = np.divide(likelyhoods, factor)
    return {"samples": values, "likelyhood": likelyhoods}



def custom_pair_plot(idata):
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
    save_plot(graphs_path(), "custom_pair_plot.pdf")




if __name__ == "__main__":
    prior_mean = [5, 3]
    idata = metropolis(samples=10000, tune=5000, n_cores=4, n_chains=4, prior_mean=prior_mean)
    prior_data = prior_samples(mean=prior_mean)
    custom_pair_plot(idata=idata)

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

