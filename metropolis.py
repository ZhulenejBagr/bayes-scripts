import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import multivariate_normal
from math import log10, pi
import pathlib
import os

def base_path():
    return pathlib.Path(__file__).parent.resolve()

generator = npr.Generator(npr.MT19937())

# Simple metropolis (not MH) implementation for one variable
def norm_pdf(value, mean, std):
    return 1 / (std * np.sqrt(2 * pi)) * np.exp(-1 / 2 * np.power((value - mean) / std, 2))

def mvn_pdf(rnd, mean, cov):
    cov = np.multiply(2 * np.pi, cov)
    factor = np.sqrt(np.linalg.det(cov))
    return multivariate_normal.pdf(rnd, mean, cov) / factor


def metropolis(samples = 10000, n_cores = 4, n_chains = 4, tune = 3000):
    # set up result array
    total_samples = samples + tune
    variables = ["U", "G"]
    samples = {variable: [None] * total_samples for variable in variables}
    accepted = 0
    rejected = 0
    
    # prior
    prior_mean = np.array([7, 5])
    prior_sigma = np.array([[4, -2], [-2, 4]])
    prior_pdf = lambda rnd: mvn_pdf(rnd, prior_mean, prior_sigma)
    prior_candidate = lambda mean: generator.multivariate_normal([mean[0], mean[1]], prior_sigma)
    # posterior
    posterior_std = 2e-4
    posterior_operator = lambda u: -1 / 80 * (3 / np.exp(u[0]) + 1 / np.exp(u[1]))
    observed_value = -1e-3 
    posterior_pdf = lambda rnd: norm_pdf(rnd - observed_value, 0, posterior_std)

    # sampling process
    g = 0
    u = prior_mean
    for iteration in range(total_samples):
        # get U candidate sample
        u_candidate = prior_candidate(u)
        g_candidate = posterior_operator(u_candidate)

        # accept/reject candidates
        u_current_probability = prior_pdf(u)
        u_candidate_probability = prior_pdf(u_candidate)

        g_candidate_probability = posterior_pdf(g_candidate)
        g_current_probability = posterior_pdf(g)
        threshold = min([1, g_candidate_probability / g_current_probability * u_candidate_probability / u_current_probability])

        random_probability = npr.uniform()

        if random_probability < threshold:
            samples["U"][iteration] = u_candidate
            samples["G"][iteration] = g_candidate_probability * u_candidate_probability
            u = u_candidate
            g = g_candidate
            accepted += 1
        else:
            samples["U"][iteration] = u
            samples["G"][iteration] = g_current_probability * u_current_probability
            rejected += 1
    print((accepted, rejected))
    return samples


if __name__ == "__main__":
    idata = metropolis(samples=20000)
    x = [d[0] for d in idata["U"]]
    y = [d[1] for d in idata["U"]]
    density = idata["G"]
    log_density = [log10(g) for g in density]
    #maximum = max(log_density)
    #log_density = [g / max(log_density) for g in log_density]
    #maximum = max(log_density)
    #minimum = min(log_density)
    maximum = max(density)
    minimum = min(density)
    colormap = plt.get_cmap('Greys')
    norm = Normalize(vmin=minimum, vmax=maximum)
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.scatter(x, y, c=density, cmap=colormap, norm=norm, s=6)
    plt.colorbar(sm, label="Hustota pravděpodobnosti", ax=plt.gca())
    plt.title("Posterior graf")
    plt.savefig(os.path.join(base_path(), "Graphs", "Metropolis.pdf"), format="pdf", dpi=300)
 