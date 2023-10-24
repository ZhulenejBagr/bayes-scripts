import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

generator = npr.Generator(npr.MT19937())

# Simple metropolis (not MH) implementation for one variable

def metropolis(samples = 10000, n_cores = 4, n_chains = 4, tune = 3000):
    # set up result array
    total_samples = samples + tune
    variables = ["U", "G"]
    samples = {variable: [None] * total_samples for variable in variables}
    accepted = 0
    rejected = 0
    
    # prior
    prior_mean = [5, 3]
    prior_sigma = [[4, -2], [-2, 4]]
    prior_pdf = lambda rnd: multivariate_normal.pdf(rnd, prior_mean, prior_sigma)
    prior_candidate = lambda mean: generator.multivariate_normal([mean[0], mean[1]], prior_sigma)
    # posterior
    posterior_std = 2e-4
    posterior_candidate = lambda mean: generator.normal(mean, posterior_std)
    posterior_operator = lambda u: -1 / 80 * (3 / np.exp(u[0]) + 1 / np.exp(u[1]))
    starting_value = -1e-3 
    posterior_pdf = lambda rnd: norm.pdf(rnd, starting_value, posterior_std)

    # sampling process
    u = prior_mean
    g = starting_value
    for iteration in range(total_samples):
        # get U candidate sample
        u_candidate = prior_candidate(u)
        # get G candidate sample
        g_mean = posterior_operator(u_candidate)
        g_candidate = posterior_candidate(g_mean)

        # accept/reject candidates
        u_current_probablity = prior_pdf(u)
        u_candidate_probability = prior_pdf(u_candidate)
        u_threshold = min([1, u_candidate_probability / u_current_probablity])
        u_random_probability = npr.uniform()

        g_current_probablity = posterior_pdf(g)
        g_candidate_probability = posterior_pdf(g_candidate)
        g_threshold = min([1, g_candidate_probability / g_current_probablity])
        g_random_probability = npr.uniform()

        if np.all([
            u_random_probability < u_threshold,
            g_random_probability < g_threshold
        ]):
            samples["U"][iteration] = u_candidate
            samples["G"][iteration] = g_candidate
            u = u_candidate
            g = g_candidate
            accepted += 1
        else:
            samples["U"][iteration] = u
            samples["G"][iteration] = g
            rejected += 1
    print((accepted, rejected))
    return samples


if __name__ == "__main__":
    idata = metropolis(samples=10000)
    x = [d[0] for d in idata["U"]]
    y = [d[1] for d in idata["U"]]
    plt.scatter(x, y)
    plt.show()
 