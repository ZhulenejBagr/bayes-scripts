import numpy as np
import numpy.random as npr
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

generator = npr.Generator(npr.MT19937())

# Simple metropolis (not MH) implementation for one variable

# params
n_samples = 5000
burn_in = 1000
std = [0.5, 0.5]
starting_value = -1 * 10**(-3)
# posterior pdf
prior_mean = [5, 3]
prior_sigma = [[4, -2], [-2, 4]]
prior_pdf = lambda rnd: multivariate_normal.pdf(rnd, prior_mean, prior_sigma)
# noise draw (?)
noise_draw = lambda: npr.normal(0, 2 * 10**(-4))
# candidate draw
#candidate_draw = lambda mean: generator.normal(mean, std)
candidate_draw = lambda mean: generator.multivariate_normal([mean[0], mean[1]], [[std[0]**2, 0],[0, std[1]**2]])
# candidate_pdf
candidate_pdf = lambda mean, rnd: norm.pdf(rnd, mean, std)

# results
samples = [None] * n_samples
accepted = 0
rejected = 0


curr_sample = [5, 3]
for iteration in range(n_samples + burn_in):
    # sample
    candidate = candidate_draw(curr_sample)

    # add noise
    candidate += noise_draw()

    # accept/reject
    candidate_probability = prior_pdf(candidate)
    current_probablity = prior_pdf(curr_sample)
    threshold = min(1, candidate_probability / current_probablity)
    random_probability = npr.uniform()
    # accept new sample
    if random_probability < threshold:
        curr_sample = candidate
        accepted += 1
    # reject new sample
    else:
        rejected += 1

    samples[iteration] = curr_sample

# remove burn in
samples = samples[burn_in:]
results = [-1 / 80 * (3 * np.exp(-1 * x[0]) + 1 * np.exp(-1 * x[1])) for x in samples]
# plot results
counts, bins = np.histogram(results, bins=100)
axis = plt.subplot()
axis.hist(results, bins=bins)
plt.show()
input()  