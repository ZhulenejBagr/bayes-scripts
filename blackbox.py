# https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html
# https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.CustomDist.html
import numpy as np

external_generator = np.random.Generator(np.random.MT19937())


def sample(mu, rng=None, size=None):
    sigma = sigma=np.array([[4, -2], [-2, 4]])
    value = external_generator.multivariate_normal(mean=mu, cov=sigma, size=size)
    return value

# norm pdf with m=0
def norm_pdf(value, sigma, mu=0):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * np.power((value - mu) / sigma, 2))

# bivariate normal distribution pdf
def bvn_pdf(value, mu, sigma):
    k = 2
    return (2 * np.pi) ** (k) * np.linalg.det(sigma) ** (-1 / 2) * np.exp(-1 / 2 * np.transpose(value - mu) * np.linalg.inv(sigma) * (value - mu))

def observation_operator(value):
    return -1 / 80 * (3 / np.exp(value[0]) + 1 / np.exp(value[1]))

def log_probability(value, mu):
    sigma = np.array([[4, -2], [-2, 4]])
    noise_sigma = 2e-4
    observed = -1e-3
    #operator = np.multiply(-1 / 80, np.add(3 / np.exp(value[0]),  1 / np.exp(value[1])))
    numerator_operator = observation_operator(value)
    numerator_mu = np.subtract(numerator_operator, observed)
    numerator = norm_pdf(numerator_mu, noise_sigma) * bvn_pdf(value, mu, sigma)

    denominator_operator = observation_operator(mu)
    denominator_mu = np.subtract(denominator_operator, observed)
    denominator = norm_pdf(denominator_mu, sigma) * bvn_pdf(mu, value, sigma)

    #likelihood = likelihood * multivariate_normal(mu, sigma).pdf(value)
    return np.log(np.divide(numerator / denominator))

