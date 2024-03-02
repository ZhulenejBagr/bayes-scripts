import numpy as np
import numpy.random as npr
from scipy.stats import multivariate_normal, norm
import arviz as az

from src.bp_simunek.samplers.idata_tools import save_idata_to_file

generator = npr.Generator(npr.MT19937())

# Simple metropolis-hastings implementation
def norm_pdf(value, mean, std):
    return norm(loc=mean, scale=std).pdf(value)

def mvn_pdf(rnd, mean, cov):
    return multivariate_normal.pdf(rnd, mean, cov)

def sample_prior(samples=10000, mean=np.array([5,3]), cov=np.array([[4,-2],[-2,4]])):
    values = generator.multivariate_normal(mean=mean, cov=cov, size=samples, check_valid='warn')
    #adj_cov = np.multiply(2 * np.pi, cov)
    #factor = np.sqrt(np.linalg.det(adj_cov))

    likelihoods = np.array(multivariate_normal(mean=mean, cov=cov).pdf(values))
    #likelihoods = np.divide(likelihoods, factor)
    return values, likelihoods

def metropolis(
        samples=10000,
        tune=3000,
        prior_mean=np.array([5, 3]),
        prior_sigma=np.array([[4, -2], [-2, 4]])):
    print(f"Sampling {samples} samples with {tune} tune")
    # set up result array
    total_samples = samples + tune
    sampled = {"U": np.zeros((0, 2)), "G": np.zeros(0)}
    likelihood = np.zeros(0)
    # prior
    prior_pdf = lambda rnd: mvn_pdf(rnd, prior_mean, prior_sigma)
    candidate_sample = lambda mean: generator.multivariate_normal([mean[0], mean[1]], prior_sigma)
    # posterior
    noise_sigma = 2e-4
    noise_operator = lambda u: -1 / 80 * (3 / np.exp(u[0]) + 1 / np.exp(u[1]))
    observed = -1e-3
    noise_pdf = lambda rnd: norm_pdf(rnd - observed, 0, noise_sigma)
    #adj_cov = np.multiply(2 * np.pi, prior_sigma)
    #factor = np.sqrt(np.linalg.det(adj_cov))

    # sampling process
    accepted = 0
    current = prior_mean
    current_operator = noise_operator(current)
    while accepted < total_samples:
        # get U candidate sample
        candidate = candidate_sample(current)
        candidate_operator = noise_operator(candidate)

        # accept/reject candidates
        current_prior = prior_pdf(current)
        candidate_prior = prior_pdf(candidate)

        current_noise = noise_pdf(current_operator)
        candidate_noise = noise_pdf(candidate_operator)
        threshold = min([1, candidate_noise / current_noise * candidate_prior / current_prior])

        random_probability = npr.uniform()

        if random_probability < threshold:
            sampled["U"] = np.concatenate((sampled["U"], candidate.reshape(1, 2)), axis=0)
            sampled["G"] = np.append(sampled["G"], candidate_operator)
            likelihood = np.append(likelihood, np.log(candidate_noise))
            current = candidate
            current_operator = candidate_operator
            accepted += 1
            print(accepted)
        else:
            continue

    # remove burn in from samples
    sampled["U"] = sampled["U"][tune:]
    sampled["G"] = sampled["G"][tune:]
    likelihood = likelihood[tune:]
    #likelihood = np.divide(likelihood, factor)

    # reshape data to match pymc
    sampled["U"] = sampled["U"].reshape((1, -1, 2))
    sampled["G"] = sampled["G"].reshape((1, -1))

    # construct idata from sampled data
    idata = az.convert_to_inference_data({
        "U": sampled["U"],
        "G_mean": sampled["G"]
        })
    # add likelyhood data to main idata
    likelihood_idata = az.convert_to_inference_data({
        "G": likelihood
    }, group="log_likelihood")
    idata.extend(likelihood_idata)

    # add prior data
    prior_samples, prior_likelihood = sample_prior(samples, mean=prior_mean, cov=prior_sigma)
    prior_samples = prior_samples.reshape(1, -1, 2)
    prior_likelihood = prior_likelihood.reshape(1, -1)
    prior_idata = az.convert_to_inference_data({
        "U": prior_samples,
        "likelihood": prior_likelihood
    }, group="prior")
    idata.extend(prior_idata)

    return idata

def sample_standard():
    idata = metropolis(tune=20000, samples=10000)
    save_idata_to_file(idata, filename="standard.custom_MH.idata")

def sample_offset():
    idata = metropolis(
        tune=20000,
        samples=10000,
        prior_mean = np.array([7, 5]),
        prior_sigma = np.array([[12, -6],[-6, 12]]))
    save_idata_to_file(idata, filename="offset.custom_MH.idata")

if __name__ == "__main__":
    sample_offset()
