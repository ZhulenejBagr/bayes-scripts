import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm

# Parameters
mu = 5
sigma = 2
a = 2
b = 9
size = 10000  # Sample size

# Generate random variable x with normal distribution
x = np.random.normal(mu, sigma, size)

# Generate random variable y with truncated normal distribution
# Define the lower and upper bounds for truncation
lower_bound = (a - mu) / sigma
upper_bound = (b - mu) / sigma

########################################################################################################################
# transform x~N(mu,sigma^2) to y~TN(mu,sigma^2,a,b)
phi_a = norm.cdf(lower_bound)
phi_b = norm.cdf(upper_bound)
phi_x = norm.cdf(x, loc=mu, scale=sigma)
y = norm.ppf((phi_b - phi_a)*phi_x + phi_a)*sigma + mu
########################################################################################################################

# Plot histograms
plt.figure(figsize=(10, 5))

# Histogram for x (normal distribution)
plt.subplot(1, 2, 1)
plt.hist(x, bins=30, density=True, color='blue', alpha=0.5, label='x~N(\mu,\sigma^2)')
plt.hist(y, bins=30, density=True, color='red', alpha=0.5, label='y~TN(\mu,\sigma^2,a,b)')
plt.title('Histogram of x (Normal Distribution)')
plt.xlabel('Value')
plt.ylabel('Frequency')
# Plot the probability density function (pdf) for comparison
xmin, xmax = plt.xlim()
x_values = np.linspace(xmin, xmax, 100)
plt.plot(x_values, norm.pdf(x_values, mu, sigma), 'b--')
# y_values = np.linspace(a, b, 100)
plt.plot(x_values, truncnorm.pdf(x_values, lower_bound, upper_bound, loc=mu, scale=sigma), 'r--')

plt.legend()
plt.grid(True)

# Generate truncated normal distribution
z = truncnorm.rvs(lower_bound, upper_bound, loc=mu, scale=sigma, size=size)
# Histogram for y (truncated normal distribution)
plt.subplot(1, 2, 2)
plt.hist(z, bins=30, density=True, color='green', alpha=0.7)
plt.title('Histogram of z (Truncated Normal Distribution)')
plt.xlabel('Value')
plt.ylabel('Frequency')
# Plot the probability density function (pdf) for comparison
z_values = np.linspace(a, b, 100)
plt.plot(z_values, truncnorm.pdf(z_values, lower_bound, upper_bound, loc=mu, scale=sigma), 'r--')

plt.tight_layout()
plt.show()
