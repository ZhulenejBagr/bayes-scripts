import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

def metropolis(samples = 10000):
    chains = 1
    tune = 2000
    observed = -1e-3
    sigma = 1e-1
    with pm.Model() as model:
        # reference: tabulka 3.1 v sekci 3.2
        # "f_U" v zadání, aka "prior pdf"
        U = pm.MvNormal('U', [5, 3], [[4, -2],[-2, 4]])
        # "G" v zadání, aka "observation operator"
        G_mean = pm.Deterministic('G_mean', -1 / 80 * (3 / np.exp(U[0]) + 1 / np.exp(U[1])))
        # y a f_Z uplně nevím, jak zakomponovat do modelu
        # současný nápad je použít G jako střední hodnotu pro normální rozdělení
        # u tohoto rozdělení určit nějaký rozptyl (podle f_Z?) a nastavit observed na y
        # G by pak mohlo být f_(U|Y) (u|y), neboli ve figure 3.1d?
        G = pm.Normal('G', mu = G_mean, sigma = sigma, observed = observed)

        trace = pm.sample(samples, tune = tune, step = pm.Metropolis(), chains = chains)
    
    return trace


if __name__ == "__main__":
    trace = metropolis()
    az.plot_pair(trace, kind="hexbin", marginals=True)
    plt.show()