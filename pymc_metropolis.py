import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import timeit as ti

def metropolis(samples = 10000, n_cores = 4, n_chains = 4):
    seed = np.random.default_rng(222)
    tune = 2000
    observed = -1e-3
    sigma = 2e-4
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

        trace = pm.sample(samples, tune = tune, step = pm.Metropolis(), chains = n_chains, cores = n_cores, random_seed=seed)
    
    return trace


if __name__ == "__main__":
    iters = 3
    t1c1 = ti.timeit('metropolis(n_cores = 1, n_chains = 1)', globals = globals(), number = iters)
    t4c1 = ti.timeit('metropolis(n_cores = 4, n_chains = 1)', globals = globals(), number = iters)
    t16c1 = ti.timeit('metropolis(n_cores = 16, n_chains = 1)', globals = globals(), number = iters)
    t1c4 = ti.timeit('metropolis(n_cores = 1, n_chains = 4)', globals = globals(), number = iters)
    t4c4 = ti.timeit('metropolis(n_cores = 4, n_chains = 4)', globals = globals(), number = iters)
    t16c4 = ti.timeit('metropolis(n_cores = 16, n_chains = 4)', globals = globals(), number = iters)
    t1c16 = ti.timeit('metropolis(n_cores = 1, n_chains = 16)', globals = globals(), number = iters)
    t4c16 = ti.timeit('metropolis(n_cores = 4, n_chains = 16)', globals = globals(), number = iters)
    t16c16 = ti.timeit('metropolis(n_cores = 16, n_chains = 16)', globals = globals(), number = iters)
    
    print(f"Execution time for 1 chain:\n1 core: {t1c1}\n4 cores: {t4c1}\n16 cores: {t16c1}\n")
    print(f"Execution time for 4 chains:\n1 core: {t1c4}\n4 cores: {t4c4}\n16 cores: {t16c4}\n")
    print(f"Execution time for 16 chains:\n1 core: {t1c16}\n4 cores: {t4c16}\n16 cores: {t16c16}\n")
    # Execution time for 1 chain:
    #1 core: 11.148524099990027
    #4 cores: 10.298176300013438
    #16 cores: 10.456789700023364

    #Execution time for 4 chains:
    #1 core: 37.46264680000604
    #4 cores: 86.86602760001551
    #16 cores: 87.1835080999881

    #Execution time for 16 chains:
    #1 core: 147.5261680999829
    #4 cores: 382.13164209999377
    #16 cores: 610.7609780000057

    plt.show()