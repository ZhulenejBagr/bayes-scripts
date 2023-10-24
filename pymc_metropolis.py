import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import timeit as ti

def metropolis(samples=10000, n_cores=4, n_chains=4, tune=3000):
    seed = np.random.default_rng(222)
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
        G = pm.Normal('G', mu=G_mean, sigma=sigma, observed=observed)
        idata = pm.sample(samples, tune=tune, step=pm.Metropolis(), chains=n_chains, cores=n_cores, random_seed=seed)
        return idata

def benchmark():
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
    
    print(f"Execution time for 1 chain:\n1 core: {t1c1}s\n4 cores: {t4c1}s\n16 cores: {t16c1}s\n")
    print(f"Execution time for 4 chains:\n1 core: {t1c4}s\n4 cores: {t4c4}s\n16 cores: {t16c4}s\n")
    print(f"Execution time for 16 chains:\n1 core: {t1c16}s\n4 cores: {t4c16}s\n16 cores: {t16c16}s\n")
    # Execution time for 1 chain:
    #1 core: 11.148524099990027s
    #4 cores: 10.298176300013438s
    #16 cores: 10.456789700023364s

    #Execution time for 4 chains:
    #1 core: 37.46264680000604s
    #4 cores: 86.86602760001551s
    #16 cores: 87.1835080999881s

    #Execution time for 16 chains:
    #1 core: 147.5261680999829s
    #4 cores: 382.13164209999377s
    #16 cores: 610.7609780000057s


if __name__ == "__main__":
    idata = metropolis(samples=5000, tune=5000, n_cores=4, n_chains=4)
    gs = 40
    az.plot_pair(
        data=idata, 
        kind="hexbin", 
        marginals=True, 
        gridsize=(round(gs * 1.73), gs), 
        hexbin_kwargs={"cmap": "Greys"}
        #var_names=['G', 'U']
        )
    #az.plot_autocorr(data)
    #az.plot_rank(data=data)
    print(az.summary(data=idata))    
    az.plot_trace(data=idata, compact=True)
    #az.plot_ppc(data=data)
    plt.show()

