import os
import matplotlib.pyplot as plt
import numpy as np

def fpermeability(k0, eps, delta, gamma, sigma0, a, b, sigma_m, sigma_tres):
    sigma_vm = 170/120*np.abs(sigma_m)
    x = 0
    y = 50
    lin = (1 + 0.1*(a * x / 50 + b * y / 50))
    kr = 1/eps *k0
    k = kr + delta * np.exp(np.log((k0 - kr) / delta) * sigma_m / sigma0)
    return np.where(sigma_vm < sigma_tres, k, k * np.exp(gamma * (sigma_vm - sigma_tres)/sigma_tres)) * lin


def fconductivity(perm):
    return 1000*9.81/0.001*perm


def plot_conductivity(config_dict, params):
    import matplotlib.pyplot as plt
    parnames = [p["name"] for p in config_dict["parameters"]]

    # init_sigma_m = -(42 + 19 + 14)*1e6/3
    sigma_tres = 55e6
    # sigma_mean is positive for graph, but goes negative into the permeability function
    sigma_mean = np.linspace(-7e6,120e6,200)
    init_sigma_m = -(params[:,parnames.index("init_stress_x")] +
                     params[:,parnames.index("init_stress_y")] +
                     params[:,parnames.index("init_stress_z")]) / 3
    # cond = conductivity(p[parnames.index("perm_k0")],
    #                     p[parnames.index("perm_eps")],
    #                     p[parnames.index("perm_delta")],
    #                     p[parnames.index("perm_gamma")],
    #                     init_sigma_m,
    #                     p[parnames.index("conductivity_a")],
    #                     p[parnames.index("conductivity_b")],
    #                     p[parnames.index("conductivity_c")],
    permeability = np.zeros((params.shape[0],len(sigma_mean)))

    # fig_cond, ax_cond = plt.subplots()
    # plt.rcParams['text.usetex'] = True
    fig, ax1 = plt.subplots()
    xax = sigma_mean / 1e6

    for i in range(params.shape[0]):
        p = params[i,:]
        # if p[4] >= p[5]:
        #     continue
        # p[[4,5]] = p[[5,4]]
        perm = fpermeability(p[parnames.index("perm_k0")],
                            p[parnames.index("perm_eps")],
                            p[parnames.index("perm_delta")],
                            p[parnames.index("perm_gamma")],
                            init_sigma_m[i],
                            p[parnames.index("conductivity_a")],
                            p[parnames.index("conductivity_b")],
                            -sigma_mean, sigma_tres)
        permeability[i,:] = perm
        # cond = fconductivity(perm)
        # ax_cond.plot(sigma_mean/1e6, cond)
        # ax_cond.scatter(init_sigma_m[i]/1e6, fconductivity(p[parnames.index("perm_k0")]))

    # plot N random samples:
    from random import randrange
    rids = [randrange(0, params.shape[0]) for i in range(0, 50)]
    for rid in rids:
        ax1.plot(xax, permeability[rid, :], linewidth=0.5, color="black", alpha=0.2)

    # plot permeability range
    q_mean = np.mean(permeability, axis=0)
    q_up = np.quantile(permeability, q=0.95, axis=0)
    q_down = np.quantile(permeability, q=0.05, axis=0)
    # q_99 = np.max(permeability, axis=0)
    # q_01 = np.min(permeability, axis=0)

    ax1.set_xlabel(r'$\sigma_m \mathrm{[MPa]}$')
    ax1.set_ylabel(r'$\kappa \mathrm{[m^2]}$', color='black')
    ax1.fill_between(xax, q_down, q_up,
                     color="red", alpha=0.2, label=None)

    # plot vertical lines of interest
    def add_vline(x, label):
        ax1.axvline(x=x / 1e6, linewidth=0.5)
        ax1.text(x / 1e6 + 0.75, 0.015, label, rotation=0, transform=ax1.get_xaxis_transform())
    def add_hline(y, label):
        ax1.axhline(y=y, linewidth=0.5)
        ax1.text(0.01, y*1.4, label, rotation=0, transform=ax1.get_yaxis_transform())

    add_vline(sigma_tres, r'$\sigma_{\mathrm{VM}c}$')
    add_vline(0, '')
    # add_vline(np.min(-init_sigma_m), r'$\sigma_0$')
    init_sigma_m_mean = np.mean(-init_sigma_m)
    add_vline(init_sigma_m_mean, r'$\sigma_{m0}$')
    # add_vline(np.max(-init_sigma_m), r'$\sigma_0$')

    # plot permeability
    ax1.plot(xax, q_mean, color="red")

    def add_annotated_point(x,y,label):
        ax1.scatter(x, y, facecolors='none', edgecolors='limegreen', marker='o',
                    zorder=100)
        ax1.annotate(label, xy=(x + 2, y),
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none',
                               boxstyle='round,pad=0.2,rounding_size=0.2'))

    perm_delta_mean = np.mean(params[:, parnames.index("perm_delta")])
    add_annotated_point(0, perm_delta_mean, r'$[0, \kappa_\delta]$')
    # ax1.scatter(0, perm_delta_mean, facecolors='none', edgecolors='limegreen', marker='o', zorder=100)
    # ax1.text(0, perm_delta_mean, r'$[0, \kappa_\delta]$', rotation=0, transform=ax1.get_yaxis_transform())
    # ax1.annotate(r'$[0, \kappa_\delta]$', xy=(0+2, perm_delta_mean),
    #              bbox=dict(facecolor='white', alpha=0.4, edgecolor='none'))
    perm_k0_mean = np.mean(params[:, parnames.index("perm_k0")])
    add_annotated_point(init_sigma_m_mean/1e6, perm_k0_mean, r'$[\sigma_{m0}, \kappa_0]$')

    perm_kr_mean = np.mean(params[:, parnames.index("perm_k0")]/params[:, parnames.index("perm_eps")])
    # perm_kr_mean = np.mean(params[:, parnames.index("perm_k0")]) / np.mean(params[:, parnames.index("perm_eps")])
    # print(perm_kr_mean)
    # ax1.scatter(init_sigma_m_mean / 1e6, perm_kr_mean, facecolors='none', edgecolors='limegreen', marker='o',
    #             zorder=100)
    add_hline(perm_kr_mean, label=r'$\kappa_r$')

    ax1.text(0.52,0.9,
             # r'$\sigma_{VM}=\frac{170}{120}\sigma_m$\\$\kappa_r=\kappa_0/\epsilon,\qquad \epsilon>1.1$',
             r'\begin{eqnarray*} \sigma_{\mathrm{VM}c} &=& \frac{170}{120}\sigma_m \\'
             r'\kappa_r &=& \kappa_0/\epsilon,\qquad \epsilon>1.1 \end{eqnarray*}',
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5,rounding_size=0.2', linewidth=0.5)
             )

    # finialize figure
    ax1.set_xlim(np.min(xax), np.max(xax))
    ax1.set_yscale('log')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig("permeability.pdf")