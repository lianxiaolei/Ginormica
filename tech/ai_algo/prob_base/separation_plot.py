# separation plot
# Author: Cameron Davidson-Pilon,2013
# see http://mdwardlab.com/sites/default/files/GreenhillWardSacks.pdf


import matplotlib.pyplot as plt
import numpy as np


def separation_plot(p, y, **kwargs):
    assert p.shape[0] == y.shape[0]
    n = p.shape[0]
    try:
        M = p.shape[1]
    except:
        p = p.reshape(n, 1)
        M = p.shape[1]

    colors_bmh = np.array(["#eeeeee", "#348ABD"])

    fig = plt.figure()

    for i in range(M):
        ax = fig.add_subplot(M, 1, i + 1)
        ix = np.argsort(p[:, i])
        # plot the different bars
        ax.bar(np.arange(n), np.ones(n), width=1.,
               color=colors_bmh[y[ix].astype(int)],
               edgecolor='none')
        ax.plot(np.arange(n + 1), np.append(p[ix, i], p[ix, i][-1]), "k",
                linewidth=1., drawstyle="steps-post")
        # create expected value bar.
        ax.vlines([(1 - p[ix, i]).sum()], [0], [1])
        plt.xlim(0, n)

    plt.tight_layout()

    return
