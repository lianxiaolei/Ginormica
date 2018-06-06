#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import pymc
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
import pymc3

reload(sys)
sys.setdefaultencoding('utf8')

figsize(12.5, 4)
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 100

colors = ['#348ABD', '#A60628']
prior = [1/21., 20/21.]
posterior = [0.087, 1 - 0.087]
plt.bar([0, .7], prior, alpha=0.7, width=0.25,
        color=colors[0], label='prior distribution', lw='3', edgecolor=colors[0])

plt.bar([0 + 0.25, .7 + 0.25], posterior,
        alpha=0.7, width=0.25, color=colors[1], label='posterior distribution', lw='3', edgecolor=colors[1])

plt.xticks([0.20, 0.95], ['Lib', 'Far'])
plt.title('sheet')
plt.ylabel('probability')
plt.legend(loc='upper left')
plt.show()

pymc3.Poisson()