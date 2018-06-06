#!/usr/bin/python
# -*- coding: utf-8 -*-

import pymc
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def mymodel():
    # Some data
    n = 5 * np.ones(4, dtype=int)
    x = np.array([-.86, -.3, -.05, .73])

    # Priors on unknown parameters
    alpha = pymc.Normal('alpha', mu=0, tau=.01)
    beta = pymc.Normal('beta', mu=0, tau=.01)

    # Arbitrary deterministic function of parameters
    @pymc.deterministic
    def theta(a=alpha, b=beta):
        """theta = logit^{-1}(a+b)"""
        return pymc.invlogit(a + b * x)

    # Binomial likelihood for data
    d = pymc.Binomial('d', n=n, p=theta, value=np.array([0., 1., 3., 5.]), observed=True)


if __name__ == '__main__':
    '''这个例子会产生10000个后验样本。这个样本会存储在Python序列化数据库中。'''
    S = pymc.MCMC(mymodel, db='pickle')
    S.sample(iter=10000, burn=5000, thin=2)
    pymc.Matplot.plot(S)

    print '-' * 60 + '牛屁的分隔线' + '-' * 60

    '''生成一个MCMC对象来处理我们的模型，导入disaster_model.py并将其作为MCMC的参数。'''
    from pymc.examples import disaster_model
    from pymc import MCMC
    M = MCMC(disaster_model)
    M.sample(iter=10000, burn=1000, thin=10)

