import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import scipy.stats as stats
import random

limit = 1000000


class ValueError(Exception):
    pass


class Gamma:
    def __init__(self, k, loc=0, scale=1):
        self.check_parameters(k, scale)
        self.k = k
        self.loc = loc
        self.scale = scale
        self.rv = stats.gamma(k, loc=loc, scale=scale)
        self.ex = stats.gamma.expect(None, (k,), loc=loc, scale=scale)
        self.dx = stats.gamma.var(k, loc=loc, scale=scale)
        self.name = 'gamma'

    def get_density_plot(self, x):
        return self.rv.pdf(x)

    def x_axe(self):
        return np.linspace(self.rv.ppf(0.0001), self.rv.ppf(0.9999), 10000)

    def check_parameters(self, k, scale):
        if k <= 0:
            raise ValueError("k must be > 0")
        if scale <= 0:
            raise ValueError("scale must be > 0")


class Norm:
    def __init__(self, loc=0, scale=1):
        self.check_parameters(scale)
        self.loc = loc
        self.scale = scale
        self.rv = stats.norm(loc, scale)
        self.ex = loc
        self.dx = scale
        self.name = 'normal'

    def get_density_plot(self, x):
        return self.rv.pdf(x)

    def x_axe(self):
        return np.linspace(self.rv.ppf(0.0001), self.rv.ppf(0.9999), 10000)

    def check_parameters(self, scale):
        if scale <= 0:
            raise ValueError("scale must be > 0")


class Binom:
    def __init__(self, n, p, loc=0, scale=1):
        self.check_parameters(n, p, scale)
        self.n = n
        self.p = p
        self.loc = loc
        self.scale = scale
        self.rv = stats.binom(n, p)
        self.ex = stats.binom.expect(None, (n, p), loc, scale)
        self.dx = stats.binom.var(n, p, loc=loc)
        self.name = 'binomial'

    def get_density_plot(self, x):
        return self.rv.pmf(x)

    def x_axe(self):
        return range(self.n + 1)

    def check_parameters(self, n, p, scale):
        if n <= 0:
            raise ValueError("n must be > 0")
        if 1 <= p <= 0:
            raise ValueError("p must be [0; 1]")
        if scale <= 0:
            raise ValueError("scale must be > 0")


class Uniform:
    def __init__(self, a, b):
        self.check_parameters(a, b)
        self.loc = a
        self.scale = b - a
        self.rv = stats.uniform(loc=self.loc, scale=self.scale)
        self.ex = stats.uniform.expect(None, loc=self.loc, scale=self.scale)
        self.dx = stats.uniform.var(self.loc, self.scale)
        self.name = 'uniform'

    def get_density_plot(self, x):
        return self.rv.pdf(x)

    def x_axe(self):
        return np.linspace(self.rv.ppf(0.0001), self.rv.ppf(0.9999), 10000)

    def check_parameters(self, a, b):
        if a > b:
            raise ValueError("a must be <= b")


class Poisson:
    def __init__(self, mu, loc=0, scale=1):
        self.check_parameters(mu, scale)
        self.mu = mu
        self.loc = loc
        self.scale = scale
        self.rv = stats.poisson(mu)
        self.ex = stats.poisson.expect(None, (mu,), loc, scale)
        self.dx = stats.poisson.var(mu, loc=loc)
        self.name = 'poisson'

    def get_density_plot(self, x):
        return self.rv.pmf(x)

    def x_axe(self):
        return range(self.r.min(), self.r.max() + 1)

    def check_parameters(self, mu, scale):
        if mu <= 0:
            raise ValueError("mu must be > 0")
        if scale <= 0:
            raise ValueError("scale must be > 0")


class Geom:
    def __init__(self, p, loc=0, scale=1):
        self.check_parameters(p, scale)
        self.p = p
        self.loc = loc
        self.scale = scale
        self.rv = stats.geom(p)
        self.ex = stats.geom.expect(None, (p,), loc, scale)
        self.dx = stats.geom.var(p, loc=loc)
        self.name = 'geometric'

    def get_density_plot(self, x):
        return self.rv.pmf(x)

    def x_axe(self):
        return range(self.r.min(), self.r.max() + 1)

    def check_parameters(self, p, scale):
        if 1 <= p <= 0:
            raise ValueError("p must be [0; 1]")
        if scale <= 0:
            raise ValueError("scale must be > 0")


class Hypergeom:
    def __init__(self, M, n, N, loc=0, scale=1):
        self.check_parameters(M, n, N, scale)
        self.M = M
        self.n = n
        self.N = N
        self.loc = loc
        self.scale = scale
        self.rv = stats.hypergeom(M, n, N)
        self.ex = stats.hypergeom.expect(None, (M, n, N,), loc, scale)
        self.dx = stats.hypergeom.var(M, n, N, loc=loc)
        self.name = 'hypergeometric'

    def get_density_plot(self, x):
        return self.rv.pmf(x)

    def x_axe(self):
        return range(max(0, self.N - self.M + self.n), min(self.n, self.N) + 1)

    def check_parameters(self, M, n, N, scale):
        if M <= 0:
            raise ValueError("M must be > 0")
        if M <= n <= 0:
            raise ValueError("n must be > [0; M]")
        if M <= N <= 0:
            raise ValueError("N must be > [0; M]")
        if scale <= 0:
            raise ValueError("scale must be > 0")


class Cauchy:
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale
        self.rv = stats.cauchy(loc=loc, scale=scale)
        self.ex = stats.cauchy.expect(None, loc=loc, scale=scale)
        self.dx = stats.cauchy.var(loc=loc, scale=scale)
        self.name = 'cauchy'

    def get_density_plot(self, x):
        return self.rv.pdf(x)

    def x_axe(self):
        return np.linspace(self.rv.ppf(0.01), self.rv.ppf(0.99), 100)


def calc_density(distrib):
    x = distrib.x_axe()
    return x, distrib.get_density_plot(x)


def density(distrib):
    r = distrib.rv.rvs(limit)
    if distrib.name == 'cauchy':
        r /= 10000
    distrib.r = r
    x, y = calc_density(distrib)
    _, ax = plt.subplots(1, 1)
    ax.hist(r, density=True, color='blue', edgecolor='black')
    ax.plot(x, y, 'r-', lw=5, label=distrib.name + ' density')
    ax.legend(loc='best', frameon=False)
    plt.show()


def LoLN(distrib):
    r = distrib.rv.rvs(limit)
    ex = distrib.ex
    _, ax = plt.subplots(1, 1)
    exs = []
    sum = 0
    for i in range(limit):
        sum += r[i]
        exs.append(sum / (i + 1))
    ax.plot(range(limit), exs, 'g-', lw=2, label=distrib.name + ' E(X)')
    plt.scatter(limit, ex, c='red', marker='*', label='expect', s=40)
    ax.legend(loc='best', frameon=False)
    plt.show()


def CLT(distrib):
    r = distrib.rv.rvs(limit)
    ex = distrib.ex
    dx = distrib.dx
    _, ax = plt.subplots(1, 1)
    dxs = []
    sum = 0
    for i in range(limit):
        sum += r[i]
        dxs.append((sum - (i + 1) * ex) / sqrt((i + 1) * dx))
    i = 10
    while i <= limit:
        print(distrib.name, i, dxs[i - 1])
        i *= 10
    ax.plot(range(limit), dxs, 'g-', lw=2, label=distrib.name + ' CLT')
    ax.legend(loc='best', frameon=False)


def check_CLT(distrib, N):
    n = 100
    ex = distrib.ex
    dx = distrib.dx
    _, ax = plt.subplots(1, 1)
    y = []
    for i in range(N):
        r = distrib.rv.rvs(n)
        y.append((sum(r) - n * ex) / sqrt(n * dx))
    ax.hist(y, density=True, color='blue', edgecolor='black', label=distrib.name+' sample')
    x, y = calc_density(Norm(0, 1))
    ax.plot(x, y, 'r-', lw=5, label='norm density', alpha=0.5)
    ax.legend(loc='best', frameon=False)
    plt.show()


if __name__ == '__main__':
    # 1
    gamma = Gamma(1.99, 10)
    norm = Norm(10, 5)
    binom = Binom(40, 0.2)
    uniform = Uniform(6, 11)
    poisson = Poisson(5)
    geom = Geom(0.7)
    hypergeom = Hypergeom(20, 4, 12)
    cauchy = Cauchy()

    density(gamma)
    density(norm)
    density(binom)
    density(uniform)
    density(poisson)
    density(geom)
    density(hypergeom)
    density(cauchy)

    # 2
    LoLN(gamma)
    LoLN(norm)
    LoLN(binom)
    LoLN(uniform)
    LoLN(poisson)
    LoLN(geom)
    LoLN(hypergeom)
    LoLN(cauchy)  # мат ожидания не существует

    # 3 стремится к 0 -> ЗБЧ, больше выборка - больше точность
    CLT(gamma)
    CLT(norm)
    CLT(binom)
    CLT(uniform)
    CLT(poisson)
    CLT(geom)
    CLT(hypergeom)

    check_CLT(gamma, 100000)
    check_CLT(norm, 100000)
    check_CLT(binom, 100000)
    check_CLT(uniform, 100000)
    check_CLT(poisson, 100000)
    check_CLT(geom, 100000)
    check_CLT(hypergeom, 100000)
