import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import scipy. stats as stats


class ValueError(Exception):
    pass


def density(distrib):
    r = distrib.rv.rvs(size=distrib.limit_val)
    distrib.r = r
    _, ax = plt.subplots(1, 1)
    x = distrib.x_axe()
    ax.hist(r, density=True, color='blue', edgecolor='black')
    ax.plot(x, distrib.get_density_plot(x), 'r-', lw=5, label=distrib.name+' density')
    ax.legend(loc='best', frameon=False)
    plt.show()


def LoLN(distrib):
    r = distrib.rv.rvs(size=distrib.limit_val)
    ex = distrib.ex
    _, ax = plt.subplots(1, 1)
    exs = []
    sum = 0
    for i in range(1, len(r)):
        sum += r[i]
        exs.append(sum / i)
    ax.plot(range(1, len(r)), exs, 'g-', lw=2, label=distrib.name+' E(X)')
    plt.scatter([len(r) + 1], ex, c='red', marker='*', label='expect', s=40)
    ax.legend(loc='best', frameon=False)
    plt.show()


def CLT(distrib):
    r = distrib.rv.rvs(2000000)
    ex = distrib.ex
    dx = distrib.dx
    _, ax = plt.subplots(1, 1)
    dxs = []
    sum = 0
    for i in range(1, len(r)):
        sum += r[i]
        dxs.append((sum - i * ex) / sqrt(i * dx))
        # dxs.append(sum/i)
    i = 10
    while i <= 1000000:
        print(distrib.name, i, dxs[i])
        i *= 10
    ax.plot(range(1, len(r)), dxs, 'g-', lw=2, label=distrib.name+' CLT')
    # plt.scatter([len(r) + 1], ex, c='red', marker='*', label='expect', s=40)
    ax.legend(loc='best', frameon=False)
    plt.show()


class Gamma:
    def __init__(self, k, loc=0, scale=1):
        self.check_parameters(k, scale)
        self.k = k
        self.loc = loc
        self.scale = scale
        self.rv = stats.gamma(k, loc=loc, scale=scale)
        self.ex = stats.gamma.expect(lambda x: x, (k,), loc=loc, scale=scale)
        self.dx = stats.gamma.var(k, loc=loc, scale=scale)
        self.name = 'gamma'
        self.limit_val = 2000000

    def get_density_plot(self, x):
        return self.rv.pdf(x)

    def x_axe(self):
        return np.linspace(self.rv.ppf(0.0000001), self.rv.ppf(0.9999999), 100)

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
        self.ex = stats.norm.expect(lambda x: x, loc=loc, scale=scale)
        self.dx = stats.norm.var(loc=loc, scale=scale)
        self.name = 'norm'
        self.limit_val = 4000

    def get_density_plot(self, x):
        return self.rv.pdf(x)

    def x_axe(self):
        return np.linspace(self.rv.ppf(0.0000001), self.rv.ppf(0.9999999), 100)

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
        self.ex = stats.binom.expect(lambda x: x, (n, p,), loc, scale)
        self.dx = stats.binom.var(n, p, loc=loc)
        self.name = 'binomial'
        self.limit_val = 6000

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
        self.ex = stats.uniform.expect(lambda x: x, loc=self.loc, scale=self.scale)
        self.dx = stats.uniform.var(self.loc, self.scale)
        self.name = 'uniform'
        self.limit_val = 1000

    def get_density_plot(self, x):
        return self.rv.pdf(x)

    def x_axe(self):
        return np.linspace(self.rv.ppf(0.0000001), self.rv.ppf(0.9999999), 100)

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
        self.ex = stats.poisson.expect(lambda x: x, (mu,), loc, scale)
        self.dx = stats.poisson.var(mu, loc=loc)
        self.name = 'poisson'
        self.limit_val = 6000
        self.r = [0]

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
        self.ex = stats.geom.expect(lambda x: x, (p,), loc, scale)
        self.dx = stats.geom.var(p, loc=loc)
        self.name = 'geometric'
        self.limit_val = 6000
        self.r = []

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
        self.ex = stats.hypergeom.expect(lambda x: x, (M, n, N,), loc, scale)
        self.dx = stats.hypergeom.var(M, n, N, loc=loc)
        self.name = 'hgypergeometric'
        self.limit_val = 6000
        self.r = []

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
        self.ex = stats.cauchy.expect(lambda x: x, loc=loc, scale=scale)
        self.dx = stats.cauchy.var(loc=loc, scale=scale)
        self.name = 'cauchy'
        self.limit_val = 6000

    def get_density_plot(self, x):
        return self.rv.pdf(x)

    def x_axe(self):
        return np.linspace(self.rv.ppf(0.001), self.rv.ppf(0.999), 100)


if __name__ == '__main__':
    # 1
    gamma = Gamma(1.99, 0)
    norm = Norm(1000, 10)
    binom = Binom(40, 0.2)
    uniform = Uniform(6, 11)
    poisson = Poisson(5)
    geom = Geom(0.7)
    hypergeom = Hypergeom(20, 4, 12)
    cauchy = Cauchy()
    # density(gamma)
    # density(norm)
    # density(binom)
    # density(uniform)
    # density(poisson)
    # density(geom)
    # density(hypergeom)
    # density(cauchy)

    # 2
    # LoLN(gamma)
    # LoLN(norm)
    # LoLN(binom)
    # LoLN(uniform)
    # LoLN(poisson)
    # LoLN(geom)
    # LoLN(hypergeom)
    # LoLN(cauchy)  # мат ожидания не существует

    # 3
    CLT(gamma)
    CLT(norm)
    CLT(binom)
    CLT(uniform)
    CLT(poisson)
    CLT(geom)
    CLT(hypergeom)
