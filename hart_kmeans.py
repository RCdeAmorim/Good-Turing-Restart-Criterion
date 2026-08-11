import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans

#W. Hart. Sequential stopping rules for random optimization methods with applications to multistart local search.
#SIAM Journal on Optimization, 9(1):270–290, 1998


def _tau_sequence(Y, n):
    taus = [n]
    while taus[-1] > 0:
        tau_j = taus[-1]
        window = Y[: tau_j - 1]
        mismatch = np.flatnonzero(window != Y[tau_j - 1])
        taus.append(int(mismatch[-1] + 1) if mismatch.size else 0)
    return taus


def _rho_hat_n(Y, n, epsilon=0.0):
    taus = _tau_sequence(Y, n)
    threshold = Y[n - 1] + epsilon
    qualifying = [j for j, t in enumerate(taus, 1) if t > 0 and Y[t - 1] <= threshold]
    rho = max(qualifying) if qualifying else 0
    tau_2 = taus[1] if len(taus) > 1 else 0
    gamma = int(np.sum(Y[tau_2:n - 1] <= threshold)) if tau_2 + 1 < n else 0
    return rho + gamma


def hart_rule_2a(Y, delta, beta, epsilon=0.01):
    n = len(Y)
    p_hat = _rho_hat_n(Y, n, epsilon) / n
    lhs = norm.cdf(2 * delta * np.sqrt(n)) - norm.cdf(-2 * delta * np.sqrt(n)) - (1 - p_hat) ** n
    return lhs >= 1 - beta


class HartKmeans:
    def __init__(self, k, delta=0.4, beta=0.025):
        self.k = k
        self.delta = delta
        self.beta = beta

    def fit(self, data):
        z_values = []
        best, best_val = None, float('inf')
        self.run_i = 0
        while True:
            self.run_i += 1
            km_i = KMeans(n_clusters=self.k, n_init=1).fit(data)
            z_values.append(km_i.inertia_)
            if km_i.inertia_ < best_val:
                best_val, best = km_i.inertia_, km_i
            if self.run_i >= 2:
                Y = np.minimum.accumulate(np.array(z_values))
                if hart_rule_2a(Y, self.delta, self.beta):
                    break
        return best
