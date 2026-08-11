import math
from sklearn.cluster import KMeans

# Corominas, A. (2023). On deciding when to stop metaheuristics: properties,
# rules and termination conditions. Operations Research Perspectives, 10, 100283.
# Section 7.2: R-property + N-rule for multi-start algorithms with a discrete
# solution distribution. N is fixed a priori (not data-adaptive):
#   N = ceil( ln(alpha) / ln(1 - epsilon) )

def corominas_N(epsilon, alpha):
    return math.ceil(math.log(alpha) / math.log(1 - epsilon))


class CorominasKmeans:
    def __init__(self, k, epsilon=0.01, alpha=0.05):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_restarts = corominas_N(epsilon, alpha)

    def fit(self, data):
        best, best_val = None, float('inf')
        self.run_i = 0
        for _ in range(self.n_restarts):
            self.run_i += 1
            km_i = KMeans(n_clusters=self.k, n_init=1).fit(data)
            if km_i.inertia_ < best_val:
                best_val, best = km_i.inertia_, km_i
        return best