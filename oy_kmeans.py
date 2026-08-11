import numpy as np
from sklearn.cluster import KMeans
#Ohsaki, Makoto, and Makoto Yamakawa. "Stopping rule of multi-start local search for structural optimization."
# Structural and Multidisciplinary Optimization 57, no. 2 (2018): 595-603.


def canonicalize(labels):
    mapping = {}
    next_id = 0
    canonical = []
    for label in labels:
        if label not in mapping:
            mapping[label] = next_id
            next_id += 1
        canonical.append(mapping[label])
    return tuple(canonical)


def likelihood_ratio(partition_counts, t):
    # s_j = n_j for each found solution, T = sum(s_j) = t (no intermediate solutions tracked)
    counts = list(partition_counts.values())
    w = len(counts)
    terms = [(1 - s_j / (t + s_j)) ** t for s_j in counts]
    return sum(terms) / w


class OYKmeans:
    def __init__(self, k, e3=0.05, min_trials=2):
        self.k = k
        self.e3 = e3
        self.min_trials = min_trials

    def fit(self, data):
        partition_counts = {}
        best, best_val = None, float('inf')
        self.run_i = 0
        while self.run_i<1000:
            self.run_i += 1
            km_i = KMeans(n_clusters=self.k, n_init=1).fit(data)
            key = canonicalize(km_i.labels_)
            partition_counts[key] = partition_counts.get(key, 0) + 1
            if km_i.inertia_ < best_val:
                best_val, best = km_i.inertia_, km_i
            if self.run_i >= self.min_trials:
                ratio = likelihood_ratio(partition_counts, self.run_i)
                if ratio < self.e3:
                    break
        return best