import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import beta


def canonicalize(labels):
    """
    Normalize a partition so two equivalent partitions
    (same grouping, different label names) map to the same tuple.
    """
    mapping = {}
    next_id = 0
    canonical = []
    for label in labels:
        if label not in mapping:
            mapping[label] = next_id
            next_id += 1
        canonical.append(mapping[label])
    return tuple(canonical)



class GTKmeans():
    def __init__(self, k: int, epsilon: float, alpha: float = 0.05) -> None:
        self.k = k
        self.epsilon = epsilon
        self.alpha = epsilon/2  # confidence level for p* lower bound
        self.run_i_min = 6

    def _p_star_lower_bound(self, c: int) -> float:
        """Clopper-Pearson one-sided lower confidence bound on p*."""
        if c == 0:
            return 0.0
        return beta.ppf(self.alpha, c, self.run_i - c + 1)

    def _empirical_bound(self) -> float:
        return ((1 - self.p_lo)**self.run_i + self.run_i * self.p_lo * (1 - self.p_lo)**(self.run_i - 1)) / self.run_i

    def fit(self, data: np.ndarray) -> KMeans:
        self.partition_counts = {}
        phi_min = float('inf')
        best_key = None
        kmeans_final = None
        self.run_i = 0

        while True:
            self.run_i += 1
            kmeans_i = KMeans(n_clusters=self.k, n_init=1).fit(data)
            key = canonicalize(kmeans_i.labels_)
            self.partition_counts[key] = self.partition_counts.get(key, 0) + 1

            if kmeans_i.inertia_ < phi_min:
                phi_min = kmeans_i.inertia_
                kmeans_final = kmeans_i
                best_key = key

            if self.run_i >= self.run_i_min:
                n_singletons = sum(1 for c in self.partition_counts.values() if c == 1)
                gt_bound = n_singletons / self.run_i
                worst_case_bound = 1 / self.run_i

                c = self.partition_counts[best_key]
                self.p_lo = self._p_star_lower_bound(c)
                emp_bound = self._empirical_bound()

                #ub = min(gt_bound, worst_case_bound, emp_bound)
                #if ub <= self.epsilon/2:
                    #break
                self.stop_criteria = np.array([gt_bound <= self.epsilon/2, worst_case_bound<= self.epsilon/2, emp_bound<= self.epsilon/2])
                if np.any(self.stop_criteria):
                    break

        return kmeans_final
    

            
            
                
            
            
            
            
        
        
        

