import numpy as np
from .selection import rl_topM_indices, random_topM_indices, mean_cosine_distance_for_indices

def evaluate(size_percents, N, X_unit, model_max, model_min):
    
    results_max = []
    results_min = []
    results_rand = []
    results_dpp = []

    for pct in size_percents:
        M = int(pct/100*N)
        print(f"=== Size {pct}% (M={M}) ===")

        idx_max = rl_topM_indices(model_max, X_unit, M)
        idx_min = rl_topM_indices(model_min, X_unit, M)

        rng = np.random.default_rng(0)
        idx_rand = random_topM_indices(X_unit, M, rng=rng)

        m_max = mean_cosine_distance_for_indices(X_unit, idx_max)
        m_min = mean_cosine_distance_for_indices(X_unit, idx_min)
        m_rand = mean_cosine_distance_for_indices(X_unit, idx_rand)

        results_max.append(m_max)
        results_min.append(m_min)
        results_rand.append(m_rand)

    results_max = np.array(results_max)
    results_min = np.array(results_min)
    results_rand = np.array(results_rand)

    return results_max, results_min, results_rand