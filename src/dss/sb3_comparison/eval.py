import numpy as np
from .selection import (
    rl_topM_indices,
    random_topM_indices,
    mean_cosine_distance_for_indices,
    trace_diversity_for_indices,
)

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

        t_max = trace_diversity_for_indices(X_unit, idx_max)
        t_min = trace_diversity_for_indices(X_unit, idx_min)
        t_rand = trace_diversity_for_indices(X_unit, idx_rand)

        results_max.append(m_max)
        results_min.append(m_min)
        results_rand.append(m_rand)

        print(
            f"    SB3 Max: cos={m_max:.4f}, trace={t_max:.4f} | "
            f"SB3 Min: cos={m_min:.4f}, trace={t_min:.4f} | "
            f"Rand: cos={m_rand:.4f}, trace={t_rand:.4f}"
        )

    results_max = np.array(results_max)
    results_min = np.array(results_min)
    results_rand = np.array(results_rand)

    return results_max, results_min, results_rand

import numpy as np

def evaluate_ppov1(size_percents, N, X_unit, ppo_v1_max, ppo_v1_min):
    results_ppov1_max = []
    results_ppov1_min = []

    for pct in size_percents:
        M = int(pct / 100 * N)
        print(f"=== PPOv1 Size {pct}% (M={M}) ===")

        # Most diverse subset
        idx_max, _ = ppo_v1_max.score_and_rank_inference(
            X_embed=X_unit,
            k=M,
            minimize=False,
        )

        # Least diverse subset
        idx_min, _ = ppo_v1_min.score_and_rank_inference(
            X_embed=X_unit,
            k=M,
            minimize=True,
        )

        m_ppov1_max = mean_cosine_distance_for_indices(X_unit, idx_max)
        m_ppov1_min = mean_cosine_distance_for_indices(X_unit, idx_min)

        t_ppov1_max = trace_diversity_for_indices(X_unit, idx_max)
        t_ppov1_min = trace_diversity_for_indices(X_unit, idx_min)

        results_ppov1_max.append(m_ppov1_max)
        results_ppov1_min.append(m_ppov1_min)

        print(
            f"    PPOv1 Max: cos={m_ppov1_max:.4f}, trace={t_ppov1_max:.4f} | "
            f"PPOv1 Min: cos={m_ppov1_min:.4f}, trace={t_ppov1_min:.4f}"
        )

    return np.array(results_ppov1_max, dtype=np.float32), np.array(results_ppov1_min, dtype=np.float32)