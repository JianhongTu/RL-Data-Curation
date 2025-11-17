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

        cos_max = mean_cosine_distance_for_indices(X_unit, idx_max)
        cos_min = mean_cosine_distance_for_indices(X_unit, idx_min)
        cos_rand = mean_cosine_distance_for_indices(X_unit, idx_rand)
        trace_max = trace_diversity_for_indices(X_unit, idx_max)
        trace_min = trace_diversity_for_indices(X_unit, idx_min)
        trace_rand = trace_diversity_for_indices(X_unit, idx_rand)

        print(
            f"    SB3 Max: cos={cos_max:.4f}, trace={trace_max:.4f} | "
            f"SB3 Min: cos={cos_min:.4f}, trace={trace_min:.4f} | "
            f"Rand: cos={cos_rand:.4f}, trace={trace_rand:.4f}"
        )

        results_max.append(cos_max)
        results_min.append(cos_min)
        results_rand.append(cos_rand)

    results_max = np.array(results_max)
    results_min = np.array(results_min)
    results_rand = np.array(results_rand)

    return results_max, results_min, results_rand

import numpy as np

def evaluate_ppov1(size_percents, N, X_unit, ppo_v1_max, ppo_v1_min=None):
    results_ppov1_max = []
    results_ppov1_min = []

    for pct in size_percents:
        M = int(pct / 100 * N)
        print(f"=== PPOv1 Size {pct}% (M={M}) ===")

        idx_max, scores = ppo_v1_max.score_and_rank_inference(
            X_embed=X_unit,
            k=M,
            minimize=False,
        )

        if ppo_v1_min is None:
            order = np.argsort(scores)
            idx_min = order[:M]
        else:
            idx_min, _ = ppo_v1_min.score_and_rank_inference(
                X_embed=X_unit,
                k=M,
                minimize=False,
            )

        m_ppov1_max = mean_cosine_distance_for_indices(X_unit, idx_max)
        m_ppov1_min = mean_cosine_distance_for_indices(X_unit, idx_min)

        trace_ppov1_max = trace_diversity_for_indices(X_unit, idx_max)
        trace_ppov1_min = trace_diversity_for_indices(X_unit, idx_min)

        print(
            f"    PPOv1 Max: cos={m_ppov1_max:.4f}, trace={trace_ppov1_max:.4f} | "
            f"PPOv1 Min (flipped): cos={m_ppov1_min:.4f}, trace={trace_ppov1_min:.4f}"
        )

        results_ppov1_max.append(m_ppov1_max)
        results_ppov1_min.append(m_ppov1_min)

    return np.array(results_ppov1_max, dtype=np.float32), np.array(results_ppov1_min, dtype=np.float32)