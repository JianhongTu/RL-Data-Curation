import numpy as np
import torch

def mean_cosine_distance_for_indices(X_unit, idx):
    """
    X_unit: (N, d) L2-normalized.
    idx: 1D array/list of selected indices.
    """
    sel = X_unit[idx]
    t = sel.shape[0]
    if t <= 1:
        return 0.0
    s = sel.sum(axis=0)                 # sum of vectors
    s2 = float(np.dot(s, s))            # ||sum||^2
    mean_sim = (s2 - t) / (t * (t - 1)) # mean cosine similarity
    return 1.0 - mean_sim               # mean cosine distance


def inclusion_logprob(model, X, batch_size=8192, device=None):
  policy = model.policy
  device = device or next(policy.parameters()).device
  out = []
  with torch.no_grad():
    for i in range(0, len(X), batch_size):
      xb = torch.as_tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
      dist = policy.get_distribution(xb)
      probs = dist.distribution.probs
      logp_inc = torch.log(probs[:, 1].clamp_min(1e-12))
      out.append(logp_inc.cpu().numpy())
  return np.concatenate(out, axis=0)

def rl_topM_indices(model, X_unit, M):
  logp = inclusion_logprob(model, X_unit)
  rank = np.argsort(-logp)
  return rank[:M]

def random_topM_indices(X_unit, M, rng=None):
  rng = rng or np.random.default_rng()
  return rng.choice(len(X_unit), size=M, replace=False)

def greedy_dpp_topM_indices(X_unit, M):
    raise NotImplementedError
