"""
Baseline Methods for Diversity-Based Data Selection
====================================================
Implements non-RL baseline methods for comparison with the paper's RL approach.

Baselines:
- Random: Uniform random sampling
- Greedy: Iteratively maximize diversity (OPTIMAL but SLOW)
- Farthest-Point: Select samples furthest from selected set
- K-means++: K-means++ initialization for diversity
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple
from envs import OnlineCovTrace


@dataclass
class BaselineResult:
    """Result from a baseline method"""
    method_name: str
    selected_indices: np.ndarray
    final_diversity: float
    time_seconds: float
    diversity_history: list = None


class BaselineSelector:
    """Container for all baseline selection methods"""
    
    def __init__(self, X_embed: np.ndarray, seed: int = 42):
        """
        Initialize baseline selector.
        
        Args:
            X_embed: Embeddings matrix (N, d)
            seed: Random seed
        """
        self.X = X_embed.astype(np.float32)
        self.N, self.d = self.X.shape
        self.rng = np.random.default_rng(seed)
    
    def random_selection(self, M: int) -> BaselineResult:
        """
        Random baseline: Uniformly sample M points.
        
        Fast but low diversity.
        
        Args:
            M: Number of samples to select
            
        Returns:
            BaselineResult with selected indices and diversity
        """
        start_time = time.time()
        
        # Random selection
        selected_indices = self.rng.choice(self.N, M, replace=False)
        
        # Compute diversity
        tracker = OnlineCovTrace(d=self.d)
        for idx in selected_indices:
            tracker.add(self.X[idx])
        
        elapsed = time.time() - start_time
        
        return BaselineResult(
            method_name="Random",
            selected_indices=selected_indices,
            final_diversity=tracker.trace_cov_unbiased,
            time_seconds=elapsed
        )
    
    def greedy_selection(self, M: int, verbose: bool = True) -> BaselineResult:
        """
        Greedy baseline: Iteratively select sample that maximizes diversity.
        
        OPTIMAL diversity but VERY SLOW (O(NM) evaluations).
        This is what the paper tries to beat!
        
        Args:
            M: Number of samples to select
            verbose: Print progress
            
        Returns:
            BaselineResult with selected indices and diversity
        """
        start_time = time.time()
        
        selected = []
        remaining = set(range(self.N))
        tracker = OnlineCovTrace(d=self.d)
        diversity_history = []
        
        if verbose:
            print(f"  Greedy selection: selecting {M} from {self.N} samples...")
        
        for step in range(M):
            best_idx = None
            best_gain = -np.inf
            
            # Try each remaining sample (SLOW!)
            for idx in remaining:
                # Compute marginal gain
                prev_div = tracker.trace_cov_unbiased
                
                # Temporarily add this sample
                tracker_temp = OnlineCovTrace(d=self.d)
                for s_idx in selected:
                    tracker_temp.add(self.X[s_idx])
                tracker_temp.add(self.X[idx])
                
                gain = tracker_temp.trace_cov_unbiased - prev_div
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            # Add best sample
            selected.append(best_idx)
            remaining.remove(best_idx)
            tracker.add(self.X[best_idx])
            diversity_history.append(tracker.trace_cov_unbiased)
            
            if verbose and (step + 1) % max(1, M // 10) == 0:
                elapsed = time.time() - start_time
                print(f"    Step {step+1}/{M}: diversity={tracker.trace_cov_unbiased:.2f}, "
                      f"time={elapsed:.1f}s")
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"  ✓ Greedy completed in {elapsed:.2f}s")
        
        return BaselineResult(
            method_name="Greedy",
            selected_indices=np.array(selected),
            final_diversity=tracker.trace_cov_unbiased,
            time_seconds=elapsed,
            diversity_history=diversity_history
        )
    
    def farthest_point_selection(self, M: int, verbose: bool = True) -> BaselineResult:
        """
        Farthest-Point Sampling: Iteratively select point furthest from selected set.
        
        Good diversity, faster than greedy but still O(NM).
        
        Args:
            M: Number of samples to select
            verbose: Print progress
            
        Returns:
            BaselineResult with selected indices and diversity
        """
        start_time = time.time()
        
        selected = []
        remaining = set(range(self.N))
        
        # Start with random point
        first_idx = self.rng.integers(0, self.N)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        if verbose:
            print(f"  Farthest-point selection: selecting {M} from {self.N} samples...")
        
        # Iteratively select farthest point
        for step in range(1, M):
            max_min_dist = -1
            farthest_idx = None
            
            # For each remaining point, find min distance to selected set
            for idx in remaining:
                min_dist = np.inf
                for s_idx in selected:
                    dist = np.linalg.norm(self.X[idx] - self.X[s_idx])
                    min_dist = min(min_dist, dist)
                
                # Select point with maximum min-distance
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest_idx = idx
            
            selected.append(farthest_idx)
            remaining.remove(farthest_idx)
            
            if verbose and (step + 1) % max(1, M // 10) == 0:
                elapsed = time.time() - start_time
                print(f"    Step {step+1}/{M}: time={elapsed:.1f}s")
        
        # Compute final diversity
        tracker = OnlineCovTrace(d=self.d)
        for idx in selected:
            tracker.add(self.X[idx])
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"  ✓ Farthest-point completed in {elapsed:.2f}s")
        
        return BaselineResult(
            method_name="Farthest-Point",
            selected_indices=np.array(selected),
            final_diversity=tracker.trace_cov_unbiased,
            time_seconds=elapsed
        )
    
    def kmeans_plus_plus_selection(self, M: int, verbose: bool = True) -> BaselineResult:
        """
        K-means++ initialization for diversity sampling.
        
        Fast and reasonably diverse. Uses k-means++ seeding algorithm.
        
        Args:
            M: Number of samples to select
            verbose: Print progress
            
        Returns:
            BaselineResult with selected indices and diversity
        """
        start_time = time.time()
        
        selected = []
        remaining = list(range(self.N))
        
        # Start with random point
        first_idx = self.rng.integers(0, self.N)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        if verbose:
            print(f"  K-means++ selection: selecting {M} from {self.N} samples...")
        
        # K-means++ sampling
        for step in range(1, M):
            # Compute distances to nearest selected point
            distances = np.zeros(len(remaining))
            for i, idx in enumerate(remaining):
                min_dist = np.inf
                for s_idx in selected:
                    dist = np.linalg.norm(self.X[idx] - self.X[s_idx]) ** 2
                    min_dist = min(min_dist, dist)
                distances[i] = min_dist
            
            # Sample proportional to distance squared (k-means++ rule)
            probs = distances / distances.sum()
            next_idx_pos = self.rng.choice(len(remaining), p=probs)
            next_idx = remaining[next_idx_pos]
            
            selected.append(next_idx)
            remaining.pop(next_idx_pos)
        
        # Compute final diversity
        tracker = OnlineCovTrace(d=self.d)
        for idx in selected:
            tracker.add(self.X[idx])
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"  ✓ K-means++ completed in {elapsed:.2f}s")
        
        return BaselineResult(
            method_name="K-means++",
            selected_indices=np.array(selected),
            final_diversity=tracker.trace_cov_unbiased,
            time_seconds=elapsed
        )
    
    def run_all_baselines(self, M: int, verbose: bool = True) -> dict:
        """
        Run all baseline methods and return results.
        
        Args:
            M: Number of samples to select
            verbose: Print progress
            
        Returns:
            Dictionary mapping method names to BaselineResults
        """
        results = {}
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running All Baseline Methods (M={M}, N={self.N})")
            print(f"{'='*60}\n")
        
        # Random (fast)
        if verbose:
            print("1. Random Baseline:")
        results['Random'] = self.random_selection(M)
        if verbose:
            print(f"  ✓ Time: {results['Random'].time_seconds:.3f}s, "
                  f"Diversity: {results['Random'].final_diversity:.2f}\n")
        
        # K-means++ (fast)
        if verbose:
            print("2. K-means++ Baseline:")
        results['K-means++'] = self.kmeans_plus_plus_selection(M, verbose)
        if verbose:
            print(f"  ✓ Time: {results['K-means++'].time_seconds:.3f}s, "
                  f"Diversity: {results['K-means++'].final_diversity:.2f}\n")
        
        # Farthest-Point (medium speed)
        if verbose:
            print("3. Farthest-Point Baseline:")
        results['Farthest-Point'] = self.farthest_point_selection(M, verbose)
        if verbose:
            print(f"  ✓ Time: {results['Farthest-Point'].time_seconds:.3f}s, "
                  f"Diversity: {results['Farthest-Point'].final_diversity:.2f}\n")
        
        # Greedy (SLOW but optimal)
        if verbose:
            print("4. Greedy Baseline (SLOW - optimal diversity):")
        results['Greedy'] = self.greedy_selection(M, verbose)
        if verbose:
            print(f"  ✓ Time: {results['Greedy'].time_seconds:.3f}s, "
                  f"Diversity: {results['Greedy'].final_diversity:.2f}\n")
        
        return results


def compare_baselines(X_embed: np.ndarray, M: int, seed: int = 42, verbose: bool = True):
    """
    Convenient function to run and compare all baselines.
    
    Args:
        X_embed: Embeddings matrix (N, d)
        M: Number of samples to select
        seed: Random seed
        verbose: Print results
        
    Returns:
        Dictionary of results
    """
    selector = BaselineSelector(X_embed, seed=seed)
    results = selector.run_all_baselines(M, verbose=verbose)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Baseline Comparison Summary")
        print(f"{'='*60}")
        print(f"{'Method':<20} {'Time (s)':<12} {'Diversity':<12} {'Speedup':<10}")
        print("-" * 60)
        
        # Sort by time
        sorted_results = sorted(results.items(), key=lambda x: x[1].time_seconds)
        
        for method_name, result in sorted_results:
            speedup = results['Greedy'].time_seconds / result.time_seconds
            print(f"{method_name:<20} {result.time_seconds:>10.3f}  "
                  f"{result.final_diversity:>10.2f}  {speedup:>8.1f}x")
        
        print(f"{'='*60}\n")
    
    return results

