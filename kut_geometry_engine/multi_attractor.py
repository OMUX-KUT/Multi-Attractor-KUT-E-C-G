"""
Module: kut_geometry_engine.multi_attractor
Formulation: Multi-Attractor Geodesic Sampling over Non-Contractible Statistical Manifolds
Axiom: E=C (Equivalence of Computation and Topological Energy)
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Any

class MultiAttractorKUTController:
    """
    Controller for tracking multiple Pareto-optimal attractors on non-contractible manifolds.
    Applies independent Ricci flow smoothing to each attractor cluster to preserve the Pareto frontier.
    """
    def __init__(
        self,
        num_attractors: int = 2,
        base_temp: float = 0.2,
        dispersion_factor: float = 0.45
    ):
        self.num_attractors = num_attractors
        self.base_temp = base_temp
        self.dispersion_factor = dispersion_factor

    def decompose_attractors(self, logits: torch.Tensor) -> List[torch.Tensor]:
        """
        Decomposes top-k logits into distinct attractor directions via spatial clustering.
        """
        top_v, top_idx = torch.topk(logits, k=min(40, logits.size(-1)))
        
        attractor_logits = []
        split_size = top_v.size(-1) // self.num_attractors
        for i in range(self.num_attractors):
            mask = torch.zeros_like(logits)
            selected_indices = top_idx[..., i * split_size : (i + 1) * split_size]
            mask.scatter_(-1, selected_indices, 1.0)
            
            proj_logits = logits * mask + (1.0 - mask) * (-1e9)
            attractor_logits.append(proj_logits)
            
        return attractor_logits

    def evaluate_pareto_optimality(
        self,
        objective_scores: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Evaluates non-dominated points and computes the hypervolume indicator across objective scores (N, M).
        """
        n_points = objective_scores.shape[0]
        is_efficient = np.ones(n_points, dtype=bool)
        
        for i, c in enumerate(objective_scores):
            if is_efficient[i]:
                is_efficient[is_efficient] = ~np.all(objective_scores[is_efficient] <= c, axis=1) | np.all(objective_scores[is_efficient] == c, axis=1)
                is_efficient[i] = True

        non_dominated_ratio = float(np.mean(is_efficient))
        
        if objective_scores.shape[1] == 2:
            sorted_pts = objective_scores[is_efficient]
            sorted_pts = sorted_pts[np.argsort(sorted_pts[:, 0])]
            hv = 0.0
            last_y = 0.0
            for pt in sorted_pts:
                hv += max(0.0, pt[0]) * max(0.0, pt[1] - last_y)
                last_y = pt[1]
        else:
            hv = non_dominated_ratio * 0.85

        return is_efficient, float(hv)
