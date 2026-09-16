"""
Module: kut_geometry_engine.patience_geometric_controller
Formulation: Information-Geometric Search Surgery and Scale-Aware Annealing
Axiom: E=C (Equivalence of Computation and Topological Information Energy)
"""

import math
from typing import Optional

class DynamicPatienceGeometricController:
    """
    Test-time search controller implementing:
    1. Marginal entropy dissipation monitoring for branch pruning.
    2. Model parameter scale-aware geometric temperature modulation.
    """
    def __init__(
        self,
        model_params_billion: float = 72.0,
        base_temp: float = 0.2,
        patience_cap: int = 1,
        entropy_floor: float = 0.05,
        marginal_threshold: float = 0.02
    ):
        self.model_params_billion = model_params_billion
        self.base_temp = base_temp
        self.patience_cap = patience_cap
        self.entropy_floor = entropy_floor
        self.marginal_threshold = marginal_threshold

        # Scaling law: beta(P) = 0.35 * (7 / P)^0.22
        self.beta = 0.35 * math.pow(7.0 / max(1.0, model_params_billion), 0.22)
        self.current_patience = 0
        self.prev_entropy: Optional[float] = None

    def step(self, current_entropy: float) -> float:
        """
        Calculates active sampling temperature based on local entropy trajectory.
        """
        if self.prev_entropy is None:
            self.prev_entropy = current_entropy
            return self.base_temp

        delta_h = self.prev_entropy - current_entropy

        # Detect non-convex saddle stagnation
        if delta_h < self.marginal_threshold and current_entropy > self.entropy_floor:
            if self.current_patience < self.patience_cap:
                self.current_patience += 1
                active_temp = self.base_temp * math.pow(1.0 + self.beta, self.current_patience)
            else:
                active_temp = self.base_temp
        else:
            self.current_patience = 0
            active_temp = self.base_temp

        self.prev_entropy = current_entropy
        return active_temp

    def should_terminate(self, current_entropy: float, flops_consumed: float, budget_flops: float) -> bool:
        """
        Termination condition: Exhausted compute budget or reached singular concentration.
        """
        if flops_consumed >= budget_flops:
            return True
        if current_entropy <= self.entropy_floor:
            return True
        return False
