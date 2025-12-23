from __future__ import annotations
from dataclasses import dataclass
import random

from src.environment.scenarios import Offer, Scenario, TierRules, UserState
from src.agents.rational import RationalAgent


@dataclass
class BoundedRationalAgent:
    """
    Like rational, but utilities are estimated with noise and limited attention.
    """
    noise_std: float = 0.6
    consider_top_k: int = 3
    seed: int = 7

    def choose_offer(self, state: UserState, scenario: Scenario, rules: TierRules) -> Offer:
        rng = random.Random(self.seed + state.day + state.points)

        base = RationalAgent()
        scored = []
        for offer in scenario.offers:
            u = base.utility(state, scenario, offer, rules)
            u_noisy = u + rng.gauss(0.0, self.noise_std)
            scored.append((u_noisy, offer))

        scored.sort(key=lambda x: x[0], reverse=True)
        shortlist = [o for _, o in scored[: min(self.consider_top_k, len(scored))]]

        # Soft choice among shortlist
        return rng.choice(shortlist)
