from __future__ import annotations
from dataclasses import dataclass
import math
import random

from src.environment.scenarios import Offer, Scenario, TierRules, UserState, downgrade_risk


@dataclass
class BehaviouralAgent:
    """
    Prospect-style distortions + present bias + effort aversion.
    Also supports reactance to AI suggestions (handled when AI is introduced).
    """
    beta_present: float = 0.65        # <1 means present bias
    loss_aversion: float = 2.0        # >1 means losses loom larger
    effort_aversion: float = 1.3      # amplifies effort costs
    choice_overload_k: float = 0.08   # penalty per extra offer
    noise_std: float = 0.4
    seed: int = 11

    def choose_offer(self, state: UserState, scenario: Scenario, rules: TierRules) -> Offer:
        rng = random.Random(self.seed + state.day + state.points)
        best_offer = scenario.offers[0]
        best_u = -1e9

        overload_penalty = self.choice_overload_k * max(0, len(scenario.offers) - 4)

        for offer in scenario.offers:
            u = self.utility(state, scenario, offer, rules) - overload_penalty
            u += rng.gauss(0.0, self.noise_std)
            if u > best_u:
                best_u = u
                best_offer = offer
        return best_offer

    def utility(self, state: UserState, scenario: Scenario, offer: Offer, rules: TierRules) -> float:
        if offer.name == "No_Action":
            # present bias makes delaying feel attractive now, but there is still a penalty
            return (1.0 / self.beta_present) * (-0.2) - scenario.delay_penalty

        immediate_gain = scenario.base_purchase_value + offer.discount_value
        points_gain = scenario.base_purchase_points + offer.bonus_points
        long_term_gain = 0.003 * points_gain

        # present bias discounts long-term
        total_gain = immediate_gain + self.beta_present * long_term_gain

        # effort feels heavier
        effort = self.effort_aversion * (scenario.time_cost + offer.effort_cost)

        # downgrade risk feels like a loss (loss aversion)
        risk = downgrade_risk(state, rules)
        perceived_loss = self.loss_aversion * (0.9 * risk)

        return total_gain - effort - perceived_loss
