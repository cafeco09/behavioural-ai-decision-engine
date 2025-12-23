from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math

from src.environment.scenarios import Offer, Scenario, TierRules, UserState, downgrade_risk


@dataclass
class RationalAgent:
    """
    Maximises expected utility without behavioural distortions.
    """
    def choose_offer(self, state: UserState, scenario: Scenario, rules: TierRules) -> Offer:
        best_offer = scenario.offers[0]
        best_u = -1e9
        for offer in scenario.offers:
            u = self.utility(state, scenario, offer, rules)
            if u > best_u:
                best_u = u
                best_offer = offer
        return best_offer

    def utility(self, state: UserState, scenario: Scenario, offer: Offer, rules: TierRules) -> float:
        if offer.name == "No_Action":
            # delaying / not acting has penalty
            return -scenario.delay_penalty

        # immediate value + expected long-term tier value proxy
        immediate = scenario.base_purchase_value + offer.discount_value
        effort = scenario.time_cost + offer.effort_cost

        # Points contribute to tier value: simple linear proxy
        points_gain = scenario.base_purchase_points + offer.bonus_points
        long_term = 0.003 * points_gain  # small weight; tune later

        # Risk of downgrade: rational agent still accounts but without loss aversion
        risk = downgrade_risk(state, rules)
        risk_cost = 0.8 * risk

        return immediate + long_term - effort - risk_cost
