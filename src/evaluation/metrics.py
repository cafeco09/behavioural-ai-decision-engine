from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from src.environment.scenarios import Offer, Scenario, TierRules, UserState, tier_from_points


@dataclass
class Outcome:
    acted: bool
    points_delta: int
    immediate_value: float
    effort_cost: float
    new_tier: str


def simulate_outcome(state: UserState, scenario: Scenario, chosen: Offer, rules: TierRules) -> Outcome:
    if chosen.name == "No_Action":
        return Outcome(
            acted=False,
            points_delta=0,
            immediate_value=0.0,
            effort_cost=0.0,
            new_tier=state.tier.value,
        )

    points_delta = scenario.base_purchase_points + chosen.bonus_points
    new_points = state.points + points_delta
    new_tier = tier_from_points(new_points, rules).value

    immediate_value = scenario.base_purchase_value + chosen.discount_value
    effort_cost = scenario.time_cost + chosen.effort_cost

    return Outcome(
        acted=True,
        points_delta=points_delta,
        immediate_value=immediate_value,
        effort_cost=effort_cost,
        new_tier=new_tier,
    )


def regret(optimal_value: float, realised_value: float) -> float:
    return max(0.0, optimal_value - realised_value)
