from __future__ import annotations
from dataclasses import dataclass
import math
import random

from src.environment.scenarios import Offer, Scenario, TierRules, UserState, Tier


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _near_threshold(state: UserState, rules: TierRules) -> float:
    """
    Returns a number in [0, 1] representing how close the user is to the next tier.
    Higher = closer to threshold.
    """
    if state.tier == Tier.BRONZE:
        target = rules.silver_threshold
    elif state.tier == Tier.SILVER:
        target = rules.gold_threshold
    else:
        return 0.0

    remaining = max(0, target - state.points)
    # Map "remaining points" to closeness: 0 points away => 1.0, far away => ~0.0
    scale = target * 0.35  # tune later
    return _sigmoid((scale - remaining) / max(1.0, scale))


def _choice_overload(scenario: Scenario) -> float:
    """
    Overload proxy in [0, 1] based on number of offers.
    """
    extra = max(0, len(scenario.offers) - 4)
    return min(1.0, extra / 6.0)


@dataclass
class BehaviouralAIResponse:
    """
    Mixed response to AI advice:
    - automation bias increases under tier pressure and (optionally) overload
    - reactance increases under overload for some users
    """
    seed: int = 23

    # How strongly tier pressure pushes following AI
    w_threshold: float = 1.4

    # How strongly overload pushes following AI via cognitive offload
    w_offload: float = 0.9

    # Reactance: higher => more likely to reject AI when overloaded
    reactance: float = 0.6

    # Baseline propensity to follow
    base: float = 0.0

    def follow_probability(self, state: UserState, scenario: Scenario, rules: TierRules) -> float:
        t = _near_threshold(state, rules)
        o = _choice_overload(scenario)

        # Offload helps following; reactance fights it (especially under overload).
        score = self.base + self.w_threshold * t + self.w_offload * o - self.reactance * o
        return _sigmoid(score)

    def choose_with_advisor(
        self,
        state: UserState,
        scenario: Scenario,
        rules: TierRules,
        advisor_recommendation: Offer,
        fallback_choice: Offer,
    ) -> Offer:
        rng = random.Random(self.seed + state.day + state.points)
        p_follow = self.follow_probability(state, scenario, rules)
        return advisor_recommendation if rng.random() < p_follow else fallback_choice
