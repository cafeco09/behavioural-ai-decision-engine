from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import random
from typing import Dict, List, Tuple


class Tier(str, Enum):
    BRONZE = "Bronze"
    SILVER = "Silver"
    GOLD = "Gold"


@dataclass(frozen=True)
class TierRules:
    # Minimum points required within the window to hold/achieve tiers
    silver_threshold: int = 400
    gold_threshold: int = 900
    window_days: int = 30


@dataclass
class UserState:
    tier: Tier
    day: int                 # 1..window_days
    points: int              # points earned so far in window
    last_window_points: int  # for downgrade risk proxy


@dataclass(frozen=True)
class Offer:
    name: str
    bonus_points: int        # extra points if chosen
    discount_value: float    # perceived immediate £ value (proxy)
    effort_cost: float       # cognitive/operational effort cost


@dataclass(frozen=True)
class Scenario:
    name: str
    offers: List[Offer]
    base_purchase_points: int
    base_purchase_value: float     # perceived utility from purchase itself
    time_cost: float               # effort cost to buy now
    delay_penalty: float           # risk/penalty for delaying in this scenario


def sample_offers_overload(rng: random.Random, k: int) -> List[Offer]:
    offers = []
    for i in range(k):
        offers.append(
            Offer(
                name=f"Offer_{i+1}",
                bonus_points=rng.choice([0, 25, 50, 100, 150]),
                discount_value=rng.choice([0.0, 1.0, 2.0, 3.0]),
                effort_cost=rng.choice([0.2, 0.5, 0.8, 1.2]),
            )
        )
    # Include "do nothing" option
    offers.append(Offer(name="No_Action", bonus_points=0, discount_value=0.0, effort_cost=0.0))
    return offers


def make_scenarios(rng_seed: int = 42, rules: TierRules = TierRules()) -> List[Scenario]:
    rng = random.Random(rng_seed)

    threshold_sprint = Scenario(
        name="Threshold_Sprint",
        offers=[
            Offer("Double_Points_Today", bonus_points=120, discount_value=0.5, effort_cost=0.6),
            Offer("Small_Discount", bonus_points=30, discount_value=2.0, effort_cost=0.4),
            Offer("No_Action", bonus_points=0, discount_value=0.0, effort_cost=0.0),
        ],
        base_purchase_points=80,
        base_purchase_value=3.0,
        time_cost=0.8,
        delay_penalty=1.5,  # delaying is costly near threshold
    )

    offer_overload = Scenario(
        name="Offer_Overload",
        offers=sample_offers_overload(rng, k=6),  # overload by design
        base_purchase_points=60,
        base_purchase_value=2.5,
        time_cost=0.7,
        delay_penalty=0.7,
    )

    ai_suggestion = Scenario(
        name="AI_Suggestion_vs_Autonomy",
        offers=[
            Offer("Bonus_Points", bonus_points=100, discount_value=0.5, effort_cost=0.6),
            Offer("Discount", bonus_points=25, discount_value=2.5, effort_cost=0.5),
            Offer("No_Action", bonus_points=0, discount_value=0.0, effort_cost=0.0),
        ],
        base_purchase_points=70,
        base_purchase_value=2.8,
        time_cost=0.7,
        delay_penalty=0.9,
    )

    return [threshold_sprint, offer_overload, ai_suggestion]


def tier_from_points(points: int, rules: TierRules) -> Tier:
    if points >= rules.gold_threshold:
        return Tier.GOLD
    if points >= rules.silver_threshold:
        return Tier.SILVER
    return Tier.BRONZE


def downgrade_risk(state: UserState, rules: TierRules) -> float:
    # Simple proxy: if last window points were below current tier threshold, risk is high.
    current_threshold = 0
    if state.tier == Tier.SILVER:
        current_threshold = rules.silver_threshold
    elif state.tier == Tier.GOLD:
        current_threshold = rules.gold_threshold

    if current_threshold == 0:
        return 0.1

    gap = current_threshold - state.last_window_points
    if gap <= 0:
        return 0.2
    return min(0.9, 0.2 + gap / (current_threshold * 1.2))
