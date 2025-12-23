from __future__ import annotations
from dataclasses import dataclass

from src.environment.scenarios import Offer, Scenario, TierRules, UserState
from src.agents.rational import RationalAgent


@dataclass
class SimpleAdvisor:
    """
    v1: Advisor recommends the offer that maximises rational utility.
    """

    def recommend(self, state: UserState, scenario: Scenario, rules: TierRules) -> Offer:
        agent = RationalAgent()
        return agent.choose_offer(state, scenario, rules)
