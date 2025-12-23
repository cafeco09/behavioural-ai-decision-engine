import random
from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.environment.scenarios import TierRules, Tier, UserState, make_scenarios
from src.agents.rational import RationalAgent
from src.agents.bounded import BoundedRationalAgent
from src.agents.behavioural import BehaviouralAgent
from src.agents.ai_response import BehaviouralAIResponse
from src.advisor.simple_advisor import SimpleAdvisor
from src.evaluation.metrics import simulate_outcome


def sample_user_state(rng: random.Random, rules: TierRules, profile: str) -> UserState:
    """
    Samples a user state to avoid trivial outcomes (e.g., always upgrading).
    'profile' controls how often users sit near thresholds vs far away.
    """
    # Tier mix: you can tune these weights later
    tier = rng.choices([Tier.BRONZE, Tier.SILVER, Tier.GOLD], weights=[0.45, 0.40, 0.15], k=1)[0]

    # Day in loyalty window (1..window_days)
    day = rng.randint(1, rules.window_days)

    # Points sampling logic by tier + profile
    def clamp(x: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, x))

    if tier == Tier.BRONZE:
        # Bronze points live below silver_threshold
        if profile == "Near threshold":
            points = rng.randint(max(0, rules.silver_threshold - 120), rules.silver_threshold - 1)
        elif profile == "Far from threshold":
            points = rng.randint(0, max(1, rules.silver_threshold - 250))
        else:  # Mixed
            points = rng.randint(0, rules.silver_threshold - 1)

        last_window_points = clamp(points + rng.randint(-150, 120), 0, rules.silver_threshold - 1)

    elif tier == Tier.SILVER:
        if profile == "Near threshold":
            points = rng.randint(max(rules.silver_threshold, rules.gold_threshold - 140), rules.gold_threshold - 1)
        elif profile == "Far from threshold":
            points = rng.randint(rules.silver_threshold, max(rules.silver_threshold + 1, rules.gold_threshold - 300))
        else:
            points = rng.randint(rules.silver_threshold, rules.gold_threshold - 1)

        # last window points around silver threshold to create plausible downgrade pressure
        last_window_points = clamp(points + rng.randint(-220, 160), 0, rules.gold_threshold - 1)

    else:  # GOLD
        # Gold points at/above gold_threshold
        points = rng.randint(rules.gold_threshold, rules.gold_threshold + 400)

        # For Gold, last_window_points often dips below gold_threshold -> downgrade risk exists
        if profile == "Near threshold":
            last_window_points = rng.randint(max(0, rules.gold_threshold - 250), rules.gold_threshold + 50)
        elif profile == "Far from threshold":
            last_window_points = rng.randint(rules.gold_threshold, rules.gold_threshold + 300)
        else:
            last_window_points = rng.randint(max(0, rules.gold_threshold - 350), rules.gold_threshold + 250)

    return UserState(tier=tier, day=day, points=points, last_window_points=last_window_points)


def label_scenario(name: str) -> str:
    # Make x-axis labels readable
    return name.replace("_", " ")


def main() -> None:
    st.set_page_config(page_title="Behavioural AI Decision Explorer", layout="wide")
    st.title("Behavioural AI Decision Explorer — Retail Loyalty")

    st.sidebar.header("Controls")

    agent_type = st.sidebar.selectbox("Decision-maker", ["Behavioural", "Bounded-rational", "Rational"], index=0)
    user_profile = st.sidebar.selectbox("User distribution", ["Mixed", "Near threshold", "Far from threshold"], index=0)

    ai_enabled = st.sidebar.toggle("AI advisor enabled", value=True)
    reactance = st.sidebar.slider("Reactance (reject AI under overload)", 0.0, 1.5, 0.6, 0.1, disabled=not ai_enabled)

    n_users = st.sidebar.slider("Simulated users", 200, 6000, 1500, 200)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: choose 'Far from threshold' to avoid trivial automatic upgrades.")

    rules = TierRules()
    scenarios = make_scenarios()

    # Agents
    if agent_type == "Rational":
        agent = RationalAgent()
    elif agent_type == "Bounded-rational":
        agent = BoundedRationalAgent()
    else:
        agent = BehaviouralAgent()

    advisor = SimpleAdvisor()
    ai_response = BehaviouralAIResponse(reactance=reactance)

    # For regret calculations we evaluate using rational utility as a consistent yardstick
    rational_eval = RationalAgent()

    rng = random.Random(int(seed))

    rows = []
    for scenario in scenarios:
        for _ in range(int(n_users)):
            state = sample_user_state(rng, rules, user_profile)

            # User's own choice
            fallback = agent.choose_offer(state, scenario, rules)

            # Advisor recommendation (rational-best)
            rec = advisor.recommend(state, scenario, rules)

            # Final choice (with or without AI)
            if ai_enabled and agent_type == "Behavioural":
                final = ai_response.choose_with_advisor(state, scenario, rules, rec, fallback)
                followed_ai = (final.name == rec.name)
            elif ai_enabled and agent_type != "Behavioural":
                # For non-behavioural agents, keep it simple: they either take their own choice or the advisor’s
                # (this keeps the demo interpretable)
                final = rec
                followed_ai = True
            else:
                final = fallback
                followed_ai = False

            outcome = simulate_outcome(state, scenario, final, rules)

            # Regret: compare to rational-best utility (counterfactual benchmark)
            # realised utility uses the same rational utility function for comparability.
            realised_u = rational_eval.utility(state, scenario, final, rules)
            optimal_offer = rational_eval.choose_offer(state, scenario, rules)
            optimal_u = rational_eval.utility(state, scenario, optimal_offer, rules)
            regret = max(0.0, optimal_u - realised_u)

            rows.append({
                "scenario": label_scenario(scenario.name),
                "agent": agent_type,
                "ai_enabled": ai_enabled,
                "reactance": reactance if ai_enabled else None,
                "tier": state.tier.value,
                "day": state.day,
                "points": state.points,
                "last_window_points": state.last_window_points,
                "final_action": final.name,
                "advisor_action": rec.name,
                "followed_ai": followed_ai,
                "acted": outcome.acted,
                "points_delta": outcome.points_delta,
                "effort": outcome.effort_cost,
                "new_tier": outcome.new_tier,
                "tier_up": (outcome.new_tier != state.tier.value) and (outcome.new_tier in ["Silver", "Gold"]),
                "regret": regret,
            })

    df = pd.DataFrame(rows)

    # ---- Summary KPIs ----
    st.subheader("Key Outcomes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completion rate", f"{df['acted'].mean():.2%}")
    c2.metric("Tier-up rate", f"{df['tier_up'].mean():.2%}")
    c3.metric("AI follow rate", f"{df['followed_ai'].mean():.2%}" if ai_enabled else "—")
    c4.metric("Mean regret", f"{df['regret'].mean():.3f}")

    # ---- Charts ----
    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Completion Rate by Scenario")
        st.bar_chart(df.groupby("scenario")["acted"].mean())

        st.subheader("Tier-up Rate by Scenario")
        st.bar_chart(df.groupby("scenario")["tier_up"].mean())

    with right:
        if ai_enabled:
            st.subheader("AI Follow Rate by Scenario")
            st.bar_chart(df.groupby("scenario")["followed_ai"].mean())

        st.subheader("Regret by Scenario (lower is better)")
        st.bar_chart(df.groupby("scenario")["regret"].mean())

    # ---- Drill-down table ----
    st.markdown("---")
    st.subheader("Sample rows (for debugging and interpretation)")
    st.dataframe(
        df[[
            "scenario", "tier", "points", "day",
            "advisor_action", "final_action",
            "followed_ai", "acted", "new_tier",
            "effort", "regret"
        ]].sample(min(30, len(df)), random_state=int(seed)).reset_index(drop=True),
        use_container_width=True
    )

    st.caption(
        "If rates look too close to 100%, switch 'User distribution' to 'Far from threshold' "
        "and/or increase reactance. The aim is to reveal when AI advice helps versus harms."
    )


if __name__ == "__main__":
    main()
