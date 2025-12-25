import random
import pandas as pd
import streamlit as st

from src.environment.scenarios import TierRules, Tier, UserState, make_scenarios
from src.agents.rational import RationalAgent
from src.agents.bounded import BoundedRationalAgent
from src.agents.behavioural import BehaviouralAgent
from src.agents.ai_response import BehaviouralAIResponse
from src.advisor.simple_advisor import SimpleAdvisor
from src.evaluation.metrics import simulate_outcome


def decides_to_act(
    effort: float,
    scenario_name: str,
    reactance: float,
    rng: random.Random
) -> bool:
    """
    Probability-based decision to act vs procrastinate.
    Keeps behaviour interpretable and breaks the '100% completion' artefact.
    """
    # Baseline motivation (conditioned on being an 'engaged' loyalty user)
    p_act = 0.95

    # Effort reduces action propensity
    p_act -= 0.20 * float(effort)

    # Overload increases avoidance
    if "Overload" in scenario_name:
        p_act -= 0.25

    # Reactance increases disengagement (only relevant when AI is present)
    p_act -= 0.15 * float(reactance)

    # Clamp to a sensible range
    p_act = max(0.05, min(0.95, p_act))

    return rng.random() < p_act


def sample_user_state(rng: random.Random, rules: TierRules, profile: str) -> UserState:
    """
    Samples heterogeneous users so outcomes are non-trivial.
    """
    tier = rng.choices(
        [Tier.BRONZE, Tier.SILVER, Tier.GOLD],
        weights=[0.45, 0.40, 0.15],
        k=1
    )[0]

    day = rng.randint(1, rules.window_days)

    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    if tier == Tier.BRONZE:
        if profile == "Near threshold":
            points = rng.randint(max(0, rules.silver_threshold - 120), rules.silver_threshold - 1)
        elif profile == "Far from threshold":
            points = rng.randint(0, max(1, rules.silver_threshold - 250))
        else:
            points = rng.randint(0, rules.silver_threshold - 1)
        last_window_points = clamp(points + rng.randint(-150, 120), 0, rules.silver_threshold - 1)

    elif tier == Tier.SILVER:
        if profile == "Near threshold":
            points = rng.randint(max(rules.silver_threshold, rules.gold_threshold - 140), rules.gold_threshold - 1)
        elif profile == "Far from threshold":
            points = rng.randint(
                rules.silver_threshold,
                max(rules.silver_threshold + 1, rules.gold_threshold - 300),
            )
        else:
            points = rng.randint(rules.silver_threshold, rules.gold_threshold - 1)
        last_window_points = clamp(points + rng.randint(-220, 160), 0, rules.gold_threshold - 1)

    else:  # GOLD
        points = rng.randint(rules.gold_threshold, rules.gold_threshold + 400)
        if profile == "Near threshold":
            last_window_points = rng.randint(
                max(0, rules.gold_threshold - 250), rules.gold_threshold + 50
            )
        elif profile == "Far from threshold":
            last_window_points = rng.randint(
                rules.gold_threshold, rules.gold_threshold + 300
            )
        else:
            last_window_points = rng.randint(
                max(0, rules.gold_threshold - 350), rules.gold_threshold + 250
            )

    return UserState(
        tier=tier,
        day=day,
        points=points,
        last_window_points=last_window_points,
    )


def main():
    st.set_page_config(page_title="Behavioural AI Decision Explorer", layout="wide")
    st.title("Behavioural AI Decision Explorer — Retail Loyalty")

    # -----------------------
    # Sidebar controls
    # -----------------------
    st.sidebar.header("Controls")

    agent_type = st.sidebar.selectbox(
        "Decision-maker",
        ["Behavioural", "Bounded-rational", "Rational"],
        index=0
    )

    user_profile = st.sidebar.selectbox(
        "User distribution",
        ["Mixed", "Near threshold", "Far from threshold"],
        index=0
    )

    ai_enabled = st.sidebar.checkbox("AI advisor enabled", value=True)

    reactance = st.sidebar.slider(
        "Reactance (reject AI under overload)",
        0.0, 1.5, 0.6, 0.1,
        disabled=not ai_enabled
    )

    n_users = st.sidebar.slider(
        "Simulated users",
        200, 6000, 1500, 200
    )

    seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: choose 'Far from threshold' to reduce automatic upgrades.")

    # -----------------------
    # Setup
    # -----------------------
    rules = TierRules()
    scenarios = make_scenarios()
    rng = random.Random(int(seed))

    if agent_type == "Rational":
        agent = RationalAgent()
    elif agent_type == "Bounded-rational":
        agent = BoundedRationalAgent()
    else:
        agent = BehaviouralAgent()

    advisor = SimpleAdvisor()
    ai_response = BehaviouralAIResponse(reactance=reactance)

    # Use a rational yardstick for regret (counterfactual benchmark)
    rational_eval = RationalAgent()

    # -----------------------
    # Simulation
    # -----------------------
    rows = []

    for scenario in scenarios:
        for _ in range(int(n_users)):
            state = sample_user_state(rng, rules, user_profile)

            fallback = agent.choose_offer(state, scenario, rules)
            rec = advisor.recommend(state, scenario, rules)

            if ai_enabled and agent_type == "Behavioural":
                final = ai_response.choose_with_advisor(state, scenario, rules, rec, fallback)
                followed_ai = (final.name == rec.name)
                reactance_used = reactance
            elif ai_enabled:
                # Keep non-behavioural demos simple/clean: follow advisor directly
                final = rec
                followed_ai = True
                reactance_used = 0.0
            else:
                final = fallback
                followed_ai = False
                reactance_used = 0.0

            # -----------------------
            # NEW: action vs no-action
            # -----------------------
            acts = decides_to_act(
                effort=getattr(final, "effort_cost", 1.0),
                scenario_name=scenario.name,
                reactance=reactance_used,
                rng=rng
            )

            if acts:
                outcome = simulate_outcome(state, scenario, final, rules)

                realised_u = rational_eval.utility(state, scenario, final, rules)
                optimal_offer = rational_eval.choose_offer(state, scenario, rules)
                optimal_u = rational_eval.utility(state, scenario, optimal_offer, rules)
                regret = max(0.0, optimal_u - realised_u)
            else:
                # Explicit procrastination / no-action outcome
                class NoActionOutcome:
                    acted = False
                    points_delta = 0
                    effort_cost = 0.0
                    new_tier = state.tier.value

                outcome = NoActionOutcome()

                # Regret of inaction: foregone optimal utility
                optimal_offer = rational_eval.choose_offer(state, scenario, rules)
                optimal_u = rational_eval.utility(state, scenario, optimal_offer, rules)
                regret = max(0.0, optimal_u)

            # Strict tier-up definition (no accidental downgrade counting)
            tier_up = (
                (state.tier.value == "Bronze" and outcome.new_tier == "Silver")
                or (state.tier.value == "Silver" and outcome.new_tier == "Gold")
            )
            tier_down = (
                (state.tier.value == "Gold" and outcome.new_tier == "Silver")
                or (state.tier.value == "Silver" and outcome.new_tier == "Bronze")
            )

            rows.append({
                "scenario": scenario.name.replace("_", " "),
                "tier": state.tier.value,
                "points": state.points,
                "day": state.day,
                "advisor_action": rec.name,
                "final_action": final.name,
                "followed_ai": followed_ai,
                "acted": outcome.acted,
                "new_tier": outcome.new_tier,
                "tier_up": tier_up,
                "tier_down": tier_down,
                "effort": getattr(outcome, "effort_cost", 0.0),
                "regret": regret
            })

    df = pd.DataFrame(rows)

    # -----------------------
    # KPIs
    # -----------------------
    st.subheader("Key Outcomes")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Completion rate", f"{df['acted'].mean():.2%}")
    c2.metric("Tier-up rate", f"{df['tier_up'].mean():.2%}")
    c3.metric("AI follow rate", f"{df['followed_ai'].mean():.2%}" if ai_enabled else "—")
    c4.metric("Mean regret", f"{df['regret'].mean():.3f}")

    # -----------------------
    # Charts
    # -----------------------
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

    # -----------------------
    # Table (old Streamlit compatible: no use_container_width)
    # -----------------------
    st.markdown("---")
    st.subheader("Sample decisions")

    st.dataframe(
        df.sample(min(30, len(df)), random_state=int(seed)).reset_index(drop=True)
    )

    st.caption(
        "If completion is still too high, increase reactance and/or switch to 'Offer Overload' scenarios "
        "by adjusting your scenario definitions. This model now includes explicit procrastination."
    )


if __name__ == "__main__":
    main()
