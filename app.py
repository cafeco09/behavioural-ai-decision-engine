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


def sample_user_state(rng: random.Random, rules: TierRules, profile: str) -> UserState:
    """
    Sample heterogeneous users so outcomes are non-trivial.
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
            points = rng.randint(rules.silver_threshold, max(rules.silver_threshold + 1, rules.gold_threshold - 300))
        else:
            points = rng.randint(rules.silver_threshold, rules.gold_threshold - 1)

        last_window_points = clamp(points + rng.randint(-220, 160), 0, rules.gold_threshold - 1)

    else:  # GOLD
        points = rng.randint(rules.gold_threshold, rules.gold_threshold + 400)

        if profile == "Near threshold":
            last_window_points = rng.randint(max(0, rules.gold_threshold - 250), rules.gold_threshold + 50)
        elif profile == "Far from threshold":
            last_window_points = rng.randint(rules.gold_threshold, rules.gold_threshold + 300)
        else:
            last_window_points = rng.randint(max(0, rules.gold_threshold - 350), rules.gold_threshold + 250)

    return UserState(
        tier=tier,
        day=day,
        points=points,
        last_window_points=last_window_points
    )


def main():
    st.set_page_config(
        page_title="Behavioural AI Decision Explorer",
        layout="wide"
    )

    st.title("Behavioural AI Decision Explorer — Retail Loyalty")

    # ======================
    # Sidebar controls
    # ======================
    st.sidebar.header("Simulation Controls")

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
    st.sidebar.caption(
        "Tip: choose 'Far from threshold' to avoid trivial upgrades."
    )

    # ======================
    # Setup
    # ======================
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
    rational_eval = RationalAgent()

    # ======================
    # Simulation
    # ======================
    rows = []

    for scenario in scenarios:
        for _ in range(int(n_users)):
            state = sample_user_state(rng, rules, user_profile)

            fallback = agent.choose_offer(state, scenario, rules)
            rec = advisor.recommend(state, scenario, rules)

            if ai_enabled and agent_type == "Behavioural":
                final = ai_response.choose_with_advisor(
                    state, scenario, rules, rec, fallback
                )
                followed_ai = (final.name == rec.name)
            elif ai_enabled:
                final = rec
                followed_ai = True
            else:
                final = fallback
                followed_ai = False

            outcome = simulate_outcome(state, scenario, final, rules)

            realised_u = rational_eval.utility(state, scenario, final, rules)
            optimal = rational_eval.choose_offer(state, scenario, rules)
            optimal_u = rational_eval.utility(state, scenario, optimal, rules)
            regret = max(0.0, optimal_u - realised_u)

            rows.append({
                "scenario": scenario.name.replace("_", " "),
                "tier": state.tier.value,
                "acted": outcome.acted,
                "tier_up": outcome.new_tier != state.tier.value,
                "followed_ai": followed_ai,
                "effort": outcome.effort_cost,
                "regret": regret
            })

    df = pd.DataFrame(rows)

    # ======================
    # KPIs
    # ======================
    st.subheader("Key Outcomes")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completion rate", f"{df['acted'].mean():.2%}")
    c2.metric("Tier-up rate", f"{df['tier_up'].mean():.2%}")
    c3.metric("AI follow rate", f"{df['followed_ai'].mean():.2%}" if ai_enabled else "—")
    c4.metric("Mean regret", f"{df['regret'].mean():.3f}")

    # ======================
    # Charts
    # ======================
    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Completion Rate by Scenario")
        st.bar_chart(df.groupby("scenario")["acted"].mean())

        st.subheader("Tier Upgrade Rate by Scenario")
        st.bar_chart(df.groupby("scenario")["tier_up"].mean())

    with right:
        if ai_enabled:
            st.subheader("AI Follow Rate by Scenario")
            st.bar_chart(df.groupby("scenario")["followed_ai"].mean())

        st.subheader("Regret by Scenario (lower is better)")
        st.bar_chart(df.groupby("scenario")["regret"].mean())

    # ======================
    # Debug table
    # ======================
    st.markdown("---")
    st.subheader("Sample decisions")

    st.dataframe(
        df.sample(min(30, len(df)), random_state=int(seed)).reset_index(drop=True),
        use_container_width=True
    )

    st.caption(
        "If results still look flat, switch to 'Far from threshold' users "
        "and increase reactance to surface AI failure modes."
    )


if __name__ == "__main__":
    main()
