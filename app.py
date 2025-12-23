import streamlit as st
import pandas as pd

from src.environment.scenarios import TierRules, Tier, UserState, make_scenarios
from src.agents.behavioural import BehaviouralAgent
from src.agents.ai_response import BehaviouralAIResponse
from src.advisor.simple_advisor import SimpleAdvisor
from src.evaluation.metrics import simulate_outcome

st.set_page_config(page_title="Behavioural AI Decision Explorer", layout="wide")

st.title("Behavioural AI Decision Explorer — Retail Loyalty")

st.sidebar.header("Simulation Controls")
reactance = st.sidebar.slider("Reactance", 0.0, 1.5, 0.6, 0.1)
n_users = st.sidebar.slider("Simulated users", 100, 2000, 500, 100)

rules = TierRules()
scenarios = make_scenarios()

agent = BehaviouralAgent()
advisor = SimpleAdvisor()
ai_response = BehaviouralAIResponse(reactance=reactance)

rows = []

for scenario in scenarios:
    for _ in range(n_users):
        state = UserState(
            tier=Tier.SILVER,
            day=22,
            points=820,
            last_window_points=880,
        )

        fallback = agent.choose_offer(state, scenario, rules)
        rec = advisor.recommend(state, scenario, rules)
        final = ai_response.choose_with_advisor(state, scenario, rules, rec, fallback)

        outcome = simulate_outcome(state, scenario, final, rules)

        rows.append({
            "scenario": scenario.name,
            "acted": outcome.acted,
            "tier_up": outcome.new_tier == "Gold",
            "effort": outcome.effort_cost,
            "followed_ai": final.name == rec.name,
        })

df = pd.DataFrame(rows)

st.subheader("Completion Rate")
st.bar_chart(df.groupby("scenario")["acted"].mean())

st.subheader("Tier Upgrade Rate")
st.bar_chart(df.groupby("scenario")["tier_up"].mean())

st.subheader("AI Follow Rate")
st.bar_chart(df.groupby("scenario")["followed_ai"].mean())
