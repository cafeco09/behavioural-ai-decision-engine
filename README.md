# Behavioural AI Decision Engine — Explainability Buffer Simulation

Code and interactive dashboard accompanying the paper:  
**“Mitigating Psychological Reactance in AI-Assisted Decision Making: A Monte Carlo Simulation Study of the Explainability Buffer.”**

**Repository:** https://github.com/cafeco09/behavioural-ai-decision-engine  
**Licence:** MIT (2025)

---

## Overview
This repository implements a behavioural simulation of human decision-making with an AI adviser. The core focus is **psychological reactance** (resistance to perceived autonomy threats) and a design hypothesis called the **Explainability Buffer**, where higher explainability reduces effective autonomy threat and preserves AI utility under disagreement.

### What you can do here
- Run an interactive **Streamlit dashboard** to explore outcomes across reactance and explainability.
- Run batch simulations to reproduce paper-style figures and summary metrics.
- Inspect and extend the simulation primitives (trust, signal strength, disagreement, regret, completion).

---

## Repository structure
```text
.
├── app.py                 # Streamlit dashboard
├── src/                   # Simulation engine modules
├── outputs/               # (recommended) generated figures and tables
├── runs/                  # (recommended) run-by-run archives from the dashboard
├── requirements.txt       # dependencies (recommended to pin versions)
├── LICENSE                # MIT License (2025)
└── README.md
