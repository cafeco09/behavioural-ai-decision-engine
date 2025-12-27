<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>README — Behavioural AI Decision Engine (Explainability Buffer)</title>
  <style>
    :root{
      --bg:#0b0f14; --card:#111826; --text:#e7eef8; --muted:#a8b3c7;
      --link:#7db2ff; --border:rgba(255,255,255,.10);
      --code:#0e1623; --accent:#69f0ae;
    }
    *{box-sizing:border-box}
    body{margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         background:var(--bg); color:var(--text); line-height:1.55;}
    .wrap{max-width:980px; margin:0 auto; padding:32px 18px 64px;}
    header{padding:22px 22px; border:1px solid var(--border); border-radius:16px;
           background:linear-gradient(180deg, rgba(125,178,255,.12), rgba(17,24,38,1));}
    h1{margin:0 0 8px; font-size:26px;}
    .subtitle{margin:0; color:var(--muted)}
    .badges{display:flex; gap:10px; flex-wrap:wrap; margin-top:14px}
    .badge{font-size:12px; padding:6px 10px; border:1px solid var(--border);
           border-radius:999px; background:rgba(255,255,255,.04); color:var(--muted)}
    a{color:var(--link); text-decoration:none}
    a:hover{text-decoration:underline}
    section{margin-top:18px; padding:18px 22px; border:1px solid var(--border);
            border-radius:16px; background:rgba(255,255,255,.03)}
    h2{margin:0 0 8px; font-size:18px}
    h3{margin:16px 0 8px; font-size:15px; color:var(--text)}
    p{margin:10px 0; color:var(--text)}
    ul{margin:8px 0 0 18px; color:var(--text)}
    li{margin:6px 0}
    .muted{color:var(--muted)}
    .callout{border-left:4px solid var(--accent); padding:10px 12px; background:rgba(105,240,174,.06);
             border-radius:10px; margin:10px 0}
    pre{margin:10px 0; padding:14px 14px; background:var(--code); border:1px solid var(--border);
        border-radius:12px; overflow:auto}
    code{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
         font-size:13px; color:#dbe7ff}
    .grid{display:grid; grid-template-columns:1fr; gap:14px}
    @media (min-width: 860px){
      .grid{grid-template-columns: 1fr 1fr}
    }
    .k{color:#b9c6e4}
    footer{margin-top:22px; color:var(--muted); font-size:12px}
  </style>
</head>

<body>
  <div class="wrap">
    <header>
      <h1>Behavioural AI Decision Engine — Explainability Buffer Simulation</h1>
      <p class="subtitle">
        Code and interactive dashboard accompanying the paper:
        <strong>“Mitigating Psychological Reactance in AI-Assisted Decision Making: A Monte Carlo Simulation Study of the Explainability Buffer.”</strong>
      </p>

      <div class="badges">
        <span class="badge">Python</span>
        <span class="badge">Streamlit Dashboard</span>
        <span class="badge">Monte Carlo Simulation</span>
        <span class="badge">Explainability × Reactance</span>
        <span class="badge">MIT License (2025)</span>
      </div>

      <p style="margin-top:14px" class="muted">
        Repository: <a href="https://github.com/cafeco09/behavioural-ai-decision-engine">https://github.com/cafeco09/behavioural-ai-decision-engine</a>
      </p>
    </header>

    <section>
      <h2>Overview</h2>
      <p>
        This repository implements a behavioural simulation of human decision-making with an AI adviser. The core focus is
        <strong>psychological reactance</strong> (resistance to perceived autonomy threats) and a design hypothesis called the
        <strong>Explainability Buffer</strong>, where higher explainability reduces effective autonomy threat and preserves AI utility
        under disagreement.
      </p>

      <div class="callout">
        <strong>What you can do here</strong>
        <ul>
          <li>Run an interactive <strong>Streamlit dashboard</strong> to explore outcomes across reactance and explainability settings.</li>
          <li>Run batch simulations to reproduce paper-style figures and summary metrics.</li>
          <li>Inspect and extend the simulation primitives (trust, signal strength, disagreement, regret, completion).</li>
        </ul>
      </div>
    </section>

    <section>
      <h2>Repository structure</h2>
      <pre><code>.
├── app.py                         # Streamlit dashboard (interactive exploration)
├── src/                           # Simulation engine modules (agents, advisor, environment, evaluation)
├── outputs/                       # Optional: generated figures, tables, and run artefacts (recommended)
├── runs/                          # Optional: run-by-run archives saved by the dashboard (recommended)
├── requirements.txt               # Dependencies (recommended to pin versions)
├── LICENSE                        # MIT License (2025)
└── README.html / README.md        # Documentation
</code></pre>
      <p class="muted">
        If your repo differs slightly, update this section to match your current folders. The key is to make it easy
        for readers to find the dashboard entry point and the figure-generation logic.
      </p>
    </section>

    <section>
      <h2>Quickstart</h2>

      <div class="grid">
        <div>
          <h3>1) Create a virtual environment</h3>
          <pre><code>python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1</code></pre>
        </div>

        <div>
          <h3>2) Install dependencies</h3>
          <pre><code>pip install -r requirements.txt</code></pre>
          <p class="muted">
            If you do not have <code>requirements.txt</code> yet, create one and pin versions for reproducibility.
          </p>
        </div>
      </div>

      <h3>3) Run the dashboard</h3>
      <pre><code>streamlit run app.py</code></pre>

      <p class="muted">
        The dashboard allows you to toggle explainability settings, sweep reactance levels, and export plots and run artefacts
        (recommended improvements below).
      </p>
    </section>

    <section>
      <h2>Reproducing paper results</h2>
      <p>
        For arXiv-friendly reproducibility, it helps to provide a non-interactive script that generates the figures and tables.
        If you add a script like <code>scripts/reproduce_paper.py</code>, document it here.
      </p>

      <h3>Suggested command</h3>
      <pre><code>python scripts/reproduce_paper.py --out outputs/</code></pre>

      <h3>Expected outputs</h3>
      <ul>
        <li><span class="k">Figures:</span> <code>outputs/fig_*.pdf</code> (or PNG)</li>
        <li><span class="k">Tables:</span> <code>outputs/summary.csv</code> and/or <code>outputs/summary.tex</code></li>
        <li><span class="k">Configs:</span> <code>outputs/run_config.json</code> (reactance grid, seeds, parameters)</li>
      </ul>

      <p class="muted">
        If you already generate figures inside the dashboard, you can wrap the same functions in a script so reviewers can reproduce results without clicking through UI.
      </p>
    </section>

    <section>
      <h2>Key concepts</h2>
      <ul>
        <li><strong>Explainability setting (<code>E</code>)</strong>: a model input, for example <code>E=0.2</code> (low) vs <code>E=0.7</code> (high).</li>
        <li><strong>Consideration probability (<code>P<sub>cons</sub></code>)</strong>: an output metric. It can exceed 0.85 and approach 0.9 under low reactance.</li>
        <li><strong>Reactance (<code>λ</code>)</strong>: resistance to advice due to perceived autonomy threat.</li>
        <li><strong>Explainability Buffer</strong>: a modelling assumption where higher explainability attenuates effective reactance.</li>
      </ul>

      <p class="muted">
        Tip: readers sometimes confuse <code>E=0.7</code> with “70% consideration”. Make the distinction explicit in your paper and in tooltips/labels in the dashboard.
      </p>
    </section>

    <section>
      <h2>Recommended improvements for paper-grade polish</h2>

      <h3>1) Save each run (reproducibility)</h3>
      <ul>
        <li>Write run artefacts to <code>runs/&lt;timestamp&gt;/</code>: <code>params.json</code>, <code>metrics.csv</code>, and exported figures.</li>
        <li>Include seed values and the reactance grid in the saved config.</li>
      </ul>

      <h3>2) Add download buttons in Streamlit (usability)</h3>
      <ul>
        <li><code>st.download_button</code> for <code>metrics.csv</code>, and for a ZIP of figures.</li>
        <li>Make “Archive plots” explicit so users know where files are saved.</li>
      </ul>

      <h3>3) Add citation metadata</h3>
      <ul>
        <li>Add <code>CITATION.cff</code> so GitHub shows a “Cite this repository” button.</li>
        <li>Tag a release, for example <code>v1.0.0</code>, aligned to the arXiv preprint version.</li>
      </ul>

      <h3>4) Pin dependencies</h3>
      <ul>
        <li>Use <code>requirements.txt</code> with pinned versions, or <code>environment.yml</code>.</li>
        <li>Document the Python version you tested with.</li>
      </ul>
    </section>

    <section>
      <h2>Licence</h2>
      <p>
        Released under the <strong>MIT License</strong> (2025). See <code>LICENSE</code> in this repository.
      </p>
      <p class="muted">
        When you reference the code in the paper, you can add: “Code and materials are available on GitHub under the MIT License.”
      </p>
    </section>

    <section>
      <h2>How to cite</h2>
      <p>
        If you add a <code>CITATION.cff</code> file, GitHub will auto-generate a citation. Until then, you can cite the paper and link to this repository.
      </p>
      <pre><code>@misc{dixit2025explainabilitybuffer,
  title        = {Mitigating Psychological Reactance in AI-Assisted Decision Making: A Monte Carlo Simulation Study of the Explainability Buffer},
  author       = {Dixit, Anshul},
  year         = {2025},
  howpublished = {\url{https://github.com/cafeco09/behavioural-ai-decision-engine}},
  note         = {Code and materials (MIT License)}
}</code></pre>
      <p class="muted">
        Replace the BibTeX key and details with your final arXiv identifier once it is published.
      </p>
    </section>

    <section>
      <h2>Contact</h2>
      <p>
        For questions or collaboration: <a href="mailto:anshuldixit589@gmail.com">anshuldixit589@gmail.com</a>.
      </p>
    </section>

    <footer>
      <p>
        This README is provided in HTML format for easy copying into documentation sites or converting to Markdown.
        If you want, I can also generate a matching <code>README.md</code> version with standard GitHub formatting.
      </p>
    </footer>
  </div>
</body>
</html>
