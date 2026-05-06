# Premise Health Case Study — Predicting (and Preventing) High-Cost Members

> A 24-hour data-science case study: explore healthcare member data, build a model
> that predicts high-cost members from **non-cost** signals, and wrap the model in
> a small **agentic AI Care Navigator** that drafts per-member preventive briefings.

## The question

A small share of members drives most of the spend in any healthcare population.
Predicting cost from cost is trivial — and useless for prevention. The interesting
question is:

> *What behavioural / clinical patterns separate high-cost members from everyone
> else, and can we spot those patterns early in currently-healthy members so we
> can intervene before costs accrue?*

This repo answers that end-to-end.

## Headline results

| Metric | Value | Meaning |
|---|---:|---|
| Cost concentration | **66%** | of total spend driven by the top 10% of members |
| Model PR-AUC (GBM) | **0.95** | vs 0.10 random baseline — strongly predictive |
| Recall @ 0.5 threshold | **89%** | of true high-cost members are flagged |
| Outreach efficiency | **89%** | of high-cost members captured by reaching just the top 10% predicted (8.9× lift) |

The model uses **only behavioural / clinical features** (utilization patterns,
diagnosis mix, chronic burden, survey sentiment) — **no dollar aggregates** — so
the same scoring generalises to currently-healthy members showing early-warning
patterns.

## Repo structure

```
.
├── README.md                       <- this file
├── case_study.ipynb                <- main deliverable: clean, end-to-end analysis
├── 01_Preparation.ipynb            <- exploratory prep notebook (raw cleaning + EDA iteration)
├── data/
│   ├── data_original/              <- original case-study CSVs (untouched)
│   ├── survey_clean.csv            <- cleaned survey table
│   ├── claims_clean.csv            <- cleaned claim-line table
│   └── member_claim_agg.csv        <- per-member cost aggregates
└── Case Study Prompt, final.pdf    <- original prompt
```

The **`case_study.ipynb`** is the polished walkthrough — read this first. The
`01_Preparation.ipynb` shows the messier exploratory work and is included only
for transparency.

## Running it

```bash
# 1. Set up environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn nltk

# Optional but recommended:
pip install shap                   # for SHAP plots in §5.3
pip install anthropic              # for live Claude agent in §6
python -c "import nltk; nltk.download('vader_lexicon')"  # for sentiment

# 2. Open the notebook
jupyter lab case_study.ipynb
# or:
code case_study.ipynb
```

The notebook is **self-contained** — it loads only from `data/` and runs end-to-end
with no extra setup. If `shap` or `nltk`/VADER isn't installed, the notebook
falls back gracefully (built-in feature importance + tiny keyword-based sentiment).

### Live Care Navigator agent (optional)

The agent in §6 runs in two modes:

- **Offline (default).** Deterministic Python loop that calls the same tools in
  the same order; no API key required. Good for reproducibility and demos.
- **Live (Claude).** Set `ANTHROPIC_API_KEY` in your environment and `pip install
  anthropic`. The notebook's `run_agent()` will switch to a live Claude tool-use
  loop using `claude-sonnet-4-6`.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## What's in the notebook

| Section | What |
|---|---|
| §1–2 | Setup, load, brief cleaning recap |
| §3   | Inner-join survey + claims, feature engineering (utilization, diagnosis & procedure mix, time-span, sentiment), define `high_cost_flag = top 10% by total_paid` |
| §4   | EDA highlights — six visuals: cost concentration (Lorenz/Pareto), cost vs chronic burden, place-of-service mix, diagnosis spend, sentiment, risk × cost |
| §5   | Logistic regression vs gradient boosting; ROC/PR curves; confusion matrix, calibration, cumulative-gain/lift; permutation importance + SHAP |
| §6   | Agentic AI Care Navigator with four tools (`get_member_profile`, `predict_high_cost_risk`, `explain_top_drivers`, `recommend_intervention`) |
| §7   | Takeaways: clinical view, operations view, executive panel, recommendations |

## Modeling decisions

- **Target.** `high_cost_flag = 1` if `total_paid` ≥ 90th percentile of cohort (~10% positive rate). Standard "rising-risk" cutoff used by health plans for care-management triage.
- **Leakage rule.** Anything derived from paid/allowed dollars in the same period as the target is dropped from the feature matrix (`total_paid`, `total_allowed`, `Concurrent Risk`, etc.). This keeps the model honest and makes it deployable on healthy members.
- **Two models.** Logistic regression for an interpretable baseline; `HistGradientBoostingClassifier` (no extra deps, handles `NaN` natively) for non-linear interactions.
- **Primary metric.** Average Precision (PR-AUC). With 10% positive rate, ROC-AUC overstates performance and accuracy is misleading.
- **Imbalance.** Both models use `class_weight='balanced'`; no resampling. Threshold tuning is left to the operations team based on outreach capacity.

## Top drivers the model surfaces

These are the signals that distinguish high-cost members — and the prevention
levers for healthy members showing the same early patterns:

- `n_inpatient`, `n_hospital_outpatient` — hospital-based care utilization
- `n_distinct_specialities` — care fragmentation
- `proc_drugs`, `proc_imaging_radiology` — heavy medical-imaging / drug usage
- `pct_primary_outpatient` — *low* primary-care share is a warning, not a good sign
- `dx_musculoskeletal`, `dx_cardiac`, `dx_endocrine_metabolic` — top spend categories
- `# of Chronic Conditions` + `chronic_low_primary` composite flag
- `survey_sentiment`, `survey_complaint_flag` — patient-experience signal

## Limitations

- **No temporal split.** The data is one period. Ideally we'd train on year-T features against year-(T+1) cost. The current setup is concurrent prediction — useful for *characterising* the cohort and *ranking* members for outreach, but not strictly forward-looking.
- **Cohort bias.** The inner join drops survey-only and claims-only members; results don't apply to those segments without revisiting the join strategy.
- **Shallow text features.** VADER sentiment + a small keyword list is robust but doesn't extract themes. An LLM-based topic-extraction pass would improve §3.5.
- **Calibration could be tighter** with isotonic / Platt scaling if the probabilities are used directly in clinical-decision rules.

## Stack

`Python 3.12` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` ·
`nltk` (VADER) · `shap` (optional) · `anthropic` (optional, for the live agent)

— *Yirang Liu · Data Scientist candidate, Premise Health*
