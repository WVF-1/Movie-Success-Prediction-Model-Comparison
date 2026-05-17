# Movie-Success-Prediction-Model-Comparison
Deriving features and modeling parameters from the three earlier analytics projects to attempt to synthesize which method and combination for predicting movie success is the best.

# 🎬 Predicting Movie Success — A Three-Lens ML Study
### May Newsletter — Movie Intelligence Series (Capstone)

The culmination of the May newsletter series. Three months of analysis established what
makes films financially successful, who drives that success, and whether audiences agree
with the box office. This capstone asks the harder question: could a model trained only
on pre-release information have predicted any of it?

Four parallel Random Forest pipelines each define "success" differently — through the
ROI, revenue, and audience rating lenses of the three preceding projects — and are
evaluated on a genuine temporal holdout of films from the last decade of the dataset.

---

## 📊 Source Data

This project consumes the three parquet files produced across the series. No additional
raw CSVs are required.

| File | Source | Key content |
|------|--------|-------------|
| `movies_clean.parquet` | Part 1, NB1 | ~4,800 films with budget, revenue, ROI, genre, release metadata |
| `directors_clean.parquet` | Part 2, NB4 | Director names, lead actor, cast size, merged with financials |
| `audience_clean.parquet` | Part 3, NB7 | Aggregated user ratings, rating std, vote count, quadrant labels |

---

## 🚀 Running the Notebooks

The notebooks must be run in order. All three parquet files must be present in the
Colab session before running NB10.

**Step 1 — Run NB10**  
Merges the three source parquets, constructs the four target variables (including the
Cinema Success Index), engineers all three feature layers, applies the temporal split,
and saves four feature matrices to `capstone_data/`.

**Step 2 — Run NB11**  
Loads the feature matrices, trains one Random Forest per pipeline using
`TimeSeriesSplit` cross-validation (5 folds), computes permutation feature importance,
and saves trained models and CV results to `capstone_models.pkl`.

**Step 3 — Run NB12**  
Evaluates all four models on the held-out 2008–2017 test set, generates prediction
case studies, and produces four newsletter-ready figures.

---

## 🤖 Model Architecture

### The Four Pipelines

| Pipeline | Target | Feature layers | Key question |
|----------|--------|---------------|--------------|
| **M1 — ROI** | ROI > 1× (binary) | Layer 1 only | Can budget + genre alone predict profitability? |
| **M2 — Revenue** | Revenue ≥ 60th pct (binary) | Layers 1 + 2 | Does knowing the director improve commercial prediction? |
| **M3 — Ratings** | avg_user_rating ≥ median (binary) | Layers 1 + 2 | Does creative attribution predict audience satisfaction? |
| **M4 — CSI** | CSI ≥ 60th pct (binary) | Layers 1 + 2 + 3 | What does the full picture predict? |

All four models use identical Random Forest hyperparameters for a fair comparison:
`n_estimators=500`, `max_features='sqrt'`, `min_samples_leaf=5`,
`class_weight='balanced'`, `random_state=42`.

### Validation Strategy

- **Training set:** Films released before 2008, sorted by release year
- **Test set:** Films released 2008–2017, held out until NB12
- **Cross-validation:** `TimeSeriesSplit` (5 folds) — each fold trains only on films
  earlier than the validation window, respecting the temporal structure of the data
  and preventing information from flowing backwards in time
- **Metrics:** AUC-ROC, F1, Precision, Recall, MSE

---

## 🎯 Target Variables

- **`success_roi`** — Binary. ROI > 1× (film more than doubled its production budget).
  The financially disciplined definition: not just profitable, but genuinely efficient.

- **`success_revenue`** — Binary. Box office revenue in the top 40% of the dataset.
  The commercial hit definition — raw scale rather than efficiency.

- **`success_rating`** — Binary. Average MovieLens user rating at or above the dataset
  median. The audience satisfaction definition, decoupled almost entirely from revenue
  (r ≈ −0.02 as established in Part 3).

- **`success_CSI`** — Binary. Cinema Success Index score in the top 40% of the dataset.
  See the dedicated section below.

---

## 🏆 Cinema Success Index (CSI)

The CSI is the capstone's signature engineered target — a single composite score that
attempts to capture success holistically rather than through any one lens.

### Construction

```
CSI = 0.4 × revenue_percentile_rank
    + 0.4 × rating_percentile_rank
    + 0.2 × vote_count_percentile_rank
```

Each component is expressed as a **percentile rank** across the full dataset (0.0–1.0),
not a raw value. This is the critical design decision: a film at the 90th revenue
percentile earned more than 90% of comparable films, but the gap between it and a film
at the 80th percentile is treated the same as any other 10-percentile gap. This makes
the three components commensurable before weighting, without any one metric dominating
by virtue of its units.

### Component breakdown

- **Revenue percentile rank (weight: 0.4)** — Commercial reach. Where does this film's
  box office sit among all films in the dataset? Captures the scale of the audience that
  actually paid to see the film.

- **Rating percentile rank (weight: 0.4)** — Audience satisfaction. Where does this
  film's average user rating sit? Equally weighted with revenue because the index
  rewards films that achieve both — a blockbuster audiences resented scores no higher
  than a beloved film nobody saw.

- **Vote count percentile rank (weight: 0.2)** — Audience reach multiplier. A film
  that is highly rated by only 12 people is less culturally impactful than one rated
  by 300. Given the lower weight (0.2) because vote count is a proxy for the two primary
  components rather than an independent dimension of success.

### Why percentile ranks?

Revenue runs to hundreds of millions of dollars. Ratings run from 0.5 to 5.0.
Vote counts range from 10 to several hundred. Adding raw values directly would let
revenue dominate by orders of magnitude. Percentile ranking collapses each to the same
0–1 scale while preserving ordinal information — relative standing within the population.

The continuous CSI score is binarised at the 60th percentile for the classification
task, consistent with the revenue target threshold.

---

## 🎬 Director Success Score

The Director Success Score operationalises a director's career track record as a single
feature. It is the centrepiece of Layer 2 and the most information-rich engineered
feature in the project.

### Construction

```
Director Success Score = 0.2 × ROI_norm
                       + 0.3 × Revenue_norm
                       + 0.5 × Rating_norm
```

Each component is normalised to [0, 1] across all directors in the training set before
weighting, so the weights reflect genuine relative importance rather than being
dominated by whichever metric has the largest raw range.

### Component breakdown

- **Normalised average ROI (weight: 0.2)** — Financial discipline across the director's
  filmography. Given the lowest weight because ROI is highly volatile — a single
  micro-budget breakout can permanently inflate a director's score even if subsequent
  films underperform.

- **Normalised average revenue (weight: 0.3)** — Commercial drawing power. Whether the
  director's name on a poster reliably brings audiences into cinemas. More stable than
  ROI across a filmography and reflective of genuine market confidence.

- **Normalised average audience rating (weight: 0.5)** — The highest weight for two
  reasons. First, audience quality assessments are the most stable signal across a
  filmography — less sensitive to budget anomalies than financial metrics. Second, it
  captures something the financial components cannot: whether the director consistently
  makes films people actually value. A director who makes expensive films audiences
  consistently rate poorly is a different risk profile from one whose films are loved
  even on modest budgets.

### Shrinkage prior

For directors with fewer than 5 qualifying films in the training set, the raw weighted
score is blended toward the population mean in proportion to film count:

```
shrinkage_weight        = min(n_films / 5, 1.0)
director_success_score  = shrinkage_weight × raw_score
                        + (1 − shrinkage_weight) × population_mean
```

A director with 1 film gets 20% of their own score and 80% of the population mean.
A director with 3 films gets 60% and 40%. A director with 5 or more films is trusted
fully. The threshold of 5 is deliberately consistent with Part 2 of the series, where
the same cutoff qualified directors for the ROI and revenue leaderboards.

Directors not present in the training set (debut films) receive the population mean —
the model treats them as unknown quantities rather than failures.

---

## 🔧 Feature Reference

### Layer 1 — Base Production Features *(all four models)*

| Feature | Description |
|---------|-------------|
| `log_budget` | Log-transformed production budget. Log scale compresses the right-skewed raw distribution so blockbuster outliers don't dominate. |
| `runtime_clean` | Runtime in minutes, median-imputed for missing values. Proxy for film scale and format. |
| `month_sin`, `month_cos` | Cyclical encoding of release month. Maps the calendar onto a circle so January and December are correctly adjacent. Captures the seasonal ROI patterns from Part 1. |
| `is_english` | Binary. Whether the film's original language is English — a proxy for primary market size and global distribution reach. |
| `genre_[name]` | One-hot encoding of primary genre (top 10 genres; remainder collapsed to Other). Genre was the strongest single predictor in Part 1's financial analysis. |
| `genre_count` | Number of genres listed. Multi-genre films target broader audiences, affecting both marketing reach and positioning. |

### Layer 2 — Creative Features *(M2, M3, M4)*

| Feature | Description |
|---------|-------------|
| `director_success_score` | Weighted composite of the director's career ROI, revenue, and audience rating, with shrinkage prior for directors with < 5 qualifying films. See dedicated section above. |
| `cast_size_clean` | Total credited cast members, median-imputed. Correlates positively with revenue (r ≈ 0.3–0.4), particularly in Action films, as established in Part 2. |
| `has_known_lead` | Binary. Whether the top-billed actor's films averaged in the top quartile of training-set revenue. Operationalises star power using only training-set evidence. |

### Layer 3 — Audience Proxy Features *(M4 only)*

| Feature | Description |
|---------|-------------|
| `vote_average_clean` | TMDB community score (0–10), median-imputed. Correlates with MovieLens user rating at r = 0.84 (Part 3), making it a useful quality proxy even without full behavioural data. |
| `log_popularity` | Log-transformed TMDB popularity metric (aggregates page views, watchlist activity, rating volume). Captures cultural momentum — a different dimension from quality or revenue. |

---

## 📊 Evaluation Figures

| Figure | Description |
|--------|-------------|
| `fig_importance_4panel.png` | Permutation feature importance for all four models in a 2×2 grid |
| `fig_radar_chart.png` | Five-axis radar chart comparing all four models across AUC, F1, Precision, Recall, and CV Stability |
| `fig_histogram_2x2.png` | Predicted probability distributions for each model, split by true class — shows how cleanly each model separates successes from failures |
| `fig_roc_curves.png` | ROC curves for all four models on the test set |

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
scipy
scikit-learn
pyarrow        # for parquet read/write
```

---

## 🗺️ Series Roadmap

| Part | Focus | Status |
|------|-------|--------|
| 1 — Financial Performance | Genre ROI, budget tiers, seasonal patterns | ✅ Complete |
| 2 — Cast, Crew & Keywords | Director ROI, actor profitability, keyword analysis | ✅ Complete |
| 3 — Audience Ratings Behaviour | Commercial success vs audience satisfaction | ✅ Complete |
| **Capstone — Predictive Intelligence** | Four-model RF comparison across all three success lenses | ✅ Complete |

---
