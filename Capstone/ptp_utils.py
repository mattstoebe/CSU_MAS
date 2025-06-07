"""
Utilities for patient-level risk scoring + score-card visualisation
==================================================================
Compatible with the `model_package` you saved in training:
    { 'pipeline': Pipeline(...),
      'thresholds': {'optimal_threshold', 'tau_low', 'tau_high'},
      'targets':    {'P_TARGET', 'R_TARGET'}, ... }
"""

from pathlib import Path
import joblib, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, shap
from matplotlib.lines import Line2D


# Data Loaders ──────────────────────────────────────────────────────────────────────────────────────
def load_model(pkg_path: str | Path):
    """Return pipeline, thresholds dict, & metadata."""
    pkg = joblib.load(pkg_path)
    return pkg["pipeline"], pkg["thresholds"], pkg.get("metadata", {})

def get_data(
        PATIENT_ID = None,
        MODEL_PKG_PATH  = "models/xgb_model.pkl",
        PATIENT_FILE    = "tbistaa556_validate.csv",      # or query DB, etc.
        COHORT_FILE     = "tbistaa556_training.csv"
        ):


    # ── load artefacts ──
    pipe, thr, meta  = load_model(MODEL_PKG_PATH)

    # ── load data rows exactly as you did in prediction phase ──
    dtype_dict = {                          # << copy-pasted from training
        'age':'float64','race7':'category','ethnic3':'category','sex2':'category',
        'sig_other':'category','tobacco':'category','alcohol':'category','drugs':'category',
        'MedianIncomeForZip':'float64','PercentAboveHighSchoolEducationForZip':'float64',
        'PercentAboveBachelorsEducationForZip':'float64','payertype':'category',
        'tbiS02':'float64','tbiS06':'float64','tbiS09':'float64',
        'ptp1_yn':'category','ptp2_yn':'category','ptp0_yn':'category',
        'ed_yn':'category','icu':'category','delirium':'category','agitated':'category',
        'lethargic':'category','comatose':'category','disoriented':'category',
        'gcs_min':'float64','gcs_max':'float64','adl_min':'float64','adl_max':'float64',
        'mobility_min':'float64','mobility_max':'float64','los_total':'float64',
        'dc_setting':'category','prehosp':'category','posthosp':'category',
        'subj_id':'int64'
    }
    df  = (pd.read_csv(PATIENT_FILE,
                    dtype=dtype_dict,
                    na_values=[' '])
            .set_index("subj_id"))

    cohort = pd.read_csv(COHORT_FILE, dtype=dtype_dict,                    
                        na_values=[' ']
        ).set_index("subj_id")

    if PATIENT_ID is None:
            # Random selection
            PATIENT_ID = np.random.choice(df.index)
            print(f"Randomly selected patient ID: {PATIENT_ID}")
        
    try:
        row = df.loc[PATIENT_ID]
    except KeyError:
        # Fallback to random if specified ID not found
        PATIENT_ID = np.random.choice(df.index)
        row = df.loc[PATIENT_ID]
        print(f"Patient not found. Using random patient ID: {PATIENT_ID}")
        
    return PATIENT_ID, pipe, thr, row, cohort  

# ── Prediction helpers ─────────────────────────────────────────────────────────
def predict_proba_single(pipe, row: pd.Series | pd.DataFrame) -> float:
    """Return scalar P(y=1) for one patient row (Series or single-row DF)."""
    if isinstance(row, pd.Series):
        row = row.to_frame().T
    return float(pipe.predict_proba(row)[:, 1])

def risk_band(p: float, τ_low: float, τ_high: float) -> str:
    return ("Low" if p < τ_low else
            "High" if p >= τ_high else
            "Moderate")

# ── Shap Helpers ─────────────────────────────────────────────────────────

def analyze_top_predictors_single(row_raw          : pd.Series | pd.DataFrame,
                                  pipe,
                                  X_train_raw      : pd.DataFrame,
                                  top_n            : int   = 5,
                                  background_size  : int   = 500):
    """
    Compute the top_n SHAP probability drivers for ONE patient record,
    but don’t actually show or style the plot here—just return table, fig, ax.

    Returns
    -------
    table : pd.DataFrame
      columns = [Feature, Impact, Value, Direction]
    fig   : matplotlib.figure.Figure
    ax    : matplotlib.axes.Axes
    """

    # 1) unwrap single row
    row_df = (row_raw
              if isinstance(row_raw, pd.DataFrame)
              else row_raw.to_frame().T)

    row_proc = pipe[:-1].transform(row_df)
    if hasattr(row_proc, "toarray"):
        row_proc = row_proc.toarray()

    bg = pipe[:-1].transform(
        X_train_raw.sample(min(background_size, len(X_train_raw)),
                           random_state=42))
    if hasattr(bg, "toarray"):
        bg = bg.toarray()

    feat_names = pipe[:-1].get_feature_names_out()

    explainer = shap.TreeExplainer(
        pipe["xgb"],
        bg,
        model_output="probability",
        feature_names=feat_names
    )
    sv = explainer(row_proc)

    contrib = sv.values[0]
    top_idx = np.argsort(np.abs(contrib))[::-1][:top_n]
    table = pd.DataFrame({
        "Feature"   : feat_names[top_idx],
        "Impact"    : contrib[top_idx],
        "Value"     : row_proc[0, top_idx],
        "Direction" : ["Increases" if v > 0 else "Decreases"
                       for v in contrib[top_idx]]
    })

    return table, sv


# ── Plotting ─────────────────────────────────────────────────────────

def manual_waterfall(
    expl, 
    max_display=7, 
    ax=None, 
    neg_color="#88c999",
    pos_color="#f18c8c"
):
    """
    Pure-Matplotlib horizontal waterfall for one shap.Explanation record.
    """
    # 1) Axes setup
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max_display * 0.6))
    else:
        fig = ax.figure

    # 2) Extract base + contributions
    base     = float(np.array(expl.base_values).ravel()[0])
    contribs = np.array(expl.values)
    feats    = list(expl.feature_names)

    df = pd.DataFrame({"feature": feats, "contrib": contribs})
    df["absval"] = df.contrib.abs()
    df = df.sort_values("absval", ascending=False)

    # 3) Top N + “other” bucket
    top_df   = df.iloc[:max_display].copy()
    other_df = df.iloc[max_display:]
    if len(other_df):
        other_sum = other_df.contrib.sum()
        other_row = pd.DataFrame([{
            "feature": f"{len(other_df)} other features",
            "contrib":  other_sum,
            "absval":   abs(other_sum)
        }])
        top_df = pd.concat([top_df, other_row], ignore_index=True)

    # 4) Compute cumulative start positions
    starts = [base]
    for c in top_df.contrib[:-1]:
        starts.append(starts[-1] + c)
    starts = np.array(starts)

    # 5) Draw bars
    y = np.arange(len(top_df))
    colors = [pos_color if c>0 else neg_color for c in top_df.contrib]
    ax.barh(y, top_df.contrib, left=starts, color=colors, edgecolor="k", height=.6)
    ax.invert_yaxis()

    # 6) Arrow connectors
    for i, c in enumerate(top_df.contrib):
        start = starts[i]
        end   = start + c
        ax.annotate(
            "", 
            xy=(end,   i), 
            xytext=(start, i),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="k")
        )

    # 7) Base & final lines
    final = base + top_df.contrib.sum()
    ax.axvline(base,  color="gray", alpha = .5, linestyle="--", lw=2)
    ax.axvline(final, color="k", linestyle="--",  lw=2)

    # 8) Labels & title
    ax.set_yticks(y)
    ax.set_yticklabels(top_df.feature)
    ax.set_xlabel("Predicted Risk")
    # ax.set_title("Change in predicted risk by feature")
    ax.set_xlim(0,1)

    # 9) Legend
    legend_elems = [
        Line2D([0], [0], color=pos_color, lw=6, label="Positive Δ"),
        Line2D([0], [0], color=neg_color, lw=6, label="Negative Δ"),
        Line2D([0], [0], color="k", lw=2, linestyle="--", label="Base value"),
        Line2D([0], [0], color="k", lw=2, linestyle="-",  label="Final value")
    ]
    ax.legend(handles=legend_elems, loc="upper right")

    return ax


def plot_risk_distribution(cohort_probs, p, tau_lo, tau_hi, ax):

    cohort_numeric = pd.to_numeric(cohort_probs, errors='coerce')
    sorted_probs   = np.sort(cohort_numeric)
    pct            = np.linspace(0, 100, len(sorted_probs))

    low  = sorted_probs < tau_lo
    mod  = (sorted_probs >= tau_lo) & (sorted_probs < tau_hi)
    high = sorted_probs >= tau_hi

    ax.fill_betweenx(pct[low],  0, sorted_probs[low],  alpha=0.8, label="Low Risk",    color="#88c999")
    ax.fill_betweenx(pct[mod],  0, sorted_probs[mod],  alpha=0.8, label="Moderate Risk", color="#ffcb58")
    ax.fill_betweenx(pct[high], 0, sorted_probs[high], alpha=0.8, label="High Risk",   color="#f18c8c")

    patient_pct = (cohort_numeric < p).mean() * 100
    ax.plot(p, patient_pct, "*", ms=15, color="k", label="This Patient")

    ax.set_xlabel("Predicted Risk")
    ax.set_ylabel("Percentile")
    ax.axvline(tau_lo, linestyle="--", color="gray", alpha=0.5)
    ax.axvline(tau_hi, linestyle="--", color="gray", alpha=0.5)


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc="lower right")

    return patient_pct

#-- Drivers ─────────────────────────────────────────────────────────────

def analyze_patient(row: pd.Series | pd.DataFrame,
                  pipe, thresholds, top_n = 5, background_df=None,
                  show=True):
    """
    Compute probability, band, percentile; optionally show plots.
    Return dict with summary numbers.
    """
    cohort_probs = pipe.predict_proba(background_df)[:, 1] 
    p  = predict_proba_single(pipe, row)
    tl, th = thresholds["tau_low"], thresholds["tau_high"]
    band = risk_band(p, tl, th)

    summary = dict(probability=round(p, 4),
                   risk_band=band)

    if cohort_probs is not None:
        summary["percentile"] = round((p > cohort_probs).mean() * 100, 1)

    tbl, sv = analyze_top_predictors_single(row,
                                        pipe,
                                        background_df,
                                        top_n=top_n)
    
    return p, band, tbl, sv, cohort_probs



def create_patient_summary(
        patient_num = None,
        top_n = 5,
        figsize=(20, 10), 
        wspace=0.4):
    """
    Two-panel figure:
      • left = manual waterfall
      • right = risk distribution
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=figsize,
        constrained_layout=True
    )
    plt.subplots_adjust(wspace=wspace)

    
    patient_num, pipe, thr, row, cohort = get_data(PATIENT_ID=patient_num)
    
    p, band, _, sv, cohort_probs = analyze_patient(row = row, pipe = pipe, thresholds=thr, background_df = cohort, top_n = top_n)

 

    # manual waterfall on the left
    manual_waterfall(sv[0], max_display=top_n, ax=ax1)
    ax1.set_xlim(left=None)   # let Matplotlib autoscale

    # risk distribution on the right
    plot_risk_distribution(
        cohort_probs, p,
        thr["tau_low"],
        thr["tau_high"],
        ax=ax2
    )

    fig.suptitle(
        f"Patient {patient_num}, {band} risk of PTP",
        fontsize=18,
        fontweight="bold"
    )

    plt

 
    # return fig