# -*- coding: utf-8 -*-
"""
XAI Analysis for the direct-coupled PV-PEM switching controller (1 MW system)
=============================================================================
Explainable-AI comparison of the MOO+RF baseline vs the RL agent, using
shallow surrogate Decision Trees and policy visualisations.

Run this AFTER the main simulation has written the per-location CSVs:
  Results_1MW_daily/Results_<LOC>/Data/rl_results_<LOC>.csv
  Results_1MW_daily/Results_<LOC>/Data/rf_results_<LOC>.csv

Key design choices (aligned with the manuscript's XAI section):
  * The surrogate tree is trained ONLY on DECISION-TIME features — the same
    quantities the agent actually observes at the moment it chooses an action
    (irradiance, temperature, spot price, hour, H2 storage, battery SoC, PEM
    temperature, and the three SoH signals). Outcome variables that are a
    CONSEQUENCE of the action (H2 produced, MPP shortfall, grid flows, profit) are
    deliberately EXCLUDED so the tree cannot "cheat" by reading the effect of
    the decision it is meant to predict.
  * "State-awareness" is quantified as the collective importance of the
    features that the memoryless MOO / MOO+RF baselines structurally cannot
    see (everything except irradiance and temperature).
  * All string-count ranges auto-scale to the data (250-400 for the 1 MW
    system), so nothing is hard-coded to a particular sizing.

In Spyder: set LOCATIONS / OUT_DIR below and press F5.
"""

# =============================================================================
# %% USER CONFIGURATION -- edit before running
# =============================================================================
import os as _os
import sys as _sys
# __file__ exists when run as a .py script, but NOT in a Colab/Jupyter cell.
try:
    _SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = '/content' if 'google.colab' in _sys.modules else _os.getcwd()

# Folder holding the Results_<LOC> dirs. The daily run writes them under
# 'Results_1MW_daily/Hourly resolution'; falls back to 'Results_1MW_daily'.
_CAND_DIRS = [_os.path.join(_SCRIPT_DIR, 'Results_1MW_daily', 'Hourly resolution'),
              _os.path.join(_SCRIPT_DIR, 'Results_1MW_daily'),
              '/content/Results_1MW_daily/Hourly resolution',
              '/content/Results_1MW_daily']
OUT_DIR         = next((d for d in _CAND_DIRS
                        if _os.path.isdir(_os.path.join(d, 'Results_Lund_2021', 'Data'))),
                       _CAND_DIRS[0])
LOCATIONS       = ['Lund_2021', 'Prague_2021', 'Odeillo_2021']  # run XAI for each
TARGET_FIDELITY = 0.85
MAX_TREE_DEPTH  = 12
RULE_DEPTH      = 3    # shallow surrogate depth used only for the printed IF/THEN rules

# =============================================================================
# %% IMPORTS
# =============================================================================
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
_spyder_running = (
    'spyder_kernels' in sys.modules or
    'SPYDER_ARGS' in _os.environ or
    'spyder' in sys.modules
)
if not _spyder_running:
    try:
        matplotlib.use('Agg')
    except Exception:
        pass
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
print("XAI imports OK.")

# =============================================================================
# %% AESTHETICS & FEATURE DEFINITIONS
# =============================================================================
# Serif style + large fonts, matching the manuscript figures. On-plot titles
# are kept short (or omitted) since captions are written in the paper text.
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'axes.titlesize':   17,
    'axes.labelsize':   18,
    'xtick.labelsize':  15,
    'ytick.labelsize':  15,
    'legend.fontsize':  15,
    'figure.titlesize': 18,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'savefig.dpi':      300,
})

C_MOO  = '#4477AA'  # MOO      (greedy)
C_HYST = '#AA3377'  # MOO+Hyst (deadband)
C_PEN  = '#EE7733'  # MOO+Pen  (soft switching charge)  -- matches CTRL_COLORS in New_RL_1MW
C_RF   = '#EE6677'  # MOO+RF   (surrogate; retired, kept so old result folders still load)
C_RL   = '#228833'  # DRL agent
C_IRR  = '#F39C12'  # irradiance

# Must match COLDSTART_THRESHOLD_C in New_RL_1MW.txt -- used only to annotate the
# keep-warm histogram with the restart-event count, not to recompute the metric.
COLDSTART_THRESHOLD_C = 50.0

# Controller registry: (display label, CSV file tag, colour). Any whose CSV is missing is
# skipped gracefully; only DRL is required.
# MOO+Pen was ABSENT from this list, so the penalised optimizer -- one of the four
# controllers in the current comparison -- was silently excluded from every XAI figure even
# though run_baselines writes pen_results_<loc>.csv. MOO+RF is retired (INCLUDE_RF_MOO=False)
# but is left in the registry so archived result folders still resolve; it simply skips when
# the CSV is absent.
CONTROLLERS = [
    ('MOO',      'moo',  C_MOO),
    ('MOO+Hyst', 'hyst', C_HYST),
    ('MOO+Pen',  'pen',  C_PEN),
    ('MOO+RF',   'rf',   C_RF),
    ('DRL',      'rl',   C_RL),
]

# --- Decision-time features (what the agent observes BEFORE acting) ----------
# Every one of these is present in the CSV; hour_of_day is derived from `step`.
DECISION_FEATURES = [
    'irradiance', 'temperature', 'price_mwh', 'hour_of_day',
    'storage', 'batt_soc', 'T_pem', 'SoH_PEM', 'SoH_PV', 'SoH_BATT',
    # Anticipatory inputs. The agent's observation carries irradiance/temperature/price 1-4 h
    # ahead (LOOKAHEAD_STEPS), and at EVALUATION the forecast noise is switched off, so what
    # it sees is exactly the weather series shifted forward. Reconstructing it here costs
    # nothing and needs no re-run. Omitting it distorted every attribution: with no forecast
    # column available, a tree explaining an anticipatory action (pre-warming at dawn because
    # sun is coming) must charge the importance to CURRENT irradiance, which rises at dawn and
    # therefore proxies the forecast almost perfectly. That is why the keep-warm figure read
    # as 93 % irradiance-driven and only ~5 % state-driven.
    'irr_fcst_1h', 'irr_fcst_2h', 'irr_fcst_3h', 'irr_fcst_4h',
    # The configuration currently held. MOO+Hyst commits a new position only once the greedy
    # optimum has moved >= 2 steps FROM THIS VALUE, so without it no tree can reproduce the
    # deadband, and its fidelity ceiling (~88 %) measures a missing feature rather than an
    # erratic policy.
    'prev_n_parallel',
]

FEATURE_LABELS = {
    'irradiance':  'Irradiance (W/m$^2$)',
    'temperature': 'Ambient Temp (°C)',
    'price_mwh':   'Spot Price (€/MWh)',
    'hour_of_day': 'Hour of Day',
    'storage':     'H$_2$ Storage (kg)',
    'batt_soc':    'Battery SoC',
    'T_pem':       'PEM Temperature (K)',
    'SoH_PEM':     'PEM Health',
    'SoH_PV':      'PV Health',
    'SoH_BATT':    'Battery Health',
    'irr_fcst_1h': 'Irradiance +1h (fcst)',
    'irr_fcst_2h': 'Irradiance +2h (fcst)',
    'irr_fcst_3h': 'Irradiance +3h (fcst)',
    'irr_fcst_4h': 'Irradiance +4h (fcst)',
    'prev_n_parallel': 'Held configuration',
}

# Features that a memoryless MOO / MOO+RF controller CANNOT see (everything
# except the instantaneous environment) → these define "state-awareness".
ENV_FEATS      = ['irradiance', 'temperature']
STATE_FEATS    = ['storage', 'batt_soc', 'T_pem', 'SoH_PEM', 'SoH_PV', 'SoH_BATT']
MARKET_FEATS   = ['price_mwh']
TIME_FEATS     = ['hour_of_day']
FORECAST_FEATS = ['irr_fcst_1h', 'irr_fcst_2h', 'irr_fcst_3h', 'irr_fcst_4h']
MEMORY_FEATS   = ['prev_n_parallel']

# What the memoryless baselines structurally lack is TWO different things, and collapsing them
# into one number hides which one the learned policy actually exploits:
#   STATE_FEATS    -- internal plant state that persists between steps (memory)
#   FORECAST_FEATS -- knowledge of what is coming (anticipation)
# They are reported separately. The baselines observe neither, so their importance on both
# should come out ~0, and that is the control which makes the DRL numbers mean anything.

# =============================================================================
# %% SURROGATE DECISION TREE
# =============================================================================

def _prepare(df, dt_hours=1.0, daytime_only=True):
    """Add derived features and return a clean copy.

    dt_hours is the timestep length in hours (1.0 hourly, 0.25 at 15-min);
    hour_of_day is the true clock hour (0-23) regardless of resolution.
    daytime_only=True keeps the generating hours (irradiance >= 80), which is
    the right subset for the string-count decision. Set False for the keep-warm
    setpoint decision, which is exercised precisely when the stack is idle.
    """
    d = df.copy()
    # Derived columns are built on the FULL series and only THEN filtered. Shifting after a
    # daytime filter would step across the removed night hours and silently produce a
    # "forecast" of the next MORNING rather than of the next hour.
    for k in (1, 2, 3, 4):
        d[f'irr_fcst_{k}h'] = d['irradiance'].shift(-k)
    if 'n_parallel' in d.columns:
        d['prev_n_parallel'] = d['n_parallel'].shift(1)
    d = d.bfill().ffill()          # only the 4 tail rows and 1 head row are affected
    if 'step' in d.columns:
        d['hour_of_day'] = (d['step'].astype(float) * dt_hours).astype(int) % 24
    else:
        d['hour_of_day'] = (np.arange(len(d)) * dt_hours).astype(int) % 24
    if daytime_only:
        d = d[d['irradiance'] >= 80]
    return d


def extract_surrogate_tree(df_active, feats, max_depth=12, target_fidelity=0.85,
                           target='n_parallel'):
    """Fit CART surrogates of increasing depth; return the shallowest tree that
    reaches target fidelity (or the deepest if none does), plus the full sweep.
    `target` is the action column the tree predicts ('n_parallel' for the string
    configuration, 'T_keepwarm_C' for the thermal warm-standby setpoint)."""
    X = df_active[feats].values
    y = np.round(df_active[target].astype(float)).astype(int).values

    # Fidelity is scored on HELD-OUT steps. Previously the tree was fitted and scored on the
    # same rows, so at depth 12 with min_samples_leaf=10 a surrogate can memorise the
    # trajectory and report a fidelity that says nothing about whether the policy is really
    # rule-expressible -- which is the entire claim the fidelity-vs-depth curve is used to
    # support. Split is stratified where possible so rare actions survive into both halves.
    strat = y if np.min(np.bincount(y - y.min())) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True, stratify=strat)

    results = {}
    for d in range(2, max_depth + 1):
        clf = DecisionTreeClassifier(max_depth=d, random_state=42, min_samples_leaf=10)
        clf.fit(X_tr, y_tr)
        results[d] = {'fidelity':    accuracy_score(y_te, clf.predict(X_te)),   # held-out
                      'fidelity_in': accuracy_score(y_tr, clf.predict(X_tr)),   # in-sample
                      'model': clf, 'leaves': clf.get_n_leaves()}

    chosen = max_depth
    for d in sorted(results):
        if results[d]['fidelity'] >= target_fidelity:
            chosen = d
            break
    return results[chosen]['model'], results[chosen]['fidelity'], results


def state_awareness_fraction(clf, feats):
    """Collective importance of the INTERNAL PLANT STATE features.

    Counts STATE_FEATS only -- tank level, battery SoC, stack temperature and the three
    state-of-health variables. These are the quantities that persist between steps and that a
    memoryless controller genuinely cannot access.

    This used to be `sum(v for k, v in imp.items() if k not in ENV_FEATS)`, i.e. everything
    except irradiance and ambient temperature, which also swept in `price_mwh` and
    `hour_of_day`. Neither is internal state: a spot price is an instantaneous exogenous input
    and a clock is not memory, so a memoryless optimizer could read both. Charging them to
    "state-awareness" inflated the headline figure. Restricting the sum to STATE_FEATS makes
    the metric mean what its name claims, and keeps it defensible against the obvious question
    of why a clock counts as plant state.
    """
    imp = dict(zip(feats, clf.feature_importances_))
    return sum(v for k, v in imp.items() if k in STATE_FEATS)


def anticipation_fraction(clf, feats):
    """Collective importance of the FORECAST features.

    Kept separate from state-awareness on purpose. Memory (where the plant is now) and
    anticipation (where it is about to be) are different capabilities, and the thermal action
    is expected to rest on the second: pre-warming is worth its energy only if production is
    imminent, which current irradiance cannot tell you at 04:00. The memoryless baselines have
    neither, so a non-trivial value here for the DRL against ~0 for the optimizers is the
    cleanest available evidence that the policy is predictive rather than merely reactive.
    """
    imp = dict(zip(feats, clf.feature_importances_))
    return sum(v for k, v in imp.items() if k in FORECAST_FEATS)

def tree_to_rules(clf, feats):
    """Walk a fitted CART and return one merged (lo, hi) interval per feature per leaf.

    A raw root-to-leaf path can test the SAME feature twice at different thresholds
    (e.g. T_pem <= 330 then, deeper down, T_pem <= 310); reporting both is redundant and
    unreadable. Only the tightest bound reached along the path -- i.e. the interval that
    actually gates the leaf -- is kept. Leaves are returned sorted by sample count so the
    rules that govern the largest share of operation are listed first.
    """
    t = clf.tree_

    def recurse(node, bounds):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            f = feats[t.feature[node]]
            thr = t.threshold[node]
            lo, hi = bounds.get(f, (-np.inf, np.inf))
            left  = {**bounds, f: (lo, min(hi, thr))}
            right = {**bounds, f: (max(lo, thr), hi)}
            yield from recurse(t.children_left[node], left)
            yield from recurse(t.children_right[node], right)
        else:
            value = t.value[node][0]
            idx = int(np.argmax(value))
            yield dict(bounds=bounds, label=clf.classes_[idx],
                       n=int(t.n_node_samples[node]), purity=float(value[idx] / value.sum()))

    leaves = list(recurse(0, {}))
    return sorted(leaves, key=lambda l: -l['n'])


def _fmt_val(feat, v):
    """Render a split threshold in physically meaningful units, matching FEATURE_LABELS."""
    if feat == 'T_pem':
        return f'{v:.1f} K ({v - 273.15:.1f} °C)'
    if feat == 'hour_of_day':
        return f'{v:.1f} h'
    if feat == 'irradiance' or feat.startswith('irr_fcst'):
        return f'{v:.0f} W/m$^2$'
    if feat in ('SoH_PEM', 'SoH_PV', 'SoH_BATT', 'batt_soc') and abs(v) <= 1.5:
        return f'{v * 100:.0f}%'
    if feat == 'storage':
        return f'{v:.0f} kg'
    if feat == 'price_mwh':
        return f'{v:.0f} €/MWh'
    if feat == 'prev_n_parallel':
        return f'{v:.0f}'
    return f'{v:.2f}'


def format_rules(leaves, feats, action_name, action_fmt=str, max_rules=8):
    """Turn tree_to_rules() output into printable IF/THEN strings, most-visited leaf first."""
    total = sum(l['n'] for l in leaves)
    lines = []
    for l in leaves[:max_rules]:
        conds = []
        for f in feats:
            if f not in l['bounds']:
                continue
            lo, hi = l['bounds'][f]
            label = FEATURE_LABELS.get(f, f)
            if np.isfinite(lo) and np.isfinite(hi):
                conds.append(f'{_fmt_val(f, lo)} < {label} <= {_fmt_val(f, hi)}')
            elif np.isfinite(hi):
                conds.append(f'{label} <= {_fmt_val(f, hi)}')
            elif np.isfinite(lo):
                conds.append(f'{label} > {_fmt_val(f, lo)}')
        share = 100.0 * l['n'] / total if total else 0.0
        lines.append(f"  IF {' AND '.join(conds) if conds else '(always)'}\n"
                     f"     THEN {action_name} = {action_fmt(l['label'])}   "
                     f"[{share:4.1f}% of steps, {l['purity']*100:.0f}% leaf purity]")
    if len(leaves) > max_rules:
        lines.append(f"  ... ({len(leaves) - max_rules} smaller-share leaves omitted)")
    return lines


def write_rules_report(ctrls, kw_bundle, feats, loc, plots_dir, rule_depth=RULE_DEPTH):
    """Distil IF/THEN rules for every controller from a SHALLOW (depth<=rule_depth) surrogate,
    so the qualitative policy -- what values trigger what action, not just which features
    matter -- can be read directly instead of inferred from the importance bars. Reuses the
    per-depth models `extract_surrogate_tree` already fitted in its depth sweep (`res`), so
    nothing is retrained here; the shallow tree's own (lower) fidelity is reported next to
    each block so the simplification from the reported-depth tree is never hidden.
    """
    lines = [f"{'='*78}\nDISTILLED CONTROL RULES -- {loc}  (depth<={rule_depth} surrogate)\n{'='*78}"]
    for c in ctrls:
        d = min(rule_depth, max(c['res']))
        model, fid_d = c['res'][d]['model'], c['res'][d]['fidelity']
        rules = format_rules(tree_to_rules(model, feats), feats, 'string config',
                             action_fmt=lambda v: f'{int(v)} strings')
        lines.append(f"\n{c['label']}  -- string-configuration decision "
                     f"(depth-{d} fidelity {fid_d*100:.0f}%, vs. {c['fid']*100:.0f}% at full depth)")
        lines += rules
    if kw_bundle and kw_bundle.get('clf') is not None:
        kfeats = kw_bundle['feats']
        d = min(rule_depth, max(kw_bundle['res']))
        model, fid_d = kw_bundle['res'][d]['model'], kw_bundle['res'][d]['fidelity']
        rules = format_rules(tree_to_rules(model, kfeats), kfeats, 'keep-warm set-point',
                             action_fmt=lambda v: f'{int(v)} °C')
        lines.append(f"\nDRL  -- thermal keep-warm decision "
                     f"(depth-{d} fidelity {fid_d*100:.0f}%, vs. {kw_bundle['fid']*100:.0f}% at full depth)")
        lines += rules
    report = '\n'.join(lines)
    print(report)
    out = plots_dir / f'xai_9_rules_{loc}.txt'
    out.write_text(report, encoding='utf-8')
    print(f"\n  -> {out}")
    return report

# =============================================================================
# %% PLOTTING FUNCTIONS
# =============================================================================

def plot_feature_importance(ax, ctrls, feats):
    """Grouped horizontal bars: one bar per controller per feature.
    ctrls is a list of (label, clf, colour)."""
    imps = {lbl: clf.feature_importances_ for lbl, clf, _ in ctrls}
    ref  = ctrls[-1][0]                                  # sort by the last controller (DRL)
    idx  = [i for i in range(len(feats)) if any(imps[l][i] > 0.01 for l, _, _ in ctrls)]
    idx  = sorted(idx, key=lambda i: imps[ref][i])
    labels = [FEATURE_LABELS.get(feats[i], feats[i]) for i in idx]
    y = np.arange(len(idx)); n = len(ctrls); h = 0.8 / n
    for k, (lbl, _clf, color) in enumerate(ctrls):
        off = (k - (n - 1) / 2) * h
        ax.barh(y + off, [imps[lbl][i] for i in idx], h, label=lbl, color=color, edgecolor='white')
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel('Relative importance in the switching decision', fontweight='bold')
    ax.legend(loc='lower right', framealpha=1.0)


def plot_macro_categories(ax, ctrls, feats):
    cats = ['Environment\n(G, T)', 'System State\n(storage, SoH, T$_{PEM}$)',
            'Market\n(price)', 'Temporal\n(hour)']
    groups = [ENV_FEATS, STATE_FEATS, MARKET_FEATS, TIME_FEATS]
    def agg(clf):
        imp = dict(zip(feats, clf.feature_importances_))
        return [sum(imp.get(f, 0) for f in g) for g in groups]
    x = np.arange(len(cats)); n = len(ctrls); w = 0.8 / n
    for k, (lbl, clf, color) in enumerate(ctrls):
        off = (k - (n - 1) / 2) * w
        ax.bar(x + off, agg(clf), w, label=lbl, color=color, edgecolor='white')
    ax.set_ylabel('Aggregated importance', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.legend(loc='upper right', framealpha=1.0)


def plot_fidelity_vs_complexity(ax, ress):
    """ress is a list of (label, res_dict, colour)."""
    markers = {'MOO': 'o', 'MOO+Hyst': '^', 'MOO+Pen': 'v', 'MOO+RF': 's', 'DRL': 'D'}
    for lbl, res, color in ress:
        depths = sorted(res)
        ax.plot(depths, [res[d]['fidelity']*100 for d in depths], marker=markers.get(lbl, 'o'),
                lw=2.5, color=color, label=f'{lbl} surrogate')
    ax.axhline(TARGET_FIDELITY*100, color='#7F8C8D', ls='--', lw=1.5, label=f'{TARGET_FIDELITY*100:.0f}% target')
    ax.set_xlabel('Decision-tree depth (model complexity)', fontweight='bold')
    ax.set_ylabel('Fidelity to controller action (%)', fontweight='bold')
    ax.legend(loc='lower right', framealpha=1.0)


def _npar_bins(dfs, step=25):
    lo = int(min(df['n_parallel'].min() for df in dfs))
    hi = int(max(df['n_parallel'].max() for df in dfs))
    lo = (lo // step) * step - step // 2
    hi = (hi // step) * step + step + step // 2
    return np.arange(lo, hi + step, step), lo, hi


def plot_action_distribution(ax, ctrls_df):
    """ctrls_df is a list of (label, df, colour)."""
    bins, lo, hi = _npar_bins([df for _, df, _ in ctrls_df])
    ax.hist([df['n_parallel'] for _, df, _ in ctrls_df], bins=bins,
            color=[c for _, _, c in ctrls_df], label=[l for l, _, _ in ctrls_df], edgecolor='white')
    ax.set_xlabel('Selected parallel strings (MOSFET configuration)', fontweight='bold')
    ax.set_ylabel('Frequency (time steps)', fontweight='bold')
    ax.legend(framealpha=1.0)


def plot_switching_heatmap(fig, axes, ctrls_df):
    """One transition-matrix panel per controller (axes flattened, 2x2 for four)."""
    bins, lo, hi = _npar_bins([df for _, df, _ in ctrls_df])
    cmaps = {'MOO': 'Blues', 'MOO+Hyst': 'Purples', 'MOO+Pen': 'Oranges', 'MOO+RF': 'Reds', 'DRL': 'Greens'}
    axes = np.array(axes).ravel()
    for ax, (lbl, df, color) in zip(axes, ctrls_df):
        t = df.copy(); t['prev_n'] = t['n_parallel'].shift(1); t = t.dropna()
        h = ax.hist2d(t['prev_n'], t['n_parallel'], bins=[bins, bins], cmap=cmaps.get(lbl, 'viridis'), cmin=1)
        cb = fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Transitions', fontweight='bold')
        ax.plot([lo, hi], [lo, hi], color='#2C3E50', ls='--', lw=1.5)
        ax.set_title(lbl, fontweight='bold', color=color, pad=8)
        ax.set_xlabel('Config at $t-1$', fontweight='bold')
        ax.set_ylabel('Config at $t$', fontweight='bold')
    for ax in axes[len(ctrls_df):]:                     # hide any unused panels
        ax.set_visible(False)
    fig.subplots_adjust(wspace=0.35, hspace=0.4)


def plot_policy_boundary(fig, axes, ctrls_df):
    """Policy boundary in the (irradiance, PEM-temperature) plane, coloured by the
    chosen string count, one panel per controller. Reveals the DRL agent's thermal
    logic (colour shifts as the stack warms) versus the memoryless baselines."""
    dfs  = [df for _, df, _ in ctrls_df]
    vmin = min(df['n_parallel'].min() for df in dfs)
    vmax = max(df['n_parallel'].max() for df in dfs)
    axes  = np.array(axes)
    ncols = axes.shape[1] if axes.ndim > 1 else len(axes)
    axes  = axes.ravel()
    n     = len(ctrls_df)
    sc = None
    for i, (ax, (lbl, df, color)) in enumerate(zip(axes, ctrls_df)):
        sc = ax.scatter(df['irradiance'], df['T_pem'] - 273.15, c=df['n_parallel'],
                        cmap='viridis', alpha=0.8, s=16, vmin=vmin, vmax=vmax, edgecolor='none')
        ax.set_title(lbl, fontweight='bold', color=color, pad=8)
        # Axis titles only on the outer edge of the grid. Labelling every panel put the
        # first row's x-label directly under the second row's title, and they collided.
        # Tick labels are kept on every panel because the panels are not on shared axes.
        if i + ncols >= n:
            ax.set_xlabel('Irradiance (W/m$^2$)', fontweight='bold')
        if i % ncols == 0:
            ax.set_ylabel('PEM Temperature (°C)', fontweight='bold')
    for ax in axes[n:]:
        ax.set_visible(False)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.tolist(), fraction=0.03, pad=0.04)
        cb.set_label('Action: parallel strings', rotation=270, labelpad=20, fontweight='bold')


def plot_timeseries_snapshot(ax, ctrls_full, dt_hours=1.0, window_hours=72):
    """Overlay every controller's string-count trajectory over a fixed window.
    ctrls_full is a list of (label, df_full, colour); the first entry sets the window."""
    ref_df = ctrls_full[0][1]
    high = ref_df[ref_df['irradiance'] > 500]
    if high.empty:
        ax.set_title('No high-irradiance window found', fontweight='bold'); return
    window = int(round(window_hours / dt_hours))
    s = high.index[0]; e = s + window
    ax2 = ax.twinx()
    sub_ref = ref_df.loc[s:e]; t_ref = (sub_ref.index - s) * dt_hours
    ax2.fill_between(t_ref, sub_ref['irradiance'], color=C_IRR, alpha=0.15)
    ax2.plot(t_ref, sub_ref['irradiance'], color=C_IRR, lw=1.5, alpha=0.5, label='Irradiance')
    styles = {'MOO': '-', 'MOO+Hyst': ':', 'MOO+Pen': (0,(3,1,1,1)), 'MOO+RF': '-.', 'DRL': '--'}
    for lbl, df, color in ctrls_full:
        sub = df.loc[s:e]; t = (sub.index - s) * dt_hours
        lw = 2.5 if lbl == 'DRL' else 1.8
        ax.step(t, sub['n_parallel'], where='post', color=color, lw=lw,
                ls=styles.get(lbl, '-'), label=lbl, alpha=0.9)
    ax.set_xlabel('Time (hours)', fontweight='bold')
    ax.set_ylabel('Action: parallel strings', fontweight='bold')
    ax2.set_ylabel('Irradiance (W/m$^2$)', fontweight='bold', color='#D68910')
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc='upper right', framealpha=1.0, ncol=2)
    ax2.grid(False)

# =============================================================================
# %% MAIN XAI EXECUTION (per location)
# =============================================================================

def run_keepwarm_xai(df_rl_full, feats, plots_dir, loc, dt_hours, target_fidelity=0.85):
    """Explain the NEW thermal warm-standby action. The keep-warm set-point is
    chosen while the stack is idle, so unlike the string-count decision this
    surrogate is fit on the FULL day (including night), not the daytime subset.
    Returns (fidelity, state_awareness) or None if the column is absent/constant."""
    if 'T_keepwarm_C' not in df_rl_full.columns:
        print("  keep-warm: T_keepwarm_C column not found (thermal action off?) -- skipped.")
        return None
    dk = _prepare(df_rl_full, dt_hours, daytime_only=False)
    kw_feats = [f for f in feats if f in dk.columns]
    n_levels = int(dk['T_keepwarm_C'].round().nunique())
    if n_levels < 2:
        lvl = float(dk['T_keepwarm_C'].iloc[0]) if len(dk) else float('nan')
        print(f"  keep-warm: agent held a single set-point ({lvl:.0f} C) -- no tree to fit.")
        return {'fid': 1.0, 'sa': 0.0, 'levels': n_levels}

    clf, fid, res = extract_surrogate_tree(dk, kw_feats, MAX_TREE_DEPTH,
                                           target_fidelity, target='T_keepwarm_C')
    sa  = state_awareness_fraction(clf, kw_feats)
    ant = anticipation_fraction(clf, kw_feats)
    print(f"  KW  surrogate fidelity: {fid*100:5.1f}%   state-awareness: {sa*100:4.1f}%   "
          f"anticipation (forecast): {ant*100:4.1f}%   ({n_levels} set-points used)")

    # Restart-event count per set-point, for the panel (a) annotation. This is the SAME
    # definition as count_cold_starts() in New_RL_1MW.txt (a step where production resumes
    # after idling below COLDSTART_THRESHOLD_C), attributed to whichever set-point the agent
    # was holding on the idle step immediately before the restart. It is reported here
    # because the bar heights below count IDLE TIME STEPS held at each set-point (thousands,
    # since one idle spell lasts many steps), which is a different and much larger quantity
    # than the number of discrete cold-start events reported elsewhere in the paper -- without
    # this annotation the two numbers read as contradictory rather than as different metrics.
    if 'T_pem' in df_rl_full.columns and 'H2_prod' in df_rl_full.columns:
        producing = (df_rl_full['H2_prod'].values > 1e-9)
        was_idle  = np.r_[True, ~producing[:-1]]
        T_prev_C  = np.r_[np.nan, df_rl_full['T_pem'].values[:-1]] - 273.15
        is_cold_restart = producing & was_idle & np.r_[False, T_prev_C[1:] < COLDSTART_THRESHOLD_C]
        restart_kw = df_rl_full['T_keepwarm_C'].round().shift(1).bfill().values if 'T_keepwarm_C' in df_rl_full.columns else None
        n_restarts_total = int(is_cold_restart.sum())
    else:
        restart_kw, n_restarts_total = None, None

    # Figure: (a) how often each set-point is chosen, (b) what drives the choice.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    levels = sorted(dk['T_keepwarm_C'].round().unique())
    counts = [int((dk['T_keepwarm_C'].round() == L).sum()) for L in levels]
    bars = axes[0].bar([str(int(L)) for L in levels], counts, color=C_RL, edgecolor='white')
    if restart_kw is not None:
        for bar, L in zip(bars, levels):
            n_r = int((restart_kw[is_cold_restart] == L).sum())
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{n_r} restarts/yr', ha='center', va='bottom', fontsize=11)
        axes[0].set_title(f'{n_restarts_total} cold-start restarts/yr total '
                           f'(<{COLDSTART_THRESHOLD_C:.0f} °C) -- distinct from the dwell time below',
                           fontsize=12)
    axes[0].set_xlabel('Warm-standby set-point (°C)', fontweight='bold')
    axes[0].set_ylabel('Idle dwell time (time steps held at set-point)', fontweight='bold')
    imp = dict(zip(kw_feats, clf.feature_importances_))
    idx = sorted([f for f in kw_feats if imp[f] > 0.01], key=lambda f: imp[f])
    axes[1].barh([FEATURE_LABELS.get(f, f) for f in idx], [imp[f] for f in idx],
                 color=C_RL, edgecolor='white')
    axes[1].set_xlabel('Relative importance in the keep-warm decision', fontweight='bold')
    out = plots_dir / f'xai_8_keepwarm_policy_{loc}.png'
    fig.tight_layout(); fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"  -> {out}")
    return {'fid': fid, 'sa': sa, 'levels': n_levels, 'clf': clf, 'res': res, 'feats': kw_feats}


def run_xai(loc, out_dir, target_fidelity=0.85):
    data_dir  = Path(out_dir) / f'Results_{loc}' / 'Data'
    plots_dir = Path(out_dir) / f'Results_{loc}' / 'Plots'

    print(f"\n{'='*60}\nEXPLAINABLE AI (XAI) -- {loc}\n{'='*60}")
    # Load every controller whose CSV exists (MOO / MOO+Hyst / MOO+RF / DRL).
    loaded = []                                    # (label, colour, df_full)
    for lbl, tag, color in CONTROLLERS:
        p = data_dir / f'{tag}_results_{loc}.csv'
        if p.exists():
            loaded.append((lbl, color, pd.read_csv(p)))
        else:
            print(f"  {lbl:9} {p.name} not found -- skipped.")
    if not any(lbl == 'DRL' for lbl, _, _ in loaded):
        print(f"  ERROR: DRL CSV required, not found in {data_dir}."); return
    print(f"  Controllers loaded: {', '.join(lbl for lbl, _, _ in loaded)}")

    n_rows   = len(loaded[0][2])
    dt_hours = round(8760.0 / n_rows, 4) if n_rows else 1.0
    print(f"  Detected timestep: {dt_hours} h  ({n_rows} steps/year)")

    feats = None
    ctrls = []                                     # per-controller distillation bundle
    for lbl, color, dff in loaded:
        dfp = _prepare(dff, dt_hours)
        if feats is None:
            feats = [f for f in DECISION_FEATURES if f in dfp.columns]
            print(f"  Decision features ({len(feats)}): {feats}")
        clf, fid, res = extract_surrogate_tree(dfp, feats, MAX_TREE_DEPTH, target_fidelity)
        sa  = state_awareness_fraction(clf, feats)
        ant = anticipation_fraction(clf, feats)
        ctrls.append(dict(label=lbl, color=color, df=dfp, df_full=dff, clf=clf, res=res, fid=fid, sa=sa, ant=ant))
        print(f"  {lbl:9} surrogate fidelity {fid*100:5.1f}%   state-awareness {sa*100:4.1f}%   "
              f"anticipation (forecast) {ant*100:4.1f}%")

    # Helper views into the bundle for the (label, X, colour) plot signatures.
    ci = [(c['label'], c['clf'],    c['color']) for c in ctrls]
    cd = [(c['label'], c['df'],     c['color']) for c in ctrls]
    cf = [(c['label'], c['df_full'],c['color']) for c in ctrls]
    cr = [(c['label'], c['res'],    c['color']) for c in ctrls]
    n  = len(ctrls)

    plots_dir.mkdir(parents=True, exist_ok=True)
    def finalize(fig, name):
        out = plots_dir / f'{name}_{loc}.png'
        fig.savefig(out, bbox_inches='tight'); plt.close(fig)
        print(f"  -> {out}")

    fig, ax = plt.subplots(figsize=(10, 7));    plot_feature_importance(ax, ci, feats);  finalize(fig, 'xai_1_feature_importance')
    fig, ax = plt.subplots(figsize=(10, 5.5));  plot_macro_categories(ax, ci, feats);    finalize(fig, 'xai_2_macro_categories')
    fig, ax = plt.subplots(figsize=(9, 5.5));   plot_fidelity_vs_complexity(ax, cr);     finalize(fig, 'xai_3_fidelity_vs_complexity')
    fig, ax = plt.subplots(figsize=(10, 5.5));  plot_action_distribution(ax, cd);        finalize(fig, 'xai_4_action_distribution')
    fig, axes = plt.subplots(2, 2, figsize=(14, 11)); plot_switching_heatmap(fig, axes, cd); finalize(fig, 'xai_5_switching_heatmap')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10)); plot_policy_boundary(fig, axes, cd);   finalize(fig, 'xai_6_policy_boundary')
    fig, ax = plt.subplots(figsize=(13, 6));    plot_timeseries_snapshot(ax, cf, dt_hours); finalize(fig, 'xai_7_behavior_snapshot')

    # --- Explain the thermal warm-standby (keep-warm) action (DRL only) --------
    rl_full = next(c['df_full'] for c in ctrls if c['label'] == 'DRL')
    kw = run_keepwarm_xai(rl_full, feats, plots_dir, loc, dt_hours, target_fidelity)

    # --- Plain-language IF/THEN rules for every controller, plus the DRL keep-warm rule ---
    write_rules_report(ctrls, kw, feats, loc, plots_dir)

    n_plots = 7 + (1 if kw and kw.get('levels', 0) >= 2 else 0)
    print(f"XAI complete -- {n_plots} plots saved to {plots_dir}")
    return {'loc': loc, 'ctrls': {c['label']: (c['fid'], c['sa'], c['ant']) for c in ctrls},
            'kw_fid': (kw or {}).get('fid'), 'kw_sa': (kw or {}).get('sa')}


if __name__ == '__main__':
    print("XAI reading from:", OUT_DIR)
    summary = []
    for _loc in LOCATIONS:
        r = run_xai(_loc, OUT_DIR, TARGET_FIDELITY)
        if r: summary.append(r)
    if summary:
        order = [lbl for lbl, _, _ in CONTROLLERS]
        print(f"\n{'='*88}\nCROSS-LOCATION XAI SUMMARY  (fidelity % / state-awareness % / anticipation %)\n{'='*88}")
        head = f"  {'Location':14}" + "".join(f"{lbl:>24}" for lbl in order) + f"{'KW(DRL)':>14}"
        print(head)
        for s in summary:
            row = f"  {s['loc']:14}"
            for lbl in order:
                if lbl in s['ctrls']:
                    fid, sa, ant = s['ctrls'][lbl]
                    row += f"{fid*100:7.1f}/{sa*100:<6.1f}/{ant*100:<7.1f}"
                else:
                    row += f"{'--':>24}"
            row += (f"{s['kw_fid']*100:13.1f}%" if s.get('kw_fid') is not None else f"{'--':>14}")
            print(row)
        print("\n  Reading: memoryless MOO / MOO+Hyst / MOO+RF should show LOW state-awareness")
        print("  (decisions ~ irradiance+temperature only); the DRL's is high (uses storage,")
        print("  SoH, price, hour, PEM temperature) -- the quantitative basis of the XAI claim.")
