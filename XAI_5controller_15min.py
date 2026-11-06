# -*- coding: utf-8 -*-
"""FINAL 15-MIN XAI for the 1 MW direct-coupled PV-PEM controller study"""

# =============================================================================
# %% USER CONFIGURATION
# =============================================================================
import os as _os
import sys as _sys

try:
    _SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = '/content' if 'google.colab' in _sys.modules else _os.getcwd()

XAI_DEMAND_MODE = 'hourly'       # 'hourly', 'daily', or 'auto'
RESULTS_ROOT_OVERRIDE = None

LOCATIONS       = ['Lund_2021', 'Prague_2021', 'Odeillo_2021']

_DEMAND_DIR_ALIASES = {
    'hourly': (
        'Hourly_demand',       # CURRENT main-code name
        'Hourly demand',
        'Hourly_resolution',
        'Hourly resolution',   # historical name
    ),
    'daily': (
        'Daily_demand',        # CURRENT main-code name
        'Daily demand',
        'Daily_resolution',
        'Daily resolution',
    ),
}

_REQUIRED_RESULT_TAGS = ('moo', 'hyst', 'prewarm', 'hystprewarm', 'rl')


def _normalise_demand_mode(mode):
    mode = str(mode).strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'h': 'hourly', 'hour': 'hourly', 'hours': 'hourly', 'hourly': 'hourly',
        'd': 'daily', 'day': 'daily', 'days': 'daily', 'daily': 'daily',
        'a': 'auto', 'auto': 'auto',
    }
    if mode not in aliases:
        raise ValueError("XAI_DEMAND_MODE must be 'hourly', 'daily', or 'auto'.")
    return aliases[mode]


def _result_tree_score(root):
    """Score a candidate root by actual site/Data folders and expected CSV files."""
    root = _os.path.abspath(_os.path.expanduser(str(root)))
    score = 0
    for loc in LOCATIONS:
        data_dir = _os.path.join(root, f'Results_{loc}', 'Data')
        if _os.path.isdir(data_dir):
            score += 10
            for tag in _REQUIRED_RESULT_TAGS:
                if _os.path.isfile(_os.path.join(data_dir, f'{tag}_results_{loc}.csv')):
                    score += 1
    return score


def _candidate_result_roots(mode):
    """Return current + legacy local/Colab/Drive result roots in priority order."""
    mode = _normalise_demand_mode(mode)
    modes = ('hourly', 'daily') if mode == 'auto' else (mode,)

    base_roots = [
        _os.path.join(_SCRIPT_DIR, 'Results_1MW_15min'),
        '/content/Results_1MW_15min',
        '/content/drive/MyDrive/PVPEM_Results/Results_1MW_15min',
        _os.path.join(_SCRIPT_DIR, 'Results_1MW_daily'),
        _os.path.join(_SCRIPT_DIR, 'Results_1MW_hourly'),
        '/content/Results_1MW_daily',
        '/content/Results_1MW_hourly',
        '/content/drive/MyDrive/PVPEM_Results/Results_1MW_daily',
        '/content/drive/MyDrive/PVPEM_Results/Results_1MW_hourly',
    ]

    candidates = []
    if RESULTS_ROOT_OVERRIDE:
        candidates.append(RESULTS_ROOT_OVERRIDE)

    for base in base_roots:
        for m in modes:
            for folder in _DEMAND_DIR_ALIASES[m]:
                candidates.append(_os.path.join(base, folder))
        candidates.append(base)

    unique = []
    seen = set()
    for d in candidates:
        nd = _os.path.abspath(_os.path.expanduser(str(d)))
        if nd not in seen:
            unique.append(nd)
            seen.add(nd)
    return unique


def _resolve_results_root(mode):
    candidates = _candidate_result_roots(mode)
    scored = [(_result_tree_score(d), -i, d) for i, d in enumerate(candidates)]
    valid = [x for x in scored if x[0] > 0]
    if not valid:
        checked = '\n'.join(f'    - {d}' for d in candidates)
        raise FileNotFoundError(
            'XAI could not locate a result tree containing Results_<site>/Data.\n'
            f'XAI_DEMAND_MODE={mode!r}. Checked:\n{checked}\n\n'
            'Set RESULTS_ROOT_OVERRIDE to the exact demand-specific result folder if your '
            'results are stored elsewhere.'
        )
    score, _, root = max(valid)
    return root, score, candidates


_XAI_MODE = _normalise_demand_mode(XAI_DEMAND_MODE)
OUT_DIR, _OUT_SCORE, _CAND_DIRS = _resolve_results_root(_XAI_MODE)

TARGET_FIDELITY = 0.85
MAX_TREE_DEPTH  = 12
RULE_DEPTH      = 3
REQUIRE_ALL_FIVE = True

EXPECTED_DT_HOURS_XAI = 0.25
FORECAST_HORIZONS_H_XAI = (1.0, 2.0, 3.0, 4.0)
FORECAST_OFFSETS_STEPS_XAI = tuple(
    int(round(h / EXPECTED_DT_HOURS_XAI)) for h in FORECAST_HORIZONS_H_XAI
)
LOOKAHEAD_STEPS_XAI = len(FORECAST_OFFSETS_STEPS_XAI)
PEM_CAPACITY_W_XAI  = 1_000_000.0
COLDSTART_THRESHOLD_C = 50.0

PREWARM_RULE_MODE_XAI = 'anticipatory'   # 'anticipatory' or 'distilled'

# =============================================================================
# %% IMPORTS
# =============================================================================
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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

warnings.filterwarnings('ignore')
print('XAI imports OK.')

# =============================================================================
# %% AESTHETICS / CONTROLLER REGISTRY
# =============================================================================
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'axes.titlesize':    16,
    'axes.labelsize':    16,
    'xtick.labelsize':   13,
    'ytick.labelsize':   13,
    'legend.fontsize':   12,
    'figure.titlesize':  18,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'savefig.dpi':       300,
})

C_MOO  = '#4477AA'
C_HYST = '#AA3377'
C_PW   = '#CCBB44'
C_HPW  = '#66CCEE'
C_RL   = '#228833'
C_IRR  = '#F39C12'
RL_LABEL = 'RL Agent'

CONTROLLERS = [
    dict(label='MOO',                 tag='moo',         color=C_MOO),
    dict(label='MOO+Hyst',            tag='hyst',        color=C_HYST),
    dict(label='MOO+PreWarm',         tag='prewarm',     color=C_PW),
    dict(label='MOO+Hyst+PreWarm',    tag='hystprewarm', color=C_HPW),
    dict(label=RL_LABEL,              tag='rl',          color=C_RL),
]
CTRL_LABELS = [c['label'] for c in CONTROLLERS]
CTRL_COLOR  = {c['label']: c['color'] for c in CONTROLLERS}

LINESTYLES = {
    'MOO': '-',
    'MOO+Hyst': ':',
    'MOO+PreWarm': (0, (5, 2)),
    'MOO+Hyst+PreWarm': (0, (5, 1, 1, 1)),
    RL_LABEL: '--',
}
MARKERS = {
    'MOO': 'o',
    'MOO+Hyst': '^',
    'MOO+PreWarm': 's',
    'MOO+Hyst+PreWarm': 'P',
    RL_LABEL: 'D',
}
CMAPS = {
    'MOO': 'Blues',
    'MOO+Hyst': 'Purples',
    'MOO+PreWarm': 'YlOrBr',
    'MOO+Hyst+PreWarm': 'GnBu',
    RL_LABEL: 'Greens',
}

# =============================================================================
# %% FEATURE DEFINITIONS
# =============================================================================
CURRENT_ENV_FEATS = ['irradiance', 'temperature']
PLANT_STATE_FEATS = ['storage', 'batt_soc', 'T_pem', 'SoH_PEM', 'SoH_PV', 'SoH_BATT']
ELECTRICAL_MEMORY_FEATS = ['V_prev', 'I_prev', 'prev_n_parallel', 'prev_load_frac']
MARKET_FEATS = ['price_mwh']
TIME_FEATS   = ['hour_of_day']


def _forecast_features(prefix):
    return [f'{prefix}_fcst_s{k}' for k in range(1, LOOKAHEAD_STEPS_XAI + 1)]


FORECAST_IRR_FEATS   = _forecast_features('irr')
FORECAST_TEMP_FEATS  = _forecast_features('temp')
FORECAST_PRICE_FEATS = _forecast_features('price')
FORECAST_FEATS       = FORECAST_IRR_FEATS + FORECAST_TEMP_FEATS + FORECAST_PRICE_FEATS

DRL_STRING_FEATURES = (
    CURRENT_ENV_FEATS
    + ['V_prev', 'I_prev', 'prev_n_parallel', 'storage', 'hour_of_day',
       'SoH_PV', 'SoH_PEM', 'batt_soc', 'T_pem', 'price_mwh',
       'prev_load_frac', 'SoH_BATT']
    + FORECAST_FEATS
)

MOO_STRING_FEATURES = ['irradiance', 'temperature', 'T_pem', 'SoH_PEM', 'SoH_PV']

STRING_FEATURES_BY_CONTROLLER = {
    'MOO': MOO_STRING_FEATURES,
    'MOO+Hyst': MOO_STRING_FEATURES + ['prev_n_parallel'],
    'MOO+PreWarm': MOO_STRING_FEATURES,
    'MOO+Hyst+PreWarm': MOO_STRING_FEATURES + ['prev_n_parallel'],
    RL_LABEL: DRL_STRING_FEATURES,
}

FEATURE_GROUPS = [
    ('Current environment\n(G, T)', CURRENT_ENV_FEATS),
    ('Plant state\n(storage, SoH, T$_{PEM}$)', PLANT_STATE_FEATS),
    ('Electrical memory\n(V/I/load/config at t-1)', ELECTRICAL_MEMORY_FEATS),
    ('Market\n(price)', MARKET_FEATS),
    ('Temporal\n(hour)', TIME_FEATS),
    ('Forecast\n(G, T, price)', FORECAST_FEATS),
]


def _feature_label(feat, dt_hours=1.0):
    base = {
        'irradiance': 'Irradiance (W/m$^2$)',
        'temperature': 'Ambient Temp (°C)',
        'price_mwh': 'Spot Price (€/MWh)',
        'hour_of_day': 'Hour of Day',
        'storage': 'H$_2$ Storage (kg)',
        'batt_soc': 'Battery SoC',
        'T_pem': 'PEM Temperature (K)',
        'SoH_PEM': 'PEM Health',
        'SoH_PV': 'PV Health',
        'SoH_BATT': 'Battery Health',
        'V_prev': 'Previous stack voltage (V)',
        'I_prev': 'Previous stack current (A)',
        'prev_n_parallel': 'Held configuration',
        'prev_load_frac': 'Previous PEM load fraction',
    }
    if feat in base:
        return base[feat]
    for prefix, title, unit in (
        ('irr_fcst_s', 'Irradiance', 'W/m$^2$'),
        ('temp_fcst_s', 'Ambient Temp', '°C'),
        ('price_fcst_s', 'Spot Price', '€/MWh'),
    ):
        if feat.startswith(prefix):
            k = int(feat.split('s')[-1])
            horizon = FORECAST_HORIZONS_H_XAI[k - 1]
            htxt = f'{horizon:g} h'
            return f'{title} +{htxt} ({unit})'
    return feat


# =============================================================================
# %% DECISION-TIME RECONSTRUCTION
# =============================================================================
def _prepare(df, dt_hours=1.0, daytime_only=True):
    """Reconstruct decision-time features from a result trajectory."""
    d = df.copy().reset_index(drop=True)

    required_now = ['irradiance', 'temperature', 'price_mwh', 'n_parallel']
    missing_now = [c for c in required_now if c not in d.columns]
    if missing_now:
        raise KeyError(f'Missing required trajectory columns: {missing_now}')

    if 'step' in d.columns:
        d['hour_of_day'] = (d['step'].astype(float) * dt_hours) % 24.0
    else:
        d['hour_of_day'] = (np.arange(len(d), dtype=float) * dt_hours) % 24.0

    d['prev_n_parallel'] = d['n_parallel'].shift(1)
    if 'V_op' in d.columns:
        d['V_prev'] = d['V_op'].shift(1)
    if 'I_op' in d.columns:
        d['I_prev'] = d['I_op'].shift(1)
    if 'P_pem_actual' in d.columns:
        d['prev_load_frac'] = (d['P_pem_actual'].shift(1) / PEM_CAPACITY_W_XAI).clip(0.0, 1.0)

    for col in PLANT_STATE_FEATS:
        if col in d.columns:
            d[col] = d[col].shift(1)

    source_by_prefix = {
        'irr': 'irradiance',
        'temp': 'temperature',
        'price': 'price_mwh',
    }
    for prefix, src in source_by_prefix.items():
        for k, offset in enumerate(FORECAST_OFFSETS_STEPS_XAI, start=1):
            out = f'{prefix}_fcst_s{k}'
            d[out] = d[src].shift(-offset)
            if len(d):
                d[out] = d[out].fillna(float(d[src].iloc[-1]))

    prev_cols = ['prev_n_parallel', 'V_prev', 'I_prev', 'prev_load_frac'] + PLANT_STATE_FEATS
    for col in prev_cols:
        if col in d.columns:
            d[col] = d[col].bfill().ffill()

    if daytime_only:
        d = d[d['irradiance'] >= 80.0].copy()

    return d.reset_index(drop=True)


def _available_features(df, requested, label, purpose):
    feats = [f for f in requested if f in df.columns]
    missing = [f for f in requested if f not in df.columns]
    if missing:
        print(f'  WARNING {label} {purpose}: missing reconstructed features {missing}; omitted.')
    if not feats:
        raise ValueError(f'No usable {purpose} features for {label}.')
    return feats


# =============================================================================
# %% SURROGATE DECISION TREE
# =============================================================================
def extract_surrogate_tree(df_active, feats, max_depth=12, target_fidelity=0.85,
                           target='n_parallel'):
    """Fit CART surrogates of increasing depth on a held-out 70/30 split."""
    if target not in df_active.columns:
        raise KeyError(f"Target '{target}' is not in dataframe")

    use = df_active[feats + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 30:
        raise ValueError(f'Only {len(use)} valid rows available for target {target}.')

    X = use[feats].to_numpy(float)
    y = np.round(use[target].astype(float)).astype(int).to_numpy()

    unique_y, counts_y = np.unique(y, return_counts=True)
    strat = y if (len(unique_y) > 1 and counts_y.min() >= 2) else None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True, stratify=strat
    )

    min_leaf = max(5, min(20, len(X_tr) // 200))
    results = {}
    for depth in range(2, max_depth + 1):
        clf = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42,
            min_samples_leaf=min_leaf,
            class_weight=None,
        )
        clf.fit(X_tr, y_tr)
        results[depth] = {
            'fidelity': accuracy_score(y_te, clf.predict(X_te)),
            'fidelity_in': accuracy_score(y_tr, clf.predict(X_tr)),
            'model': clf,
            'leaves': clf.get_n_leaves(),
            'n_train': len(y_tr),
            'n_test': len(y_te),
        }

    passing = [depth for depth in sorted(results)
               if results[depth]['fidelity'] >= target_fidelity]
    if passing:
        chosen = passing[0]   # shallowest surrogate that reaches the target
    else:
        chosen = max(sorted(results), key=lambda depth: (results[depth]['fidelity'], -depth))
    return results[chosen]['model'], results[chosen]['fidelity'], results


def _importance_map(bundle):
    return dict(zip(bundle['feats'], bundle['clf'].feature_importances_))


def _fraction(bundle, group):
    imp = _importance_map(bundle)
    return float(sum(imp.get(f, 0.0) for f in group))


def state_awareness_fraction(bundle):
    return _fraction(bundle, PLANT_STATE_FEATS)


def memory_fraction(bundle):
    return _fraction(bundle, ELECTRICAL_MEMORY_FEATS)


def anticipation_fraction(bundle):
    return _fraction(bundle, FORECAST_FEATS)


def tree_to_rules(clf, feats):
    """Return merged feature intervals for every tree leaf, largest leaves first."""
    t = clf.tree_

    def recurse(node, bounds):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            feat = feats[t.feature[node]]
            thr = t.threshold[node]
            lo, hi = bounds.get(feat, (-np.inf, np.inf))
            left  = {**bounds, feat: (lo, min(hi, thr))}
            right = {**bounds, feat: (max(lo, thr), hi)}
            yield from recurse(t.children_left[node], left)
            yield from recurse(t.children_right[node], right)
        else:
            value = t.value[node][0]
            idx = int(np.argmax(value))
            yield dict(
                bounds=bounds,
                label=clf.classes_[idx],
                n=int(t.n_node_samples[node]),
                purity=float(value[idx] / max(value.sum(), 1e-12)),
            )

    return sorted(list(recurse(0, {})), key=lambda row: -row['n'])


def _fmt_val(feat, value, dt_hours=1.0):
    if feat == 'T_pem':
        return f'{value:.1f} K ({value - 273.15:.1f} °C)'
    if feat == 'hour_of_day':
        return f'{value:.1f} h'
    if feat == 'irradiance' or feat.startswith('irr_fcst'):
        return f'{value:.0f} W/m$^2$'
    if feat == 'temperature' or feat.startswith('temp_fcst'):
        return f'{value:.1f} °C'
    if feat == 'price_mwh' or feat.startswith('price_fcst'):
        return f'{value:.0f} €/MWh'
    if feat in ('SoH_PEM', 'SoH_PV', 'SoH_BATT', 'batt_soc', 'prev_load_frac') and abs(value) <= 1.5:
        return f'{value * 100:.1f}%'
    if feat == 'storage':
        return f'{value:.0f} kg'
    if feat == 'prev_n_parallel':
        return f'{value:.0f}'
    if feat == 'V_prev':
        return f'{value:.1f} V'
    if feat == 'I_prev':
        return f'{value:.0f} A'
    return f'{value:.2f}'


def format_rules(leaves, feats, action_name, dt_hours=1.0, action_fmt=str, max_rules=8):
    total = sum(row['n'] for row in leaves)
    lines = []
    for row in leaves[:max_rules]:
        conds = []
        for feat in feats:
            if feat not in row['bounds']:
                continue
            lo, hi = row['bounds'][feat]
            label = _feature_label(feat, dt_hours)
            if np.isfinite(lo) and np.isfinite(hi):
                conds.append(f'{_fmt_val(feat, lo, dt_hours)} < {label} <= {_fmt_val(feat, hi, dt_hours)}')
            elif np.isfinite(hi):
                conds.append(f'{label} <= {_fmt_val(feat, hi, dt_hours)}')
            elif np.isfinite(lo):
                conds.append(f'{label} > {_fmt_val(feat, lo, dt_hours)}')
        share = 100.0 * row['n'] / total if total else 0.0
        lines.append(
            f"  IF {' AND '.join(conds) if conds else '(always)'}\n"
            f"     THEN {action_name} = {action_fmt(row['label'])}   "
            f"[{share:4.1f}% of training-tree steps, {row['purity']*100:.0f}% leaf purity]"
        )
    if len(leaves) > max_rules:
        lines.append(f"  ... ({len(leaves) - max_rules} smaller-share leaves omitted)")
    return lines


# =============================================================================
# %% PLOTTING HELPERS
# =============================================================================
def _all_used_features(ctrls):
    used = []
    for c in ctrls:
        for feat in c['feats']:
            if feat not in used:
                used.append(feat)
    return used


def plot_feature_importance(ax, ctrls, dt_hours):
    union = _all_used_features(ctrls)
    maps = {c['label']: _importance_map(c) for c in ctrls}
    keep = [f for f in union if max(maps[c['label']].get(f, 0.0) for c in ctrls) > 0.01]
    drl = next((c for c in ctrls if c['label'] == RL_LABEL), None)
    if drl is not None:
        ref_map = maps[RL_LABEL]
        keep.sort(key=lambda f: (ref_map.get(f, 0.0), max(maps[c['label']].get(f, 0.0) for c in ctrls)))
    else:
        keep.sort(key=lambda f: max(maps[c['label']].get(f, 0.0) for c in ctrls))

    if not keep:
        ax.text(0.5, 0.5, 'No feature importance > 1%', ha='center', va='center')
        return

    y = np.arange(len(keep))
    n = len(ctrls)
    height = 0.82 / n
    for k, c in enumerate(ctrls):
        off = (k - (n - 1) / 2.0) * height
        vals = [maps[c['label']].get(f, 0.0) for f in keep]
        ax.barh(y + off, vals, height, label=c['label'], color=c['color'], edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels([_feature_label(f, dt_hours) for f in keep])
    ax.set_xlabel('Relative importance in string-count decision', fontweight='bold')
    ax.legend(loc='lower right', framealpha=1.0, ncol=1)


def plot_macro_categories(ax, ctrls):
    cats = [name for name, _ in FEATURE_GROUPS]
    x = np.arange(len(cats))
    n = len(ctrls)
    width = 0.84 / n
    for k, c in enumerate(ctrls):
        imp = _importance_map(c)
        vals = [sum(imp.get(f, 0.0) for f in group) for _, group in FEATURE_GROUPS]
        off = (k - (n - 1) / 2.0) * width
        ax.bar(x + off, vals, width, label=c['label'], color=c['color'], edgecolor='white')
    ax.set_ylabel('Aggregated surrogate importance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10)
    ax.legend(loc='upper right', framealpha=1.0, ncol=2)


def plot_fidelity_vs_complexity(ax, ctrls):
    for c in ctrls:
        depths = sorted(c['res'])
        ax.plot(
            depths,
            [c['res'][d]['fidelity'] * 100 for d in depths],
            marker=MARKERS.get(c['label'], 'o'),
            lw=2.2,
            color=c['color'],
            label=c['label'],
        )
    ax.axhline(
        TARGET_FIDELITY * 100,
        color='#7F8C8D', ls='--', lw=1.4,
        label=f'{TARGET_FIDELITY*100:.0f}% target',
    )
    ax.set_xlabel('Decision-tree depth', fontweight='bold')
    ax.set_ylabel('Held-out fidelity to controller action (%)', fontweight='bold')
    ax.legend(loc='lower right', framealpha=1.0, ncol=2)


def _infer_npar_step(dfs):
    vals = np.unique(np.concatenate([np.asarray(df['n_parallel'].dropna(), float) for df in dfs]))
    vals = np.sort(vals)
    diffs = np.diff(vals)
    diffs = diffs[diffs > 1e-9]
    if len(diffs) == 0:
        return 25.0
    return float(np.min(diffs))


def _npar_bins(dfs):
    step = _infer_npar_step(dfs)
    lo_data = min(float(df['n_parallel'].min()) for df in dfs)
    hi_data = max(float(df['n_parallel'].max()) for df in dfs)
    edges = np.arange(lo_data - step/2.0, hi_data + 1.5*step, step)
    return edges, lo_data, hi_data


def plot_action_distribution(ax, ctrls):
    dfs = [c['df'] for c in ctrls]
    bins, _, _ = _npar_bins(dfs)
    for c in ctrls:
        ax.hist(
            c['df']['n_parallel'], bins=bins, histtype='step', linewidth=2.2,
            color=c['color'], label=c['label'], density=False,
        )
    ax.set_xlabel('Selected parallel strings (MOSFET configuration)', fontweight='bold')
    ax.set_ylabel('Daylight decision count', fontweight='bold')
    ax.legend(framealpha=1.0, ncol=2)


def plot_switching_heatmap(fig, axes, ctrls):
    dfs = [c['df_full'] for c in ctrls]
    bins, lo, hi = _npar_bins(dfs)
    axes_flat = np.asarray(axes, dtype=object).ravel()
    for ax, c in zip(axes_flat, ctrls):
        t = c['df_full'][['n_parallel']].copy()
        t['prev_n'] = t['n_parallel'].shift(1)
        t = t.dropna()
        h = ax.hist2d(
            t['prev_n'], t['n_parallel'], bins=[bins, bins],
            cmap=CMAPS.get(c['label'], 'viridis'), cmin=1,
        )
        cb = fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Transitions')
        ax.plot([lo, hi], [lo, hi], color='#2C3E50', ls='--', lw=1.2)
        ax.set_title(c['label'], fontweight='bold', color=c['color'])
        ax.set_xlabel('Config at t-1')
        ax.set_ylabel('Config at t')
    for ax in axes_flat[len(ctrls):]:
        ax.set_visible(False)


def plot_policy_boundary(fig, axes, ctrls, colorbar_ax=None):
    """Five-controller policy map with a dedicated colour-bar axis."""
    dfs = [c['df'] for c in ctrls]
    vmin = min(float(df['n_parallel'].min()) for df in dfs)
    vmax = max(float(df['n_parallel'].max()) for df in dfs)
    axes_arr = np.asarray(axes, dtype=object)
    ncols = axes_arr.shape[1] if axes_arr.ndim > 1 else len(axes_arr)
    axes_flat = axes_arr.ravel()
    sc = None

    for i, (ax, c) in enumerate(zip(axes_flat, ctrls)):
        sc = ax.scatter(
            c['df']['irradiance'], c['df']['T_pem'] - 273.15,
            c=c['df']['n_parallel'], cmap='viridis', alpha=0.75, s=14,
            vmin=vmin, vmax=vmax, edgecolor='none',
        )
        ax.set_title(c['label'], fontweight='bold', color=c['color'], pad=7)
        if i + ncols >= len(ctrls):
            ax.set_xlabel('Irradiance (W/m$^2$)', labelpad=6)
        if i % ncols == 0:
            ax.set_ylabel('PEM temperature before decision (°C)', labelpad=7)

    for ax in axes_flat[len(ctrls):]:
        ax.set_visible(False)

    if sc is not None:
        if colorbar_ax is None:
            colorbar_ax = fig.add_axes([0.915, 0.17, 0.018, 0.68])
        cb = fig.colorbar(sc, cax=colorbar_ax)
        cb.set_label('Action: parallel strings', rotation=270, labelpad=20)


def plot_timeseries_snapshot(ax, ctrls, dt_hours=1.0, window_hours=72):
    ref_df = ctrls[0]['df_full'].reset_index(drop=True)
    high = ref_df.index[ref_df['irradiance'] > 500]
    if len(high) == 0:
        ax.set_title('No high-irradiance window found')
        return
    window = max(1, int(round(window_hours / dt_hours)))
    s = int(high[0])
    e = min(len(ref_df), s + window)
    ax2 = ax.twinx()
    sub_ref = ref_df.iloc[s:e]
    t_ref = np.arange(len(sub_ref)) * dt_hours
    ax2.fill_between(t_ref, sub_ref['irradiance'].to_numpy(), color=C_IRR, alpha=0.12)
    ax2.plot(t_ref, sub_ref['irradiance'].to_numpy(), color=C_IRR, lw=1.3, alpha=0.55, label='Irradiance')

    for c in ctrls:
        sub = c['df_full'].reset_index(drop=True).iloc[s:e]
        tt = np.arange(len(sub)) * dt_hours
        ax.step(
            tt, sub['n_parallel'].to_numpy(), where='post', color=c['color'],
            lw=2.4 if c['label'] == RL_LABEL else 1.7,
            linestyle=LINESTYLES.get(c['label'], '-'), label=c['label'], alpha=0.92,
        )
    ax.set_xlabel('Time from snapshot start (h)', fontweight='bold')
    ax.set_ylabel('Parallel strings', fontweight='bold')
    ax2.set_ylabel('Irradiance (W/m$^2$)', fontweight='bold')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', framealpha=1.0, ncol=2)
    ax2.grid(False)


# =============================================================================
# %% THERMAL-ACTION XAI
# =============================================================================
def _restart_stats(df_full, threshold_C=COLDSTART_THRESHOLD_C):
    if 'H2_prod' not in df_full.columns or 'T_pem' not in df_full.columns:
        return dict(restarts=0, cold=0, mean_T=np.nan)
    producing = df_full['H2_prod'].to_numpy(float) > 1e-9
    was_idle = np.r_[True, ~producing[:-1]]
    T_prev_C = np.r_[np.nan, df_full['T_pem'].to_numpy(float)[:-1]] - 273.15
    restart = producing & was_idle
    cold = restart & np.isfinite(T_prev_C) & (T_prev_C < threshold_C)
    vals = T_prev_C[restart & np.isfinite(T_prev_C)]
    return dict(
        restarts=int(restart.sum()),
        cold=int(cold.sum()),
        mean_T=float(np.mean(vals)) if len(vals) else np.nan,
    )


def _thermal_features_for_controller(label):
    if label in ('MOO+PreWarm', 'MOO+Hyst+PreWarm'):
        if PREWARM_RULE_MODE_XAI == 'distilled':
            raise ValueError(
                "The archived distilled PreWarm rule was fitted at hourly resolution and "
                "cannot be remapped to 15-min forecast indices without refitting. Use "
                "PREWARM_RULE_MODE_XAI='anticipatory' for the final five-controller study."
            )
        return FORECAST_IRR_FEATS.copy()
    if label == RL_LABEL:
        return DRL_STRING_FEATURES.copy()
    return []


def run_keepwarm_xai(ctrls, plots_dir, loc, dt_hours, target_fidelity=0.85):
    """Compare warm-standby actions across all five controllers."""
    bundles = []
    occupancy_rows = []

    for c in ctrls:
        dff = c['df_full']
        if 'T_keepwarm_C' not in dff.columns:
            continue
        prep = _prepare(dff, dt_hours, daytime_only=False)
        if 'H2_prod' in dff.columns:
            idle_mask = dff['H2_prod'].to_numpy(float) <= 1e-9
            dk = prep.loc[idle_mask].reset_index(drop=True)
        else:
            dk = prep

        levels = sorted(np.round(dk['T_keepwarm_C'].dropna().astype(float)).astype(int).unique())
        stats = _restart_stats(dff)
        for level in levels:
            occupancy_rows.append(dict(
                controller=c['label'], level_C=int(level),
                idle_steps=int((np.round(dk['T_keepwarm_C']) == level).sum()),
                idle_fraction=float((np.round(dk['T_keepwarm_C']) == level).mean()) if len(dk) else 0.0,
                cold_starts=stats['cold'], mean_restart_T_C=stats['mean_T'],
            ))

        print(
            f"  {c['label']:<20} keep-warm levels={levels} | "
            f"restarts={stats['restarts']} cold<{COLDSTART_THRESHOLD_C:.0f}C={stats['cold']} "
            f"mean restart T={stats['mean_T']:.1f} C"
        )

        if len(levels) < 2:
            continue

        requested = _thermal_features_for_controller(c['label'])
        feats = _available_features(dk, requested, c['label'], 'keep-warm')
        clf, fid, res = extract_surrogate_tree(
            dk, feats, MAX_TREE_DEPTH, target_fidelity, target='T_keepwarm_C'
        )
        bundle = dict(
            label=c['label'], color=c['color'], df=dk, df_full=dff,
            clf=clf, res=res, fid=fid, feats=feats,
        )
        bundle['sa'] = state_awareness_fraction(bundle)
        bundle['mem'] = memory_fraction(bundle)
        bundle['ant'] = anticipation_fraction(bundle)
        bundles.append(bundle)
        print(
            f"    surrogate fidelity={fid*100:5.1f}% | plant-state={bundle['sa']*100:4.1f}% | "
            f"memory={bundle['mem']*100:4.1f}% | forecast={bundle['ant']*100:4.1f}%"
        )

    if not occupancy_rows:
        print('  No T_keepwarm_C columns found -- keep-warm XAI skipped.')
        return [], pd.DataFrame()

    occ = pd.DataFrame(occupancy_rows)
    levels_all = sorted(occ['level_C'].unique())
    x = np.arange(len(levels_all))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.7))

    n = len(ctrls)
    width = 0.82 / max(n, 1)
    for k, c in enumerate(ctrls):
        sub = occ[occ['controller'] == c['label']].set_index('level_C')
        vals = [100.0 * float(sub.loc[L, 'idle_fraction']) if L in sub.index else 0.0 for L in levels_all]
        off = (k - (n - 1) / 2.0) * width
        axes[0].bar(x + off, vals, width, label=c['label'], color=c['color'], edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{L}°C' for L in levels_all])
    axes[0].set_ylabel('Share of idle steps (%)', fontweight='bold')
    axes[0].set_xlabel('Warm-standby set-point', fontweight='bold')
    axes[0].legend(framealpha=1.0, fontsize=9, ncol=2)

    if bundles:
        union = _all_used_features(bundles)
        maps = {b['label']: _importance_map(b) for b in bundles}
        keep = [f for f in union if max(maps[b['label']].get(f, 0.0) for b in bundles) > 0.01]
        keep.sort(key=lambda f: max(maps[b['label']].get(f, 0.0) for b in bundles))
        y = np.arange(len(keep))
        h = 0.82 / len(bundles)
        for k, b in enumerate(bundles):
            off = (k - (len(bundles) - 1) / 2.0) * h
            axes[1].barh(
                y + off, [maps[b['label']].get(f, 0.0) for f in keep], h,
                label=b['label'], color=b['color'], edgecolor='white',
            )
        axes[1].set_yticks(y)
        axes[1].set_yticklabels([_feature_label(f, dt_hours) for f in keep], fontsize=10)
        axes[1].set_xlabel('Relative importance in keep-warm decision', fontweight='bold')
        axes[1].legend(framealpha=1.0, fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'All keep-warm set-points are constant', ha='center', va='center')
        axes[1].set_axis_off()

    fig.tight_layout()
    out = plots_dir / f'xai_8_keepwarm_comparison_{loc}.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out}')

    occ.to_csv(plots_dir / f'xai_keepwarm_occupancy_{loc}.csv', index=False)
    return bundles, occ


def run_chiller_xai(ctrls, plots_dir, loc, dt_hours, target_fidelity=0.85):
    """Explain DRL chiller set-point if the action is present and non-constant."""
    drl = next((c for c in ctrls if c['label'] == RL_LABEL), None)
    if drl is None or 'T_chiller_C' not in drl['df_full'].columns:
        return None
    d = _prepare(drl['df_full'], dt_hours, daytime_only=True)
    levels = sorted(np.round(d['T_chiller_C'].dropna().astype(float)).astype(int).unique())
    if len(levels) < 2:
        print(f'  DRL chiller held a single level {levels}; chiller XAI skipped.')
        return None

    feats = _available_features(d, DRL_STRING_FEATURES, RL_LABEL, 'chiller')
    clf, fid, res = extract_surrogate_tree(d, feats, MAX_TREE_DEPTH, target_fidelity, target='T_chiller_C')
    bundle = dict(label=f'{RL_LABEL} chiller', color=C_RL, df=d, df_full=drl['df_full'], clf=clf, res=res, fid=fid, feats=feats)
    bundle['sa'] = state_awareness_fraction(bundle)
    bundle['mem'] = memory_fraction(bundle)
    bundle['ant'] = anticipation_fraction(bundle)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    counts = [int((np.round(d['T_chiller_C']) == L).sum()) for L in levels]
    axes[0].bar([str(L) for L in levels], counts, color=C_RL, edgecolor='white')
    axes[0].set_xlabel('Chiller ceiling set-point (°C)', fontweight='bold')
    axes[0].set_ylabel('Daylight decision count', fontweight='bold')

    imp = _importance_map(bundle)
    keep = [f for f in feats if imp.get(f, 0.0) > 0.01]
    keep.sort(key=lambda f: imp[f])
    axes[1].barh([_feature_label(f, dt_hours) for f in keep], [imp[f] for f in keep], color=C_RL, edgecolor='white')
    axes[1].set_xlabel('Relative importance in chiller decision', fontweight='bold')
    fig.tight_layout()
    out = plots_dir / f'xai_9_chiller_policy_{loc}.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(
        f"  DRL chiller surrogate fidelity={fid*100:.1f}% | plant-state={bundle['sa']*100:.1f}% | "
        f"memory={bundle['mem']*100:.1f}% | forecast={bundle['ant']*100:.1f}%"
    )
    print(f'  -> {out}')
    return bundle


# =============================================================================
# %% RULE REPORT / TABULAR EXPORTS
# =============================================================================
def write_rules_report(ctrls, kw_bundles, ch_bundle, loc, plots_dir, dt_hours, rule_depth=RULE_DEPTH):
    lines = [
        f"{'='*88}",
        f"DISTILLED CONTROL RULES -- {loc}  (depth<={rule_depth} surrogate)",
        f"{'='*88}",
        'String-action rules use controller-specific DECISION-TIME inputs.',
    ]

    for c in ctrls:
        depth = min(rule_depth, max(c['res']))
        model = c['res'][depth]['model']
        fid_d = c['res'][depth]['fidelity']
        rules = format_rules(
            tree_to_rules(model, c['feats']), c['feats'], 'string config', dt_hours,
            action_fmt=lambda v: f'{int(v)} strings',
        )
        lines.append(
            f"\n{c['label']} -- string configuration "
            f"(depth-{depth} fidelity {fid_d*100:.0f}%, selected-depth {c['fid']*100:.0f}%)"
        )
        lines += rules

    for b in kw_bundles:
        depth = min(rule_depth, max(b['res']))
        model = b['res'][depth]['model']
        fid_d = b['res'][depth]['fidelity']
        rules = format_rules(
            tree_to_rules(model, b['feats']), b['feats'], 'keep-warm set-point', dt_hours,
            action_fmt=lambda v: f'{int(v)} °C',
        )
        lines.append(
            f"\n{b['label']} -- keep-warm decision "
            f"(depth-{depth} fidelity {fid_d*100:.0f}%, selected-depth {b['fid']*100:.0f}%)"
        )
        lines += rules

    if ch_bundle is not None:
        b = ch_bundle
        depth = min(rule_depth, max(b['res']))
        model = b['res'][depth]['model']
        fid_d = b['res'][depth]['fidelity']
        rules = format_rules(
            tree_to_rules(model, b['feats']), b['feats'], 'chiller set-point', dt_hours,
            action_fmt=lambda v: f'{int(v)} °C',
        )
        lines.append(
            f"\nDRL -- chiller decision "
            f"(depth-{depth} fidelity {fid_d*100:.0f}%, selected-depth {b['fid']*100:.0f}%)"
        )
        lines += rules

    report = '\n'.join(lines)
    print(report)
    out = plots_dir / f'xai_10_rules_{loc}.txt'
    out.write_text(report, encoding='utf-8')
    print(f'  -> {out}')
    return report


def export_surrogate_summary(ctrls, loc, plots_dir):
    rows = []
    for c in ctrls:
        rows.append(dict(
            location=loc,
            controller=c['label'],
            fidelity=c['fid'],
            plant_state_importance=c['sa'],
            electrical_memory_importance=c['mem'],
            forecast_importance=c['ant'],
            n_features=len(c['feats']),
            features=' | '.join(c['feats']),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(plots_dir / f'xai_surrogate_summary_{loc}.csv', index=False)

    imp_rows = []
    for c in ctrls:
        imp = _importance_map(c)
        for feat in c['feats']:
            imp_rows.append(dict(location=loc, controller=c['label'], feature=feat, importance=imp.get(feat, 0.0)))
    pd.DataFrame(imp_rows).to_csv(plots_dir / f'xai_feature_importance_{loc}.csv', index=False)
    return out


# =============================================================================
# %% MAIN XAI EXECUTION
# =============================================================================
def _load_five_controllers(loc, data_dir):
    loaded = []
    missing = []
    for spec in CONTROLLERS:
        p = data_dir / f"{spec['tag']}_results_{loc}.csv"
        if p.exists():
            loaded.append(dict(
                label=spec['label'], color=spec['color'], tag=spec['tag'],
                df_full=pd.read_csv(p), path=p,
            ))
        else:
            missing.append((spec['label'], p))

    if missing:
        for label, p in missing:
            print(f'  MISSING {label:<20}: {p.name}')
        if REQUIRE_ALL_FIVE:
            print('  ERROR: the 5-controller XAI comparison requires all five CSVs. Skipping location.')
            return []

    if not any(c['label'] == RL_LABEL for c in loaded):
        print('  ERROR: DRL CSV is required. Skipping location.')
        return []
    return loaded


def _infer_dt_hours(loaded):
    lengths = {c['label']: len(c['df_full']) for c in loaded}
    if len(set(lengths.values())) != 1:
        print(f'  WARNING: controller CSV lengths differ: {lengths}')
    n = lengths.get(RL_LABEL, next(iter(lengths.values())))
    return (8760.0 / n) if n else 1.0


def run_xai(loc, out_dir, target_fidelity=TARGET_FIDELITY):
    data_dir  = Path(out_dir) / f'Results_{loc}' / 'Data'
    plots_dir = Path(out_dir) / f'Results_{loc}' / 'Plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*78}\nEXPLAINABLE AI -- FIVE-CONTROLLER COMPARISON -- {loc}\n{'='*78}")
    loaded = _load_five_controllers(loc, data_dir)
    if not loaded:
        return None
    print('  Controllers loaded:', ', '.join(c['label'] for c in loaded))

    dt_hours = _infer_dt_hours(loaded)
    if not np.isclose(dt_hours, EXPECTED_DT_HOURS_XAI, rtol=0.0, atol=1e-3):
        raise ValueError(
            f"This XAI build is for 15-min trajectories (dt={EXPECTED_DT_HOURS_XAI} h), "
            f"but {loc} resolves to dt={dt_hours:g} h. Point RESULTS_ROOT_OVERRIDE to "
            "Results_1MW_15min/<Demand_demand>, not an hourly result tree."
        )
    print(f"  Detected timestep: {dt_hours:g} h  (~{round(8760/dt_hours):d} steps/year)")
    print(f"  DRL forecasts reconstructed at physical horizons {FORECAST_HORIZONS_H_XAI} h "
      f"(row offsets={FORECAST_OFFSETS_STEPS_XAI})")

    ctrls = []
    for c in loaded:
        dfp = _prepare(c['df_full'], dt_hours, daytime_only=True)
        requested = STRING_FEATURES_BY_CONTROLLER[c['label']]
        feats = _available_features(dfp, requested, c['label'], 'string-action')
        clf, fid, res = extract_surrogate_tree(
            dfp, feats, MAX_TREE_DEPTH, target_fidelity, target='n_parallel'
        )
        bundle = dict(**c, df=dfp, feats=feats, clf=clf, fid=fid, res=res)
        bundle['sa'] = state_awareness_fraction(bundle)
        bundle['mem'] = memory_fraction(bundle)
        bundle['ant'] = anticipation_fraction(bundle)
        ctrls.append(bundle)
        print(
            f"  {c['label']:<20} string fidelity={fid*100:5.1f}% | "
            f"plant-state={bundle['sa']*100:5.1f}% | memory={bundle['mem']*100:5.1f}% | "
            f"forecast={bundle['ant']*100:5.1f}% | features={len(feats)}"
        )

    union = _all_used_features(ctrls)
    fig_h = max(7.5, 0.42 * len(union) + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    plot_feature_importance(ax, ctrls, dt_hours)
    fig.tight_layout()
    out = plots_dir / f'xai_1_feature_importance_{loc}.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig); print(f'  -> {out}')

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_macro_categories(ax, ctrls)
    fig.tight_layout()
    out = plots_dir / f'xai_2_macro_categories_{loc}.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig); print(f'  -> {out}')

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_fidelity_vs_complexity(ax, ctrls)
    fig.tight_layout()
    out = plots_dir / f'xai_3_fidelity_vs_complexity_{loc}.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig); print(f'  -> {out}')

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_action_distribution(ax, ctrls)
    fig.tight_layout()
    out = plots_dir / f'xai_4_action_distribution_{loc}.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig); print(f'  -> {out}')

    ncols = 3
    nrows = int(math.ceil(len(ctrls) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2*ncols, 4.8*nrows), squeeze=False)
    plot_switching_heatmap(fig, axes, ctrls)
    fig.tight_layout()
    out = plots_dir / f'xai_5_switching_heatmap_{loc}.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig); print(f'  -> {out}')

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 5.0*nrows), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.885, bottom=0.08, top=0.94,
                        wspace=0.30, hspace=0.34)
    cax = fig.add_axes([0.91, 0.17, 0.018, 0.68])
    plot_policy_boundary(fig, axes, ctrls, colorbar_ax=cax)
    out = plots_dir / f'xai_6_policy_boundary_{loc}.png'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig); print(f'  -> {out}')

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_timeseries_snapshot(ax, ctrls, dt_hours)
    fig.tight_layout()
    out = plots_dir / f'xai_7_behavior_snapshot_{loc}.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig); print(f'  -> {out}')

    kw_bundles, kw_occ = run_keepwarm_xai(ctrls, plots_dir, loc, dt_hours, target_fidelity)

    ch_bundle = run_chiller_xai(ctrls, plots_dir, loc, dt_hours, target_fidelity)

    write_rules_report(ctrls, kw_bundles, ch_bundle, loc, plots_dir, dt_hours)

    summary_df = export_surrogate_summary(ctrls, loc, plots_dir)
    print(f"XAI complete for {loc}. Outputs saved to {plots_dir}")

    return dict(
        loc=loc,
        ctrls={c['label']: dict(fid=c['fid'], sa=c['sa'], mem=c['mem'], ant=c['ant']) for c in ctrls},
        summary_df=summary_df,
        keepwarm=kw_bundles,
        chiller=ch_bundle,
    )


if __name__ == '__main__':
    print(f'XAI demand mode: {_XAI_MODE}')
    print('XAI reading from:', OUT_DIR)
    print(f'XAI result-tree score: {_OUT_SCORE} (45 = all 5 CSVs at all 3 sites)')
    results = []
    for _loc in LOCATIONS:
        out = run_xai(_loc, OUT_DIR, TARGET_FIDELITY)
        if out is not None:
            results.append(out)

    if results:
        rows = []
        for r in results:
            for label in CTRL_LABELS:
                if label not in r['ctrls']:
                    continue
                m = r['ctrls'][label]
                rows.append(dict(
                    location=r['loc'], controller=label,
                    fidelity_pct=100*m['fid'],
                    plant_state_pct=100*m['sa'],
                    memory_pct=100*m['mem'],
                    forecast_pct=100*m['ant'],
                ))
        cross = pd.DataFrame(rows)
        cross_path = Path(OUT_DIR) / 'xai_cross_location_summary.csv'
        cross.to_csv(cross_path, index=False)

        print(f"\n{'='*100}\nCROSS-LOCATION XAI SUMMARY\n{'='*100}")
        print('  Values: held-out fidelity / plant-state importance / electrical-memory importance / forecast importance (%)')
        for loc in [r['loc'] for r in results]:
            print(f'\n  {loc}')
            sub = cross[cross['location'] == loc]
            for label in CTRL_LABELS:
                row = sub[sub['controller'] == label]
                if row.empty:
                    print(f'    {label:<20} --')
                    continue
                q = row.iloc[0]
                print(
                    f"    {label:<20} {q.fidelity_pct:5.1f} / {q.plant_state_pct:5.1f} / "
                    f"{q.memory_pct:5.1f} / {q.forecast_pct:5.1f}"
                )

        print('\nInterpretation notes:')
        print('  * MOO is greedy, but not state-blind: its physical operating-point calculation uses pre-step')
        print('    PEM temperature and PV/PEM SoH. The corrected XAI therefore allows those features.')
        print('  * MOO+Hyst additionally uses the held string configuration; memory importance is expected.')
        print('  * PreWarm baselines use forecast irradiance directly for the THERMAL rule, not for the same-step')
        print('    greedy string search. Their anticipation should therefore appear mainly in xai_8.')
        print('  * DRL is the only controller in this comparison whose string action can use the full plant state,')
        print('    electrical history, market/time and G/T/price look-ahead simultaneously.')
        print(f'\n  -> {cross_path}')
