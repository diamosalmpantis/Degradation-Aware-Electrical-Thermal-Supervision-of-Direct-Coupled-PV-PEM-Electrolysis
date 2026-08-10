# ======================================================================
# DEGRADATION + PEM REPLACEMENT SENSITIVITY
# Frozen-trajectory post-processing ONLY -- NO PPO TRAINING / NO RESIMULATION
#
# Analysis matrix:
#   degradation severity:       0.5x, 1.0x, 1.5x
#   PEM EOL SoH:                0.75, 0.80, 0.85
#   replaceable stack CAPEX:    0.50, 0.75, 1.00
#
# The common degradation factor multiplies the measured annual PEM SoH loss
# from each frozen trajectory. It does NOT alter dispatch, H2 production,
# grid use, thermal operation, or the PPO reward.
# ======================================================================

from pathlib import Path
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------------------
RESULTS_ROOT_OVERRIDE = None
# Example:
# RESULTS_ROOT_OVERRIDE = "/content/Results_1MW_daily/Daily_demand"

DEGRADATION_SCALES = (0.5, 1.0, 1.5)
PEM_EOL_SOHS       = (0.75, 0.80, 0.85)
REPLACEMENT_FRACS  = (0.50, 0.75, 1.00)

BASE_DEG_SCALE = 1.0
BASE_EOL_SOH   = 0.80
BASE_REPL_FRAC = 1.00

LOCATIONS = ("Lund_2021", "Prague_2021", "Odeillo_2021")

CONTROLLERS = {
    "MOO":                 "moo_results_{loc}.csv",
    "MOO+Hyst":            "hyst_results_{loc}.csv",
    "MOO+PreWarm":         "prewarm_results_{loc}.csv",
    "MOO+Hyst+PreWarm":    "hystprewarm_results_{loc}.csv",
    "RL Agent":            "rl_results_{loc}.csv",
}

# Frozen production values used only as an audit of the base-case calculation.
EXPECTED_BASE_LCOH = {
    "Lund_2021": {
        "MOO": 12.98, "MOO+Hyst": 13.30, "MOO+PreWarm": 9.91,
        "MOO+Hyst+PreWarm": 10.14, "RL Agent": 10.61,
    },
    "Prague_2021": {
        "MOO": 11.57, "MOO+Hyst": 11.85, "MOO+PreWarm": 9.27,
        "MOO+Hyst+PreWarm": 9.49, "RL Agent": 9.58,
    },
    "Odeillo_2021": {
        "MOO": 9.90, "MOO+Hyst": 10.07, "MOO+PreWarm": 7.83,
        "MOO+Hyst+PreWarm": 7.95, "RL Agent": 7.88,
    },
}

# ----------------------------------------------------------------------
# EXACT FROZEN ECONOMIC CONSTANTS FROM THE PRODUCTION MODEL
# ----------------------------------------------------------------------
PEM_CAPACITY_KW = 1000.0
H2_PEAK_KGH = 19.0

PV_MODULE_W = 412.5
N_SERIES_PV = 5

SITE_SIZING = {
    "Lund_2021":    (1.75, 7800.0, 550.0),  # PV ratio, tank kg, battery kWh
    "Prague_2021":  (1.75, 5300.0, 550.0),
    "Odeillo_2021": (1.25, 1600.0, 450.0),
}

LCOH_PV_EUR_KWP       = 700.0
LCOH_PEM_EUR_KW       = 1200.0
LCOH_BATT_EUR_KWH     = 200.0
LCOH_TANK_EUR_KG      = 500.0
LCOH_COMP_EUR_KGH     = 6000.0
LCOH_BOP_FRAC         = 0.20
LCOH_CTRL_EUR         = 8000.0

LCOH_BATT_CONV_EUR_KW = 75.0
LCOH_BATT_CONV_LIFE   = 15
LCOH_BATT_CONV_OM     = 0.015

LCOH_GRID_INV_EUR_KW  = 120.0
LCOH_GRID_INV_KW      = 400.0
LCOH_GRID_INV_LIFE    = 15
LCOH_GRID_INV_OM      = 0.020

LCOH_PV_LIFE          = 30
LCOH_BOP_LIFE         = 30
LCOH_BATT_LIFE        = 10
LCOH_TANK_LIFE        = 25
LCOH_COMP_LIFE        = 15

LCOH_PV_OM            = 0.010
LCOH_PEM_OM           = 0.030
LCOH_BATT_OM          = 0.010
LCOH_TANK_OM          = 0.005
LCOH_COMP_OM          = 0.020

LCOH_WACC             = 0.06
LCOH_PROJECT_LIFE     = 30

LCOH_PEM_NEW_SOH      = 1.00

O2_KG_PER_KG_H2       = 8.0
O2_PRICE_EUR_KG       = 0.10

# The frozen main daily trajectories are hourly.
DT_HOURS = 1.0


# ======================================================================
# UTILITIES
# ======================================================================
def crf(r, n):
    return r * (1.0 + r)**n / ((1.0 + r)**n - 1.0)


def ann_capex_om(capex, life, om_rate):
    return capex * crf(LCOH_WACC, life), capex * om_rate


def site_string_count(pv_ratio):
    """Exact installed parallel-string count used by the frozen plant sizing."""
    string_w = N_SERIES_PV * PV_MODULE_W
    return int(round(pv_ratio * (PEM_CAPACITY_KW * 1000.0) / string_w))


def resolve_results_root():
    if RESULTS_ROOT_OVERRIDE is not None:
        root = Path(RESULTS_ROOT_OVERRIDE)
        if root.is_dir():
            return root
        raise FileNotFoundError(f"RESULTS_ROOT_OVERRIDE does not exist: {root}")

    candidates = [
        Path("/content/Results_1MW_daily/Daily_demand"),
        Path("/content/drive/MyDrive/PVPEM_Results/Results_1MW_daily/Daily_demand"),
    ]

    for root in candidates:
        if all((root / f"Results_{loc}").is_dir() for loc in LOCATIONS):
            return root

    # Fallback recursive search in common Colab locations.
    for search_root in [Path("/content"), Path("/content/drive/MyDrive")]:
        if not search_root.exists():
            continue
        try:
            for p in search_root.rglob("Daily_demand"):
                if all((p / f"Results_{loc}").is_dir() for loc in LOCATIONS):
                    return p
        except Exception:
            pass

    raise FileNotFoundError(
        "Could not locate the frozen DAILY results tree.\n"
        "Expected e.g. /content/Results_1MW_daily/Daily_demand"
    )


def load_frozen_trajectories(root):
    data = {}
    for loc in LOCATIONS:
        ddir = root / f"Results_{loc}" / "Data"
        if not ddir.is_dir():
            raise FileNotFoundError(f"Missing Data directory: {ddir}")

        data[loc] = {}
        for ctrl, pattern in CONTROLLERS.items():
            path = ddir / pattern.format(loc=loc)
            if not path.is_file():
                raise FileNotFoundError(f"Missing frozen trajectory:\n{path}")
            df = pd.read_csv(path)

            required = {"H2_prod", "SoH_PEM", "grid_import_eur"}
            missing = required - set(df.columns)
            if missing:
                raise KeyError(f"{path.name} is missing columns: {sorted(missing)}")

            if len(df) < 8000:
                raise ValueError(
                    f"{path.name} contains only {len(df)} rows; expected a full annual "
                    "frozen daily-demand trajectory."
                )

            data[loc][ctrl] = df

    return data


def fixed_site_costs(loc):
    pv_ratio, tank_kg, batt_kwh = SITE_SIZING[loc]

    n_par_max = site_string_count(pv_ratio)
    pv_peak_kw = n_par_max * N_SERIES_PV * PV_MODULE_W / 1000.0

    pem_kw = PEM_CAPACITY_KW
    batt_kw = batt_kwh  # frozen plant uses 1C converter sizing

    capex_pv        = pv_peak_kw * LCOH_PV_EUR_KWP
    capex_pem       = pem_kw * LCOH_PEM_EUR_KW
    capex_batt      = batt_kwh * LCOH_BATT_EUR_KWH
    capex_batt_conv = batt_kw * LCOH_BATT_CONV_EUR_KW
    capex_grid_inv  = LCOH_GRID_INV_KW * LCOH_GRID_INV_EUR_KW
    capex_bop       = capex_pem * LCOH_BOP_FRAC
    capex_ctrl      = LCOH_CTRL_EUR

    capex_tank = tank_kg * LCOH_TANK_EUR_KG
    capex_comp = H2_PEAK_KGH * LCOH_COMP_EUR_KGH

    ac_pv, om_pv = ann_capex_om(capex_pv, LCOH_PV_LIFE, LCOH_PV_OM)

    # Initial PEM stack is annualized over the 30-year project exactly as in
    # the frozen production economics. Controller/scenario-specific REPEX is
    # added separately.
    ac_pem = capex_pem * crf(LCOH_WACC, LCOH_PROJECT_LIFE)
    om_pem = capex_pem * LCOH_PEM_OM

    ac_batt, om_batt = ann_capex_om(
        capex_batt, LCOH_BATT_LIFE, LCOH_BATT_OM
    )
    ac_batt_conv, om_batt_conv = ann_capex_om(
        capex_batt_conv, LCOH_BATT_CONV_LIFE, LCOH_BATT_CONV_OM
    )
    ac_grid_inv, om_grid_inv = ann_capex_om(
        capex_grid_inv, LCOH_GRID_INV_LIFE, LCOH_GRID_INV_OM
    )
    ac_bop, om_bop = ann_capex_om(
        capex_bop, LCOH_BOP_LIFE, LCOH_PEM_OM
    )
    ac_ctrl = capex_ctrl * crf(LCOH_WACC, LCOH_PROJECT_LIFE)

    fixed_prod_ann_base = (
        ac_pv + om_pv
        + ac_pem + om_pem
        + ac_batt + om_batt
        + ac_batt_conv + om_batt_conv
        + ac_grid_inv + om_grid_inv
        + ac_bop + om_bop
        + ac_ctrl
    )

    ac_tank, om_tank = ann_capex_om(
        capex_tank, LCOH_TANK_LIFE, LCOH_TANK_OM
    )
    ac_comp, om_comp = ann_capex_om(
        capex_comp, LCOH_COMP_LIFE, LCOH_COMP_OM
    )
    fixed_stor_ann = ac_tank + om_tank + ac_comp + om_comp

    return {
        "pv_peak_kw": pv_peak_kw,
        "n_par_max": n_par_max,
        "capex_pem": capex_pem,
        "ac_pem": ac_pem,
        "fixed_prod_ann_base": fixed_prod_ann_base,
        "fixed_stor_ann": fixed_stor_ann,
    }


def frozen_operating_terms(df):
    """Annual operating quantities that remain frozen in every sensitivity case."""
    sim_years = max(len(df) * DT_HOURS / 8760.0, 1e-12)

    h2_prod_yr = pd.to_numeric(df["H2_prod"], errors="coerce").fillna(0.0).sum() / sim_years

    grid_import_yr = (
        pd.to_numeric(df["grid_import_eur"], errors="coerce").fillna(0.0).sum()
        / sim_years
    )

    if "grid_export_eur" in df:
        grid_export_yr = (
            pd.to_numeric(df["grid_export_eur"], errors="coerce").fillna(0.0).sum()
            / sim_years
        )
    else:
        grid_export_yr = 0.0

    o2_revenue_yr = h2_prod_yr * O2_KG_PER_KG_H2 * O2_PRICE_EUR_KG
    variable_net = grid_import_yr - grid_export_yr - o2_revenue_yr

    soh = pd.to_numeric(df["SoH_PEM"], errors="coerce").dropna()
    if soh.empty:
        raise ValueError("No valid SoH_PEM values.")

    soh_end = float(soh.iloc[-1])

    # Evaluations start from a fresh stack at SoH=1.0. The first CSV row is
    # already after one simulation step, so do not use it as the initial SoH.
    annual_soh_loss_base = max(0.0, LCOH_PEM_NEW_SOH - soh_end) / sim_years

    return {
        "sim_years": sim_years,
        "h2_prod_yr": h2_prod_yr,
        "grid_import_yr": grid_import_yr,
        "grid_export_yr": grid_export_yr,
        "o2_revenue_yr": o2_revenue_yr,
        "variable_net": variable_net,
        "soh_end_base": soh_end,
        "annual_soh_loss_base": annual_soh_loss_base,
    }


def replacement_economics(annual_loss_base, capex_pem,
                          degradation_scale, eol_soh, replacement_frac):
    """
    Apply a common multiplicative severity factor to the measured annual PEM SoH loss.

    This is intentionally a POST-PROCESSING uncertainty analysis:
    operation is frozen; only the degradation-to-lifetime projection changes.
    """
    annual_loss = max(0.0, annual_loss_base * float(degradation_scale))
    degradation_budget = LCOH_PEM_NEW_SOH - float(eol_soh)

    if annual_loss <= 1e-12:
        life = math.inf
        years = []
    else:
        life = degradation_budget / annual_loss
        years = []
        t = life
        while t < LCOH_PROJECT_LIFE - 1e-9:
            years.append(float(t))
            t += life

    capex_each = capex_pem * float(replacement_frac)
    npv = sum(
        capex_each / ((1.0 + LCOH_WACC)**yr)
        for yr in years
    )

    project_crf = crf(LCOH_WACC, LCOH_PROJECT_LIFE)
    initial_stack_ann = capex_pem * project_crf
    repex_ann = npv * project_crf

    return {
        "annual_soh_loss_scaled": annual_loss,
        "projected_stack_life_yr": life,
        "replacement_years": years,
        "stack_replacements": len(years),
        "replacement_npv": npv,
        "repex_ann": repex_ann,
        "stack_ann_total": initial_stack_ann + repex_ann,
    }


def scenario_economics(df, loc, degradation_scale, eol_soh, replacement_frac):
    site = fixed_site_costs(loc)
    op = frozen_operating_terms(df)

    rep = replacement_economics(
        op["annual_soh_loss_base"],
        site["capex_pem"],
        degradation_scale,
        eol_soh,
        replacement_frac,
    )

    # Same structure as calculate_lcoh() in the production model:
    # replace the shared initial-stack annualized CAPEX with
    # initial stack + discounted controller/scenario-specific REPEX.
    prod_ann = (
        site["fixed_prod_ann_base"]
        - site["ac_pem"]
        + rep["stack_ann_total"]
        + op["variable_net"]
    )

    lcoh = prod_ann / max(op["h2_prod_yr"], 1e-9)
    stor_ann = prod_ann + site["fixed_stor_ann"]
    lcsh = stor_ann / max(op["h2_prod_yr"], 1e-9)

    # Equivalent end-of-year SoH if the same physical trajectory severity were
    # scaled over this one-year simulation. This is a reporting aid only.
    soh_end_scaled = LCOH_PEM_NEW_SOH - (
        op["annual_soh_loss_base"] * degradation_scale * op["sim_years"]
    )

    return {
        **op,
        **rep,
        "soh_end_scaled": soh_end_scaled,
        "prod_ann_euryr": prod_ann,
        "stor_ann_euryr": stor_ann,
        "lcoh_eurkg": lcoh,
        "lcsh_eurkg": lcsh,
    }


# ======================================================================
# FULL FACTORIAL ANALYSIS
# ======================================================================
def run_full_matrix(data):
    rows = []

    for loc in LOCATIONS:
        for ctrl, df in data[loc].items():
            for ds, eol, rf in itertools.product(
                DEGRADATION_SCALES, PEM_EOL_SOHS, REPLACEMENT_FRACS
            ):
                res = scenario_economics(df, loc, ds, eol, rf)
                rows.append({
                    "location": loc,
                    "controller": ctrl,
                    "degradation_scale": ds,
                    "pem_eol_soh": eol,
                    "replacement_capex_frac": rf,
                    "h2_prod_kgyr": res["h2_prod_yr"],
                    "base_soh_end": res["soh_end_base"],
                    "scaled_soh_end": res["soh_end_scaled"],
                    "annual_soh_loss_base": res["annual_soh_loss_base"],
                    "annual_soh_loss_scaled": res["annual_soh_loss_scaled"],
                    "stack_life_yr": res["projected_stack_life_yr"],
                    "stack_replacements_30y": res["stack_replacements"],
                    "replacement_npv_eur": res["replacement_npv"],
                    "repex_ann_euryr": res["repex_ann"],
                    "lcoh_eurkg": res["lcoh_eurkg"],
                    "lcsh_eurkg": res["lcsh_eurkg"],
                })

    return pd.DataFrame(rows)


def base_case_audit(full):
    base = full[
        np.isclose(full["degradation_scale"], BASE_DEG_SCALE)
        & np.isclose(full["pem_eol_soh"], BASE_EOL_SOH)
        & np.isclose(full["replacement_capex_frac"], BASE_REPL_FRAC)
    ].copy()

    base["expected_lcoh"] = [
        EXPECTED_BASE_LCOH[row.location][row.controller]
        for row in base.itertuples()
    ]
    base["delta_expected"] = base["lcoh_eurkg"] - base["expected_lcoh"]

    print("\n" + "="*96)
    print("BASE-CASE AUDIT -- should reproduce the frozen manuscript LCOH values")
    print("="*96)
    print(
        base[
            ["location", "controller", "lcoh_eurkg", "expected_lcoh",
             "delta_expected", "stack_life_yr", "stack_replacements_30y"]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    max_abs = float(base["delta_expected"].abs().max())
    print(f"\nMaximum |calculated - expected| LCOH = {max_abs:.3f} EUR/kg")
    if max_abs > 0.08:
        print(
            "WARNING: base-case mismatch exceeds 0.08 EUR/kg. "
            "Do not use the sensitivity results in the paper until the saved "
            "trajectory/economic-version mismatch is resolved."
        )
    else:
        print("Base-case economic audit: PASS.")

    return base


def controller_ranges(full, base):
    ranges = (
        full.groupby(["location", "controller"])
        .agg(
            lcoh_min=("lcoh_eurkg", "min"),
            lcoh_max=("lcoh_eurkg", "max"),
            lcsh_min=("lcsh_eurkg", "min"),
            lcsh_max=("lcsh_eurkg", "max"),
            life_min=("stack_life_yr", "min"),
            life_max=("stack_life_yr", "max"),
            repl_min=("stack_replacements_30y", "min"),
            repl_max=("stack_replacements_30y", "max"),
        )
        .reset_index()
    )

    base_small = base[
        ["location", "controller", "lcoh_eurkg", "lcsh_eurkg",
         "stack_life_yr", "stack_replacements_30y"]
    ].rename(columns={
        "lcoh_eurkg": "lcoh_base",
        "lcsh_eurkg": "lcsh_base",
        "stack_life_yr": "life_base",
        "stack_replacements_30y": "repl_base",
    })

    return ranges.merge(base_small, on=["location", "controller"], how="left")


def ranking_summary(full):
    rows = []

    for loc in LOCATIONS:
        d = full[full["location"] == loc]

        scenario_cols = [
            "degradation_scale", "pem_eol_soh", "replacement_capex_frac"
        ]

        piv = d.pivot_table(
            index=scenario_cols,
            columns="controller",
            values="lcoh_eurkg",
            aggfunc="first",
        )

        rl_beats_moo = float((piv["RL Agent"] < piv["MOO"]).mean() * 100.0)
        rl_beats_hpw = float(
            (piv["RL Agent"] < piv["MOO+Hyst+PreWarm"]).mean() * 100.0
        )

        winners = piv.idxmin(axis=1).value_counts()

        row = {
            "location": loc,
            "n_scenarios": len(piv),
            "RL_beats_MOO_pct": rl_beats_moo,
            "RL_beats_HystPreWarm_pct": rl_beats_hpw,
        }
        for ctrl in CONTROLLERS:
            row[f"best_count__{ctrl}"] = int(winners.get(ctrl, 0))
        rows.append(row)

    return pd.DataFrame(rows)


# ======================================================================
# PLOTS
# ======================================================================
def plot_lcoh_envelopes(ranges, outdir):
    """
    Full-factorial robustness figure:
    point = frozen base case; horizontal line = min-max LCOH over all 27 scenarios.
    This captures interactions and discrete replacement thresholds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharey=True)

    ctrl_order = list(CONTROLLERS.keys())
    y = np.arange(len(ctrl_order))

    for ax, loc in zip(axes, LOCATIONS):
        d = ranges[ranges["location"] == loc].set_index("controller").loc[ctrl_order]

        xmin = d["lcoh_min"].to_numpy(float)
        xmax = d["lcoh_max"].to_numpy(float)
        base = d["lcoh_base"].to_numpy(float)

        for yi, lo, hi, b in zip(y, xmin, xmax, base):
            ax.hlines(yi, lo, hi, linewidth=4)
            ax.plot(b, yi, "o", markersize=7)

        ax.set_yticks(y)
        ax.set_yticklabels(ctrl_order)
        ax.invert_yaxis()
        ax.set_xlabel("LCOH (EUR/kg)")
        ax.set_title(loc.replace("_2021", ""))
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Controller")
    fig.tight_layout()

    png = outdir / "degradation_replacement_LCOH_envelopes.png"
    pdf = outdir / "degradation_replacement_LCOH_envelopes.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _oat_factor_rows(full, loc, ctrl="RL Agent"):
    d = full[(full["location"] == loc) & (full["controller"] == ctrl)]

    specs = [
        (
            "Degradation severity",
            "degradation_scale",
            DEGRADATION_SCALES,
            (BASE_EOL_SOH, BASE_REPL_FRAC),
        ),
        (
            "PEM EOL SoH",
            "pem_eol_soh",
            PEM_EOL_SOHS,
            (BASE_DEG_SCALE, BASE_REPL_FRAC),
        ),
        (
            "Replacement CAPEX fraction",
            "replacement_capex_frac",
            REPLACEMENT_FRACS,
            (BASE_DEG_SCALE, BASE_EOL_SOH),
        ),
    ]

    rows = []

    for label, factor, values, fixed in specs:
        if factor == "degradation_scale":
            q = d[
                np.isclose(d["pem_eol_soh"], fixed[0])
                & np.isclose(d["replacement_capex_frac"], fixed[1])
            ]
        elif factor == "pem_eol_soh":
            q = d[
                np.isclose(d["degradation_scale"], fixed[0])
                & np.isclose(d["replacement_capex_frac"], fixed[1])
            ]
        else:
            q = d[
                np.isclose(d["degradation_scale"], fixed[0])
                & np.isclose(d["pem_eol_soh"], fixed[1])
            ]

        rows.append({
            "factor": label,
            "lcoh_min": float(q["lcoh_eurkg"].min()),
            "lcoh_max": float(q["lcoh_eurkg"].max()),
            "span": float(q["lcoh_eurkg"].max() - q["lcoh_eurkg"].min()),
        })

    return pd.DataFrame(rows).sort_values("span", ascending=True)


def plot_rl_tornado(full, base, outdir):
    """
    One-at-a-time RL tornado diagnostic.
    Other two assumptions are held at the frozen base values.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)

    for ax, loc in zip(axes, LOCATIONS):
        t = _oat_factor_rows(full, loc, "RL Agent")
        b = float(
            base[
                (base["location"] == loc)
                & (base["controller"] == "RL Agent")
            ]["lcoh_eurkg"].iloc[0]
        )

        y = np.arange(len(t))

        for yi, row in zip(y, t.itertuples()):
            ax.hlines(yi, row.lcoh_min, row.lcoh_max, linewidth=8)
            ax.plot(b, yi, "o", markersize=7)
            ax.text(row.lcoh_min, yi - 0.16, f"{row.lcoh_min:.2f}",
                    ha="left", va="bottom", fontsize=9)
            ax.text(row.lcoh_max, yi - 0.16, f"{row.lcoh_max:.2f}",
                    ha="right", va="bottom", fontsize=9)

        ax.set_yticks(y)
        ax.set_yticklabels(t["factor"])
        ax.set_xlabel("RL LCOH (EUR/kg)")
        ax.set_title(loc.replace("_2021", ""))
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    fig.tight_layout()

    png = outdir / "degradation_replacement_RL_tornado.png"
    pdf = outdir / "degradation_replacement_RL_tornado.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("="*100)
    print("DEGRADATION + PEM REPLACEMENT SENSITIVITY -- FROZEN TRAJECTORIES")
    print("="*100)
    print("NO PPO TRAINING. NO CONTROLLER RESIMULATION.")
    print("Only the degradation-to-lifetime/replacement economic projection is changed.")
    print(
        "Matrix: degradation="
        f"{DEGRADATION_SCALES}, EOL={PEM_EOL_SOHS}, "
        f"replacement fraction={REPLACEMENT_FRACS}"
    )

    root = resolve_results_root()
    print(f"\nFrozen daily results root:\n  {root}")

    outdir = root / "Sensitivity_Degradation_Replacement"
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_frozen_trajectories(root)
    full = run_full_matrix(data)

    full_path = outdir / "degradation_replacement_sensitivity_full.csv"
    full.to_csv(full_path, index=False)

    base = base_case_audit(full)
    base_path = outdir / "degradation_replacement_base_audit.csv"
    base.to_csv(base_path, index=False)

    ranges = controller_ranges(full, base)
    ranges_path = outdir / "degradation_replacement_controller_ranges.csv"
    ranges.to_csv(ranges_path, index=False)

    ranking = ranking_summary(full)
    ranking_path = outdir / "degradation_replacement_ranking_summary.csv"
    ranking.to_csv(ranking_path, index=False)

    envelope_png, envelope_pdf = plot_lcoh_envelopes(ranges, outdir)
    tornado_png, tornado_pdf = plot_rl_tornado(full, base, outdir)

    print("\n" + "="*100)
    print("FULL-FACTORIAL CONTROLLER RANGES")
    print("="*100)
    print(
        ranges[
            [
                "location", "controller",
                "lcoh_min", "lcoh_base", "lcoh_max",
                "life_min", "life_base", "life_max",
                "repl_min", "repl_base", "repl_max",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    print("\n" + "="*100)
    print("RANKING ROBUSTNESS ACROSS ALL 27 SCENARIOS PER SITE")
    print("="*100)
    print(ranking.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\nSaved:")
    for p in [
        full_path, base_path, ranges_path, ranking_path,
        envelope_png, envelope_pdf, tornado_png, tornado_pdf,
    ]:
        print(f"  {p}")

    print("\nInterpretation:")
    print("  - Use the LCOH envelope figure in the paper/Supplement: it includes all 27 combinations.")
    print("  - Use the RL tornado as a secondary diagnostic of one-at-a-time factor influence.")
    print("  - The reward and rho_cs=0.90 remain frozen; no retraining is performed.")


if __name__ == "__main__":
    main()
