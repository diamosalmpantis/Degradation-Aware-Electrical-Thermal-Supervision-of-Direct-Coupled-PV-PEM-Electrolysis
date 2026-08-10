# -*- coding: utf-8 -*-
"""
Prague 2023 temporal out-of-sample evaluation.

Protocol:
  - PPO training domain: Prague_2021
  - Evaluation domain: Prague_2023
  - Frozen model: rl_model_Prague_2021.zip
  - PPO training/adaptation on 2023: forbidden
  - Demand: daily, 78 kg/day
  - Hardware: frozen Prague sizing (1.75x PV, 5300 kg H2 tank, 550 kWh battery)

This is a wrapper around the final production physics source. It imports that
source without running its __main__ block, then reuses its baseline simulators,
PVPEMEnv, evaluate_rl(), reporting, and LCOH functions.

If the 2021 PPO model is missing, this script ABORTS. It never trains a 2023 model.
"""

from pathlib import Path
import importlib.util
import hashlib
import json
import time
import numpy as np
import pandas as pd

# ============================================================================
# USER SETTINGS
# ============================================================================
EXCEL_FILE = "/content/Irradiance_values_example_new.xlsx"
EVAL_SHEET = "Prague_2023"

# Leave as None for auto-detection.
MAIN_SOURCE_OVERRIDE = None
FROZEN_MODEL_OVERRIDE = None

OUT_ROOT = "/content/Results_1MW_UNSEEN_2023/Daily_demand"
EXPECTED_ROWS = 8760

TRAINING_ID = "Prague_2021"
SITE_ID = "Prague_2021"
EVAL_ID = "Prague_2023_FROZEN2021"

# Baselines are fast; keep True for one clean self-contained final folder.
RUN_BASELINES = True
MAKE_STANDARD_PLOTS = True


# ============================================================================
# PATHS / IMPORT
# ============================================================================
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_main_source():
    candidates = []
    if MAIN_SOURCE_OVERRIDE:
        candidates.append(Path(MAIN_SOURCE_OVERRIDE))

    for root in (Path("/content"), Path.cwd()):
        for name in (
            "New_RL_1MW_COLDSTART_EVENT_VALIDATION.py",
            "New_RL_1MW_COLDSTART_EVENT_VALIDATION(1).py",
            "New_RL_1MW_FINAL_multilocation.py",
        ):
            candidates.append(root / name)

    for p in candidates:
        if p.is_file():
            return p.resolve()

    raise FileNotFoundError(
        "Final production source not found. Upload "
        "New_RL_1MW_COLDSTART_EVENT_VALIDATION.py to /content "
        "or set MAIN_SOURCE_OVERRIDE."
    )


def resolve_frozen_model():
    candidates = []
    if FROZEN_MODEL_OVERRIDE:
        candidates.append(Path(FROZEN_MODEL_OVERRIDE))

    candidates += [
        Path("/content/Results_1MW_daily/Daily_demand/"
             "Results_Prague_2021/Model/rl_model_Prague_2021.zip"),
        Path("/content/drive/MyDrive/PVPEM_Results/Results_1MW_daily/Daily_demand/"
             "Results_Prague_2021/Model/rl_model_Prague_2021.zip"),
        Path("/content/drive/MyDrive/Results_1MW_daily/Daily_demand/"
             "Results_Prague_2021/Model/rl_model_Prague_2021.zip"),
    ]

    for p in candidates:
        if p.is_file():
            return p.resolve()

    # Exact-name fallback.
    for root in (Path("/content"), Path("/content/drive/MyDrive")):
        if not root.exists():
            continue
        try:
            hits = list(root.rglob("rl_model_Prague_2021.zip"))
        except Exception:
            hits = []
        if hits:
            hits.sort(key=lambda p: (
                "Daily_demand" not in str(p),
                "Results_Prague_2021" not in str(p),
                len(str(p)),
            ))
            return hits[0].resolve()

    raise FileNotFoundError(
        "Frozen Prague_2021 PPO model not found. This script refuses to train "
        "a replacement model. Mount/upload rl_model_Prague_2021.zip or set "
        "FROZEN_MODEL_OVERRIDE."
    )


def import_source(path):
    spec = importlib.util.spec_from_file_location("_pv_pem_final", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    required = [
        "PPO", "PVPEMEnv", "load_data_from_excel", "run_baselines",
        "evaluate_rl", "_ctrl_dfs", "_controller_headline",
        "print_controller_comparison", "audit_controller_energy_allocation",
        "audit_rl_reward_components", "calculate_lcoh",
        "project_pem_stack_replacements", "site_sizing_for",
        "set_string_ladder", "set_initial_storage",
        "plot_3way_comparison", "export_3way_excel", "plot_energy_stacks",
    ]
    missing = [x for x in required if not hasattr(m, x)]
    if missing:
        raise AttributeError(f"Production source missing: {missing}")
    return m


# ============================================================================
# FROZEN BUILD AUDIT / TRAINING LOCK
# ============================================================================
def audit_frozen_build(m):
    expected = {
        "DT_HOURS": 1.0,
        "W_H2": 0.36,
        "W_STH": 0.31,
        "W_MPPT": 0.33,
        "LAMBDA_SWITCH": 0.13,
        "W_UNDERUSE": 0.10,
        "W_KEEPWARM": 0.10,
        "W_CHILLER": 0.10,
        "W_TCYCLE": 0.04,
        "W_COLDSTART": 0.80,
        "W_WARMOPS": 0.08,
        "COLDSTART_THRESHOLD_C": 50.0,
        "LOOKAHEAD_STEPS": 4,
        "LCOH_PEM_EOL_SOH": 0.80,
        "LCOH_PEM_REPLACEMENT_FRAC": 1.00,
    }

    failures = []
    for name, exp in expected.items():
        if not hasattr(m, name):
            failures.append(f"{name}: missing")
            continue
        got = getattr(m, name)
        try:
            ok = np.isclose(float(got), float(exp), atol=1e-12, rtol=0)
        except Exception:
            ok = got == exp
        if not ok:
            failures.append(f"{name}: expected {exp}, got {got}")

    if tuple(m.KEEPWARM_LEVELS_C) != (30.0, 45.0, 60.0):
        failures.append(f"KEEPWARM_LEVELS_C={m.KEEPWARM_LEVELS_C}")
    if tuple(m.CHILLER_LEVELS_C) != (60.0, 65.0, 70.0):
        failures.append(f"CHILLER_LEVELS_C={m.CHILLER_LEVELS_C}")

    daily = float(m.hourly_H2_target) * 24.0
    if not np.isclose(daily, 78.0, atol=1e-9):
        failures.append(f"daily demand expected 78 kg/day, got {daily}")

    if failures:
        raise RuntimeError(
            "Frozen source audit failed:\n  - " + "\n  - ".join(failures)
        )


def forbid_training(m):
    def blocked(*args, **kwargs):
        raise RuntimeError(
            "TRAINING IS FORBIDDEN in the unseen-year evaluation. "
            "Use rl_model_Prague_2021.zip and model.predict() only."
        )
    m.train_rl = blocked


# ============================================================================
# INPUTS / HARDWARE
# ============================================================================
def load_2023_inputs(m):
    xlsx = Path(EXCEL_FILE)
    if not xlsx.is_file():
        raise FileNotFoundError(f"Workbook not found: {xlsx}")

    xls = pd.ExcelFile(xlsx)
    if EVAL_SHEET not in xls.sheet_names:
        raise KeyError(
            f"Missing sheet {EVAL_SHEET}. Available: {xls.sheet_names}"
        )

    raw = pd.read_excel(xlsx, sheet_name=EVAL_SHEET)
    if len(raw) != EXPECTED_ROWS:
        raise ValueError(
            f"{EVAL_SHEET} has {len(raw)} rows; expected {EXPECTED_ROWS}."
        )

    cols = list(raw.columns)
    irr_cols = [c for c in cols if "irradiance" in str(c).lower()]
    temp_cols = [c for c in cols if "temperature" in str(c).lower()]
    price_cols = [c for c in cols if "price" in str(c).lower()]

    if not irr_cols:
        raise KeyError("No Irradiance column.")
    if not temp_cols:
        raise KeyError("No Temperature column.")
    if not price_cols:
        raise KeyError(
            "No Price column. Actual 2023 price data are required; "
            "the production loader's fallback price is not allowed."
        )

    for c in (irr_cols[0], temp_cols[0], price_cols[0]):
        x = pd.to_numeric(raw[c], errors="coerce")
        if x.isna().any():
            raise ValueError(
                f"Column {c!r} has {int(x.isna().sum())} missing/non-numeric rows."
            )

    data = m.load_data_from_excel(str(xlsx), EVAL_SHEET)
    if len(data) != EXPECTED_ROWS:
        raise ValueError(
            f"Production loader returned {len(data)} rows; expected {EXPECTED_ROWS}."
        )

    irr = np.asarray([d["irradiance"] for d in data], dtype=float)
    temp = np.asarray([d["temperature"] for d in data], dtype=float)
    price = np.asarray([d["price_mwh"] for d in data], dtype=float)

    if not (np.isfinite(irr).all() and np.isfinite(temp).all()
            and np.isfinite(price).all()):
        raise ValueError("Input arrays contain NaN/inf.")

    return raw, irr, temp, price, irr_cols[0], temp_cols[0], price_cols[0]


def apply_prague_hardware(m):
    sizing = m.site_sizing_for(SITE_ID)
    if sizing is None:
        raise RuntimeError("Prague sizing not found.")

    pv_ratio, tank_kg, batt_kwh = map(float, sizing)

    if not (
        np.isclose(pv_ratio, 1.75)
        and np.isclose(tank_kg, 5300.0)
        and np.isclose(batt_kwh, 550.0)
    ):
        raise RuntimeError(
            f"Expected Prague sizing (1.75, 5300, 550), got {sizing}"
        )

    m.set_string_ladder(pv_ratio)
    m.storage_capacity = tank_kg
    m.LCOH_TANK_KG_INSTALLED = tank_kg
    m.BATT_CAPACITY_WH = batt_kwh * 1000.0
    m.BATT_MAX_P_W = batt_kwh * 1000.0
    m.set_initial_storage()

    pv_peak_kw = (
        m.N_PAR_MAX * m.N_SERIES_PV * m.PV_MODULE_W / 1000.0
    )

    return {
        "pv_ratio": pv_ratio,
        "tank_kg": tank_kg,
        "battery_kwh": batt_kwh,
        "n_par_base": int(m.N_PAR_BASE),
        "n_par_step": int(m.N_PAR_STEP),
        "n_par_max": int(m.N_PAR_MAX),
        "pv_peak_kw": float(pv_peak_kw),
        "tank_start_kg": float(m.INITIAL_H2_STORAGE),
    }


# ============================================================================
# PPO LOAD / COMPATIBILITY
# ============================================================================
def load_frozen_ppo(m, model_path, irr, temp, price, pv_ratio):
    print("\nLoading frozen 2021 PPO:")
    print(f"  {model_path}")

    model = m.PPO.load(str(model_path))

    # One-step compatibility check before the annual evaluation.
    env = m.PVPEMEnv(
        irr, temp, price, eval_mode=True, pv_scale=pv_ratio
    )
    obs, _ = env.reset()

    print("\nCompatibility audit")
    print(f"  model obs space : {model.observation_space}")
    print(f"  eval obs space  : {env.observation_space}")
    print(f"  model act space : {model.action_space}")
    print(f"  eval act space  : {env.action_space}")
    print(f"  obs shape       : {np.asarray(obs).shape}")

    if getattr(model.observation_space, "shape", None) != getattr(
        env.observation_space, "shape", None
    ):
        raise RuntimeError("Observation-space mismatch.")

    mn = getattr(model.action_space, "nvec", None)
    en = getattr(env.action_space, "nvec", None)
    if mn is not None or en is not None:
        if mn is None or en is None or not np.array_equal(mn, en):
            raise RuntimeError(
                f"Action-space mismatch: model {mn}, eval {en}"
            )

    action, _ = model.predict(obs, deterministic=True)
    print(f"  first deterministic action: {action}")

    return model


# ============================================================================
# FIVE CONTROLLERS
# ============================================================================
def run_controllers(m, model, irr, temp, price, data_dir, plots_dir, pv_ratio):
    if RUN_BASELINES:
        print("\nRunning frozen deterministic baselines on Prague_2023...")

        df_moo = m.run_baselines(
            irr, temp, price, use_rf=False, hysteresis=False,
            lambda_sw=0.0, pv_scale=pv_ratio, prewarm=None,
        )
        df_moo.to_csv(
            data_dir / "moo_results_Prague_2023.csv", index=False
        )

        df_hyst = m.run_baselines(
            irr, temp, price, use_rf=False, hysteresis=True,
            lambda_sw=0.0, pv_scale=pv_ratio, prewarm=None,
        )
        df_hyst.to_csv(
            data_dir / "hyst_results_Prague_2023.csv", index=False
        )

        df_pw = m.run_baselines(
            irr, temp, price, use_rf=False, hysteresis=False,
            lambda_sw=0.0, pv_scale=pv_ratio, prewarm=m.PREWARM_RULE,
        )
        df_pw.to_csv(
            data_dir / "prewarm_results_Prague_2023.csv", index=False
        )

        df_hpw = m.run_baselines(
            irr, temp, price, use_rf=False, hysteresis=True,
            lambda_sw=0.0, pv_scale=pv_ratio, prewarm=m.PREWARM_RULE,
        )
        df_hpw.to_csv(
            data_dir / "hystprewarm_results_Prague_2023.csv", index=False
        )
    else:
        names = {
            "moo": data_dir / "moo_results_Prague_2023.csv",
            "hyst": data_dir / "hyst_results_Prague_2023.csv",
            "pw": data_dir / "prewarm_results_Prague_2023.csv",
            "hpw": data_dir / "hystprewarm_results_Prague_2023.csv",
        }
        missing = [str(p) for p in names.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "RUN_BASELINES=False but missing:\n" + "\n".join(missing)
            )
        df_moo = pd.read_csv(names["moo"])
        df_hyst = pd.read_csv(names["hyst"])
        df_pw = pd.read_csv(names["pw"])
        df_hpw = pd.read_csv(names["hpw"])

    print("\n" + "=" * 92)
    print("FROZEN PPO OUT-OF-SAMPLE EVALUATION")
    print("  trained on       : Prague_2021")
    print("  evaluated on     : Prague_2023")
    print("  2023 PPO training: DISABLED")
    print("=" * 92)

    # This function only uses deterministic model.predict() during evaluation.
    df_rl = m.evaluate_rl(
        model, irr, temp, price, EVAL_ID,
        data_dir, plots_dir, pv_scale=pv_ratio
    )
    df_rl.to_csv(
        data_dir / "rl_results_Prague_2023_FROZEN2021.csv",
        index=False,
    )

    return df_moo, df_hyst, df_pw, df_hpw, df_rl


# ============================================================================
# REPORTS / AUDIT
# ============================================================================
def report(m, dfs, raw, cols, hardware, source_path, model_path,
           data_dir, plots_dir, audit_dir):
    df_moo, df_hyst, df_pw, df_hpw, df_rl = dfs

    kw = dict(
        df_hyst=df_hyst,
        df_pen=None,
        df_pw=df_pw,
        df_hpw=df_hpw,
    )

    ctrls = m._ctrl_dfs(
        df_moo, None, df_rl, df_hyst, None, df_pw, df_hpw
    )

    headline = m.print_controller_comparison(ctrls, EVAL_ID)
    m.audit_controller_energy_allocation(ctrls, EVAL_ID)
    m.audit_rl_reward_components(df_rl, EVAL_ID)

    lcoh = m.calculate_lcoh(
        df_moo, None, df_rl, hardware["pv_peak_kw"], **kw
    )

    if MAKE_STANDARD_PLOTS:
        m.plot_3way_comparison(
            df_moo, None, df_rl, EVAL_ID, plots_dir,
            pv_scale=hardware["pv_ratio"], **kw
        )
        m.export_3way_excel(
            df_moo, None, df_rl, EVAL_ID, data_dir, **kw
        )
        m.plot_energy_stacks(
            df_moo, None, df_rl, EVAL_ID, plots_dir, **kw
        )

    # Lifetime/replacement projection from each actual 2023 trajectory.
    stack_capex = (
        float(m.PEM_capacity) / 1000.0 * float(m.LCOH_PEM_EUR_KW)
    )

    rows = []
    for ctrl, df in ctrls.items():
        h = headline[ctrl]
        e = lcoh[ctrl]
        rep = m.project_pem_stack_replacements(
            df, stack_capex=stack_capex
        )

        rows.append({
            "controller": ctrl,
            "training_year": 2021 if ctrl == "RL Agent" else np.nan,
            "evaluation_year": 2023,
            "policy_status": (
                "Frozen Prague_2021 PPO; no adaptation"
                if ctrl == "RL Agent"
                else "Deterministic baseline"
            ),
            "h2_prod_kgyr": h["h2_prod"],
            "h2_delivered_kgyr": h["h2_del"],
            "unmet_h2_kgyr": h["unmet"],
            "demand_met_pct": h["dmet"],
            "switches_yr": h["sw"],
            "cold_starts_yr": h["cold"],
            "mean_restart_T_C": h["restartT"],
            "pem_soh_end_pct": h["soh_pem"],
            "pv_soh_end_pct": h["soh_pv"],
            "battery_soh_end_pct": h["soh_batt"],
            "battery_cycles_yr": h["batt_cyc"],
            "mean_sth_pct": h["sth"],
            "mpp_shortfall_kwhyr": h["mppt_kwh"],
            "curtailed_pv_kwhyr": h["curt_kwh"],
            "grid_import_mwhyr": h["grid_mwh"],
            "operating_profit_keuryr": h["profit_k"],
            "projected_stack_life_yr": rep["projected_stack_life_yr"],
            "stack_replacements_30y": rep["stack_replacements"],
            "lcoh_eurkg": e["lcoh"],
            "lcsh_eurkg": e["lcsh"],
        })

    summary = pd.DataFrame(rows)
    summary_path = data_dir / "UNSEEN_2023_FROZEN2021_summary.csv"
    summary.to_csv(summary_path, index=False)

    irr_col, temp_col, price_col = cols
    irr = pd.to_numeric(raw[irr_col], errors="coerce")
    temp = pd.to_numeric(raw[temp_col], errors="coerce")
    price = pd.to_numeric(raw[price_col], errors="coerce")

    manifest = {
        "experiment": "Temporal out-of-sample evaluation",
        "training_domain": TRAINING_ID,
        "evaluation_domain": "Prague_2023",
        "rl_training_on_2023": False,
        "rl_adaptation_on_2023": False,
        "deterministic_prediction": True,
        "demand_mode": "daily",
        "daily_h2_contract_kg": float(m.hourly_H2_target) * 24.0,
        "input_workbook": str(Path(EXCEL_FILE).resolve()),
        "input_sheet": EVAL_SHEET,
        "input_rows": len(raw),
        "annual_ghi_kwh_m2": float(irr.sum() / 1000.0),
        "temperature_min_C": float(temp.min()),
        "temperature_max_C": float(temp.max()),
        "price_min_eur_mwh": float(price.min()),
        "price_max_eur_mwh": float(price.max()),
        "price_mean_eur_mwh": float(price.mean()),
        "hardware": hardware,
        "production_source": str(source_path),
        "production_source_sha256": sha256(source_path),
        "frozen_model": str(model_path),
        "frozen_model_sha256": sha256(model_path),
        "reward": {
            "W_H2": m.W_H2,
            "W_STH": m.W_STH,
            "W_MPPT": m.W_MPPT,
            "LAMBDA_SWITCH": m.LAMBDA_SWITCH,
            "W_UNDERUSE": m.W_UNDERUSE,
            "W_KEEPWARM": m.W_KEEPWARM,
            "W_CHILLER": m.W_CHILLER,
            "W_TCYCLE": m.W_TCYCLE,
            "W_COLDSTART": m.W_COLDSTART,
            "W_WARMOPS": m.W_WARMOPS,
        },
    }

    (audit_dir / "UNSEEN_2023_FROZEN2021_AUDIT.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    audit_text = f"""PRAGUE 2023 TEMPORAL OUT-OF-SAMPLE AUDIT
============================================================
Training domain          : Prague_2021
Evaluation domain        : Prague_2023
RL training on 2023      : DISABLED
RL adaptation on 2023    : NONE
Prediction               : deterministic=True
Frozen model             : {model_path}
Frozen model SHA256      : {sha256(model_path)}
Production source        : {source_path}
Production source SHA256 : {sha256(source_path)}
Workbook                 : {Path(EXCEL_FILE).resolve()}
Sheet                    : {EVAL_SHEET}
Rows                     : {len(raw)}
Annual GHI               : {irr.sum()/1000.0:.2f} kWh/m2
Temperature range        : {temp.min():.2f} to {temp.max():.2f} C
Price range              : {price.min():.2f} to {price.max():.2f} EUR/MWh
Daily H2 contract        : {float(m.hourly_H2_target)*24.0:.2f} kg/day
Prague PV ratio          : {hardware['pv_ratio']:.2f}x
Installed strings        : {hardware['n_par_max']}
H2 tank                  : {hardware['tank_kg']:.0f} kg
Battery                  : {hardware['battery_kwh']:.0f} kWh

NO train_rl() call is permitted by this wrapper.
If the 2021 model is absent, execution aborts instead of training.
============================================================
"""
    audit_path = audit_dir / "UNSEEN_2023_FROZEN2021_AUDIT.txt"
    audit_path.write_text(audit_text, encoding="utf-8")

    print("\n" + "=" * 120)
    print("PAPER-READY METRICS")
    print("=" * 120)
    paper_cols = [
        "controller", "h2_prod_kgyr", "switches_yr", "cold_starts_yr",
        "mean_restart_T_C", "pem_soh_end_pct", "battery_cycles_yr",
        "grid_import_mwhyr", "demand_met_pct", "projected_stack_life_yr",
        "stack_replacements_30y", "lcoh_eurkg", "lcsh_eurkg",
    ]
    print(summary[paper_cols].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))
    print("=" * 120)

    print(f"\nSummary CSV: {summary_path}")
    print(f"Audit TXT  : {audit_path}")

    return summary


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()

    print("\n" + "#" * 100)
    print("PRAGUE 2023 UNSEEN-YEAR TEST -- FROZEN PRAGUE 2021 PPO")
    print("#" * 100)
    print("TRAINING YEAR : 2021")
    print("EVALUATION    : 2023")
    print("RL TRAINING   : FORBIDDEN")
    print("MODEL         : rl_model_Prague_2021.zip")
    print("DEMAND        : daily 78 kg/day")
    print("#" * 100)

    source_path = resolve_main_source()
    model_path = resolve_frozen_model()

    print("\nResolved:")
    print(f"  source : {source_path}")
    print(f"  model  : {model_path}")
    print(f"  input  : {EXCEL_FILE} / {EVAL_SHEET}")

    # Import does not execute the original script's __main__ training pipeline.
    m = import_source(source_path)

    audit_frozen_build(m)
    forbid_training(m)

    # Lock study mode.
    m.DEMAND_MODE = "daily"
    m.REUSE_SAVED_MODELS = True
    m.RUN_SEED_STUDY = False
    m.RUN_PARETO_STUDY = False
    m.RUN_REWARD_STUDY = False
    m.RUN_PV_SWEEP = False
    m.RUN_BATTERY_SENSITIVITY = False
    m.RUN_TANK_SENSITIVITY = False
    m.RUN_TORNADO = False
    m.INCLUDE_RF_MOO = False
    m.INCLUDE_PENALISED_MOO = False
    m.INCLUDE_HYSTERESIS_MOO = True
    m.INCLUDE_PREWARM_MOO = True
    m.INCLUDE_HYST_PREWARM_MOO = True
    m.PREWARM_RULE = "anticipatory"

    raw, irr, temp, price, irr_col, temp_col, price_col = load_2023_inputs(m)
    hardware = apply_prague_hardware(m)

    print("\nFrozen Prague hardware")
    print(
        f"  PV {hardware['pv_ratio']:.2f}x -> "
        f"{hardware['n_par_max']} strings "
        f"({hardware['n_par_base']}-{hardware['n_par_max']} "
        f"step {hardware['n_par_step']})"
    )
    print(f"  PV peak : {hardware['pv_peak_kw']:.1f} kWp")
    print(f"  tank    : {hardware['tank_kg']:.0f} kg")
    print(f"  battery : {hardware['battery_kwh']:.0f} kWh")

    run_dir = Path(OUT_ROOT) / f"Results_{EVAL_ID}"
    data_dir = run_dir / "Data"
    plots_dir = run_dir / "Plots"
    audit_dir = run_dir / "Audit"
    for d in (data_dir, plots_dir, audit_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = load_frozen_ppo(
        m, model_path, irr, temp, price, hardware["pv_ratio"]
    )

    dfs = run_controllers(
        m, model, irr, temp, price, data_dir, plots_dir,
        hardware["pv_ratio"]
    )

    report(
        m, dfs, raw, (irr_col, temp_col, price_col), hardware,
        source_path, model_path, data_dir, plots_dir, audit_dir
    )

    print("\n" + "#" * 100)
    print("UNSEEN-YEAR EVALUATION COMPLETE")
    print(f"Output: {run_dir}")
    print(f"Runtime: {(time.time()-t0)/60.0:.1f} min")
    print("VALIDITY: Prague_2021 PPO weights were loaded and never updated.")
    print("No Prague_2023 PPO model is trained or saved.")
    print("#" * 100)


if __name__ == "__main__":
    main()
