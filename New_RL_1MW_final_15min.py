# -*- coding: utf-8 -*-
"""Degradation-aware control of a 1 MW direct-coupled PV-PEM plant, 15-minute resolution.

Six-controller comparison on one plant: greedy MOO, MOO with a switching deadband, MOO with a pre-warm
rule, MOO with both, MOO with constant warm standby, and a PPO policy.

The reward settings are those frozen for the hourly study, so the two
resolutions are directly comparable.

Results saved to: <script_dir>/Results/Results_<SHEET_NAME>/
"""

# =============================================================================
# %% INSTALL DEPENDENCIES (Google Colab only -- skip if running locally)
# =============================================================================
# !pip install stable-baselines3 gymnasium scikit-learn scipy tqdm openpyxl -q

# =============================================================================
# %% USER CONFIGURATION -- Edit these before running
# =============================================================================
import os as _os
import sys as _sys

_in_colab = 'google.colab' in _sys.modules
try:
    _SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = '/content'  # Google Colab

_15MIN_EXCEL_CANDIDATES = [
    _os.path.join(_SCRIPT_DIR, 'Irradiance_values_example_new_15min.xlsx'),
    _os.path.join(_SCRIPT_DIR, 'Irradiance_values_15min.xlsx'),
    _os.path.join(_SCRIPT_DIR, 'Irradiance_values_example_new.xlsx'),
]
EXCEL_FILE = next((p for p in _15MIN_EXCEL_CANDIDATES if _os.path.isfile(p)),
                  _15MIN_EXCEL_CANDIDATES[0])
SHEET_NAME   = 'Prague_2021'
DAY_NUMBER   = 'all'            # 'all' or an integer day number

# --- PHYSICAL TIME GRID: FINAL 15-MINUTE RESOLUTION ------------------------------
DT_HOURS       = 0.25
RESOLUTION_INPUT_MODE = 'raw_15min'

STEPS_PER_HOUR = int(round(1.0 / DT_HOURS))
STEPS_PER_DAY  = int(round(24.0 / DT_HOURS))
STEPS_PER_YEAR = int(round(8760.0 / DT_HOURS))
if (STEPS_PER_HOUR, STEPS_PER_DAY, STEPS_PER_YEAR) != (4, 96, 35040):
    raise RuntimeError('15-min time-grid constants are inconsistent.')

RL_TIMESTEPS = int(STEPS_PER_YEAR * 60)   # 60 PHYSICAL years = 2,102,400 control steps

REUSE_SAVED_MODELS = False

# --- MAIN-PIPELINE TRAINING SEED -------------------------------------------------
RL_SEED = 1

# --- SEED-VARIANCE STUDY ---
RUN_SEED_STUDY =False
SEED_LIST      = (0,1,2,3,4,5,6)

# --- PARETO FRONT (switching vs yield/profit) ---
RUN_PARETO_STUDY = False
PARETO_LAMBDAS   = (0.12, 0.13, 0.14, 0.15)
PARETO_SEEDS     = (0, 1, 2)                        # median over seeds de-noises the switching axis
PARETO_ZERO_UNDERUSE = False

# --- FINAL REWARD SCALARS ---------------------------------------------------------
RUN_REWARD_STUDY = False

REWARD_CORNERS = [
    (0.12, 0.10, 0.80, 0.04, 0.10, 0.08, 'lam12'),
    (0.13, 0.10, 0.80, 0.04, 0.10, 0.08, 'lam13'),
    (0.14, 0.10, 0.80, 0.04, 0.10, 0.08, 'lam14'),
    (0.15, 0.10, 0.80, 0.04, 0.10, 0.08, 'lam15'),
]
REWARD_SEEDS       = (0, 1, 2)
REWARD_SINGLE_SEED = set()
REWARD_STUDY_TIMESTEPS = int(STEPS_PER_YEAR * 60)
REWARD_STUDY_MIN_YEARS = 20
if RUN_REWARD_STUDY and REWARD_STUDY_TIMESTEPS < int(STEPS_PER_YEAR * REWARD_STUDY_MIN_YEARS):
    raise SystemExit(
        "REWARD_STUDY_TIMESTEPS = %d (%.0f simulated years) is below the %d-year floor."
        % (REWARD_STUDY_TIMESTEPS, REWARD_STUDY_TIMESTEPS / STEPS_PER_YEAR,
           REWARD_STUDY_MIN_YEARS)
    )

REWARD_STUDY_PPO = {}

# --- HYSTERESIS / DEADBAND MOO BASELINE ---
MOO_HYSTERESIS = False
MOO_DEADBAND   = 2

# --- SWITCHING-PENALISED MOO BASELINE --------------------------------------------
MOO_LAMBDA_SW = 0.0

# =========================== THE SIX CONTROLLERS =================================
#   -------------------+----------------------+----------------------
INCLUDE_PENALISED_MOO = False
INCLUDE_ALWAYSWARM_MOO = True

MOO_PEN_LAMBDA = 0.02

# --- RULE-BASED PRE-WARM MOO BASELINE (5th controller) ---------------------------
INCLUDE_PREWARM_MOO = True
PREWARM_RULE = 'anticipatory'


if PREWARM_RULE == 'distilled':
    raise SystemExit(
        "PREWARM_RULE='distilled' is not valid for the 15-min sensitivity without refitting. "
        "Use PREWARM_RULE='anticipatory'."
    )

INCLUDE_HYST_PREWARM_MOO = True

INCLUDE_HYSTERESIS_MOO = True
INCLUDE_RF_MOO = False

# --- SLOW ANALYSIS-ONLY STUDIES (off during iteration, on for final figures) -----
RUN_BATTERY_SENSITIVITY = False   # covered by the PV x battery sizing study
RUN_TANK_SENSITIVITY    = False   # covered by the PV x tank sizing study
RUN_TORNADO             = True
RUN_PV_SWEEP            = False   # sizing done; SITE_SIZING below holds the result
USE_SITE_SIZING = True
SITE_SIZING = {
    'Lund_2021':    (1.75, 7800, 550),
    'Prague_2021':  (1.75, 5300, 550),
    'Odeillo_2021': (1.25, 1600, 450),
}

def site_sizing_for(label):
    """Per-site (pv_ratio, tank_kg, batt_kWh), matched on the SITE name."""
    if label in SITE_SIZING:
        return SITE_SIZING[label]
    site = label.split('_')[0]
    for k, v in SITE_SIZING.items():
        if k.split('_')[0] == site:
            return v
    return None

PV_MODULE_W    = 412.5
PV_MIN_FRAC    = 0.35


def site_string_config(ratio, pem_w=None, n_series=None, module_w=PV_MODULE_W,
                       n_positions=7, min_frac=PV_MIN_FRAC):
    """Per-site MOSFET string ladder for a PV oversizing `ratio`."""
    pem_w    = PEM_capacity if pem_w   is None else pem_w
    n_series = N_SERIES_PV  if n_series is None else n_series
    n_max = int(round(ratio * pem_w / (n_series * module_w)))
    step  = max(1, int(round(n_max * (1.0 - min_frac) / (n_positions - 1))))
    return n_max - step * (n_positions - 1), step, n_max


def set_string_ladder(ratio):
    """Point the module-level MOSFET ladder at a PV oversizing `ratio` (installed strings
    = ratio * PEM / string_W).
    """
    global N_PAR_BASE, N_PAR_STEP, N_PAR_MAX
    prev = (N_PAR_BASE, N_PAR_STEP, N_PAR_MAX)
    N_PAR_BASE, N_PAR_STEP, N_PAR_MAX = site_string_config(ratio)
    return prev

OUT_DIR      = _os.path.join(_SCRIPT_DIR, 'Results_1MW_15min')   # matches where the saved models/results live
_DRIVE_ROOT = '/content/drive/MyDrive'
if _in_colab and _os.path.isdir(_DRIVE_ROOT):
    OUT_DIR = _os.path.join(_DRIVE_ROOT, 'PVPEM_Results', 'Results_1MW_15min')
    _os.makedirs(OUT_DIR, exist_ok=True)
    print("  [Colab] Drive mounted -> results written to %s" % OUT_DIR)
    print("          (survives runtime disconnect; nothing to download at the end)")
elif _in_colab:
    print("  [Colab] Drive NOT mounted -> results go to /content and are LOST when the")
    print("          runtime recycles. Mount Drive first if you plan to leave this running.")

# --- H2 DEMAND CONTRACT MODE ---
DEMAND_MODE = 'daily'

DEMAND_TARGET_PCT = 98.0
BATT_NIGHT_KWH = {'Lund_2021': 550, 'Prague_2021': 550, 'Odeillo_2021': 450}
TANK_ROUND_KG      = 100
TANK_VERIFY_STEP   = 200
TANK_VERIFY_STEPS  = 8
SELF_SUFF_TARGET_PCT = 87.0

# =============================================================================
# %% IMPORTS
# =============================================================================
import sys
import time
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
_in_colab = 'google.colab' in sys.modules
_in_interactive = _spyder_running or _in_colab
if not _in_interactive:
    try:
        matplotlib.use('Agg')
    except Exception:
        pass
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
_os.environ['PYTHONWARNINGS'] = 'ignore'

plt.rcParams.update({
    'axes.labelsize': 15, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'axes.titlesize': 14, 'legend.fontsize': 12,
})


TORCH_THREADS = 8
if TORCH_THREADS:
    import torch as _torch
    _torch.set_num_threads(int(TORCH_THREADS))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial import KDTree

print("Imports OK.")

# =============================================================================
# %% COLUMN NAMES
# =============================================================================
IRRADIANCE_COL  = 'Irradiance_France'
TEMPERATURE_COL = 'Temperature_France'
PRICE_COL       = 'Price_France (EUR/MWh, EUR/tCO2)'


RL_CONFIG = dict(
    n_steps         = 2048 * STEPS_PER_HOUR,
    batch_size      = 256  * STEPS_PER_HOUR,
    n_epochs        = 10,
    gamma           = 0.99 ** DT_HOURS,
    gae_lambda      = 0.95 ** DT_HOURS,
    learning_rate   = 3e-4,
    ent_coef        = 0.005,
    clip_range      = 0.2,
    max_grad_norm   = 0.5,
    checkpoint_freq = STEPS_PER_YEAR,
    net_arch        = [256, 256],
)

# =============================================================================
# %% PHYSICAL & ECONOMIC CONSTANTS
# =============================================================================
k, q, Ns = 1.38e-23, 1.6e-19, 66   # Ns=66 series cells/module (HJT 132 half-cell format)
Iph_ref, Rs, Rsh = 8.1, 0.5, 500   # Iph_ref=8.1 (1MW PV module)
Is_ref1, Is_ref2, Is_refi = 1e-14, 1e-14, 1e-13
n1, n2, ni, Eg = 1.04, 2, 1.5, 1.12
T_ref = 25 + 273.15
rho_contact, A_cell, aSi_thickness, TCO_resistivity = 5e-3, 243, 5e-9, 3e-4
pv_module_area = 1.925
N_SERIES_PV = 5

tilt_angle_deg, sun_zenith_deg = 30, 45

delta_mem, A_cell_pem = 0.0054, 1250   # 1250 cm^2 cell (1MW)
a_an, a_cat, i_0_an, i_0_cat = 0.75, 0.4, 2.44e-4, 2.28e-2
T_ref_pem, i_lim, p_cat, p_an = 298.15, 6.1, 40, 1.0   # p_cat=40 bar (1MW)
i_cell = A_cell_pem * 3

N_SER_PEM = 134

# --- Arrhenius temperature dependence of the exchange current densities -------
PEM_I0_ARRHENIUS = True
E_ACT_AN, E_ACT_CAT = 29.5e3, 9.9e3   # J/mol — measured, see note above

LHV = 33.3 * 1000                 # Wh/kg — lower heating value (the reporting basis)
HHV = 39.4 * 1000                 # Wh/kg — higher heating value

# --- H2 PRODUCTION: FARADAY'S LAW, NOT AN EFFICIENCY CURVE -----------------------
F_FARADAY   = 96485.0     # C/mol
M_H2_KG_MOL = 2.016e-3    # kg/mol
ETA_FARADAY = 0.98        # Faradaic efficiency (H2 crossover + shunt currents)
V_TH_HHV    = 1.481


def h2_rate_faraday(I_stack_A, P_pem_W):
    """Hydrogen production rate [kg/h] from the stack operating current."""
    if P_pem_W <= 20 or I_stack_A <= 0:
        return 0.0
    rate = N_SER_PEM * float(I_stack_A) * M_H2_KG_MOL / (2.0 * F_FARADAY) * ETA_FARADAY * 3600.0
    return float(np.clip(rate, 0.0, H2_PEAK_KGH))

storage_capacity = 4320
H2_DEMAND_KG_PER_H = 3.25
hourly_H2_target  = H2_DEMAND_KG_PER_H             # kg per HOURLY settlement
step_H2_target    = H2_DEMAND_KG_PER_H * DT_HOURS  # 0.8125 kg per 15-min direct path
daily_H2_target   = H2_DEMAND_KG_PER_H * 24.0      # 78 kg per DAILY settlement
PEM_capacity     = 1000 * 1000   # 1 MW PEM
H2_PEAK_KGH      = 19.0
MPPT_NORM        = 32000.0  # normalisation (W) for the MPP-shortfall reward term.


def mppt_shortfall(P_mpp, P_op, cap=None):
    """Control-attributable MPP shortfall (W)."""
    cap = PEM_capacity if cap is None else cap
    return max(0.0, min(float(P_mpp), float(cap)) - float(P_op))

# --- Active MOSFET string ladder (set per site by the location loop) -------------
N_PAR_BASE, N_PAR_STEP, N_PAR_MAX = 250, 25, 400

# --- RL reward: MOO objective weights (identical to run_baselines) + fatigue term ---
W_H2, W_STH, W_MPPT = 0.36, 0.31, 0.33
W_UNDERUSE          = 0.10
LAMBDA_SWITCH       = 0.13    # FINAL switching-fatigue weight.

if MOO_PEN_LAMBDA is None:
    MOO_PEN_LAMBDA = LAMBDA_SWITCH

# --- THERMAL SETPOINT ACTION (novel extension) -----------------------------------
THERMAL_ACTION     = True

# --- FORECAST LOOKAHEAD ---
LOOKAHEAD_HOURS       = 4.0
FORECAST_HORIZONS_H   = (1.0, 2.0, 3.0, 4.0)
FORECAST_OFFSETS_STEPS = tuple(int(round(h / DT_HOURS)) for h in FORECAST_HORIZONS_H)
LOOKAHEAD_STEPS       = len(FORECAST_OFFSETS_STEPS)  # 4 forecast points; obs stays 26-D
if FORECAST_OFFSETS_STEPS != (4, 8, 12, 16):
    raise RuntimeError("15-min forecast offsets are inconsistent with +1,+2,+3,+4 h.")

# --- CHILLER SET-POINT ACTION ----------------------------------------------------
CHILLER_ACTION     = True
CHILLER_LEVELS_C   = (60.0, 65.0, 70.0)   # selectable cooling set-points; 70 C = datasheet

# --- THERMAL (COLD-OPERATION) DEGRADATION ----------------------------------------
COLDSTART_REWARD    = True
W_COLDSTART         = 0.80

THERMAL_DEGRADATION = True
_V_DEG_REF          = None
T_DEG_REF_K         = 343.15
KAPPA_THERM         = 1.0

# --- COLD-START EXPERIMENT -------------------------------------------------------
COLDSTART_THRESHOLD_C = 50.0

# =============================================================================
# THREE-MECHANISM PEM DEGRADATION MODEL
# =============================================================================
# M1  STEADY OPERATION -- Elizetxea-Navarro et al. (2026), Eqs. (7) and (8)
# -----------------------------------------------------------------------------
PEM_J_NOM_A_CM2   = 3.0       # A/cm^2, datasheet nominal (= i_cell / A_cell_pem)
MU_DYN_SLOPE      = 0.2499    # uV/h per A/cm^2
MU_DYN_INTERCEPT  = 2.3545    # uV/h at zero current density

MU_EVOL_SLOPE     = 5.0e-5    # uV/h per operating hour
MU_EVOL_INTERCEPT = 0.3366    # uV/h when new
MU_EVOL_MEAN      = 2.3366    # uV/h at t_op = 40,000 h = lifetime mean
PEM_LIFE_H_REF    = 80000.0   # h, reference life the ageing law is fitted over
AGEING_MODE       = 'lifetime_mean'   # 'lifetime_mean' | 'explicit' | 'off'

MU_RAMP_FACTOR    = 1.83      # x mu_dyn at |d(fl)| = 1 per hour

PEM_IDLE_FL_MIN   = 0.05
MU_IDLE_FRAC      = 0.0       # fraction of mu_dyn(0) charged while idle

# -----------------------------------------------------------------------------
# M2  THERMAL -- Chandesris et al., as reported by Maoulida et al. (2026)
# -----------------------------------------------------------------------------
THERMAL_LEVEL_DEGRADATION = True
T_DOUBLE_K        = 10.0      # K per doubling of the steady degradation rate
F_T_MAX           = 8.0       # clamp; never reached inside the 80 C ceiling

THERMAL_CYCLING_DEGRADATION = True
MU_TCYC_UV_PER_K  = 0.25      # uV per K of EXCESS per-step stack temperature swing
TCYC_DEADBAND_K   = 5.0       # swing a working cooling loop absorbs for free

# -----------------------------------------------------------------------------
# M3  ON/OFF -- Elizetxea-Navarro et al. Eq. (5)/(9), generalised
# -----------------------------------------------------------------------------
MU_SD_COLD_UV     = 100.0     # uV/event, their adopted shutdown coefficient
MU_SD_WARM_RATIO  = 0.10      # warm restart as a fraction of a cold one
MU_SD_WARM_UV     = MU_SD_COLD_UV * MU_SD_WARM_RATIO

# -----------------------------------------------------------------------------

# --- COLD-START SoH CHARGE --------------------------------
COLDSTART_SOH_COLD = 5e-5   # restart below COLDSTART_THRESHOLD_C
COLDSTART_SOH_WARM = 5e-6   # restart at or above COLDSTART_THRESHOLD_C
COLDSTART_EXCESS_DAMAGE_FRAC = max(
    0.0,
    (COLDSTART_SOH_COLD - COLDSTART_SOH_WARM) / max(COLDSTART_SOH_COLD, 1e-12)
)  # = 0.90 with the calibrated SoH charges
KEEPWARM_LEVELS_C  = (30.0, 45.0, 60.0)   # selectable warm-standby set-points.
KEEPWARM_BASELINE_C = KEEPWARM_LEVELS_C[0]
# -----------------------------------------------------------------------------
# AUXILIARY-ENERGY WEIGHTS -- DERIVED FROM THE COST MODEL, NOT ASSUMED
# -----------------------------------------------------------------------------
W_AUX_ENERGY_SCALE = 0.10   # multiplies the two auxiliary-energy weights below

W_KEEPWARM         = 0.10 * W_AUX_ENERGY_SCALE   # idle keep-warm heating energy
KEEPWARM_NORM_W    = 44000.0
W_CHILLER          = 0.10 * W_AUX_ENERGY_SCALE
# --- Thermal-cycling penalty (extends the smooth-operation objective to temperature) ---
W_TCYCLE           = 0.04
TCYCLE_NORM_C      = 34.0

# --- Steady cold-operation penalty ---
W_WARMOPS          = 0.08

BATT_CAPACITY_WH = 500 * 1000
BATT_MAX_P_W     = 500 * 1000
BATT_EFF         = 0.95
# --- Battery ageing (LFP) --------------------------------------------------------
BATT_CYCLE_LIFE       = 3000    # full equivalent cycles to end of life
BATT_EOL_LOSS         = 0.20    # SoH lost at end of cycle life (100 % -> 80 %)
BATT_CALENDAR_LOSS_PER_YR = 0.02   # calendar fade, independent of cycling. 2 %/yr is a
BATT_SOC_MIN     = 0.10         # 10 % lower cut-off
BATT_SOC_MAX     = 0.90         # 90 % upper cut-off

H2_PRICE_EUR_KG  = 5.0

# --- SECONDARY H2 OFF-TAKE -------------------------------------------------------
H2_SECONDARY_OFFTAKE      = True
H2_SECONDARY_PRICE_EUR_KG = 3.0

ALLOW_GRID_EXPORT = False
EXPORT_TANK_MIN_FRAC = 0.80

LCOH_PV_EUR_KWP       = 700.0
LCOH_PEM_EUR_KW       = 1200.0
LCOH_BATT_EUR_KWH     = 200.0    # EUR/kWh (LFP, installed)
LCOH_TANK_EUR_KG      = 500.0
LCOH_COMP_EUR_KGH     = 6000.0
LCOH_BOP_FRAC         = 0.20
LCOH_CTRL_EUR         = 8000.0
LCOH_BATT_CONV_EUR_KW = 75.0
LCOH_BATT_CONV_LIFE   = 15      # years   — power electronics replace at 15 yr
LCOH_BATT_CONV_OM     = 0.015   # 1.5 %/yr
LCOH_GRID_INV_EUR_KW  = 120.0
LCOH_GRID_INV_KW      = 400.0    # kW      — sized for max simultaneous grid power flow
LCOH_GRID_INV_LIFE    = 15      # years
LCOH_GRID_INV_OM      = 0.020   # 2.0 %/yr
LCOH_PV_LIFE          = 30
LCOH_BOP_LIFE         = 30
LCOH_BATT_LIFE        = 10       # 3000 cycles at ~1 cycle/day ≈ 8.2 yr → round to 10
LCOH_TANK_LIFE        = 25
LCOH_COMP_LIFE        = 15
LCOH_PV_OM            = 0.010    # 1.0 %/yr
LCOH_PEM_OM           = 0.030    # 3.0 %/yr (membrane/catalyst replacement, water)
LCOH_BATT_OM          = 0.010    # 1.0 %/yr
LCOH_TANK_OM          = 0.005    # 0.5 %/yr
LCOH_COMP_OM          = 0.020    # 2.0 %/yr
O2_KG_PER_KG_H2      = 8.0     # kg O2 per kg H2
O2_PRICE_EUR_KG       = 0.10    # EUR/kg — industrial grade (99.5 %+), conservative
LCOH_WACC             = 0.06     # 6 % weighted average cost of capital
LCOH_PROJECT_LIFE     = 30       # project horizon for LCOH calculation (years)
LCOH_TANK_KG_INSTALLED = 4320.0


# --- PEM stack replacement economics: degradation-driven REPEX -----------------
LCOH_PEM_NEW_SOH          = 1.00
LCOH_PEM_EOL_SOH          = 0.80
LCOH_PEM_REPLACEMENT_FRAC = 1.00


def _lcoh_crf(r, n):
    """Capital-recovery factor used by the degradation-driven replacement model."""
    return r * (1 + r)**n / ((1 + r)**n - 1)


def project_pem_stack_replacements(df, stack_capex, wacc=None, project_life=None,
                                   soh_col='SoH_PEM', soh_new=None, soh_eol=None,
                                   replacement_frac=None):
    """Project controller-specific PEM stack life and discounted replacement cost.
    """
    if soh_col not in df.columns:
        raise KeyError(f"{soh_col!r} is required for degradation-driven PEM REPEX")
    if len(df) == 0:
        raise ValueError("Cannot project PEM lifetime from an empty dataframe")

    wacc             = LCOH_WACC if wacc is None else float(wacc)
    project_life     = LCOH_PROJECT_LIFE if project_life is None else float(project_life)
    soh_new          = LCOH_PEM_NEW_SOH if soh_new is None else float(soh_new)
    soh_eol          = LCOH_PEM_EOL_SOH if soh_eol is None else float(soh_eol)
    replacement_frac = (LCOH_PEM_REPLACEMENT_FRAC if replacement_frac is None
                        else float(replacement_frac))

    if not (0.0 < soh_eol < soh_new <= 1.0):
        raise ValueError(f"Require 0 < SoH_EOL < SoH_new <= 1; got {soh_eol}, {soh_new}")
    if replacement_frac < 0:
        raise ValueError("LCOH_PEM_REPLACEMENT_FRAC must be non-negative")

    soh_series = pd.to_numeric(df[soh_col], errors='coerce').dropna()
    if soh_series.empty:
        raise ValueError(f"No valid values found in {soh_col!r}")

    soh_start = soh_new
    soh_end   = float(soh_series.iloc[-1])
    sim_years = max(float(len(df)) * float(DT_HOURS) / 8760.0, 1e-12)

    soh_loss_sim   = max(0.0, soh_start - soh_end)
    annual_soh_loss = soh_loss_sim / sim_years
    degradation_budget = soh_new - soh_eol

    if annual_soh_loss <= 1e-12:
        projected_life = np.inf
        replacement_years = []
    else:
        projected_life = degradation_budget / annual_soh_loss
        replacement_years = []
        t = projected_life
        while t < project_life - 1e-9:
            replacement_years.append(float(t))
            t += projected_life

    replacement_capex = float(stack_capex) * replacement_frac
    replacement_npv = sum(
        replacement_capex / ((1.0 + wacc) ** yr)
        for yr in replacement_years
    )

    project_crf = _lcoh_crf(wacc, project_life)
    initial_stack_ann = float(stack_capex) * project_crf
    repex_ann         = replacement_npv * project_crf
    stack_ann_total   = initial_stack_ann + repex_ann

    return {
        'soh_start': soh_start,
        'soh_end': soh_end,
        'sim_years': sim_years,
        'soh_loss_sim': soh_loss_sim,
        'annual_soh_loss': annual_soh_loss,
        'projected_stack_life_yr': projected_life,
        'replacement_years': replacement_years,
        'stack_replacements': len(replacement_years),
        'total_stacks': 1 + len(replacement_years),
        'replacement_capex_each': replacement_capex,
        'replacement_npv': replacement_npv,
        'initial_stack_ann': initial_stack_ann,
        'repex_ann': repex_ann,
        'stack_ann_total': stack_ann_total,
    }

load_points       = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.70, 0.8, 0.9, 1.0]
efficiency_points = [0, 0.78, 0.80, 0.82, 0.80, 0.79, 0.76, 0.73, 0.71, 0.70, 0.69]
TANK_INIT_MODE     = 'fraction'
TANK_INIT_FRAC     = 0.5
TANK_CYCLIC_ITERS  = 3
TANK_CYCLIC_TOL_KG = 5.0
INITIAL_H2_STORAGE = TANK_INIT_FRAC * storage_capacity
SITE_CYCLIC_S0     = {}     # 'cyclic' mode only: cache of solved per-site levels


def set_initial_storage():
    """Recompute the opening inventory from the CURRENT tank size."""
    global INITIAL_H2_STORAGE
    if TANK_INIT_MODE == 'fraction':
        INITIAL_H2_STORAGE = TANK_INIT_FRAC * storage_capacity
    elif TANK_INIT_MODE == 'empty':
        INITIAL_H2_STORAGE = 0.0
    return INITIAL_H2_STORAGE

# =============================================================================
# %% SHARED PHYSICS MODELS
# =============================================================================
def calculate_degradation_step(SoH_PV, SoH_PEM, fl, prev_fl, prev_mosf, current_mosf,
                               T_pem_K=None, diag=None, T_pem_prev_K=None):
    """Per-step health decrement for the PV array and the PEM stack."""
    pv_deg_step  = (0.005 / 8760) * DT_HOURS

    delta_fl = abs(fl - prev_fl)

    if prev_mosf == 11 and current_mosf != 11:
        switch_stress = 5e-6
    else:
        switch_stress = abs(current_mosf - prev_mosf) * 1e-7

    # --- Thermal (cold-operation) multiplier ---
    f_therm = 1.0
    if T_pem_K is not None and THERMAL_DEGRADATION:
        global _V_DEG_REF
        if _V_DEG_REF is None:
            _V_DEG_REF = float(V_cell(T_DEG_REF_K, p_cat, p_an, delta_mem, A_cell_pem,
                                      a_an, a_cat, i_cell, i_0_an, i_0_cat,
                                      T_ref_pem, i_lim))
        if _V_DEG_REF > 0:
            v_now = V_cell(float(T_pem_K), p_cat, p_an, delta_mem, A_cell_pem, a_an, a_cat,
                           i_cell, i_0_an, i_0_cat, T_ref_pem, i_lim)
            f_therm = float(max(1.0, (v_now / _V_DEG_REF) ** KAPPA_THERM))

    # --- M1: STEADY OPERATION (Elizetxea-Navarro Eqs. 7-8) -----------------------
    k_uV    = soh_per_uV()
    j_op    = PEM_J_NOM_A_CM2 * min(max(float(fl), 0.0), 1.0)
    mu_dyn  = MU_DYN_SLOPE * j_op + MU_DYN_INTERCEPT
    if float(fl) < PEM_IDLE_FL_MIN:
        mu_dyn *= MU_IDLE_FRAC
    mu_ramp = MU_RAMP_FACTOR * mu_dyn * delta_fl / DT_HOURS
    f_age   = pem_ageing_factor(SoH_PEM)
    f_T     = pem_thermal_level_factor(T_pem_K)
    scale   = f_age * f_therm * f_T * DT_HOURS * k_uV

    new_SoH_PEM = max(0.6, SoH_PEM - (mu_dyn + mu_ramp) * scale)

    if diag is not None:
        diag['load'] = diag.get('load', 0.0) + mu_dyn  * scale
        diag['rccd'] = diag.get('rccd', 0.0) + mu_ramp * scale

    # --- M2: THERMAL CYCLING -----------------------------------------------------
    if (THERMAL_CYCLING_DEGRADATION and MU_TCYC_UV_PER_K > 0.0
            and T_pem_K is not None and T_pem_prev_K is not None):
        dT_excess = max(0.0, abs(float(T_pem_K) - float(T_pem_prev_K)) - TCYC_DEADBAND_K)
        tcyc_soh  = MU_TCYC_UV_PER_K * dT_excess * k_uV
        new_SoH_PEM -= tcyc_soh
        if diag is not None:
            diag['tcyc'] = diag.get('tcyc', 0.0) + tcyc_soh

    # --- M3: ON/OFF --------------------------------------------------------------
    # --- AGGRESSIVE COLD-START PENALTY (The "MOO-crater" fix) ---
    T_cs_K = T_pem_prev_K if T_pem_prev_K is not None else T_pem_K
    if fl >= 0.05 and prev_fl < 0.05:
        b_cold = 1.0 if (T_cs_K is not None
                         and T_cs_K < (COLDSTART_THRESHOLD_C + 273.15)) else 0.0
        b_warm = 1.0 - b_cold
        sd_soh = (b_cold * MU_SD_COLD_UV + b_warm * MU_SD_WARM_UV) * k_uV
        new_SoH_PEM -= sd_soh
        if diag is not None:
            diag['cold'] = diag.get('cold', 0.0) + sd_soh

    new_SoH_PV  = max(0.6, SoH_PV  - pv_deg_step - switch_stress)
    return new_SoH_PV, new_SoH_PEM

# --- H2 storage & compression design -----------------------------------------
P_STORE_MAX_BAR = 200.0
P_DELIVER_BAR   = 180.0

def h2_compression_kwh_per_kg(P1_bar, P2_bar, T_in=313.0, eta=0.70, k=1.41, Z=1.10):
    """Multi-stage, intercooled H2 compression specific energy (kWh per kg H2)."""
    if P2_bar <= P1_bar:
        return 0.0
    ratio = P2_bar / P1_bar
    N = max(1, int(np.ceil(np.log(ratio) / np.log(2.3))))
    r = ratio ** (1.0 / N)
    R, M = 8.314, 2.016e-3
    w = N * (Z * R * T_in / M) * (k / (k - 1.0)) * (r ** ((k - 1.0) / k) - 1.0) / eta  # J/kg
    return w / 3.6e6


# =============================================================================
# %% THERMAL MODEL PARAMETERS
# =============================================================================
THERMAL_UA_W_K          = 667.0        # W/K
THERMAL_MCP_J_K         = 10_000_000.0  # J/K
CHILLER_COP             = 3.0
CHILLER_COP_LIFT_AWARE  = False        # True -> COP scales with (T_stack - T_amb)
CHILLER_COP_DT_NOM_K    = 40.0         # K, the lift at which COP == CHILLER_COP
CHILLER_COP_MIN         = 1.0
CHILLER_COP_MAX         = 8.0


def chiller_cop(T_stack_K, T_amb_K):
    """Effective COP for rejecting heat from the stack to ambient."""
    if not CHILLER_COP_LIFT_AWARE:
        return CHILLER_COP
    lift = max(1.0, float(T_stack_K) - float(T_amb_K))
    return float(min(max(CHILLER_COP * lift / CHILLER_COP_DT_NOM_K,
                         CHILLER_COP_MIN), CHILLER_COP_MAX))


def calculate_thermal_inertia(prev_T_pem, T_amb_C, P_pem_actual, prod_eff,
                              T_min_C=None, T_max_C=70.0):
    """Lumped-capacitance PEM stack thermal model with active cooling at 70 C."""
    T_amb_K = T_amb_C + 273.15
    Q_gen   = P_pem_actual * (1.0 - prod_eff) if P_pem_actual > 0 else 0.0
    U_A     = THERMAL_UA_W_K   # W/K  — stack-to-ambient overall thermal conductance
    Q_loss  = U_A * (prev_T_pem - T_amb_K)
    m_Cp    = THERMAL_MCP_J_K
    Q_net   = Q_gen - Q_loss
    delta_T = (Q_net * (DT_HOURS * 3600)) / m_Cp
    T_new   = prev_T_pem + delta_T

    P_cooling_W = 0.0
    T_MAX = T_max_C + 273.15
    if T_min_C is None:
        T_min_C = KEEPWARM_BASELINE_C
    T_MIN = T_min_C + 273.15
    if T_new > T_MAX:
        excess_heat_J    = (T_new - T_MAX) * m_Cp
        cooling_energy_J = excess_heat_J / chiller_cop(T_new, T_amb_K)
        P_cooling_W      = cooling_energy_J / (DT_HOURS * 3600)
        T_new            = T_MAX
    elif T_new < T_MIN:
        deficit_heat_J   = (T_MIN - T_new) * m_Cp
        P_cooling_W      = deficit_heat_J / (DT_HOURS * 3600)
        T_new            = T_MIN
    return T_new, P_cooling_W


def simulate_ems_routing(P_op_direct, mppt_loss, batt_energy, storage, g, n_par,
                         prev_T_pem, T_amb_C, SoH_BATT,
                         is_settlement=False, period_h2_direct=0.0,
                         is_last_step=False, annual_unmet_so_far=0.0, T_min_C=None,
                         T_max_C=70.0, pv_scale=1.0, I_op=0.0):
    """Unified EMS — physically consistent, daily H2 demand contract."""
    PERIOD_H2_DEMAND = hourly_H2_target if DEMAND_MODE == 'hourly' else daily_H2_target
    grid_bought = grid_sold = batt_charge_w = batt_discharge_w = 0.0

    P_pem_actual = P_op_direct
    I_pem        = max(0.0, float(I_op))
    pv_excess_from_clip = 0.0
    if P_pem_actual > PEM_capacity:
        pv_excess_from_clip = P_pem_actual - PEM_capacity
        if P_op_direct > 0.0:
            I_pem *= PEM_capacity / P_op_direct
        P_pem_actual = PEM_capacity

    fl = np.clip(P_pem_actual / PEM_capacity, 0, 1)
    H2_rate_kgh = h2_rate_faraday(I_pem, P_pem_actual)
    H2_prod     = H2_rate_kgh * DT_HOURS        # kg produced in THIS step

    therm_eff = (N_SER_PEM * I_pem * V_TH_HHV / P_pem_actual) if P_pem_actual > 20 else 0.0
    therm_eff = float(np.clip(therm_eff, 0.0, 1.0))

    H2_direct   = min(H2_prod, step_H2_target)
    H2_to_store = H2_prod - H2_direct

    new_storage = storage + H2_to_store
    overflow    = max(0.0, new_storage - storage_capacity)
    storage     = min(storage_capacity, new_storage)

    H2_unmet     = 0.0
    H2_delivered = H2_direct          # non-settlement hours: this hour's direct only
    is_settlement_now = is_settlement or is_last_step
    if is_settlement_now:
        served_direct    = period_h2_direct + H2_direct
        remaining_demand = max(0.0, PERIOD_H2_DEMAND - served_direct)
        if storage >= remaining_demand:
            storage       -= remaining_demand
            H2_from_store  = remaining_demand
        else:
            H2_unmet       = remaining_demand - storage
            H2_from_store  = storage
            storage        = 0.0
        H2_delivered = H2_direct + H2_from_store

        if is_last_step and storage < INITIAL_H2_STORAGE:
            H2_unmet += INITIAL_H2_STORAGE - storage

    T_new, P_cooling_W = calculate_thermal_inertia(prev_T_pem, T_amb_C, P_pem_actual, therm_eff,
                                                   T_min_C=T_min_C, T_max_C=T_max_C)

    tank_fill_fraction = storage / storage_capacity
    P_vessel_bar    = p_cat + tank_fill_fraction * (P_STORE_MAX_BAR - p_cat)
    comp_store_kwh  = h2_compression_kwh_per_kg(p_cat, P_vessel_bar)      # 40 → vessel (≤200 bar)
    comp_deliv_kwh  = h2_compression_kwh_per_kg(p_cat, P_DELIVER_BAR)     # 40 → 180 bar
    H2_stored_now   = max(0.0, H2_to_store - overflow)
    P_compression_W = (H2_direct * comp_deliv_kwh + H2_stored_now * comp_store_kwh) * 1000.0 / DT_HOURS
    P_bop_W            = P_pem_actual * 0.05
    P_idle_W           = PEM_capacity * 0.02 if P_pem_actual < (PEM_capacity * 0.05) else 0.0
    BOP_DRIVE_EFF      = 0.97
    P_aux_total        = (P_cooling_W + P_compression_W + P_bop_W + P_idle_W) / BOP_DRIVE_EFF

    P_pv_active_mppt = P_op_direct + mppt_loss
    n_unused  = max(0, N_PAR_MAX - n_par)   # per-site installed array (top switch position)
    P_unused_PV_W = (
        P_pv_active_mppt * (n_unused / n_par)
        if (n_par > 0 and P_pv_active_mppt > 0) else 0.0
    )
    pv_available = P_unused_PV_W  + pv_excess_from_clip

    P_aux_remaining = P_aux_total
    if pv_available >= P_aux_remaining:
        pv_available    -= P_aux_remaining
        P_aux_remaining  = 0.0
    else:
        P_aux_remaining -= pv_available
        pv_available     = 0.0

    if P_aux_remaining > 0:
        avail_wh_aux   = max(0.0, batt_energy - BATT_CAPACITY_WH * BATT_SOC_MIN)
        max_batt_p_aux = min(BATT_MAX_P_W, (avail_wh_aux / DT_HOURS) * BATT_EFF)
        batt_aux_w     = min(P_aux_remaining, max_batt_p_aux)
        batt_discharge_w += batt_aux_w
        batt_energy      -= (batt_aux_w * DT_HOURS) / BATT_EFF
        P_aux_remaining  -= batt_aux_w
        if P_aux_remaining > 0:
            grid_bought += P_aux_remaining

    P_pv_excess_W = pv_available
    if batt_discharge_w == 0.0 and P_pv_excess_W > 0:
        headroom_wh   = max(0.0, BATT_CAPACITY_WH * BATT_SOC_MAX - batt_energy)
        actual_charge = min(P_pv_excess_W, BATT_MAX_P_W, headroom_wh / DT_HOURS)
        batt_charge_w += actual_charge
        batt_energy   += actual_charge * DT_HOURS * BATT_EFF

        surplus   = max(0.0, P_pv_excess_W - actual_charge)
        tank_frac = storage / storage_capacity if storage_capacity > 0 else 1.0

        if ALLOW_GRID_EXPORT and H2_unmet < 1e-3 and tank_frac >= EXPORT_TANK_MIN_FRAC:
            grid_sold = surplus
        else:
            grid_sold = 0.0

    eq_full_cycles = ((batt_charge_w + batt_discharge_w) * DT_HOURS
                      / (2.0 * BATT_CAPACITY_WH))
    batt_cycle_deg = BATT_EOL_LOSS * eq_full_cycles / BATT_CYCLE_LIFE
    batt_cal_deg   = (BATT_CALENDAR_LOSS_PER_YR / 8760.0) * DT_HOURS
    SoH_BATT = max(0.0, SoH_BATT - batt_cycle_deg - batt_cal_deg)

    energy_curtailed = 0.0
    if H2_unmet > 1e-3:  # H2 not fully met
        energy_curtailed = max(0.0, P_pv_excess_W - batt_charge_w) if P_pv_excess_W > 0 else 0.0

    grid_penalty = H2_unmet * 10.0
    H2_grid = 0.0
    H2_ren  = H2_prod
    StH = float(np.clip(
        (H2_rate_kgh * LHV) / (calculate_effective_irradiance(g, 0.1 * g) * N_SERIES_PV * n_par * pv_scale * pv_module_area), 0, 0.185
    )) if g > 80 else 0.0

    return (batt_energy, storage, grid_bought, grid_sold,
            batt_charge_w, batt_discharge_w,
            H2_prod, StH, fl, P_pem_actual,
            grid_penalty, overflow,
            H2_prod, H2_grid, H2_ren,
            T_new, SoH_BATT, H2_unmet,
            H2_direct, H2_delivered,
            P_bop_W, P_compression_W, P_cooling_W, P_unused_PV_W, P_pv_excess_W)


def calculate_effective_irradiance(G, G_rear, tilt_deg=tilt_angle_deg,
                                   zen_deg=sun_zenith_deg, bifaciality=0.9, albedo=0.2):
    tilt, zen = np.radians(tilt_deg), np.radians(zen_deg)
    return max(0.0, G * np.cos(tilt - zen) + 0.3*G + albedo*G*(1 - np.cos(tilt))/2 + G_rear*bifaciality)


def calculate_iph(Iph_ref_, G, T_amb, G_ref=1000, T_ref_C=25, alpha=0.0005):
    return Iph_ref_ * (G / G_ref) * (1 + alpha * (T_amb - T_ref_C))


def enhanced_recombination(voltage, temp_C, Rs_, Rsh_, rho_c, A_c, aSi_t, TCO_r):
    R_total = Rs_ + rho_c / A_c
    S = 1e5 * np.exp(-aSi_t / 2e-9)
    return 1e-12 * (voltage / R_total) * (1 + S / (TCO_r * A_c))


def pv_model(voltage, params, irradiance):
    (Iph_ref_, Is1r, Is2r, Isir, Rs_, Rsh_, n1_, n2_, ni_, rho_c, A_c, aSi_t, TCO_r, temp_C) = params
    T_K      = temp_C + 273.15
    Iph      = calculate_iph(Iph_ref_, irradiance, temp_C)
    Vt       = k * T_K / q
    Rs_total = Rs_ + rho_c / A_c
    Id_prev  = 0.0
    for _ in range(50):
        Is1 = Is1r * (T_K/T_ref)**3 * np.exp(np.clip((Eg*q/(n1_*k))*(1/T_ref - 1/T_K), -200, 200))
        Is2 = Is2r * (T_K/T_ref)**3 * np.exp(np.clip((Eg*q/(n2_*k))*(1/T_ref - 1/T_K), -200, 200))
        Isi = Isir * (T_K/T_ref)**3 * np.exp(np.clip((Eg*q/(ni_*k))*(1/T_ref - 1/T_K), -200, 200))
        Vd  = voltage + voltage * Rs_total / Rsh_
        Id  = (Is1 * (np.exp(np.clip(Vd/(n1_*Ns*Vt), -200, 200)) - 1) +
               Is2 * (np.exp(np.clip(Vd/(n2_*Ns*Vt), -200, 200)) - 1) +
               Isi * (np.exp(np.clip(Vd/(ni_*Ns*Vt), -200, 200)) - 1))
        I_total = ((Iph*Rsh_ - Id*Rsh_ - voltage) / (Rsh_ + Rs_total) -
                   enhanced_recombination(voltage, temp_C, Rs_, Rsh_, rho_c, A_c, aSi_t, TCO_r))
        if abs(Id - Id_prev) < 1e-3:
            break
        Id_prev = Id
    return I_total


def pv_model_series(voltage, params, irradiance, num_series, num_parallel):
    return pv_model(voltage, params, irradiance) * num_parallel, voltage * num_series


def _pv_curve_vec(v_arr, params, irradiance):
    """Vectorised single-module I(V), evaluated over a whole voltage array at once.
    """
    (Iph_ref_, Is1r, Is2r, Isir, Rs_, Rsh_, n1_, n2_, ni_,
     rho_c, A_c, aSi_t, TCO_r, temp_C) = params
    v_arr    = np.asarray(v_arr, dtype=float)
    T_K      = temp_C + 273.15
    Iph      = calculate_iph(Iph_ref_, irradiance, temp_C)
    Vt       = k * T_K / q
    Rs_total = Rs_ + rho_c / A_c

    _tfac = (T_K / T_ref) ** 3
    Is1 = Is1r * _tfac * np.exp(np.clip((Eg * q / (n1_ * k)) * (1 / T_ref - 1 / T_K), -200, 200))
    Is2 = Is2r * _tfac * np.exp(np.clip((Eg * q / (n2_ * k)) * (1 / T_ref - 1 / T_K), -200, 200))
    Isi = Isir * _tfac * np.exp(np.clip((Eg * q / (ni_ * k)) * (1 / T_ref - 1 / T_K), -200, 200))

    Vd = v_arr + v_arr * Rs_total / Rsh_
    Id = (Is1 * (np.exp(np.clip(Vd / (n1_ * Ns * Vt), -200, 200)) - 1) +
          Is2 * (np.exp(np.clip(Vd / (n2_ * Ns * Vt), -200, 200)) - 1) +
          Isi * (np.exp(np.clip(Vd / (ni_ * Ns * Vt), -200, 200)) - 1))
    return ((Iph * Rsh_ - Id * Rsh_ - v_arr) / (Rsh_ + Rs_total)
            - enhanced_recombination(v_arr, temp_C, Rs_, Rsh_, rho_c, A_c, aSi_t, TCO_r))


def perturb_observe_series(params, irradiance, num_series, num_parallel, v_start=0, v_step=1, v_max=65.0):
    """Array-level P&O sweep."""
    v_arr = np.arange(v_start, v_max + 0.5 * v_step, v_step, dtype=float)
    if v_arr.size == 0:
        return [], [], [], 0
    cur = _pv_curve_vec(v_arr, params, irradiance) * num_parallel
    tot_v = v_arr * num_series
    pw = tot_v * cur

    nz = np.flatnonzero(cur <= 0)
    n = int(nz[0]) + 1 if nz.size else v_arr.size

    tot_v, cur, pw = tot_v[:n], cur[:n], pw[:n]
    mppt_idx = int(np.argmax(pw)) if n else 0
    return tot_v.tolist(), cur.tolist(), pw.tolist(), mppt_idx


def find_mppt(voltages, currents):
    V, I = np.array(voltages), np.array(currents)
    P = V * I
    if len(P) == 0:
        return 0.0, 0.0, 0.0
    idx = int(np.argmax(P))
    return V[idx], I[idx], P[idx]


def nearest_point_match(voltages_pv, currents_pv, voltages_pem, currents_pem):
    """DEPRECATED nearest-point matcher, retained only to reproduce earlier results.
    """
    V_pv = np.array(voltages_pv)[:, None]
    I_pv = np.array(currents_pv)[:, None]
    V_pm = np.array(voltages_pem)[None, :]
    I_pm = np.array(currents_pem)[None, :]
    dist = np.sqrt((V_pv - V_pm)**2 + (I_pv - I_pm)**2)
    idx  = np.unravel_index(np.argmin(dist), dist.shape)
    return float(voltages_pv[idx[0]]), float(currents_pv[idx[0]])


def find_intersection_point(voltages_pv, currents_pv, voltages_pem, currents_pem):
    """Direct-coupled operating point, or None when the curves do not cross."""
    v = np.asarray(voltages_pv, dtype=float)
    c = np.asarray(currents_pv, dtype=float)
    pv_ = np.asarray(voltages_pem, dtype=float)
    pi_ = np.asarray(currents_pem, dtype=float)
    if v.size < 2 or pv_.size < 2:
        return None

    v_lo = max(v.min(), pv_.min())
    v_hi = min(v.max(), pv_.max())
    if v_hi <= v_lo:
        return None

    vg   = np.linspace(v_lo, v_hi, 500)
    diff = np.interp(vg, v, c) - np.interp(vg, pv_, pi_)
    k    = np.where(np.diff(np.sign(diff)) != 0)[0]
    if k.size == 0:
        return None

    k = int(k[0])
    v0, v1, d0, d1 = vg[k], vg[k + 1], diff[k], diff[k + 1]
    vc = v0 - d0 * (v1 - v0) / (d1 - d0) if d1 != d0 else v0
    return float(vc), float(np.interp(vc, v, c))


def V_cell(T, p_cat_, p_an_, delta, A_cp, a_a, a_c, i, i0a, i0c, Tref, ilim,
           dV_cell=0.0):
    """Semi-empirical PEM single-cell voltage (Maroufmashat2014 + Springer1991)."""
    R, F = 8.314, 9.648e4
    k1, k2, k3, k4, k5 = 8.5e-4, R/F, 6.1078e-3, 17.694, 34.85
    k6, k7, k8, k9     = 0.1113, 3.26e-3, 1268, R/F
    sat  = k3 * np.exp(k4 * ((T-273.15) / (T-k5)))
    pc   = np.clip(p_cat_ - sat, 1e-3, 1e6)
    pa   = np.clip(p_an_  - sat, 1e-3, 1e6)
    ph2o = np.clip(sat,          1e-6, 1e6)
    Erev = 1.229 - k1*(T-298.15) + (k2/2)*T*np.log(np.clip((pc*pa**2)/ph2o, 1e-20, 1e20))

    j     = np.clip(np.asarray(i, dtype=float) / A_cp, 0.0, ilim - 1e-3)

    sigma = np.clip((k6 - k7) * np.exp(k8*(1/303 - 1/T)), 1e-9, 1e9)   # S/cm
    Vohm  = (delta / sigma) * j

    i0a_T, i0c_T = i0a, i0c
    if PEM_I0_ARRHENIUS:
        i0a_T = i0a * np.exp(-(E_ACT_AN  / R) * (1/T - 1/Tref))
        i0c_T = i0c * np.exp(-(E_ACT_CAT / R) * (1/T - 1/Tref))

    j_s   = np.clip(j, 1e-9, None)
    Vact  = (k9*T/a_a)*np.log(j_s/i0a_T) + (k9*T/a_c)*np.log(j_s/i0c_T)
    Vcon  = (k2*T/(2*a_a))*np.log(ilim/(ilim-j))
    return Erev + Vohm + Vact + Vcon + dV_cell


def V_PEM(T, p_c, p_a, delta, A_cp, a_a, a_c, i, i0a, i0c, Tref, ilim, n_series,
          dV_cell=0.0):
    return V_cell(T, p_c, p_a, delta, A_cp, a_a, a_c, i, i0a, i0c, Tref, ilim,
                  dV_cell) * n_series


# --- SoH -> polarization-curve coupling ---------------------------------------
V_DEG_EOL_FRAC       = 0.10
PEM_DEG_OVERVOLTAGE  = True
_V_BOL_CELL          = None   # memoised pristine V_cell at T_DEG_REF_K


def pem_deg_overvoltage(SoH_PEM):
    """Lumped per-cell degradation overvoltage [V] implied by the health state."""
    if not PEM_DEG_OVERVOLTAGE:
        return 0.0
    span  = max(1.0 - LCOH_PEM_EOL_SOH, 1e-9)
    f_deg = min(max((1.0 - float(SoH_PEM)) / span, 0.0), 1.5)
    return f_deg * V_DEG_EOL_FRAC * _v_bol_cell()


def _v_bol_cell():
    """Memoised beginning-of-life cell voltage [V] at the nominal operating point.
    """
    global _V_BOL_CELL
    if _V_BOL_CELL is None:
        _V_BOL_CELL = float(V_cell(T_DEG_REF_K, p_cat, p_an, delta_mem, A_cell_pem,
                                   a_an, a_cat, i_cell, i_0_an, i_0_cat,
                                   T_ref_pem, i_lim))
    return _V_BOL_CELL


def soh_per_uV():
    """SoH decrement per microvolt of cell overvoltage, from Eq. dVdeg."""
    return (1.0 - LCOH_PEM_EOL_SOH) / (V_DEG_EOL_FRAC * _v_bol_cell() * 1e6)


def pem_ageing_factor(SoH_PEM):
    """Dimensionless ageing multiplier from Elizetxea-Navarro Eq. (7)."""
    if AGEING_MODE != 'explicit':
        return 1.0
    span   = max(1.0 - LCOH_PEM_EOL_SOH, 1e-9)
    damage = min(max((1.0 - float(SoH_PEM)) / span, 0.0), 1.0)
    t_op   = damage * PEM_LIFE_H_REF
    return (MU_EVOL_SLOPE * t_op + MU_EVOL_INTERCEPT) / MU_EVOL_MEAN


def pem_thermal_level_factor(T_pem_K):
    """Hot-side acceleration, doubling every T_DOUBLE_K above the reference."""
    if not THERMAL_LEVEL_DEGRADATION or T_pem_K is None:
        return 1.0
    dT = max(0.0, float(T_pem_K) - T_DEG_REF_K)
    return float(min(2.0 ** (dT / T_DOUBLE_K), F_T_MAX))


def _sync_degradation_charges():
    """Convert the microvolt restart charges into SoH decrements."""
    global COLDSTART_SOH_COLD, COLDSTART_SOH_WARM, COLDSTART_EXCESS_DAMAGE_FRAC
    global MU_SD_WARM_UV
    MU_SD_WARM_UV      = MU_SD_COLD_UV * MU_SD_WARM_RATIO
    k                  = soh_per_uV()
    COLDSTART_SOH_COLD = MU_SD_COLD_UV * k
    COLDSTART_SOH_WARM = MU_SD_WARM_UV * k
    COLDSTART_EXCESS_DAMAGE_FRAC = max(0.0, 1.0 - MU_SD_WARM_RATIO)
    return COLDSTART_SOH_COLD, COLDSTART_SOH_WARM


def degradation_budget_check(df=None, soh_col='SoH_PEM', restarts=None,
                             cold_fraction=1.0, op_hours=None, verbose=True):
    """Effective uV/h and per-mechanism split, against the literature bands."""
    k   = soh_per_uV()
    out = {
        'soh_per_uV'       : k,
        'EOL_budget_uV'    : (1.0 - LCOH_PEM_EOL_SOH) / k,
        'M1_idle_uV_per_h' : MU_DYN_INTERCEPT,
        'M1_full_uV_per_h' : MU_DYN_SLOPE * PEM_J_NOM_A_CM2 + MU_DYN_INTERCEPT,
        'M1_ramp_factor'   : MU_RAMP_FACTOR,
        'M2_f_T_at_80C'    : pem_thermal_level_factor(353.15),
        'M2_uV_per_K'      : MU_TCYC_UV_PER_K if THERMAL_CYCLING_DEGRADATION else 0.0,
        'M2_deadband_K'    : TCYC_DEADBAND_K,
        'M3_cold_uV'       : MU_SD_COLD_UV,
        'M3_warm_uV'       : MU_SD_WARM_UV,
        'ageing_mode'      : AGEING_MODE,
    }
    if df is not None:
        soh    = pd.to_numeric(df[soh_col], errors='coerce').dropna()
        years  = max(float(len(df)) * float(DT_HOURS) / 8760.0, 1e-12)
        lost   = max(0.0, 1.0 - float(soh.iloc[-1])) / years
        if op_hours is None and 'fl' in df.columns:
            op_hours = float((pd.to_numeric(df['fl'], errors='coerce') >= 0.05).sum()) \
                       * float(DT_HOURS) / years
        out['annual_soh_loss']   = lost
        out['annual_uV']         = lost / k
        out['op_hours_per_yr']   = op_hours
        if op_hours:
            out['effective_uV_per_h'] = (lost / k) / op_hours
        if restarts is not None:
            out['M3_annual_uV'] = restarts * (cold_fraction * MU_SD_COLD_UV +
                                              (1.0 - cold_fraction) * MU_SD_WARM_UV)
    if verbose:
        for key, val in out.items():
            print('  %-20s %s' % (key, val))
    return out


_sync_degradation_charges()


def I_PEM(i_cell_, n_parallel):
    return i_cell_ * n_parallel

# =============================================================================
# %% SYSTEM SIZING REPORT
# =============================================================================

def print_system_sizing():
    """Compute and print key system parameters at STC (G=1000 W/m², T=25°C)."""
    G_stc, T_stc = 1000.0, 25.0
    N_PAR_MAX    = globals()['N_PAR_MAX']   # per-site installed strings (set by the loop)
    N_PAR_MIN    = globals()['N_PAR_BASE']
    N_SER_PEM    = globals()['N_SER_PEM']

    params_stc = [calculate_iph(Iph_ref, G_stc, T_stc),
                  Is_ref1, Is_ref2, Is_refi, Rs, Rsh,
                  n1, n2, ni, rho_contact, A_cell, aSi_thickness,
                  TCO_resistivity, T_stc]
    eff_stc = calculate_effective_irradiance(G_stc, 0.1 * G_stc)
    vols, curs, _, _ = perturb_observe_series(params_stc, eff_stc,
                                              N_SERIES_PV, N_PAR_MAX, v_step=1)
    if vols:
        Vm_max, Im_max, Pm_max = find_mppt(vols, curs)
    else:
        Vm_max = Im_max = Pm_max = 0.0

    vols_min, curs_min, _, _ = perturb_observe_series(params_stc, eff_stc,
                                                      N_SERIES_PV, N_PAR_MIN, v_step=1)
    if vols_min:
        Vm_min, Im_min, Pm_min = find_mppt(vols_min, curs_min)
    else:
        Vm_min = Im_min = Pm_min = 0.0

    T_pem_op  = 60.0 + 273.15
    pem_I_arr = np.linspace(0.0, float(I_PEM(i_cell, 1)), 200)
    V_pem_arr = V_PEM(T_pem_op, p_cat, p_an, delta_mem, A_cell_pem,
                      a_an, a_cat, pem_I_arr, i_0_an, i_0_cat,
                      T_ref_pem, i_lim, N_SER_PEM)
    P_pem_arr = V_pem_arr * pem_I_arr
    idx_rated = int(np.argmax(P_pem_arr))
    V_pem_rated = float(V_pem_arr[idx_rated])
    I_pem_rated = float(pem_I_arr[idx_rated])

    oversizing = Pm_max / PEM_capacity if PEM_capacity > 0 else 0.0
    batt_usable_kwh = BATT_CAPACITY_WH * (BATT_SOC_MAX - BATT_SOC_MIN) / 1000.0
    daily_demand_kg = daily_H2_target

    W = 62
    print(f"\n{'='*W}")
    print(f"  SYSTEM SIZING  —  STC: G=1000 W/m², T_amb=25°C, T_PEM=60°C")
    print(f"{'='*W}")
    print(f"  PV ARRAY")
    print(f"    Strings × series  : {N_PAR_MAX} par × {N_SERIES_PV} series  "
          f"({N_PAR_MAX * N_SERIES_PV} modules total)")
    print(f"    Aperture area     : {N_PAR_MAX * N_SERIES_PV * pv_module_area:.0f} m²")
    print(f"    Peak MPP (all 36) : {Pm_max/1000:.1f} kW  "
          f"[{Vm_max:.0f} V × {Im_max:.0f} A]")
    print(f"    Min config (24 s.): {Pm_min/1000:.1f} kW  "
          f"[{Vm_min:.0f} V × {Im_min:.0f} A]")
    print(f"    PV / PEM ratio    : {oversizing:.2f}×  "
          f"({'OVERSIZED' if oversizing > 1.4 else 'OK'})")
    print(f"  PEM ELECTROLYSER")
    print(f"    Rated power       : {PEM_capacity/1000:.0f} kW")
    print(f"    Rated point (60°C): {V_pem_rated:.0f} V × {I_pem_rated:.0f} A")
    print(f"    Peak H₂ rate      : {H2_PEAK_KGH:.1f} kg/h")
    print(f"    Series cells      : {N_SER_PEM}")
    print(f"  BATTERY")
    print(f"    Capacity (total)  : {BATT_CAPACITY_WH/1000:.0f} kWh")
    print(f"    Usable (10–90 %)  : {batt_usable_kwh:.0f} kWh")
    print(f"    Max power         : {BATT_MAX_P_W/1000:.0f} kW  "
          f"(C-rate {BATT_MAX_P_W/BATT_CAPACITY_WH:.2f})")
    print(f"    Initial SOC       : {BATT_SOC_MIN*100:.0f} %  (empty start)")
    print(f"  H₂ STORAGE & DEMAND")
    print(f"    Tank capacity     : {storage_capacity:.0f} kg")
    print(f"    Initial storage   : {INITIAL_H2_STORAGE:.0f} kg  "
          f"({INITIAL_H2_STORAGE/daily_demand_kg:.1f} days, repaid by year end)")
    print(f"    Daily demand      : {daily_demand_kg:.0f} kg/day")
    print(f"    Tank autonomy     : {storage_capacity/daily_demand_kg:.0f} days (full tank)")
    print(f"{'='*W}\n")
    return Pm_max / 1000.0   # kWp installed (all 36 strings at STC)


# =============================================================================
# %% MOO BASELINE & MOO+RF (Online Random Forest)
# =============================================================================

# =============================================================================
# %% FORECAST QUALITY
# =============================================================================
FORECAST_MODE           = 'perfect'   # 'perfect' | 'persistence' | 'noisy'
FORECAST_NOISE_PER_STEP = 0.10        # +/- fraction per step of horizon, 'noisy' only
FORECAST_SEED           = 12345


def forecast_value(true_v, current_v, k, rng=None):
    """What a controller is TOLD a quantity will be k steps ahead."""
    if FORECAST_MODE == 'persistence':
        return float(current_v)
    if FORECAST_MODE == 'noisy':
        r = np.random if rng is None else rng
        eps = float(r.uniform(-FORECAST_NOISE_PER_STEP * k, FORECAST_NOISE_PER_STEP * k))
        return float(true_v) * (1.0 + eps)
    return float(true_v)


_FC_RNG = np.random.RandomState(FORECAST_SEED)


def reset_forecast_rng():
    """Re-seed the baselines' forecast RNG so repeated runs are identical."""
    global _FC_RNG
    _FC_RNG = np.random.RandomState(FORECAST_SEED)
    return _FC_RNG


def prewarm_setpoint_rule(T_pem_C, g_fcst, mode='anticipatory'):
    """Hand-designed warm-standby rule -- the 5th ('MOO+PreWarm') baseline."""
    lo, mid, hi = KEEPWARM_LEVELS_C          # (30, 45, 60)
    if mode == 'always_warm':
        return hi
    if mode == 'distilled':
        g2 = float(g_fcst[1]) if len(g_fcst) > 1 else 0.0
        g3 = float(g_fcst[2]) if len(g_fcst) > 2 else 0.0
        if T_pem_C > 60.0 and g2 > 1.0:                  return hi
        if T_pem_C > 37.5 and g3 > 196.0:                return mid
        if g2 > 67.0 or g3 > 196.0:                      return hi
        return lo
    return hi if (len(g_fcst) and max(g_fcst) > 80.0) else lo


def run_baselines(irr_data, temp_data_C, price_data_mwh, use_rf=False, hysteresis=None,
                  lambda_sw=None, pv_scale=1.0, prewarm=None,
                  prewarm_offsets=FORECAST_OFFSETS_STEPS):
    _use_hyst = MOO_HYSTERESIS if hysteresis is None else hysteresis
    _lam_sw   = MOO_LAMBDA_SW  if lambda_sw  is None else float(lambda_sw)
    records = []
    SoH_PV, SoH_PEM, SoH_BATT = 1.0, 1.0, 1.0
    batt_energy         = BATT_CAPACITY_WH * BATT_SOC_MIN
    storage             = INITIAL_H2_STORAGE
    prev_mosf           = 3
    prev_fl             = 0.0
    pem_I_array         = np.linspace(0, I_PEM(i_cell, 1), 100)
    T_pem               = KEEPWARM_BASELINE_C + 273.15   # start at the standby floor
    period_h2_direct     = 0.0
    annual_h2_unmet     = 0.0
    deg_diag            = {'load': 0.0, 'rccd': 0.0, 'cold': 0.0, 'tcyc': 0.0}

    X_data, y_data = [], []
    rf_model = RandomForestRegressor(
        n_estimators=30, max_depth=10, min_samples_leaf=3,
        min_samples_split=5, n_jobs=None, random_state=42
    )
    kdtree = None
    prediction_used_count = full_optimization_count = 0

    mode_name = "MOO+RF (Online ML)" if use_rf else "Standard MOO"
    print(f"\nRunning {mode_name} Baseline over dataset...")

    for i in tqdm(range(len(irr_data)), desc=mode_name):
        g             = float(np.clip(irr_data[i], 0, 1200))
        t_amb         = float(temp_data_C[i])
        cur_price_mwh = float(price_data_mwh[i])
        buy_price_kwh  = cur_price_mwh / 1000.0
        sell_price_kwh = buy_price_kwh * 0.8

        deg_Iph_ref   = Iph_ref * SoH_PV * pv_scale
        dV_deg        = pem_deg_overvoltage(SoH_PEM)
        dynamic_pem_V = V_PEM(T_pem, p_cat, p_an, delta_mem, A_cell_pem,
                              a_an, a_cat, pem_I_array, i_0_an, i_0_cat, T_ref_pem, i_lim, N_SER_PEM,
                              dV_cell=dV_deg)

        best_score  = float('inf')
        V_op_rec = I_op_rec = 0.0   # operating point of the CHOSEN configuration
        best_n_par  = N_PAR_BASE + prev_mosf * N_PAR_STEP
        P_op_direct = mppt_loss = 0.0
        used_prediction = False

        if g >= 80:
            params_t = [deg_Iph_ref, Is_ref1, Is_ref2, Is_refi, Rs, Rsh,
                        n1, n2, ni, rho_contact, A_cell, aSi_thickness, TCO_resistivity, t_amb]
            eff_G = calculate_effective_irradiance(g, 0.1*g)

            if use_rf and kdtree is not None and len(X_data) >= 60:
                dist, _ = kdtree.query([g, t_amb])
                if dist < 20.0:
                    pred_n_par = rf_model.predict([[g, t_amb]])[0]
                    mosf_pred  = int(np.clip(round((pred_n_par - N_PAR_BASE) / N_PAR_STEP), 0, 6))
                    for mosf in sorted(set(range(max(0, mosf_pred - 2), min(6, mosf_pred + 2) + 1)) | {int(prev_mosf)}):
                        n_par = N_PAR_BASE + mosf * N_PAR_STEP
                        vols, curs, _, _ = perturb_observe_series(params_t, eff_G, N_SERIES_PV, n_par)
                        if not vols:
                            continue
                        V_mpp, I_mpp, P_mpp = find_mppt(vols, curs)
                        pt = find_intersection_point(vols, curs,
                                                     dynamic_pem_V.tolist(), pem_I_array.tolist())
                        if pt:
                            V_op, I_op = pt
                            P_op_direct_temp = V_op * I_op
                            mppt_loss_temp   = mppt_shortfall(P_mpp, P_op_direct_temp)
                            fl_t = np.clip(P_op_direct_temp / PEM_capacity, 0, 1) if P_op_direct_temp > 20 else 0
                            H2_t = h2_rate_faraday(I_op, P_op_direct_temp)
                            StH_t = float(np.clip((H2_t * LHV) / (eff_G * N_SERIES_PV * n_par * pv_scale * pv_module_area), 0, 0.185)) if g > 80 else 0.0
                            score = -DT_HOURS * (W_H2 * (H2_t / H2_PEAK_KGH) + W_STH * (StH_t / 0.185) - W_MPPT * (mppt_loss_temp / MPPT_NORM))
                            if _lam_sw > 0.0:
                                score += _lam_sw * abs(mosf - prev_mosf) * DT_HOURS
                            if score < best_score:
                                best_score  = score
                                best_n_par  = n_par
                                P_op_direct = P_op_direct_temp
                                mppt_loss   = mppt_loss_temp
                                V_op_rec, I_op_rec = V_op, I_op
                    if best_score < float('inf'):
                        used_prediction = True
                        prediction_used_count += 1

            if not used_prediction:
                full_optimization_count += 1
                for mosf in range(7):
                    n_par = N_PAR_BASE + mosf * N_PAR_STEP
                    vols, curs, _, _ = perturb_observe_series(params_t, eff_G, N_SERIES_PV, n_par)
                    if not vols:
                        continue
                    V_mpp, I_mpp, P_mpp = find_mppt(vols, curs)
                    pt = find_intersection_point(vols, curs,
                                                 dynamic_pem_V.tolist(), pem_I_array.tolist())
                    if pt:
                        V_op, I_op = pt
                        P_op_direct_temp = V_op * I_op
                        mppt_loss_temp = mppt_shortfall(P_mpp, P_op_direct_temp)

                        fl = np.clip(P_op_direct_temp / PEM_capacity, 0, 1) if P_op_direct_temp > 20 else 0
                        H2_prod_temp = h2_rate_faraday(I_op, P_op_direct_temp)
                        StH_temp = float(np.clip((H2_prod_temp * LHV) / (eff_G * N_SERIES_PV * n_par * pv_scale * pv_module_area), 0, 0.185)) if g > 80 else 0.0

                        score = -DT_HOURS * (W_H2 * (H2_prod_temp / H2_PEAK_KGH) + W_STH * (StH_temp / 0.185) - W_MPPT * (mppt_loss_temp / MPPT_NORM))
                        if _lam_sw > 0.0:
                            score += _lam_sw * abs(mosf - prev_mosf) * DT_HOURS

                        if score < best_score:
                            best_score  = score
                            best_n_par  = n_par
                            P_op_direct = P_op_direct_temp
                            mppt_loss   = mppt_loss_temp
                            V_op_rec, I_op_rec = V_op, I_op

                if use_rf and P_op_direct > 0:
                    X_data.append([g, t_amb])
                    y_data.append(best_n_par)
                    if len(X_data) == 60 or (len(X_data) > 60 and len(X_data) % 720 == 0):
                        rf_model.fit(X_data, y_data)
                        kdtree = KDTree(X_data)

            # --- optional hysteresis / deadband (engineering-floor MOO baseline) ---
            if _use_hyst and P_op_direct > 0:
                cand_mosf = int(round((best_n_par - N_PAR_BASE) / N_PAR_STEP))
                if cand_mosf != prev_mosf and abs(cand_mosf - prev_mosf) < MOO_DEADBAND:
                    n_hold = N_PAR_BASE + prev_mosf * N_PAR_STEP
                    vols_h, curs_h, _, _ = perturb_observe_series(params_t, eff_G, N_SERIES_PV, n_hold)
                    if vols_h:
                        _, _, P_mpp_h = find_mppt(vols_h, curs_h)
                        pt_h = find_intersection_point(vols_h, curs_h,
                                                       dynamic_pem_V.tolist(), pem_I_array.tolist())
                        if pt_h:
                            V_op_h, I_op_h = pt_h
                            best_n_par  = n_hold
                            P_op_direct = V_op_h * I_op_h
                            mppt_loss   = mppt_shortfall(P_mpp_h, P_op_direct)
                            V_op_rec, I_op_rec = V_op_h, I_op_h

        is_last_step = (i == len(irr_data) - 1)
        _settlement_steps = STEPS_PER_HOUR if DEMAND_MODE == 'hourly' else STEPS_PER_DAY
        is_settlement = ((i + 1) % _settlement_steps == 0)
        pass_h2_direct = period_h2_direct

        T_pem_prev_step = T_pem
        if prewarm is None:
            _T_min_C = KEEPWARM_BASELINE_C
        else:
            _g_now = float(np.clip(irr_data[i], 0, 1200))
            _fc = [
                forecast_value(
                    float(np.clip(irr_data[min(i + off, len(irr_data) - 1)], 0, 1200)),
                    _g_now, max(1, off), _FC_RNG)
                for off in prewarm_offsets
            ]
            _T_min_C = prewarm_setpoint_rule(T_pem - 273.15, _fc, mode=prewarm)
        (batt_energy, storage, grid_bought, grid_sold,
         batt_charge_w, batt_discharge_w,
         H2_prod, StH, fl, P_pem_actual,
         grid_penalty, overflow,
         H2_base, H2_grid, H2_ren,
         T_pem, SoH_BATT, H2_unmet,
         H2_direct, H2_delivered,
         BoP_W, Compression_W, Cooling_W, Unused_PV_W, PV_Excess_W) = simulate_ems_routing(
            P_op_direct, mppt_loss, batt_energy, storage, g, best_n_par,
            T_pem, t_amb, SoH_BATT,
            is_settlement=is_settlement,
            period_h2_direct=pass_h2_direct,
            is_last_step=is_last_step,
            annual_unmet_so_far=annual_h2_unmet,
            pv_scale=pv_scale,
            T_min_C=_T_min_C,
            I_op=I_op_rec,
        )

        period_h2_direct += H2_direct
        if is_settlement or is_last_step:
            period_h2_direct = 0.0

        annual_h2_unmet += H2_unmet

        mosf_idx = int(round((best_n_par - N_PAR_BASE) / N_PAR_STEP))
        SoH_PV, SoH_PEM = calculate_degradation_step(SoH_PV, SoH_PEM, fl, prev_fl,
                                                     prev_mosf, mosf_idx, T_pem_K=T_pem,
                                                     diag=deg_diag,
                                                     T_pem_prev_K=T_pem_prev_step)
        prev_mosf = mosf_idx
        prev_fl   = fl

        grid_export_eur = grid_sold   * DT_HOURS / 1000.0 * sell_price_kwh
        grid_import_eur = grid_bought * DT_HOURS / 1000.0 * buy_price_kwh
        grid_econ_eur   = grid_export_eur - grid_import_eur   # positive = net revenue
        h2_rev_eur    = H2_delivered * H2_PRICE_EUR_KG
        h2_secondary_kg  = overflow if H2_SECONDARY_OFFTAKE else 0.0
        h2_secondary_eur = h2_secondary_kg * H2_SECONDARY_PRICE_EUR_KG
        profit_eur    = h2_rev_eur + grid_econ_eur + h2_secondary_eur

        DEMAND_PER_PERIOD = hourly_H2_target if DEMAND_MODE == 'hourly' else daily_H2_target
        records.append(dict(
            step=i, irradiance=g, temperature=t_amb, price_mwh=cur_price_mwh,
            n_parallel=best_n_par,
            H2_daily_demand=DEMAND_PER_PERIOD if (is_settlement or is_last_step) else 0.0,
            H2_prod=H2_prod, StH=StH, mppt_loss=mppt_loss,
            V_op=V_op_rec, I_op=I_op_rec, P_op_W=V_op_rec * I_op_rec,
            storage=storage, T_pem=T_pem, fl=fl,
            batt_soc=batt_energy / BATT_CAPACITY_WH,
            SoH_PEM=SoH_PEM, SoH_PV=SoH_PV, SoH_BATT=SoH_BATT,
            grid_bought=grid_bought, grid_sold=grid_sold,
            grid_export_eur=grid_export_eur, grid_import_eur=grid_import_eur,
            grid_econ_eur=grid_econ_eur,
            h2_revenue_eur=h2_rev_eur, profit_eur=profit_eur,
            H2_secondary=h2_secondary_kg, h2_secondary_eur=h2_secondary_eur,
            H2_overflow=overflow,
            H2_unmet=H2_unmet, H2_direct=H2_direct, H2_delivered=H2_delivered,
            batt_charge_w=batt_charge_w, batt_discharge_w=batt_discharge_w,
            H2_grid=H2_grid, H2_ren=H2_ren,
            BoP_W=BoP_W, Compression_W=Compression_W, Cooling_W=Cooling_W,
            Unused_PV_W=Unused_PV_W, PV_Excess_W=PV_Excess_W,
            P_curtailed_W=max(0.0, PV_Excess_W - batt_charge_w - grid_sold),
            P_pem_actual=P_pem_actual,
            T_keepwarm_C=_T_min_C,
        ))

    if use_rf:
        print(f"  -> RF Predictions: {prediction_used_count} | Full Optimizations: {full_optimization_count}")

    _dsum = sum(deg_diag.values()) or 1e-30
    print(f"  -> PEM SoH breakdown (pp lost): load={100*deg_diag['load']:.3f}  "
          f"rccd={100*deg_diag['rccd']:.3f}  tcyc={100*deg_diag.get('tcyc', 0.0):.3f}  restart-event={100*deg_diag['cold']:.3f}  "
          f"(load {100*deg_diag['load']/_dsum:.0f}% / rccd {100*deg_diag['rccd']/_dsum:.0f}% "
          f"/ tcyc {100*deg_diag.get('tcyc', 0.0)/_dsum:.0f}% "
          f"/ restart {100*deg_diag['cold']/_dsum:.0f}% of total)")
    print("     restart-event term includes BOTH cold and warm restart SoH charges; "
          "use the cold-start COUNT below to distinguish them.")

    return pd.DataFrame(records)

# =============================================================================
# %% RL ENVIRONMENT
# =============================================================================

class PVPEMEnv(gym.Env):
    MOSFET_OPTIONS   = np.arange(7)
    BASE_PARALLEL    = 250
    PARALLEL_PER_MOS = 25
    N_SERIES_PV      = globals()['N_SERIES_PV']
    N_SERIES_PEM     = N_SER_PEM
    N_PARALLEL_PEM   = 1

    def __init__(self, irradiance_data, temperature_data_C, price_data_mwh,
                 lookahead_steps=LOOKAHEAD_STEPS, eval_mode=False, pv_scale=1.0):
        super().__init__()
        self.irr       = irradiance_data
        self.temp      = temperature_data_C
        self.price     = price_data_mwh
        self.lookahead = int(lookahead_steps)
        if self.lookahead != len(FORECAST_OFFSETS_STEPS):
            raise ValueError(
                f"15-min sensitivity expects {len(FORECAST_OFFSETS_STEPS)} forecast points "
                f"at {FORECAST_HORIZONS_H} h, got lookahead_steps={self.lookahead}."
            )
        self.forecast_offsets = FORECAST_OFFSETS_STEPS
        self.eval_mode = eval_mode
        self.pv_scale  = pv_scale        # legacy; kept at 1.0 (sizing is now via string count)
        self.BASE_PARALLEL    = N_PAR_BASE
        self.PARALLEL_PER_MOS = N_PAR_STEP
        if THERMAL_ACTION:
            _dims = [len(self.MOSFET_OPTIONS), len(KEEPWARM_LEVELS_C)]
            if CHILLER_ACTION:
                _dims.append(len(CHILLER_LEVELS_C))   # 3-D: (config, keep-warm, chiller)
            self.action_space = spaces.MultiDiscrete(_dims)
        else:
            self.action_space = spaces.Discrete(len(self.MOSFET_OPTIONS))

        obs_dim = 14 + (3 * self.lookahead)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        self._G_max, self._T_lo, self._T_hi    = 1200.0, -30.0, 80.0
        self._Tpem_lo = KEEPWARM_LEVELS_C[0] + 273.15
        self._Tpem_hi = max(CHILLER_LEVELS_C) + 273.15
        self._V_max    = 400.0
        self._fc_rng   = np.random.RandomState(FORECAST_SEED)
        self._n_p_max  = float(N_PAR_MAX)
        self.max_steps  = len(irradiance_data)
        self._pem_I     = np.linspace(0, I_PEM(i_cell, self.N_PARALLEL_PEM), 100)
        self._I_max_pem = float(np.max(self._pem_I))
        self._I_max    = self._I_max_pem
        self.reset()

    def _n_par(self, m):
        return self.BASE_PARALLEL + m * self.PARALLEL_PER_MOS

    @staticmethod
    def _c01(x):
        """Clip to [0,1]."""
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

    def _norm_current_state(self, G, T, V, I, n, s, hr, batt_soc, T_pem, price, prev_fl, SoH_BATT):
        _c = self._c01
        return [
            _c(G / self._G_max),
            _c((T - self._T_lo) / (self._T_hi - self._T_lo)),
            _c(V / self._V_max),
            _c(I / self._I_max),
            _c(n / self._n_p_max),
            _c(s / storage_capacity),
            _c(hr / 24.0),
            _c(self.SoH_PV),
            _c(self.SoH_PEM),
            _c(batt_soc),
            _c((T_pem - self._Tpem_lo) / (self._Tpem_hi - self._Tpem_lo)),
            _c((price - (-50)) / (500 - (-50))),
            _c(prev_fl),
            _c(SoH_BATT),   # 14th state
        ]

    def _obs(self):
        hr        = (self.step_count * DT_HOURS) % 24
        batt_soc  = self.batt_energy / BATT_CAPACITY_WH
        safe_step = min(self.step_count, self.max_steps - 1)
        cur_price = self.price[safe_step]

        base_obs = self._norm_current_state(
            self.G, self.T_C, self.V_prev, self.I_prev,
            self._n_par(self.mosf), self.storage, hr,
            batt_soc, self.T_pem, cur_price, self.prev_fl, self.SoH_BATT
        )
        future_obs = []
        for k, step_offset in enumerate(self.forecast_offsets):
            horizon_h = FORECAST_HORIZONS_H[k]
            future_idx = min(self.step_count + step_offset, self.max_steps - 1)
            fG = self.irr[future_idx]
            fT = self.temp[future_idx]
            fP = self.price[future_idx]
            if self.eval_mode:
                fG = forecast_value(fG, self.G,   horizon_h, self._fc_rng)
                fT = forecast_value(fT, self.T_C, horizon_h, self._fc_rng)
                noise_G = noise_T = 0.0
            else:
                noise_span = 0.10 * horizon_h
                noise_G, noise_T = np.random.uniform(-noise_span, noise_span, 2)
            future_obs.extend([
                float(np.clip((fG * (1 + noise_G)) / self._G_max, 0, 1)),
                float(np.clip(((fT * (1 + noise_T)) - self._T_lo) / (self._T_hi - self._T_lo), 0, 1)),
                float(np.clip((fP - (-50)) / (500 - (-50)), 0, 1)),
            ])
        return np.array(base_obs + future_obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._fc_rng         = np.random.RandomState(FORECAST_SEED)
        self.step_count      = 0
        self.G, self.T_C     = float(np.clip(self.irr[0], 0, 1200)), float(self.temp[0])
        self.mosf = self.prev_mosf = 3
        self.V_prev = self.I_prev  = 0.0
        self.prev_fl         = 0.0
        self.storage         = INITIAL_H2_STORAGE
        self.batt_energy     = BATT_CAPACITY_WH * BATT_SOC_MIN
        self.T_pem           = KEEPWARM_BASELINE_C + 273.15   # start at the standby floor
        self.T_min_C         = KEEPWARM_BASELINE_C   # set-points before the first action
        self.T_max_C         = 70.0     # decoded; overwritten every step() when enabled
        self.SoH_BATT        = 1.0
        self.period_h2_direct  = 0.0
        self.annual_h2_unmet  = 0.0
        self._deg_diag        = {'load': 0.0, 'rccd': 0.0, 'cold': 0.0, 'tcyc': 0.0}

        self.annual_grid_import_eur = 0.0
        self.annual_grid_export_eur = 0.0
        self.annual_o2_revenue = 0.0
        self.annual_h2_delivered = 0.0
        self.annual_h2_secondary_kg  = 0.0
        self.annual_h2_secondary_eur = 0.0
        self.total_switches = 0
        self.initial_soh_pem = 1.0
        self.initial_soh_batt = 1.0

        if self.eval_mode:
            self.SoH_PV, self.SoH_PEM, self.SoH_BATT = 1.0, 1.0, 1.0
        else:
            self.SoH_PV   = np.random.uniform(0.97, 1.0)
            self.SoH_PEM  = np.random.uniform(0.95, 1.0)
            self.SoH_BATT = np.random.uniform(0.95, 1.0)

        self.initial_soh_pem = self.SoH_PEM
        self.initial_soh_batt = self.SoH_BATT
        return self._obs(), {}

    def step(self, action):
        cur_G, cur_T, cur_price_mwh = self.G, self.T_C, self.price[self.step_count]
        self.prev_mosf = self.mosf
        if THERMAL_ACTION:
            _a = np.asarray(action).ravel()
            mosf_action  = int(_a[0])
            self.T_min_C = float(KEEPWARM_LEVELS_C[int(_a[1])])
            self.T_max_C = float(CHILLER_LEVELS_C[int(_a[2])]) if CHILLER_ACTION else 70.0
        else:
            mosf_action  = int(action)
            self.T_min_C = KEEPWARM_BASELINE_C
            self.T_max_C = 70.0
        if self.T_max_C < self.T_min_C:
            self.T_max_C = self.T_min_C
        requested_mosf = self.prev_mosf if cur_G < 80 else int(self.MOSFET_OPTIONS[mosf_action])
        self.mosf = requested_mosf

        buy_price_kwh  = cur_price_mwh / 1000.0
        sell_price_kwh = buy_price_kwh * 0.8
        n_par = self._n_par(self.mosf)

        P_op_direct = mppt_loss = V_op = I_op = 0.0
        valid = False

        deg_Iph_ref   = Iph_ref * self.SoH_PV * self.pv_scale
        dV_deg        = pem_deg_overvoltage(self.SoH_PEM)
        dynamic_pem_V = V_PEM(self.T_pem, p_cat, p_an, delta_mem, A_cell_pem,
                              a_an, a_cat, self._pem_I, i_0_an, i_0_cat,
                              T_ref_pem, i_lim, self.N_SERIES_PEM,
                              dV_cell=dV_deg)

        if cur_G >= 80:
            params_t = [deg_Iph_ref, Is_ref1, Is_ref2, Is_refi, Rs, Rsh,
                        n1, n2, ni, rho_contact, A_cell, aSi_thickness, TCO_resistivity, cur_T]
            eff_G = calculate_effective_irradiance(cur_G, 0.1 * cur_G)
            vols, curs, _, _ = perturb_observe_series(params_t, eff_G, self.N_SERIES_PV, n_par)

            if vols:
                V_mpp, I_mpp, _ = find_mppt(vols, curs)
                pt = find_intersection_point(vols, curs,
                                             dynamic_pem_V.tolist(), self._pem_I.tolist())

                while pt is None and self.mosf > 0:
                    self.mosf -= 1
                    n_par = self._n_par(self.mosf)
                    vols, curs, _, _ = perturb_observe_series(params_t, eff_G,
                                                              self.N_SERIES_PV, n_par)
                    if not vols:
                        break
                    V_mpp, I_mpp, _ = find_mppt(vols, curs)
                    pt = find_intersection_point(vols, curs,
                                                 dynamic_pem_V.tolist(),
                                                 self._pem_I.tolist())

                if pt:
                    V_op, I_op = pt
                    valid       = True
                    P_op_direct = V_op * I_op
                    mppt_loss   = mppt_shortfall(V_mpp * I_mpp, P_op_direct)

        is_last_step  = (self.step_count == self.max_steps - 1)
        _settlement_steps = STEPS_PER_HOUR if DEMAND_MODE == 'hourly' else STEPS_PER_DAY
        is_settlement = ((self.step_count + 1) % _settlement_steps == 0)
        pass_h2_direct = self.period_h2_direct
        T_pem_prev_step = self.T_pem
        fl_prev_step    = self.prev_fl
        (self.batt_energy, self.storage, grid_bought, grid_sold,
         batt_charge_w, batt_discharge_w,
         H2_prod, StH, fl, P_pem_actual,
         grid_penalty, overflow,
         H2_base, H2_grid, H2_ren,
         self.T_pem, self.SoH_BATT, H2_unmet,
         H2_direct, H2_delivered,
         BoP_W, Compression_W, Cooling_W, Unused_PV_W, PV_Excess_W) = simulate_ems_routing(
            P_op_direct, mppt_loss, self.batt_energy, self.storage, cur_G, n_par,
            self.T_pem, cur_T, self.SoH_BATT,
            is_settlement=is_settlement,
            period_h2_direct=pass_h2_direct,
            is_last_step=is_last_step,
            annual_unmet_so_far=self.annual_h2_unmet,
            T_min_C=self.T_min_C,
            T_max_C=self.T_max_C,
            pv_scale=self.pv_scale,
            I_op=I_op,
        )

        self.annual_h2_unmet += H2_unmet
        self.period_h2_direct += H2_direct
        if is_settlement or is_last_step:
            self.period_h2_direct = 0.0

        self.SoH_PV, self.SoH_PEM = calculate_degradation_step(
            self.SoH_PV, self.SoH_PEM, fl, self.prev_fl, self.prev_mosf, self.mosf,
            T_pem_K=self.T_pem, diag=self._deg_diag,
            T_pem_prev_K=T_pem_prev_step
        )
        self.prev_fl = fl

        grid_export_eur = grid_sold   * DT_HOURS / 1000.0 * sell_price_kwh
        grid_import_eur = grid_bought * DT_HOURS / 1000.0 * buy_price_kwh
        grid_econ_eur   = grid_export_eur - grid_import_eur   # positive = net revenue
        h2_rev_eur    = H2_delivered * H2_PRICE_EUR_KG
        h2_secondary_kg  = overflow if H2_SECONDARY_OFFTAKE else 0.0
        h2_secondary_eur = h2_secondary_kg * H2_SECONDARY_PRICE_EUR_KG
        profit_eur    = h2_rev_eur + grid_econ_eur + h2_secondary_eur


        # --- MOO multi-objective (same normalisation constants as run_baselines) ---
        moo_obj = DT_HOURS * (
              W_H2   * (H2_prod / (H2_PEAK_KGH * DT_HOURS))
            + W_STH  * (StH / 0.185)
            - W_MPPT * (mppt_loss / MPPT_NORM)
        )

        mosf_diff = abs(self.mosf - self.prev_mosf)
        switching_penalty = LAMBDA_SWITCH * mosf_diff * DT_HOURS

        # --- PEM-underutilisation penalty (fixes low-n_par camping) -----------------
        n_par_max        = self._n_par(len(self.MOSFET_OPTIONS) - 1)   # site-specific top rung
        n_unused_frac    = (n_par_max - n_par) / n_par_max
        underuse_penalty = 0.0
        if cur_G >= 80.0 and valid and fl < 0.98:
            underuse_penalty = DT_HOURS * W_UNDERUSE * n_unused_frac

        # --- Keep-warm energy penalty (thermal-action extension) --------------------
        keepwarm_penalty = 0.0
        if THERMAL_ACTION and fl < 0.05:
            keepwarm_penalty = DT_HOURS * W_KEEPWARM * (Cooling_W / KEEPWARM_NORM_W)

        # --- Chiller-energy penalty (chiller-action extension) ----------------------
        chiller_penalty = 0.0
        if THERMAL_ACTION and CHILLER_ACTION and fl >= 0.05:
            chiller_penalty = DT_HOURS * W_CHILLER * (Cooling_W / KEEPWARM_NORM_W)

        # --- Cold-start event penalty ------------------------------------------------
        coldstart_penalty = 0.0
        if COLDSTART_REWARD and fl >= 0.05 and fl_prev_step < 0.05:
            T_pre_C = T_pem_prev_step - 273.15
            if T_pre_C < COLDSTART_THRESHOLD_C:
                coldstart_penalty = W_COLDSTART * COLDSTART_EXCESS_DAMAGE_FRAC

        # --- Thermal-cycling penalty (thermal-action extension) ---------------------
        tcycle_penalty = 0.0
        if THERMAL_ACTION:
            dT_excess_C = max(0.0, abs(self.T_pem - T_pem_prev_step)
                              - TCYC_DEADBAND_K)
            tcycle_penalty = W_TCYCLE * (dT_excess_C / TCYCLE_NORM_C)

        # --- Steady cold-operation penalty (see W_WARMOPS above) --------------------
        warmops_penalty = 0.0
        if THERMAL_ACTION and fl >= 0.05:
            coldrun_depth_C = max(0.0, T_DEG_REF_K - self.T_pem)
            hotrun_depth_C  = max(0.0, self.T_pem - T_DEG_REF_K)
            warmops_penalty = DT_HOURS * W_WARMOPS * (
                (coldrun_depth_C + hotrun_depth_C) / TCYCLE_NORM_C)

        reward = (moo_obj - switching_penalty - underuse_penalty
                  - keepwarm_penalty - chiller_penalty - tcycle_penalty
                  - coldstart_penalty - warmops_penalty)
        reward = float(np.clip(reward, -2.0, 1.0))

        # --- Book-keeping for annual reporting (does NOT influence the reward) ------
        self.annual_grid_import_eur += grid_import_eur
        self.annual_grid_export_eur += grid_export_eur
        self.annual_o2_revenue      += H2_delivered * O2_KG_PER_KG_H2 * O2_PRICE_EUR_KG
        self.annual_h2_delivered    += H2_delivered
        self.annual_h2_secondary_kg  += h2_secondary_kg
        self.annual_h2_secondary_eur += h2_secondary_eur
        if mosf_diff > 0:
            self.total_switches += 1

        self.V_prev, self.I_prev = V_op, I_op
        self.step_count += 1
        done = self.step_count >= self.max_steps
        if not done:
            self.G   = float(np.clip(self.irr[self.step_count], 0, 1200))
            self.T_C = float(self.temp[self.step_count])

        _DEMAND_PER_PERIOD = hourly_H2_target if DEMAND_MODE == 'hourly' else daily_H2_target
        info = dict(
            H2_prod=H2_prod, StH=StH, mppt_loss=mppt_loss, n_parallel=n_par,
            storage=self.storage, I_op=I_op, V_op=V_op, P_op_W=V_op * I_op,
            SoH_PEM=self.SoH_PEM, SoH_PV=self.SoH_PV, SoH_BATT=self.SoH_BATT,
            batt_soc=self.batt_energy / BATT_CAPACITY_WH,
            H2_daily_demand=_DEMAND_PER_PERIOD if (is_settlement or is_last_step) else 0.0,
            T_pem=self.T_pem, grid_bought=grid_bought, grid_sold=grid_sold,
            grid_export_eur=grid_export_eur, grid_import_eur=grid_import_eur,
            grid_econ_eur=grid_econ_eur, P_pem_actual=P_pem_actual,
            batt_charge_w=batt_charge_w, batt_discharge_w=batt_discharge_w,
            H2_unmet=H2_unmet, H2_direct=H2_direct, H2_delivered=H2_delivered,
            H2_secondary=h2_secondary_kg, h2_secondary_eur=h2_secondary_eur,
            H2_overflow=overflow,
            h2_revenue_eur=h2_rev_eur, profit_eur=profit_eur,
            H2_grid=H2_grid, H2_ren=H2_ren,
            BoP_W=BoP_W, Compression_W=Compression_W, Cooling_W=Cooling_W,
            Unused_PV_W=Unused_PV_W, PV_Excess_W=PV_Excess_W,
            P_curtailed_W=max(0.0, PV_Excess_W - batt_charge_w - grid_sold),
            T_keepwarm_C=self.T_min_C, T_chiller_C=self.T_max_C,
        )
        return self._obs(), reward, False, done, info

def solve_cyclic_storage(irr, temp, price, guess=0.0, iters=None, tol=None, verbose=True):
    """Solve for the tank level the year reproduces: iterate S_end = f(S_start)."""
    global INITIAL_H2_STORAGE
    iters = TANK_CYCLIC_ITERS  if iters is None else iters
    tol   = TANK_CYCLIC_TOL_KG if tol   is None else tol
    saved, s, hist = INITIAL_H2_STORAGE, float(guess), []
    try:
        for k in range(iters):
            INITIAL_H2_STORAGE = s
            df = run_baselines(irr, temp, price, use_rf=False)
            s_end = float(df["storage"].iloc[-1])
            hist.append((s, s_end))
            if verbose:
                print(f"    cyclic pass {k+1}: start {s:8.1f} -> end {s_end:8.1f} kg "
                      f"(gap {s_end - s:+8.1f})")
            done = abs(s_end - s) <= tol
            s = s_end
            if done:
                break
        else:
            if verbose:
                print(f"    [warning] not converged in {iters} passes; using {s:.1f} kg")
    except Exception:
        INITIAL_H2_STORAGE = saved
        raise
    INITIAL_H2_STORAGE = s
    return s, hist


def apply_cyclic_storage(irr, temp, price, loc_label):
    """Set INITIAL_H2_STORAGE for this site according to TANK_INIT_MODE."""
    global INITIAL_H2_STORAGE
    if TANK_INIT_MODE != 'cyclic':
        set_initial_storage()
        print(f"  TANK START ({TANK_INIT_MODE}): {INITIAL_H2_STORAGE:.0f} kg of "
              f"{storage_capacity:.0f} kg. Repaid by year end or charged as unmet demand.")
        return INITIAL_H2_STORAGE
    cached = SITE_CYCLIC_S0.get(loc_label)
    if cached is None:
        site = loc_label.split("_")[0]
        for k, v in SITE_CYCLIC_S0.items():
            if k.split("_")[0] == site:
                cached = v
                break
    if cached is not None:
        INITIAL_H2_STORAGE = float(cached)
        print(f"  CYCLIC TANK: cached S0 = {INITIAL_H2_STORAGE:.1f} kg")
        return INITIAL_H2_STORAGE
    print(f"  CYCLIC TANK: solving the self-consistent January-1 level for {loc_label}")
    s0, _ = solve_cyclic_storage(irr, temp, price)
    print(f"  CYCLIC TANK: S0 = {s0:.1f} kg ({100*s0/storage_capacity:.0f}% of tank). "
          f"Cache as  SITE_CYCLIC_S0['{loc_label}'] = {s0:.1f}")
    return s0


# =============================================================================
# %% RL TRAINING & EVALUATION
# =============================================================================

class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rewards = []; self.ep_h2 = []; self.ep_sth = []
        self.ep_mppt = [];    self.ep_prof = []
        self.ep_count = 0
        self._reset_buffers()

    def _reset_buffers(self):
        self._r = []; self._h2 = []; self._sth = []
        self._mppt = []; self._prof = []; self._bc = []; self._bd = []

    def _on_step(self):
        info = self.locals.get('infos', [{}])[0]
        rw   = float(self.locals['rewards'][0])
        if not (np.isnan(rw) or np.isinf(rw)):
            self._r.append(rw)
            self._h2.append(info.get('H2_prod', 0))
            self._sth.append(info.get('StH', 0))
            self._mppt.append(info.get('mppt_loss', 0))
            self._prof.append(info.get('profit_eur', 0))
            self._bc.append(info.get('batt_charge_w', 0))
            self._bd.append(info.get('batt_discharge_w', 0))

        if self.locals.get('dones', [False])[0] and self._r:
            self.ep_count += 1
            ep_r    = np.sum(self._r)
            ep_h2   = np.sum(self._h2)
            v       = [s for s in self._sth if s > 0]
            ep_sth  = np.mean(v) if v else 0
            ep_mppt = np.sum(self._mppt) * DT_HOURS / 1000
            ep_prof = np.sum(self._prof)
            ep_bc   = np.sum(self._bc) * DT_HOURS / 1000
            ep_bd   = np.sum(self._bd) * DT_HOURS / 1000

            self.ep_rewards.append(ep_r)
            self.ep_h2.append(ep_h2)
            self.ep_sth.append(ep_sth)
            self.ep_mppt.append(ep_mppt)
            self.ep_prof.append(ep_prof)

            print(f"Ep {self.ep_count:02d} | Rwd: {ep_r:7.1f} | H2: {ep_h2:5.1f} kg | "
                  f"StH: {ep_sth*100:4.1f}% | "
                  f"MPP short: {ep_mppt:5.1f} kWh | Batt {ep_bc:.1f}/{ep_bd:.1f} kWh | "
                  f"Profit: EUR {ep_prof:6.0f}")
            self._reset_buffers()
        return True


def _plot_training_curves(cb, period_id, plots_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    metrics = [
        (cb.ep_rewards,               'Episode Reward',                    'steelblue'),
        (cb.ep_h2,                    'H2 Production / Episode (kg)',      'seagreen'),
        ([s * 100 for s in cb.ep_sth],'Mean StH Efficiency / Episode (%)', 'darkorange'),
        (cb.ep_mppt,                  'MPP Shortfall / Episode (kWh)',     'firebrick'),
    ]
    for ax, (data, title, color) in zip(axes.flatten(), metrics):
        if not data:
            continue
        eps = range(len(data))
        ax.plot(eps, data, alpha=0.3, color=color, linewidth=1.5)
        win = max(1, len(data) // 10)
        ax.plot(eps, pd.Series(data).rolling(win, min_periods=1).mean(), color=color, linewidth=2.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode / Simulated Year')
        ax.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.96])   # leave headroom so the suptitle never overlaps
    out_path = Path(plots_dir) / f'rl_training_curves_{period_id}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Training curves saved: {out_path}")


def train_rl(irr_data, temp_data_C, price_data_mwh, period_id, model_dir, plots_dir, pv_scale=1.0,
             seed=None):
    seed = RL_SEED if seed is None else seed
    np.random.seed(seed)
    env = PVPEMEnv(irr_data, temp_data_C, price_data_mwh, lookahead_steps=LOOKAHEAD_STEPS, pv_scale=pv_scale)
    model = PPO(
        'MlpPolicy', env, seed=seed,
        learning_rate=RL_CONFIG['learning_rate'],
        gamma=RL_CONFIG['gamma'],
        gae_lambda=RL_CONFIG['gae_lambda'],
        n_steps=RL_CONFIG['n_steps'],
        batch_size=RL_CONFIG['batch_size'],
        n_epochs=RL_CONFIG['n_epochs'],
        ent_coef=RL_CONFIG['ent_coef'],
        clip_range=RL_CONFIG['clip_range'],
        max_grad_norm=RL_CONFIG['max_grad_norm'],
        verbose=0,
        policy_kwargs=dict(net_arch=dict(pi=RL_CONFIG['net_arch'], vf=RL_CONFIG['net_arch']))
    )
    cb_track = TrainingCallback()
    cb_ckpt  = CheckpointCallback(
        save_freq=RL_CONFIG['checkpoint_freq'],
        save_path=str(model_dir),
        name_prefix=f'rl_ckpt_{period_id}'
    )
    print(f'\n--- Starting RL Training ({RL_TIMESTEPS:,} steps, seed={seed}) ---')
    model.learn(total_timesteps=RL_TIMESTEPS, callback=[cb_track, cb_ckpt], progress_bar=False)
    model_path = Path(model_dir) / f'rl_model_{period_id}'
    model.save(str(model_path))
    print(f"  -> Model saved: {model_path}.zip")

    if cb_track.ep_rewards:
        _plot_training_curves(cb_track, period_id, plots_dir)
    return model


# =============================================================================
# %% SEED-VARIANCE STUDY  — run-to-run spread of the PPO policy
# =============================================================================
def _as_seeds(seeds):
    """Accept an int (-> 0..n-1), a single-value, or any iterable; return a tuple.
    """
    if isinstance(seeds, (int, np.integer)):
        return tuple(range(int(seeds)))
    return tuple(seeds)


def _seed_summarize_run(df):
    """Headline metrics from one evaluate_rl_single_run() DataFrame."""
    n = pd.to_numeric(df['n_parallel']).to_numpy()
    switches = int(np.count_nonzero(np.diff(n) != 0))
    h2_del   = float(df['H2_delivered'].sum())
    h2_unmet = float(df['H2_unmet'].sum())
    pem_proj = project_pem_stack_replacements(df, stack_capex=1.0, replacement_frac=1.0)
    top = float(N_PAR_MAX)

    import io as _io, contextlib as _cl
    pv_peak_kw = N_PAR_MAX * N_SERIES_PV * PV_MODULE_W / 1000.0
    with _cl.redirect_stdout(_io.StringIO()):
        _eco = calculate_lcoh(df, None, df, pv_peak_kw)['RL Agent']

    return {
        'switches'      : switches,
        'H2_prod_kg'    : float(df['H2_prod'].sum()),
        'final_SoH_PEM' : float(pd.to_numeric(df['SoH_PEM']).iloc[-1] * 100.0),
        'stack_life_yr' : float(pem_proj['projected_stack_life_yr']),
        'replacements'  : int(pem_proj['stack_replacements']),
        'grid_MWh'      : float(df['grid_bought'].sum() * DT_HOURS / 1.0e6),
        'profit_kEUR'   : float(df['profit_eur'].sum() / 1000.0),
        'demand_met_pct': float(100.0 * h2_del / max(h2_del + h2_unmet, 1e-9)),
        'mean_npar'     : float(np.mean(n)),
        'top_rung_pct'  : float(100.0 * np.mean(np.isclose(n, top))),
        'unused_MWh'    : float(pd.to_numeric(df.get('Unused_PV_W', 0.0), errors='coerce').fillna(0.0).sum() * DT_HOURS / 1.0e6) if 'Unused_PV_W' in df else 0.0,
        'curtail_MWh'   : float(pd.to_numeric(df.get('P_curtailed_W', 0.0), errors='coerce').fillna(0.0).sum() * DT_HOURS / 1.0e6) if 'P_curtailed_W' in df else 0.0,
        'StH_pct'       : float(_v.mean() * 100.0) if len(_v := pd.to_numeric(
                              df['StH'])[pd.to_numeric(df['StH']) > 0]) else 0.0,
        'LCOH_EURkg'     : float(_eco['lcoh']),
        'LCSH_EURkg'     : float(_eco['lcsh']),
        'REPEX_kEURyr'   : float(_eco['pem_repex_ann'] / 1000.0),
    }


def _seed_train_one(irr, temp, price, seed):
    """Train a fresh PPO agent with a fixed seed (mirrors train_rl config).

    np.random.seed() matters MORE here than in train_rl. PPO's own seed= covers its internal
    RNG only; the env's domain randomisation (initial SoH draws) and training-time forecast
    noise call np.random.uniform() against the GLOBAL numpy stream. Without re-seeding it, that
    stream simply carries on from wherever the previous training left it, so corner N's
    randomisation depends on how many runs preceded it. Two corners nominally at "seed 0" would
    then see different noise, and the between-corner differences this study exists to measure
    would be contaminated by run ORDER. Re-seeding makes every (corner, seed) cell see an
    identical stream, which is the whole basis of the comparison.
    """
    np.random.seed(seed)
    env = PVPEMEnv(irr, temp, price, lookahead_steps=LOOKAHEAD_STEPS)
    model = PPO(
        'MlpPolicy', env, seed=seed,
        learning_rate=RL_CONFIG['learning_rate'],
        gamma=RL_CONFIG['gamma'],
        gae_lambda=RL_CONFIG['gae_lambda'],
        n_steps=RL_CONFIG['n_steps'],
        batch_size=RL_CONFIG['batch_size'],
        n_epochs=RL_CONFIG['n_epochs'],
        ent_coef=RL_CONFIG['ent_coef'],
        clip_range=RL_CONFIG['clip_range'],
        max_grad_norm=RL_CONFIG['max_grad_norm'],
        verbose=0,
        policy_kwargs=dict(net_arch=dict(pi=RL_CONFIG['net_arch'],
                                         vf=RL_CONFIG['net_arch'])),
    )
    model.learn(total_timesteps=RL_TIMESTEPS, progress_bar=False)
    return model


def seed_variance_study(irr, temp, price, period_id, plots_dir, seeds=(0, 1, 2, 3, 4)):
    """Train across seeds, evaluate each frozen policy, print & save mean +/- std.
    """
    seeds = _as_seeds(seeds)
    rows = []
    for s in seeds:
        print(f"\n=====  {period_id}  |  SEED {s}  =====")
        model = _seed_train_one(irr, temp, price, s)
        df    = evaluate_rl_single_run(model, irr, temp, price)
        m     = _seed_summarize_run(df); m['seed'] = s
        rows.append(m)
        print("   ", {k: round(v, 2) for k, v in m.items() if k != 'seed'})

    res = pd.DataFrame(rows).set_index('seed')
    print(f"\n============  SEED-VARIANCE SUMMARY  ({period_id}, n={len(seeds)})  ============")
    print(f"  {'metric':<16}{'mean':>12}{'std':>12}{'CV %':>9}")
    for col in res.columns:
        mu, sd = res[col].mean(), res[col].std(ddof=1)
        cv = 100.0 * sd / max(abs(mu), 1e-9)
        print(f"  {col:<16}{mu:>12.2f}{sd:>12.2f}{cv:>9.1f}")

    out = Path(plots_dir) / f'seed_variance_{period_id}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out)
    print(f"\n  -> per-seed metrics saved: {out}")
    return res


def pareto_switching_study(irr, temp, price, period_id, plots_dir,
                           lambdas=(0.0, 0.10, 0.18, 0.30, 0.50), seeds=(0, 1, 2),
                           zero_underuse=False):
    """Sweep the switching-fatigue weight LAMBDA_SWITCH and trace the
    switching<->performance Pareto front.
    """
    import matplotlib.pyplot as plt
    global LAMBDA_SWITCH, W_UNDERUSE
    seeds = _as_seeds(seeds)
    _saved_lambda   = LAMBDA_SWITCH
    _saved_underuse = W_UNDERUSE
    if zero_underuse:
        W_UNDERUSE = 0.0
        period_id  = f'{period_id}_noUnderuse'
        print("  [Pareto ablation] W_UNDERUSE set to 0 for this sweep.")
    rows = []
    try:
        for lam in lambdas:
            LAMBDA_SWITCH = float(lam)          # env reads this module global in the reward
            for s in seeds:
                print(f"\n=====  {period_id}  |  LAMBDA_SWITCH = {lam}  |  seed {s}  =====")
                model = _seed_train_one(irr, temp, price, s)
                df    = evaluate_rl_single_run(model, irr, temp, price)
                m     = _seed_summarize_run(df); m['lambda_sw'] = lam; m['seed'] = s
                rows.append(m)
                print("   ", {k: round(v, 2) for k, v in m.items() if k not in ('lambda_sw', 'seed')})
    finally:
        LAMBDA_SWITCH = _saved_lambda           # always restore the paper's values
        W_UNDERUSE    = _saved_underuse

    raw = pd.DataFrame(rows)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    raw.to_csv(Path(plots_dir) / f'pareto_switching_{period_id}_raw.csv', index=False)

    grp = raw.groupby('lambda_sw')
    med = grp.median(numeric_only=True)
    sd  = grp.std(numeric_only=True).fillna(0.0)
    med.to_csv(Path(plots_dir) / f'pareto_switching_{period_id}_median.csv')

    print(f"\n============  PARETO MEDIAN over {len(seeds)} seeds ({period_id})  ============")
    print(f"  {'lambda':>7}{'switches':>12}{'H2_kg':>12}{'profit_kEUR':>13}{'grid_MWh':>11}{'SoH_%':>9}")
    for lam in med.index:
        print(f"  {lam:>7.2f}{med.loc[lam,'switches']:>12.0f}{med.loc[lam,'H2_prod_kg']:>12.0f}"
              f"{med.loc[lam,'profit_kEUR']:>13.1f}{med.loc[lam,'grid_MWh']:>11.1f}{med.loc[lam,'final_SoH_PEM']:>9.2f}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].errorbar(med['switches'], med['H2_prod_kg'] / 1000.0,
                    xerr=sd['switches'], yerr=sd['H2_prod_kg'] / 1000.0,
                    fmt='o-', capsize=3, color='tab:blue')
    axs[1].errorbar(med['switches'], med['profit_kEUR'],
                    xerr=sd['switches'], yerr=sd['profit_kEUR'],
                    fmt='s-', capsize=3, color='tab:green')
    for lam in med.index:
        axs[0].annotate(f'  λ={lam:g}', (med.loc[lam, 'switches'], med.loc[lam, 'H2_prod_kg'] / 1000.0), fontsize=8)
        axs[1].annotate(f'  λ={lam:g}', (med.loc[lam, 'switches'], med.loc[lam, 'profit_kEUR']),          fontsize=8)
    axs[0].set_xlabel('MOSFET switches / yr'); axs[0].set_ylabel('H$_2$ (t/yr)')
    axs[0].set_title('Switching vs yield')
    axs[1].set_xlabel('MOSFET switches / yr'); axs[1].set_ylabel('Operating profit (k€)')
    axs[1].set_title('Switching vs profit')
    for a in axs: a.grid(alpha=0.3)
    fig.suptitle(f'Switching–performance Pareto front ({period_id}); median of {len(seeds)} seeds, labels = λ$_{{sw}}$')
    fig.tight_layout()
    figout = Path(plots_dir) / f'pareto_switching_{period_id}.png'
    fig.savefig(figout, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"\n  -> raw CSV:       pareto_switching_{period_id}_raw.csv"
          f"\n  -> median CSV:    pareto_switching_{period_id}_median.csv"
          f"\n  -> Pareto figure: {figout}")
    return med


def reward_weight_study(irr, temp, price, period_id, plots_dir,
                        corners=None, seeds=None, timesteps=None):
    """Robustness of the DRL result to the reward SHAPING weights."""
    import matplotlib.pyplot as plt
    global LAMBDA_SWITCH, W_UNDERUSE, W_COLDSTART, W_TCYCLE, W_CHILLER, W_WARMOPS, RL_TIMESTEPS

    seeds   = _as_seeds(REWARD_SEEDS if seeds is None else seeds)
    corners = REWARD_CORNERS if corners is None else corners
    budget  = timesteps or REWARD_STUDY_TIMESTEPS or RL_TIMESTEPS

    def _corner(c):
        if len(c) == 7:
            return c
        if len(c) == 6:
            lam, wu, wcs, wtc, wch, label = c
            return (lam, wu, wcs, wtc, wch, W_WARMOPS, label)
        raise ValueError("Reward corner must have 6 (legacy) or 7 fields; got %r" % (c,))

    corners = [_corner(c) for c in corners]

    saved = (LAMBDA_SWITCH, W_UNDERUSE, W_COLDSTART, W_TCYCLE, W_CHILLER, W_WARMOPS, RL_TIMESTEPS)
    saved_ppo = {k: RL_CONFIG[k] for k in REWARD_STUDY_PPO if k in RL_CONFIG}
    rows  = []
    traj_cols = ['step', 'irradiance', 'n_parallel', 'T_pem', 'H2_prod']
    trajs = []
    print("\n" + "=" * 100)
    print("REWARD-WEIGHT ROBUSTNESS STUDY - %s" % period_id)
    print("=" * 100)
    n_train = sum(1 if c[6] in REWARD_SINGLE_SEED else len(seeds) for c in corners)
    print("  %d configurations, %d seed(s) except %s at 1 -> %d trainings"
          % (len(corners), len(seeds),
             "/".join(sorted(REWARD_SINGLE_SEED)) or "none", n_train))
    print("  %s steps each (%.0f simulated years)"
          % (format(int(budget), ","), budget / STEPS_PER_YEAR))
    print("  Read the SPAN of each metric across a weight's levels, not the best row.")
    try:
        RL_TIMESTEPS = int(budget)
        RL_CONFIG.update(REWARD_STUDY_PPO)
        if REWARD_STUDY_PPO:
            print("  sweep-only PPO overrides: %s  (reference values restored afterwards)"
                  % ", ".join("%s %s->%s" % (k, saved_ppo.get(k), v)
                              for k, v in REWARD_STUDY_PPO.items()))
        for lam, wu, wcs, wtc, wch, wwo, label in corners:
            LAMBDA_SWITCH = float(lam)
            W_UNDERUSE    = float(wu)
            W_COLDSTART   = float(wcs)
            W_TCYCLE      = float(wtc)
            W_CHILLER     = float(wch)
            W_WARMOPS     = float(wwo)
            cfg_seeds = seeds[:1] if label in REWARD_SINGLE_SEED else seeds
            for s in cfg_seeds:
                print("\n=====  %s | %s  (lam=%s, w_u=%s, w_cs=%s, w_tc=%s, w_ch=%s, w_wo=%s) | seed %s of %d  ====="
                      % (period_id, label, lam, wu, wcs, wtc, wch, wwo, s, len(cfg_seeds)))
                model = _seed_train_one(irr, temp, price, s)
                df    = evaluate_rl_single_run(model, irr, temp, price)
                mm    = _seed_summarize_run(df)
                mm["cold_starts"]    = count_cold_starts(df)
                mm["mean_restart_T"] = mean_restart_temperature(df)
                mm.update(lambda_sw=lam, w_u=wu, w_coldstart=wcs, w_tcycle=wtc,
                          w_chiller=wch, w_warmops=wwo, label=label, seed=s)
                rows.append(mm)
                _tr = df[[c for c in traj_cols if c in df.columns]].copy()
                _tr['label'] = label
                _tr['seed']  = s
                trajs.append(_tr)
                print("   ", {k: round(v, 2) for k, v in mm.items()
                              if isinstance(v, (int, float))
                              and k not in ("lambda_sw", "w_u", "w_coldstart",
                                            "w_tcycle", "w_chiller", "w_warmops", "seed")})
                Path(plots_dir).mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(
                    Path(plots_dir) / ("reward_weights_%s_raw.csv" % period_id), index=False)
                print("      [saved %d/%d runs -> reward_weights_%s_raw.csv]"
                      % (len(rows), n_train, period_id))
    finally:
        (LAMBDA_SWITCH, W_UNDERUSE, W_COLDSTART,
         W_TCYCLE, W_CHILLER, W_WARMOPS, RL_TIMESTEPS) = saved
        RL_CONFIG.update(saved_ppo)
        if rows:
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(
                Path(plots_dir) / ("reward_weights_%s_raw.csv" % period_id), index=False)

    raw = pd.DataFrame(rows)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    raw.to_csv(Path(plots_dir) / ("reward_weights_%s_raw.csv" % period_id), index=False)
    if trajs:
        tpath = Path(plots_dir) / ("reward_weights_%s_trajectories.csv.gz" % period_id)
        pd.concat(trajs, ignore_index=True).to_csv(tpath, index=False,
                                                  compression='gzip')
        print("  -> trajectories: %s  (re-score any threshold metric from this)" % tpath)
    keys = ["label", "lambda_sw", "w_u", "w_coldstart", "w_tcycle", "w_chiller", "w_warmops"]
    med  = raw.groupby(keys, sort=False).median(numeric_only=True).reset_index()
    med.to_csv(Path(plots_dir) / ("reward_weights_%s_median.csv" % period_id), index=False)

    METRICS = [("switches", "switch/yr"), ("cold_starts", "cold/yr"),
               ("mean_restart_T", "restartT C"),
               ("H2_prod_kg", "H2 kg"), ("grid_MWh", "grid MWh"),
               ("final_SoH_PEM", "SoH %"), ("stack_life_yr", "life yr"),
               ("replacements", "repl/30y"), ("demand_met_pct", "demand %"),
               ("mean_npar", "mean npar"), ("top_rung_pct", "top rung %"),
               ("unused_MWh", "unused MWh"), ("curtail_MWh", "curt MWh"),
               ("LCOH_EURkg", "LCOH E/kg"), ("LCSH_EURkg", "LCSH E/kg"),
               ("REPEX_kEURyr", "REPEX kE/y")]

    print("\n" + "=" * 100)
    print("PER-CONFIGURATION RESULTS (%s, median of %d seed(s))" % (period_id, len(seeds)))
    print("=" * 100)
    hdr = "  %-10s%6s%6s%6s%6s" % ("config", "lam", "w_u", "w_cs", "w_wo")
    for _, sh in METRICS:
        hdr += "%11s" % sh
    print(hdr)
    for _, r in med.iterrows():
        line = "  %-10s%6.2f%6.2f%6.2f%6.2f" % (r["label"], r["lambda_sw"], r["w_u"], r["w_coldstart"], r["w_warmops"])
        for col, _ in METRICS:
            line += "%11.1f" % r[col]
        print(line)

    # --- the actual deliverable: how much does each swept weight move each metric? -----
    SPANS = [("lambda_sw", ["lam12", "lam13", "lam14", "lam15"], "LAMBDA_SWITCH")]
    print("\n" + "=" * 100)
    print("ROBUSTNESS SPANS - the reportable result")
    print("=" * 100)
    print("  For each weight, the min-max range of every metric over its tested levels.")
    print("  A NARROW span means the conclusion is a property of the controller.")
    print("  A WIDE span means the conclusion is a property of the weight - say so.\n")
    for _, labels_, name in SPANS:
        sub = med[med["label"].isin(labels_)]
        missing = [l for l in labels_ if l not in set(med["label"])]
        if missing:
            print("  !! %s: level(s) %s absent from this run -- the span below covers only %s "
                  "and UNDERSTATES the true range. Do not quote it as the full span."
                  % (name, "/".join(missing), "/".join(sub["label"])))
        if len(sub) < 2:
            print("  %s: not enough configurations present, skipped" % name)
            continue
        print("  %s  (configs: %s)" % (name, ", ".join(sub["label"])))
        for col, sh in METRICS:
            lo, hi = sub[col].min(), sub[col].max()
            rel = (hi - lo) / abs(lo) * 100 if abs(lo) > 1e-9 else float("nan")
            print("     %-12s %10.1f .. %10.1f   span %9.1f  (%5.1f %% of low)"
                  % (sh, lo, hi, hi - lo, rel))
        print()

    # --- publication figures for the final lambda confirmation -----------------------
    labels_all = med["label"].tolist()

    summary_rows = []
    for label_, g in raw.groupby("label", sort=False):
        rr = {"label": label_, "lambda_sw": float(g["lambda_sw"].iloc[0])}
        for col, _ in METRICS:
            rr[col + "_median"] = float(g[col].median())
            rr[col + "_min"]    = float(g[col].min())
            rr[col + "_max"]    = float(g[col].max())
            rr[col + "_std"]    = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        summary_rows.append(rr)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(Path(plots_dir) / ("reward_lambda_%s_summary.csv" % period_id), index=False)

    CONF = [("H2_prod_kg", "H2 (kg/yr)"), ("switches", "switches/yr"),
            ("cold_starts", "cold starts/yr"), ("mean_restart_T", "restart T (C)"),
            ("replacements", "stack repl./30y"), ("LCOH_EURkg", "LCOH (EUR/kg)")]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs = np.asarray(axs).ravel()
    for ax, (col, title_) in zip(axs, CONF):
        for seed_, g in raw.groupby("seed"):
            gg = g.sort_values("lambda_sw")
            ax.scatter(gg["lambda_sw"], gg[col], s=55, label="seed %s" % seed_)
        mm = raw.groupby("lambda_sw")[col].median().sort_index()
        ax.plot(mm.index, mm.values, "--o", lw=2, label="median")
        ax.set_xlabel("lambda_sw")
        ax.set_ylabel(title_)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle("Final switching-weight confirmation - %s (3 seeds, %.0f training years)"
                 % (period_id, budget / STEPS_PER_YEAR))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    calout = Path(plots_dir) / ("reward_calibration_%s.png" % period_id)
    fig.savefig(calout, dpi=180, bbox_inches="tight")
    plt.close(fig)

    SHOW = [("H2_prod_kg", "H2 kg"), ("switches", "switch/yr"),
            ("cold_starts", "cold/yr"), ("mean_restart_T", "restart T C"),
            ("final_SoH_PEM", "SoH %"), ("stack_life_yr", "life yr"),
            ("replacements", "repl/30y"), ("grid_MWh", "grid MWh"),
            ("LCOH_EURkg", "LCOH E/kg"), ("LCSH_EURkg", "LCSH E/kg")]
    ncols = 5
    nrows = int(np.ceil(len(SHOW) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(18, 3.7 * nrows))
    axs = np.atleast_1d(axs).ravel()
    x = np.arange(len(labels_all))
    for ax in axs[len(SHOW):]:
        ax.axis("off")
    for ax, (col, title_) in zip(axs, SHOW):
        vals = [raw.loc[raw["label"] == lab, col] for lab in labels_all]
        meds = np.array([v.median() for v in vals], float)
        lo   = np.array([v.min() for v in vals], float)
        hi   = np.array([v.max() for v in vals], float)
        ax.bar(x, meds, yerr=np.vstack([meds-lo, hi-meds]), capsize=4)
        for i, v in enumerate(vals):
            jitter = np.linspace(-0.08, 0.08, len(v))
            ax.scatter(np.full(len(v), i) + jitter, v, s=30, zorder=3)
        ax.set_title(title_, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_all, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Final lambda robustness - %s (median and seed range)" % period_id)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    figout = Path(plots_dir) / ("reward_weights_%s.png" % period_id)
    fig.savefig(figout, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("  -> raw CSV:       reward_weights_%s_raw.csv" % period_id)
    print("  -> median CSV:    reward_weights_%s_median.csv" % period_id)
    print("  -> summary CSV:   reward_lambda_%s_summary.csv" % period_id)
    print("  -> calibration:   %s" % calout)
    print("  -> weights plot:  %s" % figout)
    return med


def load_saved_rl_model(period_id, model_dir):
    """Returns the PPO model, or None if the file is absent so the caller can fall back
    to train_rl().
    """
    model_path = Path(model_dir) / f'rl_model_{period_id}.zip'
    if not model_path.exists():
        return None
    print(f"  -> Reusing saved RL model (no retrain): {model_path}")
    return PPO.load(str(model_path))


def evaluate_rl(model, irr_data, temp_data_C, price_data_mwh, period_id, data_dir, plots_dir, pv_scale=1.0):
    env = PVPEMEnv(irr_data, temp_data_C, price_data_mwh, eval_mode=True, pv_scale=pv_scale)
    obs, _ = env.reset()
    records = []

    print("\nEvaluating trained RL agent over dataset...")
    for step in tqdm(range(len(irr_data)), desc="RL Evaluation"):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, done, info = env.step(action)
        records.append(dict(
            step=step,
            irradiance=float(np.clip(irr_data[step], 0, 1200)),
            temperature=float(temp_data_C[step]),
            price_mwh=env.price[step],
            n_parallel=info['n_parallel'],
            V_op=info.get('V_op', 0.0),
            I_op=info.get('I_op', 0.0),
            P_op_W=info.get('P_op_W', 0.0),
            P_pem_actual=info['P_pem_actual'],
            H2_daily_demand=info.get('H2_daily_demand', 0.0),
            H2_prod=info['H2_prod'],
            StH=info['StH'],
            mppt_loss=info['mppt_loss'],
            T_pem=info['T_pem'],
            storage=info['storage'],
            SoH_PEM=info['SoH_PEM'],
            SoH_PV=info['SoH_PV'],
            SoH_BATT=info['SoH_BATT'],
            batt_soc=info['batt_soc'],
            grid_bought=info['grid_bought'],
            grid_sold=info['grid_sold'],
            grid_export_eur=info['grid_export_eur'],
            grid_import_eur=info['grid_import_eur'],
            grid_econ_eur=info['grid_econ_eur'],
            h2_revenue_eur=info['h2_revenue_eur'],
            profit_eur=info['profit_eur'],
            H2_unmet=info['H2_unmet'],
            H2_direct=info['H2_direct'],
            H2_delivered=info['H2_delivered'],
            H2_secondary=info['H2_secondary'],
            h2_secondary_eur=info['h2_secondary_eur'],
            H2_overflow=info['H2_overflow'],
            T_keepwarm_C=info.get('T_keepwarm_C', np.nan),
            T_chiller_C=info.get('T_chiller_C', np.nan),
            batt_charge_w=info['batt_charge_w'],
            batt_discharge_w=info['batt_discharge_w'],
            H2_grid=info['H2_grid'],
            H2_ren=info['H2_ren'],
            BoP_W=info['BoP_W'],
            Compression_W=info['Compression_W'],
            Cooling_W=info['Cooling_W'],
            Unused_PV_W=info['Unused_PV_W'],
            PV_Excess_W=info['PV_Excess_W'],
            P_curtailed_W=info['P_curtailed_W'],
        ))
        if done:
            break

    df = pd.DataFrame(records)
    out_csv = Path(data_dir) / f'rl_results_{period_id}.csv'
    df.to_csv(out_csv, index=False)
    print(f"  -> RL results saved: {out_csv}")

    _dd = env._deg_diag
    _dsum = sum(_dd.values()) or 1e-30
    print(f"  -> PEM SoH breakdown (pp lost): load={100*_dd['load']:.3f}  "
          f"rccd={100*_dd['rccd']:.3f}  tcyc={100*_dd.get('tcyc', 0.0):.3f}  cold-restart={100*_dd['cold']:.3f}  "
          f"(load {100*_dd['load']/_dsum:.0f}% / rccd {100*_dd['rccd']/_dsum:.0f}% "
          f"/ tcyc {100*_dd.get('tcyc', 0.0)/_dsum:.0f}% "
          f"/ cold {100*_dd['cold']/_dsum:.0f}% of total)")

    return df

# =============================================================================
# %% PLOTTING & EXCEL EXPORT
# =============================================================================

def _save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved: {path}")


def plot_polarization_curves(plots_dir):
    """PV array - PEM I-V characterisation, one figure per site."""
    plots_dir = Path(plots_dir)
    T_pem_K   = 60.0 + 273.15

    pem_I = np.linspace(0.0, float(I_PEM(i_cell, 1)), 400)
    pem_V = np.asarray(V_PEM(T_pem_K, p_cat, p_an, delta_mem, A_cell_pem,
                             a_an, a_cat, pem_I, i_0_an, i_0_cat,
                             T_ref_pem, i_lim, N_SER_PEM), float)
    i_max_pem = float(pem_I.max())
    V_at_max  = float(pem_V[int(np.argmax(pem_I))])

    params_stc = [Iph_ref, Is_ref1, Is_ref2, Is_refi, Rs, Rsh,
                  n1, n2, ni, rho_contact, A_cell, aSi_thickness,
                  TCO_resistivity, 25.0]

    def _pv(G, n_par):
        eff = calculate_effective_irradiance(G, 0.1 * G)
        v, c, _, _ = perturb_observe_series(list(params_stc), eff, N_SERIES_PV,
                                            n_par, v_step=1, v_max=70.0)
        return v, c

    def _op(vols, curs):
        """Operating point under the same rule the controllers use."""
        if not vols:
            return None
        pt = find_intersection_point(vols, curs, pem_V.tolist(), pem_I.tolist())
        return ("cross", pt[0], pt[1]) if pt else ("none", V_at_max, i_max_pem)

    for site, ratio in [(s, SITE_SIZING[s][0]) for s in SITE_SIZING]:
        base, step, n_max = site_string_config(ratio)
        ladder = [base + k * step for k in range(7)]
        label  = site.replace('_2021', '')

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ymax = [0.0, 0.0]

        for k, (ax, series, pal, ttl) in enumerate([
            (axes[0], [(G, n_max) for G in (200, 400, 600, 800, 1000)],
             plt.cm.YlOrRd(np.linspace(0.30, 1.0, 5)),
             f'Varying irradiance  ($n_{{par}}$ = {n_max}, full array)'),
            (axes[1], [(600, n) for n in ladder],
             plt.cm.Blues(np.linspace(0.32, 1.0, 7)),
             f'Seven MOSFET switch positions  (G = 600 W/m$^2$, '
             f'{ladder[0]}-{ladder[-1]}, step {step})'),
        ]):
            ax.plot(pem_V, pem_I, color='#222222', lw=2.5, ls='--',
                    label='PEM load curve', zorder=4)
            ax.axhline(i_max_pem, color='grey', lw=0.9, ls=':',
                       label=f'Nominal = 1 MW rating ({i_max_pem:.0f} A, 3 A cm$^{{-2}}$)')
            for idx, ((G, n_par), col) in enumerate(zip(series, pal)):
                vols, curs = _pv(G, n_par)
                if not vols:
                    continue
                lab = (f'G = {G} W/m$^2$' if k == 0 else
                       f'$n_{{par}}$ = {n_par}' + (' (max)' if n_par == n_max else ''))
                ax.plot(vols, curs, color=col, lw=1.9, label=lab)
                ymax[k] = max(ymax[k], float(np.max(curs)))
                Vm, Im, _ = find_mppt(vols, curs)
                ax.scatter([Vm], [Im], color=col, s=48, zorder=6,
                           edgecolors='k', linewidths=0.6)
                op = _op(vols, curs)
                if op is None:
                    continue
                kind, Vop, Iop = op
                if kind == 'cross':
                    ax.scatter([Vop], [Iop], color=col, marker='*', s=180,
                               zorder=7, edgecolors='k', linewidths=0.5)
                    ax.annotate(f'{Vop:.0f} V', xy=(Vop, Iop),
                                xytext=(9, 9 * (idx - len(series) // 2)),
                                textcoords='offset points', fontsize=7,
                                color=col, fontweight='bold',
                                arrowprops=dict(arrowstyle='->', color=col, lw=0.7,
                                                shrinkA=0, shrinkB=3),
                                va='center', ha='left', zorder=12)
                else:
                    ax.scatter([Vop], [Iop], facecolors='none', edgecolors=col,
                               marker='o', s=110, lw=1.6, zorder=7)
            ax.set_title(ttl, fontweight='bold', fontsize=11)

        from matplotlib.lines import Line2D
        proxies = [
            Line2D([], [], ls='none', marker='*', ms=11, color='0.35', mec='k',
                   mew=0.5, label='operating point (curves cross)'),
            Line2D([], [], ls='none', marker='o', ms=9, mfc='none', mec='0.35',
                   mew=1.4, label='no operating point (over 3 A cm$^{-2}$)'),
            Line2D([], [], ls='none', marker='o', ms=7, color='0.35', mec='k',
                   mew=0.5, label='array MPP'),
        ]
        for k, ax in enumerate(axes):
            ax.set_xlabel('Voltage  (V)')
            ax.set_ylabel('Current  (A)')
            ax.set_xlim(100, 330)
            ax.set_ylim(0, max(ymax[k], i_max_pem) * 1.10)
            ax.grid(alpha=0.3)
            h, l = ax.get_legend_handles_labels()
            ax.legend(h + proxies, l + [q.get_label() for q in proxies],
                      fontsize=7, loc='lower left', framealpha=0.92)

        fig.suptitle(f'{label} - direct-coupled operating point '
                     f'(PV oversizing {ratio:.2f}x, stack at 60 C)',
                     fontweight='bold', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save_fig(fig, plots_dir / f'char_00_pv_pem_polarization_{label}.png')

    print("  -> PV-PEM polarization curves saved (one figure per site).")


# --- Shared controller styling (used by every cross-controller plot/print) ------
CTRL_ORDER  = ['MOO', 'MOO+Hyst', 'MOO+PreWarm', 'MOO+Hyst+PreWarm',
               'MOO+AlwaysWarm', 'MOO+RF', 'RL Agent']
CTRL_COLORS = {'MOO': '#4477AA', 'MOO+Hyst': '#AA3377', 'MOO+PreWarm': '#CCBB44',
               'MOO+Hyst+PreWarm': '#66CCEE', 'MOO+AlwaysWarm': '#EE7733',
               'MOO+RF': '#EE6677', 'RL Agent': '#228833'}
CTRL_HATCH  = {'MOO': '', 'MOO+Hyst': '..', 'MOO+PreWarm': '--',
               'MOO+Hyst+PreWarm': 'xx', 'MOO+AlwaysWarm': '\\',
               'MOO+RF': '//', 'RL Agent': 'oo'}


def _ctrl_dfs(df_moo, df_rf, df_rl, df_hyst=None, df_pen=None, df_pw=None, df_hpw=None):
    """Ordered {label: df} for the controllers actually present."""
    d = {'MOO': df_moo}
    if df_hyst is not None:
        d['MOO+Hyst'] = df_hyst
    if df_pw is not None:
        d['MOO+PreWarm'] = df_pw
    if df_hpw is not None:
        d['MOO+Hyst+PreWarm'] = df_hpw
    if df_pen is not None:
        d['MOO+AlwaysWarm'] = df_pen
    if df_rf is not None:
        d['MOO+RF'] = df_rf
    d['RL Agent'] = df_rl
    return d


def plot_energy_stacks(df_moo, df_rf, df_rl, period_id, plots_dir, df_hyst=None, df_pen=None,
                       df_pw=None, df_hpw=None):
    """Daily stacked-area energy breakdown, one panel per controller (one panel per controller when the
    hysteresis baseline is present).
    """
    plots_dir = Path(plots_dir)
    _cd    = _ctrl_dfs(df_moo, df_rf, df_rl, df_hyst, df_pen, df_pw, df_hpw)
    labels = list(_cd.keys())
    dfs    = list(_cd.values())
    n_ctrl = len(dfs)

    def _daily_kWh(series_W, n_days):
        arr = np.array(series_W[:n_days * STEPS_PER_DAY], dtype=float) * DT_HOURS / 1000.0
        return arr.reshape(n_days, STEPS_PER_DAY).sum(axis=1)

    def _pem_kWh(df, n_days):
        arr = np.array(df['H2_prod'].values[:n_days * STEPS_PER_DAY], dtype=float)
        return (arr * LHV / (0.65 * 1000.0)).reshape(n_days, STEPS_PER_DAY).sum(axis=1)

    CONS_KEYS   = ['BoP_W', 'Compression_W', 'Cooling_W', 'batt_charge_w']
    CONS_LABELS = ['BoP', 'Compression', 'Cooling', 'Battery charge']
    CONS_COLS   = ['#4878d0', '#6acc65', '#d65f5f', '#ee854a', '#956cb4', '#8c613c']

    fig_a, axes_a = plt.subplots(n_ctrl, 1, figsize=(13, 3.0 * n_ctrl), sharex=True)
    for ax, df, lbl in zip(axes_a, dfs, labels):
        n = (len(df) // STEPS_PER_DAY) * STEPS_PER_DAY
        n_days = n // STEPS_PER_DAY
        day_x  = np.arange(n_days)

        stacks = [_pem_kWh(df, n_days)]
        slabels = ['PEM input']
        scols   = [CONS_COLS[0]]
        for key, slbl, scol in zip(CONS_KEYS, CONS_LABELS, CONS_COLS[1:]):
            if key in df.columns:
                stacks.append(_daily_kWh(df[key], n_days))
                slabels.append(slbl); scols.append(scol)

        bottom = np.zeros(n_days)
        for s, sl, sc in zip(stacks, slabels, scols):
            ax.bar(day_x, s, bottom=bottom, color=sc, label=sl,
                   width=1.0, linewidth=0)
            bottom += s

        ax.set_ylabel('kWh / day'); ax.set_title(lbl, fontweight='bold')
        ax.grid(axis='y', alpha=0.25); ax.set_xlim(-0.5, n_days - 0.5)

    axes_a[0].legend(loc='upper left', fontsize=8, ncol=3, framealpha=0.9)
    axes_a[-1].set_xlabel('Day of year')
    plt.tight_layout()
    _save_fig(fig_a, plots_dir / f'comp_7_energy_consumers_{period_id}.png')

    fig_b, axes_b = plt.subplots(n_ctrl, 1, figsize=(13, 3.0 * n_ctrl), sharex=True)
    for ax, df, lbl in zip(axes_b, dfs, labels):
        n = (len(df) // STEPS_PER_DAY) * STEPS_PER_DAY
        n_days = n // STEPS_PER_DAY
        day_x  = np.arange(n_days)

        grid_daily = _daily_kWh(df['grid_bought'], n_days)
        pem_daily  = _pem_kWh(df, n_days)

        total_daily = pem_daily.copy()
        for key in ['BoP_W', 'Compression_W', 'Cooling_W', 'batt_charge_w']:
            if key in df.columns:
                total_daily += _daily_kWh(df[key], n_days)
        total_daily = np.maximum(total_daily, 1e-6)

        solar_pct = np.clip((total_daily - grid_daily) / total_daily * 100, 0, 100)
        grid_pct  = np.clip(grid_daily / total_daily * 100, 0, 100)

        ax.bar(day_x, solar_pct, color='#f4a261', label='Solar (PV)',
               width=1.0, linewidth=0)
        ax.bar(day_x, grid_pct, bottom=solar_pct, color='#e63946',
               label='Grid import', width=1.0, linewidth=0)
        ax.set_ylim(0, 108); ax.set_ylabel('% of daily energy')
        ax.set_title(lbl, fontweight='bold')
        ax.grid(axis='y', alpha=0.25); ax.set_xlim(-0.5, n_days - 0.5)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    axes_b[-1].set_xlabel('Day of year')
    plt.tight_layout()
    _save_fig(fig_b, plots_dir / f'comp_8_energy_sources_{period_id}.png')
    print("  -> Energy stacked-area plots saved.")


def plot_3way_comparison(df_moo, df_rf, df_rl, period_id, plots_dir, df_hyst=None, df_pen=None,
                         df_pw=None, df_hpw=None, pv_scale=1.0):
    """Per-location comparison plots."""
    plots_dir = Path(plots_dir)
    ctrls = [('MOO', df_moo, '#4477AA', '-')]
    if df_hyst is not None:
        ctrls.append(('MOO+Hyst', df_hyst, '#AA3377', ':'))
    if df_pw is not None:
        ctrls.append(('MOO+PreWarm', df_pw, '#CCBB44', (0, (5, 2))))
    if df_hpw is not None:
        ctrls.append(('MOO+Hyst+PreWarm', df_hpw, '#66CCEE', (0, (5, 1, 1, 1))))
    if df_pen is not None:
        ctrls.append(('MOO+AlwaysWarm', df_pen, '#EE7733', (0, (3, 1, 1, 1))))
    if df_rf is not None:
        ctrls.append(('MOO+RF', df_rf, '#EE6677', '-.'))
    ctrls.append(('DRL', df_rl, '#228833', '--'))
    _t = lambda df: df['step'].values * DT_HOURS / 24.0

    def _line_fig(col, ylabel, fname, scale=1.0, cumulative=False, label_fmt=None):
        fig, ax = plt.subplots(figsize=(10, 4))
        for lbl, df, color, ls in ctrls:
            y = df[col].cumsum() if cumulative else df[col]
            y = y * scale
            lab = label_fmt(lbl, y) if label_fmt else lbl
            lw  = 2.2 if lbl == 'DRL' else 1.8
            ax.plot(_t(df), y, linestyle=ls, color=color, lw=lw, label=lab, alpha=0.85)
        ax.set_ylabel(ylabel); ax.set_xlabel('Day of year')
        ax.legend(fontsize=10); ax.grid(alpha=0.3); plt.tight_layout()
        _save_fig(fig, plots_dir / f'{fname}_{period_id}.png')

    fig, ax = plt.subplots(figsize=(10, 4))
    for lbl, df, color, ls in ctrls:
        ax.step(_t(df), df['n_parallel'], where='post', color=color, lw=1.5, label=lbl, alpha=0.8, ls=ls)
    npmin = min(df['n_parallel'].min() for _, df, _, _ in ctrls)
    npmax = max(df['n_parallel'].max() for _, df, _, _ in ctrls)
    pad   = max(3.0, 0.10 * (npmax - npmin)); ax.set_ylim(npmin - pad, npmax + pad)
    ax.set_ylabel('n_parallel (MOSFET position)'); ax.set_xlabel('Day of year')
    _W = N_SERIES_PV * PV_MODULE_W                                   # W per string
    _kwp = lambda n: n * _W / 1000.0                                 # connected PV (kWp)
    ax2 = ax.secondary_yaxis('right', functions=(_kwp, lambda p: p * 1000.0 / _W))
    ax2.set_ylabel('Connected PV (kWp)')
    ax.set_title(f"{period_id.replace('_2021','')} — {N_PAR_MAX} strings installed "
                 f"= {_kwp(N_PAR_MAX):.0f} kWp ({_kwp(N_PAR_MAX)/(PEM_capacity/1000.0):.2f}x PEM)",
                 fontsize=11)
    ax.legend(fontsize=11); ax.grid(alpha=0.3); plt.tight_layout()
    _save_fig(fig, plots_dir / f'comp_1_strings_{len(ctrls)}way_{period_id}.png')

    _line_fig('H2_prod',    'kg',       f'comp_2_cumulative_h2_{len(ctrls)}way',   cumulative=True,
              label_fmt=lambda l, y: f'{l} ({y.iloc[-1]:.0f} kg)')
    _line_fig('SoH_PEM',    'Health %', f'comp_3_soh_pem_{len(ctrls)}way',   scale=100.0)
    _line_fig('SoH_BATT',   'Health %', f'comp_4_soh_batt_{len(ctrls)}way',  scale=100.0)
    _line_fig('profit_eur', 'EUR',      f'comp_5_cumulative_profit_{len(ctrls)}way', cumulative=True,
              label_fmt=lambda l, y: f'{l} (EUR {y.iloc[-1]:.0f})')
    _line_fig('H2_unmet',   'kg',       f'comp_6_unmet_h2_{len(ctrls)}way',  cumulative=True)
    print(f"\nAll comparison plots saved to: {plots_dir}")


def export_3way_excel(df_moo, df_rf, df_rl, period_id, data_dir, df_hyst=None, df_pen=None,
                      df_pw=None, df_hpw=None):
    """One sheet per controller (4 when the hysteresis MOO baseline was run)."""
    ctrls = _ctrl_dfs(df_moo, df_rf, df_rl, df_hyst, df_pen, df_pw, df_hpw)
    out_path = Path(data_dir) / f'Comparison_{len(ctrls)}Way_{period_id}.xlsx'
    print(f"  Exporting multi-sheet Excel: {out_path}")
    _sheet = {'MOO': 'MOO', 'MOO+Hyst': 'MOO_Hyst', 'MOO+PreWarm': 'MOO_PreWarm',
              'MOO+Hyst+PreWarm': 'MOO_Hyst_PreWarm', 'MOO+AlwaysWarm': 'MOO_AlwaysWarm',
              'MOO+RF': 'MOO_RF', 'RL Agent': 'RL'}
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for lbl, d in ctrls.items():
            d.to_excel(writer, sheet_name=_sheet.get(lbl, lbl)[:31], index=False)
    print(f"  -> Excel saved: {out_path}")


def print_summary(df_moo, df_rf, df_rl, period_id, df_hyst=None, df_pen=None,
                  df_pw=None, df_hpw=None):
    """Detailed per-location results table, over the controllers present (
    when the hysteresis MOO baseline was run).
    """
    ctrls = _ctrl_dfs(df_moo, df_rf, df_rl, df_hyst, df_pen, df_pw, df_hpw)   # ordered {label: df}
    labels = list(ctrls.keys())

    def _safe_pct(grid_col, prod_col, df):
        return (df[grid_col].sum() / max(1e-9, df[prod_col].sum())) * 100

    def _mean_sth_pct(df):
        v = df['StH'][df['StH'] > 0]
        return float(v.mean() * 100.0) if len(v) else 0.0

    def _batt_used_kwh(df):
        return float(df['batt_soc'].max() - df['batt_soc'].min()) * BATT_CAPACITY_WH / 1000.0
    def _batt_cycles(df):
        thru = (df['batt_charge_w'].sum() + df['batt_discharge_w'].sum()) * DT_HOURS
        return thru / 2.0 / BATT_CAPACITY_WH

    def _h2_metrics(df):
        demand    = df['H2_daily_demand'].sum()
        eff_unmet = df['H2_unmet'].sum()
        deficit   = max(0.0, INITIAL_H2_STORAGE - float(df['storage'].iloc[-1]))
        return demand, demand - eff_unmet, eff_unmet, deficit

    rows = [
        ('Total H2 Produced (kg)',    lambda d: d['H2_prod'].sum()),
        ('Total H2 Demand (kg)',      lambda d: _h2_metrics(d)[0]),
        ('H2 Delivered (kg)',         lambda d: _h2_metrics(d)[1]),
        ('  incl. tank deficit(kg)',  lambda d: _h2_metrics(d)[3]),
        ('Unmet H2 (kg)',             lambda d: _h2_metrics(d)[2]),
        ('H2 from Renewables (%)',    lambda d: 100 - _safe_pct('H2_grid', 'H2_prod', d)),
        ('H2 from Grid (%)',          lambda d: _safe_pct('H2_grid', 'H2_prod', d)),
        ('Hardware Switches',         lambda d: float((d['n_parallel'].diff().fillna(0) != 0).sum())),
        ('Final PEM Health (%)',      lambda d: d['SoH_PEM'].iloc[-1] * 100),
        ('Final PV Health (%)',       lambda d: d['SoH_PV'].iloc[-1] * 100),
        ('Final Battery Health (%)',  lambda d: d['SoH_BATT'].iloc[-1] * 100),
        (f'Battery Used (kWh /{BATT_CAPACITY_WH/1000:.0f})', _batt_used_kwh),
        ('Battery Cycles (/yr)',      _batt_cycles),
        ('Final H2 Storage (kg)',     lambda d: d['storage'].iloc[-1]),
        ('Mean StH Efficiency (%)',   _mean_sth_pct),
        ('MPP Shortfall (kWh)',       lambda d: d['mppt_loss'].sum() * DT_HOURS / 1000),
        ('Grid Import (kWh)',         lambda d: d['grid_bought'].sum() * DT_HOURS / 1000),
        ('Grid Export (kWh)',         lambda d: d['grid_sold'].sum() * DT_HOURS / 1000),
        ('Curtailed PV (kWh)',        lambda d: d['P_curtailed_W'].sum() * DT_HOURS / 1000),
        ('Total Profit (EUR)',        lambda d: d['profit_eur'].sum()),
    ]

    W = 32 + 17 * len(labels)
    print(f"\n{'='*W}")
    print(f"{len(labels)}-WAY RESULTS SUMMARY ({period_id})")
    print(f"{'='*W}")
    print(f"  {'Metric':<30}" + "".join(f" | {l:<14}" for l in labels))
    print(f"  {'-'*(W-2)}")
    for name, fn in rows:
        print(f"  {name:<30}" + "".join(f" | {fn(ctrls[l]):<14.2f}" for l in labels))
    print(f"{'='*W}")


def count_cold_starts(df, threshold_C=None):
    """Number of production starts made from a COLD stack."""
    thr = COLDSTART_THRESHOLD_C if threshold_C is None else threshold_C
    if 'T_pem' not in df.columns or 'H2_prod' not in df.columns:
        return 0
    producing = (df['H2_prod'].values > 1e-9)
    was_idle  = np.r_[True, ~producing[:-1]]          # idle on the previous step
    T_prev_C  = np.r_[np.nan, df['T_pem'].values[:-1]] - 273.15
    cold      = np.r_[False, T_prev_C[1:] < thr]
    return int((producing & was_idle & cold).sum())


def restart_temperatures(df):
    """Stack temperature (deg C) at every production restart."""
    if 'T_pem' not in df.columns or 'H2_prod' not in df.columns:
        return np.array([])
    producing = (df['H2_prod'].values > 1e-9)
    was_idle  = np.r_[True, ~producing[:-1]]
    T_prev_C  = np.r_[np.nan, df['T_pem'].values[:-1]] - 273.15
    sel = producing & was_idle & np.isfinite(T_prev_C)
    return T_prev_C[sel]


def mean_restart_temperature(df):
    """Mean stack temperature at production restart, NaN if it never restarts."""
    r = restart_temperatures(df)
    return float(np.mean(r)) if r.size else float('nan')


def _controller_headline(df):
    """Headline metrics for one controller df (schema-agnostic; works for greedy MOO,
    hysteresis MOO, MOO+RF and DRL alike).
    """
    sw = int((df['n_parallel'].diff().fillna(0) != 0).sum())
    cold_starts = count_cold_starts(df)
    restart_T   = mean_restart_temperature(df)
    demand   = df['H2_daily_demand'].sum()
    eff_unmet= df['H2_unmet'].sum()      # already includes the cyclic deficit
    eff_del  = demand - eff_unmet
    dmet     = 100.0 * eff_del / max(1e-9, demand)
    sth_v    = df['StH'][df['StH'] > 0]
    sth      = sth_v.mean() * 100.0 if len(sth_v) > 0 else 0.0

    grid_mwh = df['grid_bought'].sum() * DT_HOURS / 1e6
    profit_k = df['profit_eur'].sum() / 1000.0
    h2_prod  = df['H2_prod'].sum()
    h2_sec   = df['H2_secondary'].sum() if 'H2_secondary' in df.columns else 0.0
    batt_thru= (df['batt_charge_w'].sum() + df['batt_discharge_w'].sum()) * DT_HOURS
    batt_cyc = batt_thru / 2.0 / BATT_CAPACITY_WH
    mppt_kwh = df['mppt_loss'].sum() * DT_HOURS / 1000.0
    curt_kwh = df['P_curtailed_W'].sum() * DT_HOURS / 1000.0
    grid_kwh = df['grid_bought'].sum() * DT_HOURS / 1000.0
    load_cols = [c for c in ('P_pem_actual', 'BoP_W', 'Compression_W', 'Cooling_W') if c in df.columns]
    total_load_kwh = sum(df[c].sum() for c in load_cols) * DT_HOURS / 1000.0 if load_cols else 0.0
    self_suff = 100.0 * (1.0 - grid_kwh / total_load_kwh) if total_load_kwh > 0 else np.nan

    return dict(sw=sw, cold=cold_starts, restartT=restart_T,
                soh_pem=df['SoH_PEM'].iloc[-1]*100, soh_pv=df['SoH_PV'].iloc[-1]*100,
                soh_batt=df['SoH_BATT'].iloc[-1]*100, batt_cyc=batt_cyc,
                h2_prod=h2_prod, h2_del=eff_del, unmet=eff_unmet, dmet=dmet, sth=sth,
                mppt_kwh=mppt_kwh, curt_kwh=curt_kwh, grid_mwh=grid_mwh,
                self_suff=self_suff, profit_k=profit_k, h2_sec=h2_sec)


def print_controller_comparison(controllers, period_id):
    """Compact side-by-side headline table for an ordered dict {label: df}."""
    labels = list(controllers.keys())
    M = {lbl: _controller_headline(df) for lbl, df in controllers.items()}
    rows = [
        ('Switches / yr',            'sw',       '{:>10.0f}'),
        ('Cold starts / yr',         'cold',     '{:>10.0f}'),
        ('Mean restart T (C)',       'restartT', '{:>10.1f}'),
        ('Final PEM SoH (%)',        'soh_pem',  '{:>10.3f}'),
        ('Final PV SoH (%)',         'soh_pv',   '{:>10.3f}'),
        ('Final Battery SoH (%)',    'soh_batt', '{:>10.2f}'),
        ('Battery cycles / yr',      'batt_cyc', '{:>10.1f}'),
        ('H2 produced (kg/yr)',      'h2_prod',  '{:>10.0f}'),
        ('H2 delivered (kg/yr)',     'h2_del',   '{:>10.0f}'),
        ('Unmet H2 (kg/yr)',         'unmet',    '{:>10.0f}'),
        ('Demand met (%)',           'dmet',     '{:>10.1f}'),
        ('Mean StH (%)',             'sth',      '{:>10.2f}'),
        ('MPP shortfall (kWh/yr)',   'mppt_kwh', '{:>10.0f}'),
        ('Curtailed PV (kWh/yr)',    'curt_kwh', '{:>10.0f}'),
        ('Grid import (MWh/yr)',     'grid_mwh', '{:>10.1f}'),
        ('Secondary H2 sold (kg)',   'h2_sec',   '{:>10.0f}'),
        ('Operating profit (k€/yr)', 'profit_k', '{:>10.1f}'),
    ]
    w = 26 + 12 * len(labels)
    print(f"\n{'='*w}")
    print(f"CONTROLLER COMPARISON ({len(labels)}-way) — {period_id}")
    print(f"{'='*w}")
    print(f"  {'Metric':<24}" + "".join(f"{l:>12}" for l in labels))
    print(f"  {'-'*(w-2)}")
    for name, key, fmt in rows:
        print(f"  {name:<24}" + "".join(fmt.format(M[l][key]).rjust(12) for l in labels))
    print(f"{'='*w}")
    print("  Note: the hysteresis MOO de-chatters greedy MOO (fewer switches) but a")
    print("  deadband alone does not recover the DRL's PEM state-of-health / load-smoothing gain.")
    return M


def audit_controller_energy_allocation(controllers, period_id):
    """Diagnose WHY controllers differ in H2 without retraining."""
    print(f"\n{'='*118}")
    print(f"CONTROLLER ENERGY-ALLOCATION AUDIT ({period_id})")
    print("  Low MPPT shortfall is not automatically high total-array utilisation; inspect active strings and unused PV together.")
    print(f"{'='*118}")
    print(f"  {'Controller':<20}{'H2 kg':>10}{'npar mean':>11}{'top rung%':>10}{'PEM MWh':>10}"
          f"{'unused MWh':>12}{'curt MWh':>11}{'grid MWh':>11}{'MPPT MWh':>11}")
    print(f"  {'-'*116}")

    for label, df in controllers.items():
        if len(df) == 0:
            continue
        daylight = (pd.to_numeric(df.get('irradiance', 0), errors='coerce').fillna(0.0) >= 80.0)
        dfd = df.loc[daylight] if daylight.any() else df
        npar = pd.to_numeric(dfd['n_parallel'], errors='coerce')
        nmean = float(npar.mean()) if len(npar) else np.nan
        top_pct = 100.0 * float((npar >= (N_PAR_MAX - 0.5 * N_PAR_STEP)).mean()) if len(npar) else np.nan
        h2 = float(pd.to_numeric(df['H2_prod'], errors='coerce').fillna(0.0).sum())
        pem_mwh = float(pd.to_numeric(df.get('P_pem_actual', 0), errors='coerce').fillna(0.0).sum()) * DT_HOURS / 1e6
        unused_mwh = float(pd.to_numeric(df.get('Unused_PV_W', 0), errors='coerce').fillna(0.0).sum()) * DT_HOURS / 1e6
        curt_mwh = float(pd.to_numeric(df.get('P_curtailed_W', 0), errors='coerce').fillna(0.0).sum()) * DT_HOURS / 1e6
        grid_mwh = float(pd.to_numeric(df.get('grid_bought', 0), errors='coerce').fillna(0.0).sum()) * DT_HOURS / 1e6
        mppt_mwh = float(pd.to_numeric(df.get('mppt_loss', 0), errors='coerce').fillna(0.0).sum()) * DT_HOURS / 1e6
        print(f"  {label:<20}{h2:>10.0f}{nmean:>11.1f}{top_pct:>10.1f}{pem_mwh:>10.1f}"
              f"{unused_mwh:>12.1f}{curt_mwh:>11.1f}{grid_mwh:>11.1f}{mppt_mwh:>11.1f}")

    if 'RL Agent' in controllers and 'MOO+Hyst+PreWarm' in controllers:
        r = controllers['RL Agent']; b = controllers['MOO+Hyst+PreWarm']
        dh2 = float(r['H2_prod'].sum() - b['H2_prod'].sum())
        dgrid = (float(r['grid_bought'].sum() - b['grid_bought'].sum()) * DT_HOURS / 1e6)
        dcurt = (float(r['P_curtailed_W'].sum() - b['P_curtailed_W'].sum()) * DT_HOURS / 1e6)
        print(f"\n  RL minus MOO+Hyst+PreWarm: H2 {dh2:+.0f} kg/yr, grid {dgrid:+.1f} MWh/yr, curtailment {dcurt:+.1f} MWh/yr.")

    rl = controllers.get('RL Agent')
    if rl is not None and 'T_chiller_C' in rl.columns:
        vals = pd.to_numeric(rl['T_chiller_C'], errors='coerce').dropna()
        if len(vals):
            dist = ', '.join(f"{t:.0f}C={100.0*(vals==t).mean():.1f}%" for t in CHILLER_LEVELS_C)
            print(f"  RL chiller action occupancy: {dist}")
    if rl is not None and 'T_keepwarm_C' in rl.columns:
        vals = pd.to_numeric(rl['T_keepwarm_C'], errors='coerce').dropna()
        if len(vals):
            dist = ', '.join(f"{t:.0f}C={100.0*(vals==t).mean():.1f}%" for t in KEEPWARM_LEVELS_C)
            print(f"  RL keep-warm action occupancy: {dist}")
    print(f"{'='*118}")


def audit_rl_reward_components(df, period_id=''):
    """Reconstruct the deployed RL reward term-by-term from an evaluation dataframe.
    """
    if df is None or len(df) == 0:
        return None

    h2 = pd.to_numeric(df['H2_prod'], errors='coerce').fillna(0.0).to_numpy(float)
    sth = pd.to_numeric(df['StH'], errors='coerce').fillna(0.0).to_numpy(float)
    mppt = pd.to_numeric(df['mppt_loss'], errors='coerce').fillna(0.0).to_numpy(float)
    npar = pd.to_numeric(df['n_parallel'], errors='coerce').fillna(N_PAR_BASE).to_numpy(float)
    p_pem = pd.to_numeric(df['P_pem_actual'], errors='coerce').fillna(0.0).to_numpy(float)
    cool = pd.to_numeric(df['Cooling_W'], errors='coerce').fillna(0.0).to_numpy(float)
    temp = pd.to_numeric(df['T_pem'], errors='coerce').fillna(KEEPWARM_BASELINE_C + 273.15).to_numpy(float)

    fl = np.clip(p_pem / PEM_capacity, 0.0, 1.0)
    prev_fl = np.r_[0.0, fl[:-1]]
    prev_temp = np.r_[KEEPWARM_BASELINE_C + 273.15, temp[:-1]]
    mosf = np.rint((npar - N_PAR_BASE) / max(N_PAR_STEP, 1)).astype(int)
    prev_mosf = np.r_[mosf[0], mosf[:-1]]

    terms = {}
    terms['+ H2'] = DT_HOURS * W_H2 * (h2 / (H2_PEAK_KGH * DT_HOURS))
    terms['+ StH'] = DT_HOURS * W_STH * (sth / 0.185)
    terms['- MPPT'] = DT_HOURS * W_MPPT * (mppt / MPPT_NORM)
    terms['- switch'] = DT_HOURS * LAMBDA_SWITCH * np.abs(mosf - prev_mosf)
    unused_frac = np.clip((N_PAR_MAX - npar) / max(float(N_PAR_MAX), 1.0), 0.0, 1.0)
    irr = pd.to_numeric(df['irradiance'], errors='coerce').fillna(0.0).to_numpy(float)
    valid_proxy = (irr >= 80.0) & (p_pem > 20.0)
    terms['- underuse'] = np.where(valid_proxy & (fl < 0.98),
                                    DT_HOURS * W_UNDERUSE * unused_frac, 0.0)
    terms['- keepwarm'] = np.where(fl < 0.05, DT_HOURS * W_KEEPWARM * (cool / KEEPWARM_NORM_W), 0.0)
    terms['- chiller'] = np.where((fl >= 0.05) & CHILLER_ACTION,
                                  DT_HOURS * W_CHILLER * (cool / KEEPWARM_NORM_W), 0.0)
    prev_temp_C = prev_temp - 273.15
    is_cold_restart = ((fl >= 0.05) & (prev_fl < 0.05) &
                       (prev_temp_C < COLDSTART_THRESHOLD_C) & COLDSTART_REWARD)
    terms['- coldstart'] = np.where(
        is_cold_restart,
        W_COLDSTART * COLDSTART_EXCESS_DAMAGE_FRAC,
        0.0,
    )
    terms['- tcycle'] = (W_TCYCLE * (np.abs(temp - prev_temp) / TCYCLE_NORM_C)
                         if THERMAL_ACTION else np.zeros_like(fl))
    coldrun = np.maximum(0.0, T_DEG_REF_K - temp)
    terms['- warmops'] = np.where((fl >= 0.05) & THERMAL_ACTION,
                                  DT_HOURS * W_WARMOPS * (coldrun / TCYCLE_NORM_C), 0.0)

    raw = terms['+ H2'] + terms['+ StH']
    for k in ('- MPPT', '- switch', '- underuse', '- keepwarm', '- chiller',
              '- coldstart', '- tcycle', '- warmops'):
        raw = raw - terms[k]
    clipped = np.clip(raw, -2.0, 1.0)

    print(f"\n{'='*88}")
    print(f"RL REWARD AUDIT ({period_id})")
    print("  Cumulative contribution over the evaluation year; current weights must match training.")
    print(f"{'='*88}")
    for k, arr in terms.items():
        sign = 1.0 if k.startswith('+') else -1.0
        signed_total = sign * float(np.sum(arr))
        active_steps = int(np.count_nonzero(arr > 1e-12))
        print(f"  {k:<14} total={signed_total:>11.1f}   mean/step={signed_total/len(df):>10.5f}   active={active_steps:>5d}")
    print(f"  {'raw reward':<14} total={float(raw.sum()):>11.1f}   mean/step={float(raw.mean()):>10.5f}")
    print(f"  {'clipped reward':<14} total={float(clipped.sum()):>11.1f}   mean/step={float(clipped.mean()):>10.5f}")
    print(f"{'='*88}")
    return {k: float(v.sum()) for k, v in terms.items()}

def calculate_lcoh(df_moo, df_rf, df_rl, pv_peak_kw, df_hyst=None, df_pen=None,
                   df_pw=None, df_hpw=None):
    """Levelized Cost of Hydrogen (EUR/kg) for each controller."""
    def crf(r, n):
        return r * (1 + r)**n / ((1 + r)**n - 1)

    pem_kw           = PEM_capacity / 1000.0
    batt_kw          = BATT_MAX_P_W  / 1000.0
    batt_kwh         = BATT_CAPACITY_WH / 1000.0
    h2_peak_rate_kgh = H2_PEAK_KGH

    # ---- CAPEX (EUR) ----
    capex_pv        = pv_peak_kw              * LCOH_PV_EUR_KWP
    capex_pem       = pem_kw                  * LCOH_PEM_EUR_KW
    capex_batt      = batt_kwh                * LCOH_BATT_EUR_KWH
    capex_batt_conv = batt_kw                 * LCOH_BATT_CONV_EUR_KW
    capex_grid_inv  = LCOH_GRID_INV_KW       * LCOH_GRID_INV_EUR_KW
    capex_tank      = LCOH_TANK_KG_INSTALLED * LCOH_TANK_EUR_KG
    capex_comp      = h2_peak_rate_kgh        * LCOH_COMP_EUR_KGH
    capex_bop       = capex_pem               * LCOH_BOP_FRAC
    capex_ctrl      = LCOH_CTRL_EUR

    # ---- Annualised CAPEX and O&M (EUR/yr) ----
    def ann_capex_om(capex, r, life, om_rate):
        return capex * crf(r, life), capex * om_rate

    ac_pv,        om_pv        = ann_capex_om(capex_pv,        LCOH_WACC, LCOH_PV_LIFE,        LCOH_PV_OM)
    ac_pem                         = capex_pem * crf(LCOH_WACC, LCOH_PROJECT_LIFE)
    om_pem                         = capex_pem * LCOH_PEM_OM
    ac_batt,      om_batt      = ann_capex_om(capex_batt,      LCOH_WACC, LCOH_BATT_LIFE,      LCOH_BATT_OM)
    ac_batt_conv, om_batt_conv = ann_capex_om(capex_batt_conv, LCOH_WACC, LCOH_BATT_CONV_LIFE, LCOH_BATT_CONV_OM)
    ac_grid_inv,  om_grid_inv  = ann_capex_om(capex_grid_inv,  LCOH_WACC, LCOH_GRID_INV_LIFE,  LCOH_GRID_INV_OM)
    ac_tank,      om_tank      = ann_capex_om(capex_tank,      LCOH_WACC, LCOH_TANK_LIFE,      LCOH_TANK_OM)
    ac_comp,      om_comp      = ann_capex_om(capex_comp,      LCOH_WACC, LCOH_COMP_LIFE,      LCOH_COMP_OM)
    ac_bop,       om_bop       = ann_capex_om(capex_bop,       LCOH_WACC, LCOH_BOP_LIFE,       LCOH_PEM_OM)
    ac_ctrl = capex_ctrl * crf(LCOH_WACC, LCOH_PROJECT_LIFE); om_ctrl = 0.0

    production_components = [
        ("PV array",                    capex_pv,        ac_pv,        om_pv),
        ("PEM stack (initial only)",    capex_pem,       ac_pem,       om_pem),
        ("Battery (LFP)",               capex_batt,      ac_batt,      om_batt),
        ("Batt. DC/DC converter",       capex_batt_conv, ac_batt_conv, om_batt_conv),
        ("Grid AC/DC inverter",         capex_grid_inv,  ac_grid_inv,  om_grid_inv),
        ("BoP (civil/piping)",          capex_bop,       ac_bop,       om_bop),
        ("Smart controller",            capex_ctrl,      ac_ctrl,      om_ctrl),
    ]
    storage_components = [
        ("Compressor",                                   capex_comp, ac_comp, om_comp),
        (f"H2 tank ({LCOH_TANK_KG_INSTALLED:.0f} kg)", capex_tank, ac_tank, om_tank),
    ]

    capex_prod     = sum(r[1] for r in production_components)
    ac_prod_base   = sum(r[2] for r in production_components)
    om_prod        = sum(r[3] for r in production_components)
    fixed_prod_ann_base = ac_prod_base + om_prod

    capex_stor     = sum(r[1] for r in storage_components)
    ac_stor        = sum(r[2] for r in storage_components)
    om_stor        = sum(r[3] for r in storage_components)
    fixed_stor_ann = ac_stor + om_stor

    total_capex_base     = capex_prod + capex_stor
    total_fixed_ann_base = fixed_prod_ann_base + fixed_stor_ann

    W = 128
    print(f"\n{'='*W}")
    print(f"LCOH / LCSH BREAKDOWN  (WACC={LCOH_WACC*100:.0f}%, project={LCOH_PROJECT_LIFE} yr, "
          f"PEM EOL SoH={LCOH_PEM_EOL_SOH:.2f}, replacement={LCOH_PEM_REPLACEMENT_FRAC*100:.0f}% stack CAPEX)")
    print(f"{'='*W}")
    print(f"  {'Component':<30} {'CAPEX (EUR)':>12}  {'Ann.CAPEX':>10}  {'Ann.O&M':>10}  {'Total Ann.':>10}")
    print(f"  {'-'*(W-2)}")
    print("  --- PRODUCTION chain BASE (before controller-specific PEM REPEX) ---")
    for name, cap, ac, om in production_components:
        print(f"  {name:<30} {cap:>12,.0f}  {ac:>10,.0f}  {om:>10,.0f}  {ac+om:>10,.0f}")
    print(f"  {'  subtotal (production)':<30} {capex_prod:>12,.0f}  {ac_prod_base:>10,.0f}  {om_prod:>10,.0f}  {fixed_prod_ann_base:>10,.0f}")
    print("  --- STORAGE add-on (LCSH) ---")
    for name, cap, ac, om in storage_components:
        print(f"  {name:<30} {cap:>12,.0f}  {ac:>10,.0f}  {om:>10,.0f}  {ac+om:>10,.0f}")
    print(f"  {'  subtotal (storage)':<30} {capex_stor:>12,.0f}  {ac_stor:>10,.0f}  {om_stor:>10,.0f}  {fixed_stor_ann:>10,.0f}")
    print(f"  {'-'*(W-2)}")
    print(f"  {'BASE TOTAL (before REPEX)':<30} {total_capex_base:>12,.0f}  "
          f"{ac_prod_base+ac_stor:>10,.0f}  {om_prod+om_stor:>10,.0f}  {total_fixed_ann_base:>10,.0f}")

    hdr1 = f"  {'Controller':<20} {'H2 Prod':>9} {'CF_PEM':>7} {'Grid Imp':>9} {'Grid Exp':>9} {'O2 Rev':>9}"
    hdr2 = f" {'REPEXann':>10} {'Prod.Ann':>10} {'LCOH':>9} {'Stor.Ann':>10} {'LCSH':>9}"
    print(f"\n{hdr1}{hdr2}")
    hdr1u = f"  {'':20} {'(kg/yr)':>9} {'(%)':>7} {'(EUR/yr)':>9} {'(EUR/yr)':>9} {'(EUR/yr)':>9}"
    hdr2u = f" {'(EUR/yr)':>10} {'(EUR/yr)':>10} {'(EUR/kg)':>9} {'(EUR/yr)':>10} {'(EUR/kg)':>9}"
    print(f"{hdr1u}{hdr2u}")
    print(f"  {'-'*(W-2)}")

    pem_max_annual_kg = H2_PEAK_KGH * 8760
    loc_results = {}
    deg_rows = []

    _ctrl_list = [('MOO', df_moo)]
    if df_hyst is not None:
        _ctrl_list.append(('MOO+Hyst', df_hyst))
    if df_pw is not None:
        _ctrl_list.append(('MOO+PreWarm', df_pw))
    if df_hpw is not None:
        _ctrl_list.append(('MOO+Hyst+PreWarm', df_hpw))
    if df_pen is not None:
        _ctrl_list.append(('MOO+AlwaysWarm', df_pen))
    if df_rf is not None:
        _ctrl_list.append(('MOO+RF', df_rf))
    _ctrl_list.append(('RL Agent', df_rl))

    for label, df in _ctrl_list:
        h2_prod = df['H2_prod'].sum()
        sim_years = max(len(df) * DT_HOURS / 8760.0, 1e-12)
        h2_prod_annualised = h2_prod / sim_years
        cf_pem  = 100.0 * h2_prod_annualised / max(pem_max_annual_kg, 1)

        grid_export_yr = (df['grid_export_eur'].sum() if 'grid_export_eur' in df.columns else 0.0) / sim_years
        grid_import_yr = (df['grid_import_eur'].sum() if 'grid_import_eur' in df.columns else 0.0) / sim_years
        h2_prod_yr     = h2_prod / sim_years
        o2_revenue_yr  = h2_prod_yr * O2_KG_PER_KG_H2 * O2_PRICE_EUR_KG

        h2_secondary_yr = ((df['h2_secondary_eur'].sum() if 'h2_secondary_eur' in df.columns else 0.0)
                           / sim_years)
        variable_net = grid_import_yr - grid_export_yr - o2_revenue_yr

        pem_econ = project_pem_stack_replacements(
            df, capex_pem, wacc=LCOH_WACC, project_life=LCOH_PROJECT_LIFE
        )
        prod_ann = (fixed_prod_ann_base - ac_pem
                    + pem_econ['stack_ann_total']
                    + variable_net)
        lcoh = prod_ann / max(h2_prod_yr, 1e-6)

        stor_ann = prod_ann + fixed_stor_ann
        lcsh     = stor_ann / max(h2_prod_yr, 1e-6)

        print(f"  {label:<20} {h2_prod_yr:>9.0f} {cf_pem:>7.1f}"
              f" {grid_import_yr:>9.0f} {grid_export_yr:>9.0f} {o2_revenue_yr:>9.0f}"
              f" {pem_econ['repex_ann']:>10.0f} {prod_ann:>10.0f} {lcoh:>9.2f}"
              f" {stor_ann:>10.0f} {lcsh:>9.2f}")

        deg_rows.append((label, pem_econ))
        loc_results[label] = {
            'h2_prod': h2_prod_yr, 'cf_pem': cf_pem,
            'grid_import_yr': grid_import_yr, 'grid_export_yr': grid_export_yr,
            'o2_revenue_yr': o2_revenue_yr, 'h2_secondary_yr': h2_secondary_yr,
            'soh_start': pem_econ['soh_start'], 'soh_end': pem_econ['soh_end'],
            'sim_years': pem_econ['sim_years'],
            'annual_soh_loss': pem_econ['annual_soh_loss'],
            'projected_stack_life_yr': pem_econ['projected_stack_life_yr'],
            'stack_replacements': pem_econ['stack_replacements'],
            'total_stacks': pem_econ['total_stacks'],
            'replacement_years': pem_econ['replacement_years'],
            'replacement_npv': pem_econ['replacement_npv'],
            'pem_repex_ann': pem_econ['repex_ann'],
            'pem_stack_ann_total': pem_econ['stack_ann_total'],
            'prod_ann': prod_ann, 'stor_ann': stor_ann,
            'lcoh': lcoh, 'lcsh': lcsh,
        }

    print("\n  PEM STACK LIFE / REPLACEMENT PROJECTION")
    print(f"  {'Controller':<20} {'SoH end':>9} {'Loss/yr':>10} {'Life yr':>10} {'Repl.':>7} {'Stacks':>7} {'REPEX NPV':>12} {'Years':>22}")
    print(f"  {'-'*(W-2)}")
    for label, p in deg_rows:
        life_txt = f"{p['projected_stack_life_yr']:.2f}" if np.isfinite(p['projected_stack_life_yr']) else "inf"
        yrs = p['replacement_years']
        if not yrs:
            years_txt = '-'
        elif len(yrs) <= 4:
            years_txt = ','.join(f"{y:.1f}" for y in yrs)
        else:
            years_txt = ','.join(f"{y:.1f}" for y in yrs[:3]) + f",...({len(yrs)})"
        print(f"  {label:<20} {p['soh_end']:>9.4f} {p['annual_soh_loss']:>10.4f} {life_txt:>10}"
              f" {p['stack_replacements']:>7d} {p['total_stacks']:>7d} {p['replacement_npv']:>12,.0f} {years_txt:>22}")

    print(f"\n  PEM life = ({LCOH_PEM_NEW_SOH:.2f} - {LCOH_PEM_EOL_SOH:.2f}) / annual SoH loss.")
    print("  Replacements are discrete and occur strictly before project end; a stack due exactly at year 30 is not purchased.")
    print("  REPEX is discounted to NPV, then annualised over the project; PEM O&M remains an annual fraction of initial stack CAPEX.")
    print("  LCOH = production-chain annual cost + controller-specific PEM REPEX + net grid/O2 cost, divided by annual H2 production.")
    print("  LCSH = LCOH numerator + compressor/tank annual cost, divided by the same H2 production.")
    print(f"{'='*W}")
    return loc_results

def pv_oversizing_sweep(irr, temp, price, loc_label, out_dir,
                        ratios=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)):
    """Find the optimal PV oversizing ratio (PV nameplate / PEM rating) WITH the battery
    in place, using the MOO baseline only (fast, no RL training).
    """
    import io as _io, contextlib as _cl
    global N_PAR_BASE, N_PAR_STEP, N_PAR_MAX
    _ladder0 = (N_PAR_BASE, N_PAR_STEP, N_PAR_MAX)   # restored at the end of the sweep
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pem_kw = PEM_capacity / 1000.0
    n_par_installed = N_PAR_MAX      # baseline installed parallel strings
    R0 = (n_par_installed * N_SERIES_PV * PV_MODULE_W / 1000.0) / pem_kw
    print(f"\n{'='*88}\nPV-OVERSIZING SWEEP (MOO baseline, battery in place) — {loc_label}\n{'='*88}")
    print(f"  baseline array = {n_par_installed}x{N_SERIES_PV}x{PV_MODULE_W:.0f} W = "
          f"{R0*pem_kw*1000:.0f} kWp  (ratio {R0:.2f})")
    print(f"  {'ratio':>6}{'PV_kWp':>9}{'demand%':>9}{'H2_t/yr':>9}{'curtail_MWh':>12}"
          f"{'grid_MWh':>10}{'2ndH2_kg':>10}{'LCOH':>8}{'profit_k€':>11}")
    rows = []
    for R in ratios:
        set_string_ladder(R)                  # installed strings for this PV ratio
        df = run_baselines(irr, temp, price, use_rf=False)
        h  = _controller_headline(df)
        pv_kw = R * pem_kw                                  # NAMEPLATE kWp for CAPEX
        with _cl.redirect_stdout(_io.StringIO()):
            lcoh = calculate_lcoh(df, df, df, pv_kw)['MOO']['lcoh']
        rows.append(dict(ratio=R, pv_kw=pv_kw, demand=h['dmet'], h2_t=h['h2_prod']/1000.0,
                         curt=h['curt_kwh']/1000.0, grid=h['grid_mwh'], sec=h['h2_sec'],
                         lcoh=lcoh, profit=h['profit_k']))
        print(f"  {R:>6.2f}{pv_kw:>9.0f}{h['dmet']:>9.1f}{h['h2_prod']/1000:>9.2f}"
              f"{h['curt_kwh']/1000:>12.1f}{h['grid_mwh']:>10.1f}{h['h2_sec']:>10.0f}"
              f"{lcoh:>8.2f}{h['profit_k']:>11.1f}")

    N_PAR_BASE, N_PAR_STEP, N_PAR_MAX = _ladder0      # restore the site's own array
    dfres = pd.DataFrame(rows)
    dfres.to_csv(out_dir / f'pv_oversizing_{loc_label}.csv', index=False)
    best_dem = dfres['demand'].max()
    ok = dfres[dfres['demand'] >= best_dem - 0.5]
    knee = ok.loc[ok['lcoh'].idxmin()]
    print(f"\n  Best demand fulfilment reached: {best_dem:.1f}%")
    print(f"  Lowest-LCOH size that still reaches it: ratio {knee['ratio']:.2f} "
          f"({knee['pv_kw']:.0f} kWp), LCOH {knee['lcoh']:.2f} EUR/kg, profit {knee['profit']:.1f} k€")
    print(f"  -> the battery lets you meet demand at this ratio; more PV only adds curtailment/2nd-market.")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(dfres['ratio'], dfres['demand'], 'o-', color='tab:blue'); ax[0].set_ylabel('Demand met (%)')
    ax[1].plot(dfres['ratio'], dfres['lcoh'],  'o-', color='tab:red');  ax[1].set_ylabel('LCOH (€/kg, nameplate)')
    ax[2].plot(dfres['ratio'], dfres['profit'],'o-', color='tab:green');ax[2].set_ylabel('Operating profit (k€/yr)')
    for a in ax:
        a.set_xlabel('PV oversizing ratio (PV nameplate / PEM)'); a.grid(alpha=0.3)
        a.axvline(knee['ratio'], color='grey', ls='--', lw=1)
    fig.suptitle(f'PV oversizing sweep with battery — {loc_label}')
    fig.tight_layout(); fig.savefig(out_dir / f'pv_oversizing_{loc_label}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {out_dir / f'pv_oversizing_{loc_label}.png'}")
    return dfres


def _annualised(capex_eur, life_yr, om_frac):
    """Annualised CAPEX + O&M (EUR/yr) at the project WACC."""
    r = LCOH_WACC
    crf = r * (1 + r) ** life_yr / ((1 + r) ** life_yr - 1)
    return capex_eur * (crf + om_frac)


def sizing_study(irr, temp, price, loc_label, out_dir, sweep='battery',
                 pv_ratios=(1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3),
                 batt_kwhs=(50, 500, 1000), tank_kgs=(1000, 3000, 5000, 7000, 9000),
                 fixed_batt_kwh=500, fixed_tank_kg=3000):
    """Headline metric is SELF-SUFFICIENCY % (renewable share of the plant electrical
    load) — the grid-independence goal.
    """
    global BATT_CAPACITY_WH, BATT_MAX_P_W, storage_capacity, INITIAL_H2_STORAGE
    global N_PAR_BASE, N_PAR_STEP, N_PAR_MAX
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pem_kw = PEM_capacity / 1000.0
    c_rate = BATT_MAX_P_W / BATT_CAPACITY_WH
    orig = (BATT_CAPACITY_WH, BATT_MAX_P_W, storage_capacity, INITIAL_H2_STORAGE)
    _ladder0 = (N_PAR_BASE, N_PAR_STEP, N_PAR_MAX)

    if sweep == 'battery':
        y_vals, y_label, y_unit, fname = batt_kwhs, 'Battery size (kWh)', 'kWh', 'batt'
    else:
        y_vals, y_label, y_unit, fname = tank_kgs, 'H$_2$ tank size (kg)', 'kg', 'tank'

    SS   = np.full((len(pv_ratios), len(y_vals)), np.nan)   # self-sufficiency (%)
    DEM  = np.full_like(SS, np.nan)                         # demand met (%)
    GRID = np.full_like(SS, np.nan)                         # grid import (MWh/yr)
    NV   = np.full_like(SS, np.nan)                         # net value (k EUR/yr)
    rows = []
    print(f"\n{'='*94}\nSIZING STUDY  PV x {sweep.upper()}  (MOO, self-sufficiency headline) — {loc_label}\n{'='*94}")
    print(f"  {len(pv_ratios)}x{len(y_vals)} = {len(pv_ratios)*len(y_vals)} MOO runs | demand "
          f"= {hourly_H2_target*24:.0f} kg/day | secondary-H2 excluded from value")
    try:
        for i, R in enumerate(pv_ratios):
            pv_kw  = site_string_config(R)[2] * N_SERIES_PV * PV_MODULE_W / 1000.0
            pv_ann = _annualised(pv_kw * LCOH_PV_EUR_KWP, LCOH_PV_LIFE, LCOH_PV_OM)
            for j, yv in enumerate(y_vals):
                bk = yv if sweep == 'battery' else fixed_batt_kwh
                tk = yv if sweep == 'tank'    else fixed_tank_kg
                BATT_CAPACITY_WH  = float(bk * 1000.0); BATT_MAX_P_W = BATT_CAPACITY_WH * c_rate
                storage_capacity  = float(tk)
                batt_ann = (_annualised(bk * LCOH_BATT_EUR_KWH, LCOH_BATT_LIFE, LCOH_BATT_OM)
                            + _annualised((BATT_MAX_P_W/1000.0) * LCOH_BATT_CONV_EUR_KW, LCOH_BATT_LIFE, LCOH_BATT_OM))
                tank_ann = _annualised(tk * LCOH_TANK_EUR_KG, LCOH_TANK_LIFE, LCOH_TANK_OM)
                set_string_ladder(R)          # installed strings for this PV ratio
                if TANK_INIT_MODE == 'cyclic':
                    solve_cyclic_storage(irr, temp, price, guess=0.0, verbose=False)
                else:
                    set_initial_storage()
                df = run_baselines(irr, temp, price, use_rf=False)
                h  = _controller_headline(df)
                firm_rev = df['H2_delivered'].sum() * H2_PRICE_EUR_KG
                o2_rev   = df['H2_prod'].sum() * O2_KG_PER_KG_H2 * O2_PRICE_EUR_KG
                grid_imp = df['grid_import_eur'].sum() if 'grid_import_eur' in df.columns else 0.0
                grid_exp = df['grid_export_eur'].sum() if 'grid_export_eur' in df.columns else 0.0
                nv = (firm_rev + o2_rev + grid_exp - grid_imp - pv_ann - batt_ann - tank_ann) / 1000.0
                SS[i, j], DEM[i, j], GRID[i, j], NV[i, j] = h['self_suff'], h['dmet'], h['grid_mwh'], nv

                final_storage = df['storage'].iloc[-1] if 'storage' in df.columns else 0.0

                rows.append(dict(location=loc_label, sweep=sweep,
                                 pv_ratio=R, pv_kw=pv_kw, batt_kwh=bk, tank_kg=tk,
                                 self_suff=h['self_suff'], demand=h['dmet'], grid_mwh=h['grid_mwh'],
                                 curt_mwh=h['curt_kwh']/1000.0, net_value_k=nv,
                                 grid_import_eur=grid_imp, grid_export_eur=grid_exp,
                                 h2_delivered_kg=df['H2_delivered'].sum(), h2_prod_kg=df['H2_prod'].sum(),
                                 capex_ann_k=(pv_ann + batt_ann + tank_ann) / 1000.0,
                                 initial_storage=INITIAL_H2_STORAGE, final_storage=final_storage))
                print(f"  PV {R:.2f}x | {y_unit} {yv:>5} | self-suf {h['self_suff']:5.1f}% | "
                      f"demand {h['dmet']:5.1f}% | grid {h['grid_mwh']:6.1f} MWh | NV {nv:7.1f} k EUR")
                pd.DataFrame(rows).to_csv(out_dir / f'sizing_pv_{fname}_{loc_label}.csv', index=False)
    finally:
        BATT_CAPACITY_WH, BATT_MAX_P_W, storage_capacity, INITIAL_H2_STORAGE = orig
        N_PAR_BASE, N_PAR_STEP, N_PAR_MAX = _ladder0
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f'sizing_pv_{fname}_{loc_label}.csv', index=False)
            print(f"  [saved {len(rows)} rows -> {out_dir / f'sizing_pv_{fname}_{loc_label}.csv'}]")

    dfres = pd.DataFrame(rows); dfres.to_csv(out_dir / f'sizing_pv_{fname}_{loc_label}.csv', index=False)
    i_opt, j_opt = np.unravel_index(np.nanargmax(NV), NV.shape)
    print(f"\n  Economic optimum (max net value): PV {pv_ratios[i_opt]:.2f}x + {y_vals[j_opt]} {y_unit}  "
          f"-> self-suf {SS[i_opt,j_opt]:.1f}%, demand {DEM[i_opt,j_opt]:.1f}%, NV {NV[i_opt,j_opt]:.1f} k EUR")

    def _heat(ax, M, title, cmap, fmt='{:.0f}'):
        im = ax.imshow(M, origin='lower', aspect='auto', cmap=cmap)
        ax.set_xticks(range(len(y_vals))); ax.set_xticklabels(y_vals)
        ax.set_yticks(range(len(pv_ratios))); ax.set_yticklabels([f'{r:g}' for r in pv_ratios])
        ax.set_xlabel(y_label); ax.set_ylabel('PV oversizing (x PEM)'); ax.set_title(title)
        rng = (np.nanmax(M) - np.nanmin(M)) or 1.0
        for a in range(len(pv_ratios)):
            for b in range(len(y_vals)):
                ax.text(b, a, fmt.format(M[a, b]), ha='center', va='center', fontsize=7,
                        color='white' if (np.nanmax(M)-M[a,b]) > 0.5*rng else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig, ax = plt.subplots(1, 4, figsize=(22, 5.0))
    _heat(ax[0], SS,   'Self-sufficiency (%)',              'YlGn',     '{:.1f}')
    _heat(ax[1], DEM,  'Demand fulfilment (%)',             'YlGnBu',  '{:.1f}')
    _heat(ax[2], GRID, 'Grid import (MWh/yr)',              'magma_r', '{:.0f}')
    _heat(ax[3], NV,   'Grid-savings net value (k EUR/yr)', 'viridis', '{:.0f}')
    for a in ax:
        a.scatter([j_opt], [i_opt], marker='*', s=260, edgecolor='k', facecolor='gold', zorder=5)
    fig.tight_layout(); fig.savefig(out_dir / f'sizing_pv_{fname}_{loc_label}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: sizing_pv_{fname}_{loc_label}.png")
    return dfres


def verify_designs(reco, series, dem_target, verbose=True):
    """Re-simulate each recommended design and report what it ACTUALLY delivers."""
    global storage_capacity, LCOH_TANK_KG_INSTALLED, BATT_CAPACITY_WH, BATT_MAX_P_W
    global N_PAR_BASE, N_PAR_STEP, N_PAR_MAX, INITIAL_H2_STORAGE
    keep = (storage_capacity, LCOH_TANK_KG_INSTALLED, BATT_CAPACITY_WH, BATT_MAX_P_W,
            N_PAR_BASE, N_PAR_STEP, N_PAR_MAX, INITIAL_H2_STORAGE)
    if verbose:
        print(f"\n{'='*104}\nVERIFYING RECOMMENDED DESIGNS (re-simulating each at a buildable size)\n{'='*104}")
    try:
        for r in reco:
            key = next((k for k in series if k.startswith(r['loc'])), None)
            if key is None:
                r['verified'] = False
                continue
            irr, temp, price = series[key]
            batt = float(BATT_NIGHT_KWH.get(key, r['batt']))
            tank = float(np.ceil(r['tank'] / TANK_ROUND_KG) * TANK_ROUND_KG)
            set_string_ladder(r['pv'])
            BATT_CAPACITY_WH = batt * 1000.0
            BATT_MAX_P_W     = BATT_CAPACITY_WH
            for step in range(TANK_VERIFY_STEPS + 1):
                storage_capacity       = tank
                LCOH_TANK_KG_INSTALLED = tank
                s0 = set_initial_storage()
                h  = _controller_headline(run_baselines(irr, temp, price, use_rf=False))
                if verbose:
                    print(f"  {r['loc']:<9} PV {r['pv']:.2f}x  tank {tank:>6.0f} kg  "
                          f"batt {batt:>4.0f} kWh  S0 {s0:>7.1f} kg  ->  demand {h['dmet']:6.2f} %"
                          + ("  OK" if h['dmet'] >= dem_target else "  short, growing tank"))
                if h['dmet'] >= dem_target or step == TANK_VERIFY_STEPS:
                    break
                tank += TANK_VERIFY_STEP
            r.update(tank=tank, batt=batt, s0=s0, demand=h['dmet'], ss=h['self_suff'],
                     grid=h['grid_mwh'], verified=bool(h['dmet'] >= dem_target))
    finally:
        (storage_capacity, LCOH_TANK_KG_INSTALLED, BATT_CAPACITY_WH, BATT_MAX_P_W,
         N_PAR_BASE, N_PAR_STEP, N_PAR_MAX, INITIAL_H2_STORAGE) = keep
    return reco


def plot_sizing_pareto(base_out_dir, dem_target=None, ss_target=None, batt_min=500.0,
                       tol=0.05, pv_override=None, batt_override=None, tank_override=None,
                       series=None):
    """Publication figures + recommended per-site sizing from the sizing-study CSVs.
    """
    import glob
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    dem_target = DEMAND_TARGET_PCT    if dem_target is None else dem_target
    ss_target  = SELF_SUFF_TARGET_PCT if ss_target  is None else ss_target
    pv_override = dict(pv_override or {})
    batt_override = dict(batt_override or {})
    tank_override = dict(tank_override or {})

    base_out_dir = Path(base_out_dir)
    files = sorted(glob.glob(str(base_out_dir / '**' / 'sizing_pv_*.csv'), recursive=True))
    if not files:
        print("  (sizing figures: no sizing CSVs found under", base_out_dir, ")"); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['loc'] = df['location'].str.replace('_2021', '', regex=False)
    locs = list(dict.fromkeys(df['loc']))
    COL = {lc: c for lc, c in zip(locs, ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE'])}
    pvs = sorted(df['pv_ratio'].unique())

    for _lc, _pv in list(pv_override.items()):
        if not any(np.isclose(p, _pv) for p in pvs):
            _near = min(pvs, key=lambda p: abs(p - _pv))
            print(f"  NOTE: pv_override['{_lc}'] = {_pv:g}x is not in the swept ladder "
                  f"{[round(float(p), 3) for p in pvs]} -> snapping to {_near:g}x.")
            pv_override[_lc] = float(_near)

    all_batts = sorted(df[df['sweep'] == 'battery']['batt_kwh'].unique())
    batts = [b for b in all_batts if b >= batt_min]

    for _lc, _b in list(batt_override.items()):
        if not any(np.isclose(b, _b) for b in all_batts):
            _near = min(all_batts, key=lambda b: abs(b - _b))
            print(f"  NOTE: batt_override['{_lc}'] = {_b:g}kWh is not in the swept ladder "
                  f"{[round(float(b), 1) for b in all_batts]} -> snapping to {_near:g}kWh.")
            batt_override[_lc] = float(_near)

    demand_kgday = daily_H2_target
    CF = 100.0 * demand_kgday / (H2_PEAK_KGH * 24.0)
    TITLE_CF = f'1 MW PV–PEM, {CF:.0f}% capacity factor'
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12})
    has_grideur = 'grid_import_eur' in df.columns

    # ---- annualised CAPEX + LCOH / LCOS ----
    FIXED_PROD_ANN = 199042 + 39808 + 5902 + 581
    COMP_ANN       = 15493
    def _crf(life): r = LCOH_WACC; return r*(1+r)**life/((1+r)**life-1)
    def _ann(cap, life, om): return cap*(_crf(life)+om)
    CR = 1.0
    pv_ann   = lambda pv:  _ann(pv*1000*LCOH_PV_EUR_KWP, LCOH_PV_LIFE, LCOH_PV_OM)
    batt_ann = lambda kwh: (_ann(kwh*LCOH_BATT_EUR_KWH, LCOH_BATT_LIFE, LCOH_BATT_OM)
                            + _ann(kwh*CR*LCOH_BATT_CONV_EUR_KW, LCOH_BATT_LIFE, LCOH_BATT_OM))
    tank_ann = lambda kg:  _ann(kg*LCOH_TANK_EUR_KG, LCOH_TANK_LIFE, LCOH_TANK_OM)

    def _costs(pv, tank, batt, demand, grid_eur):
        h2 = demand/100.0 * demand_kgday * 365.0
        o2 = h2 * O2_KG_PER_KG_H2 * O2_PRICE_EUR_KG
        prod = FIXED_PROD_ANN + pv_ann(pv) + batt_ann(batt) + grid_eur - o2
        stor = prod + COMP_ANN + tank_ann(tank)
        return prod/max(h2, 1e-6), stor/max(h2, 1e-6)

    # ---- per-site design ----
    reco = []
    has_storage_cols = 'initial_storage' in df.columns and 'final_storage' in df.columns

    for lc in locs:
        cands = []
        for pv in pvs:
            if lc in pv_override and not np.isclose(pv, pv_override[lc]):
                continue

            gt_full = df[(df['loc'] == lc) & (df['sweep'] == 'tank') & np.isclose(df['pv_ratio'], pv)]

            gt = gt_full.sort_values('tank_kg')

            if not len(gt) or gt['demand'].max() < dem_target:
                continue

            if lc in tank_override:
                tank = float(tank_override[lc])
                dem = (float(np.interp(tank, gt['tank_kg'], gt['demand']))
                       if tank <= gt['tank_kg'].max() else float(gt['demand'].iloc[-1]))
            elif gt['demand'].iloc[0] >= dem_target:
                tank, dem = float(gt['tank_kg'].iloc[0]), float(gt['demand'].iloc[0])
            else:
                tank, dem = float(np.interp(dem_target, gt['demand'], gt['tank_kg'])), dem_target

            valid_batts = [batt_override[lc]] if lc in batt_override else batts

            for bk in valid_batts:
                gb = df[(df['loc'] == lc) & (df['sweep'] == 'battery') &
                        np.isclose(df['pv_ratio'], pv) & np.isclose(df['batt_kwh'], bk)]
                if not len(gb): continue
                ss, grid_mwh = float(gb['self_suff'].iloc[0]), float(gb['grid_mwh'].iloc[0])
                grid_eur = float(gb['grid_import_eur'].iloc[0]) if has_grideur else grid_mwh*1000*0.08
                lcoh, lcos = _costs(pv, tank, bk, dem, grid_eur)
                cands.append(dict(loc=lc, pv=pv, pv_kw=pv*1000, tank=tank, batt=bk, demand=dem, ss=ss,
                                  grid=grid_mwh, capex=(pv_ann(pv)+tank_ann(tank)+batt_ann(bk))/1000,
                                  lcoh=lcoh, lcos=lcos))

        pool = [c for c in cands if c['ss'] >= ss_target] or cands
        if not pool:
            continue

        best_cost = min(c['lcos'] for c in pool)
        near = [c for c in pool if c['lcos'] <= best_cost * (1 + tol)]
        reco.append(min(near, key=lambda c: (c['pv'], c['capex'], c['tank'])))

    for r in reco:
        r.setdefault('s0', float('nan'))
        r.setdefault('verified', False)
    if series:
        verify_designs(reco, series, dem_target)

    RC = {r['loc']: r for r in reco}
    locs = [lc for lc in locs if lc in RC]
    if not locs:
        print("  (sizing figures: no site had a feasible design.)")
        return

    # ---- FIG 1: 3-axis ----
    fig = plt.figure(figsize=(11.5, 10.4))
    _gs = fig.add_gridspec(2, 1, height_ratios=[6.0, 1.0], hspace=0.16)
    ax  = fig.add_subplot(_gs[0], projection='3d')
    axt = fig.add_subplot(_gs[1]); axt.axis('off')

    for lc in locs:
        d = df[df['loc'] == lc]
        ax.scatter(d['capex_ann_k'], d['demand'], d['self_suff'], s=24, alpha=0.30,
                   color=COL[lc], edgecolor='none', label=lc, depthshade=False)

    _ordered = sorted(reco, key=lambda c: (c['demand'], c['ss']))
    for r in _ordered:                       # short tag only — the numbers live in the table
        ax.text(r['capex'], r['demand'], r['ss'] + 0.7, f" {r['loc']}",
                fontsize=10.5, fontweight='bold', color=COL[r['loc']],
                ha='left', va='bottom')
    for r in _ordered:
        ax.scatter([r['capex']], [r['demand']], [r['ss']], s=430, marker='*',
                   color=COL[r['loc']], edgecolor='k', lw=0.9, depthshade=False)

    ax.set_xlabel('Annualised CAPEX (kEUR/yr)', labelpad=10)
    ax.set_ylabel('Demand fulfilment (%)', labelpad=10)
    ax.set_zlabel('Self-sufficiency (%)', labelpad=6)
    ax.set_box_aspect((1, 1, 0.85), zoom=1.08)
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=20, azim=-60)
    ax.legend(title='Location', loc='upper left', framealpha=0.92)

    # ---- specification table (reserved axes: can never overlap the cloud) ----
    _hdr = ['Site', 'PV', 'H$_2$ tank', 'Battery', 'S$_0$', 'Demand met']
    _rows, _verified = [], any(r.get('verified') for r in reco)
    for r in reco:
        _rows.append([r['loc'], f"{r['pv']:.2f}$\\times$", f"{r['tank']:,.0f} kg",
                      f"{r['batt']:.0f} kWh",
                      f"{r['s0']:,.0f} kg" if np.isfinite(r.get('s0', np.nan)) else '--',
                      f"{r['demand']:.2f} %"])
    _tb = axt.table(cellText=_rows, colLabels=_hdr, cellLoc='center', loc='center')
    _tb.auto_set_font_size(False); _tb.set_fontsize(10.5); _tb.scale(1.0, 1.45)
    for (_i, _j), _cell in _tb.get_celld().items():
        _cell.set_edgecolor('0.80')
        if _i == 0:
            _cell.get_text().set_fontweight('bold'); _cell.set_facecolor('0.94')
        elif _j == 0:
            _cell.get_text().set_color(COL[_rows[_i - 1][0]])
            _cell.get_text().set_fontweight('bold')
    axt.annotate('demand met is re-simulated at the tank size shown' if _verified else
                 'demand met is INTERPOLATED, not simulated',
                 xy=(0.5, -0.02), xycoords='axes fraction', ha='center', va='top',
                 fontsize=9, style='italic', color='0.35')
    fig.savefig(base_out_dir / 'sizing_3axis.png', dpi=200, bbox_inches='tight',
                pad_inches=0.30)
    plt.close(fig)

    # ---- FIG 2: demand vs tank ----
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    for lc in locs:
        pv = RC[lc]['pv']
        g = df[(df['loc'] == lc) & (df['sweep'] == 'tank') & np.isclose(df['pv_ratio'], pv)].sort_values('tank_kg')
        ax.plot(g['tank_kg'], g['demand'], 'o-', color=COL[lc], lw=2, ms=7, label=f"{lc} ({pv:g}x PV)")
        ax.plot([RC[lc]['tank']], [RC[lc]['demand']], '*', color=COL[lc], ms=17, mec='k', mew=0.6, zorder=5)
        ax.annotate(f"{RC[lc]['tank']:.0f} kg", (RC[lc]['tank'], RC[lc]['demand']), textcoords='offset points', xytext=(4, -15), fontsize=9, fontweight='bold', color=COL[lc])
    ax.axhline(dem_target, color='grey', ls='--', lw=1)
    ax.set_xlabel('H$_2$ tank size (kg)'); ax.set_ylabel('Demand fulfilment (%)')
    ax.legend(title='Location'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(base_out_dir / 'sizing_demand_vs_tank.png', dpi=200, bbox_inches='tight'); plt.close(fig)

    # ---- FIG 3: self-sufficiency vs PV ----
    fig, ax = plt.subplots(figsize=(7.6, 5.3)); ax.axhspan(86, 92, color='grey', alpha=0.12)
    for lc in locs:
        bk = RC[lc]['batt']
        g = df[(df['loc'] == lc) & (df['sweep'] == 'battery') & np.isclose(df['batt_kwh'], bk)].sort_values('pv_ratio')
        ax.plot(g['pv_ratio'], g['self_suff'], 'o-', color=COL[lc], lw=2, ms=7, label=f"{lc} ({int(bk)} kWh)")
        ax.plot([RC[lc]['pv']], [RC[lc]['ss']], '*', color=COL[lc], ms=16, mec='k', mew=0.6, zorder=5)
    ax.set_xlabel('PV oversizing (x PEM)'); ax.set_ylabel('Self-sufficiency (%)')
    ax.legend(title='Location'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(base_out_dir / 'sizing_selfsuff_vs_pv.png', dpi=200, bbox_inches='tight'); plt.close(fig)

    # ---- FIG 4: battery role ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for lc in locs:
        pv = RC[lc]['pv']
        g = df[(df['loc'] == lc) & (df['sweep'] == 'battery') & np.isclose(df['pv_ratio'], pv)].sort_values('batt_kwh')
        a1.plot(g['batt_kwh'], g['self_suff'], 'o-', color=COL[lc], lw=2, ms=7, label=f"{lc} ({pv:g}x)")
        a2.plot(g['batt_kwh'], g['grid_mwh'],  'o-', color=COL[lc], lw=2, ms=7, label=f"{lc} ({pv:g}x)")
        a1.plot([RC[lc]['batt']], [RC[lc]['ss']],   '*', color=COL[lc], ms=16, mec='k', mew=0.6, zorder=5)
        a2.plot([RC[lc]['batt']], [RC[lc]['grid']], '*', color=COL[lc], ms=16, mec='k', mew=0.6, zorder=5)
    a1.axhspan(86, 92, color='grey', alpha=0.12)
    for a in (a1, a2): a.legend(title='Location'); a.grid(alpha=0.3)
    a1.set_xlabel('Battery size (kWh)'); a1.set_ylabel('Self-sufficiency (%)'); a1.set_title('Battery raises self-sufficiency')
    a2.set_xlabel('Battery size (kWh)'); a2.set_ylabel('Grid import (MWh/yr)'); a2.set_title('Battery cuts grid import')
    fig.tight_layout(); fig.savefig(base_out_dir / 'sizing_battery_role.png', dpi=200, bbox_inches='tight'); plt.close(fig)

    # ---- table ----
    print(f"\n{'='*104}\nRECOMMENDED PER-SITE SIZING  ({TITLE_CF} = {demand_kgday:.0f} kg/day; "
          f"demand>={dem_target:.0f}%, self-suff>={ss_target:.0f}%, SUSTAINABLE)\n{'='*104}")
    print(f"  {'site':<10}{'PV x':>6}{'PV kWp':>9}{'tank kg':>10}{'batt kWh':>10}{'S0 kg':>9}"
          f"{'demand%':>9}{'self-suf%':>11}{'grid MWh':>10}{'LCOH E/kg':>11}{'LCOS E/kg':>11}")
    for r in reco:
        _s0 = f"{r['s0']:>9.0f}" if np.isfinite(r.get('s0', np.nan)) else f"{'-':>9}"
        print(f"  {r['loc']:<10}{r['pv']:>6.2f}{r['pv_kw']:>9.0f}{r['tank']:>10.0f}{r['batt']:>10.0f}"
              f"{_s0}{r['demand']:>9.2f}{r['ss']:>11.1f}{r['grid']:>10.1f}"
              f"{r['lcoh']:>11.2f}{r['lcos']:>11.2f}")
    if any(r.get('verified') for r in reco):
        print("\n  demand% is MEASURED: each design was re-simulated at the tank size shown.")
        print("  Paste into SITE_SIZING:")
        for r in reco:
            _k = next((k for k in SITE_SIZING if k.startswith(r['loc'])), r['loc'])
            print(f"    '{_k}': ({r['pv']:.2f}, {r['tank']:.0f}, {r['batt']:.0f}),")
    else:
        print("\n  demand% is INTERPOLATED, not simulated. Pass series=... to verify it.")
    return reco


def battery_sensitivity_analysis(irr, temp, price, pv_peak_kw, loc_label, out_dir):
    """Sweep battery size (kWh) using only the MOO baseline (fast, no RL training).
    """
    import matplotlib.pyplot as plt

    global BATT_CAPACITY_WH, BATT_MAX_P_W

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curr_kwh = BATT_CAPACITY_WH / 1000.0
    c_rate   = BATT_MAX_P_W / BATT_CAPACITY_WH   # kept constant across sweep

    _batt_mults    = [0.1, 0.25, 0.5, 1.0, 2.0]
    batt_sizes_kwh = [max(10.0, round(curr_kwh * m / 10.0) * 10.0) for m in _batt_mults]
    seen = set(); batt_sizes_kwh_u = []
    for v in batt_sizes_kwh:
        if v not in seen: seen.add(v); batt_sizes_kwh_u.append(v)
    batt_sizes_kwh = batt_sizes_kwh_u
    _batt_mults    = [v / curr_kwh for v in batt_sizes_kwh]   # recompute exact multiples
    batt_labels    = [f'{m:.2g}× ({s:.0f} kWh)' for m, s in zip(_batt_mults, batt_sizes_kwh)]

    def _crf(r, n):
        return r * (1 + r)**n / ((1 + r)**n - 1)

    pem_kw        = PEM_capacity / 1000.0
    h2_peak_kgh   = H2_PEAK_KGH
    batt_power_kw = BATT_MAX_P_W / 1000.0

    def _ann(capex, r, life, om):
        return capex * _crf(r, life) + capex * om

    pem_stack_capex = pem_kw * LCOH_PEM_EUR_KW
    fixed_prod_ann = sum([
        _ann(pv_peak_kw          * LCOH_PV_EUR_KWP,       LCOH_WACC, LCOH_PV_LIFE,        LCOH_PV_OM),
        pem_stack_capex * LCOH_PEM_OM,
        _ann(batt_power_kw       * LCOH_BATT_CONV_EUR_KW,  LCOH_WACC, LCOH_BATT_CONV_LIFE, LCOH_BATT_CONV_OM),
        _ann(LCOH_GRID_INV_KW    * LCOH_GRID_INV_EUR_KW,   LCOH_WACC, LCOH_GRID_INV_LIFE,  LCOH_GRID_INV_OM),
        _ann(pem_kw * LCOH_PEM_EUR_KW * LCOH_BOP_FRAC,    LCOH_WACC, LCOH_BOP_LIFE,       LCOH_PEM_OM),
        LCOH_CTRL_EUR * _crf(LCOH_WACC, LCOH_PROJECT_LIFE),
    ])
    stor_ann = sum([
        _ann(LCOH_TANK_KG_INSTALLED * LCOH_TANK_EUR_KG, LCOH_WACC, LCOH_TANK_LIFE, LCOH_TANK_OM),
        _ann(h2_peak_kgh * LCOH_COMP_EUR_KGH,           LCOH_WACC, LCOH_COMP_LIFE, LCOH_COMP_OM),
    ])

    print(f"\n{'='*100}")
    print(f"  Battery sensitivity: {loc_label}   (MOO baseline, {len(batt_sizes_kwh)} sizes, "
          f"nominal = {curr_kwh:.0f} kWh)")
    print(f"{'='*100}")
    print(f"  {'Battery':>18} {'Pmax(kW)':>9} {'Cyc/yr':>7} "
          f"{'GImp(kWh)':>11} {'Curt(kWh)':>11} {'BattAnn(€)':>11} {'LCOH':>8} {'LCSH':>8}")
    print(f"  {'-'*98}")

    results    = []
    orig_wh    = BATT_CAPACITY_WH
    orig_pw    = BATT_MAX_P_W

    for batt_kwh, blabel in zip(batt_sizes_kwh, batt_labels):
        BATT_CAPACITY_WH = batt_kwh * 1000.0
        BATT_MAX_P_W     = min(batt_kwh * 1000.0 * c_rate, 2_500_000.0)  # cap at 2.5 MW (1MW system)

        try:
            df = run_baselines(irr, temp, price, use_rf=False)
        finally:
            BATT_CAPACITY_WH = orig_wh
            BATT_MAX_P_W     = orig_pw

        h2p      = df['H2_prod'].sum()
        cf       = 100.0 * h2p / (H2_PEAK_KGH * 8760)
        gimp_kwh = df['grid_bought'].sum() * DT_HOURS / 1000.0  # power samples -> kWh
        gimp     = df['grid_import_eur'].sum()
        grev     = df['grid_export_eur'].sum()
        o2r      = h2p * O2_KG_PER_KG_H2 * O2_PRICE_EUR_KG
        pmax     = min(batt_kwh * c_rate, 2500.0)   # kW display cap (1MW system)
        cycles   = (df['batt_charge_w'].sum() + df['batt_discharge_w'].sum()) * DT_HOURS / 2.0 / (batt_kwh * 1000.0)
        curt_kwh = df['P_curtailed_W'].sum() * DT_HOURS / 1000.0 if 'P_curtailed_W' in df.columns else 0.0

        batt_ann = _ann(batt_kwh * LCOH_BATT_EUR_KWH, LCOH_WACC, LCOH_BATT_LIFE, LCOH_BATT_OM)
        pem_econ = project_pem_stack_replacements(df, pem_stack_capex, wacc=LCOH_WACC)
        prod_ann = fixed_prod_ann + pem_econ['stack_ann_total'] + batt_ann + gimp - grev - o2r
        lcoh     = prod_ann / max(h2p, 1e-6)
        lcsh     = (prod_ann + stor_ann) / max(h2p, 1e-6)

        marker = " ◄ current" if abs(batt_kwh - curr_kwh) < curr_kwh*0.02 else ""
        print(f"  {blabel:>18} {pmax:>9.0f} {cycles:>7.1f} "
              f"{gimp_kwh:>11.0f} {curt_kwh:>11.0f} {batt_ann:>11.0f} {lcoh:>8.2f} {lcsh:>8.2f}{marker}")

        results.append({
            'batt_kwh': batt_kwh, 'pmax_kw': pmax,
            'cycles': cycles, 'grid_imp_kwh': gimp_kwh, 'curt_kwh': curt_kwh,
            'batt_ann': batt_ann,
            'lcoh': lcoh, 'lcsh': lcsh,
        })

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 13, 'axes.labelsize': 15,
        'axes.titlesize': 14, 'legend.fontsize': 12,
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    bx     = [r['batt_kwh']     for r in results]
    cycy   = [r['cycles']       for r in results]
    lcohy  = [r['lcoh']         for r in results]
    lcshy  = [r['lcsh']         for r in results]
    banky  = [r['batt_ann']     for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bx_idx = list(range(len(batt_sizes_kwh)))   # integer positions for the x-axis

    def _panel(ax, ydata, ylabel, title, color, yref=None, secondary=None, mark_opt=True):
        ax.plot(bx_idx, ydata, 'o-', color=color, lw=1.8, ms=5, zorder=3)
        curr_idx = min(range(len(batt_sizes_kwh)), key=lambda i: abs(batt_sizes_kwh[i] - curr_kwh))
        ax.axvline(curr_idx, color='black', lw=1.1, ls='--',
                   label=f'Current ({curr_kwh:.0f} kWh = 1×)')
        if mark_opt:
            opt_idx = ydata.index(min(ydata))
            ax.scatter(bx_idx[opt_idx], ydata[opt_idx], marker='*', color='red',
                       s=130, zorder=5, label=f'Lowest: {batt_labels[opt_idx]}')
        if yref is not None:
            ax.axhline(yref, color='#888888', lw=0.9, ls=':', label=f'Ref {yref:.0f}')
        ax.set_xticks(bx_idx)
        ax.set_xticklabels(batt_labels, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel('Battery Capacity (× nominal)')
        ax.set_ylabel(ylabel, color=color)
        ax.set_title(title)
        ax.yaxis.grid(True, linestyle='--', alpha=0.45)
        ax.set_axisbelow(True)
        ax.legend(framealpha=0.85, loc='best')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if secondary is not None:
            ax2 = ax.twinx()
            ax2.plot(bx_idx, secondary, 's--', color='gray', lw=0.9, ms=4, alpha=0.55)
            ax2.set_ylabel('Battery Ann. CAPEX+O&M (EUR/yr)', color='gray', fontsize=7.5)
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)

    _panel(axes[0], lcohy, 'LCOH (EUR kg⁻¹)', 'LCOH vs Battery Size', '#228833',
           secondary=banky, mark_opt=True)
    _panel(axes[1], cycy,  'Battery cycles (per yr)',
           'Battery Utilisation', '#EE6677', mark_opt=False)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'Battery_Sensitivity_{loc_label}.{ext}',
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"\n  Figure saved: Battery_Sensitivity_{loc_label}.pdf/.png")
    return results


# =============================================================================
# %% USE TRAINED RL IN SENSITIVITY ANALYSIS (faster than MOO re-runs)
# =============================================================================

def evaluate_rl_single_run(model, irr_data, temp_data_C, price_data_mwh, pv_scale=1.0):
    """Run one full year through trained RL agent (eval mode)."""
    env = PVPEMEnv(irr_data, temp_data_C, price_data_mwh, eval_mode=True, pv_scale=pv_scale)
    obs, _ = env.reset()
    records = []

    for step in range(len(irr_data)):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, done, info = env.step(action)
        info = dict(info, step=step, irradiance=float(np.clip(irr_data[step], 0, 1200)))
        records.append(info)
        if done:
            break

    return pd.DataFrame(records)


def tank_sensitivity_analysis_rl(model, irr, temp, price, pv_peak_kw, loc_label, out_dir):
    """Tank sensitivity analysis using trained RL agent (much faster than MOO)."""
    import matplotlib.pyplot as plt

    global storage_capacity, LCOH_TANK_KG_INSTALLED

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_demand_kg = daily_H2_target
    _auto_days      = [7, 30, 60, 90, 120]
    tank_sizes_kg   = [max(1.0, round(daily_demand_kg * d)) for d in _auto_days]
    seen_t = set(); tank_sizes_u = []
    for v in tank_sizes_kg:
        if v not in seen_t: seen_t.add(v); tank_sizes_u.append(v)
    tank_sizes_kg = tank_sizes_u
    _auto_days    = [v / max(daily_demand_kg, 1e-6) for v in tank_sizes_kg]
    tank_labels   = [f'{d:.0f}d ({s:.0f} kg)' for d, s in zip(_auto_days, tank_sizes_kg)]

    curr_kg       = storage_capacity

    def _crf(r, n): return r * (1 + r)**n / ((1 + r)**n - 1)
    def _ann(capex, r, life, om): return capex * _crf(r, life) + capex * om

    pem_kw   = PEM_capacity / 1000.0
    batt_kwh = BATT_CAPACITY_WH / 1000.0
    batt_kw  = BATT_MAX_P_W    / 1000.0

    pem_stack_capex = pem_kw * LCOH_PEM_EUR_KW
    fixed_prod_ann = sum([
        _ann(pv_peak_kw         * LCOH_PV_EUR_KWP,       LCOH_WACC, LCOH_PV_LIFE,        LCOH_PV_OM),
        pem_stack_capex * LCOH_PEM_OM,
        _ann(batt_kwh           * LCOH_BATT_EUR_KWH,      LCOH_WACC, LCOH_BATT_LIFE,      LCOH_BATT_OM),
        _ann(batt_kw            * LCOH_BATT_CONV_EUR_KW,  LCOH_WACC, LCOH_BATT_CONV_LIFE, LCOH_BATT_CONV_OM),
        _ann(LCOH_GRID_INV_KW   * LCOH_GRID_INV_EUR_KW,   LCOH_WACC, LCOH_GRID_INV_LIFE,  LCOH_GRID_INV_OM),
        _ann(pem_kw * LCOH_PEM_EUR_KW * LCOH_BOP_FRAC,   LCOH_WACC, LCOH_BOP_LIFE,       LCOH_PEM_OM),
        LCOH_CTRL_EUR * _crf(LCOH_WACC, LCOH_PROJECT_LIFE),
    ])
    comp_ann = _ann(H2_PEAK_KGH * LCOH_COMP_EUR_KGH, LCOH_WACC, LCOH_COMP_LIFE, LCOH_COMP_OM)

    print(f"\n{'='*110}")
    print(f"  H₂ Storage Tank Sensitivity (RL Agent): {loc_label}   ({len(tank_sizes_kg)} sizes)")
    print(f"{'='*110}")
    print(f"  {'Tank size':>15} {'H2prod':>8} {'H2del':>8} "
          f"{'Overflow':>9} {'Unmet':>8} {'Fulfill%':>9} "
          f"{'LCOH_prod':>10} {'LCSH':>8} {'TankAnn':>9}")
    print(f"  {'-'*108}")

    results  = []
    orig_sc  = storage_capacity
    orig_tk  = LCOH_TANK_KG_INSTALLED

    for tank_kg, tlabel in zip(tank_sizes_kg, tank_labels):
        storage_capacity       = float(tank_kg)
        LCOH_TANK_KG_INSTALLED = float(tank_kg)
        set_initial_storage()

        try:
            df = evaluate_rl_single_run(model, irr, temp, price)
        finally:
            storage_capacity       = orig_sc
            LCOH_TANK_KG_INSTALLED = orig_tk

        h2p          = df['H2_prod'].sum()
        h2d          = df['H2_delivered'].sum()
        h2_unmet     = df['H2_unmet'].sum()
        annual_demand= df['H2_daily_demand'].sum()
        final_stor   = df['storage'].iloc[-1]
        overflow_est = df['H2_overflow'].sum()
        fulfill_pct  = 100.0 * h2d / max(annual_demand, 1e-6)
        autonomy     = tank_kg / max(annual_demand / 365.0, 1e-6)

        gimp  = df['grid_import_eur'].sum()
        grev  = df['grid_export_eur'].sum()
        o2r   = h2p * O2_KG_PER_KG_H2 * O2_PRICE_EUR_KG

        tank_ann = _ann(tank_kg * LCOH_TANK_EUR_KG, LCOH_WACC, LCOH_TANK_LIFE, LCOH_TANK_OM)
        pem_econ = project_pem_stack_replacements(df, pem_stack_capex, wacc=LCOH_WACC)
        prod_ann = fixed_prod_ann + pem_econ['stack_ann_total'] + gimp - grev - o2r
        lcoh_prod = prod_ann / max(h2p, 1e-6)
        lcoh_del  = prod_ann / max(h2d, 1e-6)
        stor_ann = comp_ann + tank_ann
        lcsh     = (prod_ann + stor_ann) / max(h2p, 1e-6)

        marker = " ◄ current" if abs(tank_kg - curr_kg) < 1 else ""
        print(f"  {tlabel:>15} {h2p:>8.0f} {h2d:>8.0f} "
              f"{overflow_est:>9.0f} {h2_unmet:>8.0f} {fulfill_pct:>9.1f} "
              f"{lcoh_prod:>10.2f} {lcsh:>8.2f} {tank_ann:>9.0f}{marker}")

        results.append({
            'tank_kg':     tank_kg,
            'autonomy_d':  autonomy,
            'h2_prod':     h2p,
            'h2_del':      h2d,
            'overflow':    overflow_est,
            'h2_unmet':    h2_unmet,
            'fulfill_pct': fulfill_pct,
            'tank_ann':    tank_ann,
            'lcoh_prod':   lcoh_prod,
            'lcoh_del':    lcoh_del,
            'lcsh':        lcsh,
        })

    plt.rcParams.update({'font.family': 'serif', 'font.size': 13,
                         'axes.titlesize': 14, 'axes.labelsize': 15,
                         'xtick.labelsize': 13, 'ytick.labelsize': 13,
                         'legend.fontsize': 12, 'savefig.dpi': 300})
    xi      = list(range(len(results)))
    fly     = [r['fulfill_pct'] for r in results]
    lprod_y = [r['lcoh_prod']   for r in results]
    lcsh_y  = [r['lcsh']        for r in results]
    curr_i = min(range(len(results)), key=lambda i: abs(results[i]['tank_kg'] - curr_kg))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))

    axL.plot(xi, fly, 'o-', color='#4477AA', lw=1.8, ms=6)
    axL.axhline(100, color='green', ls=':', lw=1.0, label='100% demand met')
    axL.axvline(curr_i, color='black', ls='--', lw=1.0, label=f'Current ({curr_kg:.0f} kg)')
    axL.set_ylabel('Demand Fulfillment (%)'); axL.set_title('Fulfillment plateau')
    axL.set_ylim(min(fly) * 0.98, 101.5)

    axR.plot(xi, lprod_y, 'o-', color='#228833', lw=1.8, ms=6, label='LCOH (production)')
    axR.plot(xi, lcsh_y,  's-', color='#AA55CC', lw=1.8, ms=6, label='LCSH (production + storage)')
    axR.axvline(curr_i, color='black', ls='--', lw=1.0, label=f'Current ({curr_kg:.0f} kg)')
    axR.set_ylabel('Levelised Cost (EUR kg⁻¹)')
    axR.set_title('Cost vs tank size')

    for ax in (axL, axR):
        ax.set_xticks(xi); ax.set_xticklabels(tank_labels, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel('Tank size (autonomy days)')
        ax.grid(True, ls='--', alpha=0.4); ax.set_axisbelow(True)
        ax.legend(fontsize=10, framealpha=0.85)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'Tank_Sensitivity_{loc_label}.{ext}',
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Figure saved: Tank_Sensitivity_{loc_label}.pdf/.png")

    return results


# =============================================================================
# %% TORNADO PLOTS: Parameter Sensitivity for LCOH & H2 Fulfillment
# =============================================================================

def tornado_sensitivity_analysis(irr, temp, price, pv_peak_kw, loc_label, out_dir, model=None, pv_scale=1.0):
    """Tornado plots: show % change in LCOH & H2 fulfillment when each parameter ±20%.
    """
    import matplotlib.pyplot as plt

    global PEM_capacity, BATT_CAPACITY_WH, BATT_MAX_P_W, storage_capacity, H2_PRICE_EUR_KG, LCOH_WACC
    global irradiance_scaling_factor  # will define

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    irr = np.array(irr)
    temp = np.array(temp)
    price = np.array(price)

    print(f"\n{'='*90}")
    print(f"  TORNADO SENSITIVITY: {loc_label}")
    print(f"{'='*90}")

    # ---- Proper annualised production-chain fixed cost (mirrors calculate_lcoh) ----
    def _crf(r, n):
        return r * (1 + r) ** n / ((1 + r) ** n - 1)

    def _prod_ann_fixed(pv_kw, wacc, pem_w, batt_wh, batt_w):
        def ann(capex, life, om):
            return capex * _crf(wacc, life) + capex * om
        pem_kw = pem_w / 1000.0
        pem_stack_capex = pem_kw * LCOH_PEM_EUR_KW
        return (ann(pv_kw           * LCOH_PV_EUR_KWP,          LCOH_PV_LIFE,        LCOH_PV_OM)
              + pem_stack_capex * LCOH_PEM_OM
              + ann(batt_wh / 1000.0 * LCOH_BATT_EUR_KWH,       LCOH_BATT_LIFE,      LCOH_BATT_OM)
              + ann(batt_w  / 1000.0 * LCOH_BATT_CONV_EUR_KW,   LCOH_BATT_CONV_LIFE, LCOH_BATT_CONV_OM)
              + ann(LCOH_GRID_INV_KW * LCOH_GRID_INV_EUR_KW,    LCOH_GRID_INV_LIFE,  LCOH_GRID_INV_OM)
              + ann(pem_kw * LCOH_PEM_EUR_KW * LCOH_BOP_FRAC,   LCOH_BOP_LIFE,       LCOH_PEM_OM)
              + LCOH_CTRL_EUR * _crf(wacc, LCOH_PROJECT_LIFE))

    orig_pem     = PEM_capacity
    orig_batt_wh = BATT_CAPACITY_WH
    orig_batt_w  = BATT_MAX_P_W
    orig_tank    = storage_capacity
    orig_wacc    = LCOH_WACC
    orig_price   = price.copy()

    def _lcoh_of(df, pv_kw, wacc, pem_w, batt_wh, batt_w):
        h2   = df['H2_prod'].sum()
        gimp = df['grid_import_eur'].sum() if 'grid_import_eur' in df.columns else 0.0
        gexp = df['grid_export_eur'].sum() if 'grid_export_eur' in df.columns else 0.0
        o2   = h2 * O2_KG_PER_KG_H2 * O2_PRICE_EUR_KG
        fixed = _prod_ann_fixed(pv_kw, wacc, pem_w, batt_wh, batt_w)
        pem_stack_capex = (pem_w / 1000.0) * LCOH_PEM_EUR_KW
        pem_econ = project_pem_stack_replacements(df, pem_stack_capex, wacc=wacc)
        return (fixed + pem_econ['stack_ann_total'] + gimp - gexp - o2) / max(h2, 1e-6)

    baseline_df   = evaluate_rl_single_run(model, irr, temp, price, pv_scale=pv_scale) if model else run_baselines(irr, temp, price, pv_scale=pv_scale)
    baseline_lcoh = _lcoh_of(baseline_df, pv_peak_kw, orig_wacc, orig_pem, orig_batt_wh, orig_batt_w)
    baseline_fulfill = 100.0 * baseline_df['H2_delivered'].sum() / max(baseline_df['H2_daily_demand'].sum(), 1e-6)
    print(f"  Baseline LCOH: {baseline_lcoh:.2f} EUR/kg | H2 Fulfillment: {baseline_fulfill:.1f}%\n")

    parameters = {
        'PV Capacity ±20%':  (-0.2,  +0.2,  'PV_scale'),
        'PEM Capacity ±20%': (-0.2,  +0.2,  'PEM_scale'),
        'Battery Cap. ±20%': (-0.2,  +0.2,  'BATT_scale'),
        'WACC 4-8%':         (-0.02, +0.02, 'WACC'),
        'Electricity ±30%':  (-0.3,  +0.3,  'elec_price'),
    }
    tornado_data = {name: {'low': None, 'high': None} for name in parameters.keys()}

    def _scenario_lcoh(param_type, delta):
        global PEM_capacity, BATT_CAPACITY_WH, BATT_MAX_P_W, storage_capacity
        global N_PAR_BASE, N_PAR_STEP, N_PAR_MAX
        pv_kw, wacc, pem_w = pv_peak_kw, orig_wacc, orig_pem
        batt_wh, batt_w    = orig_batt_wh, orig_batt_w
        if param_type == 'PV_scale':
            pv_kw = pv_peak_kw * (1 + delta)
            _lad = set_string_ladder(pv_kw / (PEM_capacity / 1000.0))
            df = evaluate_rl_single_run(model, irr, temp, price) if model else run_baselines(irr, temp, price)
            N_PAR_BASE, N_PAR_STEP, N_PAR_MAX = _lad
        elif param_type == 'PEM_scale':
            pem_w = orig_pem * (1 + delta); PEM_capacity = int(pem_w)
            df = evaluate_rl_single_run(model, irr, temp, price, pv_scale=pv_scale) if model else run_baselines(irr, temp, price, pv_scale=pv_scale)
            PEM_capacity = orig_pem
        elif param_type == 'BATT_scale':
            batt_wh, batt_w = orig_batt_wh * (1 + delta), orig_batt_w * (1 + delta)
            BATT_CAPACITY_WH, BATT_MAX_P_W = int(batt_wh), int(batt_w)
            df = evaluate_rl_single_run(model, irr, temp, price, pv_scale=pv_scale) if model else run_baselines(irr, temp, price, pv_scale=pv_scale)
            BATT_CAPACITY_WH, BATT_MAX_P_W = orig_batt_wh, orig_batt_w
        elif param_type == 'WACC':
            wacc = orig_wacc + delta                               # purely economic — sim unchanged
            df = evaluate_rl_single_run(model, irr, temp, price, pv_scale=pv_scale) if model else run_baselines(irr, temp, price, pv_scale=pv_scale)
        elif param_type == 'elec_price':
            df = evaluate_rl_single_run(model, irr, temp, orig_price * (1 + delta), pv_scale=pv_scale) if model else run_baselines(irr, temp, orig_price * (1 + delta), pv_scale=pv_scale)
        elif param_type == 'tank_scale':
            storage_capacity = int(orig_tank * (1 + delta))        # tank is storage-chain → LCOH ~flat (shows in LCSH)
            df = evaluate_rl_single_run(model, irr, temp, price, pv_scale=pv_scale) if model else run_baselines(irr, temp, price, pv_scale=pv_scale)
            storage_capacity = orig_tank
        return _lcoh_of(df, pv_kw, wacc, pem_w, batt_wh, batt_w)

    for param_name, (dlo, dhi, ptype) in parameters.items():
        print(f"  Sweeping: {param_name}...")
        lcoh_lo = _scenario_lcoh(ptype, dlo)
        lcoh_hi = _scenario_lcoh(ptype, dhi)
        tornado_data[param_name]['low']  = (lcoh_lo - baseline_lcoh) / baseline_lcoh * 100
        tornado_data[param_name]['high'] = (lcoh_hi - baseline_lcoh) / baseline_lcoh * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    param_names = list(tornado_data.keys())
    lows = [tornado_data[p]['low'] for p in param_names]
    highs = [tornado_data[p]['high'] for p in param_names]

    widths = [abs(h - l) for h, l in zip(highs, lows)]
    sorted_idx = sorted(range(len(widths)), key=lambda i: widths[i], reverse=True)

    param_names_sorted = [param_names[i] for i in sorted_idx]
    lows_sorted = [lows[i] for i in sorted_idx]
    highs_sorted = [highs[i] for i in sorted_idx]

    y_pos = np.arange(len(param_names_sorted))
    ax.barh(y_pos, lows_sorted, left=0, height=0.6, color='#EE6677', label='Low scenario')
    ax.barh(y_pos, highs_sorted, left=0, height=0.6, color='#228833', label='High scenario')

    ax.axvline(0, color='black', lw=1.5, linestyle='-')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names_sorted)
    ax.set_xlabel('% Change in LCOH', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    fig.savefig(out_dir / f'Tornado_LCOH_{loc_label}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✓ Tornado plot saved: Tornado_LCOH_{loc_label}.png\n")

    return tornado_data


def tank_sensitivity_analysis(irr, temp, price, pv_peak_kw, loc_label, out_dir):
    """Sweep H2 storage tank size (kg) using only the MOO baseline (fast, no RL
    training).
    """
    import matplotlib.pyplot as plt

    global storage_capacity, LCOH_TANK_KG_INSTALLED

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_demand_kg = daily_H2_target
    _auto_days      = [7, 14, 30, 60, 90, 120]
    tank_sizes_kg   = [max(1.0, round(daily_demand_kg * d)) for d in _auto_days]
    seen_t = set(); tank_sizes_u = []
    for v in tank_sizes_kg:
        if v not in seen_t: seen_t.add(v); tank_sizes_u.append(v)
    tank_sizes_kg = tank_sizes_u
    _auto_days    = [v / max(daily_demand_kg, 1e-6) for v in tank_sizes_kg]   # recompute exact days
    tank_labels   = [f'{d:.0f}d ({s:.0f} kg)' for d, s in zip(_auto_days, tank_sizes_kg)]

    curr_kg       = storage_capacity

    def _crf(r, n): return r * (1 + r)**n / ((1 + r)**n - 1)
    def _ann(capex, r, life, om): return capex * _crf(r, life) + capex * om

    pem_kw   = PEM_capacity / 1000.0
    batt_kwh = BATT_CAPACITY_WH / 1000.0
    batt_kw  = BATT_MAX_P_W    / 1000.0

    pem_stack_capex = pem_kw * LCOH_PEM_EUR_KW
    fixed_prod_ann = sum([
        _ann(pv_peak_kw         * LCOH_PV_EUR_KWP,       LCOH_WACC, LCOH_PV_LIFE,        LCOH_PV_OM),
        pem_stack_capex * LCOH_PEM_OM,
        _ann(batt_kwh           * LCOH_BATT_EUR_KWH,      LCOH_WACC, LCOH_BATT_LIFE,      LCOH_BATT_OM),
        _ann(batt_kw            * LCOH_BATT_CONV_EUR_KW,  LCOH_WACC, LCOH_BATT_CONV_LIFE, LCOH_BATT_CONV_OM),
        _ann(LCOH_GRID_INV_KW   * LCOH_GRID_INV_EUR_KW,   LCOH_WACC, LCOH_GRID_INV_LIFE,  LCOH_GRID_INV_OM),
        _ann(pem_kw * LCOH_PEM_EUR_KW * LCOH_BOP_FRAC,   LCOH_WACC, LCOH_BOP_LIFE,       LCOH_PEM_OM),
        LCOH_CTRL_EUR * _crf(LCOH_WACC, LCOH_PROJECT_LIFE),
    ])
    comp_ann = _ann(H2_PEAK_KGH * LCOH_COMP_EUR_KGH, LCOH_WACC, LCOH_COMP_LIFE, LCOH_COMP_OM)

    print(f"\n{'='*110}")
    print(f"  H₂ Storage Tank Sensitivity: {loc_label}   ({len(tank_sizes_kg)} sizes, "
          f"daily demand = {daily_demand_kg:.2f} kg/day)")
    print(f"{'='*110}")
    print(f"  {'Tank size':>15} {'H2prod':>8} {'H2del':>8} "
          f"{'Overflow':>9} {'Unmet':>8} {'Fulfill%':>9} "
          f"{'LCOH_prod':>10} {'LCSH':>8} {'TankAnn':>9}")
    print(f"  {'-'*108}")

    results  = []
    orig_sc  = storage_capacity
    orig_tk  = LCOH_TANK_KG_INSTALLED

    for tank_kg, tlabel in zip(tank_sizes_kg, tank_labels):
        storage_capacity       = float(tank_kg)
        LCOH_TANK_KG_INSTALLED = float(tank_kg)
        set_initial_storage()

        try:
            df = run_baselines(irr, temp, price, use_rf=False)
        finally:
            storage_capacity       = orig_sc
            LCOH_TANK_KG_INSTALLED = orig_tk

        h2p          = df['H2_prod'].sum()
        h2d          = df['H2_delivered'].sum()
        h2_unmet     = df['H2_unmet'].sum()
        annual_demand= df['H2_daily_demand'].sum()   # actual annual contract from records
        final_stor   = df['storage'].iloc[-1]
        overflow_est = df['H2_overflow'].sum()
        fulfill_pct  = 100.0 * h2d / max(annual_demand, 1e-6)
        autonomy     = tank_kg / max(annual_demand / 365.0, 1e-6)

        gimp  = df['grid_import_eur'].sum()
        grev  = df['grid_export_eur'].sum()
        o2r   = h2p * O2_KG_PER_KG_H2 * O2_PRICE_EUR_KG

        tank_ann = _ann(tank_kg * LCOH_TANK_EUR_KG, LCOH_WACC, LCOH_TANK_LIFE, LCOH_TANK_OM)

        pem_econ = project_pem_stack_replacements(df, pem_stack_capex, wacc=LCOH_WACC)
        prod_ann = fixed_prod_ann + pem_econ['stack_ann_total'] + gimp - grev - o2r
        lcoh_prod = prod_ann / max(h2p, 1e-6)

        lcoh_del  = prod_ann / max(h2d, 1e-6)

        stor_ann = comp_ann + tank_ann
        lcsh     = (prod_ann + stor_ann) / max(h2p, 1e-6)

        marker = " ◄ current" if abs(tank_kg - curr_kg) < 1 else ""
        print(f"  {tlabel:>15} {h2p:>8.0f} {h2d:>8.0f} "
              f"{overflow_est:>9.0f} {h2_unmet:>8.0f} {fulfill_pct:>9.1f} "
              f"{lcoh_prod:>10.2f} {lcsh:>8.2f} {tank_ann:>9.0f}{marker}")

        results.append({
            'tank_kg':     tank_kg,
            'autonomy_d':  autonomy,
            'h2_prod':     h2p,
            'h2_del':      h2d,
            'overflow':    overflow_est,
            'h2_unmet':    h2_unmet,
            'fulfill_pct': fulfill_pct,
            'tank_ann':    tank_ann,
            'lcoh_prod':   lcoh_prod,
            'lcoh_del':    lcoh_del,
            'lcsh':        lcsh,
        })

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9,
        'axes.titlesize': 10, 'legend.fontsize': 8,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    tx       = [r['tank_kg']     for r in results]
    tx_idx   = list(range(len(results)))              # integer positions for x-axis
    autox    = [r['autonomy_d']  for r in results]
    h2dy     = [r['h2_del']      for r in results]
    ofy      = [r['overflow']    for r in results]
    fly      = [r['fulfill_pct'] for r in results]
    lcohy    = [r['lcoh_del']    for r in results]
    lcshy    = [r['lcsh']        for r in results]
    tanky    = [r['tank_ann']    for r in results]
    unmety   = [r['h2_unmet']    for r in results]
    try:
        curr_tidx = next(i for i, v in enumerate(tx) if abs(v - curr_kg) < 1)
    except StopIteration:
        curr_tidx = min(range(len(tx)), key=lambda i: abs(tx[i] - curr_kg))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    def _panel(ax, ydata, ylabel, title, color, best='min', secondary=None):
        ax.plot(tx_idx, ydata, 'o-', color=color, lw=1.8, ms=5, zorder=3)
        ax.axvline(curr_tidx, color='black', lw=1.1, ls='--',
                   label=f'Current ({curr_kg:.0f} kg)')
        if best == 'min':
            opt_idx = ydata.index(min(ydata))
        elif best == 'max':
            opt_idx = ydata.index(max(ydata))
        else:
            opt_idx = None

        if opt_idx is not None:
            ax.scatter(tx_idx[opt_idx], ydata[opt_idx], marker='*',
                       color='red', s=130, zorder=5,
                       label=f'Optimum: {tank_labels[opt_idx]}')
        ax.set_xticks(tx_idx)
        ax.set_xticklabels(tank_labels, rotation=45, ha='right', fontsize=7.5)
        ax.set_xlabel('Tank Size (autonomy days at daily demand)')
        ax.set_ylabel(ylabel, color=color)
        ax.set_title(title)
        ax.yaxis.grid(True, linestyle='--', alpha=0.45)
        ax.set_axisbelow(True)
        ax.legend(framealpha=0.85, loc='best', fontsize=7.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if secondary is not None:
            ax2 = ax.twinx()
            ax2.plot(tx_idx, secondary, 's--', color='gray', lw=0.9, ms=4, alpha=0.55)
            ax2.set_ylabel('Tank Ann. CAPEX+O&M (EUR/yr)', color='gray', fontsize=7.5)
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)

    _panel(axes[0, 0], fly,   'Demand Fulfillment (%)',
           'H₂ Demand Fulfillment vs Tank Size\n(plateaus once seasonal buffer is sufficient)',
           '#4477AA', best='max')

    _panel(axes[0, 1], lcohy, 'LCOH_delivery (EUR kg⁻¹ H₂)',
           'LCOH (Delivery Basis) vs Tank Size\n(U-shape: tiny tank = low fulfillment; big = high CAPEX)',
           '#228833', best='min', secondary=tanky)

    _panel(axes[0, 2], lcshy, 'LCSH (EUR kg⁻¹ H₂)',
           'LCSH — Production + Storage vs Tank Size\n(rises with tank due to CAPEX)',
           '#AA55CC', best='min', secondary=tanky)

    _panel(axes[1, 0], ofy,   'Annual H₂ Overflow (kg yr⁻¹)',
           'H₂ Overflow (tank-full waste) vs Tank Size\n(falls then flattens — production ≠ wasted)',
           '#EE6677', best=None)
    axes[1, 0].axhline(0, color='green', lw=0.9, ls=':', alpha=0.7, label='Zero overflow')
    axes[1, 0].legend(fontsize=7.5)

    _panel(axes[1, 1], unmety, 'Annual H₂ Unmet (kg yr⁻¹)',
           'Unmet Demand vs Tank Size\n(falls as tank buffers low-irradiance days)',
           '#FF8C00', best=None)
    axes[1, 1].axhline(0, color='green', lw=0.9, ls=':', alpha=0.7, label='Zero unmet')
    axes[1, 1].legend(fontsize=7.5)

    ax6 = axes[1, 2]
    ax6.plot(tx_idx, lcohy, 'o-', color='#228833', lw=1.8, ms=5, label='LCOH (delivery)')
    ax6.plot(tx_idx, lcshy, 's-', color='#AA55CC', lw=1.8, ms=5, label='LCSH (production+storage)')
    ax6.axvline(curr_tidx, color='black', lw=1.1, ls='--', label=f'Current ({curr_kg:.0f} kg)')
    opt_del_idx  = lcohy.index(min(lcohy))
    opt_lcsh_idx = lcshy.index(min(lcshy))
    ax6.axvline(opt_del_idx,  color='#228833', lw=0.9, ls=':', alpha=0.7,
                label=f'LCOH_del min @ {tank_labels[opt_del_idx]}')
    ax6.axvline(opt_lcsh_idx, color='#AA55CC', lw=0.9, ls=':', alpha=0.7,
                label=f'LCSH min @ {tank_labels[opt_lcsh_idx]}')
    ax6.set_xticks(tx_idx)
    ax6.set_xticklabels(tank_labels, rotation=45, ha='right', fontsize=7.5)
    ax6.set_xlabel('Tank Size (autonomy days at daily demand)')
    ax6.set_ylabel('Levelised Cost (EUR kg⁻¹ H₂)')
    ax6.set_title('LCOH (delivery) vs LCSH — Optimal Tank\n(choose between cost metric priorities)')
    ax6.yaxis.grid(True, linestyle='--', alpha=0.45)
    ax6.set_axisbelow(True)
    ax6.legend(framealpha=0.85, fontsize=7.5)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'Tank_Sensitivity_{loc_label}.{ext}',
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"\n  Figure saved: Tank_Sensitivity_{loc_label}.pdf/.png")
    return results


def plot_cross_location_comparison(all_results, all_lcoh, out_dir):
    """Five publication-quality figures comparing all locations × controllers."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family':        'serif',
        'font.serif':         ['Times New Roman', 'DejaVu Serif'],
        'font.size':          13,
        'axes.titlesize':     14,
        'axes.labelsize':     15,
        'xtick.labelsize':    13,
        'ytick.labelsize':    13,
        'legend.fontsize':    12,
        'axes.linewidth':     0.8,
        'grid.linewidth':     0.5,
        'grid.alpha':         0.4,
        'figure.dpi':         120,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.05,
    })

    LOC_LABELS  = list(all_results.keys())
    CTRL_LABELS = list(all_results[LOC_LABELS[0]].keys())
    LOC_SHORT   = [l.replace('_2021', '') for l in LOC_LABELS]

    COLORS  = [CTRL_COLORS[c] for c in CTRL_LABELS]
    HATCHES = [CTRL_HATCH[c]  for c in CTRL_LABELS]
    N_LOC   = len(LOC_LABELS)
    N_CTRL  = len(CTRL_LABELS)
    W_BAR   = 0.8 / N_CTRL
    OFFS    = (np.arange(N_CTRL) - (N_CTRL - 1) / 2.0) * W_BAR
    X       = np.arange(N_LOC)
    PEM_MAX_KG = H2_PEAK_KGH * 8760.0

    def _switches(df):
        return int(df['n_parallel'].diff().fillna(0).ne(0).sum())

    met = {}
    for loc, ctrl_dfs in all_results.items():
        met[loc] = {}
        for ctrl, df in ctrl_dfs.items():
            met[loc][ctrl] = {
                'profit':        df['profit_eur'].sum(),
                'grid_imp_kwh':  df['grid_bought'].sum() * DT_HOURS / 1000.0,
                'curtailed_kwh': df['P_curtailed_W'].sum() * DT_HOURS / 1000.0,
                'soh_pem':       df['SoH_PEM'].iloc[-1]  * 100.0,
                'soh_pv':        df['SoH_PV'].iloc[-1]   * 100.0,
                'soh_batt':      df['SoH_BATT'].iloc[-1] * 100.0,
                'switches':      _switches(df),
                **all_lcoh[loc][ctrl],
            }

    def _save(fig, name):
        for ext in ('pdf', 'png'):
            fig.savefig(out_dir / f'{name}.{ext}', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

    def _bar_group(ax, key, ylabel, title, fmt='{:.0f}', annotate=False,
                   ref_lines=None):
        """Grouped bars: N_LOC groups, N_CTRL bars each."""
        for ci, ctrl in enumerate(CTRL_LABELS):
            vals = [met[loc][ctrl][key] for loc in LOC_LABELS]
            bars = ax.bar(X + OFFS[ci], vals, width=W_BAR, label=ctrl,
                          color=COLORS[ci], hatch=HATCHES[ci],
                          edgecolor='white', linewidth=0.5)
            if annotate:
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.015,
                            fmt.format(v), ha='center', va='bottom',
                            fontsize=10, fontweight='bold', rotation=90)
        if ref_lines:
            for y_val, label, ls, col in ref_lines:
                ax.axhline(y_val, color=col, lw=0.9, ls=ls, alpha=0.7, label=label)
        ax.set_xticks(X); ax.set_xticklabels(LOC_SHORT)
        ax.set_ylabel(ylabel); ax.set_title(title, pad=6)
        ax.yaxis.grid(True, linestyle='--'); ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))

    _bar_group(axes1[0, 0], 'h2_prod',      'Annual H₂ Production (kg yr⁻¹)',
               'H₂ Production')
    _bar_group(axes1[0, 1], 'cf_pem',       'PEM Capacity Factor (%)',
               'PEM Capacity Factor', fmt='{:.1f}')
    _bar_group(axes1[1, 0], 'grid_imp_kwh', 'Annual Grid Import (kWh yr⁻¹)',
               'Grid Import for Auxiliaries')
    _bar_group(axes1[1, 1], 'profit',       'Annual Profit (EUR yr⁻¹)',
               'Annual Profit (H₂ revenue + grid economy)')

    _h1, _l1 = axes1[0, 0].get_legend_handles_labels()
    fig1.legend(_h1, _l1, loc='lower center', ncol=len(_l1), fontsize=9,
                framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig1.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig1, 'Fig1_H2_Production_Overview')

    fig2, (ax2l, ax2r) = plt.subplots(1, 2, figsize=(11, 5))

    _bar_group(ax2l, 'lcoh', 'LCOH (EUR kg⁻¹ H₂)',
               'LCOH — Production Cost\n(excl. compressor & tank)',
               fmt='{:.2f}', annotate=True)
    _bar_group(ax2r, 'lcsh', 'LCSH (EUR kg⁻¹ H₂)',
               'LCSH — Stored Hydrogen Cost\n(incl. compressor & tank)',
               fmt='{:.2f}', annotate=True)

    _h2, _l2 = ax2l.get_legend_handles_labels()
    fig2.legend(_h2, _l2, loc='lower center', ncol=len(_l2), fontsize=9,
                framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig2.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig2, 'Fig2_LCOH_LCSH')

    fig3, axes3 = plt.subplots(1, N_LOC, figsize=(5.2 * N_LOC, 6.5), sharey=True)
    if N_LOC == 1:
        axes3 = [axes3]

    C_FIXED  = '#5588CC';  H_FIXED  = ''
    C_IMP    = '#DD8833';  H_IMP    = '..'
    C_EXP    = '#66BB44';  H_EXP    = '//'
    C_O2     = '#AA55CC';  H_O2     = 'xx'
    C_STOR   = '#CC5555';  H_STOR   = '\\\\'

    xs_ctrl = np.arange(N_CTRL)
    WB3     = 0.50

    for li, (ax3, loc) in enumerate(zip(axes3, LOC_LABELS)):
        for ci, ctrl in enumerate(CTRL_LABELS):
            m   = met[loc][ctrl]
            h2p = max(m['h2_prod'], 1e-6)

            g_imp = m['grid_import_yr']  / h2p
            g_exp = m['grid_export_yr']  / h2p
            o2r   = m['o2_revenue_yr']   / h2p
            fixed = m['lcoh'] + g_exp + o2r - g_imp

            stor_addon = m['lcsh'] - m['lcoh']

            ax3.bar(xs_ctrl[ci], fixed,      bottom=0,     width=WB3,
                    color=C_FIXED, hatch=H_FIXED, edgecolor='white', lw=0.5,
                    label='Fixed CAPEX+OæM' if (li == 0 and ci == 0) else '')
            ax3.bar(xs_ctrl[ci], g_imp,      bottom=fixed, width=WB3,
                    color=C_IMP,   hatch=H_IMP,   edgecolor='white', lw=0.5,
                    label='Grid Import'            if (li == 0 and ci == 0) else '')
            ax3.bar(xs_ctrl[ci], stor_addon, bottom=fixed + g_imp, width=WB3,
                    color=C_STOR,  hatch=H_STOR,  edgecolor='white', lw=0.5,
                    label='Storage add-on (LCSH)'  if (li == 0 and ci == 0) else '')

            ax3.bar(xs_ctrl[ci], -g_exp, bottom=0,     width=WB3,
                    color=C_EXP, hatch=H_EXP, edgecolor='white', lw=0.5,
                    label='Grid Export Revenue'    if (li == 0 and ci == 0) else '')
            ax3.bar(xs_ctrl[ci], -o2r,  bottom=-g_exp, width=WB3,
                    color=C_O2,  hatch=H_O2,  edgecolor='white', lw=0.5,
                    label='O₂ By-product Revenue' if (li == 0 and ci == 0) else '')

            ax3.scatter(xs_ctrl[ci], m['lcoh'], marker='D', color='black',
                        s=55, zorder=6,
                        label='Net LCOH' if (li == 0 and ci == 0) else '')
            ax3.annotate(f"{m['lcoh']:.1f}", (xs_ctrl[ci], m['lcoh']),
                         xytext=(-9, 0), textcoords='offset points',
                         ha='right', va='center', fontsize=11, fontweight='bold',
                         color='black',
                         bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

            ax3.scatter(xs_ctrl[ci], m['lcsh'], marker='o', color='#B22222',
                        s=50, zorder=6, edgecolors='black', linewidths=0.6,
                        label='Net LCSH' if (li == 0 and ci == 0) else '')
            ax3.annotate(f"{m['lcsh']:.1f}", (xs_ctrl[ci], m['lcsh']),
                         xytext=(9, 0), textcoords='offset points',
                         ha='left', va='center', fontsize=11, fontweight='bold',
                         color='#8B0000',
                         bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))

        ax3.axhline(0, color='black', lw=0.7)
        ax3.set_xticks(xs_ctrl)
        ax3.set_xticklabels(CTRL_LABELS, rotation=15)
        ax3.set_title(LOC_SHORT[li], fontweight='bold')
        ax3.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax3.set_axisbelow(True)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        if li == 0:
            ax3.set_ylabel('EUR per kg H₂')

    leg_handles = [
        mpatches.Patch(fc=C_FIXED, hatch=H_FIXED, ec='gray', label='Fixed CAPEX + O&M'),
        mpatches.Patch(fc=C_IMP,   hatch=H_IMP,   ec='gray', label='Grid Import Cost'),
        mpatches.Patch(fc=C_STOR,  hatch=H_STOR,  ec='gray', label='Storage add-on (LCSH)'),
        mpatches.Patch(fc=C_EXP,   hatch=H_EXP,   ec='gray', label='Grid Export Revenue (−)'),
        mpatches.Patch(fc=C_O2,    hatch=H_O2,    ec='gray', label='O₂ By-product Revenue (−)'),
        plt.Line2D([0], [0], marker='D', color='black',   lw=0, ms=6, label='Net LCOH'),
        plt.Line2D([0], [0], marker='o', color='#CC5555', lw=0, ms=6, label='Net LCSH'),
    ]
    fig3.legend(handles=leg_handles, loc='lower center', ncol=4,
                framealpha=0.92, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig3.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig3, 'Fig3_LCOH_Cost_Breakdown')

    fig4, axes4 = plt.subplots(1, 3, figsize=(13, 5))

    _bar_group(axes4[0], 'switches', 'MOSFET Switching Events yr⁻¹',
               'Switching Events\n(fewer = less hardware fatigue)', fmt='{:.0f}')
    _bar_group(axes4[1], 'soh_pem',  'Final PEM SoH (%)',
               'PEM State of Health\n(end of simulated year)', fmt='{:.2f}')
    _bar_group(axes4[2], 'soh_pv',   'Final PV SoH (%)',
               'PV State of Health\n(end of simulated year)', fmt='{:.3f}')

    for ax4, key in [(axes4[1], 'soh_pem'), (axes4[2], 'soh_pv')]:
        all_vals = [met[loc][ctrl][key] for loc in LOC_LABELS for ctrl in CTRL_LABELS]
        lo = min(all_vals)
        ax4.set_ylim(lo * 0.9985, max(all_vals) * 1.002)

    _h4, _l4 = axes4[0].get_legend_handles_labels()
    fig4.legend(_h4, _l4, loc='lower center', ncol=len(_l4), fontsize=9,
                framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig4.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig4, 'Fig4_Operational_Performance')

    fig5, axes5 = plt.subplots(2, 2, figsize=(11, 8))

    adv_panels = [
        (axes5[0, 0], 'h2_prod',  'Δ H₂ Production (kg yr⁻¹)',
         'H₂ Production Gain vs MOO',        +1.0, '{:+.0f}'),
        (axes5[0, 1], 'switches', 'Δ Switching Events yr⁻¹',
         'Switching Reduction vs MOO\n(negative = fewer switches)',  -1.0, '{:+.0f}'),
        (axes5[1, 0], 'lcoh',     'Δ LCOH (EUR kg⁻¹)',
         'LCOH Reduction vs MOO\n(negative = cheaper)',              -1.0, '{:+.2f}'),
        (axes5[1, 1], 'soh_pem',  'Δ Final PEM SoH (pp)',
         'PEM Longevity Improvement vs MOO',      +1.0, '{:+.3f}'),
    ]

    X5      = np.arange(N_LOC)
    OTHERS  = [c for c in CTRL_LABELS if c != 'MOO']     # every controller vs the MOO baseline
    NB      = len(OTHERS)
    W5      = 0.8 / NB
    OFF5    = (np.arange(NB) - (NB - 1) / 2.0) * W5

    for ax5, key, ylabel, title, sign, fmt in adv_panels:
        all_d = []
        for bi, ctrl in enumerate(OTHERS):
            dvals = [sign * (met[loc][ctrl][key] - met[loc]['MOO'][key]) for loc in LOC_LABELS]
            all_d += dvals
            bars = ax5.bar(X5 + OFF5[bi], dvals, width=W5, label=f'{ctrl} vs MOO',
                           color=CTRL_COLORS[ctrl], hatch=CTRL_HATCH[ctrl], edgecolor='white', lw=0.5)
            max_abs = max((abs(v) for v in all_d), default=1.0) or 1.0
            for bar, v in zip(bars, dvals):
                if abs(v) < 1e-9:
                    continue
                y_text = v + max_abs * 0.04 if v >= 0 else v - max_abs * 0.04
                ax5.text(bar.get_x() + bar.get_width() / 2, y_text, fmt.format(v),
                         ha='center', va=('bottom' if v >= 0 else 'top'),
                         fontsize=9, fontweight='bold')

        ax5.axhline(0, color='black', lw=0.8)
        ax5.set_xticks(X5); ax5.set_xticklabels(LOC_SHORT)
        ax5.set_ylabel(ylabel); ax5.set_title(title, pad=6)
        ax5.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax5.set_axisbelow(True)
        ax5.legend(framealpha=0.85)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)

    fig5.tight_layout()
    _save(fig5, 'Fig5_RL_Advantage')

    print(f"\n  Publication figures saved to: {out_dir}")
    for name in ['Fig1_H2_Production_Overview', 'Fig2_LCOH_LCSH',
                 'Fig3_LCOH_Cost_Breakdown', 'Fig4_Operational_Performance',
                 'Fig5_RL_Advantage']:
        print(f"    {name}.pdf  +  .png")


def plot_cross_location_technical(all_results, out_dir):
    """Additional TECHNICAL cross-location figures (physics/operation, not economics):
    Fig6  — Switching ↔ MPPT-loss trade-off (the Pareto that defines RL's value) Fig7
    — Technical efficiency: MPP shortfall, Solar-to-H2 efficiency, PV curtailment Fig8
    — Monthly H2 production profile per location (seasonal / climate signature) Fig9
    — Component State-of-Health: PEM, PV, Battery (longevity) Saved as PDF (vector) +
    PNG (300 dpi) in out_dir.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    LOCS   = list(all_results.keys())
    CTRL   = list(all_results[LOCS[0]].keys())
    SHORT  = [l.replace('_2021', '') for l in LOCS]
    COLORS = [CTRL_COLORS[c] for c in CTRL]
    HATCH  = [CTRL_HATCH[c]  for c in CTRL]
    MARK   = ['o', 's', '^', 'D', 'v']            # one per location
    N      = len(LOCS); NC = len(CTRL); X = np.arange(N)
    WB     = 0.8 / NC; OFF = (np.arange(NC) - (NC - 1) / 2.0) * WB

    dfs = {loc: all_results[loc] for loc in LOCS}   # all_results[loc] is already {label: df}

    def _switches(df): return int(df['n_parallel'].diff().fillna(0).ne(0).sum())
    def _mppt(df):     return df['mppt_loss'].sum() * DT_HOURS / 1000.0
    def _sth(df):
        v = df['StH'][df['StH'] > 0]; return float(v.mean() * 100.0) if len(v) else 0.0
    def _curt(df):     return df['P_curtailed_W'].sum() * DT_HOURS / 1000.0 if 'P_curtailed_W' in df.columns else 0.0
    def _soh(df, key): return df[key].iloc[-1] * 100.0

    def _save(fig, name):
        for ext in ('pdf', 'png'):
            fig.savefig(out_dir / f'{name}.{ext}', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

    def _grouped(ax, fn, ylabel, title, zoom=False):
        allv = []
        for ci, ctrl in enumerate(CTRL):
            vals = [fn(dfs[loc][ctrl]) for loc in LOCS]; allv += vals
            ax.bar(X + OFF[ci], vals, width=WB, label=ctrl, color=COLORS[ci],
                   hatch=HATCH[ci], edgecolor='white', linewidth=0.5)
        ax.set_xticks(X); ax.set_xticklabels(SHORT)
        ax.set_ylabel(ylabel); ax.set_title(title, pad=6)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if zoom:                                   # zoom near-identical SoH values
            lo, hi = min(allv), max(allv)
            ax.set_ylim(lo - (hi - lo) * 0.6 - 1e-6, hi + (hi - lo) * 0.6 + 1e-6)

    def _fig_legend(fig, ax, ncol):
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=ncol, fontsize=9,
                   framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig6, ax = plt.subplots(figsize=(8.5, 6))
    _mppt_all = []
    for li, loc in enumerate(LOCS):
        for ci, ctrl in enumerate(CTRL):
            df = dfs[loc][ctrl]
            ax.scatter(_switches(df), _mppt(df), s=150, color=COLORS[ci],
                       marker=MARK[li % len(MARK)], edgecolors='k', linewidths=0.8, zorder=3)
    ax.set_xlabel('Annual MOSFET switching events  (→ more hardware fatigue)')
    ax.set_ylabel('Annual MPP shortfall (kWh)  (capped at stack power)')
    ax.grid(alpha=0.3)
    ctrl_leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[i],
                       markersize=11, markeredgecolor='k', label=CTRL[i]) for i in range(NC)]
    loc_leg  = [Line2D([0], [0], marker=MARK[i % len(MARK)], color='w', markerfacecolor='gray',
                       markersize=11, markeredgecolor='k', label=SHORT[i]) for i in range(N)]
    _l1 = ax.legend(handles=ctrl_leg, title='Controller (colour)', loc='upper left',
                    bbox_to_anchor=(1.02, 1.0), framealpha=0.9, borderaxespad=0.0)
    ax.add_artist(_l1)
    ax.legend(handles=loc_leg, title='Location (marker)', loc='upper left',
              bbox_to_anchor=(1.02, 0.60), framealpha=0.9, borderaxespad=0.0)
    fig6.tight_layout()
    _save(fig6, 'Fig6_Switching_MPPT_Tradeoff')

    fig7, a7 = plt.subplots(1, 3, figsize=(15, 5))
    _grouped(a7[0], _mppt, 'MPP shortfall (kWh yr⁻¹)', 'MPP Shortfall\n(off-MPP operation)')
    _grouped(a7[1], _sth,  'Mean StH efficiency (%)',   'Solar-to-Hydrogen Efficiency', zoom=True)
    _grouped(a7[2], _curt, 'Curtailed PV (kWh yr⁻¹)',   'Curtailed PV Energy\n(surplus not used)')
    _fig_legend(fig7, a7[0], 3)
    fig7.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig7, 'Fig7_Technical_Efficiency')

    n_h = min(len(next(iter(dfs.values()))['MOO']), STEPS_PER_YEAR)
    month = (pd.to_datetime('2021-01-01') + pd.to_timedelta(np.arange(n_h) * DT_HOURS, 'h')).month.to_numpy()
    mx    = np.arange(1, 13)
    fig8, a8 = plt.subplots(1, N, figsize=(5.3 * N, 5), sharey=True)
    if N == 1:
        a8 = [a8]
    for li, loc in enumerate(LOCS):
        for ci, ctrl in enumerate(CTRL):
            h2 = dfs[loc][ctrl]['H2_prod'].values[:n_h]
            monthly = np.array([h2[month == m].sum() for m in mx])
            a8[li].bar(mx + OFF[ci], monthly, width=WB, color=COLORS[ci],
                       hatch=HATCH[ci], edgecolor='white', linewidth=0.3, label=ctrl)
        a8[li].set_title(SHORT[li], fontweight='bold')
        a8[li].set_xlabel('Month'); a8[li].set_xticks(mx)
        a8[li].yaxis.grid(True, linestyle='--', alpha=0.5); a8[li].set_axisbelow(True)
        a8[li].spines['top'].set_visible(False); a8[li].spines['right'].set_visible(False)
    a8[0].set_ylabel('Monthly H₂ production (kg)')
    _fig_legend(fig8, a8[0], 3)
    fig8.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig8, 'Fig8_Monthly_H2_Profile')

    fig9, a9 = plt.subplots(1, 3, figsize=(15, 5))
    _grouped(a9[0], lambda d: _soh(d, 'SoH_PEM'),  'Final PEM SoH (%)',     'PEM Electrolyser', zoom=True)
    _grouped(a9[1], lambda d: _soh(d, 'SoH_PV'),   'Final PV SoH (%)',      'PV Array',         zoom=True)
    _grouped(a9[2], lambda d: _soh(d, 'SoH_BATT'), 'Final Battery SoH (%)', 'Battery',          zoom=True)
    _fig_legend(fig9, a9[0], 3)
    fig9.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig9, 'Fig9_Component_Health')

    print(f"\n  Technical figures saved to: {out_dir}")
    for name in ['Fig6_Switching_MPPT_Tradeoff', 'Fig7_Technical_Efficiency',
                 'Fig8_Monthly_H2_Profile', 'Fig9_Component_Health']:
        print(f"    {name}.pdf  +  .png")


def plot_computational_cost(rl_model, irr, temp, price, out_dir, n_samples=150):
    """Per-DECISION inference time for the FOUR compared controllers — the on-line cost
    once trained: • MOO      — full perturb-&-observe optimisation over all MOSFET
    positions every step • MOO+Hyst — the same optimisation, then a deadband gate on
    the resulting move • MOO+Pen  — the same optimisation with the switching charge
    added inside the score • RL Agent — one policy-network forward pass The three MOO
    variants are TIMED SEPARATELY rather than assumed equal: the deadband and the
    switching term are cheap, but "cheap" is a measurement, not an assumption, and the
    point of the figure is that only the learned policy changes the COST CLASS (it
    does not enumerate the ladder at all).
    """
    import matplotlib.pyplot as plt
    import time as _time
    from sklearn.ensemble import RandomForestRegressor

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    irr = np.asarray(irr, float); temp = np.asarray(temp, float)
    day = np.where(irr >= 80)[0]
    rng = np.random.default_rng(0)
    samp = rng.choice(day, size=min(n_samples, len(day)), replace=False)

    N_SER_PV  = N_SERIES_PV
    N_SER_PEM = PVPEMEnv.N_SERIES_PEM
    pem_I = np.linspace(0, I_PEM(i_cell, 1), 100)
    pem_V = V_PEM(60 + 273.15, p_cat, p_an, delta_mem, A_cell_pem, a_an, a_cat,
                  pem_I, i_0_an, i_0_cat, T_ref_pem, i_lim, N_SER_PEM)
    pemVl, pemIl = pem_V.tolist(), pem_I.tolist()

    def _moo_decision(g, t, prev_mosf=None, lam=0.0):
        params = [Iph_ref, Is_ref1, Is_ref2, Is_refi, Rs, Rsh, n1, n2, ni,
                  rho_contact, A_cell, aSi_thickness, TCO_resistivity, t]
        eff = calculate_effective_irradiance(g, 0.1 * g)
        best_score, best_n, best_mosf = float('inf'), PVPEMEnv.BASE_PARALLEL, 0
        for mosf in range(7):
            n_par = PVPEMEnv.BASE_PARALLEL + mosf * PVPEMEnv.PARALLEL_PER_MOS
            vols, curs, _, _ = perturb_observe_series(params, eff, N_SER_PV, n_par)
            if not vols:
                continue
            Vm, Im, Pm = find_mppt(vols, curs)
            pt = find_intersection_point(vols, curs, pemVl, pemIl)
            if pt:
                Pod = pt[0] * pt[1]; ml = mppt_shortfall(Pm, Pod)
                fl = np.clip(Pod / PEM_capacity, 0, 1) if Pod > 20 else 0
                h2 = h2_rate_faraday(pt[1], Pod)
                sth = float(np.clip((h2 * LHV) / (eff * N_SER_PV * n_par * pv_module_area), 0, 0.185)) if g > 80 else 0.0
                sc = -(W_H2 * (h2 / H2_PEAK_KGH) + W_STH * (sth / 0.185) - W_MPPT * (ml / MPPT_NORM))
                if lam and prev_mosf is not None:
                    sc += lam * abs(mosf - prev_mosf)
                if sc < best_score:
                    best_score, best_n, best_mosf = sc, n_par, mosf
        return best_n, best_mosf

    def _moo_plain(g, t):
        return _moo_decision(g, t)[0]

    def _moo_pen(g, t, prev_mosf):
        return _moo_decision(g, t, prev_mosf=prev_mosf, lam=MOO_PEN_LAMBDA)[0]

    def _moo_hyst(g, t, prev_mosf):
        n_par, mosf = _moo_decision(g, t)
        if mosf != prev_mosf and abs(mosf - prev_mosf) < MOO_DEADBAND:
            return PVPEMEnv.BASE_PARALLEL + prev_mosf * PVPEMEnv.PARALLEL_PER_MOS
        return n_par

    X = [[irr[i], temp[i]] for i in samp]
    _dec = [_moo_decision(irr[i], temp[i]) for i in samp]
    y    = [d[0] for d in _dec]
    _prev_mosf = [_dec[0][1]] + [d[1] for d in _dec[:-1]]
    if INCLUDE_RF_MOO:
        rf = RandomForestRegressor(n_estimators=30, max_depth=10, min_samples_leaf=3, random_state=42)
        rf.fit(X, y)

    env = PVPEMEnv(irr, temp, price, eval_mode=True); obs, _ = env.reset()
    obs_list = [obs]
    for _ in range(30):
        obs, _, _, done, _ = env.step(env.action_space.sample())
        obs_list.append(obs)
        if done:
            break

    def _per_call(fn, args, repeats=5):
        for a in args[:5]:
            fn(a)                                            # warm-up
        best = float('inf')
        for _ in range(repeats):
            t0 = _time.perf_counter()
            for a in args:
                fn(a)
            best = min(best, (_time.perf_counter() - t0) / len(args))
        return best                                          # seconds / decision

    _gt      = [(irr[i], temp[i]) for i in samp]
    _gt_prev = [(g, t, p) for (g, t), p in zip(_gt, _prev_mosf)]

    times = {}
    times['MOO']      = _per_call(lambda a: _moo_plain(*a), _gt)
    times['MOO+Hyst'] = _per_call(lambda a: _moo_hyst(*a),  _gt_prev)
    times['MOO+Pen']  = _per_call(lambda a: _moo_pen(*a),   _gt_prev)
    if INCLUDE_RF_MOO:
        times['MOO+RF'] = _per_call(lambda a: rf.predict([a]), [[irr[i], temp[i]] for i in samp])
    try:
        times['RL Agent'] = _per_call(lambda o: rl_model.predict(o, deterministic=True), obs_list)
    except Exception as _e:
        print(f"  (RL timing skipped: {_e})"); times['RL Agent'] = float('nan')
    t_moo = times['MOO']

    t_rl_net = float('nan')
    try:
        import torch as _th
        _pol = rl_model.policy
        _pol.set_training_mode(False)
        _obs_t = _th.as_tensor(np.asarray(obs_list, dtype=np.float32))
        with _th.no_grad():
            _pol._predict(_obs_t[:1], deterministic=True)      # warm-up
            t_rl_net = _per_call(lambda o: _pol._predict(o, deterministic=True),
                                 [_obs_t[i:i+1] for i in range(len(obs_list))])
    except Exception as _e:
        print(f"  (policy-only timing skipped: {_e})")

    def _fmt(t):
        return f"{t*1e6:.1f} us" if t < 1e-3 else f"{t*1e3:.2f} ms"
    _DESC = {'MOO':      'full optimisation',
             'MOO+Hyst': 'optimisation + deadband',
             'MOO+Pen':  'optimisation + switch charge',
             'MOO+RF':   'RF prediction',
             'RL Agent': 'SB3 predict'}
    print(f"\n{'='*70}\n  COMPUTATIONAL COST — time per control decision (once trained)\n{'='*70}")
    for c, s in times.items():
        if np.isnan(s):
            continue
        rel = "" if c == 'MOO' else f"   ({t_moo/s:.1f}x faster than MOO)"
        print(f"  {c:<9} ({_DESC[c]:<28}): {_fmt(s)}{rel}")
    if not np.isnan(t_rl_net):
        print(f"  {'RL Agent':<9} ({'policy forward pass':<28}): {_fmt(t_rl_net)}"
              f"   ({t_moo/t_rl_net:.1f}x faster than MOO)")
    print(f"  All three MOO variants enumerate the same {len(MOSFET_OPTIONS) if 'MOSFET_OPTIONS' in dir() else 7} "
          f"ladder positions per step, so their")
    print(f"  cost grows with the ladder; the deadband and the switching charge add only a")
    print(f"  comparison and one addition. The policy's cost does not grow with the ladder.")
    print(f"  NOTE: the PV sweep is vectorised (~36x faster than the original per-voltage")
    print(f"        loop), so every MOO variant is far cheaper than in earlier versions.")
    print(f"{'='*70}")

    _WRAP = {'MOO':      'MOO\n(optimise every step)',
             'MOO+Hyst': 'MOO+Hyst\n(optimise + deadband)',
             'MOO+Pen':  'MOO+Pen\n(optimise + switch charge)',
             'MOO+RF':   'MOO+RF\n(RF prediction)',
             'RL Agent': 'RL Agent\n(NN inference)'}
    ctrls   = [c for c in CTRL_ORDER if c in times]
    labels  = [_WRAP[c] for c in ctrls]
    secs    = [times[c] for c in ctrls]
    cols    = [CTRL_COLORS[c] for c in ctrls]
    fig, ax = plt.subplots(figsize=(2.0 * len(ctrls) + 1.5, 5.5))
    heights = [s * 1e3 for s in secs]                        # ms
    bars = ax.bar(range(len(ctrls)), heights, color=cols, edgecolor='white', width=0.62)
    ax.set_yscale('log'); ax.set_ylabel('Time per control decision  (ms, log scale)')
    ax.set_xticks(range(len(ctrls))); ax.set_xticklabels(labels, fontsize=10)
    for i, (b, s) in enumerate(zip(bars, secs)):
        if np.isnan(s):
            continue
        lab = _fmt(s) + (f"\n{t_moo/s:.1f}x faster" if ctrls[i] != 'MOO' else "")
        ax.text(b.get_x() + b.get_width()/2, s*1e3*1.25, lab, ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(axis='y', ls='--', alpha=0.4); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_ylim(top=np.nanmax(heights) * 4)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'computational_time_comparison.{ext}', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Figure saved: computational_time_comparison.pdf/.png")
    if not np.isnan(t_rl_net):
        times['RL Agent (policy only)'] = t_rl_net
    return times


def plot_controller_radar(all_results, all_lcoh, out_dir, decision_times=None):
    """At-a-glance multi-criteria radar over the compared controllers, averaged across
    locations.
    """
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    LOCS = list(all_results.keys())
    CTRL = list(all_results[LOCS[0]].keys())
    COLS = [CTRL_COLORS[c] for c in CTRL]

    def _sth_pct(df):
        v = df['StH'][df['StH'] > 0]
        return float(v.mean() * 100.0) if len(v) else 0.0

    def _agg(ctrl):
        return dict(
            h2   = np.mean([all_results[l][ctrl]['H2_prod'].sum() for l in LOCS]),
            sw   = np.mean([int(all_results[l][ctrl]['n_parallel'].diff().fillna(0).ne(0).sum()) for l in LOCS]),
            pem  = np.mean([all_results[l][ctrl]['SoH_PEM'].iloc[-1] * 100 for l in LOCS]),
            pv   = np.mean([all_results[l][ctrl]['SoH_PV'].iloc[-1] * 100 for l in LOCS]),
            sth  = np.mean([_sth_pct(all_results[l][ctrl]) for l in LOCS]),
            cur  = np.mean([all_results[l][ctrl]['P_curtailed_W'].sum() * DT_HOURS / 1e6
                            for l in LOCS]),
            mppt = np.mean([all_results[l][ctrl]['mppt_loss'].sum() * DT_HOURS / 1000 for l in LOCS]),
            lcoh = np.mean([all_lcoh[l][ctrl]['lcoh'] for l in LOCS]),
        )
    M = {c: _agg(c) for c in CTRL}
    if decision_times is not None:
        spd = {c: float(decision_times.get(c, decision_times.get('MOO', np.nan))) for c in CTRL}
        if not np.all(np.isfinite(list(spd.values()))):
            spd = None
    else:
        spd = None

    axes_def = [('H₂ production', 'h2', True), ('Switching gentleness', 'sw', False),
                ('PEM longevity', 'pem', True), ('PV longevity', 'pv', True),
                ('Solar-to-H₂ efficiency', 'sth', True),
                ('Curtailment avoidance', 'cur', False),
                ('Economy (low LCOH)', 'lcoh', False),
                ('MPP tracking', 'mppt', False)]
    if spd is not None:
        axes_def.append(('Decision speed', 'spd', False))

    norm = {}
    for label, key, hib in axes_def:
        v = np.array([(spd[c] if key == 'spd' else M[c][key]) for c in CTRL], float)
        lo, hi = v.min(), v.max()
        n = np.ones_like(v) if hi - lo < 1e-12 else ((v - lo) / (hi - lo) if hib else (hi - v) / (hi - lo))
        norm[label] = 0.2 + 0.8 * n

    labels = [a[0] for a in axes_def]
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(10.0, 10.0), subplot_kw=dict(polar=True))
    for ci, c in enumerate(CTRL):
        vals = [norm[l][ci] for l in labels]; vals += vals[:1]
        ax.plot(ang, vals, color=COLS[ci], lw=2.2, label=c)
        ax.fill(ang, vals, color=COLS[ci], alpha=0.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([])
    ax.set_yticklabels([]); ax.set_ylim(0, 1.05)
    for a, lab in zip(ang[:-1], labels):
        ca, sa = np.cos(a), np.sin(a)
        ha = 'left' if ca > 0.25 else ('right' if ca < -0.25 else 'center')
        va = 'bottom' if sa > 0.25 else ('top' if sa < -0.25 else 'center')
        ax.text(a, 1.18, lab, ha=ha, va=va, fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=len(CTRL),
              framealpha=0.9, fontsize=12)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'Fig10_Controller_Radar.{ext}', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print("  Radar figure saved: Fig10_Controller_Radar.pdf/.png")


def print_cross_location_lcoh(all_lcoh):
    """Print a cross-location LCOH / LCSH comparison table."""
    locations   = list(all_lcoh.keys())
    controllers = [c for c in CTRL_ORDER if c in all_lcoh[locations[0]]]
    W = 110
    print(f"\n\n{'#'*W}")
    print(f"#  CROSS-LOCATION SUMMARY  —  LCOH / LCSH  (EUR/kg H2, same {hourly_H2_target*24:.0f} kg/day demand everywhere)")
    print(f"{'#'*W}")

    # ---------- H2 production & CF_PEM ----------
    print(f"\n  {'':20}", end='')
    for loc in locations:
        print(f"  {loc:>28}", end='')
    print()
    print(f"  {'Metric / Controller':<20}", end='')
    for loc in locations:
        for ctrl in controllers:
            print(f"  {ctrl:>8}", end='')
    print()
    print(f"  {'-'*(W-2)}")

    def row(title, key, fmt='.0f'):
        print(f"  {title:<20}", end='')
        for loc in locations:
            for ctrl in controllers:
                val = all_lcoh[loc][ctrl][key]
                print(f"  {val:>8{fmt}}", end='')
        print()

    row('H2 prod (kg/yr)',    'h2_prod',        '.0f')
    row('CF_PEM (%)',         'cf_pem',          '.1f')
    row('Grid export (EUR/yr)','grid_export_yr', '.0f')
    row('O2 revenue (EUR/yr)','o2_revenue_yr',  '.0f')
    row('LCOH (EUR/kg)',      'lcoh',            '.2f')
    row('LCSH (EUR/kg)',      'lcsh',            '.2f')

    # ---------- LCOH delta vs MOO per location ----------
    print(f"\n  RL vs MOO  ΔLCOH (EUR/kg) — negative = RL is cheaper")
    print(f"  {'-'*(W-2)}")
    for loc in locations:
        moo_lcoh = all_lcoh[loc]['MOO']['lcoh']
        rl_lcoh  = all_lcoh[loc]['RL Agent']['lcoh']
        delta    = rl_lcoh - moo_lcoh
        pct      = 100.0 * delta / max(moo_lcoh, 1e-6)
        print(f"  {loc:<25}  MOO={moo_lcoh:.2f}  RL={rl_lcoh:.2f}  "
              f"Δ={delta:+.2f} EUR/kg  ({pct:+.1f}%)")

    print(f"\n  LCOH = production-chain cost / H2_produced  (no compressor, no tank)")
    print(f"  LCSH = LCOH + compressor + H2 tank cost  / same H2_produced")
    print(f"{'#'*W}\n")


# =============================================================================
# %% DATA LOADING
# =============================================================================

def load_data_from_excel(excel_path, sheet_name):
    print(f"Loading: {excel_path} | requested site/sheet: {sheet_name}")
    if not _os.path.isfile(excel_path):
        raise FileNotFoundError(
            f"15-min input workbook not found: {excel_path}\n"
            f"Tried: {_15MIN_EXCEL_CANDIDATES}"
        )
    xls = pd.ExcelFile(excel_path)
    site_token = str(sheet_name).split('_')[0].lower()
    site_sheets = [sh for sh in xls.sheet_names if site_token in sh.lower()]
    preferred_15 = [sh for sh in site_sheets
                    if ('15' in sh.lower() or 'quarter' in sh.lower() or 'min' in sh.lower())]
    if preferred_15:
        resolved_sheet = preferred_15[0]
    elif sheet_name in xls.sheet_names:
        resolved_sheet = sheet_name
    elif site_sheets:
        resolved_sheet = site_sheets[0]
    else:
        raise KeyError(f"No sheet for site '{site_token}' in {xls.sheet_names}")
    print(f"  Resolved 15-min sheet: {resolved_sheet}")
    df = pd.read_excel(xls, sheet_name=resolved_sheet)
    if str(DAY_NUMBER).lower() == 'all' and len(df) != STEPS_PER_YEAR:
        raise ValueError(
            f"15-min full-year study requires {STEPS_PER_YEAR} rows (2021 = 35040), "
            f"but sheet '{resolved_sheet}' has {len(df)}. Refusing to run an hourly or "
            "incomplete dataset as a 15-min resolution experiment."
        )

    def _find_col(frame, keyword, label):
        matches = [c for c in frame.columns if keyword.lower() in c.lower()]
        if not matches:
            raise KeyError(
                f"No column containing '{keyword}' found in sheet '{label}'.\n"
                f"  Available columns: {list(frame.columns)}"
            )
        if len(matches) > 1:
            print(f"  INFO: multiple '{keyword}' columns in '{label}' {matches}; using '{matches[0]}'.")
        return matches[0]

    irr_col  = _find_col(df, 'Irradiance', resolved_sheet)
    temp_col = _find_col(df, 'Temperature', resolved_sheet)
    price_matches = [c for c in df.columns if 'price' in c.lower()]
    price_col = price_matches[0] if price_matches else None

    irr_vals = pd.to_numeric(df[irr_col], errors='coerce').fillna(0.0).to_numpy(float)
    temp_raw = pd.to_numeric(df[temp_col], errors='coerce').fillna(15.0).to_numpy(float)
    temp_vals = np.where(temp_raw < 150.0, temp_raw, temp_raw - 273.15)
    if price_col is not None:
        price_vals = pd.to_numeric(df[price_col], errors='coerce').fillna(100.0).to_numpy(float)
        print(f"  Columns used: irr='{irr_col}'  temp='{temp_col}'  price='{price_col}'")
    else:
        price_vals = np.full(len(df), 100.0, dtype=float)
        print(f"  Columns used: irr='{irr_col}'  temp='{temp_col}'  price=fallback 100 EUR/MWh")

    raw_ghi = float(np.sum(np.clip(irr_vals, 0.0, None)) * DT_HOURS / 1000.0)

    if RESOLUTION_INPUT_MODE == 'hourly_mean_matched':
        hourly_sheet = None
        if sheet_name in xls.sheet_names and sheet_name != resolved_sheet:
            hourly_sheet = sheet_name
        else:
            non15 = [sh for sh in site_sheets
                     if sh != resolved_sheet and not ('15' in sh.lower() or 'quarter' in sh.lower() or 'min' in sh.lower())]
            for sh in non15:
                probe = pd.read_excel(xls, sheet_name=sh, nrows=5)
                hourly_sheet = sh
                break
        if hourly_sheet is None:
            raise KeyError(
                f"RESOLUTION_INPUT_MODE='hourly_mean_matched' requires an hourly reference "
                f"sheet for {sheet_name}, but only found {site_sheets}."
            )
        hdf = pd.read_excel(xls, sheet_name=hourly_sheet)
        if len(hdf) != 8760:
            raise ValueError(
                f"Hourly reference sheet '{hourly_sheet}' must contain 8760 rows; got {len(hdf)}."
            )
        h_irr_col = _find_col(hdf, 'Irradiance', hourly_sheet)
        h_temp_col = _find_col(hdf, 'Temperature', hourly_sheet)
        h_price_matches = [c for c in hdf.columns if 'price' in c.lower()]
        h_price_col = h_price_matches[0] if h_price_matches else None

        hG = pd.to_numeric(hdf[h_irr_col], errors='coerce').fillna(0.0).to_numpy(float)
        hTraw = pd.to_numeric(hdf[h_temp_col], errors='coerce').fillna(15.0).to_numpy(float)
        hT = np.where(hTraw < 150.0, hTraw, hTraw - 273.15)
        hP = (pd.to_numeric(hdf[h_price_col], errors='coerce').fillna(100.0).to_numpy(float)
              if h_price_col is not None else np.full(8760, 100.0, dtype=float))

        if len(irr_vals) != 4 * len(hG):
            raise ValueError("15-min/hourly reference lengths are not 4:1; cannot create matched sensitivity input.")

        qG = np.clip(irr_vals, 0.0, None).reshape(-1, 4)
        qmean = qG.mean(axis=1)
        target = np.clip(hG, 0.0, 1200.0)
        for hh in range(len(target)):
            tgt = float(target[hh])
            if tgt <= 0.0:
                qG[hh, :] = 0.0
                continue
            if qmean[hh] > 1e-12:
                q = qG[hh, :] * (tgt / qmean[hh])
            else:
                q = np.full(4, tgt, dtype=float)

            required_sum = 4.0 * tgt
            q = np.clip(q, 0.0, 1200.0)
            for _ in range(8):
                deficit = required_sum - float(q.sum())
                if abs(deficit) < 1e-9:
                    break
                if deficit > 0.0:
                    headroom = np.maximum(0.0, 1200.0 - q)
                    hsum = float(headroom.sum())
                    if hsum <= 1e-12:
                        break
                    q += deficit * headroom / hsum
                    q = np.minimum(q, 1200.0)
                else:
                    positive = np.maximum(q, 0.0)
                    psum = float(positive.sum())
                    if psum <= 1e-12:
                        break
                    q += deficit * positive / psum
                    q = np.maximum(q, 0.0)

            residual = required_sum - float(q.sum())
            if abs(residual) > 1e-7:
                if residual > 0:
                    candidates = np.where(q < 1200.0 - 1e-9)[0]
                else:
                    candidates = np.where(q > 1e-9)[0]
                if len(candidates):
                    jj = int(candidates[0])
                    q[jj] = np.clip(q[jj] + residual, 0.0, 1200.0)

            if abs(float(q.mean()) - tgt) > 1e-6:
                raise RuntimeError(
                    f"Could not preserve hourly irradiance mean at hour {hh}: "
                    f"target={tgt}, obtained={q.mean()}."
                )
            qG[hh, :] = q
        irr_vals = qG.reshape(-1)

        qT = temp_vals.reshape(-1, 4)
        qT += (hT - qT.mean(axis=1))[:, None]
        temp_vals = qT.reshape(-1)

        price_vals = np.repeat(hP, 4)

        matched_ghi = float(np.sum(irr_vals) * DT_HOURS / 1000.0)
        hourly_ghi = float(np.sum(np.clip(hG, 0.0, 1200.0)) / 1000.0)
        simulated_ghi = float(np.sum(np.clip(irr_vals, 0.0, 1200.0)) * DT_HOURS / 1000.0)
        print(f"  RESOLUTION INPUT: hourly-mean matched to '{hourly_sheet}'")
        print(f"    raw 15-min GHI       = {raw_ghi:.1f} kWh/m2/yr")
        print(f"    hourly reference     = {hourly_ghi:.1f} kWh/m2/yr")
        print(f"    matched 15-min GHI   = {matched_ghi:.1f} kWh/m2/yr")
        print(f"    simulator-bounded GHI= {simulated_ghi:.1f} kWh/m2/yr")
        print(f"    matched peak G       = {float(np.max(irr_vals)):.1f} W/m2")
        if abs(matched_ghi - hourly_ghi) > 0.05:
            raise RuntimeError("Hourly-mean matching failed to conserve annual GHI.")
        if abs(simulated_ghi - hourly_ghi) > 0.05:
            raise RuntimeError("Simulator irradiance bound changed annual GHI after matching.")
    elif RESOLUTION_INPUT_MODE == 'raw_15min':
        print(f"  RESOLUTION INPUT: NATIVE 15-min sheet '{resolved_sheet}' (unchanged)")
        print(f"    native 15-min GHI = {raw_ghi:.1f} kWh/m2/yr")

        hourly_sheet = None
        non15 = [sh for sh in site_sheets
                 if sh != resolved_sheet and not ('15' in sh.lower() or 'quarter' in sh.lower() or 'min' in sh.lower())]
        canonical_hourly = f"{str(sheet_name).split('_')[0]}_2021"
        if canonical_hourly in xls.sheet_names:
            hourly_sheet = canonical_hourly
        elif non15:
            hourly_sheet = non15[0]

        if hourly_sheet is not None:
            hdf = pd.read_excel(xls, sheet_name=hourly_sheet)
            if len(hdf) != 8760:
                raise ValueError(f"Hourly audit sheet '{hourly_sheet}' must contain 8760 rows; got {len(hdf)}.")
            h_irr_col = _find_col(hdf, 'Irradiance', hourly_sheet)
            h_temp_col = _find_col(hdf, 'Temperature', hourly_sheet)
            h_price_matches = [c for c in hdf.columns if 'price' in c.lower()]
            h_price_col = h_price_matches[0] if h_price_matches else None
            hG = pd.to_numeric(hdf[h_irr_col], errors='coerce').fillna(0.0).to_numpy(float)
            hTraw = pd.to_numeric(hdf[h_temp_col], errors='coerce').fillna(15.0).to_numpy(float)
            hT = np.where(hTraw < 150.0, hTraw, hTraw - 273.15)
            hP = (pd.to_numeric(hdf[h_price_col], errors='coerce').fillna(100.0).to_numpy(float)
                  if h_price_col is not None else np.full(8760, 100.0, dtype=float))
            if len(irr_vals) != 4 * len(hG):
                raise ValueError(f"Native 15-min sheet has {len(irr_vals)} rows but hourly sheet has {len(hG)}; expected exact 4:1 ratio.")

            hourly_ghi = float(np.sum(np.clip(hG, 0.0, 1200.0)) / 1000.0)
            simulator_native_ghi = float(np.sum(np.clip(irr_vals, 0.0, 1200.0)) * DT_HOURS / 1000.0)
            annual_diff_pct = 100.0 * (simulator_native_ghi - hourly_ghi) / max(hourly_ghi, 1e-12)
            qGmean = np.clip(irr_vals, 0.0, 1200.0).reshape(-1, 4).mean(axis=1)
            qTmean = temp_vals.reshape(-1, 4).mean(axis=1)
            dG = qGmean - np.clip(hG, 0.0, 1200.0)
            dT = qTmean - hT

            print(f"    hourly audit sheet = '{hourly_sheet}'")
            print(f"    hourly-sheet GHI    = {hourly_ghi:.1f} kWh/m2/yr")
            print(f"    simulator native GHI= {simulator_native_ghi:.1f} kWh/m2/yr")
            print(f"    annual GHI difference= {annual_diff_pct:+.3f}%")
            print(f"    quartet-vs-hourly G: MAE={np.mean(np.abs(dG)):.3f}, RMSE={np.sqrt(np.mean(dG**2)):.3f}, max={np.max(np.abs(dG)):.3f} W/m2")
            print(f"    quartet-vs-hourly T: MAE={np.mean(np.abs(dT)):.3f}, max={np.max(np.abs(dT)):.3f} C")
            if h_price_col is not None:
                qP = price_vals.reshape(-1, 4)
                dP = qP - hP[:, None]
                print(f"    native-vs-hourly price: max |dP|={np.max(np.abs(dP)):.6g} EUR/MWh")
            if abs(annual_diff_pct) > 5.0:
                raise RuntimeError("Native 15-min and hourly sheets differ by >5% annual GHI; this would confound the resolution comparison.")
            elif abs(annual_diff_pct) > 1.0:
                print("    WARNING: native 15-min and hourly sheets differ by >1% annual GHI; report this input difference explicitly.")
            else:
                print("    INPUT CHECK PASS: annual solar resource is consistent between native 15-min and hourly sheets.")
        else:
            print("    NOTE: no 8760-row hourly audit sheet found; native 15-min data are still used unchanged.")
    else:
        raise ValueError("RESOLUTION_INPUT_MODE must be 'hourly_mean_matched' or 'raw_15min'.")

    data = [
        {'irradiance': float(irr_vals[i]),
         'temperature': float(temp_vals[i]),
         'price_mwh': float(price_vals[i])}
        for i in range(len(irr_vals))
    ]
    print(f"  -> {len(data)} 15-minute records loaded.")
    return data

# =============================================================================
# %% MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    t_start = time.time()
    print(f'\nFINAL 15-MIN BUILD V6 DAILY-NATIVE: DT={DT_HOURS:g} h, {STEPS_PER_DAY} steps/day, {STEPS_PER_YEAR} steps/year, forecast points={FORECAST_HORIZONS_H} h (row offsets={FORECAST_OFFSETS_STEPS})')
    print(f'PPO: n_steps={RL_CONFIG["n_steps"]}, batch={RL_CONFIG["batch_size"]}, gamma={RL_CONFIG["gamma"]:.6f}, gae_lambda={RL_CONFIG["gae_lambda"]:.6f}')
    print('  RESOLUTION-SCALING AUDIT:')
    print('    continuous x DT : H2, StH, MPPT, underuse, keep-warm, chiller, warm-operation')
    print('    x DT regularizer : switching (matches frozen hourly implementation)')
    print('    event, no x DT  : cold-start; temperature-cycle uses total variation')
    print('    physical x DT   : H2 mass, grid/battery energy, thermal inertia, base/calendar degradation')
    print('    RCCD            : base degradation x DT and |Delta fl|/DT -> total-variation invariant')
    print(f'    demand          : {daily_H2_target/float(STEPS_PER_DAY):.4f} kg/15-min direct path; {daily_H2_target:.2f} kg settled every {STEPS_PER_DAY} steps (daily)')
    print(f'    lambda_sw step  : {LAMBDA_SWITCH:.3f} x DT={DT_HOURS:.2f} -> {LAMBDA_SWITCH*DT_HOURS:.5f} per rung-step')


    # ---- DAILY demand contract -- locked for this resolution comparison ---------
    DEMAND_MODE = 'daily'
    print("\n" + "="*60)
    print("  DEMAND CONTRACT MODE -- LOCKED")
    print("="*60)
    print(f"  daily: {daily_H2_target:.2f} kg/day settled every {STEPS_PER_DAY} quarter-hour steps")
    print(f"  direct target per 15-min step: {daily_H2_target / STEPS_PER_DAY:.4f} kg")
    print("  -> This matches the 60-min DAILY production study; only the control/data resolution changes.\n")
    _demand_folder = 'Daily_demand'
    OUT_DIR = _os.path.join(OUT_DIR, _demand_folder)
    _os.makedirs(OUT_DIR, exist_ok=True)
    print(f"  -> Demand-specific output root: {OUT_DIR}\n")

    # ---- Location definitions — sheet names must match tabs in the Excel file ----
    ALL_LOCATIONS = {
        'Lund_2021':    'Lund_15min',
        'Prague_2021':  'Prague_15min',
        'Odeillo_2021': 'Odeillo_15min',
    }

    # ---- Location selection prompt — asked once, mirrors the demand-mode prompt ----
    _loc_items = list(ALL_LOCATIONS.items())
    print("\n" + "="*60)
    print("  LOCATION SELECTION")
    print("="*60)
    print("  [0] ALL locations  — run Lund, Prague and Odeillo (full study)")
    for _i, (_lbl, _sheet) in enumerate(_loc_items, start=1):
        print(f"  [{_i}] {_lbl}")
    print("="*60)
    while True:
        _lc = input(f"  Select location [0-{len(_loc_items)}, default=0]: ").strip()
        if _lc in ('', '0'):
            LOCATIONS = dict(_loc_items)
            break
        elif _lc.isdigit() and 1 <= int(_lc) <= len(_loc_items):
            _lbl, _sheet = _loc_items[int(_lc) - 1]
            LOCATIONS = {_lbl: _sheet}
            break
        else:
            print(f"  Please enter a number between 0 and {len(_loc_items)}.")
    if len(LOCATIONS) == 1:
        print(f"  -> Running SINGLE location: {next(iter(LOCATIONS))}")
        print("     (cross-location comparison figures need >=2 locations → skipped)\n")
    else:
        print(f"  -> Running ALL {len(LOCATIONS)} locations.\n")

    # ---- Hardware sizing is the same everywhere — print once ----
    pv_peak_kw = print_system_sizing()

    # ---- Polarisation curve plot — physics-only, location-independent ----
    _polar_dir = Path(OUT_DIR) / 'Polarization'
    _polar_dir.mkdir(parents=True, exist_ok=True)
    print("Generating PV-PEM polarization curves (once)...")
    plot_polarization_curves(_polar_dir)

    # ---- Loop over locations ----
    all_lcoh    = {}   # {location_label: results_dict from calculate_lcoh}
    all_results = {}
    sizing_series = {}

    for loc_label, sheet in LOCATIONS.items():
        print(f"\n{'#'*70}")
        print(f"#  LOCATION: {loc_label}  (sheet: {sheet})")
        print(f"{'#'*70}")

        period_id = loc_label if str(DAY_NUMBER).lower() == 'all' else f'{loc_label}_day{DAY_NUMBER}'
        run_dir   = Path(OUT_DIR) / f'Results_{period_id}'
        plots_dir = run_dir / 'Plots'
        data_dir  = run_dir / 'Data'
        model_dir = run_dir / 'Model'
        for d in [plots_dir, data_dir, model_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"  Period : {period_id}")
        print(f"  RL steps: {RL_TIMESTEPS:,}")
        print(f"  Output  : {run_dir}")

        data  = load_data_from_excel(EXCEL_FILE, sheet)
        irr   = [x['irradiance']  for x in data]
        temp  = [x['temperature'] for x in data]
        price = [x['price_mwh']   for x in data]

        _ghi_kwh_m2 = float(np.sum(np.clip(np.asarray(irr, dtype=float), 0.0, 1200.0))
                            * DT_HOURS / 1000.0)
        _daylight_h = float(np.sum(np.asarray(irr, dtype=float) >= 80.0) * DT_HOURS)
        print(f"  INPUT AUDIT: native integrated GHI={_ghi_kwh_m2:.1f} kWh/m2/yr, "
              f"G>=80 duration={_daylight_h:.1f} h/yr")

        # ---- per-site hardware from the sizing study (applied to ALL controllers) ----
        site_pv_scale = 1.0
        set_string_ladder(1.08)          # default array unless the site overrides it
        _sz = site_sizing_for(loc_label)
        if USE_SITE_SIZING and _sz is None:
            raise SystemExit(
                "SITE_SIZING has no entry matching '%s'. Known keys: %s. "
                "Add one or fix the location label -- refusing to run with the "
                "default array/tank, which would invalidate the comparison."
                % (loc_label, list(SITE_SIZING)))
        if USE_SITE_SIZING and _sz is not None:
            _ratio, _tank, _batt = _sz
            set_string_ladder(_ratio)
            _b, _s, _m = N_PAR_BASE, N_PAR_STEP, N_PAR_MAX
            storage_capacity       = float(_tank)                   # H2 tank (physical, used by the sim)
            LCOH_TANK_KG_INSTALLED = float(_tank)                    # H2 tank (cost side — keep LCSH consistent)
            set_initial_storage()
            BATT_CAPACITY_WH   = float(_batt * 1000.0)
            BATT_MAX_P_W       = BATT_CAPACITY_WH * 1.0              # hold C-rate at 1.00
            pv_peak_kw         = _m * N_SERIES_PV * PV_MODULE_W / 1000.0   # N_ser x N_par x W_mod
            print(f"  SITE SIZING: PV {_ratio:g}x -> {_m} strings "
                  f"({_b}-{_m} in steps of {_s}) = {pv_peak_kw:.0f} kWp, "
                  f"tank {_tank} kg, battery {_batt} kWh")

        if RUN_SEED_STUDY:
            seed_variance_study(irr, temp, price, period_id, plots_dir, seeds=SEED_LIST)
            continue

        if RUN_PARETO_STUDY:
            pareto_switching_study(irr, temp, price, period_id, plots_dir,
                                   lambdas=PARETO_LAMBDAS, seeds=PARETO_SEEDS,
                                   zero_underuse=PARETO_ZERO_UNDERUSE)
            continue

        if RUN_REWARD_STUDY:
            reward_weight_study(irr, temp, price, period_id, plots_dir,
                                corners=REWARD_CORNERS, seeds=REWARD_SEEDS)
            continue

        if RUN_PV_SWEEP:
            _sz = run_dir / 'Sizing_Study'
            sizing_study(irr, temp, price, period_id, _sz, sweep='battery')
            sizing_study(irr, temp, price, period_id, _sz, sweep='tank')
            sizing_series[loc_label] = (irr, temp, price)
            continue

        apply_cyclic_storage(irr, temp, price, period_id)

        df_moo = run_baselines(irr, temp, price, use_rf=False, hysteresis=False, pv_scale=site_pv_scale)
        df_moo.to_csv(data_dir / f'moo_results_{period_id}.csv', index=False)
        print(f"  -> MOO done.")

        df_moo_hyst = None
        if INCLUDE_HYSTERESIS_MOO:
            df_moo_hyst = run_baselines(irr, temp, price, use_rf=False,
                                        hysteresis=True, lambda_sw=0.0,
                                        pv_scale=site_pv_scale)
            df_moo_hyst.to_csv(data_dir / f'hyst_results_{period_id}.csv', index=False)
            print(f"  -> MOO+Hyst (deadband={MOO_DEADBAND}) done.")

        df_moo_pen = None
        if INCLUDE_ALWAYSWARM_MOO:
            df_moo_pen = run_baselines(irr, temp, price, use_rf=False, hysteresis=False,
                                       lambda_sw=0.0, pv_scale=site_pv_scale,
                                       prewarm='always_warm')
            df_moo_pen.to_csv(data_dir / f'alwayswarm_results_{period_id}.csv',
                              index=False)
            print("  -> MOO+AlwaysWarm (constant %.0f C standby) done."
                  % KEEPWARM_LEVELS_C[-1])
        elif INCLUDE_PENALISED_MOO:
            df_moo_pen = run_baselines(irr, temp, price, use_rf=False,
                                       hysteresis=False, lambda_sw=MOO_PEN_LAMBDA,
                                       pv_scale=site_pv_scale)
            df_moo_pen.to_csv(data_dir / f'pen_results_{period_id}.csv', index=False)
            print(f"  -> MOO+Pen (lambda_sw={MOO_PEN_LAMBDA}) done.")

        df_moo_pw = None
        if INCLUDE_PREWARM_MOO:
            df_moo_pw = run_baselines(irr, temp, price, use_rf=False, hysteresis=False,
                                      lambda_sw=0.0, pv_scale=site_pv_scale,
                                      prewarm=PREWARM_RULE)
            df_moo_pw.to_csv(data_dir / f'prewarm_results_{period_id}.csv', index=False)
            print(f"  -> MOO+PreWarm (rule='{PREWARM_RULE}') done.")

        df_moo_hpw = None
        if INCLUDE_HYST_PREWARM_MOO:
            df_moo_hpw = run_baselines(irr, temp, price, use_rf=False, hysteresis=True,
                                       lambda_sw=0.0, pv_scale=site_pv_scale,
                                       prewarm=PREWARM_RULE)
            df_moo_hpw.to_csv(data_dir / f'hystprewarm_results_{period_id}.csv', index=False)
            print(f"  -> MOO+Hyst+PreWarm (deadband={MOO_DEADBAND}, rule='{PREWARM_RULE}') done.")

        # ---- 15-min PRE-FLIGHT CHECK BEFORE ANY PPO TRAINING --------------------
        print("\n" + "="*92)
        print("15-MIN DETERMINISTIC PRE-FLIGHT (before PPO)")
        print(f"  forecast horizons: {FORECAST_HORIZONS_H} h | row offsets: {FORECAST_OFFSETS_STEPS}")
        print(f"  daily contract: {daily_H2_target:.2f} kg settled every {STEPS_PER_DAY} steps; direct target={daily_H2_target/STEPS_PER_DAY:.4f} kg/step")
        print(f"  reward switch regularizer: lambda={LAMBDA_SWITCH:.3f} x DT={DT_HOURS:.2f} "
              f"= {LAMBDA_SWITCH*DT_HOURS:.5f} per rung-step")
        print(f"  {'Controller':<22}{'H2 kg/yr':>11}{'Switches':>11}{'Cold':>8}{'Restart T C':>13}{'PEM SoH %':>11}")
        print("  " + "-"*78)
        _preflight_ctrls = [
            ("MOO", df_moo),
            ("MOO+Hyst", df_moo_hyst),
            ("MOO+PreWarm", df_moo_pw),
            ("MOO+Hyst+PreWarm", df_moo_hpw),
            ("MOO+AlwaysWarm" if INCLUDE_ALWAYSWARM_MOO else "MOO+Pen", df_moo_pen),
        ]
        for _lbl, _df in _preflight_ctrls:
            if _df is None:
                continue
            _h2 = float(pd.to_numeric(_df['H2_prod'], errors='coerce').fillna(0.0).sum())
            _n = pd.to_numeric(_df['n_parallel'], errors='coerce').to_numpy(float)
            _sw = int(np.count_nonzero(np.diff(_n) != 0)) if len(_n) > 1 else 0
            _cold = int(count_cold_starts(_df))
            _rt = float(mean_restart_temperature(_df))
            _soh = 100.0 * float(pd.to_numeric(_df['SoH_PEM'], errors='coerce').dropna().iloc[-1])
            print(f"  {_lbl:<22}{_h2:>11.0f}{_sw:>11d}{_cold:>8d}{_rt:>13.1f}{_soh:>11.3f}")
        print("="*92)
        if df_moo_pw is not None and df_moo_hpw is not None:
            _pw_cold = count_cold_starts(df_moo_pw)
            _hpw_cold = count_cold_starts(df_moo_hpw)
            if max(_pw_cold, _hpw_cold) > 20:
                print("  WARNING: pre-warm baselines still show many true <50 C restarts. "
                      "Inspect the native 15-min meteorology / forecast sequence before trusting the run.")
            else:
                print("  PreWarm cold-start count is in the expected low range; "
                      "restart-event SoH above may still be non-zero because warm restarts carry 5e-6 damage.")
        print()

        df_rf = None
        if INCLUDE_RF_MOO:
            df_rf = run_baselines(irr, temp, price, use_rf=True, pv_scale=site_pv_scale)
            df_rf.to_csv(data_dir / f'rf_results_{period_id}.csv', index=False)
            print(f"  -> MOO+RF done.")

        model = None
        if REUSE_SAVED_MODELS:
            model = load_saved_rl_model(period_id, model_dir)
            if model is None:
                print("  -> REUSE_SAVED_MODELS is True but no saved model was found "
                      "for this location — training a fresh one.")
        if model is None:
            model = train_rl(irr, temp, price, period_id, model_dir, plots_dir, pv_scale=site_pv_scale)
        df_rl = evaluate_rl(model, irr, temp, price, period_id, data_dir, plots_dir, pv_scale=site_pv_scale)

        _kw = dict(df_hyst=df_moo_hyst, df_pen=df_moo_pen,
                   df_pw=df_moo_pw, df_hpw=df_moo_hpw)
        plot_3way_comparison(df_moo, df_rf, df_rl, period_id, plots_dir,
                             pv_scale=site_pv_scale, **_kw)
        export_3way_excel(df_moo, df_rf, df_rl, period_id, data_dir, **_kw)
        plot_energy_stacks(df_moo, df_rf, df_rl, period_id, plots_dir, **_kw)

        print_summary(df_moo, df_rf, df_rl, period_id, **_kw)
        _ctrls = _ctrl_dfs(df_moo, df_rf, df_rl, df_moo_hyst, df_moo_pen,
                           df_moo_pw, df_moo_hpw)
        print_controller_comparison(_ctrls, period_id)
        audit_controller_energy_allocation(_ctrls, period_id)
        audit_rl_reward_components(df_rl, period_id)

        all_lcoh[loc_label]    = calculate_lcoh(df_moo, df_rf, df_rl, pv_peak_kw, **_kw)
        all_results[loc_label] = _ctrls

        if RUN_BATTERY_SENSITIVITY:
            batt_dir = run_dir / 'Battery_Sensitivity'
            battery_sensitivity_analysis(irr, temp, price, pv_peak_kw,
                                         loc_label, batt_dir)

        if RUN_TANK_SENSITIVITY:
            tank_dir = run_dir / 'Tank_Sensitivity'
            tank_sensitivity_analysis_rl(model, irr, temp, price, pv_peak_kw,
                                         loc_label, tank_dir)

        if RUN_TORNADO:
            tornado_dir = run_dir / 'Tornado_Sensitivity'
            tornado_sensitivity_analysis(irr, temp, price, pv_peak_kw,
                                         loc_label, tornado_dir, model=model,
                                         pv_scale=site_pv_scale)

    if RUN_SEED_STUDY or RUN_PARETO_STUDY or RUN_REWARD_STUDY or RUN_PV_SWEEP:
        elapsed = time.time() - t_start
        _which = ('Seed-variance' if RUN_SEED_STUDY else
                  'Pareto-front'  if RUN_PARETO_STUDY else
                  'Reward-weight' if RUN_REWARD_STUDY else 'PV-oversizing')
        if RUN_PV_SWEEP:
            plot_sizing_pareto(OUT_DIR, batt_min=500.0, tol=0.03,
                               pv_override={'Odeillo': 1.25},
                               tank_override={'Odeillo': 1400},
                               series=sizing_series)
        print(f"\n{_which} study complete. Total runtime: {elapsed/60:.1f} min")
        print("Results (CSV + figures) saved under each run's Plots folder.")
        raise SystemExit

    figs_dir = Path(OUT_DIR) / 'CrossLocation_Figures'
    print(f"\nTiming per-decision computational cost (MOO vs MOO+Hyst vs MOO+Pen vs RL)...")
    decision_times = plot_computational_cost(model, irr, temp, price, figs_dir)

    # ---- Cross-location summaries & figures (only meaningful for >=2 locations) ----
    if len(all_results) >= 2:
        print_cross_location_lcoh(all_lcoh)

        print(f"\nGenerating cross-location publication figures -> {figs_dir}")
        plot_cross_location_comparison(all_results, all_lcoh, figs_dir)
        plot_cross_location_technical(all_results, figs_dir)
        plot_controller_radar(all_results, all_lcoh, figs_dir, decision_times)
    else:
        print("\nSingle-location run — cross-location summary and figures skipped.")

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed/60:.1f} min")
    print(f"All results saved under: {OUT_DIR}")
