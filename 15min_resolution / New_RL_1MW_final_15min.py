# -*- coding: utf-8 -*-
"""
FINAL 15-MIN DRL PV-PEM CONTROL  —  1 MW SYSTEM
===========================================
Five-controller, event-aligned 15-minute resolution study — V6 DAILY + NATIVE-15MIN + FROZEN-REWARD BUILD
for a direct-coupled PV-PEM green hydrogen system.

Run in Spyder: press F5 (Run file)
Results saved to: <script_dir>/Results/Results_<SHEET_NAME>/
"""

# =============================================================================
# %% INSTALL DEPENDENCIES (Google Colab only -- skip if running locally)
# =============================================================================
# Uncomment and run this cell first in Colab, then re-run the rest:
# !pip install stable-baselines3 gymnasium scikit-learn scipy tqdm openpyxl -q

# =============================================================================
# %% USER CONFIGURATION -- Edit these before running
# =============================================================================
import os as _os
import sys as _sys

# Detect environment: Colab uploads files to /content/
_in_colab = 'google.colab' in _sys.modules
try:
    _SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = '/content'  # Google Colab

# 15-minute meteorological input. The resolver tries dedicated 15-min files first and
# only then the historical workbook name. load_data_from_excel() refuses a full-year
# 8760-row hourly sheet, so an hourly file cannot silently masquerade as a 15-min study.
_15MIN_EXCEL_CANDIDATES = [
    _os.path.join(_SCRIPT_DIR, 'Irradiance_values_example_new_15min.xlsx'),
    _os.path.join(_SCRIPT_DIR, 'Irradiance_values_15min.xlsx'),
    _os.path.join(_SCRIPT_DIR, 'Irradiance_values_example_new.xlsx'),
]
EXCEL_FILE = next((p for p in _15MIN_EXCEL_CANDIDATES if _os.path.isfile(p)),
                  _15MIN_EXCEL_CANDIDATES[0])
SHEET_NAME   = 'Prague_2021'   # display/default site; 15-min sheet aliases are auto-resolved
DAY_NUMBER   = 'all'            # 'all' or an integer day number

# --- PHYSICAL TIME GRID: FINAL 15-MINUTE RESOLUTION ------------------------------
DT_HOURS       = 0.25
# For the paper's timestep sensitivity, use the real 15-min intra-hour shape but
# force each four-sample block to have the SAME hourly mean irradiance/temperature
# and price as the final hourly Prague input. This isolates timestep/sub-hourly
# variability from a different annual resource. Set to 'raw_15min' only for a
# separate meteorological-resolution study, not for the 60-vs-15 min sensitivity.
RESOLUTION_INPUT_MODE = 'raw_15min'  # FINAL: use the native 35040-row 15-min sheet without altering it

STEPS_PER_HOUR = int(round(1.0 / DT_HOURS))
STEPS_PER_DAY  = int(round(24.0 / DT_HOURS))
STEPS_PER_YEAR = int(round(8760.0 / DT_HOURS))
if (STEPS_PER_HOUR, STEPS_PER_DAY, STEPS_PER_YEAR) != (4, 96, 35040):
    raise RuntimeError('15-min time-grid constants are inconsistent.')

RL_TIMESTEPS = int(STEPS_PER_YEAR * 60)   # 60 PHYSICAL years = 2,102,400 control steps
                                 # 80 was the principled figure but too slow to iterate on, so
                                 # 60 is the deliberate compromise (1.5x the original budget
                                 # against a 3x larger action space).

# Set True to SKIP RL training and reuse the saved rl_model_<location>.zip found in each
# Results_<loc>/Model/ folder from a previous run.
REUSE_SAVED_MODELS = False

# --- MAIN-PIPELINE TRAINING SEED -------------------------------------------------
# train_rl() previously passed no seed to PPO() at all, so every "final" run (the one
# that actually produces the numbers reported for the paper) used whatever unseeded
# state numpy/torch happened to start in -- not reproducible, and there was no way to
# know afterwards which run you'd gotten. This project's own thermal-management
# behaviour is documented as seed-bimodal (reward-weight study notes), so an unseeded
# main run silently samples from that bimodality with no record of which side it landed
# on. Change RL_SEED to explicitly try a different one; the value actually used is
# printed at the start of every training run.
RL_SEED = 1

# --- SEED-VARIANCE STUDY (reviewer robustness check) ---
# When True, for each selected location the script trains the RL agent once per seed in
# SEED_LIST, evaluates each, and prints/saves mean +/- std of the headline metrics (switches,
# PEM SoH, H2, grid, profit, demand-met, StH).
RUN_SEED_STUDY = False
SEED_LIST      = (0, 1, 2)

# --- PARETO FRONT (switching vs yield/profit) ---
# When True, for each selected location the script retrains the RL agent once per switching-
# fatigue weight in PARETO_LAMBDAS (fixed seed to isolate lambda), records switches / H2 /
# profit / grid, and saves a CSV + frontier plot.
RUN_PARETO_STUDY = False
PARETO_LAMBDAS   = (0.12, 0.13, 0.14, 0.15)   # archived final switching-coefficient bracket
PARETO_SEEDS     = (0, 1, 2)                        # median over seeds de-noises the switching axis
# The under-use penalty w_u (W_UNDERUSE) also suppresses switching (it parks the agent at high
# string count), so it confounds a pure lambda_sw sweep.
PARETO_ZERO_UNDERUSE = False

# --- FINAL REWARD SCALARS ---------------------------------------------------------
# The corrected Faraday/LHV reward study is complete. A 25-year / 2-seed screen fixed
# the thermal shaping terms, followed by a 60-year / 3-seed confirmation over
# lambda_sw = 0.12, 0.13, 0.14 and 0.15. The final selection is lambda_sw = 0.13:
# it is the lowest tested switching penalty for which ALL confirmation seeds remain in
# the warm-restart / three-stack-replacement regime. Values >=0.14 further suppress
# switching but do not improve the replacement class and reduce mean H2 production;
# lambda_sw = 0.12 is rejected because its three seeds span 3-5 replacements and
# substantially higher lifetime cost.
#
# Final retained shaping scalars:
#   W_UNDERUSE = W_KEEPWARM = W_CHILLER = 0.10
#   W_TCYCLE = 0.04, W_COLDSTART = 0.80, W_WARMOPS = 0.08
#   LAMBDA_SWITCH = 0.13
#
# The reward study is disabled for production runs. The archived bracket below is kept
# only for reproducibility; toggling RUN_REWARD_STUDY=True intentionally skips the normal
# multi-location controller pipeline.
RUN_REWARD_STUDY = False
# FINAL FROZEN REWARD. The event-aligned cold-start formulation has passed Lund, Prague
# and Odeillo validation. The 15-min study changes time discretisation only; no reward
# scalar is retuned here.

REWARD_CORNERS = [
    # (lambda_sw, w_u, w_coldstart, w_tcycle, w_chiller, w_warmops, label)
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

# PPO settings used for the SWEEP ONLY, applied on top of RL_CONFIG and restored afterwards.
REWARD_STUDY_PPO = {}

# --- HYSTERESIS / DEADBAND MOO BASELINE (reviewer point 1: "engineering floor") ---
# When True, run_baselines (MOO and MOO+RF) holds the previous string config unless the greedy
# optimum moves by >= MOO_DEADBAND string-steps, suppressing adjacent-config chatter WITHOUT any
# learning.
MOO_HYSTERESIS = False
MOO_DEADBAND   = 2      # switch only if |Δconfig| >= this many 25-string steps

# --- SWITCHING-PENALISED MOO BASELINE --------------------------------------------
# The soft-penalty counterpart of the deadband.
MOO_LAMBDA_SW = 0.0     # 0.0 => plain greedy MOO. Set per-call, not globally; the main
                        # loop passes LAMBDA_SWITCH for the penalised variant only.

# =========================== THE FIVE CONTROLLERS ================================
# Restructured 2026-08-06 into a 2x2 ablation plus the learned policy, so each baseline
# isolates one actuator and the DRL is measured against the strongest hand-designed rule:
#
#                       | no thermal rule      | pre-warm rule
#   -------------------+----------------------+----------------------
#   greedy strings     | MOO                  | MOO+PreWarm
#   deadband strings   | MOO+Hyst             | MOO+Hyst+PreWarm
#   -------------------+----------------------+----------------------
#   learned, both      |                 DRL
#
# MOO+Hyst isolates switching suppression, MOO+PreWarm isolates thermal anticipation,
# MOO+Hyst+PreWarm is the ceiling of hand engineering (both rules, no learning), and the
# DRL's margin over THAT is what the learning is worth.
#
# MOO+Pen is retired from the main comparison (2026-08-06). It occupied the same cell as
# MOO+Hyst -- switching suppression without a thermal action -- while needing an apology
# for its decoupled lambda (see MOO_PEN_LAMBDA), so the deadband represents that cell more
# cleanly. Set this True to reproduce it for the supplement.
INCLUDE_PENALISED_MOO = False

# The switching charge used by the MOO+Pen baseline. Defined separately from the DRL's
# LAMBDA_SWITCH so the two CAN be decoupled. DECOUPLED 2026-08-03: in the historical
# calibration, lambda_sw=0.15 was equal to the then-current DRL setting and the Lund run showed MOO+Pen pinned in the 445-655-string band for 98% of the year,
# NEVER reaching above 760 of the 340-970 ladder (verified directly from the run's n_parallel
# trajectory), because a myopic per-step optimizer can only pay a switching cost the SAME step's
# objective covers, unlike the DRL's bootstrapped value function, which can justify a costly
# move for a return that arrives later. Demand met collapsed to 88.1% and PEM SoH was the worst
# of all four controllers despite the fewest switches. 0.02 is a reasoned starting point (roughly
# an order of magnitude below the DRL's lambda), NOT a swept value -- rerun and check the
# n_parallel distribution no longer pins below the top third of the ladder before trusting it.
MOO_PEN_LAMBDA = 0.02

# --- RULE-BASED PRE-WARM MOO BASELINE (5th controller) ---------------------------
# Greedy string search + a hand-designed warm-standby rule on the same heater the DRL agent
# commands (see prewarm_setpoint_rule). Its purpose is to answer the sharpest objection to
# the durability result: the DRL is the ONLY controller in the 4-way comparison that can act
# on stack temperature, and ~97 % of its state-of-health advantage comes from the per-restart
# term, so "state-awareness protects the stack" and "a heater protects the stack" are not
# separated without this control. Cheap to run -- deterministic, no training, one pass/site.
INCLUDE_PREWARM_MOO = True
PREWARM_RULE = 'anticipatory'   # 'anticipatory' = the 2-input engineering rule (hold 60 C if
                                #   any irradiance above cut-in is forecast within the horizon).
                                # 'distilled'    = the depth-3 tree read off the trained agent's
                                #   own keep-warm action at 99 % fidelity (xai_9). Run BOTH: the
                                #   first bounds what a rule can do, the second checks the
                                #   distillation is faithful enough to deploy.


# The archived 'distilled' rule was extracted from hourly forecast indices and is not
# resolution-transferable without refitting. The final engineering comparator is the
# physical anticipatory rule, which naturally extends to the full 4-h / 16-step horizon.
if PREWARM_RULE == 'distilled':
    raise SystemExit(
        "PREWARM_RULE='distilled' is not valid for the 15-min sensitivity without refitting. "
        "Use PREWARM_RULE='anticipatory'."
    )

# Both rules at once: the deadband on the strings AND the pre-warm rule on the heater. This
# is the controller the DRL actually has to beat -- everything the agent can do, done by
# hand. If its stack health matches the DRL's, the contribution is the RULE (which the
# distillation gives you) rather than the policy; if a gap remains, that gap is the value of
# learning the trade-off instead of being told it. Deterministic, so one pass per site.
INCLUDE_HYST_PREWARM_MOO = True

# Deadband MOO — switching suppression with no thermal action.
INCLUDE_HYSTERESIS_MOO = True
# MOO+RF reproduced greedy MOO EXACTLY at every site (identical switches, H2, SoH, LCOH):
# its RF narrows the search to the predicted rung +/-2, a window that always contains MOO's
# optimum, so it selects the same configuration. It therefore adds a column to every table
# without adding information. Off by default -> the comparison is 4-way (MOO, MOO+Hyst,
# MOO+Pen, DRL). The decision-latency benchmark still times the surrogate, which is the one
# place it says something MOO does not.
INCLUDE_RF_MOO = False

# --- SLOW ANALYSIS-ONLY STUDIES (off during iteration, on for final figures) -----
#   The battery-size, H2-tank-size and LCOH tornado sweeps each re-simulate the
#   plant many times per location and dominate the runtime. Keep them OFF while
#   experimenting; flip to True only when regenerating the final paper figures.
RUN_BATTERY_SENSITIVITY = False   # covered by the PV x battery sizing study
RUN_TANK_SENSITIVITY    = False   # covered by the PV x tank sizing study
RUN_TORNADO             = True    # LCOH parameter sensitivity (WACC/PEM/electricity/...) — keep
# One-off PV-oversizing sweep (MOO baseline only) to find the optimal PV/PEM ratio
# WITH the battery, before committing to a final array size. When True the pipeline
# runs only this sweep per location and skips training/evaluation.
RUN_PV_SWEEP            = False   # sizing done; SITE_SIZING below holds the result
# Per-site hardware from the sizing study: (PV oversizing x, H2 tank kg, battery kWh). When
# USE_SITE_SIZING is True the location loop applies each site's own PV, H2 tank and battery, so
# ONE run does all three at their optimised sizes.
USE_SITE_SIZING = True   # MUST be True for the reward study too, or it trains on the
                         # default 1.08x / 4320 kg plant instead of the final design
SITE_SIZING = {   # RE-SWEPT 2026-08-06 on the post-Faraday production model at the lowered
                  # 78 kg/day demand (17 % PEM capacity factor). Acceptance: demand >= 98 %,
                  # self-sufficiency >= 87 %, and the tank must return to its opening stock by
                  # year end (SUSTAINABLE). Greedy MOO baseline, no hysteresis, no pre-warm --
                  # the plant is specified by a controller with no free parameter, so the
                  # five-way comparison is five controllers on ONE plant rather than five
                  # controller+plant pairs. (PV oversizing x, H2 tank kg, battery kWh);
                  # opening stock S0 is half the tank, applied in solve_cyclic_storage.
                  #   site      PV x   tank    batt    S0   demand%  self-suf%  LCOH E/kg
                  #   Lund      1.75   7800    550    3900    98.09     88.9      13.06
                  #   Odeillo   1.25   1600    450     800    98.32     92.0      12.30
                  #   Prague    1.75   5300    550    2650    98.28     91.1      13.17
                  # Every optimum is INTERIOR to the swept grid (PV 1.0-2.5x), so no corner
                  # solutions and no need to widen and re-sweep.
                  # SANITY: against the previous 90 kg/day pre-Faraday sweep (Lund 2.00/8200,
                  # Prague 1.75/5700, Odeillo 1.25/1400) the plant barely moved -- the 11-19 %
                  # production loss and the 13 % demand cut very nearly cancelled, which is the
                  # confirmation that the 0.87 scaling used to set 78 kg/day was right.
    'Lund_2021':    (1.75, 7800, 550),
    'Prague_2021':  (1.75, 5300, 550),
    'Odeillo_2021': (1.25, 1600, 450),
}

def site_sizing_for(label):
    """Per-site (pv_ratio, tank_kg, batt_kWh), matched on the SITE name.

    SITE_SIZING is keyed by the hourly labels ('Prague_2021'), but the 15-minute
    study uses 'Prague_15min'. A plain `label in SITE_SIZING` test therefore fails
    for every 15-minute location and the run silently falls back to the default
    1.08x array and 4320 kg tank -- i.e. it simulates a different plant and the
    comparison against the hourly results is meaningless. Matching on the site
    name keeps both resolutions on the same hardware.
    """
    if label in SITE_SIZING:
        return SITE_SIZING[label]
    site = label.split('_')[0]
    for k, v in SITE_SIZING.items():
        if k.split('_')[0] == site:
            return v
    return None

PV_MODULE_W    = 412.5    # module nameplate at STC (W). 450 x 66/72: same cell, six fewer
                          # of them. Used for capacity accounting, so every per-site string
                          # count changes with it and the sizing sweep must be rerun.
PV_MIN_FRAC    = 0.35     # smallest switch position as a fraction of the installed array
                          # 0.625 -> 0.42 -> 0.35.


def site_string_config(ratio, pem_w=None, n_series=None, module_w=PV_MODULE_W,
                       n_positions=7, min_frac=PV_MIN_FRAC):
    """Per-site MOSFET string ladder for a PV oversizing `ratio`.

    installed strings  N_par_max = ratio * PEM_W / (N_series * module_W)
    switch positions   n_positions values from min_frac*N_par_max up to N_par_max.
    Returns (base_parallel, step, n_par_max) as ints, e.g. 1.0x -> (232, 23, 370).
    """
    pem_w    = PEM_capacity if pem_w   is None else pem_w
    n_series = N_SERIES_PV  if n_series is None else n_series
    n_max = int(round(ratio * pem_w / (n_series * module_w)))
    step  = max(1, int(round(n_max * (1.0 - min_frac) / (n_positions - 1))))
    return n_max - step * (n_positions - 1), step, n_max


def set_string_ladder(ratio):
    """Point the module-level MOSFET ladder at a PV oversizing `ratio` (installed
    strings = ratio * PEM / string_W). Returns the previous (base, step, max) so a
    caller sweeping several ratios can restore it. Every consumer -- run_baselines,
    PVPEMEnv, the degradation model -- reads these globals, so this is the single
    switch that resizes the array."""
    global N_PAR_BASE, N_PAR_STEP, N_PAR_MAX
    prev = (N_PAR_BASE, N_PAR_STEP, N_PAR_MAX)
    N_PAR_BASE, N_PAR_STEP, N_PAR_MAX = site_string_config(ratio)
    return prev

OUT_DIR      = _os.path.join(_SCRIPT_DIR, 'Results_1MW_15min')   # matches where the saved models/results live
# Colab wipes /content when the runtime recycles, and a study left running unattended will
# outlive the browser tab. If Google Drive is mounted, write everything there instead, so the
# results survive a disconnect with no action needed at the end. Mount it BEFORE running:
#     from google.colab import drive; drive.mount('/content/drive')
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
# 'daily' : 6 kg/day settled once at end of each calendar day. Storage can buffer intra-day
# shortfalls → more flexible. 'hourly' : 0.25 kg/h must be met every hour individually.
DEMAND_MODE = 'daily'  # FINAL resolution comparison: same 78 kg/day contract as the 60-min daily study

# Demand-fulfilment target the sizing selection aims at. 100 % is deliberately NOT the
# goal: reaching it needs 7000/7000/1400 kg of tank against 6000/4000/600, i.e. +182
# kEUR/yr to buy the last ~3.5 pp at Lund and Prague, against a 5 EUR/kg product. The
# selection therefore takes the LEANEST design clearing this bar.
DEMAND_TARGET_PCT = 98.0
# Battery per site, from the MEASURED p95 overnight auxiliary load (BoP + compression +
# thermal over each contiguous G<80 block) through the 0.80 usable SoC window at 0.95
# efficiency: Lund 553, Prague 551, Odeillo 459 kWh. The sizing sweep's battery axis
# (50/500/1000 kWh) is far too coarse to resolve a 450-550 kWh answer, so the recommendation
# takes the measured value and the sweep is used only for the self-sufficiency trend.
BATT_NIGHT_KWH = {'Lund_2021': 550, 'Prague_2021': 550, 'Odeillo_2021': 450}
# The recommended tank is INTERPOLATED between swept sizes, so its demand figure is the
# target by construction, not a simulated result. verify_designs() re-simulates each
# recommendation at a round, buildable tank size and grows it until it genuinely clears the
# target. Without this Prague was reported at 95.0 % when 4292 kg actually delivers 94.0 %.
TANK_ROUND_KG      = 100
TANK_VERIFY_STEP   = 200
TANK_VERIFY_STEPS  = 8
# Minimum renewable share of the plant electrical load for a design to qualify.
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
# Let Colab/Spyder manage their own backend; force Agg only in headless scripts.
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

# Global font defaults — larger axis labels and tick numbers for publication figures.
# (Per-figure rcParams below inherit / override these; kept consistent everywhere.)
plt.rcParams.update({
    'axes.labelsize': 15, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'axes.titlesize': 14, 'legend.fontsize': 12,
})

# =============================================================================
# FIGURE CAPTIONS (on-plot titles are intentionally removed — paste these into the manuscript as
# figure captions). Kept here so the wording is preserved.

# PyTorch CPU threads. Left at None PyTorch either takes a single thread or oversubscribes the
# machine and thrashes on context switches; neither is what you want for an MLP this small. Set
# to the physical core count.
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

# DT_HOURS is defined in USER CONFIGURATION above (0.25 h for this build).

RL_CONFIG = dict(
    # Preserve the hourly study's PHYSICAL rollout horizon and minibatch fraction:
    # 2048 h / 256 h-equivalent samples -> 8192 quarter-hours / 1024 samples.
    n_steps         = 2048 * STEPS_PER_HOUR,
    batch_size      = 256  * STEPS_PER_HOUR,
    n_epochs        = 10,
    # 0.99 and 0.95 were hourly discount factors. Exponentiating by DT preserves the
    # same discount/GAE decay per physical hour at 15-min resolution.
    gamma           = 0.99 ** DT_HOURS,
    gae_lambda      = 0.95 ** DT_HOURS,
    learning_rate   = 3e-4,
    ent_coef        = 0.005,  # lowered 0.02→0.005: high entropy kept the policy random,
                              # so at eval the deterministic argmax jittered between
                              # adjacent n_par → RL switched MORE than MOO. Sharper policy.
    clip_range      = 0.2,
    max_grad_norm   = 0.5,
    checkpoint_freq = STEPS_PER_YEAR,
    net_arch        = [256, 256],
)

# =============================================================================
# %% PHYSICAL & ECONOMIC CONSTANTS
# =============================================================================
k, q, Ns = 1.38e-23, 1.6e-19, 66   # Ns=66 series cells/module (HJT 132 half-cell format)
# CHANGED 72 -> 66 on 2026-07-30 to match the module to the datasheet-fixed stack. The PEM is
# validated against NREL/TP-5B00-79055 and is not adjustable; the PV module is, so the match is
# made on the PV side.
Iph_ref, Rs, Rsh = 8.1, 0.5, 500   # Iph_ref=8.1 (1MW PV module)
Is_ref1, Is_ref2, Is_refi = 1e-14, 1e-14, 1e-13
n1, n2, ni, Eg = 1.04, 2, 1.5, 1.12
T_ref = 25 + 273.15
rho_contact, A_cell, aSi_thickness, TCO_resistivity = 5e-3, 243, 5e-9, 3e-4
pv_module_area = 1.925 # m^2, 2.1 x 66/72 (132 half-cell format). Enters the StH
                       # denominator, so it must track the cell count or efficiency is
                       # mis-stated.
N_SERIES_PV = 5        # 5 modules/string: array Vmp ~275 V against the 270 V stack, +1.8 %.
                       # History 6 -> 5. Six was chosen while the stack was mis-sized at 182
                       # cells/356 V.

tilt_angle_deg, sun_zenith_deg = 30, 45

delta_mem, A_cell_pem = 0.0054, 1250   # 1250 cm^2 cell (1MW)
a_an, a_cat, i_0_an, i_0_cat = 0.75, 0.4, 2.44e-4, 2.28e-2
T_ref_pem, i_lim, p_cat, p_an = 298.15, 6.1, 40, 1.0   # p_cat=40 bar (1MW)
i_cell = A_cell_pem * 3   # 3750 A = the datasheet NOMINAL operating current (3 A/cm^2).
                          # NREL/TP-5B00-79055 (Giner ELX Allagash, CRADA CRD-18-00742) lists
                          # "Nominal Operating Current: 3,750 A (current density 3 A/cm^2)" for
                          # the 1250 cm^2 cell.

# Series cells in the PEM stack, sized directly to the datasheet rather than to the PV.
N_SER_PEM = 134

# --- Arrhenius temperature dependence of the exchange current densities -------
# i0a/i0c are quoted at T_ref_pem = 298.15 K.
PEM_I0_ARRHENIUS = True
E_ACT_AN, E_ACT_CAT = 29.5e3, 9.9e3   # J/mol — measured, see note above

LHV = 33.3 * 1000                 # Wh/kg — lower heating value (the reporting basis)
HHV = 39.4 * 1000                 # Wh/kg — higher heating value

# --- H2 PRODUCTION: FARADAY'S LAW, NOT AN EFFICIENCY CURVE -----------------------
# Corrected 2026-08-06. The `efficiency_points` table further down is a datasheet SYSTEM
# efficiency quoted on the HHV basis. Checked against this file's own V_cell at 60 C, the
# curve reads 0.82/0.80/0.79/0.76 at 0.3/0.4/0.5/0.6 load against 1.481/V_cell (HHV) =
# 0.815/0.797/0.783/0.771, while 1.254/V_cell (LHV) gives 0.690/0.675/0.663/0.653 — it
# tracks the HHV column to within 0.005 and is nowhere near the LHV one. It was previously
# used as `H2 = P*eff/LHV`, an HHV-basis efficiency divided by an LHV energy content, which
# yielded 11-19 % more hydrogen than Faraday's law permits at the same current, i.e. an
# apparent Faradaic efficiency of 111-119 %. Production is now taken from the operating
# CURRENT — which is what physically sets it in a direct-coupled stack — and the curve is
# kept only where it belongs, in the thermal balance.
F_FARADAY   = 96485.0     # C/mol
M_H2_KG_MOL = 2.016e-3    # kg/mol
ETA_FARADAY = 0.98        # Faradaic efficiency (H2 crossover + shunt currents)
V_TH_HHV    = 1.481       # V/cell — thermoneutral (HHV) cell voltage. Waste heat per cell is
                          # I*(V_cell - V_TH_HHV); at V_cell < V_TH the reaction is endothermic,
                          # which no PEM stack reaches in practice, so Q_gen is floored at 0.


def h2_rate_faraday(I_stack_A, P_pem_W):
    """Hydrogen production rate [kg/h] from the stack operating current.

    Single definition used by BOTH the EMS mass balance and the MOO candidate scoring, so
    the controller cannot rank string counts on one production model and then be settled on
    another. P_pem_W only gates the cut-in (the stack is idle below ~20 W of coupled power).
    """
    if P_pem_W <= 20 or I_stack_A <= 0:
        return 0.0
    rate = N_SER_PEM * float(I_stack_A) * M_H2_KG_MOL / (2.0 * F_FARADAY) * ETA_FARADAY * 3600.0
    return float(np.clip(rate, 0.0, H2_PEAK_KGH))

storage_capacity = 4320           # kg — right-sized to the demand-fulfillment knee (~60 days
                                 # autonomy). Tank sensitivity shows 100% fulfillment is
                                 # reached here; larger tanks only raise LCSH. INITIAL is half.
# 78 kg/day (3.25 kg/h). LOWERED FROM 90 on 2026-08-06 alongside the Faraday correction
# (see the F_FARADAY block): production now comes from the operating current instead of an
# HHV-basis efficiency curve divided by LHV, which removes an 11-19 % over-production. The
# demand is an absolute number and did not shrink with it, so at 90 kg/day the existing
# sizing would have fallen from ~98 % fulfilment to roughly 85 %.
#
# 78 = 90 x 0.87, where 0.87 is the ESTIMATED annual mean of the removed bias (per-load it
# runs +4.6 % at 0.1 load, plateaus near +19 % over 0.3-0.5, falls to +11 % at nominal; the
# annual figure is a load-duration-weighted average of that, plausible band 0.84-0.90).
# It is an estimate, NOT a measurement -- confirm it against the sizing sweep and move it if
# the 98 % knee no longer sits inside a sensible oversizing band.
#
# Note the PEM capacity factor is largely unaffected either way: calculate_lcoh divides by
# H2_PEAK_KGH * 8760, and that ceiling fell 21 -> 19 with the same physics, so CF moves ~4 %.
# What the correction actually threatened was demand fulfilment, which is why the DEMAND is
# the quantity being adjusted here rather than the plant.
#
# History: 72 was too easy (good sites met it at any PV size); 108 suited the PVGIS data but
# was out of reach under the lower CAMS irradiance; 90 was the compromise. The CAMS sizing
# sweep showed Lund plateauing at ~87 % demand even at 1.4x PV (100 % needs ~6x, i.e. the
# northern winter resource is the binding constraint, not the controller). Those sweep
# figures predate the polarization-curve correction, the widened PV ladder AND this Faraday
# fix -- re-read every one of them from the new sizing run before quoting.
H2_DEMAND_KG_PER_H = 3.25
hourly_H2_target  = H2_DEMAND_KG_PER_H             # kg per HOURLY settlement
step_H2_target    = H2_DEMAND_KG_PER_H * DT_HOURS  # 0.8125 kg per 15-min direct path
daily_H2_target   = H2_DEMAND_KG_PER_H * 24.0      # 78 kg per DAILY settlement
PEM_capacity     = 1000 * 1000   # 1 MW PEM
H2_PEAK_KGH      = 19.0     # peak H2 rate (kg/h). Faraday ceiling at the datasheet nominal
                            # current: 134 cells x 3750 A x M_H2/(2F) x 0.98 = 18.5 kg/h, so
                            # 19.0 is a guard, not a binding cap. Was 21.0, which was ABOVE the
                            # Faraday limit and therefore never bound at all.
MPPT_NORM        = 32000.0  # normalisation (W) for the MPP-shortfall reward term.
                            # 26667 -> 89000 -> 32000.


def mppt_shortfall(P_mpp, P_op, cap=None):
    """Control-attributable MPP shortfall (W).

    The power an ideal tracker could have delivered TO THE STACK, minus what the
    direct-coupled operating point actually delivers.

    The `min(P_mpp, cap)` is the correction. When the array's maximum-power point
    exceeds what the electrolyzer can absorb, the surplus is routed to the battery
    or curtailed -- it is a sizing outcome, not a tracking failure, and no
    controller can recover it. Charging it to the controller (as `P_mpp - P_op`
    did) made this metric report PEM saturation rather than coupling quality:
    89-100 % of the reported annual total was saturation, single steps reached
    622 kW, and the DRL/MOO ranking inverted between the daily and hourly demand
    settlements even though the electrical match barely changes between them.

    With the cap applied the term isolates what the string-count decision actually
    controls, and it exposes the real trade-off: holding a configuration to avoid
    switching costs mismatch, which is what the switching-vs-mismatch figure is
    meant to show. Energy above the cap is already accounted separately as
    PV_Excess_W / P_curtailed_W.
    """
    cap = PEM_capacity if cap is None else cap
    return max(0.0, min(float(P_mpp), float(cap)) - float(P_op))

# --- Active MOSFET string ladder (set per site by the location loop) -------------
# n_par = N_PAR_BASE + mosf * N_PAR_STEP,  mosf = 0..6,  top position = N_PAR_MAX.
# Defaults are the 1.08x baseline array (400 strings, 250-400 in steps of 25); the
# location loop overwrites them from SITE_SIZING via site_string_config().
N_PAR_BASE, N_PAR_STEP, N_PAR_MAX = 250, 25, 400

# --- RL reward: MOO objective weights (identical to run_baselines) + fatigue term ---
# The RL reward reuses the SAME multi-objective as MOO/MOO+RF: moo_obj = W_H2*(H2/1.6) +
# W_STH*(StH/0.185) - W_MPPT*(mppt_loss/2000) so a converged PPO policy reproduces MOO's H2 /
# StH / MPPT performance.
W_H2, W_STH, W_MPPT = 0.36, 0.31, 0.33
W_UNDERUSE          = 0.10    # RL-only penalty for starving the PEM: a flat charge on idle
                             # strings whenever the PEM is not yet saturated (fl < 0.98), scaled
                             # by the unused fraction of the installed array. REDUCED 0.30 ->
                             # 0.10 on 2026-07-29 with the MPPT_NORM fix above.
LAMBDA_SWITCH       = 0.13    # FINAL switching-fatigue weight.
                             # Selected from the 60-year, 3-seed lambda bracket
                             # {0.12, 0.13, 0.14, 0.15}. 0.13 is the lowest tested value
                             # with 3/3 seeds in the three-replacement class, while giving
                             # the highest mean H2 and lowest mean LCOH/LCSH among the
                             # robust 0.13-0.15 candidates.

# Resolve the MOO+Pen charge now that LAMBDA_SWITCH exists. Equal by default; set
# MOO_PEN_LAMBDA to a float above to decouple them (see the note at its definition).
if MOO_PEN_LAMBDA is None:
    MOO_PEN_LAMBDA = LAMBDA_SWITCH

# --- THERMAL SETPOINT ACTION (novel extension) -----------------------------------
# When THERMAL_ACTION is on, the agent's action is 2-D: (string config, warm-standby setpoint).
THERMAL_ACTION     = True

# --- FORECAST LOOKAHEAD (PROVISIONAL — raised 2 -> 4 on 2026-08-03, not yet swept) ---
# Found comparing MOO vs DRL restart temperatures at Lund: the agent pre-warms (raises its
# keep-warm floor above the cheapest 30 C rung) in the hour before only 7.6% of its restarts,
# and lands cold (<40 C) in 32.4% of those cheap-floor cases -- versus 0% cold whenever it DOES
# pre-warm. The expected-value math already favours pre-warming (~0.41 reward gained from
# avoiding a cold-start charge against ~0.14 spent heating for 4h), so the agent is not under-
# incentivised, it cannot SEE far enough ahead to act on the incentive: at lookahead=2 the
# observation carries irradiance/temperature only 2h out, short of the ~4h the keep-warm
# heater needs to move the floor meaningfully (cf. the idle cooldown time constant
# tau = mCp/UA ~= 4.2h). Raised to 4h to match that physical timescale. Retrain and re-check
# the pre-restart floor-choice table before trusting this value; it is reasoned, not measured.
LOOKAHEAD_HOURS       = 4.0
# Time-resolution sensitivity must preserve the information structure of the
# hourly controller: four forecast points at +1,+2,+3,+4 physical hours.
# At DT=0.25 h these are rows +4,+8,+12,+16, respectively.
FORECAST_HORIZONS_H   = (1.0, 2.0, 3.0, 4.0)
FORECAST_OFFSETS_STEPS = tuple(int(round(h / DT_HOURS)) for h in FORECAST_HORIZONS_H)
LOOKAHEAD_STEPS       = len(FORECAST_OFFSETS_STEPS)  # 4 forecast points; obs stays 26-D
if FORECAST_OFFSETS_STEPS != (4, 8, 12, 16):
    raise RuntimeError("15-min forecast offsets are inconsistent with +1,+2,+3,+4 h.")

# --- CHILLER SET-POINT ACTION ----------------------------------------------------
# The second half of the thermal actuator. The keep-warm heater sets the floor the idle stack is
# held at; the chiller sets the ceiling the loaded stack is cooled to.
CHILLER_ACTION     = True
CHILLER_LEVELS_C   = (60.0, 65.0, 70.0)   # selectable cooling set-points; 70 C = datasheet
                                           # operating temperature = the baselines' fixed value.

# --- THERMAL (COLD-OPERATION) DEGRADATION ----------------------------------------
# See calculate_degradation_step(). Degradation is charged in proportion to the cell voltage the
# stack actually sits at, which the polarization model already raises when the stack is cold.
COLDSTART_REWARD    = True
# RAISED 0.40 -> 0.80 on 2026-08-03, together with W_TCYCLE 0.08 -> 0.04. See the block at
# W_TCYCLE for the full derivation; in short, these two terms were mechanically opposed and
# the pre-warming action they are meant to encourage was left with a net incentive of
# +0.046 reward per restart, roughly a quarter of its gross benefit.
W_COLDSTART         = 0.80

THERMAL_DEGRADATION = True
_V_DEG_REF          = None   # memoised V_cell at T_DEG_REF_K; see calculate_degradation_step
T_DEG_REF_K         = 343.15   # 70 C, the datasheet operating point = reference (f_therm = 1)
KAPPA_THERM         = 1.0      # exponent on the voltage ratio; 1.0 = strictly proportional.
                               # Calibrate by sweep; do not report a tuned value as measured.

# --- COLD-START EXPERIMENT -------------------------------------------------------
# A production start counts as COLD when the stack was still below this temperature at the
# moment it resumed producing (see count_cold_starts()).
COLDSTART_THRESHOLD_C = 50.0
# MOVED 45 -> 50 on 2026-07-30.

# --- COLD-START SoH CHARGE (recalibrated 2026-08-03) --------------------------------
# Per-event SoH decrement charged in calculate_degradation_step() when the stack resumes
# production. RESCALED 5e-4/5e-5 -> 5e-5/5e-6 at the same time as the cold-start
# classification was fixed to use the PRE-restart temperature.
#
# Why: with the old (post-restart) classification only ~16 % of restarts were charged the
# cold rate, so 5e-4 delivered ~4.9 pp/yr. Fixing the classification makes ~95 % of the
# baselines' restarts cold, which at 5e-4 would have given ~19 pp/yr from this term alone
# and ~21 pp/yr in total -- a stack crossing the model's historical 80 % health reference
# inside ONE year. NOTE: LCOH no longer assumes a fixed 10-year stack interval; the economic
# EOL is fixed consistently at SoH=0.80 through LCOH_PEM_EOL_SOH; replacement timing is projected from each controller's SoH loss.
#
# Calibration target: the base terms (f_load + f_rccd) already realise ~1.8 pp/yr. Setting
# the cold charge to 5e-5 puts a daily-cycling baseline at ~380 x 5e-5 = 1.9 pp/yr from
# restarts, so total ~3.7 pp/yr -- i.e. daily start/stop cycling roughly DOUBLES the
# steady-operation rate, which is the effect direction reported for intermittent PEM duty
# [Ursua2012, Rakousky2016]. That historical calibration referenced an 80 % health floor;
# the economic replacement model below uses the same historical 80 % SoH EOL
# convention, with replacement timing projected independently for each controller. A controller that
# pre-warms every restart instead pays 380 x 5e-6 = 0.19 pp/yr, so the mechanism still
# separates controllers by ~1.7 pp without the un-physical crater.
#
# The cold/warm SoH charges also define the SHAPE of the cold-start reward event.
# A classified cold restart receives a fixed penalty proportional to the incremental
# cold-start damage relative to a warm restart. This aligns the reward with the same
# discrete 50 C classifier used by the degradation and reporting models and removes the
# near-threshold loophole of the former depth-only penalty.
COLDSTART_SOH_COLD = 5e-5   # restart below COLDSTART_THRESHOLD_C
COLDSTART_SOH_WARM = 5e-6   # restart at or above COLDSTART_THRESHOLD_C
COLDSTART_EXCESS_DAMAGE_FRAC = max(
    0.0,
    (COLDSTART_SOH_COLD - COLDSTART_SOH_WARM) / max(COLDSTART_SOH_COLD, 1e-12)
)  # = 0.90 with the calibrated SoH charges
KEEPWARM_LEVELS_C  = (30.0, 45.0, 60.0)   # selectable warm-standby set-points.
                                           # FLOOR HISTORY: 10 -> 30 -> 40 -> 30 C. 10 C let the
                                           # agent park the stack near ambient and reheat daily,
                                           # an un-physical deep thermal cycle.
KEEPWARM_BASELINE_C = KEEPWARM_LEVELS_C[0]
W_KEEPWARM         = 0.10        # reward penalty weight on idle keep-warm heating energy
KEEPWARM_NORM_W    = 44000.0     # normalisation (W) for the thermal-management energy terms.
                                 # 50000 -> 90000 -> 44000. Re-measured after the sizing change
                                 # by stepping the ENV at each keep-warm rung for a full year at
                                 # all three sites (MOO cannot be used: it holds the 30 C floor,
                                 # so it only sees the cheapest rung and reports ~26 kW).
                                 # Idle load p99 by rung, sites pooled: 30 C 26.3 kW, 45 C 36.3 kW,
                                 # 60 C 46.3 kW; pooled p99 = 44.1 kW. At 90000 the term ran at
                                 # 0.49 of its intended scale, i.e. W_KEEPWARM = 0.10 behaved
                                 # like 0.05. The two normalisers that carry the headline result
                                 # were checked at the same time and did NOT move: MPPT_NORM
                                 # p99/const = 1.01, TCYCLE_NORM_C = 0.87.
                                 # NOTE for the manuscript: w_keepwarm was never one of the
                                 # weights swept in the reward-robustness study (that swept
                                 # lambda_sw, w_u, w_cs, w_tc, w_ch), so the reported spans are
                                 # unaffected by this change.
W_CHILLER          = 0.10
# --- Thermal-cycling penalty (extends the smooth-operation objective to temperature) ---
# Charges per-step PEM stack-temperature swings |dT_pem|. W_TCYCLE remains fixed at 0.04
# from the completed reward screen. The separate cold-start term is now a threshold-aligned
# event charge, so W_TCYCLE prices continuous thermal motion while W_COLDSTART prices the
# discrete restart class used by the degradation model.
W_TCYCLE           = 0.04
TCYCLE_NORM_C      = 34.0        # per-step stack-temperature swing that normalises to 1.0.
                                 # RECALIBRATED 20 -> 34 on 2026-07-30 by reward_audit.py: the
                                 # observed |dT_pem| reaches p99 = 34 C and max = 40 C per step,
                                 # so a 20 C reference put the term above 1 and inflated W_TCYCLE
                                 # by ~1.7x relative to its written value.

# --- Steady cold-operation penalty (PROVISIONAL — added 2026-08-03, not yet swept) ---
# Gap found comparing MOO vs DRL at Lund: DRL's mean T_pem was 40.9 C against MOO's fixed-
# ceiling 44.6 C (DRL sits below the datasheet 70 C ceiling 55% of active hours, favouring the
# ~60 C electrical sweet spot near the array MPP voltage). None of the existing thermal terms
# price this: W_TCYCLE only charges the per-step SWING, and W_COLDSTART only fires once at the
# idle->producing transition. A stack that settles at a lower ceiling and STAYS there pays
# neither, yet f_therm in calculate_degradation_step() charges every step spent below
# T_DEG_REF_K regardless. That gap let the agent capture the electrical-matching benefit of a
# cooler set-point for free while the SAME physical mechanism quietly cost more PEM SoH than
# MOO's fixed-70C baseline (measured: 90.92% vs 93.29% final SoH at Lund, reproduced at
# Prague). This term prices the identical physical quantity f_therm already uses (depth below
# the 70 C reference), so the trade the agent is actually making becomes visible in the reward
# instead of only in the post-hoc SoH metric. The weight below is a reasoned default matching
# W_TCYCLE's scale, NOT a measured normaliser -- retrain and re-audit before trusting it, the
# same way every other weight in this file was (see KEEPWARM_NORM_W, TCYCLE_NORM_C above).
W_WARMOPS          = 0.08

BATT_CAPACITY_WH = 500 * 1000    # 500 kWh — right-sized as a sub-hourly SMOOTHING buffer
                                 # (was 2133 kWh, which barely cycled). At 500 kWh the pack
                                 # actually charge/discharges over the day, exposing its role
                                 # in absorbing transients; sweep with RUN_BATTERY_SENSITIVITY.
BATT_MAX_P_W     = 500 * 1000    # 500 kW = exactly 1.00 C at 500 kWh — the pack's continuous
                                 # power rating, a standard figure for utility LFP. HISTORY: was
                                 # 533 kW, justified as "sized to absorb the unused- string MPPT
                                 # power without clipping". That justification was written for
                                 # the old 1.08x array, whose diverted branch peaked at 405 kW.
BATT_EFF         = 0.95
# --- Battery ageing (LFP) --------------------------------------------------------
# Cycle life is quoted as "N full equivalent cycles to X % of initial capacity", so the per-
# cycle SoH loss is BATT_EOL_LOSS / BATT_CYCLE_LIFE.
BATT_CYCLE_LIFE       = 3000    # full equivalent cycles to end of life
BATT_EOL_LOSS         = 0.20    # SoH lost at end of cycle life (100 % -> 80 %)
BATT_CALENDAR_LOSS_PER_YR = 0.02   # calendar fade, independent of cycling. 2 %/yr is a
                                   # representative LFP figure at moderate temperature and
                                   # mid SoC; it dominates for a lightly-cycled pack and is
                                   # what stops an unused battery from ageing at zero.
BATT_SOC_MIN     = 0.10         # 10 % lower cut-off
BATT_SOC_MAX     = 0.90         # 90 % upper cut-off

H2_PRICE_EUR_KG  = 5.0

# --- SECONDARY H2 OFF-TAKE -------------------------------------------------------
# The primary firm contract (hourly_H2_target, the SAF hydrotreater) must always be met and
# still drives the "demand-met %" metric and the seasonal-storage story.
H2_SECONDARY_OFFTAKE      = True
H2_SECONDARY_PRICE_EUR_KG = 3.0     # < H2_PRICE_EUR_KG (firm SAF price); merchant wholesale

# Grid export policy. The system is designed as direct-coupled PV-PEM with the grid used only as
# an auxiliary backup — NOT as an electricity-selling business.
ALLOW_GRID_EXPORT = False
# If export is ever enabled, it is gated by TANK LEVEL as well as demand: the seasonal
# H2 buffer is filled first, and only the genuine surplus above this fraction is exported.
EXPORT_TANK_MIN_FRAC = 0.80

# =============================================================================
# TECHNO-ECONOMIC REFERENCES (for the manuscript bibliography) [R1] IEA (2019), "The Future of
# Hydrogen." — PEM CAPEX, H2 cost benchmarks. [R2] IRENA (2020), "Green Hydrogen Cost
# Reduction." — electrolyzer/PV CAPEX, WACC, LCOH.
LCOH_PV_EUR_KWP       = 700.0    # EUR/kWp installed (bifacial, utility-scale, no inverter) [R2]
LCOH_PEM_EUR_KW       = 1200.0   # EUR/kW_PEM (labelled stack + BoP in the original cost basis) [R1,R2]
                                  # NOTE: capex_bop is also added separately below. For REPEX,
                                  # LCOH_PEM_REPLACEMENT_FRAC must reflect the truly replaceable
                                  # stack share; the default 1.0 is intentionally conservative.
LCOH_BATT_EUR_KWH     = 200.0    # EUR/kWh (LFP, installed)
LCOH_TANK_EUR_KG      = 500.0    # EUR/kg H2 — 200 bar steel storage bank; conservative vs €1393/kg @350 bar [R5], within pressurized-vessel range [R3]; range ~350-600, tornado brackets it
LCOH_COMP_EUR_KGH     = 6000.0   # EUR/(kg/h) — small multi-stage 40→200 bar H2 compressor (~21 kg/h ≈ 21 kW) [R4,R5]; range ~4000-10000, tornado brackets it
LCOH_BOP_FRAC         = 0.20     # BoP as fraction of PEM CAPEX (piping, sensors, civil)
LCOH_CTRL_EUR         = 8000.0   # Fixed cost of smart controller hardware (EUR)
# Power electronics — bidirectional converters
LCOH_BATT_CONV_EUR_KW = 75.0    # EUR/kW  — battery bidirectional DC/DC converter (MPPT branch)
LCOH_BATT_CONV_LIFE   = 15      # years   — power electronics replace at 15 yr
LCOH_BATT_CONV_OM     = 0.015   # 1.5 %/yr
LCOH_GRID_INV_EUR_KW  = 120.0   # EUR/kW  — bidirectional AC/DC grid inverter (import + export)
LCOH_GRID_INV_KW      = 400.0    # kW      — sized for max simultaneous grid power flow
LCOH_GRID_INV_LIFE    = 15      # years
LCOH_GRID_INV_OM      = 0.020   # 2.0 %/yr
# Lifetimes (years)
LCOH_PV_LIFE          = 30
LCOH_BOP_LIFE         = 30       # years; civil/piping BoP life is separate from stack life.
                                  # PEM STACK life is NOT fixed: it is controller-specific and
                                  # projected from simulated SoH to LCOH_PEM_EOL_SOH below.
LCOH_BATT_LIFE        = 10       # 3000 cycles at ~1 cycle/day ≈ 8.2 yr → round to 10
LCOH_TANK_LIFE        = 25
LCOH_COMP_LIFE        = 15
# Annual O&M as fraction of CAPEX
LCOH_PV_OM            = 0.010    # 1.0 %/yr
LCOH_PEM_OM           = 0.030    # 3.0 %/yr (membrane/catalyst replacement, water)
LCOH_BATT_OM          = 0.010    # 1.0 %/yr
LCOH_TANK_OM          = 0.005    # 0.5 %/yr
LCOH_COMP_OM          = 0.020    # 2.0 %/yr
# Oxygen by-product revenue
# PEM produces 8 kg O2 per kg H2 (stoichiometric: 2H2O → 2H2 + O2, M_O2/M_H2 = 16)
O2_KG_PER_KG_H2      = 8.0     # kg O2 per kg H2
O2_PRICE_EUR_KG       = 0.10    # EUR/kg — industrial grade (99.5 %+), conservative
# Finance
LCOH_WACC             = 0.06     # 6 % weighted average cost of capital
LCOH_PROJECT_LIFE     = 30       # project horizon for LCOH calculation (years)
# Installed tank for LCOH is sized for a practical buffer, NOT the 1000 kg
# simulation horizon buffer.  For 6 kg/day contract: 5-day buffer = 30 kg.
LCOH_TANK_KG_INSTALLED = 4320.0   # kg — matches the right-sized physical tank (60-day buffer)


# --- PEM stack replacement economics: degradation-driven REPEX -----------------
# The operational simulation starts every evaluation/controller at a NEW stack (SoH = 1.0)
# and records the one-period SoH loss. LCOH projects that measured degradation rate forward
# to an economic end-of-life threshold, schedules DISCRETE stack replacements over the
# 30-year project, discounts each replacement to year 0, and annualises the resulting NPV.
#
# SINGLE PEM END-OF-LIFE DEFINITION USED THROUGHOUT THE ECONOMIC MODEL:
#   new stack SoH = 1.00; replace at SoH = 0.80.
# The stack lifetime is therefore inferred from each controller's measured annual SoH loss:
#   life_yr = (1.00 - 0.80) / annual_SoH_loss.
# Do not re-introduce a fixed PEM stack lifetime: it would erase the controller-specific durability
# benefit that the SoH model is intended to measure.
LCOH_PEM_NEW_SOH          = 1.00
LCOH_PEM_EOL_SOH          = 0.80
# Fraction of the initial PEM CAPEX paid at each STACK replacement. 1.0 is conservative and
# assumes LCOH_PEM_EUR_KW is the replaceable stack cost. If that EUR/kW value is a complete
# installed electrolyser cost, replace this with the documented stack-only fraction.
LCOH_PEM_REPLACEMENT_FRAC = 1.00


def _lcoh_crf(r, n):
    """Capital-recovery factor used by the degradation-driven replacement model."""
    return r * (1 + r)**n / ((1 + r)**n - 1)


def project_pem_stack_replacements(df, stack_capex, wacc=None, project_life=None,
                                   soh_col='SoH_PEM', soh_new=None, soh_eol=None,
                                   replacement_frac=None):
    """Project controller-specific PEM stack life and discounted replacement cost.

    The evaluation trajectories in this script start at SoH=1.0 for both MOO and RL.
    The simulated SoH loss is annualised using the ACTUAL trajectory duration
    ``len(df) * DT_HOURS / 8760``; therefore the same function is valid for hourly and
    15-minute evaluations and for partial-year datasets.

    Stack life [yr] = (SoH_new - SoH_EOL) / annual_SoH_loss

    Replacements occur at life, 2*life, ... strictly before project end. A replacement
    exactly at year 30 is not purchased because the project ends at that instant.
    Future replacement expenditures are discounted to year 0 and then converted to an
    equivalent annual cost over the project horizon.
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

    # Both run_baselines() and PVPEMEnv(eval_mode=True) explicitly initialise SoH_PEM=1.0.
    # Use that true pre-step state rather than df.iloc[0], which is already AFTER one step of wear.
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
        # At the model's present SoH floor this loop is small (<~120 replacements / 30 yr).
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
# Opening tank inventory, and how the year is closed.
#   'fraction' — start at TANK_INIT_FRAC x tank. The plant must FINISH at or above that
#                level or the shortfall is charged as unmet demand, so the opening stock
#                is a LOAN to be repaid, not a gift, and the year stays repeatable. This
#                is the design rule: carry a 50 % strategic reserve into every winter.
#   'cyclic'   — solve the level the year reproduces (solve_cyclic_storage).
#   'empty'    — start at zero.
TANK_INIT_MODE     = 'fraction'
TANK_INIT_FRAC     = 0.5
TANK_CYCLIC_ITERS  = 3      # 'cyclic' mode: max fixed-point passes
TANK_CYCLIC_TOL_KG = 5.0
INITIAL_H2_STORAGE = TANK_INIT_FRAC * storage_capacity
# Solved on the sizing above (2 passes each; f is constant once the tank fills, so the
# fixed point is exact). Clear these whenever PV/tank sizing or the demand contract moves.
SITE_CYCLIC_S0     = {}     # 'cyclic' mode only: cache of solved per-site levels


def set_initial_storage():
    """Recompute the opening inventory from the CURRENT tank size.

    Must be called after any change to storage_capacity, or the sweep silently reuses the
    previous site's opening stock and every tank size is compared on a different basis.
    """
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
    """Per-step health decrement for the PV array and the PEM stack.

    T_pem_K       stack temperature at the END of this step (drives f_therm).
    T_pem_prev_K  stack temperature at the END of the PREVIOUS step, i.e. the
                  temperature the stack was actually at when current began to
                  flow. This is what classifies a restart as cold or warm.
                  Falls back to T_pem_K when omitted (legacy behaviour).

    WHY THE TWO ARE SEPARATE (bug fixed 2026-08-03). Both callers overwrite their
    T_pem with the post-step value before calling this function, so the cold-start
    branch was classifying restarts by the temperature AFTER the restart hour's own
    production heating. At 1 MW and m_Cp = 1e7 J/K a full-load hour raises the stack
    by >100 K, so that post-step temperature is set by how sunny the restart hour
    was, not by whether the stack had been kept warm. Measured on the four seeded
    daily runs: corr(T_post, irradiance_at_restart) = +0.78..+0.85, whereas
    corr(T_pre, irradiance) = +0.05..+0.17. The metric was therefore scoring
    sunshine, not thermal management, and the keep-warm action the agent controls
    had almost no influence on the degradation it was scored by -- a weak, noisy
    action->outcome gradient that shows up as large seed-to-seed swings in PEM SoH.
    It also contradicted this function's own comment ("380 restarts/yr * 0.0005 =
    19 % SoH loss/yr for the MOO baseline"): with the post-step temperature only
    64/397 restarts at Lund were charged as cold, giving 4.9 pp, not 19 pp.
    The reward, reporting metric and degradation classifier all use the pre-step
    temperature and the same COLDSTART_THRESHOLD_C, so the three conventions are aligned.
    """
    pv_deg_step  = (0.005 / 8760) * DT_HOURS
    pem_base_deg = (0.03  / 8760) * DT_HOURS

    f_load   = 0.31 + (0.69 * fl)
    delta_fl = abs(fl - prev_fl)
    f_rccd   = 1.83 * delta_fl / DT_HOURS

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

    dynamic_multiplier = (f_load + f_rccd) * f_therm
    new_SoH_PEM = max(0.6, SoH_PEM - (pem_base_deg * dynamic_multiplier))

    if diag is not None:
        diag['load'] = diag.get('load', 0.0) + pem_base_deg * f_load * f_therm
        diag['rccd'] = diag.get('rccd', 0.0) + pem_base_deg * f_rccd * f_therm

    # --- AGGRESSIVE COLD-START PENALTY (The "MOO-crater" fix) ---
    # Trigger if transitioning from idle (prev_fl < 0.05) to operational (fl >= 0.05).
    # Classified by the temperature the stack was at when current began to flow (the
    # END of the previous, idle step) -- NOT by the post-production temperature. See
    # the docstring for the measurement that motivated this.
    T_cs_K = T_pem_prev_K if T_pem_prev_K is not None else T_pem_K
    if fl >= 0.05 and prev_fl < 0.05:
        if T_cs_K is not None and T_cs_K < (COLDSTART_THRESHOLD_C + 273.15):
            new_SoH_PEM -= COLDSTART_SOH_COLD
            if diag is not None:
                diag['cold'] = diag.get('cold', 0.0) + COLDSTART_SOH_COLD
        else:
            # Warm restart: mechanical stress only, an order of magnitude cheaper.
            new_SoH_PEM -= COLDSTART_SOH_WARM
            if diag is not None:
                diag['cold'] = diag.get('cold', 0.0) + COLDSTART_SOH_WARM

    new_SoH_PV  = max(0.6, SoH_PV  - pv_deg_step - switch_stress)
    return new_SoH_PV, new_SoH_PEM
'''
def calculate_degradation_step(SoH_PV, SoH_PEM, fl, prev_fl, prev_mosf, current_mosf,
                              T_pem_K=None):
    """Per-step health decrement for the PV array and the PEM stack.

    T_pem_K (optional) enables the THERMAL degradation multiplier described below. When
    omitted the function reduces exactly to its previous behaviour, so any caller that has
    not been updated is unaffected rather than silently mis-scaled.
    """
    pv_deg_step  = (0.005 / 8760) * DT_HOURS
    pem_base_deg = (0.03  / 8760) * DT_HOURS

    f_load   = 0.31 + (0.69 * fl)
    delta_fl = abs(fl - prev_fl)
    # delta_fl is a per-step change, but it is multiplied below by pem_base_deg, which
    # already carries a DT_HOURS factor. Dividing by DT_HOURS here leaves the rate-of-
    # change damage proportional to |df_l| alone, so it no longer shrinks purely because
    # the sampling interval got shorter. (PEM SoH rose from 98.12 to 98.36 % at 15-min
    # on identical weather before this fix, which is unphysical.) No-op at DT_HOURS = 1.
    f_rccd   = 1.83 * delta_fl / DT_HOURS

    if prev_mosf == 11 and current_mosf != 11:
        switch_stress = 5e-6
    else:
        switch_stress = abs(current_mosf - prev_mosf) * 1e-7

    # --- Thermal (cold-operation) multiplier -----------------------------------------
    # Anode-catalyst dissolution and membrane oxidative attack in a PEM electrolyser are
    # driven by CELL POTENTIAL, not by current alone. The polarization model already in
    # this file reproduces the relevant fact: at a fixed current density the cell sits at a
    # HIGHER voltage when the stack is cold, because the activation term carries the
    # Arrhenius temperature dependence of i0a/i0c (see PEM_I0_ARRHENIUS above). At the
    # nominal 3 A/cm^2 the model gives ~2.11 V/cell at 30 C against ~1.99 V/cell at 70 C.
    #
    # Charging degradation in proportion to that voltage ratio is therefore a CONSEQUENCE
    # of physics already present in the model, not a new mechanism bolted on: running cold
    # is intrinsically more damaging per unit hydrogen, and a cold restart is simply the
    # worst case of it. This replaces the previous behaviour, in which stack temperature
    # did not enter the degradation model at all and a cold start was completely free.
    #
    # The exponent is deliberately left at 1.0 (strictly proportional). Potential-driven
    # Ir dissolution is known to be super-linear in overpotential, so a value > 1 is
    # physically arguable, but this file has no measured coefficient to justify a specific
    # one. 1.0 is the minimally-assumptive choice; treat KAPPA_THERM as a quantity to be
    # established by the reward/degradation sensitivity sweep and, ultimately, cited --
    # do NOT report a tuned value as if it were measured.
    f_therm = 1.0
    if T_pem_K is not None and THERMAL_DEGRADATION:
        # v_ref depends only on module constants, so it is computed once and memoised.
        # Evaluating it per step tripled the V_cell call count (1 -> 3 per step) and made
        # this function one of the most expensive in the simulation for no reason.
        global _V_DEG_REF
        if _V_DEG_REF is None:
            _V_DEG_REF = float(V_cell(T_DEG_REF_K, p_cat, p_an, delta_mem, A_cell_pem,
                                      a_an, a_cat, i_cell, i_0_an, i_0_cat,
                                      T_ref_pem, i_lim))
        if _V_DEG_REF > 0:
            v_now = V_cell(float(T_pem_K), p_cat, p_an, delta_mem, A_cell_pem, a_an, a_cat,
                           i_cell, i_0_an, i_0_cat, T_ref_pem, i_lim)
            f_therm = float(max(1.0, (v_now / _V_DEG_REF) ** KAPPA_THERM))

    dynamic_multiplier = (f_load + f_rccd) * f_therm
    new_SoH_PEM = max(0.6, SoH_PEM - (pem_base_deg * dynamic_multiplier))
    new_SoH_PV  = max(0.6, SoH_PV  - pv_deg_step - switch_stress)
    return new_SoH_PV, new_SoH_PEM
'''

# --- H2 storage & compression design -----------------------------------------
# H2 leaves the PEM cathode at p_cat (40 bar).
P_STORE_MAX_BAR = 200.0     # buffer design pressure (200 bar steel storage bank) [R5]
P_DELIVER_BAR   = 180.0     # HDO/SAF reactor delivery pressure — ALL H2 must reach at least
                            # this before use (no 40 bar direct path). Bio-oil HDO ~100-200 bar [R6]

def h2_compression_kwh_per_kg(P1_bar, P2_bar, T_in=313.0, eta=0.70, k=1.41, Z=1.10):
    """Multi-stage, intercooled H2 compression specific energy (kWh per kg H2). [R4,R5]
    Stages sized for a per-stage pressure ratio ≤ ~2.3; eta lumps isentropic +
    mechanical + motor efficiency. From 40→200 bar this gives ≈1.0 kWh/kg."""
    if P2_bar <= P1_bar:
        return 0.0
    ratio = P2_bar / P1_bar
    N = max(1, int(np.ceil(np.log(ratio) / np.log(2.3))))
    r = ratio ** (1.0 / N)
    R, M = 8.314, 2.016e-3
    w = N * (Z * R * T_in / M) * (k / (k - 1.0)) * (r ** ((k - 1.0) / k) - 1.0) / eta  # J/kg
    return w / 3.6e6


def calculate_thermal_inertia(prev_T_pem, T_amb_C, P_pem_actual, prod_eff,
                              T_min_C=None, T_max_C=70.0):
    """Lumped-capacitance PEM stack thermal model with active cooling at 70 C.

    `prod_eff` is the THERMAL (thermoneutral/HHV-referenced) efficiency N_s*I*V_th/P, so
    that P*(1 - prod_eff) is the waste heat. The caller derives it from the same operating
    current that sets the hydrogen rate; passing an LHV-referenced number here would
    over-state the heat by ~18 %.

    T_min_C sets the warm-standby (keep-warm) set-point. Normally an automatic BoP
    value of 40 C; the thermal-action agent may choose it so it can trade standby
    heating energy against a warmer, more efficient restart.

    T_max_C sets the chiller set-point, i.e. the ceiling the loaded stack is cooled to.
    Normally the fixed 70 C datasheet operating temperature; the chiller-action agent may
    choose it from CHILLER_LEVELS_C to trade chiller electricity (COP ~ 3) against the
    higher stack voltage a cooler stack presents to the array. Both set-points return
    their electrical cost through P_cooling_W, so the reward already prices them.

    Structure follows the validated single-node energy balance of Espinosa-Lopez
    et al. (46 kW PEM), scaled to 1 MW. The stack thermal mass is the sum of the
    metallic stack hardware (~2.5 t Ti/steel, cp~500 J/kg/K -> 1.25e6 J/K) and the
    circulating de-ionised water inventory (~2.1 t, cp 4186 J/kg/K -> 8.75e6 J/K),
    m_Cp ~ 1.0e7 J/K. This gives a full-load 60->70 C warm-up of ~7 min and a
    natural (idle) cooldown time constant tau = m_Cp/U_A ~ 4.2 h.
    """
    T_amb_K = T_amb_C + 273.15
    Q_gen   = P_pem_actual * (1.0 - prod_eff) if P_pem_actual > 0 else 0.0
    U_A     = 667.0          # W/K  — stack-to-ambient overall thermal conductance
    Q_loss  = U_A * (prev_T_pem - T_amb_K)
    m_Cp    = 10_000_000.0   # J/K  — stack metal + circulating-water inventory (1 MW)
    Q_net   = Q_gen - Q_loss
    delta_T = (Q_net * (DT_HOURS * 3600)) / m_Cp
    T_new   = prev_T_pem + delta_T

    P_cooling_W = 0.0   # electrical load of active thermal management (chiller OR keep-warm heater)
    # Cooling set-point. NREL/TP-5B00-79055 states the Giner ELX stack's operating temperature
    # as 70 C, and the 1 MW rating (3 A/cm^2, 266 V) is quoted there, so the default holds the
    # datasheet operating point.
    T_MAX = T_max_C + 273.15
    # Default to the shared BoP standby floor rather than a literal, so baselines and the
    # agent's lowest rung stay locked together (see KEEPWARM_BASELINE_C).
    if T_min_C is None:
        T_min_C = KEEPWARM_BASELINE_C
    T_MIN = T_min_C + 273.15  # warm-standby set-point (agent- or BoP-chosen keep-warm)
    if T_new > T_MAX:
        excess_heat_J    = (T_new - T_MAX) * m_Cp
        cooling_energy_J = excess_heat_J / 3.0  # chiller COP ~ 3
        P_cooling_W      = cooling_energy_J / (DT_HOURS * 3600)
        T_new            = T_MAX
    elif T_new < T_MIN:
        # Standby heating holds the idle stack at 40 C (resistive, COP 1);
        # the electrical keep-warm load is charged to the balance of plant.
        deficit_heat_J   = (T_MIN - T_new) * m_Cp
        P_cooling_W      = deficit_heat_J / (DT_HOURS * 3600)
        T_new            = T_MIN
    return T_new, P_cooling_W


def simulate_ems_routing(P_op_direct, mppt_loss, batt_energy, storage, g, n_par,
                         prev_T_pem, T_amb_C, SoH_BATT,
                         is_settlement=False, period_h2_direct=0.0,
                         is_last_step=False, annual_unmet_so_far=0.0, T_min_C=None,
                         T_max_C=70.0, pv_scale=1.0, I_op=0.0):
    """
    Unified EMS — physically consistent, daily H2 demand contract.

    Two-path H2 delivery (BOTH compressed — reactor is at P_DELIVER_BAR = 180 bar):
      • Direct path  : instantaneous production → compressed 40→180 bar → consumer.
      • Storage path : surplus production → compressed 40→(≤200 bar) → tank →
                       let-down to 180 bar → consumer at end of day.

    Split-string PV architecture:
      • n_par strings → PEM (direct coupling).
      • (40 - n_par) strings → battery MPPT converter.
      Excess beyond battery capacity → exported to grid (revenue at sell_price_kwh).

    Demand mode is controlled by the global DEMAND_MODE:
      'daily'  -> settle 78 kg every STEPS_PER_DAY=96 control steps
      'hourly' -> settle 3.25 kg every STEPS_PER_HOUR=4 control steps
    Direct demand is still represented continuously as 0.8125 kg per quarter-hour.
    Returns 25 values.
    """
    PERIOD_H2_DEMAND = hourly_H2_target if DEMAND_MODE == 'hourly' else daily_H2_target
    grid_bought = grid_sold = batt_charge_w = batt_discharge_w = 0.0

    # -- 1. PEM POWER (protected) --
    # The stack CURRENT is clipped with the power, otherwise Faraday's law below would be
    # handed the unclipped current against a clipped power and over-report production in
    # exactly the hours the rectifier is saturated.
    P_pem_actual = P_op_direct
    I_pem        = max(0.0, float(I_op))
    pv_excess_from_clip = 0.0
    if P_pem_actual > PEM_capacity:
        pv_excess_from_clip = P_pem_actual - PEM_capacity
        if P_op_direct > 0.0:
            I_pem *= PEM_capacity / P_op_direct
        P_pem_actual = PEM_capacity

    # -- 2. H2 PRODUCTION (Faraday's law) --
    # n_dot_H2 = N_series * I / (2F)  [mol/s]  ->  x M_H2 x 3600 -> kg/h. This is a RATE.
    # Intensive quantities (StH, the MOO/RL objective) need the rate; the mass balance needs
    # the AMOUNT produced in this step. They coincide only at DT_HOURS = 1, so keep both.
    # See the F_FARADAY block for why this replaced (P * prod_eff) / LHV.
    fl = np.clip(P_pem_actual / PEM_capacity, 0, 1)
    H2_rate_kgh = h2_rate_faraday(I_pem, P_pem_actual)
    H2_prod     = H2_rate_kgh * DT_HOURS        # kg produced in THIS step

    # Thermal efficiency for the energy balance, from the SAME current. Waste heat is
    # P - N_series*I*V_th (the enthalpy of reaction leaves as H2, the rest as heat), so the
    # fraction passed to the thermal model is N_series*I*V_th / P. Expressing it this way
    # keeps calculate_thermal_inertia()'s `P*(1 - eff)` form unchanged while making the heat
    # consistent with the hydrogen — the two used to come from different places.
    therm_eff = (N_SER_PEM * I_pem * V_TH_HHV / P_pem_actual) if P_pem_actual > 20 else 0.0
    therm_eff = float(np.clip(therm_eff, 0.0, 1.0))

    # Direct delivery: meet instantaneous demand from this hour's production.
    # H2_direct bypasses the storage tank, but is STILL compressed to the reactor
    # delivery pressure (40→180 bar) — see the compression block below.
    H2_direct   = min(H2_prod, step_H2_target)
    H2_to_store = H2_prod - H2_direct

    new_storage = storage + H2_to_store
    overflow    = max(0.0, new_storage - storage_capacity)
    storage     = min(storage_capacity, new_storage)

    # -- 3. PERIODIC DEMAND SETTLEMENT (hourly or daily physical boundary) --
    # Between settlement boundaries, H2_delivered contains only the direct quarter-hour path.
    # At the physical hourly/daily boundary, storage supplies any remaining contract demand.
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
        # Only add H2_from_store (storage draw) to this step's H2_delivered.
        # served_direct was already recorded in earlier hourly steps → no double-count.
        H2_delivered = H2_direct + H2_from_store

        # PERIODIC BOUNDARY CONDITION. Ending below the opening inventory means the plant
        # consumed hydrogen it did not replace, which the next year cannot repeat, so the
        # deficit is charged as unmet demand.
        if is_last_step and storage < INITIAL_H2_STORAGE:
            H2_unmet += INITIAL_H2_STORAGE - storage

    # -- 4. THERMAL MODEL & AUXILIARY LOADS --
    T_new, P_cooling_W = calculate_thermal_inertia(prev_T_pem, T_amb_C, P_pem_actual, therm_eff,
                                                   T_min_C=T_min_C, T_max_C=T_max_C)

    tank_fill_fraction = storage / storage_capacity
    # Physically-based compression.
    P_vessel_bar    = p_cat + tank_fill_fraction * (P_STORE_MAX_BAR - p_cat)
    comp_store_kwh  = h2_compression_kwh_per_kg(p_cat, P_vessel_bar)      # 40 → vessel (≤200 bar)
    comp_deliv_kwh  = h2_compression_kwh_per_kg(p_cat, P_DELIVER_BAR)     # 40 → 180 bar
    # Only H2 that actually ENTERS the tank is compressed. The overflow (surplus above
    # the tank buffer) is sold to the secondary off-taker at the cathode pressure
    # (~40 bar) and is neither compressed nor stored — so it carries no compression load.
    H2_stored_now   = max(0.0, H2_to_store - overflow)
    # kg x kWh/kg x 1000 = Wh for this step; divide by DT_HOURS to get W.
    P_compression_W = (H2_direct * comp_deliv_kwh + H2_stored_now * comp_store_kwh) * 1000.0 / DT_HOURS
    P_bop_W            = P_pem_actual * 0.05
    P_idle_W           = PEM_capacity * 0.02 if P_pem_actual < (PEM_capacity * 0.05) else 0.0
    # Every auxiliary load (multi-stage compressor, chiller, circulation pumps, keep-warm
    # heater) is a motor or resistive load fed from the common DC bus through a variable-
    # frequency drive, so the bus must supply MORE than the shaft/thermal demand above by the
    # drive's conversion loss.
    BOP_DRIVE_EFF      = 0.97
    P_aux_total        = (P_cooling_W + P_compression_W + P_bop_W + P_idle_W) / BOP_DRIVE_EFF

    # -- 5. PV POWER BUDGET (split-string architecture) --
    #   n_par strings           → MOSFET_PEM closed  → DC bus A → PEM
    #   (400 - n_par) strings    → MOSFET_BATT closed → MPPT converter → battery
    P_pv_active_mppt = P_op_direct + mppt_loss
    n_unused  = max(0, N_PAR_MAX - n_par)   # per-site installed array (top switch position)
    P_unused_PV_W = (
        P_pv_active_mppt * (n_unused / n_par)
        if (n_par > 0 and P_pv_active_mppt > 0) else 0.0
    )
    pv_available = P_unused_PV_W  + pv_excess_from_clip

    # -- 6. SATISFY AUXILIARIES: PV surplus → battery → grid (last resort) --
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

    # -- 7. CHARGE BATTERY FROM REMAINING PV SURPLUS; export to grid ONLY if H2 is met --
    P_pv_excess_W = pv_available
    if batt_discharge_w == 0.0 and P_pv_excess_W > 0:
        headroom_wh   = max(0.0, BATT_CAPACITY_WH * BATT_SOC_MAX - batt_energy)
        actual_charge = min(P_pv_excess_W, BATT_MAX_P_W, headroom_wh / DT_HOURS)
        batt_charge_w += actual_charge
        batt_energy   += actual_charge * DT_HOURS * BATT_EFF

        # Grid export is allowed only if (a) globally enabled, (b) H2 demand is met, AND (c) the
        # tank is above its seasonal-buffer level (fill storage first, export only the genuine
        # surplus).
        surplus   = max(0.0, P_pv_excess_W - actual_charge)
        tank_frac = storage / storage_capacity if storage_capacity > 0 else 1.0

        if ALLOW_GRID_EXPORT and H2_unmet < 1e-3 and tank_frac >= EXPORT_TANK_MIN_FRAC:
            grid_sold = surplus
        else:
            grid_sold = 0.0  # export off / H2 unmet / buffer not yet full → surplus curtailed

    # -- 8. BATTERY DEGRADATION (cycle-throughput + calendar) --
    # CORRECTED 2026-07-30. The previous line was batt_deg_step = (charge + discharge)*DT /
    # (3000 * 0.8 * C) which overstated cycle ageing by 12.5x and cost the pack 12-28 % of its
    # health in a single year (Odeillo DRL finished at 71.9 % SoH).
    eq_full_cycles = ((batt_charge_w + batt_discharge_w) * DT_HOURS
                      / (2.0 * BATT_CAPACITY_WH))
    batt_cycle_deg = BATT_EOL_LOSS * eq_full_cycles / BATT_CYCLE_LIFE
    # Calendar ageing proceeds whether or not the pack is cycled; a model in which an idle
    # battery never ages is unphysical and would let a controller "preserve" the pack for
    # free simply by not using it.
    batt_cal_deg   = (BATT_CALENDAR_LOSS_PER_YR / 8760.0) * DT_HOURS
    SoH_BATT = max(0.0, SoH_BATT - batt_cycle_deg - batt_cal_deg)

    # -- 9. CURTAILMENT TRACKING (Option C: gated export) --
    # If H2 unmet, curtail surplus instead of exporting to grid
    energy_curtailed = 0.0
    if H2_unmet > 1e-3:  # H2 not fully met
        energy_curtailed = max(0.0, P_pv_excess_W - batt_charge_w) if P_pv_excess_W > 0 else 0.0

    # -- 10. SYSTEM-LEVEL METRICS --
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

    Numerically identical to calling pv_model() per point, but the scalar version was the
    single largest cost in the whole simulation: perturb_observe_series() called it ~66
    times per sweep and each call issued ~10 NumPy operations on SCALARS, where the
    per-call dispatch overhead dwarfs the arithmetic. One sweep therefore cost well over a
    thousand NumPy calls; here it costs about ten.

    The `for _ in range(50)` loop in pv_model() is NOT a fixed-point iteration and is not
    reproduced: Is1/Is2/Isi depend only on temperature, Vd only on voltage, and Id only on
    those, so nothing in the loop body depends on I_total. Id is bit-identical on every
    pass, and the convergence test therefore always exits by the second iteration. The
    loop recomputed the same value twice and discarded the first result.
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
    """Array-level P&O sweep. Same outputs as the original scalar while-loop, including
    the truncation at the first non-positive current, which is what bounds the curve at
    open circuit."""
    v_arr = np.arange(v_start, v_max + 0.5 * v_step, v_step, dtype=float)
    if v_arr.size == 0:
        return [], [], [], 0
    cur = _pv_curve_vec(v_arr, params, irradiance) * num_parallel
    tot_v = v_arr * num_series
    pw = tot_v * cur

    # The scalar loop appended the point, then broke AFTER the first current <= 0, so that
    # point is included and everything past it is not.
    nz = np.flatnonzero(cur <= 0)
    n = int(nz[0]) + 1 if nz.size else v_arr.size

    tot_v, cur, pw = tot_v[:n], cur[:n], pw[:n]
    # ">" not ">=", matching the original: the FIRST maximum wins on ties.
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

    Returns the PV-curve point closest in the (V, I) plane to any point of the PEM
    curve. It ALWAYS returns a point, including when the two curves never cross,
    in which case the returned pair is not a solution of the coupled circuit and
    can lie outside the modelled PEM voltage range. Superseded by
    find_intersection_point(); see the note there.
    """
    V_pv = np.array(voltages_pv)[:, None]
    I_pv = np.array(currents_pv)[:, None]
    V_pm = np.array(voltages_pem)[None, :]
    I_pm = np.array(currents_pem)[None, :]
    dist = np.sqrt((V_pv - V_pm)**2 + (I_pv - I_pm)**2)
    idx  = np.unravel_index(np.argmin(dist), dist.shape)
    return float(voltages_pv[idx[0]]), float(currents_pv[idx[0]])


def find_intersection_point(voltages_pv, currents_pv, voltages_pem, currents_pem):
    """Direct-coupled operating point, or None when the curves do not cross.

    In a directly coupled array-stack pair the operating point is fixed by Kirchhoff:
    common voltage, common current. It is therefore the point where the PV and PEM
    I-V characteristics genuinely intersect, detected here as a sign change of
    (I_pv - I_pem) over the shared voltage range and located by linear interpolation.

    None means the array over-supplies the stack: across the whole shared range the
    PV curve lies above the PEM curve, so no common (V, I) exists at or below the
    3 A/cm^2 datasheet current density. The stack cannot be operated in that string
    configuration and the step produces no hydrogen. The controller must trim strings
    to stay coupled, which is the purpose of the MOSFET ladder.

    This replaces nearest_point_match(), which returned the geometrically closest PV
    point whether or not a crossing existed. That function always succeeded, so an
    over-supplied configuration silently yielded an operating point outside the
    modelled PEM range (up to ~396 V against a 367 V curve maximum) and appeared to
    the optimiser as a feasible, high-power choice.
    """
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


def V_cell(T, p_cat_, p_an_, delta, A_cp, a_a, a_c, i, i0a, i0c, Tref, ilim):
    """Semi-empirical PEM single-cell voltage (Maroufmashat2014 + Springer1991).

    `i` is the TOTAL cell current in A; `A_cp` the active area in cm^2, so the
    current density is j = i / A_cp in A/cm^2. `ilim` is likewise a current
    DENSITY (6.1 A/cm^2).

    Two terms were corrected on 2026-07-26:

    * Vohm — previously `delta / (A_cp*k6 - k7*exp(...))`, which (a) placed the
      cell area inside the conductivity, (b) applied the Springer exponential to
      only the second constant instead of the whole (k6 - k7) bracket, inverting
      its temperature dependence, and (c) omitted the current entirely. The term
      evaluated to ~39 uV/cell — effectively absent — which left the stack a stiff
      voltage sink and made the direct-coupling match artificially good. The
      correct Springer form is sigma(T) = (k6 - k7)*exp(k8*(1/303 - 1/T)) S/cm,
      giving Vohm = (delta/sigma)*j, ~103 mV/cell at 3 A/cm^2 and 60 C.
    * Vcon — the current density was hard-coded as `1`, making the concentration
      overpotential constant at all loads. Now uses the actual j.
    * Vact — was driven by the TOTAL cell current in A while i0a/i0c are quoted in
      A/cm^2, overstating the argument of both logarithms by a factor A_cp = 1250.
      That inflated the cell voltage to ~2.6 V at 2 A/cm^2 against a real ~1.95 V.
      Now driven by j, giving 1.82 / 1.93 / 2.02 V at 1 / 2 / 3 A/cm^2 and 60 C.

    NOTE: i0a/i0c are temperature-independent unless PEM_I0_ARRHENIUS is set, in
    which case `Tref` (previously accepted and never used) is the reference
    temperature at which they are quoted. With the flag off, Vact rises linearly
    with T and the model predicts a hotter stack is slightly LESS efficient —
    read the thermal results with that in mind.
    """
    R, F = 8.314, 9.648e4
    k1, k2, k3, k4, k5 = 8.5e-4, R/F, 6.1078e-3, 17.694, 34.85
    k6, k7, k8, k9     = 0.1113, 3.26e-3, 1268, R/F
    sat  = k3 * np.exp(k4 * ((T-273.15) / (T-k5)))
    pc   = np.clip(p_cat_ - sat, 1e-3, 1e6)
    pa   = np.clip(p_an_  - sat, 1e-3, 1e6)
    ph2o = np.clip(sat,          1e-6, 1e6)
    Erev = 1.229 - k1*(T-298.15) + (k2/2)*T*np.log(np.clip((pc*pa**2)/ph2o, 1e-20, 1e20))

    # current density [A/cm^2], capped just below the limiting value
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
    return Erev + Vohm + Vact + Vcon


def V_PEM(T, p_c, p_a, delta, A_cp, a_a, a_c, i, i0a, i0c, Tref, ilim, n_series):
    return V_cell(T, p_c, p_a, delta, A_cp, a_a, a_c, i, i0a, i0c, Tref, ilim) * n_series


def I_PEM(i_cell_, n_parallel):
    return i_cell_ * n_parallel

# =============================================================================
# %% SYSTEM SIZING REPORT
# =============================================================================

def print_system_sizing():
    """
    Compute and print key system parameters at STC (G=1000 W/m², T=25°C).
    PV MPP is found numerically using the same physics model as the simulation.
    """
    G_stc, T_stc = 1000.0, 25.0
    N_PAR_MAX    = globals()['N_PAR_MAX']   # per-site installed strings (set by the loop)
    N_PAR_MIN    = globals()['N_PAR_BASE']
    N_SER_PEM    = globals()['N_SER_PEM']

    # PV at STC — full array (n_par = 40)
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

    # PV at STC — minimum active strings (n_par = 24, baseline)
    vols_min, curs_min, _, _ = perturb_observe_series(params_stc, eff_stc,
                                                      N_SERIES_PV, N_PAR_MIN, v_step=1)
    if vols_min:
        Vm_min, Im_min, Pm_min = find_mppt(vols_min, curs_min)
    else:
        Vm_min = Im_min = Pm_min = 0.0

    # PEM at 60 °C operating temperature, rated current
    T_pem_op  = 60.0 + 273.15
    pem_I_arr = np.linspace(0.0, float(I_PEM(i_cell, 1)), 200)
    V_pem_arr = V_PEM(T_pem_op, p_cat, p_an, delta_mem, A_cell_pem,
                      a_an, a_cat, pem_I_arr, i_0_an, i_0_cat,
                      T_ref_pem, i_lim, N_SER_PEM)
    # PEM power curve → rated point
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

def prewarm_setpoint_rule(T_pem_C, g_fcst, mode='anticipatory'):
    """Hand-designed warm-standby rule -- the 5th ('MOO+PreWarm') baseline.

    The point of this controller is to separate what the EXTRA ACTUATOR is worth from
    what the LEARNING is worth. The DRL agent is the only controller that commands the
    keep-warm set-point, and the mechanism decomposition attributes ~97 % of its stack
    health advantage to the per-restart term, so 'DRL beats the baselines on durability'
    and 'a heater beats no heater' are not separated by the 4-way comparison alone.
    This gives a memoryless-plus-forecast rule the SAME heater and the same ladder.

    mode='anticipatory' -- the minimal rule an engineer would write: hold the 60 C rung
        whenever any irradiance above the cut-in is expected inside the forecast horizon,
        otherwise fall back to the 30 C floor. Two inputs, no training, no model.
    mode='distilled'    -- the depth-3 tree distilled from the trained agent's own
        keep-warm action at 99 % fidelity (xai_9). Reproduced verbatim so the rule the
        agent actually learned can be run as a controller in its own right.

    T_pem_C : current stack temperature (deg C)
    g_fcst  : irradiance forecasts (W/m2) at +1,+2,+3,+4 physical hours.
    Returns a set-point drawn from KEEPWARM_LEVELS_C, so the rule can never spend less
    thermal energy than the baselines' fixed floor nor more than the agent's ceiling.
    """
    lo, mid, hi = KEEPWARM_LEVELS_C          # (30, 45, 60)
    if mode == 'distilled':
        g2 = float(g_fcst[1]) if len(g_fcst) > 1 else 0.0
        g3 = float(g_fcst[2]) if len(g_fcst) > 2 else 0.0
        if T_pem_C > 60.0 and g2 > 1.0:                  return hi
        if T_pem_C > 37.5 and g3 > 196.0:                return mid
        if g2 > 67.0 or g3 > 196.0:                      return hi
        return lo
    # 'anticipatory' (default). 80 W/m2 is the direct-coupling cut-in used throughout
    # this file (see the `g > 80` guards in the production terms) -- the irradiance below
    # which no operating point exists and the stack is idled.
    return hi if (len(g_fcst) and max(g_fcst) > 80.0) else lo


def run_baselines(irr_data, temp_data_C, price_data_mwh, use_rf=False, hysteresis=None,
                  lambda_sw=None, pv_scale=1.0, prewarm=None,
                  prewarm_offsets=FORECAST_OFFSETS_STEPS):
    # hysteresis: None -> use the global MOO_HYSTERESIS; True/False -> override it, so greedy-
    # MOO and hysteresis-MOO can both be produced in a single execution. lambda_sw: None -> use
    # the global MOO_LAMBDA_SW; a float overrides it.
    # prewarm: None -> the fixed BoP floor KEEPWARM_BASELINE_C, i.e. the three baselines of the
    #   paper. 'anticipatory' or 'distilled' -> run the rule-based warm-standby controller of
    #   prewarm_setpoint_rule() on top of the SAME greedy string search, giving the 5th baseline.
    #   The string-count logic is untouched, so any difference against greedy MOO is the thermal
    #   action alone, and any residual difference against the DRL is what the LEARNING adds.
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
    deg_diag            = {'load': 0.0, 'rccd': 0.0, 'cold': 0.0}

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
        # Floor must match PVPEMEnv.step()'s 0.7 (and Eq. delta_mem_eff in the paper).
        # Was 0.1 here, so at low SoH the baselines would have been handed a much thicker
        # effective membrane than the agent -- a controller-family asymmetry in the shared
        # physics. Latent while SoH stays above 0.7, but the cold-start recalibration makes
        # low-SoH excursions more likely, so it is aligned now rather than left to bind.
        deg_delta_mem = delta_mem / max(SoH_PEM, 0.7)
        dynamic_pem_V = V_PEM(T_pem, p_cat, p_an, deg_delta_mem, A_cell_pem,
                              a_an, a_cat, pem_I_array, i_0_an, i_0_cat, T_ref_pem, i_lim, N_SER_PEM)

        best_score  = float('inf')
        V_op_rec = I_op_rec = 0.0   # operating point of the CHOSEN configuration
        # Held, not top-rung: the optimiser below only runs at g >= 80, so this default IS
        # the night configuration. It must match the env's hold rule or the two families
        # of controller are counted on different conventions.
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
                    # RF narrows the search to the predicted position ± 2 (plus the previous
                    # position) and takes the GLOBAL best by the SAME objective full MOO uses —
                    # no hysteresis / dead-band.
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
                            # Soft switching charge (penalised-MOO baseline). Score is
                            # MINIMISED, so a positive charge discourages moving away from
                            # prev_mosf. Identical form to the DRL reward's switching term.
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

                        # REWARD-BASED SCORING (aligned with RL)
                        fl = np.clip(P_op_direct_temp / PEM_capacity, 0, 1) if P_op_direct_temp > 20 else 0
                        H2_prod_temp = h2_rate_faraday(I_op, P_op_direct_temp)
                        StH_temp = float(np.clip((H2_prod_temp * LHV) / (eff_G * N_SERIES_PV * n_par * pv_scale * pv_module_area), 0, 0.185)) if g > 80 else 0.0

                        # Score: same multi-objective the RL reward maximises (shared
                        # W_H2/W_STH/W_MPPT constants) so MOO and RL optimise identically.
                        score = -DT_HOURS * (W_H2 * (H2_prod_temp / H2_PEAK_KGH) + W_STH * (StH_temp / 0.185) - W_MPPT * (mppt_loss_temp / MPPT_NORM))
                        # Soft switching charge (penalised-MOO baseline). Score is MINIMISED, so
                        # a positive charge discourages moving away from prev_mosf.
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
            # Hold the previous config unless the greedy optimum moves by >= MOO_DEADBAND
            # string-steps; recompute the held operating point so P_op_direct / mppt_loss
            # stay consistent with best_n_par. Daytime only (P_op_direct > 0).
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

        # Stack temperature BEFORE this step, i.e. the temperature the stack is at when
        # current begins to flow. Captured here because the tuple-unpack below overwrites
        # T_pem with the post-step value, and the cold-start classification needs the
        # pre-step one (see calculate_degradation_step).
        T_pem_prev_step = T_pem
        # Warm-standby set-point for this step. None -> the fixed BoP floor (the three
        # baselines of the paper); a rule name -> the 5th, rule-based thermal baseline.
        if prewarm is None:
            _T_min_C = KEEPWARM_BASELINE_C
        else:
            # Match the hourly baseline's information exactly: forecasts at
            # +1,+2,+3,+4 physical hours. Near the end of the year clamp to the
            # final row, exactly as PVPEMEnv._obs() does.
            _fc = [
                float(np.clip(irr_data[min(i + off, len(irr_data) - 1)], 0, 1200))
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
            I_op=I_op_rec,     # operating current of the CHOSEN configuration — sets H2 by Faraday
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
        # Secondary off-take: surplus H2 that overflowed the tank is sold rather than dumped.
        h2_secondary_kg  = overflow if H2_SECONDARY_OFFTAKE else 0.0
        h2_secondary_eur = h2_secondary_kg * H2_SECONDARY_PRICE_EUR_KG
        profit_eur    = h2_rev_eur + grid_econ_eur + h2_secondary_eur

        DEMAND_PER_PERIOD = hourly_H2_target if DEMAND_MODE == 'hourly' else daily_H2_target
        records.append(dict(
            step=i, irradiance=g, temperature=t_amb, price_mwh=cur_price_mwh,
            n_parallel=best_n_par,
            H2_daily_demand=DEMAND_PER_PERIOD if (is_settlement or is_last_step) else 0.0,
            H2_prod=H2_prod, StH=StH, mppt_loss=mppt_loss,
            # Direct-coupled operating point. Zero when the curves do not cross,
            # i.e. the array over-supplies the 3 A/cm^2 stack limit at this n_par.
            V_op=V_op_rec, I_op=I_op_rec, P_op_W=V_op_rec * I_op_rec,
            storage=storage, T_pem=T_pem,
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
            T_keepwarm_C=_T_min_C,  # the fixed BoP floor, or the rule's set-point if prewarm is on.
            # Was a hardcoded 40.0, which mis-reported the baselines whenever the ladder
            # floor was not 40 -- the Final_daily run logged 40 while running at 30.
        ))

    if use_rf:
        print(f"  -> RF Predictions: {prediction_used_count} | Full Optimizations: {full_optimization_count}")

    _dsum = sum(deg_diag.values()) or 1e-30
    print(f"  -> PEM SoH breakdown (pp lost): load={100*deg_diag['load']:.3f}  "
          f"rccd={100*deg_diag['rccd']:.3f}  restart-event={100*deg_diag['cold']:.3f}  "
          f"(load {100*deg_diag['load']/_dsum:.0f}% / rccd {100*deg_diag['rccd']/_dsum:.0f}% "
          f"/ restart {100*deg_diag['cold']/_dsum:.0f}% of total)")
    print("     restart-event term includes BOTH cold and warm restart SoH charges; "
          "use the cold-start COUNT below to distinguish them.")

    return pd.DataFrame(records)

# =============================================================================
# %% RL ENVIRONMENT
# =============================================================================

class PVPEMEnv(gym.Env):
    # mosf 0..6 -> n_par = BASE_PARALLEL + mosf * PARALLEL_PER_MOS. The ladder is taken
    # from the module-level N_PAR_BASE/N_PAR_STEP at construction time, so each site runs
    # its own installed string count (see site_string_config).
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
        # Per-site string ladder: the installed array is set by N_par, not by scaling Iph.
        self.BASE_PARALLEL    = N_PAR_BASE
        self.PARALLEL_PER_MOS = N_PAR_STEP
        # Action: string config only (Discrete) or (string config, keep-warm level)
        # when the thermal-setpoint extension is enabled (MultiDiscrete).
        if THERMAL_ACTION:
            _dims = [len(self.MOSFET_OPTIONS), len(KEEPWARM_LEVELS_C)]
            if CHILLER_ACTION:
                _dims.append(len(CHILLER_LEVELS_C))   # 3-D: (config, keep-warm, chiller)
            self.action_space = spaces.MultiDiscrete(_dims)
        else:
            self.action_space = spaces.Discrete(len(self.MOSFET_OPTIONS))

        # 14 base states + 3 variables at each of four physical forecast horizons (+1,+2,+3,+4 h)
        obs_dim = 14 + (3 * self.lookahead)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        self._G_max, self._T_lo, self._T_hi    = 1200.0, -30.0, 80.0
        # Stack-temperature normalisers, derived from the set-point ladders rather than written
        # out: calculate_thermal_inertia() clamps T_pem to [T_min, T_max], so the reachable band
        # is exactly [keep-warm floor, highest chiller set-point].
        self._Tpem_lo = KEEPWARM_LEVELS_C[0] + 273.15
        self._Tpem_hi = max(CHILLER_LEVELS_C) + 273.15
        # Observation normalisers. n_par and array current MUST track the installed array:
        # with a per-site string count (e.g. 593 at 1.6x) the old fixed 400/4500 A limits
        # saturated the observation at 1.0 and hid the state from the policy.
        self._V_max    = 400.0
        self._n_p_max  = float(N_PAR_MAX)
        self.max_steps  = len(irradiance_data)
        self._pem_I     = np.linspace(0, I_PEM(i_cell, self.N_PARALLEL_PEM), 100)
        self._I_max_pem = float(np.max(self._pem_I))
        # Operating current is the SHARED direct-coupled current, so it can never exceed the
        # stack's own 3 A/cm^2 ceiling (3750 A) no matter how large the array is. Scaling this
        # normaliser with N_PAR_MAX therefore did not "track the installed array", it just
        # inflated the denominator: measured max I_op is 3750 A against a 9731 A denominator at
        # Lund, so the current observation only ever spanned 0..0.39 and the policy saw a
        # compressed version of one of its most informative states. Normalise by the physical
        # ceiling instead.
        self._I_max    = self._I_max_pem
        self.reset()

    def _n_par(self, m):
        return self.BASE_PARALLEL + m * self.PARALLEL_PER_MOS

    @staticmethod
    def _c01(x):
        """clip to [0,1]. Plain min/max on a Python float, NOT np.clip: np.clip on a
        scalar dispatches through _wrapfunc and costs ~2 us, and this observation vector
        issues fourteen of them every step. That showed up as ~46 np.clip calls per env
        step and roughly half the total step time once the PV sweep was vectorised."""
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
            # Stack temperature, normalised over the band the stack can actually occupy.
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
            # TRAIN-ONLY forecast augmentation copied from the hourly policy in
            # PHYSICAL time: ±10% at +1 h, ±20% at +2 h, ±30% at +3 h,
            # ±40% at +4 h. The previous 15-min conversion incorrectly used the
            # quarter-hour index (1..16), giving ±160% at +4 h.
            if self.eval_mode:
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
        self._deg_diag        = {'load': 0.0, 'rccd': 0.0, 'cold': 0.0}

        # LCOH tracking (for reward calculation)
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
            # Narrow randomisation around the EVALUATION regime. Evaluation always starts at
            # SoH=1.0 and only drifts to ~0.98 over a year, so training on the old wide bands
            # (PEM 0.60–1.0) meant ~90% of training happened in health states never seen at
            # eval.
            self.SoH_PV   = np.random.uniform(0.97, 1.0)
            self.SoH_PEM  = np.random.uniform(0.95, 1.0)
            self.SoH_BATT = np.random.uniform(0.95, 1.0)

        self.initial_soh_pem = self.SoH_PEM
        self.initial_soh_batt = self.SoH_BATT
        return self._obs(), {}

    def step(self, action):
        cur_G, cur_T, cur_price_mwh = self.G, self.T_C, self.price[self.step_count]
        self.prev_mosf = self.mosf
        # Decode the action: string-config index (+ keep-warm and chiller set-points when
        # the thermal extensions are on). The chiller rung defaults to the datasheet 70 C
        # whenever CHILLER_ACTION is off, which is what the baselines hold.
        if THERMAL_ACTION:
            _a = np.asarray(action).ravel()
            mosf_action  = int(_a[0])
            self.T_min_C = float(KEEPWARM_LEVELS_C[int(_a[1])])
            self.T_max_C = float(CHILLER_LEVELS_C[int(_a[2])]) if CHILLER_ACTION else 70.0
        else:
            mosf_action  = int(action)
            self.T_min_C = KEEPWARM_BASELINE_C
            self.T_max_C = 70.0
        # The ceiling must stay above the floor: the keep-warm ladder tops out at 60 C and the
        # chiller ladder bottoms at 60 C, so they can coincide but never invert.
        if self.T_max_C < self.T_min_C:
            self.T_max_C = self.T_min_C
        #self.mosf = len(self.MOSFET_OPTIONS) - 1 if cur_G < 80 else int(self.MOSFET_OPTIONS[mosf_action])
        # NIGHT: HOLD the previous configuration, do not snap to the top rung.
        # At G < 80 there is no generation, so the string count is electrically irrelevant --
        # it changes no power, no hydrogen and no cost. Snapping it to the top every night
        # therefore bought nothing, but it charged two hardware switches per day to any
        # controller whose DAYTIME preference was not the top rung, i.e. to exactly the
        # controllers that carry a switching penalty. Measured on the previous run: Prague's
        # DRL sat on the top rung for 11 % of daylight hours against MOO's 57 % and was
        # charged 421 dawn/dusk transitions against MOO's 10, which alone turned a +10.5 %
        # daytime difference into the +43.0 % that was reported. Holding is also the more
        # physical convention: no plant reconfigures its array at midnight for no reason.
        requested_mosf = self.prev_mosf if cur_G < 80 else int(self.MOSFET_OPTIONS[mosf_action])
        self.mosf = requested_mosf

        buy_price_kwh  = cur_price_mwh / 1000.0
        sell_price_kwh = buy_price_kwh * 0.8
        n_par = self._n_par(self.mosf)

        P_op_direct = mppt_loss = V_op = I_op = 0.0
        valid = False

        deg_Iph_ref   = Iph_ref * self.SoH_PV * self.pv_scale
        deg_delta_mem = delta_mem / max(self.SoH_PEM, 0.7)
        dynamic_pem_V = V_PEM(self.T_pem, p_cat, p_an, deg_delta_mem, A_cell_pem,
                              a_an, a_cat, self._pem_I, i_0_an, i_0_cat,
                              T_ref_pem, i_lim, self.N_SERIES_PEM)

        if cur_G >= 80:
            params_t = [deg_Iph_ref, Is_ref1, Is_ref2, Is_refi, Rs, Rsh,
                        n1, n2, ni, rho_contact, A_cell, aSi_thickness, TCO_resistivity, cur_T]
            eff_G = calculate_effective_irradiance(cur_G, 0.1 * cur_G)
            vols, curs, _, _ = perturb_observe_series(params_t, eff_G, self.N_SERIES_PV, n_par)

            if vols:
                V_mpp, I_mpp, _ = find_mppt(vols, curs)
                # The stack is limited to the 3 A/cm^2 datasheet current density, so a string
                # count that drives the array past that limit has NO valid direct-coupled
                # operating point.
                pt = find_intersection_point(vols, curs,
                                             dynamic_pem_V.tolist(), self._pem_I.tolist())

                # If the requested string count over-supplies the stack there is no crossing, so
                # the hardware steps DOWN the ladder until a valid operating point exists.
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
                # else: P_op_direct and mppt_loss stay 0. No operating point means no
                # generation on the stack branch, so nothing is recoverable from it
                # either. The unused strings still feed the battery via P_unused_PV_W.

        is_last_step  = (self.step_count == self.max_steps - 1)
        # Same physical contract as the baselines: hourly settlement every 4 quarter-hours,
        # or daily settlement every 96 quarter-hours.
        _settlement_steps = STEPS_PER_HOUR if DEMAND_MODE == 'hourly' else STEPS_PER_DAY
        is_settlement = ((self.step_count + 1) % _settlement_steps == 0)
        pass_h2_direct = self.period_h2_direct
        T_pem_prev_step = self.T_pem   # stack temperature BEFORE this step (thermal-cycling penalty)
        # Load fraction BEFORE this step. Captured here because self.prev_fl is overwritten
        # further down, well before the reward is assembled -- reading it at the penalty
        # would compare fl against itself and the idle->producing edge could never fire.
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
            I_op=I_op,         # operating current of the CHOSEN configuration — sets H2 by Faraday
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
        # Secondary off-take: surplus H2 that overflowed the tank is sold rather than dumped.
        h2_secondary_kg  = overflow if H2_SECONDARY_OFFTAKE else 0.0
        h2_secondary_eur = h2_secondary_kg * H2_SECONDARY_PRICE_EUR_KG
        profit_eur    = h2_rev_eur + grid_econ_eur + h2_secondary_eur

        # ═══════════════════════════════════════════════════════════════════════════
        # REWARD = MOO multi-objective − switching-fatigue penalty The positive term is
        # IDENTICAL to the MOO / MOO+RF objective in run_baselines() (shared W_H2 / W_STH /
        # W_MPPT constants): moo_obj = W_H2·(H2_prod/1.6) + W_STH·(StH/0.185) −
        # W_MPPT·(mppt_loss/2000) Because the reward optimises the SAME quantities MOO does, a
        # converged PPO policy reproduces MOO's H2 production, StH and MPPT loss — that is the
        # goal.

        # --- MOO multi-objective (same normalisation constants as run_baselines) ---
        # Continuous reward density integrated over the physical step. At DT=1 h this is
        # numerically identical to the frozen hourly formulation; at DT=0.25 h four
        # quarter-hours sum to the same one-hour contribution.
        moo_obj = DT_HOURS * (
              W_H2   * (H2_prod / (H2_PEAK_KGH * DT_HOURS))
            + W_STH  * (StH / 0.185)
            - W_MPPT * (mppt_loss / MPPT_NORM)
        )

        # --- Switching-fatigue penalty (state-aware advantage of PPO over greedy MOO)
        # CHARGE THE REALIZED MOVE, NOT THE REQUEST. self.mosf has already been stepped
        # down by the feasibility loop above whenever the requested string count
        # over-supplies the stack and no I-V crossing exists. Charging
        # |requested - prev| billed the agent for MOSFET transitions the hardware never
        # made: holding a constant top-rung request for one year was charged 4731 events
        # / sum|d|=16190 against 2542 events / sum|d|=3265 actually realized, i.e. 420
        # reward points per year of penalty for moves that did not happen (267 at hourly
        # resolution). Because run_baselines only ever scores rungs that HAVE a valid
        # crossing, and counts its switches from the realized n_parallel, that charge fell
        # on the learned policy alone and taught it to stop requesting high string counts
        # -- the direct cause of the agent plateauing ~12 % below the greedy MOO on annual
        # hydrogen while showing a LOWER MPPT shortfall. Realized-move charging is both
        # the physical truth (the switches move to self.mosf) and the only basis on which
        # the DRL and the optimizer baselines are counted alike.
        mosf_diff = abs(self.mosf - self.prev_mosf)
        # FROZEN HOURLY REWARD SEMANTICS:
        # the production source uses lambda_sw * |delta rung| * DT_HOURS.  At hourly
        # resolution DT_HOURS=1, so this factor is numerically hidden.  Preserve the
        # same functional form at 15 min: the switching term is a timestep-normalized
        # regularizer whose local trade against the continuous operating objective must
        # shrink with the decision interval.  Cold-start degradation remains a true
        # fixed event charge below.
        switching_penalty = LAMBDA_SWITCH * mosf_diff * DT_HOURS

        # --- PEM-underutilisation penalty (fixes low-n_par camping) -----------------
        # FLAT penalty on strings left idle while the PEM is not yet saturated: those strings
        # could be added to drive more current into the PEM (higher fl → more H2) instead of
        # being diverted to battery/grid.
        n_par_max        = self._n_par(len(self.MOSFET_OPTIONS) - 1)   # site-specific top rung
        n_unused_frac    = (n_par_max - n_par) / n_par_max
        # Do NOT charge unused strings at night or when the PV/PEM curves do not intersect.
        # In those states the agent cannot turn the omitted strings into H2, so a penalty would
        # be action-independent noise (the old formulation fired for ~8600/8760 steps).
        underuse_penalty = 0.0
        if cur_G >= 80.0 and valid and fl < 0.98:
            underuse_penalty = DT_HOURS * W_UNDERUSE * n_unused_frac

        # --- Keep-warm energy penalty (thermal-action extension) --------------------
        # At idle (fl < 0.05) the thermal-management load Cooling_W is the resistive warm-
        # standby heating the agent controls via its keep-warm set-point.
        keepwarm_penalty = 0.0
        if THERMAL_ACTION and fl < 0.05:
            keepwarm_penalty = DT_HOURS * W_KEEPWARM * (Cooling_W / KEEPWARM_NORM_W)

        # --- Chiller-energy penalty (chiller-action extension) ----------------------
        # P_cooling_W carries EITHER standby heating (stack below T_min, idle) OR chiller work
        # (stack above T_max, loaded) — never both in one step, since the two branches of
        # calculate_thermal_inertia() are mutually exclusive.
        chiller_penalty = 0.0
        if THERMAL_ACTION and CHILLER_ACTION and fl >= 0.05:
            chiller_penalty = DT_HOURS * W_CHILLER * (Cooling_W / KEEPWARM_NORM_W)

        # --- Cold-start event penalty ------------------------------------------------
        # Fires only on idle -> producing transitions classified as cold by the SAME
        # pre-restart 50 C threshold used by the degradation and reporting models.
        #
        # IMPORTANT: this is intentionally a fixed event charge, not a penalty that tends
        # to zero as T_pre approaches 50 C from below. The degradation model applies the
        # full COLDSTART_SOH_COLD charge to every restart below 50 C, so the former depth-
        # only reward admitted a loophole in which PPO could park near 45-49 C while still
        # receiving the full cold-start degradation charge.
        coldstart_penalty = 0.0
        if COLDSTART_REWARD and fl >= 0.05 and fl_prev_step < 0.05:
            T_pre_C = T_pem_prev_step - 273.15
            if T_pre_C < COLDSTART_THRESHOLD_C:
                coldstart_penalty = W_COLDSTART * COLDSTART_EXCESS_DAMAGE_FRAC

        # --- Thermal-cycling penalty (thermal-action extension) ---------------------
        # Charge per-step PEM stack-temperature swings. The load-based RCCD SoH term does not
        # price the thermal/mechanical fatigue of cooling then reheating the stack, so without
        # this the agent minimises standby heating by deep-cycling the temperature.
        tcycle_penalty = 0.0
        if THERMAL_ACTION:
            tcycle_penalty = W_TCYCLE * (abs(self.T_pem - T_pem_prev_step) / TCYCLE_NORM_C)

        # --- Steady cold-operation penalty (see W_WARMOPS above) --------------------
        # Charges every producing step for the depth below the 70 C degradation reference,
        # the same quantity f_therm prices in calculate_degradation_step(). Unlike
        # coldstart_penalty this is NOT gated on the idle->producing transition, so it also
        # catches a stack that settles at a cooler ceiling and stays there.
        warmops_penalty = 0.0
        if THERMAL_ACTION and fl >= 0.05:
            coldrun_depth_C = max(0.0, T_DEG_REF_K - self.T_pem)
            warmops_penalty = DT_HOURS * W_WARMOPS * (coldrun_depth_C / TCYCLE_NORM_C)

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
    """Solve for the tank level the year reproduces: iterate S_end = f(S_start).

    Once the tank fills to capacity the initial condition is erased, so f is effectively
    constant there and this converges in one or two passes. Returns (S0, history).
    """
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
    # on-plot title removed — caption: "RL Training Convergence -- <location>"
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
    # PPO's own seed= covers its internal RNG and the vec-env wrapper, but this env's
    # domain randomization (initial SoH draws) and training-time forecast noise call
    # np.random.uniform() directly against the GLOBAL numpy RNG, which PPO's seed does
    # NOT touch. Without this, two runs with the same seed= still diverge.
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
    Guards against 'int object is not iterable' when a study flag is set to e.g. 3."""
    if isinstance(seeds, (int, np.integer)):
        return tuple(range(int(seeds)))
    return tuple(seeds)


def _seed_summarize_run(df):
    """Headline metrics from one evaluate_rl_single_run() DataFrame."""
    n = pd.to_numeric(df['n_parallel']).to_numpy()
    # Counts every change in n_parallel (incl. forced G<80 night moves), so it
    # tracks but may slightly exceed the reported "MOSFET switches/yr". The
    # cross-seed STD (the deliverable) is robust to this definition.
    switches = int(np.count_nonzero(np.diff(n) != 0))
    h2_del   = float(df['H2_delivered'].sum())
    h2_unmet = float(df['H2_unmet'].sum())
    pem_proj = project_pem_stack_replacements(df, stack_capex=1.0, replacement_frac=1.0)
    top = float(N_PAR_MAX)

    # Economic metrics for the same frozen policy. calculate_lcoh() already contains the
    # controller-specific SoH->lifetime->discrete-REPEX logic, so reuse it rather than
    # duplicating the cost model inside the reward study. Suppress its verbose table here.
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
        # Daytime-only mean, matching calculate_lcoh / the results tables / the
        # cross-location figures. Averaging over ALL steps includes night zeros.
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
    """Train across seeds, evaluate each frozen policy, print & save mean +/- std."""
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
    switching<->performance Pareto front. Trains SEEDS agents per lambda and
    reports the median with std error bars, so the noisy switching axis is
    de-biased against single-seed variance. If zero_underuse=True, the under-use
    penalty W_UNDERUSE is held at 0 for the whole sweep to isolate LAMBDA_SWITCH's
    own effect on switching (ablation)."""
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
    """Robustness of the DRL result to the reward SHAPING weights.

    Trains one agent per entry in `corners` and reports, for each swept weight, the SPAN
    of every headline metric across that weight's levels. The span is the deliverable: a
    conclusion that holds across a plausible range of a weight is attributable to the
    controller, while one that appears at a single value is attributable to the weight.

    The base objective (W_H2 / W_STH / W_MPPT) is NOT swept -- it is fixed by
    Almpantis2026 -- so every difference reported here comes from the shaping terms.

    Each corner is (lambda_sw, w_u, w_coldstart, w_tcycle, w_chiller, w_warmops, label).
    """
    import matplotlib.pyplot as plt
    global LAMBDA_SWITCH, W_UNDERUSE, W_COLDSTART, W_TCYCLE, W_CHILLER, W_WARMOPS, RL_TIMESTEPS

    seeds   = _as_seeds(REWARD_SEEDS if seeds is None else seeds)
    corners = REWARD_CORNERS if corners is None else corners
    budget  = timesteps or REWARD_STUDY_TIMESTEPS or RL_TIMESTEPS

    def _corner(c):
        # Backward-compatible with old 6-field corner lists: inherit current W_WARMOPS.
        if len(c) == 7:
            return c
        if len(c) == 6:
            lam, wu, wcs, wtc, wch, label = c
            return (lam, wu, wcs, wtc, wch, W_WARMOPS, label)
        raise ValueError("Reward corner must have 6 (legacy) or 7 fields; got %r" % (c,))

    corners = [_corner(c) for c in corners]

    # Every global this study perturbs is restored in the finally block. RL_TIMESTEPS is
    # included because _seed_train_one() reads it, and leaving the sweep's shorter budget
    # in place would silently under-train everything that runs afterwards.
    saved = (LAMBDA_SWITCH, W_UNDERUSE, W_COLDSTART, W_TCYCLE, W_CHILLER, W_WARMOPS, RL_TIMESTEPS)
    # RL_CONFIG is mutated in place (it is read by _seed_train_one at call time), so the
    # original entries are captured here and put back in the finally block. Only the keys
    # actually present in REWARD_STUDY_PPO are touched.
    saved_ppo = {k: RL_CONFIG[k] for k in REWARD_STUDY_PPO if k in RL_CONFIG}
    rows  = []
    # Per-step trajectories, kept so that any THRESHOLD-based metric can be recomputed
    # afterwards without retraining.
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
                # Dump after EVERY training. The default study is 18 trainings and is normally
                # left unattended; writing only at the end means one disconnect at run 7
                # throws away 30 minutes of compute. Completed rows then survive on disk and
                # the remaining configurations can be re-run on their own.
                Path(plots_dir).mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(
                    Path(plots_dir) / ("reward_weights_%s_raw.csv" % period_id), index=False)
                print("      [saved %d/%d runs -> reward_weights_%s_raw.csv]"
                      % (len(rows), n_train, period_id))
    finally:
        (LAMBDA_SWITCH, W_UNDERUSE, W_COLDSTART,
         W_TCYCLE, W_CHILLER, W_WARMOPS, RL_TIMESTEPS) = saved
        RL_CONFIG.update(saved_ppo)
        if rows:      # keep whatever finished, even if the sweep was interrupted
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

    # switches and cold starts lead: they are the two claims this study exists to test.
    # mean_restart_T is the threshold-free check on the cold-start count.
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
    # One-factor-at-a-time spans around `ref`; all base MOO weights remain unchanged.
    SPANS = [("lambda_sw", ["lam12", "lam13", "lam14", "lam15"], "LAMBDA_SWITCH")]
    print("\n" + "=" * 100)
    print("ROBUSTNESS SPANS - the reportable result")
    print("=" * 100)
    print("  For each weight, the min-max range of every metric over its tested levels.")
    print("  A NARROW span means the conclusion is a property of the controller.")
    print("  A WIDE span means the conclusion is a property of the weight - say so.\n")
    for _, labels_, name in SPANS:
        sub = med[med["label"].isin(labels_)]
        # A requested level that never ran does not raise -- it just shrinks the span. Say so,
        # because a silently truncated span reads as a robustness result rather than a gap.
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

    # Flat per-configuration summary with median/min/max/std for reviewer tables.
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

    # Figure A: the six quantities that decide the final lambda. Show every seed, plus median.
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

    # Figure B: compact median bars with min-max whiskers for the broader metric set.
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
    """Load a previously trained RL policy (rl_model_<period_id>.zip) for reuse
    without retraining. Returns the PPO model, or None if the file is absent so
    the caller can fall back to train_rl(). Only .predict() is needed downstream
    (evaluate_rl / plot_computational_cost build their own eval env)."""
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
            # Read the INPUT series, not env.G / env.T_C. step() advances those to the NEXT
            # hour before returning, so recording them here stored the weather of step+1
            # alongside the result of step, shifting the whole column by one. It showed up as
            # 391 hours per year where the DRL produced hydrogen at a recorded irradiance
            # below the 80 W/m2 cut-in, and it made every irradiance-keyed analysis of the RL
            # results (switching-by-daylight, XAI, any G-conditioned plot) off by one hour.
            # run_baselines never had this: it records the loop's own g.
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
          f"rccd={100*_dd['rccd']:.3f}  cold-restart={100*_dd['cold']:.3f}  "
          f"(load {100*_dd['load']/_dsum:.0f}% / rccd {100*_dd['rccd']/_dsum:.0f}% "
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
    """PV array - PEM I-V characterisation, one figure per site.

    Panel 1  five irradiance levels at the site's full array (top switch position).
    Panel 2  the site's own seven MOSFET switch positions at G = 600 W/m2, an
             irradiance that straddles the coupled/over-supplied boundary at all
             three sites (at G <= 400 every position couples, at G = 1000 none do).

    String counts are derived per site from site_string_config(), never hard-coded.
    The earlier version fixed them at n_par = 400 and [250, 300, 350, 400], which
    correspond to no site in the current study.

    A configuration whose array current exceeds the 3 A/cm2 datasheet limit has no
    valid operating point: the curves do not cross and the stack cannot be operated.
    Those cases are drawn as an open marker on the PEM curve maximum and labelled
    "no operating point", which is what the controller must avoid by trimming strings.
    """
    plots_dir = Path(plots_dir)
    T_pem_K   = 60.0 + 273.15

    pem_I = np.linspace(0.0, float(I_PEM(i_cell, 1)), 400)
    pem_V = np.asarray(V_PEM(T_pem_K, p_cat, p_an, delta_mem, A_cell_pem,
                             a_an, a_cat, pem_I, i_0_an, i_0_cat,
                             T_ref_pem, i_lim, N_SER_PEM), float)
    i_max_pem = float(pem_I.max())
    # The datasheet nominal current IS the 1 MW rating (3 A/cm^2), so the rated line and the
    # max-current line coincide and only one horizontal reference is drawn.
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
            # Upper limit follows the 66-cell module: the 5-module array reaches open circuit
            # near 300 V and the stack tops out at 270 V, so 415 V left roughly a third of the
            # axis empty and squashed the region where the curves actually cross.
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
# Canonical order + colours/hatches, keyed by the SAME labels calculate_lcoh emits, so
# all_results, all_lcoh and every figure stay aligned. Order follows the ablation: no rule,
# one rule, one rule, both rules, learned.
CTRL_ORDER  = ['MOO', 'MOO+Hyst', 'MOO+PreWarm', 'MOO+Hyst+PreWarm',
               'MOO+Pen', 'MOO+RF', 'RL Agent']
CTRL_COLORS = {'MOO': '#4477AA', 'MOO+Hyst': '#AA3377', 'MOO+PreWarm': '#CCBB44',
               'MOO+Hyst+PreWarm': '#66CCEE', 'MOO+Pen': '#EE7733',
               'MOO+RF': '#EE6677', 'RL Agent': '#228833'}
CTRL_HATCH  = {'MOO': '', 'MOO+Hyst': '..', 'MOO+PreWarm': '--',
               'MOO+Hyst+PreWarm': 'xx', 'MOO+Pen': '\\',
               'MOO+RF': '//', 'RL Agent': 'oo'}


def _ctrl_dfs(df_moo, df_rf, df_rl, df_hyst=None, df_pen=None, df_pw=None, df_hpw=None):
    """Ordered {label: df} for the controllers actually present.

    The four baselines form a 2x2 over the two actuators the DRL commands: 'MOO+Hyst' adds a
    hard deadband on the string count, 'MOO+PreWarm' a forecast rule on the warm-standby
    set-point, 'MOO+Hyst+PreWarm' both. Order runs from the unfiltered optimiser through
    increasing amounts of hand-designed structure to the learned policy, so summary tables
    read left to right as increasing sophistication and the DRL column is always last.
    'MOO+Pen' (soft -lambda|delta n| charge) is retired from the default set; it shares the
    deadband's cell and is kept only for the supplement.
    """
    d = {'MOO': df_moo}
    if df_hyst is not None:
        d['MOO+Hyst'] = df_hyst
    if df_pw is not None:
        d['MOO+PreWarm'] = df_pw
    if df_hpw is not None:
        d['MOO+Hyst+PreWarm'] = df_hpw
    if df_pen is not None:
        d['MOO+Pen'] = df_pen
    if df_rf is not None:
        d['MOO+RF'] = df_rf
    d['RL Agent'] = df_rl
    return d


def plot_energy_stacks(df_moo, df_rf, df_rl, period_id, plots_dir, df_hyst=None, df_pen=None,
                       df_pw=None, df_hpw=None):
    """
    Daily stacked-area energy breakdown, one panel per controller (4-way when the
    hysteresis baseline is present).
    Figure A — consumers : PEM electrical input, BoP, Compression, Cooling,
                           Battery charge, Grid export  (kWh/day).
    Figure B — sources   : Solar fraction vs Grid import (% of daily load).
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
        # Electrical input ≈ H2_prod × LHV / average_efficiency
        # LHV is in Wh/kg; /1000 → kWh/kg. The divisor is the LHV-basis system efficiency,
        # 0.80 -> 0.65: with production now set by Faraday's law the stack runs at
        # 1.254/V_cell ≈ 0.63-0.68 across the operating band, not 0.80 (which was the old
        # HHV-basis curve read as if it were LHV).
        arr = np.array(df['H2_prod'].values[:n_days * STEPS_PER_DAY], dtype=float)
        return (arr * LHV / (0.65 * 1000.0)).reshape(n_days, STEPS_PER_DAY).sum(axis=1)

    # ── Figure A: energy consumers ────────────────────────────────────────
    CONS_KEYS   = ['BoP_W', 'Compression_W', 'Cooling_W', 'batt_charge_w']
    CONS_LABELS = ['BoP', 'Compression', 'Cooling', 'Battery charge']
    CONS_COLS   = ['#4878d0', '#6acc65', '#d65f5f', '#ee854a', '#956cb4', '#8c613c']
    # first colour is for PEM; 'Grid export' dropped (export disabled → always zero)

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
    # on-plot title removed — caption: "Daily Energy Consumer Breakdown -- <location>"
    plt.tight_layout()
    _save_fig(fig_a, plots_dir / f'comp_7_energy_consumers_{period_id}.png')

    # ── Figure B: energy sources (% solar vs % grid) ─────────────────────
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
    # on-plot title removed — caption: "Daily Energy Source — Solar vs Grid (<location>)"
    plt.tight_layout()
    _save_fig(fig_b, plots_dir / f'comp_8_energy_sources_{period_id}.png')
    print("  -> Energy stacked-area plots saved.")


def plot_3way_comparison(df_moo, df_rf, df_rl, period_id, plots_dir, df_hyst=None, df_pen=None,
                         df_pw=None, df_hpw=None, pv_scale=1.0):
    """Per-location comparison plots. N-way: every baseline the main loop produced is drawn,
    in the ablation order of _ctrl_dfs (no rule / deadband / pre-warm / both / learned)."""
    plots_dir = Path(plots_dir)
    # x-axis in calendar days so hourly and 15-min plots share a 0-365 axis.
    # Controller registry (label, df, colour, linestyle) — shared Tol palette with
    # the XAI and cross-location figures for a consistent look across the paper.
    ctrls = [('MOO', df_moo, '#4477AA', '-')]
    if df_hyst is not None:
        ctrls.append(('MOO+Hyst', df_hyst, '#AA3377', ':'))
    if df_pw is not None:
        ctrls.append(('MOO+PreWarm', df_pw, '#CCBB44', (0, (5, 2))))
    if df_hpw is not None:
        ctrls.append(('MOO+Hyst+PreWarm', df_hpw, '#66CCEE', (0, (5, 1, 1, 1))))
    if df_pen is not None:
        ctrls.append(('MOO+Pen', df_pen, '#EE7733', (0, (3, 1, 1, 1))))
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
            # linestyle MUST be a keyword: passed positionally matplotlib reads it as a
            # format string, which rejects the dash-tuple used by MOO+Pen.
            ax.plot(_t(df), y, linestyle=ls, color=color, lw=lw, label=lab, alpha=0.85)
        ax.set_ylabel(ylabel); ax.set_xlabel('Day of year')
        ax.legend(fontsize=10); ax.grid(alpha=0.3); plt.tight_layout()
        _save_fig(fig, plots_dir / f'{fname}_{period_id}.png')

    # 1. PV parallel strings (step plot) — auto-scaled y-axis, system-size agnostic. SIZING:
    # n_parallel is the real installed string count, so its range differs by site (Odeillo
    # 232-370, Prague 327-519, Lund 371-593 at 1.0x/1.4x/1.6x).
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
              'MOO+Hyst+PreWarm': 'MOO_Hyst_PreWarm', 'MOO+Pen': 'MOO_Pen',
              'MOO+RF': 'MOO_RF', 'RL Agent': 'RL'}
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for lbl, d in ctrls.items():
            d.to_excel(writer, sheet_name=_sheet.get(lbl, lbl)[:31], index=False)
    print(f"  -> Excel saved: {out_path}")


def print_summary(df_moo, df_rf, df_rl, period_id, df_hyst=None, df_pen=None,
                  df_pw=None, df_hpw=None):
    """Detailed per-location results table, N-way over the controllers present
    (4-way when the hysteresis MOO baseline was run)."""
    ctrls = _ctrl_dfs(df_moo, df_rf, df_rl, df_hyst, df_pen, df_pw, df_hpw)   # ordered {label: df}
    labels = list(ctrls.keys())

    def _safe_pct(grid_col, prod_col, df):
        return (df[grid_col].sum() / max(1e-9, df[prod_col].sum())) * 100

    # Mean solar-to-hydrogen efficiency over daylight hours (StH > 0), in %.
    def _mean_sth_pct(df):
        v = df['StH'][df['StH'] > 0]
        return float(v.mean() * 100.0) if len(v) else 0.0

    # Battery utilisation — how much of the installed pack is actually exercised.
    def _batt_used_kwh(df):
        return float(df['batt_soc'].max() - df['batt_soc'].min()) * BATT_CAPACITY_WH / 1000.0
    def _batt_cycles(df):
        thru = (df['batt_charge_w'].sum() + df['batt_discharge_w'].sum()) * DT_HOURS
        return thru / 2.0 / BATT_CAPACITY_WH

    # There is no year-end liquidation to reconcile: H2_unmet already carries the cyclic
    # deficit, charged once in simulate_ems_routing, so the contract closes directly.
    def _h2_metrics(df):
        # H2_unmet already carries the cyclic deficit (charged once, in simulate_ems_routing),
        # and there is no liquidation draw to reconcile, so the contract closes directly.
        demand    = df['H2_daily_demand'].sum()
        eff_unmet = df['H2_unmet'].sum()
        deficit   = max(0.0, INITIAL_H2_STORAGE - float(df['storage'].iloc[-1]))
        return demand, demand - eff_unmet, eff_unmet, deficit

    # (row label, function of df) — evaluated for every controller
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
    """Number of production starts made from a COLD stack.

    A cold start is a step where the electrolyzer begins producing after an idle spell
    while the stack was still below `threshold_C`. Restarting cold means climbing the
    high-overpotential end of the polarization curve and imposing a large thermal
    excursion on the membrane and seals, so a controller that anticipates sunrise and
    pre-warms the stack should show fewer of these than one that only reacts.

    Computed POST-HOC from the recorded trajectory, so it applies identically to all four
    controllers with no change to the simulation loop and no risk of the metric itself
    perturbing a result.
    """
    thr = COLDSTART_THRESHOLD_C if threshold_C is None else threshold_C
    if 'T_pem' not in df.columns or 'H2_prod' not in df.columns:
        return 0
    producing = (df['H2_prod'].values > 1e-9)
    was_idle  = np.r_[True, ~producing[:-1]]          # idle on the previous step
    T_prev_C  = np.r_[np.nan, df['T_pem'].values[:-1]] - 273.15
    cold      = np.r_[False, T_prev_C[1:] < thr]
    return int((producing & was_idle & cold).sum())


def restart_temperatures(df):
    """Stack temperature (deg C) at every production restart.

    The continuous companion to count_cold_starts(). A threshold count is convenient but
    discontinuous, so its value depends on where the threshold sits relative to the
    keep-warm ladder; the mean restart temperature has no threshold and therefore no such
    artefact. Report both: the count is the headline, this is the check on it.
    """
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
    hysteresis MOO, MOO+RF and DRL alike)."""
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
    # Self-sufficiency = renewable share of the total plant electrical load
    # (PEM + BoP + compression + thermal). grid_bought is the only non-renewable input.
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
    """Compact side-by-side headline table for an ordered dict {label: df}.
    Promotes the hysteresis MOO into the main comparison (4- or 5-way)."""
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
    """Diagnose WHY controllers differ in H2 without retraining.

    The RL reward's MPPT term is computed only on the strings connected to the PEM. A policy
    can therefore show a small `mppt_loss` while still allocating fewer of the INSTALLED PV
    strings to the PEM. This audit reports the string allocation and the energy that remains
    on the unused-string/battery branch, which is the quantity needed to interpret a low-MPPT
    / low-H2 result correctly.
    """
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

    This is a diagnostic only. It assumes the current module-level reward constants are the
    same values used to train the supplied policy. Large cumulative penalties identify which
    shaping terms deserve an ablation; they do not by themselves prove causality.
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
    # Evaluation CSV does not store the local `valid` boolean; positive PEM power is the
    # observable proxy for a valid PV/PEM operating point. This mirrors the corrected reward
    # closely and, crucially, excludes night/no-intersection steps.
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
    """
    Levelized Cost of Hydrogen (EUR/kg) for each controller.

    PEM stack CAPEX is controller-specific: the simulated end-of-period SoH is projected
    to LCOH_PEM_EOL_SOH, discrete stack replacements are scheduled over the project,
    discounted to NPV, and annualised over LCOH_PROJECT_LIFE. This makes the economic
    benefit of thermal / switching strategies visible in LCOH rather than leaving every
    controller on the same hard-coded PEM lifetime.

    CAPEX components: PV, PEM stack, Battery, H2 tank, Compressor, BoP, Controller.
    OPEX components:  O&M (fraction of CAPEX) + grid electricity cost.

    Uses global LCOH_* parameters defined in PHYSICAL & ECONOMIC CONSTANTS.
    pv_peak_kw : installed PV peak capacity in kW (from print_system_sizing).
    """
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
    # Initial stack is annualised over the PROJECT. Controller-specific REPEX is added below.
    ac_pem                         = capex_pem * crf(LCOH_WACC, LCOH_PROJECT_LIFE)
    om_pem                         = capex_pem * LCOH_PEM_OM
    ac_batt,      om_batt      = ann_capex_om(capex_batt,      LCOH_WACC, LCOH_BATT_LIFE,      LCOH_BATT_OM)
    ac_batt_conv, om_batt_conv = ann_capex_om(capex_batt_conv, LCOH_WACC, LCOH_BATT_CONV_LIFE, LCOH_BATT_CONV_OM)
    ac_grid_inv,  om_grid_inv  = ann_capex_om(capex_grid_inv,  LCOH_WACC, LCOH_GRID_INV_LIFE,  LCOH_GRID_INV_OM)
    ac_tank,      om_tank      = ann_capex_om(capex_tank,      LCOH_WACC, LCOH_TANK_LIFE,      LCOH_TANK_OM)
    ac_comp,      om_comp      = ann_capex_om(capex_comp,      LCOH_WACC, LCOH_COMP_LIFE,      LCOH_COMP_OM)
    # BoP is not a stack replacement item: give civil/piping its own fixed service life; only the PEM stack is SoH-driven.
    ac_bop,       om_bop       = ann_capex_om(capex_bop,       LCOH_WACC, LCOH_BOP_LIFE,       LCOH_PEM_OM)
    ac_ctrl = capex_ctrl * crf(LCOH_WACC, LCOH_PROJECT_LIFE); om_ctrl = 0.0

    # Base production chain contains ONE initial stack. REPEX differs by controller and is
    # therefore not included in this shared component table.
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
        _ctrl_list.append(('MOO+Pen', df_pen))
    if df_rf is not None:
        _ctrl_list.append(('MOO+RF', df_rf))
    _ctrl_list.append(('RL Agent', df_rl))

    for label, df in _ctrl_list:
        h2_prod = df['H2_prod'].sum()
        # Annual capacity factor should remain annualised for partial-year / sub-hourly datasets.
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
        # fixed_prod_ann_base already contains the annualised INITIAL stack. Replace that
        # initial-only CAPEX charge with initial + discounted controller-specific REPEX.
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
    in place, using the MOO baseline only (fast, no RL training). For each ratio it scales
    the array photocurrent, re-runs the plant, and reports demand fulfilment, curtailment,
    NAMEPLATE LCOH and operating profit, so the size can be chosen from the curve rather
    than assumed. Uses nameplate CAPEX (pv_kw = ratio x PEM_kW), not the bifacial peak."""
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
    # "optimal" = smallest ratio whose demand fulfilment is within 0.5 pp of the best achieved
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
    """Sizing study (MOO baseline, fast) sweeping PV oversizing (rows) against a second
    variable (columns): sweep='battery' -> battery size (tank fixed); sweep='tank' ->
    H2-tank size (battery fixed). Headline metric is SELF-SUFFICIENCY % (renewable share
    of the plant electrical load) — the grid-independence goal. Also maps demand
    fulfilment, grid import, and a grid-savings net value (firm-H2 + O2 + grid econ -
    annualised PV/battery/tank CAPEX; secondary-H2 sales excluded). The economic optimum
    (max net value) is starred. Mutates the relevant globals transiently (restored).
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
            # site_string_config rounds to an integer string count, so R x PEM is not
            # exactly what gets built. Cost the array that does.
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
                # Each (PV, tank) cell has its own cyclic opening level. Solving it per cell
                # is what keeps the comparison fair: a fixed fraction of capacity would hand
                # every larger tank a larger loan and penalise it for not repaying.
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

                # Capture final storage for cyclic sustainability check
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
                # Incremental save: the full sweep takes ~15 min, so dump after EVERY run.
                # A dropped Colab session then still leaves the completed rows on disk.
                pd.DataFrame(rows).to_csv(out_dir / f'sizing_pv_{fname}_{loc_label}.csv', index=False)
    finally:
        BATT_CAPACITY_WH, BATT_MAX_P_W, storage_capacity, INITIAL_H2_STORAGE = orig
        N_PAR_BASE, N_PAR_STEP, N_PAR_MAX = _ladder0
        if rows:      # keep whatever finished, even if the sweep was interrupted
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
    # on-plot title removed — caption: "PV x <sweep> sizing, <location>, <demand> kg/day"
    fig.tight_layout(); fig.savefig(out_dir / f'sizing_pv_{fname}_{loc_label}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: sizing_pv_{fname}_{loc_label}.png")
    return dfres


def verify_designs(reco, series, dem_target, verbose=True):
    """Re-simulate each recommended design and report what it ACTUALLY delivers.

    `series` is {location_label: (irr, temp, price)}. Each design's tank is first rounded up
    to TANK_ROUND_KG, then grown in TANK_VERIFY_STEP increments until the measured demand
    fulfilment clears `dem_target`. Adds measured 'demand', 'ss', 'grid' and the cyclic
    opening stock 's0' to every entry, and marks it 'verified'.
    """
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
    REVISED OPTIMIZATION GOAL: Searches for the design with the SMALLEST TANK within
    the acceptable cost tolerance (`tol`). Cyclic sustainability enforced.
    NEW: `batt_override` allows forcing specific battery sizes for specific locations
    (e.g., larger batteries for high-irradiance sites like Odeillo).
    `tank_override` pins a specific swept tank size for a location instead of letting
    the cost tie-break pick the leanest one that clears dem_target -- e.g. Odeillo's
    deployed 1400 kg store is larger than the 1200 kg the pure-cost search returns,
    because the DRL agent needs the extra headroom the MOO-only search doesn't price in.
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

    # Snap PV overrides
    for _lc, _pv in list(pv_override.items()):
        if not any(np.isclose(p, _pv) for p in pvs):
            _near = min(pvs, key=lambda p: abs(p - _pv))
            print(f"  NOTE: pv_override['{_lc}'] = {_pv:g}x is not in the swept ladder "
                  f"{[round(float(p), 3) for p in pvs]} -> snapping to {_near:g}x.")
            pv_override[_lc] = float(_near)

    # Available batteries
    all_batts = sorted(df[df['sweep'] == 'battery']['batt_kwh'].unique())
    batts = [b for b in all_batts if b >= batt_min]

    # Snap Battery overrides
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

            # NO hard cyclic filter here. The cyclic deficit is ALREADY charged into
            # 'demand': simulate_ems_routing adds (initial - final) to H2_unmet on the last
            # step, so a design that fails to repay its opening stock is penalised in the
            # demand column itself. Filtering AGAIN on final >= initial double-counts it, and
            # it does something much worse: a design only fully repays when its deficit is
            # zero, i.e. only at ~100 % demand, so the filter discarded every row below 100 %
            # and forced the recommendation there no matter what dem_target said.
            gt = gt_full.sort_values('tank_kg')

            if not len(gt) or gt['demand'].max() < dem_target:
                continue

            if lc in tank_override:
                # The override need not be a swept grid point -- verify_designs() below
                # re-simulates at this exact size regardless, so it is used directly rather
                # than snapped to the nearest swept value (which silently discarded any
                # override outside the coarse sizing ladder, e.g. Odeillo's 1400 kg against
                # a {1000,3000,5000,...} sweep).
                tank = float(tank_override[lc])
                dem = (float(np.interp(tank, gt['tank_kg'], gt['demand']))
                       if tank <= gt['tank_kg'].max() else float(gt['demand'].iloc[-1]))
            elif gt['demand'].iloc[0] >= dem_target:
                tank, dem = float(gt['tank_kg'].iloc[0]), float(gt['demand'].iloc[0])
            else:
                tank, dem = float(np.interp(dem_target, gt['demand'], gt['tank_kg'])), dem_target

            # Filter battery sizes if an override is provided for this location
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
        # Among designs within `tol` of the cheapest, take the LEANEST ARRAY.
        #
        # At 700 EUR/kWp of PV against 500 EUR/kg of tank the two are near-perfect economic
        # substitutes: at Lund every ratio from 2.25x to 3.0x reaches the target for
        # 359.6-366.0 kEUR/yr, a spread of 1.8 %, which is well inside the model's own
        # uncertainty. Ranking on cost therefore picks the edge of whatever range happened to
        # be swept (3.0x here), and ranking on tank picks the other edge for the same reason.
        # Neither is a real optimum. Inside a flat band the defensible tie-break is the
        # smaller array: less land, less interconnection and less curtailment for the same
        # delivered hydrogen at the same cost.
        reco.append(min(near, key=lambda c: (c['pv'], c['capex'], c['tank'])))

    # The pick is interpolated between swept tank sizes, so re-simulate it at a buildable
    # size before it is reported or plotted. Everything downstream then shows measured
    # numbers rather than the target the interpolation was solved for.
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
    # The full specification goes in a TABLE under the cube, not in callouts on it. Sites are
    # selected against the same demand target, so their stars land at nearly the same
    # (demand, self-sufficiency) and any multi-line callout drawn at the marker collides with
    # its neighbour's — staggering only moved the collision around. A reserved axes cannot
    # overlap the data by construction, and it reads as a figure caption should.
    # hspace must be generous: the 3D x-axis label is drawn well BELOW the axes box (it
    # follows the rotated floor edge), so a table packed tight underneath clips it.
    fig = plt.figure(figsize=(11.5, 10.4))
    _gs = fig.add_gridspec(2, 1, height_ratios=[6.0, 1.0], hspace=0.16)
    ax  = fig.add_subplot(_gs[0], projection='3d')
    axt = fig.add_subplot(_gs[1]); axt.axis('off')

    for lc in locs:
        d = df[df['loc'] == lc]
        # depthshade=False keeps each site's colour constant with distance; with it on, the
        # far half of every cloud washes out to grey and the three sites stop being separable.
        ax.scatter(d['capex_ann_k'], d['demand'], d['self_suff'], s=24, alpha=0.30,
                   color=COL[lc], edgecolor='none', label=lc, depthshade=False)

    _ordered = sorted(reco, key=lambda c: (c['demand'], c['ss']))
    for r in _ordered:                       # short tag only — the numbers live in the table
        ax.text(r['capex'], r['demand'], r['ss'] + 0.7, f" {r['loc']}",
                fontsize=10.5, fontweight='bold', color=COL[r['loc']],
                ha='left', va='bottom')
    # Stars LAST: mplot3d paints in call order and ignores zorder between artists, so a label
    # drawn afterwards would sit on top of a neighbouring site's marker.
    for r in _ordered:
        ax.scatter([r['capex']], [r['demand']], [r['ss']], s=430, marker='*',
                   color=COL[r['loc']], edgecolor='k', lw=0.9, depthshade=False)

    ax.set_xlabel('Annualised CAPEX (kEUR/yr)', labelpad=10)
    ax.set_ylabel('Demand fulfilment (%)', labelpad=10)
    ax.set_zlabel('Self-sufficiency (%)', labelpad=6)
    # A 3D axes reserves a square bounding box, so most of the panel was empty and the cube
    # sat small in the middle. zoom fills it. Two traps, both verified by rendering: (1)
    # fig.tight_layout() on a 3D axes clips the z label away entirely — it is why the label
    # was missing before, so it is deliberately NOT called here; (2) zoom above ~1.1 pushes
    # the label past the tight bbox and clips it again. 1.08 + pad_inches is the safe pair.
    ax.set_box_aspect((1, 1, 0.85), zoom=1.08)
    ax.grid(True, alpha=0.25)
    # on-plot title removed — caption: "Per-site optimal sizing of the <plant>. Stars mark the
    # recommended design (>= <dem>% demand, >= <ss>% self-sufficiency, cyclic H2 inventory)."
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
    # Provenance as a footnote, not a title: whether the demand column was simulated or
    # merely interpolated changes how the table may be cited, so it travels with the figure.
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
    # on-plot title removed — caption: "Seasonal H2 storage drives demand fulfilment;
    # stars mark the recommended per-site tank."
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
    # on-plot title removed — caption: "PV oversizing raises grid self-sufficiency; shaded
    # band = 86-92 %; stars mark the chosen PV."
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
    # on-plot title removed — caption: "Battery role; stars mark the chosen battery per site."
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
    """
    Sweep battery size (kWh) using only the MOO baseline (fast, no RL training).
    For each size: run simulation, compute LCOH and LCSH, then plot.

    Answers: "Is our 400 kWh battery the right size, and why doesn't a bigger
    battery pay off in H2 revenue alone?"

    Saves:  Battery_Sensitivity_<loc_label>.pdf / .png
    Returns: list of result dicts (one per battery size).
    """
    import matplotlib.pyplot as plt

    global BATT_CAPACITY_WH, BATT_MAX_P_W

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curr_kwh = BATT_CAPACITY_WH / 1000.0
    c_rate   = BATT_MAX_P_W / BATT_CAPACITY_WH   # kept constant across sweep

    # Sweep as multiples of the current battery — automatically re-centres when
    # the system is scaled up, keeping the sensitivity analysis always proportional.
    _batt_mults    = [0.1, 0.25, 0.5, 1.0, 2.0]
    batt_sizes_kwh = [max(10.0, round(curr_kwh * m / 10.0) * 10.0) for m in _batt_mults]
    # Remove duplicates while preserving order (can happen for very small curr_kwh)
    seen = set(); batt_sizes_kwh_u = []
    for v in batt_sizes_kwh:
        if v not in seen: seen.add(v); batt_sizes_kwh_u.append(v)
    batt_sizes_kwh = batt_sizes_kwh_u
    _batt_mults    = [v / curr_kwh for v in batt_sizes_kwh]   # recompute exact multiples
    batt_labels    = [f'{m:.2g}× ({s:.0f} kWh)' for m, s in zip(_batt_mults, batt_sizes_kwh)]

    # ── Pre-compute the fixed (non-battery) annualized costs ──────────────────
    def _crf(r, n):
        return r * (1 + r)**n / ((1 + r)**n - 1)

    pem_kw        = PEM_capacity / 1000.0
    h2_peak_kgh   = H2_PEAK_KGH
    batt_power_kw = BATT_MAX_P_W / 1000.0   # original converter size — kept fixed

    def _ann(capex, r, life, om):
        return capex * _crf(r, life) + capex * om

    # Fixed production cost EXCLUDING PEM stack CAPEX; the initial stack + REPEX
    # is controller/scenario-specific and added after each simulation from its SoH trajectory.
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

    # ── Sweep ─────────────────────────────────────────────────────────────────
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

    # ── Figure ────────────────────────────────────────────────────────────────
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
    # on-plot title removed — caption in FIGURE CAPTIONS (Battery_Sensitivity)

    bx_idx = list(range(len(batt_sizes_kwh)))   # integer positions for the x-axis

    def _panel(ax, ydata, ylabel, title, color, yref=None, secondary=None, mark_opt=True):
        ax.plot(bx_idx, ydata, 'o-', color=color, lw=1.8, ms=5, zorder=3)
        curr_idx = min(range(len(batt_sizes_kwh)), key=lambda i: abs(batt_sizes_kwh[i] - curr_kwh))
        ax.axvline(curr_idx, color='black', lw=1.1, ls='--',
                   label=f'Current ({curr_kwh:.0f} kWh = 1×)')
        if mark_opt:   # only meaningful for cost panels (lowest LCOH); NOT for utilisation
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
    """
    Run one full year through trained RL agent (eval mode).
    Returns DataFrame with hourly data for sensitivity metrics.
    """
    env = PVPEMEnv(irr_data, temp_data_C, price_data_mwh, eval_mode=True, pv_scale=pv_scale)
    obs, _ = env.reset()
    records = []

    for step in range(len(irr_data)):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, done, info = env.step(action)
        # Carry the step index and this step's OWN irradiance. The info dict has neither, so
        # the reward study's trajectory dump silently lost both columns and any diagnostic
        # that needs to separate daylight from night could not be run on it afterwards.
        info = dict(info, step=step, irradiance=float(np.clip(irr_data[step], 0, 1200)))
        records.append(info)
        if done:
            break

    return pd.DataFrame(records)


def tank_sensitivity_analysis_rl(model, irr, temp, price, pv_peak_kw, loc_label, out_dir):
    """
    Tank sensitivity analysis using trained RL agent (much faster than MOO).

    Args:
        model: trained PPO model
        irr, temp, price: time series data
        pv_peak_kw: installed PV capacity
        loc_label: location name
        out_dir: output directory

    Returns:
        list of result dicts with sensitivity metrics
    """
    import matplotlib.pyplot as plt

    global storage_capacity, LCOH_TANK_KG_INSTALLED

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_demand_kg = daily_H2_target
    _auto_days      = [7, 30, 60, 90, 120]
    tank_sizes_kg   = [max(1.0, round(daily_demand_kg * d)) for d in _auto_days]
    # Remove duplicates
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

    # ── Compact figure: demand fulfillment + cost trade-off ────────────────────
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
    # on-plot title removed — caption in FIGURE CAPTIONS (Tank_Sensitivity)

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
    """
    Tornado plots: show % change in LCOH & H2 fulfillment when each parameter ±20%.

    Parameters swept:
      • PV capacity (±20%)
      • PEM capacity (±20%)
      • Battery capacity (±20%)
      • H2 price (±50% range: EUR 2.5 to 7.5)
      • Grid electricity price (±30%)
      • Tank capacity (±30%)

    Args:
        irr, temp, price: time series
        pv_peak_kw: baseline PV capacity
        loc_label: location name
        out_dir: output directory
        model: (optional) trained RL model. If None, uses MOO baseline.
    """
    import matplotlib.pyplot as plt

    global PEM_capacity, BATT_CAPACITY_WH, BATT_MAX_P_W, storage_capacity, H2_PRICE_EUR_KG, LCOH_WACC
    global irradiance_scaling_factor  # will define

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy arrays for arithmetic operations
    irr = np.array(irr)
    temp = np.array(temp)
    price = np.array(price)

    # Baseline run (current configuration)
    print(f"\n{'='*90}")
    print(f"  TORNADO SENSITIVITY: {loc_label}")
    print(f"{'='*90}")

    # ---- Proper annualised production-chain fixed cost (mirrors calculate_lcoh) ----
    # This is recomputed PER SCENARIO from the swept parameters. The previous version hard-coded
    # it at 29,895 EUR/yr, so WACC / Battery / PEM / PV sweeps — which act almost entirely
    # through annualised CAPEX — produced no LCOH change.
    def _crf(r, n):
        return r * (1 + r) ** n / ((1 + r) ** n - 1)

    def _prod_ann_fixed(pv_kw, wacc, pem_w, batt_wh, batt_w):
        def ann(capex, life, om):
            return capex * _crf(wacc, life) + capex * om
        pem_kw = pem_w / 1000.0
        pem_stack_capex = pem_kw * LCOH_PEM_EUR_KW
        # Stack CAPEX is omitted here and injected by _lcoh_of() from the scenario's SoH.
        return (ann(pv_kw           * LCOH_PV_EUR_KWP,          LCOH_PV_LIFE,        LCOH_PV_OM)
              + pem_stack_capex * LCOH_PEM_OM
              + ann(batt_wh / 1000.0 * LCOH_BATT_EUR_KWH,       LCOH_BATT_LIFE,      LCOH_BATT_OM)
              + ann(batt_w  / 1000.0 * LCOH_BATT_CONV_EUR_KW,   LCOH_BATT_CONV_LIFE, LCOH_BATT_CONV_OM)
              + ann(LCOH_GRID_INV_KW * LCOH_GRID_INV_EUR_KW,    LCOH_GRID_INV_LIFE,  LCOH_GRID_INV_OM)
              + ann(pem_kw * LCOH_PEM_EUR_KW * LCOH_BOP_FRAC,   LCOH_BOP_LIFE,       LCOH_PEM_OM)
              + LCOH_CTRL_EUR * _crf(wacc, LCOH_PROJECT_LIFE))

    # Save originals
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
        # Runs the sim for one parameter perturbation and returns its LCOH, mutating
        # the relevant global only transiently. Cost-relevant scenario values are passed
        # explicitly to _lcoh_of so the annualised CAPEX reflects the perturbed parameter.
        global PEM_capacity, BATT_CAPACITY_WH, BATT_MAX_P_W, storage_capacity
        global N_PAR_BASE, N_PAR_STEP, N_PAR_MAX
        pv_kw, wacc, pem_w = pv_peak_kw, orig_wacc, orig_pem
        batt_wh, batt_w    = orig_batt_wh, orig_batt_w
        if param_type == 'PV_scale':
            # More panels -> more CAPEX AND more yield. Resize the installed string
            # count (exact), then restore the site's own array afterwards.
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

    # Plot tornado
    fig, ax = plt.subplots(figsize=(10, 6))
    param_names = list(tornado_data.keys())
    lows = [tornado_data[p]['low'] for p in param_names]
    highs = [tornado_data[p]['high'] for p in param_names]

    # Sort by impact (width of bar)
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
    # on-plot title removed — caption: "Tornado Plot: LCOH Sensitivity -- <location>"
    ax.grid(alpha=0.3, axis='x')
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    fig.savefig(out_dir / f'Tornado_LCOH_{loc_label}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✓ Tornado plot saved: Tornado_LCOH_{loc_label}.png\n")

    return tornado_data


def tank_sensitivity_analysis(irr, temp, price, pv_peak_kw, loc_label, out_dir):
    """
    Sweep H2 storage tank size (kg) using only the MOO baseline (fast, no RL training).

    Two globals are updated together for each size:
      storage_capacity       — physical tank ceiling in the simulation
      LCOH_TANK_KG_INSTALLED — tank size used in LCOH cost calculation

    Key metrics tracked:
      H2 production   — nearly constant (PEM output set by PV, not tank)
      H2 delivered    — rises with tank (more buffer for low-irradiance periods)
      H2 overflow     — falls with tank (estimated from mass balance)
      Demand fulfil % — rises then plateaus once seasonal buffer is sufficient
      LCOH (production) — FLAT: the tank is not in the production chain, so the
                          production cost per kg is independent of tank size
      LCSH (prod+storage) — rises monotonically (tank dominates storage-chain CAPEX)

    Saves:  Tank_Sensitivity_<loc_label>.pdf / .png
    Returns: list of result dicts.
    """
    import matplotlib.pyplot as plt

    global storage_capacity, LCOH_TANK_KG_INSTALLED

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sweep as autonomy-days × daily demand — automatically re-centres when
    # the system is scaled up, keeping the sweep always proportional to sizing.
    daily_demand_kg = daily_H2_target
    _auto_days      = [7, 14, 30, 60, 90, 120]
    tank_sizes_kg   = [max(1.0, round(daily_demand_kg * d)) for d in _auto_days]
    # Remove duplicates (can arise when daily_demand is very small)
    seen_t = set(); tank_sizes_u = []
    for v in tank_sizes_kg:
        if v not in seen_t: seen_t.add(v); tank_sizes_u.append(v)
    tank_sizes_kg = tank_sizes_u
    _auto_days    = [v / max(daily_demand_kg, 1e-6) for v in tank_sizes_kg]   # recompute exact days
    tank_labels   = [f'{d:.0f}d ({s:.0f} kg)' for d, s in zip(_auto_days, tank_sizes_kg)]

    curr_kg       = storage_capacity

    # ── Fixed annualised costs that don't change with tank size ──────────────
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
    # Compressor is fixed regardless of tank volume (sized for PEM peak flow)
    comp_ann = _ann(H2_PEAK_KGH * LCOH_COMP_EUR_KGH, LCOH_WACC, LCOH_COMP_LIFE, LCOH_COMP_OM)

    # ── Sweep ─────────────────────────────────────────────────────────────────
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
        # Recorded directly. The old estimate assumed an empty opening tank, which is not
        # what the cyclic boundary uses, so it under-reported overflow by INITIAL_H2_STORAGE.
        overflow_est = df['H2_overflow'].sum()
        fulfill_pct  = 100.0 * h2d / max(annual_demand, 1e-6)
        autonomy     = tank_kg / max(annual_demand / 365.0, 1e-6)

        gimp  = df['grid_import_eur'].sum()
        grev  = df['grid_export_eur'].sum()
        o2r   = h2p * O2_KG_PER_KG_H2 * O2_PRICE_EUR_KG

        # Tank CAPEX/O&M for this size
        tank_ann = _ann(tank_kg * LCOH_TANK_EUR_KG, LCOH_WACC, LCOH_TANK_LIFE, LCOH_TANK_OM)

        # LCOH (production) — tank not in production chain; PEM REPEX follows this scenario's SoH.
        pem_econ = project_pem_stack_replacements(df, pem_stack_capex, wacc=LCOH_WACC)
        prod_ann = fixed_prod_ann + pem_econ['stack_ann_total'] + gimp - grev - o2r
        lcoh_prod = prod_ann / max(h2p, 1e-6)

        # LCOH (delivery) — denominator is H2 delivered
        lcoh_del  = prod_ann / max(h2d, 1e-6)

        # LCSH — adds compressor + this tank size
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

    # ── Figure ────────────────────────────────────────────────────────────────
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
    # Find index of closest tank size to current capacity
    try:
        curr_tidx = next(i for i, v in enumerate(tx) if abs(v - curr_kg) < 1)
    except StopIteration:
        curr_tidx = min(range(len(tx)), key=lambda i: abs(tx[i] - curr_kg))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    # on-plot title removed — caption in FIGURE CAPTIONS (Tank_Sensitivity)  [MOO variant, unused]

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

    # Panel 6: combined LCOH_del & LCSH on same axes for direct comparison
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
    """
    Five publication-quality figures comparing all locations × controllers.

    Saved as both PDF (vector) and PNG (300 dpi) in out_dir/:
      Fig1_H2_Production_Overview.pdf/.png
      Fig2_LCOH_LCSH.pdf/.png
      Fig3_LCOH_Cost_Breakdown.pdf/.png
      Fig4_Operational_Performance.pdf/.png
      Fig5_RL_Advantage.pdf/.png

    Parameters
    ----------
    all_results : {loc_label: {ctrl_label: df}}  (4-way incl. MOO+Hyst when present)
    all_lcoh    : {loc_label: {ctrl_label: metrics_dict}}  (from calculate_lcoh)
    out_dir     : Path or str
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Style ──────────────────────────────────────────────────────────────────
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
    CTRL_LABELS = list(all_results[LOC_LABELS[0]].keys())   # 4-way incl. hysteresis when present
    LOC_SHORT   = [l.replace('_2021', '') for l in LOC_LABELS]

    COLORS  = [CTRL_COLORS[c] for c in CTRL_LABELS]
    HATCHES = [CTRL_HATCH[c]  for c in CTRL_LABELS]
    N_LOC   = len(LOC_LABELS)
    N_CTRL  = len(CTRL_LABELS)
    W_BAR   = 0.8 / N_CTRL
    OFFS    = (np.arange(N_CTRL) - (N_CTRL - 1) / 2.0) * W_BAR
    X       = np.arange(N_LOC)
    PEM_MAX_KG = H2_PEAK_KGH * 8760.0

    # ── Collect metrics ────────────────────────────────────────────────────────
    def _switches(df):
        return int(df['n_parallel'].diff().fillna(0).ne(0).sum())

    met = {}
    for loc, ctrl_dfs in all_results.items():
        met[loc] = {}
        for ctrl, df in ctrl_dfs.items():
            met[loc][ctrl] = {
                # operational metrics from raw dataframe
                'profit':        df['profit_eur'].sum(),
                'grid_imp_kwh':  df['grid_bought'].sum() * DT_HOURS / 1000.0,
                'curtailed_kwh': df['P_curtailed_W'].sum() * DT_HOURS / 1000.0,
                'soh_pem':       df['SoH_PEM'].iloc[-1]  * 100.0,
                'soh_pv':        df['SoH_PV'].iloc[-1]   * 100.0,
                'soh_batt':      df['SoH_BATT'].iloc[-1] * 100.0,
                'switches':      _switches(df),
                # economic metrics from LCOH function (h2_prod, lcoh, lcsh, etc.)
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
        # extra top headroom for the rotated value labels; legend is placed once per
        # figure OUTSIDE the axes (below), so it never overlaps bars or numbers.
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — H2 Production & Energy Overview   (2 × 2)
    # ═══════════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))
    # (figure-level title omitted — added as a caption in the manuscript)

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

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — LCOH & LCSH side by side   (1 × 2)
    # ═══════════════════════════════════════════════════════════════════════════
    fig2, (ax2l, ax2r) = plt.subplots(1, 2, figsize=(11, 5))
    # (figure-level title omitted — added as a caption in the manuscript)

    # No cost benchmark line: the reference value is scenario-dependent and was being
    # read as a target the design must hit, which it is not.
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

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 3 — LCOH Cost Component Breakdown   (one subplot per location)
    # ═══════════════════════════════════════════════════════════════════════════
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

            # Per-kg breakdown:
            # LCOH = fixed_prod/h2 + grid_import/h2 - grid_export/h2 - o2/h2
            # fixed_prod/h2 = LCOH + grid_export/h2 + o2/h2 - grid_import/h2
            g_imp = m['grid_import_yr']  / h2p
            g_exp = m['grid_export_yr']  / h2p
            o2r   = m['o2_revenue_yr']   / h2p
            fixed = m['lcoh'] + g_exp + o2r - g_imp   # gross fixed cost per kg

            # Storage add-on per kg (LCSH - LCOH)
            stor_addon = m['lcsh'] - m['lcoh']

            # Positive stacks (costs)
            ax3.bar(xs_ctrl[ci], fixed,      bottom=0,     width=WB3,
                    color=C_FIXED, hatch=H_FIXED, edgecolor='white', lw=0.5,
                    label='Fixed CAPEX+OæM' if (li == 0 and ci == 0) else '')
            ax3.bar(xs_ctrl[ci], g_imp,      bottom=fixed, width=WB3,
                    color=C_IMP,   hatch=H_IMP,   edgecolor='white', lw=0.5,
                    label='Grid Import'            if (li == 0 and ci == 0) else '')
            ax3.bar(xs_ctrl[ci], stor_addon, bottom=fixed + g_imp, width=WB3,
                    color=C_STOR,  hatch=H_STOR,  edgecolor='white', lw=0.5,
                    label='Storage add-on (LCSH)'  if (li == 0 and ci == 0) else '')

            # Negative stacks (revenue offsets)
            ax3.bar(xs_ctrl[ci], -g_exp, bottom=0,     width=WB3,
                    color=C_EXP, hatch=H_EXP, edgecolor='white', lw=0.5,
                    label='Grid Export Revenue'    if (li == 0 and ci == 0) else '')
            ax3.bar(xs_ctrl[ci], -o2r,  bottom=-g_exp, width=WB3,
                    color=C_O2,  hatch=H_O2,  edgecolor='white', lw=0.5,
                    label='O₂ By-product Revenue' if (li == 0 and ci == 0) else '')

            # Net LCOH marker (diamond) — label to the LEFT of the marker, bold, boxed
            ax3.scatter(xs_ctrl[ci], m['lcoh'], marker='D', color='black',
                        s=55, zorder=6,
                        label='Net LCOH' if (li == 0 and ci == 0) else '')
            ax3.annotate(f"{m['lcoh']:.1f}", (xs_ctrl[ci], m['lcoh']),
                         xytext=(-9, 0), textcoords='offset points',
                         ha='right', va='center', fontsize=11, fontweight='bold',
                         color='black',
                         bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

            # Net LCSH marker (circle) — label to the RIGHT, bold, dark red on white box
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
    # Legend as a horizontal strip BELOW the panels (outside the data area).
    fig3.legend(handles=leg_handles, loc='lower center', ncol=4,
                framealpha=0.92, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig3.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig3, 'Fig3_LCOH_Cost_Breakdown')

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 4 — Operational Performance   (1 × 3)
    # ═══════════════════════════════════════════════════════════════════════════
    fig4, axes4 = plt.subplots(1, 3, figsize=(13, 5))
    # (figure-level title omitted — added as a caption in the manuscript)

    _bar_group(axes4[0], 'switches', 'MOSFET Switching Events yr⁻¹',
               'Switching Events\n(fewer = less hardware fatigue)', fmt='{:.0f}')
    _bar_group(axes4[1], 'soh_pem',  'Final PEM SoH (%)',
               'PEM State of Health\n(end of simulated year)', fmt='{:.2f}')
    _bar_group(axes4[2], 'soh_pv',   'Final PV SoH (%)',
               'PV State of Health\n(end of simulated year)', fmt='{:.3f}')

    # Zoom SoH axes so small differences are visible
    for ax4, key in [(axes4[1], 'soh_pem'), (axes4[2], 'soh_pv')]:
        all_vals = [met[loc][ctrl][key] for loc in LOC_LABELS for ctrl in CTRL_LABELS]
        lo = min(all_vals)
        ax4.set_ylim(lo * 0.9985, max(all_vals) * 1.002)

    _h4, _l4 = axes4[0].get_legend_handles_labels()
    fig4.legend(_h4, _l4, loc='lower center', ncol=len(_l4), fontsize=9,
                framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig4.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig4, 'Fig4_Operational_Performance')

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE 5 — RL vs MOO Advantage   (2 × 2 delta plots)
    # ═══════════════════════════════════════════════════════════════════════════
    fig5, axes5 = plt.subplots(2, 2, figsize=(11, 8))
    # (figure-level title omitted — added as a caption in the manuscript)

    # sign = +1 means "higher is better", -1 means "lower is better"
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
    """
    Additional TECHNICAL cross-location figures (physics/operation, not economics):
      Fig6  — Switching ↔ MPPT-loss trade-off (the Pareto that defines RL's value)
      Fig7  — Technical efficiency: MPP shortfall, Solar-to-H2 efficiency, PV curtailment
      Fig8  — Monthly H2 production profile per location (seasonal / climate signature)
      Fig9  — Component State-of-Health: PEM, PV, Battery (longevity)
    Saved as PDF (vector) + PNG (300 dpi) in out_dir.
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
    CTRL   = list(all_results[LOCS[0]].keys())    # 4-way incl. hysteresis when present
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

    # ── FIG 6 — Switching ↔ MPPT-loss trade-off (Pareto) ───────────────────────
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
    # BOTH legends placed OUTSIDE the axes at the top-right, stacked vertically,
    # so the data area stays completely clean.
    _l1 = ax.legend(handles=ctrl_leg, title='Controller (colour)', loc='upper left',
                    bbox_to_anchor=(1.02, 1.0), framealpha=0.9, borderaxespad=0.0)
    ax.add_artist(_l1)
    ax.legend(handles=loc_leg, title='Location (marker)', loc='upper left',
              bbox_to_anchor=(1.02, 0.60), framealpha=0.9, borderaxespad=0.0)
    fig6.tight_layout()
    _save(fig6, 'Fig6_Switching_MPPT_Tradeoff')

    # ── FIG 7 — Technical efficiency (1 × 3) ───────────────────────────────────
    fig7, a7 = plt.subplots(1, 3, figsize=(15, 5))
    # (figure-level title omitted — added as a caption in the manuscript)
    _grouped(a7[0], _mppt, 'MPP shortfall (kWh yr⁻¹)', 'MPP Shortfall\n(off-MPP operation)')
    _grouped(a7[1], _sth,  'Mean StH efficiency (%)',   'Solar-to-Hydrogen Efficiency', zoom=True)
    _grouped(a7[2], _curt, 'Curtailed PV (kWh yr⁻¹)',   'Curtailed PV Energy\n(surplus not used)')
    _fig_legend(fig7, a7[0], 3)
    fig7.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig7, 'Fig7_Technical_Efficiency')

    # ── FIG 8 — Monthly H2 production per location (seasonal signature) ─────────
    n_h = min(len(next(iter(dfs.values()))['MOO']), STEPS_PER_YEAR)
    month = (pd.to_datetime('2021-01-01') + pd.to_timedelta(np.arange(n_h) * DT_HOURS, 'h')).month.to_numpy()
    mx    = np.arange(1, 13)
    fig8, a8 = plt.subplots(1, N, figsize=(5.3 * N, 5), sharey=True)
    if N == 1:
        a8 = [a8]
    # (figure-level title omitted — added as a caption in the manuscript)
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

    # ── FIG 9 — Component State-of-Health (1 × 3) ──────────────────────────────
    fig9, a9 = plt.subplots(1, 3, figsize=(15, 5))
    # (figure-level title omitted — added as a caption in the manuscript)
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
    """
    Per-DECISION inference time for the FOUR compared controllers — the on-line cost
    once trained:
      • MOO      — full perturb-&-observe optimisation over all MOSFET positions every step
      • MOO+Hyst — the same optimisation, then a deadband gate on the resulting move
      • MOO+Pen  — the same optimisation with the switching charge added inside the score
      • RL Agent — one policy-network forward pass
    The three MOO variants are TIMED SEPARATELY rather than assumed equal: the deadband
    and the switching term are cheap, but "cheap" is a measurement, not an assumption, and
    the point of the figure is that only the learned policy changes the COST CLASS (it does
    not enumerate the ladder at all). MOO+RF is timed as well when INCLUDE_RF_MOO is on;
    its forest is trained offline and that training is NOT counted here.

    Returns {controller_label: seconds_per_decision}, plus a 'RL Agent (policy only)' entry
    when the bare forward pass could be timed. Saves computational_time_comparison.pdf/.png.
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

    # One optimiser body, parameterised exactly the way run_baselines parameterises it:
    # lam > 0 adds the switching charge to the per-position score (the MOO+Pen mechanism),
    # and prev_mosf supplies the rung the charge is measured against. Sharing the body is
    # what makes the three timings comparable — the only difference between them is the
    # work each mechanism actually adds.
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
        # Full optimisation, then the deadband gate — the extra cost is the gate only.
        n_par, mosf = _moo_decision(g, t)
        if mosf != prev_mosf and abs(mosf - prev_mosf) < MOO_DEADBAND:
            return PVPEMEnv.BASE_PARALLEL + prev_mosf * PVPEMEnv.PARALLEL_PER_MOS
        return n_par

    # Plain-MOO answers for every sample. Reused twice: as RF training targets, and as the
    # realistic "previous rung" each penalised/deadband decision is measured against, so
    # those two branches are genuinely exercised instead of short-circuiting on a constant.
    X = [[irr[i], temp[i]] for i in samp]
    _dec = [_moo_decision(irr[i], temp[i]) for i in samp]
    y    = [d[0] for d in _dec]
    _prev_mosf = [_dec[0][1]] + [d[1] for d in _dec[:-1]]
    if INCLUDE_RF_MOO:
        rf = RandomForestRegressor(n_estimators=30, max_depth=10, min_samples_leaf=3, random_state=42)
        rf.fit(X, y)

    # a handful of RL observation vectors
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

    # Policy-only inference, bypassing SB3's predict() wrapper. predict() re-validates and
    # reshapes the observation, builds a tensor, enters no_grad and post-processes the action
    # on every call; that Python overhead is several times the forward pass itself and belongs
    # to the training framework, not to the controller. A deployed agent exports the network
    # (TorchScript/ONNX) and pays only the forward pass, so both numbers are reported.
    t_rl_net = float('nan')
    try:
        import torch as _th
        _pol = rl_model.policy
        _pol.set_training_mode(False)
        _obs_t = _th.as_tensor(np.asarray(obs_list, dtype=np.float32))
        # _predict() is the ACTOR only. policy(...) would also run the value network, which
        # a deployed controller never evaluates, and timing that overstates inference cost.
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
    # on-plot title removed — caption in FIGURE CAPTIONS (Computational-cost figure)
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
    """
    At-a-glance multi-criteria radar over the compared controllers, averaged across
    locations. Every axis is oriented so OUTER = better, min-max normalised across the
    controllers (with a 0.2 floor so the weakest stays visible):
      H2 production · Switching gentleness · PEM longevity · PV longevity ·
      Solar-to-H2 efficiency · Curtailment avoidance · Economy (low LCOH) ·
      MPP tracking · Decision speed (if timing is supplied).

    Battery longevity was dropped as an axis: the four controllers finish within a few
    hundredths of a percentage point of each other on SoH_BATT, so min-max normalising it
    stretches numerical noise across the full radius and reads as a large difference. The
    durability that the controllers genuinely separate on is the stack, and the axes that
    replace it (PEM and PV state-of-health, and solar-to-hydrogen efficiency) are the ones
    the manuscript actually reports.
    """
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    LOCS = list(all_results.keys())
    CTRL = list(all_results[LOCS[0]].keys())       # 4-way incl. hysteresis when present
    COLS = [CTRL_COLORS[c] for c in CTRL]

    # Mean solar-to-hydrogen efficiency over producing steps, matching print_summary's
    # definition (StH > 0 only, so night steps do not dilute it).
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
    # decision_times is the {controller: seconds} dict returned by plot_computational_cost,
    # so every controller now carries its OWN measured time instead of the deadband and the
    # penalised variant being assigned MOO's by assumption. Anything missing from the dict
    # (e.g. a controller whose timing was skipped) falls back to MOO's, since all three MOO
    # variants run the same solver.
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
    # Hide the default tick labels and place each axis label OUTSIDE the ring,
    # aligned by its angle so long labels never overlap the plotted polygons.
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([])
    ax.set_yticklabels([]); ax.set_ylim(0, 1.05)
    for a, lab in zip(ang[:-1], labels):
        ca, sa = np.cos(a), np.sin(a)
        ha = 'left' if ca > 0.25 else ('right' if ca < -0.25 else 'center')
        va = 'bottom' if sa > 0.25 else ('top' if sa < -0.25 else 'center')
        ax.text(a, 1.18, lab, ha=ha, va=va, fontsize=12)
    # Legend below the axis labels. With eight or nine axes a label falls near the bottom
    # of the ring, so the legend cannot sit in that gap at -0.03; it is pushed clear.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=len(CTRL),
              framealpha=0.9, fontsize=12)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'Fig10_Controller_Radar.{ext}', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print("  Radar figure saved: Fig10_Controller_Radar.pdf/.png")


def print_cross_location_lcoh(all_lcoh):
    """
    Print a cross-location LCOH / LCSH comparison table.

    Parameters
    ----------
    all_lcoh : dict  {location_label: {controller_label: metrics_dict}}
               as returned by calculate_lcoh() for each location.
    """
    locations   = list(all_lcoh.keys())
    controllers = [c for c in CTRL_ORDER if c in all_lcoh[locations[0]]]   # 4-way incl. hysteresis
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
        # Prefer the exact hourly sheet used by the final hourly study (e.g. Prague_2021).
        hourly_sheet = None
        if sheet_name in xls.sheet_names and sheet_name != resolved_sheet:
            hourly_sheet = sheet_name
        else:
            non15 = [sh for sh in site_sheets
                     if sh != resolved_sheet and not ('15' in sh.lower() or 'quarter' in sh.lower() or 'min' in sh.lower())]
            for sh in non15:
                probe = pd.read_excel(xls, sheet_name=sh, nrows=5)
                # Exact row count is checked after selecting the first plausible site sheet.
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

        # Irradiance: preserve each hour's quarter-hour shape but force the quartet
        # mean to equal the exact hourly reference value. Enforce the same 0..1200 W/m2
        # physical bound used by the simulator WITHOUT losing the hourly mean through
        # a later hidden clip.
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

            # Preserve the target quartet sum under the simulator's irradiance bound.
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

            # Tiny floating-point residual goes to any sample with available headroom.
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

        # Temperature: preserve intra-hour deviations but match the hourly mean exactly.
        qT = temp_vals.reshape(-1, 4)
        qT += (hT - qT.mean(axis=1))[:, None]
        temp_vals = qT.reshape(-1)

        # Price is hourly/day-ahead in the reference study: repeat the exact hourly value.
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
        # FINAL DAILY RESOLUTION STUDY: use the native 15-min measurements exactly as
        # stored in the workbook. The hourly sheet is loaded ONLY for an audit; it
        # does not rescale, shift, repeat or otherwise modify the 15-min input.
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
        # Display label : EXACT native 15-min Excel sheet.
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
    all_results = {}   # {location_label: {ctrl_label: df}}  4-way incl. MOO+Hyst
    sizing_series = {} # {location_label: (irr, temp, price)} — used to verify the sizing pick

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

        # Native 15-min resource audit. Exact comparison with the corresponding
        # 8760-row sheet is already printed inside load_data_from_excel().
        _ghi_kwh_m2 = float(np.sum(np.clip(np.asarray(irr, dtype=float), 0.0, 1200.0))
                            * DT_HOURS / 1000.0)
        _daylight_h = float(np.sum(np.asarray(irr, dtype=float) >= 80.0) * DT_HOURS)
        print(f"  INPUT AUDIT: native integrated GHI={_ghi_kwh_m2:.1f} kWh/m2/yr, "
              f"G>=80 duration={_daylight_h:.1f} h/yr")

        # ---- per-site hardware from the sizing study (applied to ALL controllers) ----
        # PV oversizing is realised as the installed PARALLEL STRING COUNT (exact), not by
        # scaling the photocurrent, so pv_scale stays at 1.0 for every controller.
        site_pv_scale = 1.0
        set_string_ladder(1.08)          # default array unless the site overrides it
        _sz = site_sizing_for(loc_label)
        if USE_SITE_SIZING and _sz is None:
            # Never fall through quietly: an unmatched label means the run would
            # simulate the default plant while claiming to be this site.
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

        # 0. (optional) SEED-VARIANCE STUDY — train N seeds, report mean +/- std,
        #    then skip the normal pipeline for this location.
        if RUN_SEED_STUDY:
            seed_variance_study(irr, temp, price, period_id, plots_dir, seeds=SEED_LIST)
            continue

        # 0b. (optional) PARETO FRONT — sweep LAMBDA_SWITCH, then skip the pipeline.
        if RUN_PARETO_STUDY:
            pareto_switching_study(irr, temp, price, period_id, plots_dir,
                                   lambdas=PARETO_LAMBDAS, seeds=PARETO_SEEDS,
                                   zero_underuse=PARETO_ZERO_UNDERUSE)
            continue

        # 0c. (optional) REWARD-WEIGHT CORNER STUDY — 2-factor sensitivity, then skip.
        if RUN_REWARD_STUDY:
            reward_weight_study(irr, temp, price, period_id, plots_dir,
                                corners=REWARD_CORNERS, seeds=REWARD_SEEDS)
            continue

        # 0d. (optional) SIZING STUDY — MOO-only, self-sufficiency headline. Runs BOTH
        #     PV x battery and PV x H2-tank sweeps, then skips the pipeline.
        if RUN_PV_SWEEP:
            _sz = run_dir / 'Sizing_Study'
            sizing_study(irr, temp, price, period_id, _sz, sweep='battery')
            sizing_study(irr, temp, price, period_id, _sz, sweep='tank')
            # Kept so plot_sizing_pareto can RE-SIMULATE its recommendation instead of
            # reporting the interpolated target as if it were a result.
            sizing_series[loc_label] = (irr, temp, price)
            continue

        # 0e. CYCLIC TANK — solve the January-1 inventory the year reproduces on 31 Dec.
        #     Every controller then opens from the SAME level, so none gets a head start.
        apply_cyclic_storage(irr, temp, price, period_id)

        # 1. MOO Baseline (greedy, memoryless)
        df_moo = run_baselines(irr, temp, price, use_rf=False, hysteresis=False, pv_scale=site_pv_scale)
        df_moo.to_csv(data_dir / f'moo_results_{period_id}.csv', index=False)
        print(f"  -> MOO done.")

        # 1b. Hard-deadband MOO — hold unless the greedy optimum moves >= MOO_DEADBAND steps.
        df_moo_hyst = None
        if INCLUDE_HYSTERESIS_MOO:
            df_moo_hyst = run_baselines(irr, temp, price, use_rf=False,
                                        hysteresis=True, lambda_sw=0.0,
                                        pv_scale=site_pv_scale)
            df_moo_hyst.to_csv(data_dir / f'hyst_results_{period_id}.csv', index=False)
            print(f"  -> MOO+Hyst (deadband={MOO_DEADBAND}) done.")

        # 1c. Soft-penalty MOO — an explicit -lambda|delta n_par| charge inside the score.
        # NOTE the charge is MOO_PEN_LAMBDA (0.02), NOT the DRL's LAMBDA_SWITCH (0.13): at the
        # DRL's value this controller pins in the lower third of the ladder and demand collapses
        # (see the note at MOO_PEN_LAMBDA). The comparison is therefore not like-for-like on the
        # switching price, and the paper says so explicitly.
        df_moo_pen = None
        if INCLUDE_PENALISED_MOO:
            df_moo_pen = run_baselines(irr, temp, price, use_rf=False,
                                       hysteresis=False, lambda_sw=MOO_PEN_LAMBDA,
                                       pv_scale=site_pv_scale)
            df_moo_pen.to_csv(data_dir / f'pen_results_{period_id}.csv', index=False)
            print(f"  -> MOO+Pen (lambda_sw={MOO_PEN_LAMBDA}) done.")

        # 1d. Rule-based pre-warm MOO — greedy string search + a hand-designed warm-standby
        # rule on the SAME heater the DRL agent commands. This is the control that separates
        # "the extra thermal actuator is worth something" from "the LEARNING is worth
        # something": if it recovers most of the DRL's ~1.7 pp stack-health advantage, the
        # contribution is a deployable rule that RL discovered; if it does not, the residual
        # is a genuine learning gain measured against the strongest rule-based alternative.
        # Deterministic and untrained, so one pass per site is enough (no seeds needed).
        df_moo_pw = None
        if INCLUDE_PREWARM_MOO:
            df_moo_pw = run_baselines(irr, temp, price, use_rf=False, hysteresis=False,
                                      lambda_sw=0.0, pv_scale=site_pv_scale,
                                      prewarm=PREWARM_RULE)
            df_moo_pw.to_csv(data_dir / f'prewarm_results_{period_id}.csv', index=False)
            print(f"  -> MOO+PreWarm (rule='{PREWARM_RULE}') done.")

        # 1e. BOTH hand-designed rules — deadband on the strings AND pre-warm on the heater.
        # The ceiling of hand engineering: everything the DRL can actuate, actuated by rule.
        # The DRL's margin over THIS controller, not over greedy MOO, is what the learning is
        # worth, and it is the number a referee will look for first.
        df_moo_hpw = None
        if INCLUDE_HYST_PREWARM_MOO:
            df_moo_hpw = run_baselines(irr, temp, price, use_rf=False, hysteresis=True,
                                       lambda_sw=0.0, pv_scale=site_pv_scale,
                                       prewarm=PREWARM_RULE)
            df_moo_hpw.to_csv(data_dir / f'hystprewarm_results_{period_id}.csv', index=False)
            print(f"  -> MOO+Hyst+PreWarm (deadband={MOO_DEADBAND}, rule='{PREWARM_RULE}') done.")

        # ---- 15-min PRE-FLIGHT CHECK BEFORE ANY PPO TRAINING --------------------
        # This makes resolution mistakes visible in <3 minutes rather than after 60 PPO years.
        # "restart-event" in the degradation print above includes warm-restart damage too;
        # the actual cold-start count below uses the declared 50 C pre-restart classifier.
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

        # 2. MOO + RF surrogate (off by default — it reproduces greedy MOO exactly)
        df_rf = None
        if INCLUDE_RF_MOO:
            df_rf = run_baselines(irr, temp, price, use_rf=True, pv_scale=site_pv_scale)
            df_rf.to_csv(data_dir / f'rf_results_{period_id}.csv', index=False)
            print(f"  -> MOO+RF done.")

        # 3. RL — reuse a saved model if requested (and available), else train fresh
        model = None
        if REUSE_SAVED_MODELS:
            model = load_saved_rl_model(period_id, model_dir)
            if model is None:
                print("  -> REUSE_SAVED_MODELS is True but no saved model was found "
                      "for this location — training a fresh one.")
        if model is None:
            model = train_rl(irr, temp, price, period_id, model_dir, plots_dir, pv_scale=site_pv_scale)
        df_rl = evaluate_rl(model, irr, temp, price, period_id, data_dir, plots_dir, pv_scale=site_pv_scale)

        # 4. Plots & Excel export
        _kw = dict(df_hyst=df_moo_hyst, df_pen=df_moo_pen,
                   df_pw=df_moo_pw, df_hpw=df_moo_hpw)
        plot_3way_comparison(df_moo, df_rf, df_rl, period_id, plots_dir,
                             pv_scale=site_pv_scale, **_kw)
        export_3way_excel(df_moo, df_rf, df_rl, period_id, data_dir, **_kw)
        plot_energy_stacks(df_moo, df_rf, df_rl, period_id, plots_dir, **_kw)

        # 5. Per-location summary + N-way comparison over every baseline produced above
        print_summary(df_moo, df_rf, df_rl, period_id, **_kw)
        _ctrls = _ctrl_dfs(df_moo, df_rf, df_rl, df_moo_hyst, df_moo_pen,
                           df_moo_pw, df_moo_hpw)
        print_controller_comparison(_ctrls, period_id)
        audit_controller_energy_allocation(_ctrls, period_id)
        audit_rl_reward_components(df_rl, period_id)

        # 6. Per-location LCOH / LCSH (also returns results dict)
        all_lcoh[loc_label]    = calculate_lcoh(df_moo, df_rf, df_rl, pv_peak_kw, **_kw)
        all_results[loc_label] = _ctrls

        # 7. Battery size sensitivity (MOO only) — justifies the chosen battery size
        if RUN_BATTERY_SENSITIVITY:
            batt_dir = run_dir / 'Battery_Sensitivity'
            battery_sensitivity_analysis(irr, temp, price, pv_peak_kw,
                                         loc_label, batt_dir)

        # 8. H2 tank size sensitivity using trained RL (faster than MOO re-runs)
        if RUN_TANK_SENSITIVITY:
            tank_dir = run_dir / 'Tank_Sensitivity'
            tank_sensitivity_analysis_rl(model, irr, temp, price, pv_peak_kw,
                                         loc_label, tank_dir)

        # 9. Tornado plots: parameter sensitivity for LCOH (RL-based)
        if RUN_TORNADO:
            tornado_dir = run_dir / 'Tornado_Sensitivity'
            tornado_sensitivity_analysis(irr, temp, price, pv_peak_kw,
                                         loc_label, tornado_dir, model=model,
                                         pv_scale=site_pv_scale)

    # ---- If running the seed-variance study, we are done: skip everything below
    #      (no 'model'/'all_results' were produced because each location 'continue'd).
    if RUN_SEED_STUDY or RUN_PARETO_STUDY or RUN_REWARD_STUDY or RUN_PV_SWEEP:
        elapsed = time.time() - t_start
        _which = ('Seed-variance' if RUN_SEED_STUDY else
                  'Pareto-front'  if RUN_PARETO_STUDY else
                  'Reward-weight' if RUN_REWARD_STUDY else 'PV-oversizing')
        if RUN_PV_SWEEP:
            # No pv_override: the previous {'Prague': 1.2} was chosen under the old polarization
            # curve and the old (0.75..1.6) ladder, and 1.2x is not even a point in the current
            # sweep.
            # Odeillo: the pure-cost search clears dem_target at 1200 kg, but the deployed
            # design uses 1400 kg (SITE_SIZING) for the extra buffer the DRL agent needs --
            # pv_override pins the PV ratio the search already agrees on (1.25x) so only the
            # tank is forced, not re-optimized. Keys are the SHORT location name
            # (plot_sizing_pareto strips '_2021' into its own 'loc' column internally), not
            # the loc_label used everywhere else in this file -- a long-form key here is
            # silently ignored rather than raising, so it is easy to pass the wrong one.
            plot_sizing_pareto(OUT_DIR, batt_min=500.0, tol=0.03,
                               pv_override={'Odeillo': 1.25},
                               tank_override={'Odeillo': 1400},
                               series=sizing_series)
        print(f"\n{_which} study complete. Total runtime: {elapsed/60:.1f} min")
        print("Results (CSV + figures) saved under each run's Plots folder.")
        raise SystemExit

    # ---- Computational-cost comparison (deployment inference time; hardware-only, run once)
    figs_dir = Path(OUT_DIR) / 'CrossLocation_Figures'
    print(f"\nTiming per-decision computational cost (MOO vs MOO+Hyst vs MOO+Pen vs RL)...")
    decision_times = plot_computational_cost(model, irr, temp, price, figs_dir)

    # ---- Cross-location summaries & figures (only meaningful for >=2 locations) ----
    if len(all_results) >= 2:
        # Cross-location text summary
        print_cross_location_lcoh(all_lcoh)

        # Cross-location publication figures (economics + technical + radar summary)
        print(f"\nGenerating cross-location publication figures -> {figs_dir}")
        plot_cross_location_comparison(all_results, all_lcoh, figs_dir)
        plot_cross_location_technical(all_results, figs_dir)
        plot_controller_radar(all_results, all_lcoh, figs_dir, decision_times)
    else:
        print("\nSingle-location run — cross-location summary and figures skipped.")

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed/60:.1f} min")
    print(f"All results saved under: {OUT_DIR}")
