# Degradation-Aware Deep RL for Direct-Coupled PV–PEM Electrolysis

Simulation, control and analysis code for *"Degradation-aware deep reinforcement learning
for direct-coupled PV–PEM electrolysis: cold-start mitigation, stack lifetime and
levelized hydrogen cost."*

A 1 MW photovoltaic array is coupled directly to a PEM electrolyzer with no intermediate
DC–DC stage. Electrical adaptability comes from a reconfigurable string matrix that
changes the number of active parallel PV strings. The supervisory problem is to coordinate
that electrical decision with the thermal conditions that govern restart damage and stack
lifetime.

Five controllers are compared on identical plant physics, degradation and economics over
complete meteorological years at Lund, Prague and Odeillo:

| Controller | Description |
|---|---|
| MOO | Greedy multi-objective optimizer, re-solved every timestep |
| MOO + Hyst | Adds a hysteresis deadband on the string decision |
| MOO + PreWarm | Adds a forecast-based warm-standby rule |
| MOO + Hyst + PreWarm | Both deterministic protections |
| PPO | Learned policy that jointly sets the string count, the warm-standby floor and the chiller ceiling |

## Repository layout

```
Hourly_resolution/     primary study, 60-minute supervisory timestep
15min_resolution/      sub-hourly resolution sensitivity
requirements.txt
LICENSE
```

### `Hourly_resolution/`

| File | Description |
|---|---|
| `New_RL_1MW_final.py` | Main pipeline: PV and PEM physics, thermal and degradation models, the four deterministic baselines, PPO training and evaluation, LCOH and replacement economics. |
| `Unseen_PPO.py` | Temporal out-of-sample test. Loads the frozen Prague-2021 PPO weights and evaluates them without retraining on unseen 2023 irradiance, temperature and price data. |
| `XAI_5controller.py` | Post-hoc explainability. Reconstructs decision-time states without post-action leakage, fits depth-3 decision-tree surrogates on a chronological 70/30 holdout, and runs the inference-time benchmark. |
| `degradation_replacement_sensitivity.py` | 27-case frozen-trajectory sweep over degradation severity, end-of-life SoH threshold and replaceable stack-CAPEX fraction. |
| `degradation_replacement_sensitivity_COLDSTART_AXIS.py` | 135-case extension adding the cold-to-warm restart damage ratio (1, 2, 5, 10, 20) as a fourth axis. This is the sensitivity that decides the LCOH ranking. |
| `make_coldstart_trace.py` | Generates the cold-start mechanism figure from a saved per-step trajectory. |
| `make_aux_table.py` | Decomposes auxiliary electricity into keep-warm, chiller and balance-of-plant, and reports the self-supplied fraction. |

### `15min_resolution/`

| File | Description |
|---|---|
| `New_RL_1MW_final_15min.py` | The same pipeline at a 15-minute timestep, with resolution-consistent reward scaling and forecast horizons. Not a frozen-policy transfer: the agent is retrained at the finer resolution, so the test isolates the resolution sensitivity of the control formulation. |
| `XAI_5controller_FINAL_15min.py` | Explainability analysis for the 15-minute runs. |

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10 or later. Developed against Stable-Baselines3 and Gymnasium.

## Input data

Both main scripts read a single Excel workbook with one sheet per site, holding the
irradiance, ambient temperature and electricity price series:

- hourly: `Irradiance_values_example_new.xlsx`
- 15-minute: `Irradiance_values_example_new_15min.xlsx`

The workbook is not included in this repository. Place it beside the script, or under
`/content/` when running on Google Colab.

## Usage

Run the main pipeline first. It is interactive and prompts for the demand mode and the
site to simulate:

```bash
python New_RL_1MW_final.py
```

Results, saved models and per-step trajectories are written to `Results_1MW_daily/`
(`Results_1MW_15min/` for the sub-hourly script). Everything downstream reads that
directory.

The analysis scripts are run afterwards against those outputs:

```bash
python XAI_5controller.py
python degradation_replacement_sensitivity_COLDSTART_AXIS.py
```

Each accepts a `RESULTS_ROOT_OVERRIDE` near the top of the file if the results live
elsewhere.

## Configuration

The behaviour of the main pipeline is set by module-level flags in the first ~350 lines,
not by command-line arguments. The ones that matter most:

| Flag | Default | Effect |
|---|---|---|
| `DEMAND_MODE` | `'daily'` | Daily or hourly hydrogen contract. The published primary results use the daily contract. |
| `DEMAND_TARGET_PCT` | `98.0` | Demand coverage target used to size the tank and PV array. |
| `USE_SITE_SIZING` | `True` | Uses the per-site sizing in `SITE_SIZING`. Must stay `True` to reproduce the published results, including the reward study. |
| `INCLUDE_HYSTERESIS_MOO`, `INCLUDE_PREWARM_MOO`, `INCLUDE_HYST_PREWARM_MOO` | `True` | The three deterministic ablation baselines. |
| `INCLUDE_PENALISED_MOO`, `INCLUDE_RF_MOO` | `False` | Switching-penalized MOO and the random-forest-accelerated MOO from the earlier studies. Reported in the supplementary material only. |
| `RUN_TORNADO` | `True` | LCOH parameter sensitivity over WACC, stack cost, electricity price and related terms. |
| `RUN_PV_SWEEP`, `RUN_BATTERY_SENSITIVITY`, `RUN_TANK_SENSITIVITY` | `False` | Sizing studies. Already performed; the outcome is fixed in `SITE_SIZING`. Re-enable only to redo the sizing. |
| `RUN_SEED_STUDY`, `RUN_PARETO_STUDY`, `RUN_REWARD_STUDY` | `False` | Multi-seed robustness, Pareto sweep and reward-coefficient calibration. |

## Reproducing the paper

1. Run `New_RL_1MW_final.py` once per site with `DEMAND_MODE = 'daily'`.
2. Run `XAI_5controller.py` for the surrogate fidelities and the timing benchmark.
3. Run `degradation_replacement_sensitivity_COLDSTART_AXIS.py` for the 135-case LCOH matrix.
4. Run `Unseen_PPO.py` for the frozen-policy 2023 evaluation.
5. Run `New_RL_1MW_final_15min.py` and `XAI_5controller_FINAL_15min.py` for the
   resolution sensitivity.

Training is stochastic. PPO results vary between seeds at the level reported in the
multi-seed section of the paper, so a single run will not reproduce the published numbers
digit for digit.

## Evaluation protocol

The short-horizon irradiance and temperature forecasts supplied to the agent and to the
pre-warm baselines are perturbed during PPO training only. At evaluation all five
controllers read the realized series. The comparison between controllers is therefore
internally consistent, but it is conditional on forecast quality a deployed plant would
not have. This is stated in the paper and is the principal open validation.

## Data availability

The irradiance, temperature and price time series and the full simulation outputs are
available on reasonable request. See the paper's Data and Code Availability statement.

## Funding

Supported by the Horizon Europe project CIRCULAR FUELS (grant agreement No. 101118239),
funded by the European Union.

## Citation

Citation details will be added on publication.

## License

See [LICENSE](LICENSE).

