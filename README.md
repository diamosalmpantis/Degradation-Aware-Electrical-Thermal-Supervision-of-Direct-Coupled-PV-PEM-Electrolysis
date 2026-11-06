# Degradation-aware control of a directly coupled PV–PEM hydrogen plant

Simulation and control code for a 1 MW photovoltaic array coupled **directly** to a PEM
electrolyser — no DC–DC stage between them. The array is reconfigured by switching
parallel strings in and out, which is the only electrical actuator available, and the
stack's thermal state is the only other one. The study asks a narrow question: once stack
degradation is priced into the objective, how much of the achievable cost reduction needs a
learned policy, and how much a plain engineering rule already delivers?

Six controllers are compared on one identical plant, so the contribution of each added
capability can be read off separately.

## The controller ladder

| Controller | String switching | Thermal action | Anticipation |
|---|---|---|---|
| `MOO` | greedy per-step optimum | none | none |
| `MOO+Hyst` | deadband | none | none |
| `MOO+PreWarm` | greedy | pre-warm rule | forecast-driven |
| `MOO+Hyst+PreWarm` | deadband | pre-warm rule | forecast-driven |
| `MOO+AlwaysWarm` | greedy | constant 60 °C standby | none |
| `PPO` | learned | learned | learned |

`MOO+Hyst+PreWarm` is deliberately the hardest baseline: everything the agent can do, done
by hand. Two further controllers (`MOO+RF`, a random-forest surrogate, and `MOO+Pen`, a
penalised variant) are implemented but disabled by default — `MOO+RF` reproduced greedy
`MOO` exactly at every site, because its search window always contains `MOO`'s optimum.

## What the study finds

- **Constant warm standby wins on cost.** `MOO+AlwaysWarm` gives the lowest levelized cost
  of hydrogen at all three sites and in all 21 PPO training runs. Seven-seed means are
  10.59 ± 0.32 €/kg at Lund, 9.34 ± 0.25 at Prague and 7.97 ± 0.32 at Odeillo.
- **Cold starts dominate the damage.** Under greedy control they account for 67–71 % of
  annual stack health loss.
- **The learned policy wins on everything except cost**, cutting switching by 29–55 % and
  auxiliary grid energy by 33–38 %.
- **Forecast quality separates the two approaches.** Replacing perfect foresight with
  persistence forecasts costs the pre-warm rules 56–65 % of their benefit, against 1–18 %
  for the learned policy, which degrades gracefully because it does not depend on a single
  forecast threshold.

The negative result is the point: a memoryless 60 °C set-point is cheaper than a trained
agent, and the agent's value lies in actuator wear and forecast robustness rather than in
LCOH.

## Repository layout

**Main simulation**

| File | Purpose |
|---|---|
| `New_RL_1MW_final.py` | The pipeline. Six controllers, hourly resolution, per-site sizing, LCOH and degradation accounting. Everything else imports this. |
| `New_RL_1MW_final_15min.py` | The same study at native 15-minute resolution, on the reward settings frozen for the hourly run. |
| `Unseen_PPO.py` | Temporal out-of-sample test: a policy trained on 2021 evaluated on Prague 2023. |

**Sensitivity and robustness**

| File | Purpose |
|---|---|
| `forecast_robustness_study.py` | How much of each controller's benefit survives a realistic forecast. |
| `thermal_sensitivity_study.py` | Sensitivity of the conclusion to the thermal model itself. |
| `degradation_replacement_sensitivity.py` | 27-case sweep over degradation severity, end-of-life SoH and replaceable stack CAPEX. |
| `degradation_replacement_sensitivity_COLDSTART_AXIS.py` | The same, with cold/warm restart damage added as a fourth axis. |
| `check_degradation_calibration.py` | Verifies the degradation model against its literature band. Run before retraining. |

**Explainability**

| File | Purpose |
|---|---|
| `XAI_6controller.py` | Policy maps and action attribution across all six controllers (hourly). |
| `XAI_5controller_15min.py` | The 15-minute counterpart. Covers five controllers — `MOO+AlwaysWarm` is not part of that comparison. |

**Figures and tables**

| File | Output |
|---|---|
| `make_radar_6way.py` | Controller trade-space radar. |
| `make_coldstart_trace.py` | The cold-start mechanism figure. |
| `make_aux_table.py` | LaTeX body of the auxiliary-energy decomposition table. |
| `make_thermal_management_fig.py` | Thermal-management figure. |
| `make_thermal_management_pptx.py` | The same as an editable slide. |
| `make_topology_schematic.py`, `make_topology_pptx.py`, `make_topology_pptx_v2.py` | Plant topology schematic. `_v2` is current: the battery serves auxiliaries only. |

## Getting started

```bash
pip install numpy pandas matplotlib scipy scikit-learn gymnasium stable-baselines3 torch tqdm openpyxl python-pptx
```

The scripts expect an irradiance workbook named `Irradiance_values_example_new.xlsx` **in
the same directory**, with one sheet per site (`Lund_2021`, `Prague_2021`, `Odeillo_2021`).
It is not included here — see *Data* below.

Then run the main pipeline, which prompts for the site:

```bash
python New_RL_1MW_final.py
```

Results are written to `Results/Results_<SHEET_NAME>/`, split into `Plots/`, `Data/` and
`Model/`. The scripts also run unmodified in Spyder (F5) and in Google Colab.

All configuration sits in a single block at the top of `New_RL_1MW_final.py`. The switches
worth knowing:

| Switch | Effect |
|---|---|
| `REUSE_SAVED_MODELS` | Skip training and reload `rl_model_<site>.zip` from a previous run. |
| `RL_TIMESTEPS` | Training budget. Default is 60 simulated years. |
| `RL_SEED` | Training seed for the production policy. |
| `RUN_SEED_STUDY`, `SEED_LIST` | Retrain once per seed and report mean ± SD. Default seeds 0–6. |
| `USE_SITE_SIZING` | Apply each site's own PV ratio, tank and battery. Keep this on. |
| `RUN_TORNADO` | LCOH parameter sensitivity. |
| `RUN_PV_SWEEP`, `RUN_BATTERY_SENSITIVITY`, `RUN_TANK_SENSITIVITY` | Sizing sweeps. Slow; off during iteration. |

Training the full six-controller comparison across three sites and seven seeds is a long
job — 21 PPO runs of 60 simulated years each. For a single-site check, leave
`RUN_SEED_STUDY = False`.

## Plant configuration

A 1 MW PEM stack, 5 modules in series per PV string, and a 7-position string ladder running
from 35 % of the installed array to 100 %. Hydrogen is stored at up to 200 bar against a
daily delivery contract. Per-site hardware comes from the sizing study:

| Site | PV oversizing | H₂ tank | Battery |
|---|---|---|---|
| Lund | 1.75× | 7800 kg | 550 kWh |
| Prague | 1.75× | 5300 kg | 550 kWh |
| Odeillo | 1.25× | 1600 kg | 450 kWh |

Each optimum is interior to the swept grid, so none is a corner solution. The battery
serves auxiliaries only.

## Degradation model

Three mechanisms, all declared as cell overvoltage in µV and converted once onto the SoH
scale. Nothing is fitted here: the regressions are used as published.

1. **Steady operation** — load-dependent and ageing rates from the measured
   polarization-curve evolution of a square-wave campaign (Elizetxea-Navarro et al., 2026,
   Eqs. 7–8), evaluated at the stack's own current density. A ramp-amplification factor
   covers the profile dependence the regressions do not resolve.
2. **Thermal** — the degradation rate doubles per 10 K above the reference temperature
   (Chandesris et al., as reported by Maoulida et al., 2026), applied one-sidedly: the
   measurement brackets 333–353 K and says nothing about running colder, so no credit is
   given for letting the stack go cold. A separate term charges thermal cycling beyond what
   a working cooling loop absorbs.
3. **On/off** — 100 µV per cold restart against 10 µV per warm restart, the mutually
   exclusive classification turning on the stack temperature at the moment production
   resumes.

End of life is a 10 % rise in cell voltage; the economics replace the stack at SoH 0.80.

## Data

The irradiance workbook is derived from the CAMS radiation service and is not redistributed
here. Any hourly or 15-minute global-horizontal irradiance and ambient-temperature series
for the three sites will work, as long as the sheet names and column order match what
`New_RL_1MW_final.py` reads.

## Citation

This code accompanies a manuscript under review at *Solar Energy Advances*
(SEJA-D-26-00084). Please check the repository for an updated reference before citing; the
published DOI will be added here once available.

## License

No license file is included yet. Until one is added, all rights are reserved — if you want
to reuse this, please get in touch.
