# DRL-controler-for-direct-PV-PEM-coupling

# State-Aware Deep RL for Direct-Coupled PV–PEM Reconfiguration

Simulation and control code for *"State-aware deep reinforcement learning for
direct-coupled PV–PEM hydrogen supply to sustainable aviation fuel synthesis."*

Four control strategies are compared on identical physics for a 1 MW direct-coupled
PV–PEM electrolyzer stack across three European sites (Lund, Prague, Odeillo): a greedy
multi-objective optimizer (MOO), MOO with a hysteresis deadband, MOO with a switching
penalty, and a state-aware PPO (Deep Reinforcement Learning) agent that jointly sets the
active PV string count and the electrolyzer's warm-standby/cooling thermal band.

## Contents

| File | Description |
|---|---|
| `New_RL_1MW.py` | Main simulation, physics model, MOO baselines, and PPO training/evaluation pipeline at hourly resolution. |
| `New_RL_15min.py` | Same pipeline at 15-minute temporal resolution (sub-hourly robustness check). |
| `xai_analysis.py` | Post-hoc explainability: decision-tree surrogates distilled from each controller's action history (hourly runs). |
| `xai_analysis_15min.py` | Same explainability analysis for the 15-minute runs. |

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended (developed and tested with Stable-Baselines3 / Gymnasium).

## Usage

Each script is self-contained and runs from the command line:

```bash
python New_RL_1MW.py
```

The pipeline prompts for the demand mode (daily/hourly contract) and the site(s) to run
(Lund, Prague, Odeillo). Input irradiance/temperature/price data is read from an Excel
workbook (`Irradiance_values_example_new.xlsx`, one sheet per site — not included in this
repository) expected in the same directory as the script, or under `/content/` when run
on Google Colab.

`xai_analysis.py` / `xai_analysis_15min.py` are run afterward, against the CSV outputs
produced by the corresponding main script, to fit the decision-tree surrogates and
generate the explainability figures.

## Data availability

The irradiance/temperature/price time series and full simulation outputs are available
upon reasonable request. See the paper's Data and Code Availability statement.

## Funding

This work was supported by the Horizon Europe project CIRCULAR FUELS (grant agreement
No. 101118239), funded by the European Union.

## Citation

If you use this code, please cite the associated paper (citation details to be added
upon publication).

## License

See [LICENSE](LICENSE).
