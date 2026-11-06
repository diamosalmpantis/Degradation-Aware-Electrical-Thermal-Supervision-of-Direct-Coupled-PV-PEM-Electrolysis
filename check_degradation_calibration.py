# -*- coding: utf-8 -*-
"""Degradation calibration check -- run this BEFORE retraining."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import New_RL_1MW_final as m

SITE = sys.argv[1] if len(sys.argv) > 1 else 'Prague_2021'
UV_BAND = (20.0, 50.0)          # Maoulida et al., frequent ON/OFF cycling


def apply_site(label):
    """Reproduce the per-site hardware the main script applies in its location loop.
    """
    m.set_string_ladder(1.08)
    sz = m.site_sizing_for(label)
    if sz is None:
        raise SystemExit("No SITE_SIZING entry for %r. Known: %s"
                         % (label, list(m.SITE_SIZING)))
    ratio, tank, batt = sz
    m.set_string_ladder(ratio)
    m.storage_capacity = float(tank)
    m.LCOH_TANK_KG_INSTALLED = float(tank)
    m.set_initial_storage()
    m.BATT_CAPACITY_WH = float(batt) * 1000.0
    m.BATT_MAX_P_W = m.BATT_CAPACITY_WH * 1.0
    return ratio, tank, batt


def split(df, label):
    """Per-mechanism annual split, recomputed from the simulated trajectory."""
    k = m.soh_per_uV()
    n = len(df)
    years = max(n * float(m.DT_HOURS) / 8760.0, 1e-12)

    fl = pd.to_numeric(df['fl'], errors='coerce').fillna(0.0).to_numpy()
    T = pd.to_numeric(df.get('T_pem', pd.Series(np.full(n, m.T_DEG_REF_K))),
                      errors='coerce').fillna(m.T_DEG_REF_K).to_numpy()
    if np.nanmax(T) < 200.0:                       # stored in Celsius
        T = T + 273.15

    prev_fl = np.concatenate(([0.0], fl[:-1]))
    prev_T = np.concatenate(([T[0]], T[:-1]))

    producing = fl >= 0.05
    op_hours = producing.sum() * float(m.DT_HOURS) / years

    j = m.PEM_J_NOM_A_CM2 * np.clip(fl, 0.0, 1.0)
    mu_dyn = m.MU_DYN_SLOPE * j + m.MU_DYN_INTERCEPT
    mu_dyn = np.where(fl < getattr(m, 'PEM_IDLE_FL_MIN', 0.05),
                      mu_dyn * getattr(m, 'MU_IDLE_FRAC', 0.0), mu_dyn)
    mu_ramp = m.MU_RAMP_FACTOR * mu_dyn * np.abs(fl - prev_fl) / float(m.DT_HOURS)
    f_T = np.array([m.pem_thermal_level_factor(t) for t in T])
    m1 = float(((mu_dyn + mu_ramp) * f_T).sum()) * float(m.DT_HOURS) / years

    dT_exc = np.maximum(0.0, np.abs(T - prev_T) - m.TCYC_DEADBAND_K)
    m2 = float(m.MU_TCYC_UV_PER_K * dT_exc.sum()) / years

    restart = producing & (prev_fl < 0.05)
    cold = restart & (prev_T < (m.COLDSTART_THRESHOLD_C + 273.15))
    n_cold, n_warm = int(cold.sum()), int((restart & ~cold).sum())
    m3 = (n_cold * m.MU_SD_COLD_UV + n_warm * m.MU_SD_WARM_UV) / years

    soh_end = float(pd.to_numeric(df['SoH_PEM'], errors='coerce').dropna().iloc[-1])
    realised = (1.0 - soh_end) / k / years
    total = m1 + m2 + m3
    eff = total / op_hours if op_hours else float('nan')
    budget = (1.0 - m.LCOH_PEM_EOL_SOH) / k

    print('\n%s' % label)
    print('  restarts/yr        %6.0f cold  %6.0f warm' % (n_cold / years, n_warm / years))
    print('  operating h/yr     %6.0f' % op_hours)
    print('  mean f_T           %6.3f   (1.000 = cooling loop holds 70 C)' % f_T.mean())
    print('  M1 steady       %9.0f uV/yr  %5.1f %%' % (m1, 100 * m1 / total))
    print('  M2 thermal cyc  %9.0f uV/yr  %5.1f %%' % (m2, 100 * m2 / total))
    print('  M3 on/off       %9.0f uV/yr  %5.1f %%' % (m3, 100 * m3 / total))
    print('  TOTAL           %9.0f uV/yr  (%.1f %% of the %.0f uV EoL budget)'
          % (total, 100 * total / budget, budget))
    print('  realised in sim %9.0f uV/yr  (cross-check, should match TOTAL)' % realised)
    print('  effective rate  %9.1f uV/h   [literature band %.0f-%.0f]'
          % (eff, UV_BAND[0], UV_BAND[1]))
    print('  projected life  %9.2f yr' % ((1.0 - m.LCOH_PEM_EOL_SOH) / (realised * k))
          if realised > 0 else '  projected life        inf')
    return eff, m2, total


print('=' * 68)
print('  DEGRADATION CALIBRATION CHECK  --  site %s' % SITE)
print('=' * 68)
print('\n--- model constants ---')
m.degradation_budget_check()

ratio, tank, batt = apply_site(SITE)
print('\n  site hardware: PV %gx, tank %g kg, battery %g kWh' % (ratio, tank, batt))

def find_workbook():
    """EXCEL_FILE is pinned next to the module; look up the tree if it moved."""
    name = os.path.basename(m.EXCEL_FILE)
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(5):
        cand = os.path.join(here, *(['..'] * up), name)
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    raise SystemExit('Cannot find %s near %s' % (name, here))


m.EXCEL_FILE = find_workbook()
data = m.load_data_from_excel(m.EXCEL_FILE, SITE)
irr = [x['irradiance'] for x in data]
temp = [x['temperature'] for x in data]
price = [x['price_mwh'] for x in data]

print('\n--- running baselines (no training) ---')
eff_greedy, m2_greedy, tot_greedy = split(
    m.run_baselines(irr, temp, price, use_rf=False, hysteresis=False), 'greedy MOO')
eff_pre, m2_pre, tot_pre = split(
    m.run_baselines(irr, temp, price, use_rf=False, hysteresis=False,
                    prewarm='anticipatory'), 'MOO + PreWarm')

print('\n' + '=' * 68)
print('  VERDICT')
print('=' * 68)
lo, hi = UV_BAND
if eff_greedy > hi:
    k_new = m.MU_TCYC_UV_PER_K - (tot_greedy - hi * tot_greedy / eff_greedy) / max(
        m2_greedy / max(m.MU_TCYC_UV_PER_K, 1e-9), 1e-9)
    print('  Greedy MOO at %.1f uV/h is ABOVE the band. Lower MU_TCYC_UV_PER_K'
          ' towards %.3f and rerun.' % (eff_greedy, max(k_new, 0.0)))
elif eff_greedy < lo:
    print('  Greedy MOO at %.1f uV/h sits BELOW the %.0f uV/h band edge.' % (eff_greedy, lo))
    print('  That is the conservative side and is acceptable; raising'
          ' MU_TCYC_UV_PER_K to about %.3f would reach it.'
          % (m.MU_TCYC_UV_PER_K * (1 + (lo * (tot_greedy / eff_greedy) - tot_greedy)
                                   / max(m2_greedy, 1e-9))))
else:
    print('  Greedy MOO at %.1f uV/h is inside the %.0f-%.0f uV/h band. '
          'MU_TCYC_UV_PER_K = %.3f stands.' % (eff_greedy, lo, hi, m.MU_TCYC_UV_PER_K))
print('  Thermal management separates the two controllers by %.0f uV/yr (%.1f %%).'
      % (tot_greedy - tot_pre, 100 * (tot_greedy - tot_pre) / max(tot_greedy, 1e-9)))
print('  M2 share: greedy %.1f %%, prewarm %.1f %%  (design intent: subordinate to M3)'
      % (100 * m2_greedy / max(tot_greedy, 1e-9), 100 * m2_pre / max(tot_pre, 1e-9)))
