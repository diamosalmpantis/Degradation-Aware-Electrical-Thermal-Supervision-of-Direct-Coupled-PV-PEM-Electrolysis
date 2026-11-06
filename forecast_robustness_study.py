# -*- coding: utf-8 -*-
"""How much of each controller's benefit survives a real forecast?"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import New_RL_1MW_final as m                                     # noqa: E402
from stable_baselines3 import PPO                                # noqa: E402

MODES = ('perfect', 'persistence', 'noisy')
RESULTS_ROOT = os.path.join(_HERE, 'Results_1MW_daily', 'Daily_demand')


def find_workbook(name='Irradiance_values_example_new.xlsx'):
    d = _HERE
    for _ in range(6):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
        d = os.path.dirname(d)
    raise FileNotFoundError(name)


def load_site(site):
    m.EXCEL_FILE = find_workbook()
    data = m.load_data_from_excel(m.EXCEL_FILE, site)
    return ([x['irradiance'] for x in data],
            [x['temperature'] for x in data],
            [x['price_mwh'] for x in data])


def model_path(site):
    return os.path.join(RESULTS_ROOT, 'Results_%s' % site, 'Model',
                        'rl_model_%s.zip' % site)


def summarise(df, label, mode):
    """Pull the comparable quantities out of one controller-year."""
    if 'fl' in df.columns:
        fl = pd.to_numeric(df['fl'], errors='coerce').fillna(0.0).to_numpy()
    else:
        pw = pd.to_numeric(df['P_pem_actual'], errors='coerce').fillna(0.0).to_numpy()
        fl = np.clip(pw / m.PEM_capacity, 0.0, 1.0)
    prev = np.concatenate(([0.0], fl[:-1]))
    restart = (fl >= 0.05) & (prev < 0.05)

    T = pd.to_numeric(df.get('T_pem', pd.Series(dtype=float)),
                      errors='coerce').to_numpy()
    T_prev = np.concatenate(([np.nan], T[:-1])) - 273.15
    cold = int(np.sum(restart & (T_prev < m.COLDSTART_THRESHOLD_C)))
    warm = int(np.sum(restart)) - cold
    mean_T = float(np.nanmean(T_prev[restart])) if restart.any() else float('nan')

    soh = pd.to_numeric(df['SoH_PEM'], errors='coerce').to_numpy()
    loss = float(soh[0] - soh[-1])
    life = (1.0 - m.LCOH_PEM_EOL_SOH) / loss if loss > 0 else float('inf')

    h2 = float(pd.to_numeric(df['H2_prod'], errors='coerce').sum())

    gb = pd.to_numeric(df.get('grid_bought', pd.Series(dtype=float)),
                       errors='coerce').fillna(0.0).to_numpy()
    grid_mwh = float(gb.sum()) * float(m.DT_HOURS) / 1e6

    npar = pd.to_numeric(df.get('n_parallel', pd.Series(dtype=float)),
                         errors='coerce').ffill().to_numpy()
    switches = int(np.sum(np.abs(np.diff(npar)) > 0.5)) if npar.size > 1 else 0

    return dict(controller=label, forecast=mode, H2_kg=h2,
                grid_MWh=grid_mwh, switches=switches,
                cold_starts=cold, warm_starts=warm, mean_restart_T=mean_T,
                SoH_end=float(soh[-1]), SoH_loss=loss, life_yr=life)


def run_site(site):
    print('\n' + '=' * 78)
    print('  FORECAST ROBUSTNESS  --  %s' % site)
    print('=' * 78)

    mp = model_path(site)
    if not os.path.isfile(mp):
        print('  !! no trained model at %s' % mp)
        print('     run New_RL_1MW_final.py for this site first; skipping.')
        return None

    ratio, tank, batt = m.SITE_SIZING[site]
    m.set_string_ladder(ratio)
    m.BATT_CAPACITY_WH       = float(batt) * 1000.0
    m.BATT_MAX_P_W           = m.BATT_CAPACITY_WH
    m.storage_capacity       = float(tank)
    m.LCOH_TANK_KG_INSTALLED = float(tank)
    if hasattr(m, 'set_initial_storage'):
        m.set_initial_storage()
    print('  hardware: PV %.2fx, tank %d kg, battery %d kWh' % (ratio, tank, batt))

    irr, temp, price = load_site(site)
    model = PPO.load(mp)
    print('  loaded frozen policy: %s' % os.path.basename(mp))

    pv_peak_kw = m.N_PAR_MAX * m.N_SERIES_PV * m.PV_MODULE_W / 1000.0

    rows = []
    for mode in MODES:
        m.FORECAST_MODE = mode
        m.reset_forecast_rng()
        print('\n  --- forecast = %s ---' % mode)
        frames = {}

        for label, kw in (('MOO', dict(hysteresis=False)),
                          ('MOO+Hyst', dict(hysteresis=True)),
                          ('MOO+PreWarm', dict(hysteresis=False,
                                               prewarm='anticipatory')),
                          ('MOO+Hyst+PreWarm', dict(hysteresis=True,
                                                    prewarm='anticipatory')),
                          ('MOO+AlwaysWarm', dict(hysteresis=False,
                                                  prewarm='always_warm'))):
            m.reset_forecast_rng()
            df = m.run_baselines(irr, temp, price, use_rf=False, **kw)
            if isinstance(df, tuple):
                df = df[0]
            frames[label] = df
            rows.append(summarise(df, label, mode))
            print('      %-18s H2 %7.0f  cold %4d  SoH %.3f'
                  % (label, rows[-1]['H2_kg'], rows[-1]['cold_starts'],
                     rows[-1]['SoH_end']))

        m.reset_forecast_rng()
        df_rl = m.evaluate_rl_single_run(model, irr, temp, price)
        frames['RL Agent'] = df_rl
        rows.append(summarise(df_rl, 'RL Agent', mode))
        print('      %-18s H2 %7.0f  cold %4d  SoH %.3f'
              % ('RL Agent', rows[-1]['H2_kg'], rows[-1]['cold_starts'],
                 rows[-1]['SoH_end']))

        try:
            eco = m.calculate_lcoh(
                frames['MOO'], None, frames['RL Agent'], pv_peak_kw,
                df_hyst=frames.get('MOO+Hyst'),
                df_pen=frames.get('MOO+AlwaysWarm'),
                df_pw=frames.get('MOO+PreWarm'),
                df_hpw=frames.get('MOO+Hyst+PreWarm'))
        except Exception as exc:                              # noqa: BLE001
            print('      [LCOH unavailable for this mode: %s]' % exc)
            eco = {}
        for r in rows:
            if r['forecast'] != mode:
                continue
            e = eco.get(r['controller'])
            if e is None and r['controller'] == 'MOO+AlwaysWarm':
                e = eco.get('MOO+Pen')
            e = e or {}
            r['LCOH'] = float(e.get('lcoh', float('nan'))) if e else float('nan')
            r['REPEX_ann'] = float(e.get('repex_ann', float('nan'))) if e else float('nan')

    m.FORECAST_MODE = 'perfect'
    return pd.DataFrame(rows)


def report(df, site):
    print('\n' + '=' * 78)
    print('  COLD STARTS PER YEAR  --  %s   (the quantity the forecast decides)' % site)
    print('=' * 78)
    piv = df.pivot(index='controller', columns='forecast', values='cold_starts')
    piv = piv.reindex(columns=[c for c in MODES if c in piv.columns])
    print(piv.to_string())

    print('\n  PROJECTED STACK LIFE (yr)')
    pl = df.pivot(index='controller', columns='forecast', values='life_yr')
    print(pl.reindex(columns=[c for c in MODES if c in pl.columns]).round(2).to_string())

    print('\n  HYDROGEN (kg/yr)')
    ph = df.pivot(index='controller', columns='forecast', values='H2_kg')
    print(ph.reindex(columns=[c for c in MODES if c in ph.columns]).round(0).to_string())

    if 'LCOH' in df.columns and df['LCOH'].notna().any():
        print('\n  LCOH (EUR/kg)')
        pc = df.pivot(index='controller', columns='forecast', values='LCOH')
        print(pc.reindex(columns=[c for c in MODES if c in pc.columns])
                .round(2).to_string())

    print('\n  GRID IMPORT (MWh/yr)  -- what warm standby costs')
    pg = df.pivot(index='controller', columns='forecast', values='grid_MWh')
    print(pg.reindex(columns=[c for c in MODES if c in pg.columns]).round(1).to_string())

    if 'perfect' in piv.columns and 'persistence' in piv.columns:
        print('\n  DEGRADATION FROM PERFECT -> PERSISTENCE (cold starts added)')
        d = (piv['persistence'] - piv['perfect']).sort_values(ascending=False)
        for k, v in d.items():
            print('    %-18s %+6d' % (k, v))
        print('\n  The controller that gains fewest cold starts is the one whose')
        print('  restart protection did not depend on being told the future.')


def main():
    sites = sys.argv[1:] or list(m.SITE_SIZING.keys())
    allout = []
    for s in sites:
        if s not in m.SITE_SIZING:
            print('unknown site %r; known: %s' % (s, ', '.join(m.SITE_SIZING)))
            continue
        d = run_site(s)
        if d is None:
            continue
        d['site'] = s
        report(d, s)
        allout.append(d)

    if allout:
        out = pd.concat(allout, ignore_index=True)
        p = os.path.join(RESULTS_ROOT, 'forecast_robustness.csv')
        os.makedirs(RESULTS_ROOT, exist_ok=True)
        out.to_csv(p, index=False)
        print('\n  saved: %s' % p)


if __name__ == '__main__':
    main()
