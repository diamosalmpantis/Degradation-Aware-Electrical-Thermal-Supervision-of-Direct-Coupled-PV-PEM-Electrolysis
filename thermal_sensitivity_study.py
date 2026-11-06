# -*- coding: utf-8 -*-
"""Sensitivity of the result to the thermal model itself."""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import New_RL_1MW_final as m                                     # noqa: E402
from stable_baselines3 import PPO                                # noqa: E402

RESULTS_ROOT = os.path.join(_HERE, 'Results_1MW_daily', 'Daily_demand')

CASES = [
    ('baseline',      None,                      None),
    ('U_A -30%',      'THERMAL_UA_W_K',          0.70),
    ('U_A +30%',      'THERMAL_UA_W_K',          1.30),
    ('m_Cp -30%',     'THERMAL_MCP_J_K',         0.70),
    ('m_Cp +30%',     'THERMAL_MCP_J_K',         1.30),
    ('COP -30%',      'CHILLER_COP',             0.70),
    ('COP +30%',      'CHILLER_COP',             1.30),
    ('COP lift-aware', 'CHILLER_COP_LIFT_AWARE', True),
]


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


def apply_hardware(site):
    ratio, tank, batt = m.SITE_SIZING[site]
    m.set_string_ladder(ratio)
    m.BATT_CAPACITY_WH       = float(batt) * 1000.0
    m.BATT_MAX_P_W           = m.BATT_CAPACITY_WH
    m.storage_capacity       = float(tank)
    m.LCOH_TANK_KG_INSTALLED = float(tank)
    if hasattr(m, 'set_initial_storage'):
        m.set_initial_storage()
    return ratio


def repex_annual(life_yr, pem_capex=None, wacc=None, horizon=None):
    """Discrete stack replacements -> NPV -> annualised."""
    pem_capex = 1200000.0 if pem_capex is None else pem_capex
    r = m.LCOH_WACC if wacc is None else wacc
    n = m.LCOH_PROJECT_LIFE if horizon is None else horizon
    if not np.isfinite(life_yr) or life_yr <= 0:
        return 0.0, 0
    years, j = [], 1
    while j * life_yr < n:
        years.append(j * life_yr)
        j += 1
    npv = sum(pem_capex / (1.0 + r) ** y for y in years)
    crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
    return npv * crf, len(years)


def summarise(df, label, case):
    if 'fl' in df.columns:
        fl = pd.to_numeric(df['fl'], errors='coerce').fillna(0.0).to_numpy()
    else:
        pw = pd.to_numeric(df['P_pem_actual'], errors='coerce').fillna(0.0).to_numpy()
        fl = np.clip(pw / m.PEM_capacity, 0.0, 1.0)
    prev = np.concatenate(([0.0], fl[:-1]))
    restart = (fl >= 0.05) & (prev < 0.05)

    T = pd.to_numeric(df['T_pem'], errors='coerce').to_numpy()
    T_prev = np.concatenate(([np.nan], T[:-1])) - 273.15
    cold = int(np.sum(restart & (T_prev < m.COLDSTART_THRESHOLD_C)))

    soh = pd.to_numeric(df['SoH_PEM'], errors='coerce').to_numpy()
    loss = float(soh[0] - soh[-1])
    life = (1.0 - m.LCOH_PEM_EOL_SOH) / loss if loss > 0 else float('inf')
    h2 = float(pd.to_numeric(df['H2_prod'], errors='coerce').sum())
    gb = pd.to_numeric(df.get('grid_bought', pd.Series(dtype=float)),
                       errors='coerce').fillna(0.0).to_numpy()
    return dict(controller=label, case=case, cold_starts=cold, SoH_loss=loss,
                life_yr=life, H2_kg=h2,
                grid_MWh=float(gb.sum()) * float(m.DT_HOURS) / 1e6)


CONTROLLERS = (('MOO', dict(hysteresis=False)),
               ('MOO+Hyst', dict(hysteresis=True)),
               ('MOO+PreWarm', dict(hysteresis=False, prewarm='anticipatory')),
               ('MOO+Hyst+PreWarm', dict(hysteresis=True, prewarm='anticipatory')),
               ('MOO+AlwaysWarm', dict(hysteresis=False, prewarm='always_warm')))


def run_site(site):
    print('\n' + '=' * 78)
    print('  THERMAL SENSITIVITY  --  %s' % site)
    print('=' * 78)
    mp = os.path.join(RESULTS_ROOT, 'Results_%s' % site, 'Model',
                      'rl_model_%s.zip' % site)
    if not os.path.isfile(mp):
        print('  !! no trained model at %s -- run New_RL_1MW_final.py first.' % mp)
        return None

    apply_hardware(site)
    irr, temp, price = load_site(site)
    model = PPO.load(mp)

    base = {'THERMAL_UA_W_K': m.THERMAL_UA_W_K,
            'THERMAL_MCP_J_K': m.THERMAL_MCP_J_K,
            'CHILLER_COP': m.CHILLER_COP,
            'CHILLER_COP_LIFT_AWARE': m.CHILLER_COP_LIFT_AWARE}

    rows = []
    for case, attr, val in CASES:
        for k, v in base.items():
            setattr(m, k, v)
        if attr is not None:
            setattr(m, attr, val if isinstance(val, bool) else base[attr] * val)
        m._V_DEG_REF = None                      # force the thermal reference to re-derive
        print('\n  --- %s ---' % case)

        for label, kw in CONTROLLERS:
            df = m.run_baselines(irr, temp, price, use_rf=False, **kw)
            if isinstance(df, tuple):
                df = df[0]
            rows.append(summarise(df, label, case))
            print('      %-18s cold %4d  life %6.2f yr'
                  % (label, rows[-1]['cold_starts'], rows[-1]['life_yr']))

        df_rl = m.evaluate_rl_single_run(model, irr, temp, price)
        rows.append(summarise(df_rl, 'RL Agent', case))
        print('      %-18s cold %4d  life %6.2f yr'
              % ('RL Agent', rows[-1]['cold_starts'], rows[-1]['life_yr']))

    for k, v in base.items():                    # restore
        setattr(m, k, v)
    m._V_DEG_REF = None

    d = pd.DataFrame(rows)
    d['REPEX_ann'], d['n_repl'] = zip(*[repex_annual(x) for x in d['life_yr']])
    d['site'] = site
    return d


def report(d, site):
    order = [c for c, _ in CONTROLLERS] + ['RL Agent']
    print('\n' + '=' * 78)
    print('  PART A -- PROJECTED STACK LIFE (yr) BY THERMAL CASE  --  %s' % site)
    print('=' * 78)
    pv = d.pivot(index='controller', columns='case', values='life_yr')
    pv = pv.reindex(index=[c for c in order if c in pv.index],
                    columns=[c for c, _, _ in CASES if c in pv.columns])
    print(pv.round(2).to_string())

    print('\n  COLD STARTS BY THERMAL CASE')
    pc = d.pivot(index='controller', columns='case', values='cold_starts')
    print(pc.reindex(index=[c for c in order if c in pc.index],
                     columns=[c for c, _, _ in CASES if c in pc.columns]).to_string())

    print('\n  ANNUALISED REPLACEMENT COST (EUR/yr) -- what the ranking actually turns on')
    pr = d.pivot(index='controller', columns='case', values='REPEX_ann')
    print(pr.reindex(index=[c for c in order if c in pr.index],
                     columns=[c for c, _, _ in CASES if c in pr.columns])
            .round(0).to_string())

    print('\n  ORDERING BY REPLACEMENT COST (cheapest first)')
    flips = []
    ref = None
    for case in [c for c, _, _ in CASES if c in pr.columns]:
        o = list(pr[case].sort_values().index)
        print('    %-16s %s' % (case, ' < '.join(o)))
        if ref is None:
            ref = o
        elif o != ref:
            flips.append(case)
    print('\n  -> ordering %s' % ('UNCHANGED in every case' if not flips
                                  else 'CHANGES in: ' + ', '.join(flips)))

    # ---- PART B -------------------------------------------------------------
    print('\n' + '=' * 78)
    print('  PART B -- STACK-LIFE DISCONTINUITY (replacements are a step function)')
    print('=' * 78)
    b = d[d['case'] == 'baseline']
    print('  %-18s %8s %7s | %s' % ('controller', 'life', 'n_repl',
                                    'REPEX_ann at life -5% / base / +5%'))
    for _, r in b.iterrows():
        lo, nlo = repex_annual(r['life_yr'] * 0.95)
        mid, nmid = repex_annual(r['life_yr'])
        hi, nhi = repex_annual(r['life_yr'] * 1.05)
        flag = '  <== CROSSES A BOUNDARY' if (nlo != nmid or nhi != nmid) else ''
        print('  %-18s %8.2f %7d | %9.0f (%d) %9.0f (%d) %9.0f (%d)%s'
              % (r['controller'], r['life_yr'], nmid,
                 lo, nlo, mid, nmid, hi, nhi, flag))
    print('\n  A controller flagged above can gain or lose a whole stack from a few')
    print('  percent of SoH, so its LCOH must be reported with that spread.')


def main():
    sites = sys.argv[1:] or list(m.SITE_SIZING.keys())
    out = []
    for s in sites:
        if s not in m.SITE_SIZING:
            print('unknown site %r' % s)
            continue
        d = run_site(s)
        if d is None:
            continue
        report(d, s)
        out.append(d)
    if out:
        p = os.path.join(RESULTS_ROOT, 'thermal_sensitivity.csv')
        pd.concat(out, ignore_index=True).to_csv(p, index=False)
        print('\n  saved: %s' % p)


if __name__ == '__main__':
    main()
