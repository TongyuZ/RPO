"""
Extract RPO total / current / noncurrent from wide-format XBRL tag file.

Usage : python extract_rpo_from_tags.py rpo_sample_100.csv
Output: rpo_summary.csv (one row per filing), rpo_facts_long.csv (audit trail)

Definitions:
  rpo_total      = transaction price allocated to remaining performance obligations
  rpo_current    = portion expected to be recognized within ~12 months
  rpo_noncurrent = rpo_total - rpo_current
"""

import ast
import re
import sys
import pandas as pd
from datetime import timedelta

ID_COLS = ['CIK', 'Company', 'FiscalYear']
TIMING_AXIS = 'ExpectedTimingOfSatisfactionStartDateAxis'
C_TOTAL  = 'RevenueRemainingPerformanceObligation'
C_PERIOD = 'RevenueRemainingPerformanceObligationExpectedTimingOfSatisfactionPeriod1'
C_PCT_RE = re.compile(r'RevenueRemainingPerformanceObligation.*Percentage$', re.I)
# custom/extension tags meaning "RPO within 12 months"
C_CUR_RE = re.compile(r'RemainingPerformanceObligation.*(?:12|Twelve).?Months?', re.I)
TOL = timedelta(days=45)
YEAR = timedelta(days=366)


def parse_dims(s):
    """-> (timing_member, has_other_dims, other_axes_tuple, other_key)"""
    if not isinstance(s, str) or s.strip() in ('', '[]', 'nan'):
        return None, False, (), ''
    try:
        dims = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None, True, ('unparsed',), s
    timing, others = None, []
    for d in dims:
        if TIMING_AXIS in str(d.get('dimension', '')):
            timing = str(d.get('member', '')).strip()
        else:
            others.append((str(d.get('dimension', '')), str(d.get('member', ''))))
    axes = tuple(sorted(a for a, _ in others))
    key = ';'.join(f'{a}={m}' for a, m in sorted(others))
    return timing, bool(others), axes, key


def duration_days(p):
    m = re.match(r'P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?$', str(p).strip())
    if not m or not any(m.groups()):
        return None
    y, mo, d = (int(g) if g else 0 for g in m.groups())
    return y * 365 + mo * 30 + d


def to_float(v):
    try:
        return float(str(v).replace(',', ''))
    except (ValueError, TypeError):
        return None


def melt_tags(df):
    n_tags = max(int(m.group(1)) for c in df.columns
                 if (m := re.match(r'Tag_(\d+)_name$', c)))
    frames, idc = [], [c for c in ID_COLS if c in df.columns]
    fields = ['name', 'value_raw', 'unit', 'period_type',
              'instant', 'startDate', 'endDate', 'dimensions']
    for i in range(1, n_tags + 1):
        cols = {f'Tag_{i}_{f}': f for f in fields if f'Tag_{i}_{f}' in df.columns}
        if not cols:
            continue
        sub = df[idc + list(cols)].rename(columns=cols).dropna(subset=['name'])
        if len(sub):
            frames.append(sub)
    long = pd.concat(frames, ignore_index=True)
    long = long[long['name'].str.contains('RemainingPerformanceObligation', na=False)]
    long = long[~long['name'].str.contains('TextBlock|PracticalExpedient', na=False)]
    parsed = long['dimensions'].map(parse_dims)
    long['timing_member'] = parsed.map(lambda t: t[0])
    long['has_other_dims'] = parsed.map(lambda t: t[1])
    long['other_axes'] = parsed.map(lambda t: t[2])
    long['other_key'] = parsed.map(lambda t: t[3])
    long['value'] = long['value_raw'].map(to_float)
    long['ref_date'] = long['instant'].fillna(long['endDate'])
    return long


def bucket_span_days(member, start, next_start, dur_map):
    """Bucket window in days. The tagged Period1 duration is authoritative,
    EXCEPT the classic filer typo 'PnY' meaning 'n months' (detected when the
    next bucket starts ~n months later). Without Period1, use bucket spacing."""
    spacing = (next_start - start).days if next_start is not None else None
    p = str(dur_map.get(member, '')).strip()
    dd = duration_days(p)
    if dd is None:
        return spacing, ''
    m = re.match(r'P(\d+)Y$', p)
    if m and spacing is not None and abs(spacing - int(m.group(1)) * 30.4) <= 75:
        return spacing, 'period1_year_month_typo_fixed'     # e.g. P12Y next to a 12-month gap
    return dd, ''


def walk_buckets(b, dur_map):
    """b: DataFrame with timing_member + value, consecutive timing buckets.
    Anchor the 12m cutoff at the FIRST bucket's start (robust to filer date errors).
    Returns (current, remaining_sum, n, flag)."""
    b = b.dropna(subset=['value']).copy()
    b['start'] = pd.to_datetime(b['timing_member'], errors='coerce')
    b = b.dropna(subset=['start']).sort_values('start')
    n = len(b)
    if n == 0:
        return None, None, 0, 'unparsed_bucket_dates'
    starts = list(b['start'])
    vals = list(b['value'])
    members = list(b['timing_member'])
    cutoff = starts[0] + YEAR
    cur, i, flags = 0.0, 0, set()
    while i < n:
        nxt = starts[i + 1] if i + 1 < n else None
        span, fl = bucket_span_days(members[i], starts[i], nxt, dur_map)
        if fl:
            flags.add(fl)
        end = starts[i] + timedelta(days=span) if span else None
        if end is not None and end <= cutoff + TOL:
            cur += vals[i]; i += 1; continue
        if end is None:
            if i == 0:
                # single bucket, unknown span: assume the customary <=12m disclosure
                return vals[0], sum(vals[1:]), n, 'assumed_first_bucket_12m'
            break                                  # remaining are noncurrent
        if starts[i] >= cutoff - TOL:
            break                                  # clean cut: rest is noncurrent
        flags.add('first_bucket_spans_gt_12m')     # e.g. P24M / P5Y first bucket
        return None, None, n, '|'.join(sorted(flags))
    return cur, sum(vals[i:]), n, '|'.join(sorted(flags))


def pick_as_of(dates, fy):
    """Choose the reporting date anchored to FiscalYear, never blindly max().
    Priority: latest date IN the fiscal year > latest date BEFORE it (flag)
    > latest date after/future (stronger flag). Guards against filer typos
    and percentage facts tagged with future recognition-window contexts."""
    ds = sorted({d for d in dates if isinstance(d, str) and len(d) >= 4})
    if not ds:
        return None, ''
    if fy is None:
        return ds[-1], ''
    same = [d for d in ds if int(d[:4]) == fy]
    if same:
        return same[-1], ''
    past = [d for d in ds if int(d[:4]) < fy]
    if past:
        return past[-1], 'as_of_before_fiscalyear'
    return ds[-1], 'as_of_after_fiscalyear'


def summarize(g):
    out = {'as_of_date': None, 'rpo_total': None, 'rpo_current': None,
           'rpo_noncurrent': None, 'pct_within_12m': None,
           'n_timing_buckets': 0, 'method': None, 'flag': []}

    amt = g[(g['name'] == C_TOTAL) & ~g['has_other_dims']]
    totals = amt[amt['timing_member'].isna()].dropna(subset=['value'])
    seg = g[(g['name'] == C_TOTAL) & g['has_other_dims'] &
            g['timing_member'].isna()].dropna(subset=['value'])
    ref = pd.concat([totals['ref_date'], amt['ref_date'], seg['ref_date']]).dropna()
    if ref.empty:
        ref = g['ref_date'].dropna()
        if ref.empty:
            out['flag'] = ''
            return out
    try:
        fy = int(float(g['FiscalYear'].iloc[0])) if 'FiscalYear' in g else None
    except (ValueError, TypeError):
        fy = None
    inst, fl = pick_as_of(ref, fy)
    if fl:
        out['flag'].append(fl)
    out['as_of_date'] = inst

    if (totals['ref_date'] == inst).any():
        out['rpo_total'] = totals.loc[totals['ref_date'] == inst, 'value'].iloc[0]
        out['method'] = 'reported_total'
    else:
        # fallback: sum RPO disclosed per segment/product (one shared axis,
        # distinct members, same date) when no consolidated total is tagged
        s = seg[seg['ref_date'] == inst]
        if len(s) and s['other_axes'].nunique() == 1 \
                and s['other_key'].is_unique and 'unparsed' not in s['other_axes'].iloc[0]:
            out['rpo_total'] = s['value'].sum()
            out['method'] = 'sum_of_segments'
            out['flag'].append('total_summed_across_segments')

    per = g[(g['name'] == C_PERIOD) & (g['ref_date'] == inst)]
    dur_map = dict(zip(per['timing_member'], per['value_raw']))

    # ---- dollar timing buckets
    bk = amt[amt['timing_member'].notna() & (amt['ref_date'] == inst)]
    if not bk.empty:
        cur, rest, n, fl = walk_buckets(bk, dur_map)
        out['n_timing_buckets'] = n
        if fl:
            out['flag'].append(fl)
        if out['rpo_total'] is None and cur is not None:
            out['rpo_total'] = cur + rest
            out['method'] = 'sum_of_buckets'
        if cur is not None:
            out['rpo_current'] = cur
            out['method'] += '+dollar_buckets'

    # ---- custom "within 12 months" extension tags (e.g. ...Obligation12months)
    if out['rpo_current'] is None:
        cust = g[g['name'].str.contains(C_CUR_RE, na=False) & g['value'].notna() &
                 g['unit'].astype(str).str.contains('USD', na=False) &
                 (g['ref_date'] == inst)]
        if not cust.empty:
            out['rpo_current'] = cust['value'].iloc[0]
            out['method'] = (out['method'] or '') + '+custom_12m_tag'

    # ---- percentage buckets
    pc = g[g['name'].str.match(C_PCT_RE) & g['timing_member'].notna() &
           ~g['has_other_dims']].dropna(subset=['value']).copy()
    if not pc.empty:
        exact = pc[pc['ref_date'] == inst]
        if len(exact):
            pc = exact
        else:
            past = pc[pc['ref_date'] <= inst]['ref_date'].dropna()
            if len(past):
                pc = pc[pc['ref_date'] == past.max()]
                out['flag'].append('pct_ref_date_off_as_of')
        scale = 100 if pc['value'].sum() <= 1.5 else 1
        pc['value'] = pc['value'] * scale
        cur_p, _, _, fl = walk_buckets(pc, dur_map)
        if cur_p is not None and 0 < cur_p <= 100:
            out['pct_within_12m'] = round(cur_p, 2)
        elif fl:
            out['flag'].append('pct_' + fl)
        if out['rpo_current'] is None and out['rpo_total'] is not None \
                and out['pct_within_12m'] is not None:
            out['rpo_current'] = round(out['rpo_total'] * out['pct_within_12m'] / 100, 0)
            out['method'] = (out['method'] or '') + '+pct_derived'

    # ---- noncurrent = total - current (authoritative when total is reported)
    if out['rpo_total'] is not None and out['rpo_current'] is not None:
        out['rpo_noncurrent'] = out['rpo_total'] - out['rpo_current']
        if out['rpo_noncurrent'] < 0:
            out['flag'].append('current_exceeds_total')

    # ---- sanity flags
    if out['rpo_total'] is not None and out['rpo_total'] >= 1e11:
        out['flag'].append('check_magnitude')
    out['flag'] = '|'.join(out['flag'])
    return out


def main(path):
    df = pd.read_csv(path, dtype=str)
    long = melt_tags(df)
    long.to_csv('rpo_facts_long.csv', index=False)

    keys = [c for c in ID_COLS if c in long.columns]
    rows = []
    for kv, g in long.groupby(keys, dropna=False):
        rec = dict(zip(keys, kv if isinstance(kv, tuple) else (kv,)))
        rec.update(summarize(g))
        rows.append(rec)
    summary = df[keys].drop_duplicates().merge(pd.DataFrame(rows), on=keys, how='left')
    summary.to_csv('rpo_summary.csv', index=False)

    print(f'{len(summary)} filings -> rpo_summary.csv')
    print(f"  with rpo_total     : {summary['rpo_total'].notna().sum()}")
    print(f"  with rpo_current   : {summary['rpo_current'].notna().sum()}")
    print(f"  with rpo_noncurrent: {summary['rpo_noncurrent'].notna().sum()}")
    print('  flags:', summary['flag'].value_counts(dropna=False).to_dict())


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'rpo_sample_100.csv')
