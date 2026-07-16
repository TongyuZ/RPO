"""
Extract RPO total and cumulative recognition horizons from a wide-format XBRL tag file.

Usage:
    python extract_rpo_horizons.py rpo_sample_100.csv

Outputs:
    rpo_summary_horizons.csv  one row per filing
    rpo_facts_long.csv        all retained RPO facts for audit

Main output columns:
    rpo_total
    rpo_within_12m
    rpo_within_24m
    rpo_within_36m
    ... automatically extended to the latest disclosed bucket
    rpo_thereafter

Important rule:
    Values are cumulative. For example, rpo_within_24m means the amount expected
    to be recognized from the reporting date through month 24.

    If the first disclosed bucket spans 24 months, rpo_within_12m remains missing
    and rpo_within_24m is populated. The program never divides a multi-year bucket
    evenly across years.
"""

import ast
import math
import re
import sys
from datetime import timedelta

import pandas as pd


ID_COLS = ["CIK", "Company", "FiscalYear"]
TIMING_AXIS = "ExpectedTimingOfSatisfactionStartDateAxis"
C_TOTAL = "RevenueRemainingPerformanceObligation"
C_PERIOD = "RevenueRemainingPerformanceObligationExpectedTimingOfSatisfactionPeriod1"
C_PCT_RE = re.compile(r"RevenueRemainingPerformanceObligation.*Percentage$", re.I)

# Extension tags such as RemainingPerformanceObligation12Months,
# RemainingPerformanceObligationWithinTwentyFourMonths, etc.
NUMBER_WORDS = {
    "twelve": 12,
    "twentyfour": 24,
    "thirtysix": 36,
    "fortyeight": 48,
    "sixty": 60,
    "seventytwo": 72,
    "eightyfour": 84,
    "ninetysix": 96,
    "onehundredeight": 108,
    "onehundredtwenty": 120,
}

TOL_DAYS = 45
DAYS_PER_MONTH = 365.25 / 12


def parse_dims(s):
    """Return timing member and information about all other dimensions."""
    if not isinstance(s, str) or s.strip() in ("", "[]", "nan"):
        return None, False, (), ""
    try:
        dims = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None, True, ("unparsed",), s

    timing, others = None, []
    for d in dims:
        dimension = str(d.get("dimension", ""))
        member = str(d.get("member", "")).strip()
        if TIMING_AXIS in dimension:
            timing = member
        else:
            others.append((dimension, member))

    axes = tuple(sorted(a for a, _ in others))
    key = ";".join(f"{a}={m}" for a, m in sorted(others))
    return timing, bool(others), axes, key


def duration_days(value):
    """Convert an ISO-8601 duration such as P12M, P2Y, or P30D to days."""
    match = re.fullmatch(r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?", str(value).strip())
    if not match or not any(match.groups()):
        return None
    years, months, days = (int(x) if x else 0 for x in match.groups())
    return years * 365.25 + months * DAYS_PER_MONTH + days


def to_float(value):
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


def melt_tags(df):
    tag_numbers = [
        int(match.group(1))
        for column in df.columns
        if (match := re.match(r"Tag_(\d+)_name$", column))
    ]
    if not tag_numbers:
        raise ValueError("No Tag_n_name columns were found in the input file.")

    frames = []
    id_cols = [column for column in ID_COLS if column in df.columns]
    fields = [
        "name", "value_raw", "unit", "period_type", "instant",
        "startDate", "endDate", "dimensions",
    ]

    for i in range(1, max(tag_numbers) + 1):
        rename = {
            f"Tag_{i}_{field}": field
            for field in fields
            if f"Tag_{i}_{field}" in df.columns
        }
        if "Tag_%d_name" % i not in rename:
            continue
        sub = df[id_cols + list(rename)].rename(columns=rename)
        sub = sub.dropna(subset=["name"])
        if not sub.empty:
            frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=id_cols + fields)

    long = pd.concat(frames, ignore_index=True)
    long = long[long["name"].str.contains("RemainingPerformanceObligation", na=False)]
    long = long[~long["name"].str.contains("TextBlock|PracticalExpedient", na=False)]

    for field in fields:
        if field not in long.columns:
            long[field] = None

    parsed = long["dimensions"].map(parse_dims)
    long["timing_member"] = parsed.map(lambda item: item[0])
    long["has_other_dims"] = parsed.map(lambda item: item[1])
    long["other_axes"] = parsed.map(lambda item: item[2])
    long["other_key"] = parsed.map(lambda item: item[3])
    long["value"] = long["value_raw"].map(to_float)
    long["ref_date"] = long["instant"].fillna(long["endDate"])
    long["ref_date_parsed"] = pd.to_datetime(long["ref_date"], errors="coerce")
    return long


def bucket_span_days(member, start, next_start, duration_map):
    """Determine the disclosed bucket length in days."""
    spacing = (next_start - start).days if next_start is not None else None
    duration = str(duration_map.get(member, "")).strip()
    days = duration_days(duration)

    if days is None:
        return spacing, ""

    # Some filers incorrectly use P12Y where bucket spacing clearly indicates 12 months.
    typo = re.fullmatch(r"P(\d+)Y", duration)
    if typo and spacing is not None:
        alleged_months = int(typo.group(1))
        if abs(spacing - alleged_months * DAYS_PER_MONTH) <= 75:
            return spacing, "period1_year_month_typo_fixed"

    return days, ""


def horizon_months_from_days(days):
    """Map a bucket endpoint to the nearest 12-month cumulative horizon."""
    if days is None or days <= 0:
        return None
    raw_months = days / DAYS_PER_MONTH
    nearest_year = max(1, int(round(raw_months / 12)))
    return nearest_year * 12


def amount_horizons(bucket_df, duration_map, as_of):
    """Convert mutually exclusive dollar intervals into cumulative horizons."""
    b = bucket_df.dropna(subset=["value"]).copy()
    b["start"] = pd.to_datetime(b["timing_member"], errors="coerce")
    b = b.dropna(subset=["start"]).sort_values("start")
    if b.empty:
        return {}, None, 0, ["unparsed_bucket_dates"]

    interval_rows, flags = [], set()
    starts = list(b["start"])
    for i, row in b.reset_index(drop=True).iterrows():
        start_date = row["start"]
        next_start = starts[i + 1] if i + 1 < len(starts) else None
        span, flag = bucket_span_days(row["timing_member"], start_date, next_start, duration_map)
        if flag:
            flags.add(flag)
        if span is None:
            flags.add("undated_final_bucket_excluded_from_horizons")
            continue
        end_date = start_date + timedelta(days=float(span))
        elapsed = (end_date - as_of).days
        horizon = horizon_months_from_days(elapsed)
        if horizon is None:
            flags.add("invalid_bucket_horizon")
            continue
        interval_rows.append((horizon, float(row["value"])))

    cumulative, horizons = 0.0, {}
    for horizon, value in sorted(interval_rows):
        cumulative += value
        horizons[horizon] = cumulative
    return horizons, b["value"].sum(), len(b), sorted(flags)


def percentage_horizons(percent_df, period_facts, as_of):
    """Interpret percentage facts as mutually exclusive recognition intervals.

    Example:
      52% starting at the report date for P24M -> interval 0-24 months
      17% starting 24 months later for P12M -> interval 24-36 months

    Output is cumulative: within_24m=52 and within_36m=69.
    """
    p = percent_df.dropna(subset=["value"]).copy()
    p["start"] = pd.to_datetime(p["timing_member"], errors="coerce")
    p = p.dropna(subset=["start"])
    if p.empty:
        return {}, 0, ["unparsed_percentage_dates"]

    # Normalize proportions stored as 0.52/0.17 to percentage points.
    if p["value"].abs().max() <= 1.0 + 1e-9:
        p["pct"] = p["value"] * 100
    else:
        p["pct"] = p["value"]

    pf = period_facts.copy()
    pf["start"] = pd.to_datetime(pf["timing_member"], errors="coerce")
    pf = pf.dropna(subset=["start"])
    duration_lists = (
        pf.groupby("start")["value_raw"]
          .apply(lambda x: [str(v).strip() for v in x if pd.notna(v)])
          .to_dict()
    )

    rows, flags = [], set()
    for _, row in p.sort_values("start").iterrows():
        start_date = row["start"]
        durations = duration_lists.get(start_date, [])
        valid = [(d, duration_days(d)) for d in durations]
        valid = [(d, days) for d, days in valid if days is not None]

        if len(valid) == 0:
            flags.add("missing_percentage_period")
            continue
        if len(valid) > 1:
            # Prefer the duration that gives a clean annual endpoint from report date.
            scored = []
            for d, days in valid:
                elapsed = (start_date - as_of).days + days
                months = elapsed / DAYS_PER_MONTH
                score = abs(months / 12 - round(months / 12))
                scored.append((score, d, days))
            scored.sort()
            _, chosen_duration, span_days = scored[0]
            flags.add("multiple_percentage_periods_best_match_used")
        else:
            chosen_duration, span_days = valid[0]

        end_date = start_date + timedelta(days=float(span_days))
        elapsed_days = (end_date - as_of).days
        horizon = horizon_months_from_days(elapsed_days)
        if horizon is None:
            flags.add("invalid_percentage_horizon")
            continue

        expected_days = horizon * DAYS_PER_MONTH
        if abs(elapsed_days - expected_days) > TOL_DAYS:
            flags.add(f"nonannual_percentage_endpoint_{round(elapsed_days / DAYS_PER_MONTH)}m")

        rows.append({
            "start": start_date,
            "horizon": horizon,
            "pct": float(row["pct"]),
            "duration": chosen_duration,
        })

    if not rows:
        return {}, 0, sorted(flags)

    intervals = pd.DataFrame(rows).drop_duplicates(
        subset=["start", "horizon", "pct", "duration"]
    )

    # If exact duplicate tags remain after dimension filtering, keep one.
    cumulative, horizons = 0.0, {}
    for horizon, grp in intervals.groupby("horizon", sort=True):
        cumulative += grp["pct"].sum()
        horizons[int(horizon)] = cumulative

    if cumulative > 100 + 1e-6:
        flags.add("percentage_intervals_exceed_100")
    return horizons, len(intervals), sorted(flags)

def extract_custom_horizon_months(tag_name):
    """Infer a month horizon from an extension tag name."""
    normalized = re.sub(r"[^a-z0-9]", "", str(tag_name).lower())

    numeric = re.search(r"(12|24|36|48|60|72|84|96|108|120)months?", normalized)
    if numeric:
        return int(numeric.group(1))

    for word, months in NUMBER_WORDS.items():
        if word + "month" in normalized or word + "months" in normalized:
            return months
    return None


def choose_as_of_date(g, totals, amount_facts, segment_facts):
    candidates = pd.concat([
        totals["ref_date_parsed"],
        amount_facts["ref_date_parsed"],
        segment_facts["ref_date_parsed"],
    ]).dropna()
    if candidates.empty:
        candidates = g["ref_date_parsed"].dropna()
    return candidates.max() if not candidates.empty else None


def summarize(g):
    out = {
        "as_of_date": None,
        "rpo_total": None,
        "rpo_thereafter": None,
        "n_timing_buckets": 0,
        "n_percentage_intervals": 0,
        "max_horizon_months": None,
        "method": None,
        "flag": [],
    }

    amount = g[(g["name"] == C_TOTAL) & ~g["has_other_dims"]]
    totals = amount[amount["timing_member"].isna()].dropna(subset=["value"])
    segments = g[
        (g["name"] == C_TOTAL)
        & g["has_other_dims"]
        & g["timing_member"].isna()
    ].dropna(subset=["value"])

    as_of = choose_as_of_date(g, totals, amount, segments)
    if as_of is None or pd.isna(as_of):
        out["flag"] = "missing_as_of_date"
        return out

    out["as_of_date"] = as_of.strftime("%Y-%m-%d")

    current_totals = totals[totals["ref_date_parsed"] == as_of]
    if not current_totals.empty:
        out["rpo_total"] = current_totals["value"].iloc[0]
        out["method"] = "reported_total"
    else:
        current_segments = segments[segments["ref_date_parsed"] == as_of]
        if (
            not current_segments.empty
            and current_segments["other_axes"].nunique() == 1
            and current_segments["other_key"].is_unique
            and "unparsed" not in current_segments["other_axes"].iloc[0]
        ):
            out["rpo_total"] = current_segments["value"].sum()
            out["method"] = "sum_of_segments"
            out["flag"].append("total_summed_across_segments")

    period_facts = g[(g["name"] == C_PERIOD) & (g["ref_date_parsed"] == as_of)]
    duration_map = dict(zip(period_facts["timing_member"], period_facts["value_raw"]))

    # 1. Dollar timing buckets
    dollar_buckets = amount[
        amount["timing_member"].notna()
        & (amount["ref_date_parsed"] == as_of)
    ]
    dollar_horizons = {}
    if not dollar_buckets.empty:
        dollar_horizons, disclosed_sum, n, flags = amount_horizons(
            dollar_buckets, duration_map, as_of
        )
        out["n_timing_buckets"] = n
        out["flag"].extend(flags)
        for months, value in dollar_horizons.items():
            out[f"rpo_within_{months}m"] = value

        if out["rpo_total"] is None and disclosed_sum is not None:
            out["rpo_total"] = disclosed_sum
            out["method"] = "sum_of_buckets"
        if dollar_horizons:
            out["method"] = (out["method"] or "") + "+dollar_horizons"

    # 2. Custom extension tags for explicit horizons
    custom = g[
        g["value"].notna()
        & g["unit"].astype(str).str.contains("USD", case=False, na=False)
        & (g["ref_date_parsed"] == as_of)
        & (g["name"] != C_TOTAL)
    ].copy()
    if not custom.empty:
        custom["horizon_months"] = custom["name"].map(extract_custom_horizon_months)
        custom = custom.dropna(subset=["horizon_months"])
        for _, row in custom.iterrows():
            months = int(row["horizon_months"])
            column = f"rpo_within_{months}m"
            if out.get(column) is None:
                out[column] = row["value"]
                out["method"] = (out["method"] or "") + "+custom_horizon_tag"

    # 3. Percentage timing buckets, converted to amounts when total exists
    percentages = g[
        g["name"].str.match(C_PCT_RE, na=False)
        & g["timing_member"].notna()
        & ~g["has_other_dims"]
    ].dropna(subset=["value"]).copy()

    if not percentages.empty:
        pct_date = percentages["ref_date_parsed"].dropna().max()
        if pd.notna(pct_date):
            percentages = percentages[percentages["ref_date_parsed"] == pct_date]

        pct_horizons, n_pct_intervals, flags = percentage_horizons(
            percentages, period_facts, as_of
        )
        out["flag"].extend("pct_" + flag for flag in flags)
        out["n_percentage_intervals"] = n_pct_intervals

        for months, pct in pct_horizons.items():
            if not (0 < pct <= 100 + 1e-6):
                out["flag"].append(f"invalid_pct_within_{months}m")
                continue

            out[f"pct_within_{months}m"] = round(pct, 2)
            amount_column = f"rpo_within_{months}m"
            if out.get(amount_column) is None and out["rpo_total"] is not None:
                out[amount_column] = round(out["rpo_total"] * pct / 100, 0)
                out["method"] = (out["method"] or "") + "+pct_horizons_derived"

    horizon_columns = sorted(
        [
            (int(match.group(1)), column)
            for column in out
            if (match := re.fullmatch(r"rpo_within_(\d+)m", column))
            and out[column] is not None
        ],
        key=lambda item: item[0],
    )

    if horizon_columns:
        max_months, max_column = horizon_columns[-1]
        out["max_horizon_months"] = max_months
        if out["rpo_total"] is not None:
            out["rpo_thereafter"] = out["rpo_total"] - out[max_column]
            if out["rpo_thereafter"] < -1:
                out["flag"].append("maximum_horizon_exceeds_total")

    # Backward-compatible aliases
    out["rpo_current"] = out.get("rpo_within_12m")
    if out["rpo_total"] is not None and out["rpo_current"] is not None:
        out["rpo_noncurrent"] = out["rpo_total"] - out["rpo_current"]
    else:
        out["rpo_noncurrent"] = None

    if out["rpo_total"] is not None and out["rpo_total"] >= 1e11:
        out["flag"].append("check_magnitude")

    fiscal_year = g["FiscalYear"].iloc[0] if "FiscalYear" in g else None
    try:
        if fiscal_year and abs(as_of.year - int(float(fiscal_year))) > 1:
            out["flag"].append("as_of_vs_fiscalyear_mismatch")
    except (ValueError, TypeError):
        pass

    out["flag"] = "|".join(dict.fromkeys(flag for flag in out["flag"] if flag))
    return out


def order_summary_columns(summary, keys):
    horizon_amounts = sorted(
        [c for c in summary.columns if re.fullmatch(r"rpo_within_\d+m", c)],
        key=lambda c: int(re.search(r"\d+", c).group()),
    )
    horizon_pcts = sorted(
        [c for c in summary.columns if re.fullmatch(r"pct_within_\d+m", c)],
        key=lambda c: int(re.search(r"\d+", c).group()),
    )
    fixed = [
        "as_of_date", "rpo_total", *horizon_amounts, "rpo_thereafter",
        *horizon_pcts, "rpo_current", "rpo_noncurrent",
        "n_timing_buckets", "n_percentage_intervals", "max_horizon_months", "method", "flag",
    ]
    return summary[keys + [c for c in fixed if c in summary.columns]]


def main(path):
    df = pd.read_csv(path, dtype=str)
    long = melt_tags(df)
    long.to_csv("rpo_facts_long.csv", index=False)

    keys = [column for column in ID_COLS if column in long.columns]
    rows = []
    for key_values, group in long.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        record = dict(zip(keys, key_values))
        record.update(summarize(group))
        rows.append(record)

    extracted = pd.DataFrame(rows)
    summary = df[keys].drop_duplicates().merge(extracted, on=keys, how="left")
    summary = order_summary_columns(summary, keys)
    summary.to_csv("rpo_summary_horizons.csv", index=False)

    horizon_cols = [c for c in summary.columns if re.fullmatch(r"rpo_within_\d+m", c)]
    print(f"{len(summary)} filings -> rpo_summary_horizons.csv")
    print(f"  with rpo_total: {summary['rpo_total'].notna().sum()}")
    for column in horizon_cols:
        print(f"  with {column}: {summary[column].notna().sum()}")
    print(f"  with rpo_thereafter: {summary['rpo_thereafter'].notna().sum()}")
    print("  flags:", summary["flag"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rpo_sample_100.csv")
