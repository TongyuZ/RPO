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
C_CURRENT = "RevenueRemainingPerformanceObligationCurrent"
C_NONCURRENT = "RevenueRemainingPerformanceObligationNoncurrent"
C_PCT_RE = re.compile(r"RevenueRemainingPerformanceObligation.*Percentage$", re.I)
C_ANY_PCT_RE = re.compile(
    r"RevenueRemainingPerformanceObligation.*(?:Percentage|PercentRecognized)$",
    re.I,
)

# Only these one-dimensional member sets are treated as potentially additive.
# Cross-dimensional facts (for example Segment x TypeOfArrangement) are slices,
# not company totals, and must never be summed into rpo_total.
ADDITIVE_AXIS_SUFFIXES = (
    "StatementBusinessSegmentsAxis",
    "ProductOrServiceAxis",
)

# Generic number vocabulary for extension tags and dimension members.  XBRL
# names are commonly CamelCase (ThirtySixMonths), numeric (36Months), ordinal
# (FourthFiscalYear), or abbreviated (36Mos / 3Yrs).
SMALL_CARDINALS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
)
TENS_CARDINALS = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}
SMALL_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth",
    18: "eighteenth", 19: "nineteenth",
}
TENS_ORDINALS = {
    20: "twentieth", 30: "thirtieth", 40: "fortieth", 50: "fiftieth",
    60: "sixtieth", 70: "seventieth", 80: "eightieth", 90: "ninetieth",
}

TOL_DAYS = 45
DAYS_PER_MONTH = 365.25 / 12


def compact_cardinal(n, include_and=False):
    """Return a compact English cardinal, e.g. 36 -> thirtysix."""
    if not 0 <= n < 1000:
        return None
    if n < 20:
        return SMALL_CARDINALS[n]
    if n < 100:
        tens, remainder = divmod(n, 10)
        return TENS_CARDINALS[tens * 10] + (SMALL_CARDINALS[remainder] if remainder else "")
    hundreds, remainder = divmod(n, 100)
    prefix = SMALL_CARDINALS[hundreds] + "hundred"
    if not remainder:
        return prefix
    connector = "and" if include_and else ""
    return prefix + connector + compact_cardinal(remainder, include_and)


def compact_ordinal(n):
    """Return a compact English ordinal, e.g. 36 -> thirtysixth."""
    if not 1 <= n < 1000:
        return None
    if n in SMALL_ORDINALS:
        return SMALL_ORDINALS[n]
    if n in TENS_ORDINALS:
        return TENS_ORDINALS[n]
    if n < 100:
        tens, remainder = divmod(n, 10)
        return TENS_CARDINALS[tens * 10] + SMALL_ORDINALS[remainder]
    hundreds, remainder = divmod(n, 100)
    prefix = SMALL_CARDINALS[hundreds] + "hundred"
    if not remainder:
        return prefix + "th"
    return prefix + compact_ordinal(remainder)


CARDINAL_WORD_VALUES = {}
ORDINAL_WORD_VALUES = {}
for _number in range(1, 1000):
    CARDINAL_WORD_VALUES[compact_cardinal(_number)] = _number
    CARDINAL_WORD_VALUES[compact_cardinal(_number, include_and=True)] = _number
    ORDINAL_WORD_VALUES[compact_ordinal(_number)] = _number


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


def parse_segment_dims(s):
    """Return one additive segment/product member plus remaining dimensions."""
    if not isinstance(s, str) or s.strip() in ("", "[]", "nan"):
        return None, None, ""
    try:
        dims = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None, None, ""

    candidates = []
    for d in dims:
        dimension = str(d.get("dimension", ""))
        member = str(d.get("member", "")).strip()
        if dimension.endswith(ADDITIVE_AXIS_SUFFIXES):
            candidates.append((dimension, member))
    if len(candidates) != 1:
        return None, None, ""

    segment_axis, segment_member = candidates[0]
    remaining = []
    for d in dims:
        dimension = str(d.get("dimension", ""))
        member = str(d.get("member", "")).strip()
        if dimension == segment_axis and member == segment_member:
            continue
        if TIMING_AXIS in dimension:
            continue
        remaining.append((dimension, member))
    remaining_key = ";".join(f"{a}={m}" for a, m in sorted(remaining))
    return segment_axis, segment_member, remaining_key


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
    parsed_segment = long["dimensions"].map(parse_segment_dims)
    long["segment_axis"] = parsed_segment.map(lambda item: item[0])
    long["segment_member"] = parsed_segment.map(lambda item: item[1])
    long["segment_other_key"] = parsed_segment.map(lambda item: item[2])
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
    distinct_starts = sorted(p["start"].drop_duplicates())
    next_start_by_start = {
        start: distinct_starts[i + 1] if i + 1 < len(distinct_starts) else None
        for i, start in enumerate(distinct_starts)
    }

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

        # Apply the same common XBRL typo correction used for dollar buckets:
        # P12Y is sometimes filed where the adjacent timing-axis dates make it
        # unambiguously a 12-month interval.
        next_start = next_start_by_start.get(start_date)
        typo = re.fullmatch(r"P(\d+)Y", chosen_duration)
        if typo and next_start is not None:
            alleged_months = int(typo.group(1))
            spacing = (next_start - start_date).days
            if abs(spacing - alleged_months * DAYS_PER_MONTH) <= 75:
                span_days = spacing
                flags.add("period1_year_month_typo_fixed")

        rows.append({
            "start": start_date,
            "pct": float(row["pct"]),
            "duration": chosen_duration,
            "span_days": float(span_days),
        })

    if not rows:
        return {}, 0, sorted(flags)

    intervals = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["start", "pct", "duration", "span_days"])
        .sort_values(["start", "span_days", "pct"])
    )

    # Expected-timing percentages describe mutually exclusive intervals.  Some
    # filings contain a bad timing-axis start date that makes a later interval
    # overlap an earlier one.  Abbott 2025 is one example: 52% is tagged P24M
    # from the report date, while the following 17%/P12M interval is given a
    # start only 12 months later.  Treating raw endpoints independently puts
    # both facts in within_24m and incorrectly reports 69%.  Preserve the first
    # interval and shift only a strictly later, overlapping start to the prior
    # interval's end.  Rows sharing the same raw start remain in the same bucket.
    normalized_rows = []
    previous_end = None
    for raw_start, same_start in intervals.groupby("start", sort=True):
        effective_start = raw_start
        if (
            previous_end is not None
            and raw_start < previous_end - timedelta(days=TOL_DAYS)
        ):
            effective_start = previous_end
            flags.add("overlapping_percentage_intervals_sequenced")

        group_ends = []
        for _, row in same_start.iterrows():
            end_date = effective_start + timedelta(days=row["span_days"])
            elapsed_days = (end_date - as_of).days
            horizon = horizon_months_from_days(elapsed_days)
            if horizon is None:
                flags.add("invalid_percentage_horizon")
                continue

            expected_days = horizon * DAYS_PER_MONTH
            if abs(elapsed_days - expected_days) > TOL_DAYS:
                flags.add(
                    f"nonannual_percentage_endpoint_"
                    f"{round(elapsed_days / DAYS_PER_MONTH)}m"
                )

            normalized_rows.append({
                "start": raw_start,
                "effective_start": effective_start,
                "horizon": horizon,
                "pct": row["pct"],
                "duration": row["duration"],
            })
            group_ends.append(end_date)

        if group_ends:
            group_end = max(group_ends)
            previous_end = max(previous_end, group_end) if previous_end is not None else group_end

    if not normalized_rows:
        return {}, 0, sorted(flags)

    intervals = pd.DataFrame(normalized_rows)

    # If exact duplicate tags remain after dimension filtering, keep one.
    cumulative, horizons = 0.0, {}
    for horizon, grp in intervals.groupby("horizon", sort=True):
        cumulative += grp["pct"].sum()
        horizons[int(horizon)] = cumulative

    if cumulative > 100 + 1e-6:
        flags.add("percentage_intervals_exceed_100")
    return horizons, len(intervals), sorted(flags)

def extract_duration_months_from_label(label):
    """Extract a duration from a normalized XBRL label.

    Supports numeric/cardinal/ordinal months, years, fiscal years, calendar
    years, quarters, and common abbreviations.  It returns only the duration;
    callers remain responsible for deciding whether the phrase is cumulative.
    """
    normalized = re.sub(r"[^a-z0-9]", "", str(label).lower())
    unit_multipliers = (
        ("fiscalyears", 12), ("fiscalyear", 12),
        ("calendaryears", 12), ("calendaryear", 12),
        ("quarters", 3), ("quarter", 3), ("qtrs", 3), ("qtr", 3),
        ("months", 1), ("month", 1), ("mos", 1), ("mo", 1),
        ("years", 12), ("year", 12), ("yrs", 12), ("yr", 12),
    )

    # Prefer explicit digits when present.
    for unit, multiplier in unit_multipliers:
        match = re.search(rf"(\d+){unit}", normalized)
        if match:
            months = int(match.group(1)) * multiplier
            return months if 0 < months <= 12000 else None

    # Match the longest English token first so onehundredtwenty wins over twenty.
    word_values = {**CARDINAL_WORD_VALUES, **ORDINAL_WORD_VALUES}
    for word in sorted(word_values, key=len, reverse=True):
        for unit, multiplier in unit_multipliers:
            if word + unit in normalized:
                months = word_values[word] * multiplier
                return months if 0 < months <= 12000 else None
    return None


def extract_custom_horizon_months(tag_name):
    """Infer a cumulative month horizon from an extension tag name."""
    normalized = re.sub(r"[^a-z0-9]", "", str(tag_name).lower())
    if any(word in normalized for word in (
        "subsequent", "thereafter", "beyond", "morethan", "after",
        "between", "from",
    )):
        return None
    return extract_duration_months_from_label(normalized)


def is_additive_axis_set(axes):
    """True only for one safe, potentially exhaustive business/product axis."""
    return (
        isinstance(axes, tuple)
        and len(axes) == 1
        and axes[0].endswith(ADDITIVE_AXIS_SUFFIXES)
    )


def append_method(out, label):
    existing = out.get("method")
    out["method"] = f"{existing}+{label}" if existing else label


def normalize_percentage(value):
    """Return percentage points for either 0.52 or 52 representations."""
    return float(value) * 100 if abs(float(value)) <= 1.0 + 1e-9 else float(value)


def member_horizon_months(other_key):
    """Infer safe cumulative horizons from extension dimension members."""
    normalized = re.sub(r"[^a-z0-9]", "", str(other_key).lower())

    # These words describe a later interval or an open-ended range, not a
    # cumulative "within" horizon.  They require start+duration sequencing.
    if any(word in normalized for word in (
        "subsequent", "thereafter", "beyond", "morethan", "after",
        "between", "from", "remainder",
    )):
        return None

    if "withinnextfiscalyear" in normalized or "nextfiscalyear" in normalized:
        return 12
    if "currentmember" in normalized or "currentportionmember" in normalized:
        return 12

    cumulative_markers = (
        "within", "next", "first", "upto", "upthrough", "byendof",
        "nolaterthan", "notlaterthan", "overthenext", "duringthenext",
    )
    if not any(marker in normalized for marker in cumulative_markers):
        return None
    return extract_duration_months_from_label(normalized)


def cumulative_member_percentage_horizons(g, as_of):
    """Read already-cumulative percentages encoded in dimension members."""
    facts = g[
        g["name"].str.match(C_ANY_PCT_RE, na=False)
        & g["has_other_dims"]
        & g["segment_member"].isna()
        & g["timing_member"].isna()
        & (g["ref_date_parsed"] == as_of)
    ].dropna(subset=["value"]).copy()
    if facts.empty:
        return {}, []

    facts["horizon"] = facts["other_key"].map(member_horizon_months)
    facts = facts.dropna(subset=["horizon"])
    if facts.empty:
        return {}, []

    horizons, flags = {}, []
    for horizon, grp in facts.groupby("horizon", sort=True):
        values = sorted({round(normalize_percentage(v), 8) for v in grp["value"]})
        if len(values) != 1:
            flags.append(f"conflicting_member_pct_within_{int(horizon)}m")
            continue
        horizons[int(horizon)] = values[0]
    return horizons, flags


def dimensioned_amount_horizons(amount_facts, period_facts, as_of):
    """Aggregate timed dollar facts only across one safe additive axis.

    This handles disclosures such as Tesla's ProductOrServiceAxis amounts.  It
    deliberately does not combine facts from different or nested axis sets.
    """
    facts = amount_facts[
        amount_facts["timing_member"].notna()
        & amount_facts["has_other_dims"]
        & (amount_facts["ref_date_parsed"] == as_of)
        & amount_facts["other_axes"].map(is_additive_axis_set)
    ].dropna(subset=["value"]).copy()
    if facts.empty:
        return {}, 0, []

    axis_sets = facts["other_axes"].drop_duplicates().tolist()
    if len(axis_sets) != 1:
        return {}, 0, ["multiple_dimensioned_timing_axis_sets_not_summed"]

    facts = facts.drop_duplicates(
        subset=["timing_member", "other_key", "value", "ref_date"]
    )
    interval_rows, flags = [], set()
    for _, row in facts.iterrows():
        matches = period_facts[
            (period_facts["timing_member"] == row["timing_member"])
            & (period_facts["other_key"] == row["other_key"])
        ]
        durations = sorted({
            str(v).strip() for v in matches["value_raw"]
            if duration_days(v) is not None
        })
        if not durations:
            sibling_periods = period_facts[
                (period_facts["timing_member"] == row["timing_member"])
                & (period_facts["other_axes"] == row["other_axes"])
            ]
            durations = sorted({
                str(v).strip() for v in sibling_periods["value_raw"]
                if duration_days(v) is not None
            })
            if len(durations) == 1:
                flags.add("dimensioned_amount_period_inferred_from_sibling_member")
        if len(durations) != 1:
            flags.add("missing_or_ambiguous_dimensioned_amount_period")
            continue

        span_days = duration_days(durations[0])
        start = pd.to_datetime(row["timing_member"], errors="coerce")
        if pd.isna(start):
            flags.add("unparsed_dimensioned_amount_start")
            continue
        end = start + timedelta(days=float(span_days))
        horizon = horizon_months_from_days((end - as_of).days)
        if horizon is None:
            flags.add("invalid_dimensioned_amount_horizon")
            continue
        interval_rows.append((horizon, float(row["value"])))

    cumulative, horizons = 0.0, {}
    for horizon, value in sorted(interval_rows):
        cumulative += value
        horizons[int(horizon)] = cumulative
    if horizons:
        flags.add("timing_amounts_summed_across_additive_members")
    return horizons, len(interval_rows), sorted(flags)


def match_dimensioned_amount_periods(amount_facts, period_facts, as_of):
    """Pair an undated dimensioned amount with a period sharing other_key.

    The pair is audit information, not a within-X-month amount: a P25Y period
    tells us the outer recognition horizon but not the annual allocation.
    """
    undated = amount_facts[
        amount_facts["has_other_dims"]
        & amount_facts["timing_member"].isna()
        & (amount_facts["ref_date_parsed"] == as_of)
    ].dropna(subset=["value"])
    timed_keys = set(
        amount_facts.loc[
            amount_facts["timing_member"].notna()
            & (amount_facts["ref_date_parsed"] == as_of),
            "other_key",
        ]
    )

    pairs = []
    for _, amount in undated.iterrows():
        # When a timed dollar fact exists for the same slice, the period belongs
        # to that timed fact; pairing it to the undated total would be ambiguous.
        if amount["other_key"] in timed_keys:
            continue
        matches = period_facts[
            (period_facts["other_key"] == amount["other_key"])
            & period_facts["timing_member"].notna()
        ]
        for _, period in matches.iterrows():
            span_days = duration_days(period["value_raw"])
            start = pd.to_datetime(period["timing_member"], errors="coerce")
            if span_days is None or pd.isna(start):
                continue
            end = start + timedelta(days=float(span_days))
            pairs.append({
                "amount": float(amount["value"]),
                "start": start.strftime("%Y-%m-%d"),
                "duration": str(period["value_raw"]).strip(),
                "end": end.strftime("%Y-%m-%d"),
                "horizon_months": horizon_months_from_days((end - as_of).days),
                "dimensions": amount["other_key"],
            })

    unique = []
    seen = set()
    for pair in pairs:
        key = tuple(pair.items())
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique


def qname_local(value):
    """Return the readable local part of an XBRL QName."""
    return str(value).split(":", 1)[-1]


def segment_display_name(member, remaining_key):
    """Build a readable segment name without hiding intersecting dimensions."""
    name = qname_local(member)
    if not remaining_key:
        return name
    extra_members = []
    for item in str(remaining_key).split(";"):
        if "=" in item:
            extra_members.append(qname_local(item.split("=", 1)[1]))
    return name + (" | " + " | ".join(extra_members) if extra_members else "")


def single_segment_amount_horizons(segment_amounts, all_period_facts, as_of):
    """Calculate cumulative dollar horizons for one segment/member slice."""
    timed = segment_amounts[
        segment_amounts["timing_member"].notna()
        & (segment_amounts["ref_date_parsed"] == as_of)
    ].dropna(subset=["value"]).drop_duplicates(
        subset=["timing_member", "other_key", "value", "ref_date"]
    )
    interval_rows, flags = [], set()
    for _, row in timed.iterrows():
        exact = all_period_facts[
            (all_period_facts["timing_member"] == row["timing_member"])
            & (all_period_facts["other_key"] == row["other_key"])
        ]
        durations = sorted({
            str(v).strip() for v in exact["value_raw"]
            if duration_days(v) is not None
        })
        if not durations:
            siblings = all_period_facts[
                (all_period_facts["timing_member"] == row["timing_member"])
                & (all_period_facts["segment_axis"] == row["segment_axis"])
            ]
            durations = sorted({
                str(v).strip() for v in siblings["value_raw"]
                if duration_days(v) is not None
            })
            if len(durations) == 1:
                flags.add("period_inferred_from_sibling_segment")
        if len(durations) != 1:
            flags.add("missing_or_ambiguous_segment_period")
            continue

        start = pd.to_datetime(row["timing_member"], errors="coerce")
        if pd.isna(start):
            flags.add("unparsed_segment_start")
            continue
        end = start + timedelta(days=float(duration_days(durations[0])))
        horizon = horizon_months_from_days((end - as_of).days)
        if horizon is None:
            flags.add("invalid_segment_horizon")
            continue
        interval_rows.append((horizon, float(row["value"])))

    cumulative, horizons = 0.0, {}
    for horizon, value in sorted(interval_rows):
        cumulative += value
        horizons[int(horizon)] = cumulative
    return horizons, sorted(flags)


def summarize_segments(g, period_facts, as_of):
    """Flatten segment -> value -> horizons into Segment_n_* output columns."""
    numeric_segment_facts = g[
        g["segment_member"].notna()
        & (g["ref_date_parsed"] == as_of)
        & g["value"].notna()
    ]
    identities = sorted({
        (row["segment_axis"], row["segment_member"], row["segment_other_key"])
        for _, row in numeric_segment_facts.iterrows()
    })

    result, flags = {}, []
    for index, (axis, member, remaining_key) in enumerate(identities, 1):
        prefix = f"Segment_{index}"
        sg = g[
            (g["segment_axis"] == axis)
            & (g["segment_member"] == member)
            & (g["segment_other_key"] == remaining_key)
            & (g["ref_date_parsed"] == as_of)
        ]
        amounts = sg[sg["name"] == C_TOTAL]
        undated_values = amounts.loc[
            amounts["timing_member"].isna(), "value"
        ].dropna().drop_duplicates()

        result[prefix] = segment_display_name(member, remaining_key)
        result[f"{prefix}_axis"] = qname_local(axis)
        result[f"{prefix}_dimensions"] = sg["other_key"].dropna().iloc[0]
        if len(undated_values) == 1:
            result[f"{prefix}_value"] = float(undated_values.iloc[0])
        elif len(undated_values) > 1:
            flags.append(f"{prefix}_conflicting_values")

        horizons, horizon_flags = single_segment_amount_horizons(
            amounts, period_facts, as_of
        )
        flags.extend(f"{prefix}_{flag}" for flag in horizon_flags)

        segment_current = sg[sg["name"] == C_CURRENT]["value"].dropna().drop_duplicates()
        if len(segment_current) == 1 and 12 not in horizons:
            horizons[12] = float(segment_current.iloc[0])

        segment_percentages = sg[
            sg["name"].str.match(C_PCT_RE, na=False)
            & sg["timing_member"].notna()
        ]
        if not segment_percentages.empty:
            segment_periods = period_facts[
                period_facts["other_key"].isin(sg["other_key"].dropna().unique())
            ]
            pct_horizons, _, pct_flags = percentage_horizons(
                segment_percentages, segment_periods, as_of
            )
            flags.extend(f"{prefix}_pct_{flag}" for flag in pct_flags)
            if len(undated_values) == 1:
                segment_total = float(undated_values.iloc[0])
                for months, pct in pct_horizons.items():
                    horizons.setdefault(months, round(segment_total * pct / 100, 0))
                    result[f"{prefix}_pct_within_{months}months"] = round(pct, 2)

        for months, value in sorted(horizons.items()):
            result[f"{prefix}_within_{months}months"] = value

        pairs = match_dimensioned_amount_periods(amounts, period_facts, as_of)
        if len(pairs) == 1:
            pair = pairs[0]
            result[f"{prefix}_recognition_start"] = pair["start"]
            result[f"{prefix}_recognition_duration"] = pair["duration"]
            result[f"{prefix}_recognition_end"] = pair["end"]
            result[f"{prefix}_max_horizon_months"] = pair["horizon_months"]
        elif len(pairs) > 1:
            flags.append(f"{prefix}_multiple_period_pairs")

    result["n_segments"] = len(identities)
    return result, flags


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
        "n_dimensioned_period_pairs": 0,
        "dimensioned_rpo_amount": None,
        "dimensioned_recognition_start": None,
        "dimensioned_recognition_duration": None,
        "dimensioned_recognition_end": None,
        "dimensioned_max_horizon_months": None,
        "dimensioned_period_details": None,
        "max_horizon_months": None,
        "method": None,
        "flag": [],
    }
    company_level_horizons = set()

    all_amounts = g[g["name"] == C_TOTAL]
    amount = all_amounts[~all_amounts["has_other_dims"]]
    totals = amount[amount["timing_member"].isna()].dropna(subset=["value"])
    segments = all_amounts[
        all_amounts["has_other_dims"]
        & all_amounts["timing_member"].isna()
    ].dropna(subset=["value"])

    as_of = choose_as_of_date(g, totals, all_amounts, segments)
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
        eligible_segments = current_segments[
            current_segments["other_axes"].map(is_additive_axis_set)
        ]
        if (
            len(eligible_segments) >= 2
            and eligible_segments["other_axes"].nunique() == 1
            and eligible_segments["other_key"].is_unique
        ):
            out["rpo_total"] = eligible_segments["value"].sum()
            out["method"] = "sum_of_segments"
            out["flag"].append("total_summed_across_segments")
        elif not current_segments.empty:
            out["flag"].append("dimensioned_amounts_not_used_as_company_total")

    current_facts = g[
        (g["name"] == C_CURRENT)
        & ~g["has_other_dims"]
        & (g["ref_date_parsed"] == as_of)
    ].dropna(subset=["value"])
    noncurrent_facts = g[
        (g["name"] == C_NONCURRENT)
        & ~g["has_other_dims"]
        & (g["ref_date_parsed"] == as_of)
    ].dropna(subset=["value"])
    current_value = (
        current_facts["value"].drop_duplicates().iloc[0]
        if current_facts["value"].nunique() == 1 else None
    )
    noncurrent_value = (
        noncurrent_facts["value"].drop_duplicates().iloc[0]
        if noncurrent_facts["value"].nunique() == 1 else None
    )
    if current_value is not None:
        out["rpo_within_12m"] = float(current_value)
        company_level_horizons.add(12)
        append_method(out, "reported_current")
    if out["rpo_total"] is None and current_value is not None and noncurrent_value is not None:
        out["rpo_total"] = float(current_value + noncurrent_value)
        append_method(out, "current_plus_noncurrent_total")
    elif out["rpo_total"] is not None and current_value is None and noncurrent_value is not None:
        inferred_current = out["rpo_total"] - float(noncurrent_value)
        if inferred_current >= 0:
            out["rpo_within_12m"] = inferred_current
            company_level_horizons.add(12)
            append_method(out, "current_derived_from_noncurrent")

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
            column = f"rpo_within_{months}m"
            if out.get(column) is None:
                out[column] = value
            company_level_horizons.add(int(months))

        if out["rpo_total"] is None and disclosed_sum is not None:
            out["rpo_total"] = disclosed_sum
            out["method"] = "sum_of_buckets"
        if dollar_horizons:
            append_method(out, "dollar_horizons")

    # 1b. Timed dollar facts split across one additive product/segment axis.
    dimensioned_horizons, n_dimensioned, flags = dimensioned_amount_horizons(
        all_amounts, period_facts, as_of
    )
    out["flag"].extend(flags)
    if dimensioned_horizons:
        out["n_timing_buckets"] += n_dimensioned
        for months, value in dimensioned_horizons.items():
            column = f"rpo_within_{months}m"
            if out.get(column) is None:
                out[column] = value
                append_method(out, "dimensioned_dollar_horizons")

    # 2. Custom extension tags for explicit horizons
    custom = g[
        g["value"].notna()
        & g["unit"].astype(str).str.contains("USD", case=False, na=False)
        & (g["ref_date_parsed"] == as_of)
        & (g["name"] != C_TOTAL)
        & g["segment_member"].isna()
    ].copy()
    if not custom.empty:
        custom["horizon_months"] = custom["name"].map(extract_custom_horizon_months)
        custom = custom.dropna(subset=["horizon_months"])
        for _, row in custom.iterrows():
            months = int(row["horizon_months"])
            column = f"rpo_within_{months}m"
            if out.get(column) is None:
                out[column] = row["value"]
                append_method(out, "custom_horizon_tag")
            company_level_horizons.add(months)

    # 2b. Cumulative percentage horizons encoded in extension members, such as
    # Boeing's WithinNextFiscalYearMember / WithinNext4FiscalYearsMember.
    member_pct_horizons, flags = cumulative_member_percentage_horizons(g, as_of)
    out["flag"].extend("member_" + flag for flag in flags)
    for months, pct in member_pct_horizons.items():
        if not (0 < pct <= 100 + 1e-6):
            out["flag"].append(f"invalid_member_pct_within_{months}m")
            continue
        pct_column = f"pct_within_{months}m"
        if out.get(pct_column) is None:
            out[pct_column] = round(pct, 2)
        company_level_horizons.add(int(months))
        amount_column = f"rpo_within_{months}m"
        if out.get(amount_column) is None and out["rpo_total"] is not None:
            out[amount_column] = round(out["rpo_total"] * pct / 100, 0)
            append_method(out, "member_pct_horizon_derived")

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

            pct_column = f"pct_within_{months}m"
            if out.get(pct_column) is None:
                out[pct_column] = round(pct, 2)
            company_level_horizons.add(int(months))
            amount_column = f"rpo_within_{months}m"
            if out.get(amount_column) is None and out["rpo_total"] is not None:
                out[amount_column] = round(out["rpo_total"] * pct / 100, 0)
                append_method(out, "pct_horizons_derived")

    # 4. Pair dimensioned undated amounts with a same-slice period for audit.
    # These pairs disclose an outer recognition horizon, not annual allocation.
    dimensioned_pairs = match_dimensioned_amount_periods(
        all_amounts, period_facts, as_of
    )
    out["n_dimensioned_period_pairs"] = len(dimensioned_pairs)
    if dimensioned_pairs:
        out["flag"].append("dimensioned_amount_period_pair_not_annualized")
        if len(dimensioned_pairs) == 1:
            pair = dimensioned_pairs[0]
            out["dimensioned_rpo_amount"] = pair["amount"]
            out["dimensioned_recognition_start"] = pair["start"]
            out["dimensioned_recognition_duration"] = pair["duration"]
            out["dimensioned_recognition_end"] = pair["end"]
            out["dimensioned_max_horizon_months"] = pair["horizon_months"]
        out["dimensioned_period_details"] = " | ".join(
            f"amount={pair['amount']:g};start={pair['start']};"
            f"duration={pair['duration']};end={pair['end']};"
            f"horizon_months={pair['horizon_months']};"
            f"dimensions={pair['dimensions']}"
            for pair in dimensioned_pairs
        )

    # 5. Hierarchical wide output requested by the user: Total -> time and
    # Segment_n -> segment value -> segment-specific time.  Company-level time
    # facts are never copied into a segment, and segment time is not presented
    # as a company-level Total_* fact.
    out["Total_RPO"] = out["rpo_total"]
    for months in sorted(company_level_horizons):
        amount_value = out.get(f"rpo_within_{months}m")
        pct_value = out.get(f"pct_within_{months}m")
        if amount_value is not None:
            out[f"Total_within_{months}months"] = amount_value
        if pct_value is not None:
            out[f"Total_pct_within_{months}months"] = pct_value
    if company_level_horizons and out["rpo_total"] is not None:
        last_company_month = max(company_level_horizons)
        last_company_value = out.get(f"rpo_within_{last_company_month}m")
        if last_company_value is not None:
            out["Total_thereafter"] = out["rpo_total"] - last_company_value

    segment_output, segment_flags = summarize_segments(g, period_facts, as_of)
    out.update(segment_output)
    out["flag"].extend(segment_flags)

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

    if (
        noncurrent_value is not None
        and out["rpo_total"] is not None
        and out.get("rpo_within_12m") is not None
    ):
        expected_noncurrent = out["rpo_total"] - out["rpo_within_12m"]
        tolerance = max(1.0, abs(out["rpo_total"]) * 1e-6)
        if abs(expected_noncurrent - float(noncurrent_value)) > tolerance:
            out["flag"].append("reported_noncurrent_does_not_reconcile")

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
    total_amounts = sorted(
        [c for c in summary.columns if re.fullmatch(r"Total_within_\d+months", c)],
        key=lambda c: int(re.search(r"\d+", c).group()),
    )
    total_pcts = sorted(
        [c for c in summary.columns if re.fullmatch(r"Total_pct_within_\d+months", c)],
        key=lambda c: int(re.search(r"\d+", c).group()),
    )

    def segment_column_key(column):
        match = re.match(r"Segment_(\d+)(.*)$", column)
        index, suffix = int(match.group(1)), match.group(2)
        if suffix == "":
            rank = (0, 0)
        elif suffix == "_axis":
            rank = (1, 0)
        elif suffix == "_value":
            rank = (2, 0)
        elif (horizon := re.fullmatch(r"_within_(\d+)months", suffix)):
            rank = (3, int(horizon.group(1)))
        elif (pct := re.fullmatch(r"_pct_within_(\d+)months", suffix)):
            rank = (4, int(pct.group(1)))
        elif suffix.startswith("_recognition_") or suffix == "_max_horizon_months":
            rank = (5, suffix)
        elif suffix == "_dimensions":
            rank = (6, 0)
        else:
            rank = (7, suffix)
        return index, rank

    segment_columns = sorted(
        [c for c in summary.columns if re.match(r"Segment_\d+(?:_|$)", c)],
        key=segment_column_key,
    )
    fixed = [
        "as_of_date", "Total_RPO", *total_amounts, "Total_thereafter",
        *total_pcts, "n_segments", *segment_columns,
        "as_of_date", "rpo_total", *horizon_amounts, "rpo_thereafter",
        *horizon_pcts, "rpo_current", "rpo_noncurrent",
        "n_timing_buckets", "n_percentage_intervals", "max_horizon_months",
        "n_dimensioned_period_pairs", "dimensioned_rpo_amount",
        "dimensioned_recognition_start", "dimensioned_recognition_duration",
        "dimensioned_recognition_end", "dimensioned_max_horizon_months",
        "dimensioned_period_details", "method", "flag",
    ]
    ordered = []
    for column in keys + fixed:
        if column in summary.columns and column not in ordered:
            ordered.append(column)
    return summary[ordered]


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
