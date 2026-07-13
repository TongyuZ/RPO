"""
Company universe sourced from Compustat (via WRDS), instead of SEC's
full ticker list. See SAS.txt: pulls gvkey/cik/fyear from comp.funda
(standard STD/INDL/D/C/USD "primary record" filter) and exports it to
gvkey_cik_fyear.csv.

Using Compustat's own company set is a cleaner definition of "US public
company" than SEC's company_tickers.json + ad hoc filters, since it's
already the standard North America Compustat universe.
"""

import pandas as pd


def load_gvkey_cik_fyear(csv_path: str = "gvkey_cik_fyear.csv") -> pd.DataFrame:
    """
    Load the gvkey/cik/fyear link table produced by SAS.txt. One row per
    (gvkey, cik, fyear). `cik` is cleaned to a zero-padded 10-digit
    string; rows with a missing/non-numeric cik (e.g. gvkeys with no
    EDGAR CIK) are dropped, since SAS.txt doesn't filter those out
    before formatting cik_str.
    """
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    if "gvkey" not in df.columns or "fyear" not in df.columns:
        raise ValueError(f"{csv_path} is missing expected columns 'gvkey'/'fyear': {list(df.columns)}")

    # cik_str (already zero-padded by SAS) if present, else raw cik
    cik_col = "cik_str" if "cik_str" in df.columns else "cik"
    cik_digits = df[cik_col].astype(str).str.extract(r"(\d+)")[0]

    before = len(df)
    df = df.assign(cik=cik_digits).dropna(subset=["cik", "gvkey"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} row(s) with missing gvkey/cik.")

    df["cik"] = df["cik"].str.zfill(10)
    df["gvkey"] = df["gvkey"].astype(str).str.strip()
    df["fyear"] = pd.to_numeric(df["fyear"], errors="coerce").astype("Int64")

    keep_cols = [c for c in ["gvkey", "cik", "fyear", "tic", "cusip"] if c in df.columns]
    return df[keep_cols].drop_duplicates().reset_index(drop=True)


def get_gvkey_cik_map(csv_path: str = "gvkey_cik_fyear.csv") -> pd.DataFrame:
    """
    Company-level gvkey<->cik map (one row per cik), collapsed across
    fyear. A given cik should map to a single gvkey; if a cik shows up
    with more than one gvkey (e.g. a corporate restructuring recorded
    under two gvkeys), the first one is kept and a warning is printed —
    check gvkey_cik_fyear.csv directly for those cases if it matters
    for your analysis.
    """
    df = load_gvkey_cik_fyear(csv_path)

    multi = df.groupby("cik")["gvkey"].nunique()
    dupes = multi[multi > 1]
    if len(dupes):
        print(f"Warning: {len(dupes)} cik(s) map to more than one gvkey (keeping the first). "
              f"e.g. {list(dupes.index[:5])}")

    return (
        df.sort_values(["cik", "fyear"])
        .drop_duplicates(subset=["cik"], keep="first")[["gvkey", "cik"]]
        .reset_index(drop=True)
    )
