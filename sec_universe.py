"""
Universe of CIKs to scrape.

SEC's company_tickers.json is the standard list of SEC-registered,
exchange-listed companies (ticker, CIK, name). It's a practical starting
point for "US companies" since foreign private issuers mostly file
20-F/40-F rather than 10-K (and get filtered out naturally once we
keep only 10-K filings downstream). For a stricter US-only filter,
html_generator's output also carries `stateOfIncorporation`, `sic`,
`sicDescription`, and `exchanges` per company so you can filter before
running the XBRL step once exact criteria are decided.
"""

import os

import pandas as pd

from sec_http import sec_get_json

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def get_all_sec_companies(cache_path: str = "sec_company_tickers.csv", refresh: bool = False) -> pd.DataFrame:
    if not refresh and os.path.exists(cache_path):
        return pd.read_csv(cache_path, dtype={"cik": str})

    data = sec_get_json(TICKERS_URL)

    df = pd.DataFrame(data.values()).rename(columns={"cik_str": "cik", "title": "company"})
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df = df.drop_duplicates(subset="cik").sort_values("cik").reset_index(drop=True)

    df.to_csv(cache_path, index=False)
    return df
