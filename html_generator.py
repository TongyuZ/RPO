import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from sec_http import sec_get_json

OUTPUT_COLUMNS = [
    "Company", "CIK", "fiscalYear", "html",
    "stateOfIncorporation", "sic", "sicDescription", "exchanges",
]


def cik_pad(cik: str) -> str:
    s = str(cik).strip()
    if not s.isdigit():
        raise ValueError(f"Bad CIK: {cik}")
    return str(int(s)).zfill(10)


def accession_nodash(acc: str) -> str:
    return acc.replace("-", "")


def archives_base(cik10: str, acc: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{accession_nodash(acc)}/"


def fetch_submissions_json(cik: str) -> dict:
    cik10 = cik_pad(cik)
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    return sec_get_json(url)


def extract_10k_rows(block: dict, cik10: str, start_year: int, end_year: int, include_amended: bool):
    forms = block.get("form", [])
    filing_dates = block.get("filingDate", [])
    report_dates = block.get("reportDate", [])
    accessions = block.get("accessionNumber", [])
    primary_docs = block.get("primaryDocument", [])

    out = []
    for form, fdate, rdate, acc, pdoc in zip(forms, filing_dates, report_dates, accessions, primary_docs):
        if include_amended:
            if form not in ("10-K", "10-K/A"):
                continue
        else:
            if form != "10-K":
                continue

        # use reportDate year as fiscal year filter (preferred for 10-K)
        try:
            fy = datetime.strptime(rdate, "%Y-%m-%d").year
        except Exception:
            try:
                fy = datetime.strptime(fdate, "%Y-%m-%d").year
            except Exception:
                continue

        if fy < start_year or fy > end_year:
            continue

        base = archives_base(cik10, acc)
        out.append({
            "CIK": cik10,
            "Company": None,  # fill later
            "filingDate": fdate,
            "reportDate": rdate,
            "fiscalYear": fy,
            "accessionNumber": acc,
            "html": base + pdoc,
        })
    return out


def get_10k_html_urls_from_submissions(cik: str, start_year=2013, end_year=2025, include_amended=False) -> pd.DataFrame:
    data = fetch_submissions_json(cik)
    cik10 = cik_pad(cik)

    rows = []
    recent = data.get("filings", {}).get("recent", {})
    rows.extend(extract_10k_rows(recent, cik10, start_year, end_year, include_amended))

    for f in data.get("filings", {}).get("files", []):
        name = f.get("name")
        if not name:
            continue
        chunk_url = "https://data.sec.gov/submissions/" + name
        chunk = sec_get_json(chunk_url)
        rows.extend(extract_10k_rows(chunk, cik10, start_year, end_year, include_amended))

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = (pd.DataFrame(rows)
          .drop_duplicates(subset=["accessionNumber"])
          .sort_values(["CIK", "fiscalYear", "filingDate"])
          .reset_index(drop=True))

    df["Company"] = data.get("name")
    df["stateOfIncorporation"] = data.get("stateOfIncorporation")
    df["sic"] = data.get("sic")
    df["sicDescription"] = data.get("sicDescription")
    exchanges = [e for e in (data.get("exchanges") or []) if e]
    df["exchanges"] = ", ".join(exchanges) if exchanges else None

    return df[OUTPUT_COLUMNS]


def _load_checkpoint(out_xlsx: str):
    existing = pd.read_excel(out_xlsx, sheet_name="10K_HTML", dtype={"CIK": str})
    done_ciks = set(existing["CIK"].dropna().unique())
    return existing, done_ciks


def _save(out_xlsx: str, done_dfs: list, errors: list) -> pd.DataFrame:
    df_all = pd.concat(done_dfs, ignore_index=True) if done_dfs else pd.DataFrame(columns=OUTPUT_COLUMNS)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_all.to_excel(writer, index=False, sheet_name="10K_HTML")
        if errors:
            pd.DataFrame(errors).to_excel(writer, index=False, sheet_name="Errors")
    return df_all


def export_company_cik_year_html(cik_ls,
                                  out_xlsx="company_cik_fy_html_10k.xlsx",
                                  start_year=2013,
                                  end_year=2025,
                                  include_amended=False,
                                  max_workers=6,
                                  checkpoint_every=50,
                                  resume=True):
    """
    Pull 10-K filing HTML URLs for every CIK in `cik_ls`, fiscal years
    [start_year, end_year]. Uses a small thread pool (this is I/O bound;
    the shared sec_http rate limiter caps the actual request rate
    regardless of worker count). Checkpoints to `out_xlsx` every
    `checkpoint_every` companies and, when `resume=True`, skips CIKs
    already present in an existing `out_xlsx` so an interrupted run can
    continue where it left off.
    """
    if isinstance(cik_ls, str):
        raise TypeError("cik_ls must be a LIST, e.g. ['0001318605','0000320193']")

    cik_ls = [cik_pad(c) for c in cik_ls]

    done_dfs = []
    done_ciks = set()
    errors = []

    if resume and os.path.exists(out_xlsx):
        try:
            existing, done_ciks = _load_checkpoint(out_xlsx)
            done_dfs.append(existing)
            print(f"Resuming: {len(done_ciks)} CIKs already done, skipping them.")
        except Exception as e:
            print(f"Could not read existing checkpoint ({e}), starting fresh.")

    todo = [c for c in cik_ls if c not in done_ciks]
    print(f"{len(todo)} CIKs to process ({len(done_ciks)} already done).")

    lock = threading.Lock()
    completed = 0

    def _worker(cik):
        return cik, get_10k_html_urls_from_submissions(cik, start_year, end_year, include_amended)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, c): c for c in todo}
        for fut in as_completed(futures):
            cik = futures[fut]
            with lock:
                completed += 1
                try:
                    _, df = fut.result()
                    done_dfs.append(df)
                except Exception as e:
                    errors.append({"CIK": cik, "error": str(e)})
                    print(f"  Failed CIK={cik}: {e}")

                if completed % 25 == 0 or completed == len(todo):
                    print(f"[{completed}/{len(todo)}] processed this run")

                if completed % checkpoint_every == 0:
                    _save(out_xlsx, done_dfs, errors)
                    print(f"Checkpoint saved: {out_xlsx}")

    df_all = _save(out_xlsx, done_dfs, errors)
    print(f"Saved: {out_xlsx} ({len(df_all)} rows)")
    return df_all
