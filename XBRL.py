import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from bs4 import BeautifulSoup

from sec_http import sec_get_text

RPO_TEXT_PATTERN = re.compile(
    r"remaining performance obligation",
    re.I
)

RPO_TAG_PATTERN = re.compile(
    r"RemainingPerformanceObligation",
    re.I
)


def filing_folder_url(html_url):
    return html_url.rsplit("/", 1)[0] + "/"


def clean_text(x):
    if x is None:
        return ""
    return " ".join(x.split())


def parse_xbrl_value(text):
    if text is None:
        return None

    x = text.replace(",", "").replace("$", "").replace("%", "").strip()

    try:
        return float(x)
    except ValueError:
        return text


def find_xbrl_instance_url(html_url):
    """
    从 filing folder 里找到真正的 XBRL instance XML。
    避免抓到 FilingSummary.xml、cal.xml、def.xml、lab.xml、pre.xml。
    """

    folder = filing_folder_url(html_url)
    index_html = sec_get_text(folder)
    soup = BeautifulSoup(index_html, "lxml")

    xml_candidates = []

    for a in soup.find_all("a"):
        href = a.get("href")

        if not href:
            continue

        file_name = href.split("/")[-1]

        if not file_name.lower().endswith(".xml"):
            continue

        lower_name = file_name.lower()

        # 排除明显不是 instance 的 XML
        if any(x in lower_name for x in [
            "filingsummary",
            "_cal",
            "_def",
            "_lab",
            "_pre",
            "cal.xml",
            "def.xml",
            "lab.xml",
            "pre.xml"
        ]):
            continue

        xml_candidates.append(folder + file_name)

    for xml_url in xml_candidates:
        try:
            xml_text = sec_get_text(xml_url)
            xml_soup = BeautifulSoup(xml_text, "lxml-xml")

            # 真正的 XBRL instance 通常有 xbrl root
            if xml_soup.find("xbrl"):
                return xml_url

        except Exception as e:
            print(f"Skip {xml_url}: {e}")
            continue

    return None


def clean_textblock_fact(value):
    soup = BeautifulSoup(value, "html.parser")
    return " ".join(soup.get_text(" ").split())


def build_context_map(xml_text):
    """
    读取 XBRL XML 里的所有 <context>，
    建立 contextRef -> context 信息 的字典。
    """

    soup = BeautifulSoup(xml_text, "lxml-xml")

    context_map = {}

    for ctx in soup.find_all("context"):

        ctx_id = ctx.get("id")

        if not ctx_id:
            continue

        # entity / CIK
        identifier_tag = ctx.find("identifier")

        entity_identifier = None
        entity_scheme = None

        if identifier_tag:
            entity_identifier = clean_text(identifier_tag.get_text())
            entity_scheme = identifier_tag.get("scheme")

        # period
        period = ctx.find("period")

        instant = None
        start_date = None
        end_date = None
        forever = False
        period_type = None

        if period:
            instant_tag = period.find("instant")
            start_tag = period.find("startDate")
            end_tag = period.find("endDate")
            forever_tag = period.find("forever")

            if instant_tag:
                instant = clean_text(instant_tag.get_text())
                period_type = "instant"

            elif start_tag and end_tag:
                start_date = clean_text(start_tag.get_text())
                end_date = clean_text(end_tag.get_text())
                period_type = "duration"

            elif forever_tag:
                forever = True
                period_type = "forever"

        # dimensions: segment + scenario
        dimensions = []

        for location_name in ["segment", "scenario"]:
            location = ctx.find(location_name)

            if not location:
                continue

            # explicitMember
            for member in location.find_all("explicitMember"):
                dimensions.append({
                    "location": location_name,
                    "member_type": "explicit",
                    "dimension": member.get("dimension"),
                    "member": clean_text(member.get_text())
                })

            # typedMember
            for member in location.find_all("typedMember"):
                dimensions.append({
                    "location": location_name,
                    "member_type": "typed",
                    "dimension": member.get("dimension"),
                    "member": clean_text(member.get_text())
                })

        context_map[ctx_id] = {
            "contextRef": ctx_id,
            "entity_identifier": entity_identifier,
            "entity_scheme": entity_scheme,
            "period_type": period_type,
            "instant": instant,
            "startDate": start_date,
            "endDate": end_date,
            "forever": forever,
            "dimensions": dimensions
        }

    return context_map


def build_unit_map(xml_text):
    """
    读取 XBRL XML 里的所有 <unit>，
    建立 unitRef -> unit 的字典。
    """

    soup = BeautifulSoup(xml_text, "lxml-xml")

    unit_map = {}

    for unit in soup.find_all("unit"):

        unit_id = unit.get("id")

        if not unit_id:
            continue

        measures = []

        for measure in unit.find_all("measure"):
            value = clean_text(measure.get_text())

            if value:
                measures.append(value)

        unit_map[unit_id] = " / ".join(measures)

    return unit_map


def extract_rpo_from_xbrl_xml(xml_url):
    """
    从 XBRL instance XML 提取：
    1. RPO numeric tags
    2. RPO-related TextBlock
    """

    xml = sec_get_text(xml_url)
    soup = BeautifulSoup(xml, "lxml-xml")

    # Context reference & Unit
    context_map = build_context_map(xml)
    unit_map = build_unit_map(xml)

    rows = []

    for tag in soup.find_all():
        tag_name = tag.name

        if not tag_name:
            continue

        if (
            "RemainingPerformanceObligation" in tag_name
            and any(x in tag_name for x in [
                "Axis",
                ".domain",
                "Domain",
            ])
        ):
            continue

        value_raw = clean_text(tag.get_text(" ", strip=True))

        if not value_raw:
            continue

        context_ref = tag.get("contextRef") or tag.get("contextref")
        context_info = context_map.get(context_ref, {})
        unit_ref = tag.get("unitRef") or tag.get("unitref")
        unit = unit_map.get(unit_ref)

        # 1. RPO numeric tags
        if RPO_TAG_PATTERN.search(tag_name):

            rows.append({
                "source": "xbrl_numeric",
                "xml_url": xml_url,
                "tag_name": tag_name,
                "value_raw": value_raw,
                "unit": unit,
                "period_type": context_info.get("period_type"),
                "instant": context_info.get("instant"),
                "startDate": context_info.get("startDate"),
                "endDate": context_info.get("endDate"),
                "forever": context_info.get("forever"),
                "dimensions": context_info.get("dimensions")
            })

        # 2. RPO TextBlock
        tag_name_lower = tag_name.lower()

        if "textblock" in tag_name_lower:
            m = RPO_TEXT_PATTERN.search(value_raw)
            if m:
                rows.append({
                    "source": "xbrl_textblock",
                    "xml_url": xml_url,
                    "tag_name": tag_name,
                    "value_raw": value_raw,
                    "unit": unit,
                    "period_type": context_info.get("period_type"),
                    "instant": context_info.get("instant"),
                    "startDate": context_info.get("startDate"),
                    "endDate": context_info.get("endDate"),
                    "forever": context_info.get("forever"),
                    "dimensions": context_info.get("dimensions")
                })

    return rows


def process_filing(html_url):
    """
    输入 10-K HTML URL，
    输出一个 summary row + detailed RPO tags。
    """

    xml_url = find_xbrl_instance_url(html_url)

    if xml_url:
        rpo_tags = extract_rpo_from_xbrl_xml(xml_url)
    else:
        rpo_tags = []

    row = {
        "html_url": html_url,
        "xml_url": xml_url,
        "num_rpo_tags": len(rpo_tags)
    }

    for i, item in enumerate(rpo_tags, 1):
        row[f"Tag_{i}_source"] = item["source"]
        row[f"Tag_{i}_name"] = item["tag_name"]
        row[f"Tag_{i}_value_raw"] = item["value_raw"]
        row[f"Tag_{i}_unit"] = item["unit"]
        row[f"Tag_{i}_period_type"] = item["period_type"]
        row[f"Tag_{i}_instant"] = item["instant"]
        row[f"Tag_{i}_startDate"] = item["startDate"]
        row[f"Tag_{i}_endDate"] = item["endDate"]
        row[f"Tag_{i}_dimensions"] = item["dimensions"]

    return row, rpo_tags


def _process_one(filing: dict) -> dict:
    url = filing.get("html")
    try:
        if pd.isna(url) or not str(url).strip():
            raise ValueError("Missing HTML URL")
        row, _ = process_filing(url)
        row["error"] = None
    except Exception as e:
        row = {"html_url": url, "xml_url": None, "num_rpo_tags": 0, "error": str(e)}

    row["CIK"] = filing.get("CIK")
    row["Company"] = filing.get("Company")
    row["FiscalYear"] = filing.get("fiscalYear")
    if "gvkey" in filing:
        row["gvkey"] = filing.get("gvkey")
    return row


def _save(output_prefix: str, rows: list) -> pd.DataFrame:
    df_out = pd.DataFrame(rows)
    df_out.to_excel(f"{output_prefix}.xlsx", index=False)
    df_out.to_csv(f"{output_prefix}.csv", index=False, encoding="utf-8-sig")
    return df_out


def run_rpo_pipeline(df: pd.DataFrame = None,
                      input_path: str = None,
                      output_prefix: str = "rpo_new_xbrl_output",
                      max_workers: int = 6,
                      checkpoint_every: int = 100,
                      resume: bool = True) -> pd.DataFrame:
    """
    Run RPO extraction over every filing in `df` (or the excel/csv at
    `input_path`), which must have columns `html`, `CIK`, `Company`,
    `fiscalYear` (the shape produced by
    html_generator.export_company_cik_year_html). Checkpoints to
    `{output_prefix}.xlsx/.csv` every `checkpoint_every` filings; when
    `resume=True`, skips html URLs already present in an existing
    `{output_prefix}.csv` so an interrupted run can continue.
    """
    if df is None:
        if input_path is None:
            raise ValueError("Provide either df or input_path")
        df = pd.read_excel(input_path) if str(input_path).endswith((".xlsx", ".xls")) else pd.read_csv(input_path)

    out_csv = f"{output_prefix}.csv"

    rows = []
    done_urls = set()

    if resume and os.path.exists(out_csv):
        try:
            existing = pd.read_csv(out_csv, encoding="utf-8-sig")
            rows.extend(existing.to_dict("records"))
            done_urls = set(existing["html_url"].dropna())
            print(f"Resuming: {len(done_urls)} filings already processed, skipping them.")
        except Exception as e:
            print(f"Could not read existing checkpoint ({e}), starting fresh.")

    todo = [
        filing for _, filing in df.iterrows()
        if str(filing.get("html", "")).strip() and filing["html"] not in done_urls
    ]
    print(f"{len(todo)} filings to process ({len(done_urls)} already done).")

    lock = threading.Lock()
    completed = 0
    total = len(todo)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_process_one, filing) for filing in todo]
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                rows.append(row)
                completed += 1

                if completed % 25 == 0 or completed == total:
                    print(f"[{completed}/{total}] processed this run")

                if completed % checkpoint_every == 0:
                    _save(output_prefix, rows)
                    print(f"Checkpoint saved: {output_prefix}.xlsx/.csv")

    df_out = _save(output_prefix, rows)
    print(f"Done. Total rows: {len(df_out)}")
    return df_out
