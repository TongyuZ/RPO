"""
Shared HTTP helpers for talking to SEC EDGAR (www.sec.gov / data.sec.gov).

Centralizes:
- a single User-Agent (SEC requires one that identifies the requester)
- a process-wide rate limiter shared across threads, so concurrent
  scraping in html_generator.py and XBRL.py never collectively exceeds
  a safe request rate against SEC's servers
- retry-with-backoff for transient errors / 429s
"""

import threading
import time

import requests

HEADERS = {"User-Agent": "Research Script (contact: tz25@fsu.edu)"}


class RateLimiter:
    """Thread-safe limiter: at most `rate` calls/second across all callers."""

    def __init__(self, rate: float = 8.0):
        self._min_interval = 1.0 / rate
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            sleep_for = self._min_interval - (now - self._last)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()


# Shared across every module/thread in the process. SEC asks for a
# reasonable, well-identified request rate; 8/sec leaves headroom.
_limiter = RateLimiter(rate=8.0)


def sec_get(url: str, timeout: int = 30, max_retries: int = 4) -> requests.Response:
    last_exc = None
    for attempt in range(max_retries):
        _limiter.wait()
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            time.sleep(min(2 ** attempt, 10))
    raise last_exc


def sec_get_text(url: str, **kwargs) -> str:
    return sec_get(url, **kwargs).text


def sec_get_json(url: str, **kwargs) -> dict:
    return sec_get(url, **kwargs).json()
