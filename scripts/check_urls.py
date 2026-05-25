"""Pre-flight URL validator — test reachability before bulk ingestion.

Reads a text file with one URL per line (ignores blank lines and lines
starting with '#'), performs a HEAD request against each, and reports:

  status_code  content_type               size   url
  ----------   -------------------------  -----  --------------------------
  200          application/pdf            5.8 MB https://army.mil/fm.pdf
  403          -                          -      https://paho.org/doc.pdf
  TIMEOUT      -                          -      https://slow.example.com

Also generates a filtered file ``<input>.ok.txt`` with only URLs that
returned 2xx, so you can paste it directly into the bulk-URL UI.

Usage:
    python -m scripts.check_urls survival/urls_alpha.txt
    python -m scripts.check_urls survival/urls_beta.txt --timeout 30
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx


_USER_AGENT = "NCN/0.1 (pre-flight URL validator)"


def _load_urls(path: Path) -> list[str]:
    """Parse a URL-list file: skip blank lines and '#' comments."""
    urls: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def _human_size(bytes_count: int | None) -> str:
    if not bytes_count:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_count < 1024:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f} TB"


async def _check_one(url: str, timeout: float) -> dict:
    """HEAD request with redirect following; fall back to GET-stream if needed."""
    headers = {"User-Agent": _USER_AGENT}
    result = {"url": url, "status": "ERROR", "content_type": "-", "size": "-", "ok": False}
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            # Some servers don't like HEAD — try it first, fallback to a streamed GET
            try:
                resp = await client.head(url, headers=headers)
            except httpx.HTTPError:
                resp = None

            if resp is None or resp.status_code >= 400:
                # Try a tiny GET to see if the URL actually serves content
                async with client.stream("GET", url, headers=headers) as gresp:
                    status = gresp.status_code
                    content_type = gresp.headers.get("content-type", "-")
                    size = gresp.headers.get("content-length")
                    size_val = int(size) if size and size.isdigit() else None
            else:
                status = resp.status_code
                content_type = resp.headers.get("content-type", "-")
                size = resp.headers.get("content-length")
                size_val = int(size) if size and size.isdigit() else None

            result["status"] = str(status)
            result["content_type"] = (content_type or "-").split(";")[0].strip() or "-"
            result["size"] = _human_size(size_val)
            result["ok"] = 200 <= status < 300
    except httpx.TimeoutException:
        result["status"] = "TIMEOUT"
    except httpx.HTTPError as exc:
        result["status"] = f"HTTP_ERR"
        result["content_type"] = str(exc)[:40]
    except Exception as exc:
        result["status"] = "ERROR"
        result["content_type"] = str(exc)[:40]
    return result


async def _check_batch(urls: list[str], timeout: float, concurrent: int) -> list[dict]:
    """Check URLs in parallel (capped by a semaphore)."""
    sem = asyncio.Semaphore(concurrent)

    async def guarded(u):
        async with sem:
            return await _check_one(u, timeout)

    return await asyncio.gather(*[guarded(u) for u in urls])


def _print_report(results: list[dict]) -> None:
    print()
    print(f"{'status':<10} {'content-type':<30} {'size':>10}  url")
    print(f"{'-' * 10} {'-' * 30} {'-' * 10}  {'-' * 60}")
    for r in results:
        marker = "OK " if r["ok"] else "XX "
        print(
            f"{marker}{r['status']:<7} {r['content_type']:<30} "
            f"{r['size']:>10}  {r['url']}"
        )
    print()
    ok_count = sum(1 for r in results if r["ok"])
    print(f"Summary: {ok_count}/{len(results)} reachable\n")


def _write_filtered_file(input_path: Path, results: list[dict]) -> Path | None:
    """Write only the OK URLs to <input>.ok.txt. Returns the new path."""
    ok_urls = [r["url"] for r in results if r["ok"]]
    if not ok_urls:
        print("  (no OK urls -> no filtered file generated)\n")
        return None
    out_path = input_path.with_suffix(".ok.txt")
    out_path.write_text(
        "# Filtered by scripts/check_urls.py — only reachable URLs\n"
        + "\n".join(ok_urls)
        + "\n",
        encoding="utf-8",
    )
    print(f"  Filtered list (reachable only) -> {out_path}\n")
    return out_path


async def main(args) -> int:
    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    urls = _load_urls(path)
    if not urls:
        print(f"ERROR: no URLs found in {path}")
        return 1

    print(f"Checking {len(urls)} URLs from {path} (timeout={args.timeout}s, "
          f"concurrent={args.concurrent})...")

    results = await _check_batch(urls, args.timeout, args.concurrent)
    _print_report(results)
    _write_filtered_file(path, results)

    failures = [r for r in results if not r["ok"]]
    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to URL-list file (one URL per line)")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="Per-request timeout in seconds (default: 20)")
    parser.add_argument("--concurrent", type=int, default=4,
                        help="Max concurrent checks (default: 4)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with non-zero code if any URL fails")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
