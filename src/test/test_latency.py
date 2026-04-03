"""
src/test/latency_benchmark.py
------------------------------
Sequential latency benchmark for the RecSys serving API.

Experiment
----------
  - Takes a month (e.g. 2018-07) and uses the last day as cutoff
  - Builds a globally chronological request list:
      (user_id, as_of_date) sorted by actual interaction timestamp
  - Runs the list sequentially against GET /recommendations/{user_id}
  - Repeats for 4 conditions:
      GPU + cache ON
      GPU + cache OFF
      CPU + cache ON
      CPU + cache OFF
  - Reports P50, P95, P99, mean latency + cache hit rate per condition

Usage
-----
  python src/test/latency_benchmark.py --month 2018-07
  python src/test/latency_benchmark.py --month 2018-09 --top_n 20 --base_url http://localhost:8001
"""

import argparse
import calendar
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from rich.console import Console
from rich.table import Table

from src.data.db_simulator import MovieLensDB

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--month",    type=str, default="2018-07",
                   help="Month to benchmark, format YYYY-MM (default: 2018-07)")
    p.add_argument("--top_n",   type=int, default=10,
                   help="top_n param for /recommendations (default: 10)")
    p.add_argument("--base_url", type=str, default="http://localhost:8001",
                   help="API base URL (default: http://localhost:8001)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Build request list
# ─────────────────────────────────────────────────────────────────────────────

def build_request_list(month_str: str) -> list[dict]:
    """
    Filter all interactions within the given month.
    Sort globally by timestamp (chronological).
    Return list of {user_id, as_of_date} dicts.

    as_of_date = the date of that specific interaction
    → each request uses the history the user had UP TO that moment
    """
    year, mon = map(int, month_str.split("-"))
    last_day  = calendar.monthrange(year, mon)[1]
    start     = pd.Timestamp(f"{year}-{mon:02d}-01")
    end       = pd.Timestamp(f"{year}-{mon:02d}-{last_day}")

    db = MovieLensDB()
    db.load_data()

    df = db.ratings_df.copy()
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    df = df.sort_values("timestamp").reset_index(drop=True)

    requests_list = [
        {
            "user_id":    int(row["userId"]),
            "as_of_date": str(row["timestamp"].date()),
        }
        for _, row in df.iterrows()
    ]

    console.print(
        f"[bold cyan]Month:[/bold cyan] {month_str}  |  "
        f"[bold cyan]Cutoff:[/bold cyan] {end.date()}  |  "
        f"[bold cyan]Total requests:[/bold cyan] {len(requests_list):,}  |  "
        f"[bold cyan]Unique users:[/bold cyan] {df['userId'].nunique():,}"
    )
    return requests_list


# ─────────────────────────────────────────────────────────────────────────────
# Server control helpers
# ─────────────────────────────────────────────────────────────────────────────

def clear_cache(base_url: str):
    res = requests.get(f"{base_url}/cache/clear", timeout=10)
    res.raise_for_status()


def get_cache_stats(base_url: str) -> dict:
    res = requests.get(f"{base_url}/cache/stats", timeout=10)
    res.raise_for_status()
    return res.json()


def check_health(base_url: str) -> dict:
    res = requests.get(f"{base_url}/health", timeout=10)
    res.raise_for_status()
    return res.json()


# ─────────────────────────────────────────────────────────────────────────────
# Run one experiment condition
# ─────────────────────────────────────────────────────────────────────────────

def run_condition(
    requests_list: list[dict],
    base_url:      str,
    top_n:         int,
    label:         str,
) -> dict:
    """
    Fire all requests sequentially, collect latency_ms from response.
    Returns stats dict.
    """
    clear_cache(base_url)

    latencies = []
    errors    = 0

    console.print(f"\n[bold yellow]Running:[/bold yellow] {label}  ({len(requests_list):,} requests)")

    for i, req in enumerate(requests_list):
        url = (
            f"{base_url}/recommendations/{req['user_id']}"
            f"?as_of_date={req['as_of_date']}&top_n={top_n}"
        )
        try:
            t0  = time.perf_counter()
            res = requests.get(url, timeout=30)
            t1  = time.perf_counter()

            if res.status_code == 200:
                # use server-reported latency (excludes network overhead)
                data = res.json()
                latencies.append(data["latency_ms"])
            else:
                errors += 1

        except Exception:
            errors += 1

        if (i + 1) % 100 == 0:
            console.print(
                f"  {i+1:>5}/{len(requests_list)}  "
                f"p50={np.percentile(latencies, 50):.1f}ms  "
                f"errors={errors}"
            )

    latencies = np.array(latencies)
    stats     = get_cache_stats(base_url)

    l2  = stats["L2_candidate_cache"]
    l1  = stats["L1_output_cache"]

    return {
        "label":             label,
        "n_requests":        len(requests_list),
        "errors":            errors,
        "p50_ms":            round(float(np.percentile(latencies, 50)),  3),
        "p95_ms":            round(float(np.percentile(latencies, 95)),  3),
        "p99_ms":            round(float(np.percentile(latencies, 99)),  3),
        "mean_ms":           round(float(np.mean(latencies)),            3),
        "l1_hit_rate_pct":   l1["hit_rate_pct"],
        "l2_hit_rate_pct":   l2["hit_rate_pct"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print results table
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: list[dict]):
    table = Table(title="Latency Benchmark Results", show_lines=True)

    table.add_column("Condition",        style="bold cyan",  no_wrap=True)
    table.add_column("Requests",         justify="right")
    table.add_column("P50 (ms)",         justify="right", style="green")
    table.add_column("P95 (ms)",         justify="right", style="yellow")
    table.add_column("P99 (ms)",         justify="right", style="red")
    table.add_column("Mean (ms)",        justify="right")
    table.add_column("L1 Hit %",         justify="right")
    table.add_column("L2 Hit %",         justify="right")
    table.add_column("Errors",           justify="right", style="red")

    for r in results:
        table.add_row(
            r["label"],
            str(r["n_requests"]),
            str(r["p50_ms"]),
            str(r["p95_ms"]),
            str(r["p99_ms"]),
            str(r["mean_ms"]),
            f"{r['l1_hit_rate_pct']}%",
            f"{r['l2_hit_rate_pct']}%",
            str(r["errors"]),
        )

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── health check ──────────────────────────────────────────────────────────
    try:
        health = check_health(args.base_url)
        console.print(
            f"[bold green]Server healthy[/bold green]  →  "
            f"device={health['device']}"
        )
    except Exception as e:
        console.print(f"[bold red]Server not reachable:[/bold red] {e}")
        return

    # ── build request list once (shared across all conditions) ────────────────
    requests_list = build_request_list(args.month)

    if not requests_list:
        console.print("[bold red]No interactions found for this month.[/bold red]")
        return

    # ── 4 conditions ─────────────────────────────────────────────────────────
    # The script assumes the server is already running with the right
    # FORCE_CPU / CACHE_ENABLED settings. Instructions printed below.
    # Run the script once per condition, results saved to CSV each time.

    console.print("\n[bold]NOTE:[/bold] Run this script once per condition.")
    console.print("Set env vars before starting the server:\n")
    console.print("  GPU  + Cache ON  :  FORCE_CPU=0  CACHE_ENABLED=1")
    console.print("  GPU  + Cache OFF :  FORCE_CPU=0  CACHE_ENABLED=0")
    console.print("  CPU  + Cache ON  :  FORCE_CPU=1  CACHE_ENABLED=1")
    console.print("  CPU  + Cache OFF :  FORCE_CPU=1  CACHE_ENABLED=0\n")

    device = health["device"]
    cache_enabled_flag = input("Is cache ON for this run? (y/n): ").strip().lower() == "y"
    cache_label        = "cache=ON" if cache_enabled_flag else "cache=OFF"
    label              = f"{device.upper()} | {cache_label}"

    result = run_condition(
        requests_list = requests_list,
        base_url      = args.base_url,
        top_n         = args.top_n,
        label         = label,
    )

    # ── print ─────────────────────────────────────────────────────────────────
    print_results([result])

    # ── save CSV (append mode so all 4 runs accumulate) ───────────────────────
    out_path = f"results/latency_benchmark_{args.month}.csv"
    import os
    os.makedirs("results", exist_ok=True)

    df_row = pd.DataFrame([{**result, "month": args.month, "top_n": args.top_n}])
    write_header = not os.path.exists(out_path)
    df_row.to_csv(out_path, mode="a", header=write_header, index=False)

    console.print(f"\n[bold green]Results appended →[/bold green] {out_path}")
    console.print("Run again with a different server config to add another condition.")


if __name__ == "__main__":
    main()