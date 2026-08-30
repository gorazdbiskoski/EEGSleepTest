"""Run every optimizer that does not yet have both of its result CSVs, then
write a status manifest.

Each optimizer writes ``<prefix>_haaglanden.csv`` and ``<prefix>_sleep_edf.csv``
into ``results/``. A script counts as done only when BOTH exist -- a script can
be killed (or swallow an exception) after finishing one dataset and still exit 0,
which is exactly how genetic_algorithm.py left a half-finished run behind.

Usage:
    python run_missing.py                 # run outstanding scripts, 3 at a time
    python run_missing.py --dry-run       # show what would run, run nothing
    python run_missing.py --jobs 2        # lower concurrency if RAM is tight
    python run_missing.py --only random_search.py
"""

import argparse
import concurrent.futures
import datetime
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
CACHE_DIR = os.path.join(REPO_ROOT, "data", "cache")
MANIFEST_PATH = os.path.join(REPO_ROOT, "RUN_MANIFEST.md")

# Script -> result CSV prefix. Must be explicit: the prefix does not always
# match the script name (hill_climbing_with_simulated_annealing writes
# "basin_hopping_*"), so inferring it from the filename would misreport.
SCRIPT_OUTPUTS = {
    "ant_colony_optimization.py": "aco_results",
    "bayesian_optimization_algorithm.py": "bayesian_optimization_results",
    "genetic_algorithm.py": "ga_results",
    "grid_search.py": "grid_search_results",
    "hill_climbing.py": "hill_climbing_results",
    "hill_climbing_with_simulated_annealing.py": "basin_hopping_results",
    "pso_optimizer.py": "pso_results",
    "random_search.py": "random_search_results",
    "simulated_annealing.py": "simulated_annealing_results",
    "tabu_search.py": "tabu_search_results",
}

DATASET_SUFFIXES = ("haaglanden", "sleep_edf")

EXPECTED_CACHE = ("haaglanden_8000_42.npz", "sleep_edfx_8000_42.npz")

ERROR_TAIL_LINES = 25


def expected_csvs(script):
    prefix = SCRIPT_OUTPUTS[script]
    return [f"{prefix}_{suffix}.csv" for suffix in DATASET_SUFFIXES]


def present_csvs(script, results_dir):
    return [
        name
        for name in expected_csvs(script)
        if os.path.exists(os.path.join(results_dir, name))
    ]


def find_outstanding(results_dir, only=None):
    """Scripts with fewer than both dataset CSVs on disk."""
    outstanding, complete = [], []
    for script in SCRIPT_OUTPUTS:
        if only and script not in only:
            continue
        if len(present_csvs(script, results_dir)) == len(DATASET_SUFFIXES):
            complete.append(script)
        else:
            outstanding.append(script)
    return outstanding, complete


def check_cache():
    """Workers only read the cache, but if it were missing each would decode
    ~24 GB of EDF and race on the write. Refuse to fan out in that case."""
    missing = [
        name for name in EXPECTED_CACHE
        if not os.path.exists(os.path.join(CACHE_DIR, name))
    ]
    return missing


def child_env(threads_per_job):
    env = dict(os.environ)
    # Children import `data`/`model` from the repo root.
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # Without caps, N processes each grab every core and thrash.
    env["OMP_NUM_THREADS"] = str(threads_per_job)
    env["TF_NUM_INTRAOP_THREADS"] = str(threads_per_job)
    env["TF_NUM_INTEROP_THREADS"] = "2"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return env


def run_script(script, env, results_dir):
    """Run one optimizer to completion, teeing output to its own log."""
    log_path = os.path.join(LOGS_DIR, f"{os.path.splitext(script)[0]}.log")
    started = time.time()

    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(f"=== {script} started {datetime.datetime.now().isoformat()} ===\n")
        log.flush()
        proc = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, script)],
            check=False,
            cwd=BASE_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - started

    return {
        "script": script,
        "returncode": proc.returncode,
        "elapsed": elapsed,
        "log_path": log_path,
        "csvs": present_csvs(script, results_dir),
    }


def read_tail(path, n_lines):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return ["(log unreadable)"]
    return lines[-n_lines:] if lines else ["(log empty)"]


def classify(record):
    """COMPLETE / FAILED / PARTIAL -- exit code alone is not enough."""
    if record["returncode"] != 0:
        return "FAILED"
    if len(record["csvs"]) < len(DATASET_SUFFIXES):
        return "PARTIAL"
    return "COMPLETE"


def humanize(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def write_manifest(records, skipped, jobs, threads_per_job, started_at,
                   total_elapsed, results_dir):
    rows = []
    for script in SCRIPT_OUTPUTS:
        record = next((r for r in records if r["script"] == script), None)
        if record is None:
            rows.append({
                "script": script,
                "status": "SKIPPED",
                "returncode": "-",
                "elapsed": "-",
                "csvs": present_csvs(script, results_dir),
                "log_path": None,
            })
        else:
            rows.append({
                "script": script,
                "status": classify(record),
                "returncode": record["returncode"],
                "elapsed": humanize(record["elapsed"]),
                "csvs": record["csvs"],
                "log_path": record["log_path"],
            })

    problems = [r for r in rows if r["status"] in ("FAILED", "PARTIAL")]
    succeeded = [r for r in rows if r["status"] == "COMPLETE"]

    out = []
    out.append("# Optimizer Run Manifest\n")
    out.append(
        f"Generated **{started_at.strftime('%Y-%m-%d %H:%M:%S')}** - "
        f"total wall clock **{humanize(total_elapsed)}**, "
        f"`--jobs {jobs}`, {threads_per_job} threads/worker.\n"
    )
    out.append(
        "A script counts as complete only when **both** dataset CSVs exist "
        "(`*_haaglanden.csv` and `*_sleep_edf.csv`). Exit code alone is not "
        "sufficient: a script can be killed after finishing one dataset and "
        "still report success.\n"
    )

    out.append("| Status | Meaning |")
    out.append("|---|---|")
    out.append("| `COMPLETE` | exit 0 and both CSVs present |")
    out.append("| `FAILED` | non-zero exit code |")
    out.append("| `PARTIAL` | exit 0 but fewer than 2 CSVs - silent failure |")
    out.append("| `SKIPPED` | already had both CSVs before this run |")
    out.append("")

    if problems:
        names = ", ".join(f"`{r['script']}`" for r in problems)
        out.append(
            f"> **{len(problems)} script(s) did not fully compute:** {names}\n"
        )
    else:
        out.append("> **All scripts that ran completed successfully.**\n")

    out.append("## Status\n")
    out.append("| Script | Status | Exit | Duration | CSVs | Log |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        log_cell = (
            f"`results/logs/{os.path.basename(r['log_path'])}`"
            if r["log_path"] else "-"
        )
        out.append(
            f"| `{r['script']}` | **{r['status']}** | {r['returncode']} | "
            f"{r['elapsed']} | {len(r['csvs'])}/2 | {log_cell} |"
        )
    out.append("")

    out.append(
        f"**{len(succeeded)} complete this run | {len(problems)} problem(s) | "
        f"{len(skipped)} skipped**\n"
    )

    if problems:
        out.append("## Failures\n")
        for r in problems:
            out.append(f"### `{r['script']}` - {r['status']}\n")
            out.append(f"- Exit code: `{r['returncode']}`")
            out.append(f"- Duration: {r['elapsed']}")
            missing = [c for c in expected_csvs(r["script"]) if c not in r["csvs"]]
            out.append(f"- Produced: {r['csvs'] or 'nothing'}")
            out.append(f"- Missing: {missing}")
            out.append(f"- Full log: `results/logs/{os.path.basename(r['log_path'])}`\n")
            out.append(f"Last {ERROR_TAIL_LINES} lines:\n")
            out.append("```")
            out.extend(read_tail(r["log_path"], ERROR_TAIL_LINES))
            out.append("```\n")

        out.append("## Next steps\n")
        for r in problems:
            if r["status"] == "FAILED":
                out.append(
                    f"- `{r['script']}` - crashed. Read the traceback in its log, "
                    f"fix, then re-run: `python run_missing.py --only {r['script']}`"
                )
            else:
                out.append(
                    f"- `{r['script']}` - exited 0 but only wrote "
                    f"{len(r['csvs'])}/2 CSVs, so it was almost certainly "
                    f"interrupted. Re-run: "
                    f"`python run_missing.py --only {r['script']}`"
                )
        out.append("")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))

    return rows, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=3,
                        help="concurrent optimizers (default 3)")
    parser.add_argument("--threads-per-job", type=int, default=6,
                        help="thread cap per worker (default 6)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would run, then exit")
    parser.add_argument("--only", nargs="+", metavar="SCRIPT",
                        help="restrict to these script filenames")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help=argparse.SUPPRESS)  # test hook
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)

    if args.only:
        unknown = [s for s in args.only if s not in SCRIPT_OUTPUTS]
        if unknown:
            parser.error(f"unknown script(s): {unknown}")

    outstanding, complete = find_outstanding(results_dir, only=args.only)

    print(f"results dir: {results_dir}\n")
    print(f"{'SCRIPT':<48} {'CSVs':<6} ACTION")
    print("-" * 72)
    for script in SCRIPT_OUTPUTS:
        if args.only and script not in args.only:
            continue
        n = len(present_csvs(script, results_dir))
        action = "run" if script in outstanding else "skip (complete)"
        print(f"{script:<48} {n}/2    {action}")
    print("-" * 72)
    print(f"{len(outstanding)} to run, {len(complete)} already complete\n")

    if args.dry_run:
        print("--dry-run: nothing executed.")
        return 0

    if not outstanding:
        print("Nothing to run.")
        return 0

    missing_cache = check_cache()
    if missing_cache:
        print(
            f"ERROR: dataset cache missing {missing_cache} in {CACHE_DIR}.\n"
            f"With {args.jobs} workers each would decode ~24 GB of EDF and race "
            f"on the cache write.\nBuild it once first:\n"
            f"  python -c \"import sys; sys.path.insert(0, r'{REPO_ROOT}'); "
            f"from data.global_data_loader import get_data_all_datasets; "
            f"get_data_all_datasets()\"",
            file=sys.stderr,
        )
        return 2

    os.makedirs(LOGS_DIR, exist_ok=True)
    env = child_env(args.threads_per_job)
    started_at = datetime.datetime.now()
    t0 = time.time()

    print(f"Running {len(outstanding)} script(s), {args.jobs} at a time "
          f"({args.threads_per_job} threads each). Logs: {LOGS_DIR}\n")

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_script, script, env, results_dir): script
            for script in outstanding
        }
        for future in concurrent.futures.as_completed(futures):
            script = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # launching failed, not the script itself
                record = {
                    "script": script, "returncode": -1, "elapsed": 0.0,
                    "log_path": os.path.join(
                        LOGS_DIR, f"{os.path.splitext(script)[0]}.log"),
                    "csvs": present_csvs(script, results_dir),
                }
                print(f"  !! {script} could not be launched: {exc}")

            records.append(record)
            status = classify(record)
            print(f"  [{status:<8}] {script:<48} "
                  f"exit={record['returncode']} "
                  f"{humanize(record['elapsed']):>7} "
                  f"csvs={len(record['csvs'])}/2")

    total_elapsed = time.time() - t0
    rows, problems = write_manifest(
        records, complete, args.jobs, args.threads_per_job,
        started_at, total_elapsed, results_dir,
    )

    print(f"\n{'=' * 72}")
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Total wall clock: {humanize(total_elapsed)}")
    if problems:
        print(f"{len(problems)} script(s) did not fully compute:")
        for r in problems:
            print(f"  - {r['script']} ({r['status']})")
        return 1
    print("All scripts that ran completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
