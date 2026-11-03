#!/usr/bin/env python3
"""
Strong-scaling sweep for the OpenMP Random Forest, with CPU energy.

Datasets and hyperparameters are the ones reported in the article
(report/template/artigo.tex, Table 1). Energy comes from python/measure_energy.py:
Intel RAPL package counter read through Windows PDH. The OpenMP binary itself
does not report joules.

Protocol:
  - each dataset runs N blocks (REPEATS); a block runs every thread count once,
    in an order shuffled with a recorded seed (--order-seed);
  - one discarded warmup run per dataset, before its first block;
  - idle power is measured again before each dataset;
  - no thread affinity is set (libgomp on MinGW does not support OMP_PLACES /
    OMP_PROC_BIND); the Windows scheduler places the threads.

On Windows, from the RandomForest directory:

    python run_openmp_experiments.py

Writes one line per measured run to openmp_runs.csv and the per-configuration
statistics (mean, median, standard deviation) to openmp_scaling.csv.
To recompute the statistics from an existing openmp_runs.csv:

    python run_openmp_experiments.py --aggregate-only
"""

import argparse
import csv
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

from measure_energy import EnergyMonitor  # noqa: E402

THREADS = [1, 2, 4, 8, 12, 16, 32]

# name, data, target (None = last column), trees, max_depth, min_samples, mtry, repeats
DATASETS = [
    ("iris", ROOT / "data" / "iris.csv", None, 300, 12, 5, 2, 30),
    ("breast-cancer", ROOT / "data" / "breast-cancer.csv", "diagnosis", 300, 6, 5, 3, 30),
    ("letter-recognition", ROOT / "data" / "letter-recognition.data", "0", 300, 30, 2, 4, 10),
    ("covtype", ROOT / "data" / "covtype.csv", "Cover_Type", 50, 30, 2, 18, 5),
]

RUNS_PATH = ROOT / "openmp_runs.csv"
OUT_PATH = ROOT / "openmp_scaling.csv"

RUN_COLUMNS = [
    "run_index",
    "start_time",
    "dataset",
    "block",
    "pos_in_block",
    "order_seed",
    "threads",
    "train_accuracy",
    "test_accuracy",
    "train_wall_s",
    "predict_wall_s",
    "measure_dt_s",
    "cpu_j",
    "dynamic_j",
    "cpu_power_w",
    "cpu_edp",
    "idle_power_w",
]

# Metrics summarized per (dataset, threads).
METRICS = ["train_wall_s", "predict_wall_s", "measure_dt_s", "cpu_j", "cpu_power_w", "cpu_edp"]


def find_exe() -> Path:
    for name in ("rf.exe", "rf"):
        path = ROOT / "openmp" / name
        if path.is_file():
            return path
    sys.exit(
        "OpenMP binary not found (openmp/rf.exe or openmp/rf). "
        "Build it with 'make' inside openmp/."
    )


def build() -> None:
    print("Building openmp/rf ...")
    for make in ("make", "mingw32-make"):
        try:
            result = subprocess.run([make, "-C", str(ROOT / "openmp")])
        except FileNotFoundError:
            continue
        if result.returncode != 0:
            print(f"{make} failed; using an existing binary if one is present.")
        return
    print("make was not found; using an existing binary if one is present.")


def parse_stdout(text: str) -> dict:
    res = {}
    for line in text.splitlines():
        for token in line.split():
            if "=" in token:
                key, value = token.split("=", 1)
                res[key] = value
    return res


def fnum(text, default=0.0) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def command(exe, spec):
    name, data, target, trees, depth, min_samples, mtry, _ = spec
    if not data.is_file():
        sys.exit(f"dataset file not found: {data}")
    cmd = [
        str(exe),
        "--data", str(data),
        "--trees", str(trees),
        "--max-depth", str(depth),
        "--min-samples", str(min_samples),
        "--mtry", str(mtry),
        "--test-frac", "0.2",
        "--seed", "42",
    ]
    if target is not None:
        cmd.extend(["--target", target])
    return cmd


def run_once(monitor, cmd, name, threads, idle_power_w):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_DYNAMIC"] = "false"
    measured = monitor.measure_command(cmd, idle_power_w=idle_power_w)
    if measured.returncode != 0:
        sys.stderr.write(measured.stderr or "")
        sys.exit(f"{name} threads={threads} failed (exit {measured.returncode})")
    return measured, parse_stdout(measured.stdout or "")


def run_sweep(args) -> None:
    exe = find_exe()
    names = [d[0] for d in DATASETS]
    unknown = [n for n in (args.datasets or []) if n not in names]
    if unknown:
        sys.exit(f"unknown dataset(s): {', '.join(unknown)}; choose from {', '.join(names)}")
    selected = [d for d in DATASETS if not args.datasets or d[0] in args.datasets]

    existing = []
    if RUNS_PATH.is_file() and RUNS_PATH.stat().st_size > 0:
        with RUNS_PATH.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != RUN_COLUMNS:
                sys.exit(f"{RUNS_PATH} has a different header; move it away first")
            existing = list(reader)
    if existing and not (args.append or args.overwrite):
        sys.exit(
            f"{RUNS_PATH} already has {len(existing)} runs. "
            "Use --append to add runs or --overwrite to replace them."
        )
    append = args.append and RUNS_PATH.is_file() and RUNS_PATH.stat().st_size > 0
    mode = "a" if append else "w"
    run_index = len(existing) if append else 0
    # In append mode, block numbers continue per dataset so every block gets
    # its own shuffled order and (dataset, block) stays unique.
    first_block = {}
    if append:
        for r in existing:
            first_block[r["dataset"]] = max(first_block.get(r["dataset"], -1), int(r["block"])) + 1

    with EnergyMonitor(use_gpu=False) as monitor, RUNS_PATH.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS)
        if mode == "w":
            writer.writeheader()
        handle.flush()

        if not monitor.cpu_active:
            print(
                "RAPL/PDH counter is not active. Energy columns will be 0. "
                "Run this on Windows where the Energy Meter counter is available."
            )

        for spec in selected:
            name = spec[0]
            repeats = args.repeats or spec[7]
            cmd = command(exe, spec)

            time.sleep(args.pause)
            idle_power_w = monitor.measure_idle_power(duration_s=args.idle_seconds)
            print(f"==> {name}: {repeats} blocks of {THREADS}, idle {idle_power_w:.2f} W")

            # Warmup: loads the file into the OS cache; discarded.
            run_once(monitor, cmd, name, max(THREADS), idle_power_w)

            start_block = first_block.get(name, 0)
            for block in range(start_block, start_block + repeats):
                order = list(THREADS)
                random.Random(f"{args.order_seed}-{name}-{block}").shuffle(order)
                for pos, threads in enumerate(order):
                    time.sleep(args.pause)
                    start = datetime.now().isoformat(timespec="seconds")
                    m, parsed = run_once(monitor, cmd, name, threads, idle_power_w)
                    writer.writerow({
                        "run_index": run_index,
                        "start_time": start,
                        "dataset": name,
                        "block": block,
                        "pos_in_block": pos,
                        "order_seed": args.order_seed,
                        "threads": threads,
                        "train_accuracy": parsed.get("train_accuracy", ""),
                        "test_accuracy": parsed.get("test_accuracy", ""),
                        "train_wall_s": parsed.get("train_wall_s", ""),
                        "predict_wall_s": parsed.get("predict_wall_s", ""),
                        "measure_dt_s": f"{m.dt:.6f}",
                        "cpu_j": f"{m.cpu_j:.6f}",
                        "dynamic_j": f"{m.dynamic_j:.6f}",
                        "cpu_power_w": f"{m.cpu_power_w:.6f}",
                        "cpu_edp": f"{m.cpu_j * m.dt:.6f}",
                        "idle_power_w": f"{idle_power_w:.6f}",
                    })
                    handle.flush()
                    run_index += 1
                    print(
                        f"    block {block} p={threads:2d}  train={fnum(parsed.get('train_wall_s')):.4f}s  "
                        f"dt={m.dt:.3f}s  cpu={m.cpu_j:.2f} J"
                    )


def summarize(values):
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return {
        "mean": mean,
        "median": statistics.median(values),
        "std": std,
        "cv": std / mean if mean > 0 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate() -> None:
    with RUNS_PATH.open(newline="", encoding="utf-8") as fh:
        runs = list(csv.DictReader(fh))
    if not runs:
        sys.exit(f"no runs in {RUNS_PATH}")

    groups = {}
    for r in runs:
        groups.setdefault((r["dataset"], int(r["threads"])), []).append(r)

    columns = ["dataset", "threads", "n", "test_accuracy"]
    for metric in METRICS:
        columns += [f"{metric}_{s}" for s in ("mean", "median", "std", "cv", "min", "max")]
    columns += [
        "speedup_train", "speedup_train_min", "speedup_train_max",
        "speedup_predict", "speedup_predict_min", "speedup_predict_max",
        "speedup_wall", "speedup_cpu_energy", "speedup_cpu_edp",
    ]

    rows = []
    for (name, threads), group in sorted(groups.items()):
        row = {"dataset": name, "threads": threads, "n": len(group)}
        accs = sorted({g["test_accuracy"] for g in group})
        row["test_accuracy"] = "|".join(accs)
        for metric in METRICS:
            for key, value in summarize([fnum(g[metric]) for g in group]).items():
                row[f"{metric}_{key}"] = value
        rows.append(row)

    base = {r["dataset"]: r for r in rows if r["threads"] == 1}
    for row in rows:
        b = base.get(row["dataset"])
        if b is None:
            continue
        group = groups[(row["dataset"], row["threads"])]
        for stage, metric in (("train", "train_wall_s"), ("predict", "predict_wall_s")):
            t1 = b[f"{metric}_median"]
            tp = row[f"{metric}_median"]
            row[f"speedup_{stage}"] = t1 / tp if tp > 0 else 0.0
            # Per-run speedup against the single-thread median: its range
            # shows whether a gain holds in every run.
            per_run = [t1 / fnum(g[metric]) for g in group if fnum(g[metric]) > 0]
            row[f"speedup_{stage}_min"] = min(per_run) if per_run else 0.0
            row[f"speedup_{stage}_max"] = max(per_run) if per_run else 0.0
        for col, metric in (
            ("speedup_wall", "measure_dt_s"),
            ("speedup_cpu_energy", "cpu_j"),
            ("speedup_cpu_edp", "cpu_edp"),
        ):
            v = row[f"{metric}_median"]
            row[col] = b[f"{metric}_median"] / v if v > 0 else 0.0

    with OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()
            })
    print(f"Wrote {OUT_PATH} ({len(rows)} configurations, {len(runs)} runs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=0,
                        help="Blocks per dataset; 0 uses the per-dataset value in DATASETS.")
    parser.add_argument("--datasets", nargs="*", help="Run only these datasets.")
    parser.add_argument("--order-seed", type=int, default=20261103,
                        help="Seed that shuffles the thread order inside each block.")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="Seconds of rest before each run (default: 1).")
    parser.add_argument("--idle-seconds", type=float, default=10.0,
                        help="Seconds used to sample idle power before each dataset (default: 10).")
    parser.add_argument("--append", action="store_true",
                        help="Append to openmp_runs.csv; block numbers continue per dataset.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing openmp_runs.csv.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only recompute openmp_scaling.csv from openmp_runs.csv.")
    parser.add_argument("--no-build", action="store_true", help="Skip make.")
    args = parser.parse_args()
    if args.repeats < 0:
        sys.exit("--repeats must be >= 0")

    if not args.aggregate_only:
        if not args.no_build:
            build()
        run_sweep(args)
    aggregate()


if __name__ == "__main__":
    main()
