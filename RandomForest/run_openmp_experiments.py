#!/usr/bin/env python3
"""
Strong-scaling sweep for the OpenMP Random Forest, with CPU energy.

Datasets and hyperparameters match python/benchmark_comparison.py.
Energy comes from python/measure_energy.py: Intel RAPL through Windows PDH
(package counter) and, if present, NVIDIA NVML. The OpenMP binary itself
does not report joules.

On Windows, from the RandomForest directory:

    python run_openmp_experiments.py

Writes openmp_scaling.csv and raw logs under openmp_scaling_logs/.
cpu_j / cpu_edp are the package-counter figures for the CPU run.
edp is the monitor total (CPU + GPU) times wall time, same definition
used by benchmark_comparison.py.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

from measure_energy import EnergyMonitor  # noqa: E402

THREADS = [1, 2, 4, 8, 16, 32]

# name, data, target (None = last column), trees, max_depth, min_samples, mtry
DATASETS = [
    ("breast-cancer", ROOT / "data" / "breast-cancer.csv", "diagnosis", 300, 6, 5, 3),
    ("covtype", ROOT / "data" / "covtype.csv", "Cover_Type", 50, 30, 2, 18),
    ("iris", ROOT / "data" / "iris.csv", None, 300, 12, 5, 2),
    ("letter-recognition", ROOT / "data" / "letter-recognition.data", "0", 300, 30, 2, 4),
]

OUT_PATH = ROOT / "openmp_scaling.csv"
LOG_DIR = ROOT / "openmp_scaling_logs"

COLUMNS = [
    "dataset",
    "threads",
    "train_samples",
    "test_samples",
    "features",
    "classes",
    "trees",
    "mtry",
    "max_depth",
    "min_samples",
    "seed",
    "train_accuracy",
    "test_accuracy",
    "train_wall_s",
    "predict_wall_s",
    "wall_s",
    "speedup_train",
    "speedup_predict",
    "speedup_wall",
    "measure_dt_s",
    "cpu_j",
    "gpu_j",
    "total_j",
    "dynamic_j",
    "cpu_power_w",
    "avg_power_w",
    "edp",
    "cpu_edp",
    "dynamic_edp",
    "speedup_cpu_energy",
    "speedup_edp",
    "speedup_cpu_edp",
    "idle_power_w",
]


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
    try:
        result = subprocess.run(["make", "-C", str(ROOT / "openmp")])
    except FileNotFoundError:
        print("make was not found; using an existing binary if one is present.")
        return
    if result.returncode != 0:
        print("make failed; using an existing binary if one is present.")


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


def ratio(base: float, value: float) -> float:
    if base > 0.0 and value > 0.0:
        return base / value
    return 0.0


def average(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def run_case(monitor, exe, spec, threads, repeats, warmup, idle_power_w):
    name, data, target, trees, depth, min_samples, mtry = spec
    if not data.is_file():
        sys.exit(f"dataset file not found: {data}")

    cmd = [
        str(exe),
        "--data",
        str(data),
        "--trees",
        str(trees),
        "--max-depth",
        str(depth),
        "--min-samples",
        str(min_samples),
        "--mtry",
        str(mtry),
        "--test-frac",
        "0.2",
        "--seed",
        "42",
    ]
    if target is not None:
        cmd.extend(["--target", target])

    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_DYNAMIC"] = "false"

    print(f"==> {name}  OMP_NUM_THREADS={threads}")

    if warmup:
        warm = monitor.measure_command(cmd, idle_power_w=idle_power_w)
        if warm.returncode != 0:
            sys.stderr.write(warm.stderr or "")
            sys.exit(f"{name} warmup failed (exit {warm.returncode})")
        time.sleep(1.0)

    measurements = []
    for i in range(repeats):
        measured = monitor.measure_command(cmd, idle_power_w=idle_power_w)
        if measured.returncode != 0:
            sys.stderr.write(measured.stderr or "")
            sys.exit(f"{name} threads={threads} failed (exit {measured.returncode})")
        measurements.append(measured)
        if i < repeats - 1:
            time.sleep(1.0)

    parsed_runs = [parse_stdout(m.stdout or "") for m in measurements]
    parsed = parsed_runs[-1]
    train_s = average(fnum(p.get("train_wall_s")) for p in parsed_runs)
    predict_s = average(fnum(p.get("predict_wall_s")) for p in parsed_runs)
    wall_s = average(fnum(p.get("wall_s")) for p in parsed_runs)

    dt = average(m.dt for m in measurements)
    cpu_j = average(m.cpu_j for m in measurements)
    gpu_j = average(m.gpu_j for m in measurements)
    total_j = average(m.total_j for m in measurements)
    dynamic_j = average(m.dynamic_j for m in measurements)
    cpu_power_w = average(m.cpu_power_w for m in measurements)
    avg_power_w = average(m.avg_power_w for m in measurements)
    edp = average(m.edp for m in measurements)
    dynamic_edp = average(m.dynamic_edp for m in measurements)
    cpu_edp = cpu_j * dt

    log_path = LOG_DIR / f"{name}_t{threads}.log"
    log_body = measurements[-1].stdout or ""
    if measurements[-1].stderr:
        log_body += "\n" + measurements[-1].stderr
    log_body += (
        f"\n# energy cpu_j={cpu_j:.6f} gpu_j={gpu_j:.6f} total_j={total_j:.6f} "
        f"dynamic_j={dynamic_j:.6f} cpu_power_w={cpu_power_w:.6f} "
        f"edp={edp:.6f} cpu_edp={cpu_edp:.6f} dynamic_edp={dynamic_edp:.6f} "
        f"measure_dt_s={dt:.6f} repeats={repeats}\n"
    )
    log_path.write_text(log_body, encoding="utf-8")

    print(
        f"    wall={wall_s:.4f}s  cpu={cpu_j:.4f} J  "
        f"power={cpu_power_w:.2f} W  cpu_edp={cpu_edp:.4f} J*s"
    )

    return {
        "dataset": name,
        "threads": parsed.get("threads", threads),
        "train_samples": parsed.get("train", ""),
        "test_samples": parsed.get("test", ""),
        "features": parsed.get("features", ""),
        "classes": parsed.get("classes", ""),
        "trees": parsed.get("trees", ""),
        "mtry": parsed.get("mtry", ""),
        "max_depth": parsed.get("max_depth", ""),
        "min_samples": parsed.get("min_samples", ""),
        "seed": parsed.get("seed", ""),
        "train_accuracy": parsed.get("train_accuracy", ""),
        "test_accuracy": parsed.get("test_accuracy", ""),
        "train_wall_s": f"{train_s:.6f}",
        "predict_wall_s": f"{predict_s:.6f}",
        "wall_s": f"{wall_s:.6f}",
        "measure_dt_s": f"{dt:.6f}",
        "cpu_j": f"{cpu_j:.6f}",
        "gpu_j": f"{gpu_j:.6f}",
        "total_j": f"{total_j:.6f}",
        "dynamic_j": f"{dynamic_j:.6f}",
        "cpu_power_w": f"{cpu_power_w:.6f}",
        "avg_power_w": f"{avg_power_w:.6f}",
        "edp": f"{edp:.6f}",
        "cpu_edp": f"{cpu_edp:.6f}",
        "dynamic_edp": f"{dynamic_edp:.6f}",
        "idle_power_w": f"{idle_power_w:.6f}",
        "_train_s": train_s,
        "_predict_s": predict_s,
        "_wall_s": wall_s,
        "_cpu_j": cpu_j,
        "_edp": edp,
        "_cpu_edp": cpu_edp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Measured repetitions averaged into each CSV row (default: 1).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Discard one preliminary run before measuring.",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=3.0,
        help="Seconds used to sample idle power before the sweep (default: 3).",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        sys.exit("--repeats must be >= 1")

    build()
    exe = find_exe()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    baselines = {}
    with EnergyMonitor() as monitor, OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        handle.flush()

        if monitor.cpu_active:
            print("RAPL package counter is active.")
        else:
            print(
                "RAPL/PDH counter is not active. Energy columns will be 0. "
                "Run this on Windows where the Energy Meter counter is available."
            )
        if monitor.nvml_active:
            print("NVML is active; gpu_j is included in total_j and edp.")

        print(f"Measuring idle power for {args.idle_seconds:.1f} s ...")
        idle_power_w = monitor.measure_idle_power(duration_s=args.idle_seconds)
        print(f"Idle power: {idle_power_w:.2f} W")

        for spec in DATASETS:
            name = spec[0]
            for threads in THREADS:
                row = run_case(
                    monitor,
                    exe,
                    spec,
                    threads,
                    args.repeats,
                    args.warmup,
                    idle_power_w,
                )
                if threads == 1:
                    baselines[name] = row
                    row["speedup_train"] = "1.0000"
                    row["speedup_predict"] = "1.0000"
                    row["speedup_wall"] = "1.0000"
                    row["speedup_cpu_energy"] = "1.0000"
                    row["speedup_edp"] = "1.0000"
                    row["speedup_cpu_edp"] = "1.0000"
                else:
                    base = baselines[name]
                    row["speedup_train"] = f"{ratio(base['_train_s'], row['_train_s']):.4f}"
                    row["speedup_predict"] = f"{ratio(base['_predict_s'], row['_predict_s']):.4f}"
                    row["speedup_wall"] = f"{ratio(base['_wall_s'], row['_wall_s']):.4f}"
                    row["speedup_cpu_energy"] = f"{ratio(base['_cpu_j'], row['_cpu_j']):.4f}"
                    row["speedup_edp"] = f"{ratio(base['_edp'], row['_edp']):.4f}"
                    row["speedup_cpu_edp"] = f"{ratio(base['_cpu_edp'], row['_cpu_edp']):.4f}"

                writer.writerow(row)
                handle.flush()

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
