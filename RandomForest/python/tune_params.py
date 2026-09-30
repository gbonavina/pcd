#!/usr/bin/env python3
"""
tune_params.py

Busca em grade (grid search) dos hiperparametros do Random Forest usando a
implementacao OpenMP (`openmp/rf.exe`) como referencia, por ser a mais rapida
na CPU e numericamente equivalente as versoes Sequencial e CUDA.

Para cada combinacao de (trees, max_depth, min_samples, mtry) o script executa
o binario com varias sementes de particionamento (holdout) e agrega a acuracia
media de teste. A configuracao escolhida e a mais economica cujo desempenho
esteja dentro de uma tolerancia da melhor acuracia observada (principio da
parcimonia: evita gastar tempo/energia por ganho estatisticamente irrelevante).

Uso:
    python tune_params.py                      # todos os datasets do grid padrao
    python tune_params.py --dataset iris       # apenas um dataset
    python tune_params.py --seeds 42 7 1       # sementes personalizadas
    python tune_params.py --export-csv sweep.csv
"""

import argparse
import itertools
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPENMP_EXE = os.path.join(REPO_ROOT, "openmp", "rf.exe")
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Grades de busca por dataset. Os intervalos cobrem desde arvores rasas
# (subajustadas) ate arvores praticamente completas, e mtry de baixo a alto
# em torno de sqrt(n_features).
GRIDS: Dict[str, Dict[str, Any]] = {
    "iris": {
        "data": os.path.join(DATA_DIR, "iris.csv"),
        "target": None,
        "test_frac": 0.20,
        "trees": [100, 200, 300],
        "max_depth": [4, 8, 12, 16],
        "min_samples": [2, 5],
        "mtry": [1, 2, 3, 4],
    },
    "breast-cancer": {
        "data": os.path.join(DATA_DIR, "breast-cancer.csv"),
        "target": "diagnosis",
        "test_frac": 0.20,
        "trees": [100, 200, 300],
        "max_depth": [6, 10, 14, 20],
        "min_samples": [2, 5],
        "mtry": [3, 5, 8, 11, 15],
    },
    "sales_data": {
        "data": os.path.join(DATA_DIR, "sales_data.csv"),
        "target": "Product_Category",
        "test_frac": 0.20,
        "trees": [100, 200, 300],
        "max_depth": [6, 10, 16, 24],
        "min_samples": [2, 5],
        "mtry": [3, 5, 8, 11, 13],
    },
    "letter-recog": {
        "data": os.path.join(DATA_DIR, "letter-recognition.data"),
        "target": "0",
        "test_frac": 0.20,
        "trees": [100, 200, 300],
        "max_depth": [12, 20, 30],
        "min_samples": [2, 5],
        "mtry": [2, 4, 6, 8],
    },
    # covtype ja foi calibrado previamente (depth 30, mtry 18, trees 50).
    # Mantido fora do grid padrao por custo: cada execucao leva dezenas de
    # segundos. Use --dataset covtype para reavaliar.
    "covtype": {
        "data": os.path.join(DATA_DIR, "covtype.csv"),
        "target": "Cover_Type",
        "test_frac": 0.20,
        "trees": [50],
        "max_depth": [20, 30],
        "min_samples": [2],
        "mtry": [8, 18],
        "default_off": True,
    },
}


def parse_kv(out: str) -> Dict[str, str]:
    res: Dict[str, str] = {}
    for line in out.splitlines():
        for token in line.split():
            if "=" in token:
                k, v = token.split("=", 1)
                res[k] = v
    return res


def run_once(
    data: str,
    target: Optional[str],
    trees: int,
    max_depth: int,
    min_samples: int,
    mtry: int,
    test_frac: float,
    seed: int,
) -> Optional[Dict[str, float]]:
    cmd = [
        OPENMP_EXE,
        "--data", data,
        "--trees", str(trees),
        "--max-depth", str(max_depth),
        "--min-samples", str(min_samples),
        "--mtry", str(mtry),
        "--test-frac", f"{test_frac:.4f}",
        "--seed", str(seed),
    ]
    if target:
        cmd += ["--target", target]

    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        sys.stderr.write(f"falha ao executar: {' '.join(cmd)}\n{proc.stderr}\n")
        return None

    kv = parse_kv(proc.stdout)
    try:
        return {
            "test_acc": float(kv["test_accuracy"]),
            "train_acc": float(kv["train_accuracy"]),
            "train_s": float(kv["train_wall_s"]),
            "features": float(kv["features"]),
        }
    except (KeyError, ValueError):
        sys.stderr.write(f"saida inesperada:\n{proc.stdout}\n")
        return None


def sweep(name: str, cfg: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    combos = list(
        itertools.product(
            cfg["trees"], cfg["max_depth"], cfg["min_samples"], cfg["mtry"]
        )
    )
    print(f"\n=== {name}: {len(combos)} configuracoes x {len(seeds)} sementes ===")
    results: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    for idx, (trees, depth, min_s, mtry) in enumerate(combos, start=1):
        accs: List[float] = []
        train_accs: List[float] = []
        times: List[float] = []
        n_features = 0
        for seed in seeds:
            r = run_once(
                cfg["data"], cfg["target"], trees, depth, min_s, mtry,
                cfg["test_frac"], seed,
            )
            if r is None:
                break
            accs.append(r["test_acc"])
            train_accs.append(r["train_acc"])
            times.append(r["train_s"])
            n_features = int(r["features"])
        if len(accs) != len(seeds):
            continue

        # mtry e saturado em n_features pelo binario; descarta duplicatas.
        eff_mtry = min(mtry, n_features)
        results.append(
            {
                "dataset": name,
                "trees": trees,
                "max_depth": depth,
                "min_samples": min_s,
                "mtry": eff_mtry,
                "mean_test_acc": statistics.fmean(accs),
                "std_test_acc": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                "min_test_acc": min(accs),
                "mean_train_acc": statistics.fmean(train_accs),
                "mean_train_s": statistics.fmean(times),
            }
        )
        print(
            f"  [{idx:3d}/{len(combos)}] trees={trees:<4} depth={depth:<3} "
            f"min_samples={min_s} mtry={eff_mtry:<3} -> acc={results[-1]['mean_test_acc']:.4f} "
            f"(+-{results[-1]['std_test_acc']:.4f})  train={results[-1]['mean_train_s']:.3f}s"
        )

    print(f"  concluido em {time.perf_counter() - t_start:.1f}s")
    return results


def pick_best(
    results: List[Dict[str, Any]], tolerance: float, prefer: str = "robust"
) -> Dict[str, Any]:
    """
    Seleciona entre as configuracoes cuja acuracia media fica a menos de
    `tolerance` da melhor observada (o "plato" de desempenho):

    - prefer="robust"   : maior numero de arvores (menor variancia do ensemble e
                          janela de medicao de energia mais confiavel) e, dentro
                          disso, o menor custo de treino.
    - prefer="cheapest" : menor custo de treino em termos absolutos.
    """
    best_acc = max(r["mean_test_acc"] for r in results)
    viable = [r for r in results if r["mean_test_acc"] >= best_acc - tolerance]
    if prefer == "cheapest":
        viable.sort(key=lambda r: (r["mean_train_s"], -r["mean_test_acc"]))
    else:
        viable.sort(key=lambda r: (-r["trees"], r["mean_train_s"], -r["mean_test_acc"]))
    return viable[0]


def main():
    ap = argparse.ArgumentParser(
        description="Grid search de hiperparametros do Random Forest via binario OpenMP."
    )
    ap.add_argument(
        "--dataset", action="append", default=None,
        help="Dataset a avaliar (repetivel). Padrao: todos exceto covtype.",
    )
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 7, 1, 2024, 123],
        help="Sementes de particionamento holdout usadas na media.",
    )
    ap.add_argument(
        "--tolerance", type=float, default=0.0025,
        help="Margem de acuracia (absoluta) para preferir a configuracao mais barata.",
    )
    ap.add_argument("--threads", type=int, default=None, help="Valor de OMP_NUM_THREADS.")
    ap.add_argument(
        "--prefer", choices=["robust", "cheapest"], default="robust",
        help="Criterio de desempate dentro do plato de acuracia.",
    )
    ap.add_argument("--top", type=int, default=8, help="Quantas linhas exibir no ranking.")
    ap.add_argument("--export-csv", type=str, default=None, help="Salva a varredura completa em CSV.")
    args = ap.parse_args()

    if not os.path.exists(OPENMP_EXE):
        sys.exit(f"binario OpenMP nao encontrado: {OPENMP_EXE} (rode 'make' em openmp/)")

    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    names = args.dataset or [k for k, v in GRIDS.items() if not v.get("default_off")]
    all_results: List[Dict[str, Any]] = []
    best_by_dataset: Dict[str, Dict[str, Any]] = {}

    for name in names:
        if name not in GRIDS:
            sys.exit(f"dataset desconhecido: {name} (opcoes: {', '.join(GRIDS)})")
        res = sweep(name, GRIDS[name], args.seeds)
        if not res:
            continue
        all_results.extend(res)
        best_by_dataset[name] = pick_best(res, args.tolerance, args.prefer)

        ranked = sorted(res, key=lambda r: -r["mean_test_acc"])[: args.top]
        print(f"\n  Top {len(ranked)} por acuracia media ({name}):")
        print("   trees depth min_s mtry | acc_media  desvio  acc_treino  treino(s)")
        for r in ranked:
            print(
                f"   {r['trees']:<5} {r['max_depth']:<5} {r['min_samples']:<5} {r['mtry']:<4} | "
                f"{r['mean_test_acc']:.4f}     {r['std_test_acc']:.4f}  "
                f"{r['mean_train_acc']:.4f}      {r['mean_train_s']:.3f}"
            )

    print("\n================ CONFIGURACOES ESCOLHIDAS ================")
    for name, b in best_by_dataset.items():
        print(
            f"{name:>14}: trees={b['trees']} max_depth={b['max_depth']} "
            f"min_samples={b['min_samples']} mtry={b['mtry']} "
            f"-> acc={b['mean_test_acc']:.4f} (+-{b['std_test_acc']:.4f}), "
            f"treino={b['mean_train_s']:.3f}s"
        )

    print("\nBloco pronto para BENCHMARKS em benchmark_comparison.py:")
    for name, b in best_by_dataset.items():
        cfg = GRIDS[name]
        target = f'"{cfg["target"]}"' if cfg["target"] else "None"
        print(
            f'    {{"name": "{name}", "target": {target}, "trees": {b["trees"]}, '
            f'"max_depth": {b["max_depth"]}, "min_samples": {b["min_samples"]}, '
            f'"mtry": {b["mtry"]}, "test_frac": {cfg["test_frac"]}, "seed": 42}},'
        )

    if args.export_csv and all_results:
        import csv

        with open(args.export_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
        print(f"\nVarredura completa exportada para: {args.export_csv}")


if __name__ == "__main__":
    main()
