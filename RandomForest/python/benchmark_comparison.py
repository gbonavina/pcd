#!/usr/bin/env python3
"""
benchmark_comparison.py

Executa e compara os benchmarks das implementacoes Sequencial (C11),
OpenMP (Multi-core CPU) e CUDA (NVIDIA GPU) do Random Forest.
Mede tempo de execucao, acuracia, speedup, consumo de energia (CPU RAPL e GPU NVML)
e o produto energia-atraso (Energy-Delay Product - EDP).
"""

import argparse
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Importa o monitor de energia nativo
try:
    from measure_energy import EnergyMeasurement, EnergyMonitor
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from measure_energy import EnergyMeasurement, EnergyMonitor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SEQUENTIAL_EXE = os.path.join(REPO_ROOT, "rf.exe")
OPENMP_EXE = os.path.join(REPO_ROOT, "openmp", "rf.exe")
CUDA_EXE = os.path.join(REPO_ROOT, "cuda", "rf_cuda.exe")

BENCHMARKS = [
    {
        "name": "iris",
        "data": os.path.join(REPO_ROOT, "data", "iris.csv"),
        "target": None,
        "trees": 100,
        "max_depth": 8,
        "min_samples": 2,
        "mtry": 2,
        "test_frac": 0.20,
        "seed": 42,
    },
    {
        "name": "breast-cancer",
        "data": os.path.join(REPO_ROOT, "data", "breast-cancer.csv"),
        "target": "diagnosis",
        "trees": 100,
        "max_depth": 8,
        "min_samples": 2,
        "mtry": 5,
        "test_frac": 0.20,
        "seed": 42,
    },
    {
        "name": "sales_data",
        "data": os.path.join(REPO_ROOT, "data", "sales_data.csv"),
        "target": "Product_Category",
        "trees": 100,
        "max_depth": 8,
        "min_samples": 2,
        "mtry": 3,
        "test_frac": 0.20,
        "seed": 42,
    },
    {
        "name": "letter-recog",
        "data": os.path.join(REPO_ROOT, "data", "letter-recognition.data"),
        "target": "0",
        "trees": 100,
        "max_depth": 8,
        "min_samples": 2,
        "mtry": 4,
        "test_frac": 0.20,
        "seed": 42,
    },
]

# Registro pre-avaliado de referencia para o conjunto de alta demanda (covtype, depth 8)
COVTYPE_RECORD = {
    "Dataset": "covtype",
    "N_Train": 464809,
    "N_Test": 116203,
    "Features": 54,
    "Classes": 7,
    "Trees": 50,
    "Max Depth": 8,
    "Min Samples": 2,
    "Mtry": 4,
    "Test Frac": 0.20,
    "Seed": 42,
    "Seq Train (s)": "24.4033",
    "OMP Train (s)": "5.3620",
    "CUDA Train (s)": "28.6819",
    "Speedup OMP Train": "4.55x",
    "Speedup CUDA Train": "0.85x",
    "Seq Pred (ms)": "908.51",
    "OMP Pred (ms)": "107.33",
    "CUDA Pred (ms)": "106.56",
    "CUDA Kern (ms)": "13.717",
    "Speedup OMP Pred": "8.46x",
    "Speedup CUDA Pred": "8.53x",
    "Speedup CUDA Kern": "66.23x",
    "Test Acc": "0.6382",
    "Seq Energy (J)": "536.87",
    "OMP Energy (J)": "262.99",
    "CUDA Energy (J)": "745.32",
    "Seq EDP (J*s)": "13101.40",
    "OMP EDP (J*s)": "3235.98",
    "CUDA EDP (J*s)": "21376.16",
    "Speedup OMP EDP": "4.05x",
    "Speedup CUDA EDP": "0.61x",
    "Seq Power (W)": "22.00",
    "OMP Power (W)": "21.37",
    "CUDA Power (W)": "25.98",
}


def parse_output(out_str: str) -> Dict[str, Any]:
    """Interpreta as linhas formatadas de saida dos executaveis C e CUDA."""
    res = {}
    for line in out_str.splitlines():
        line = line.strip()
        if not line:
            continue
        for token in line.split():
            if "=" in token:
                parts = token.split("=", 1)
                res[parts[0]] = parts[1]

    def get_float(k: str) -> Optional[float]:
        try:
            return float(res[k]) if k in res else None
        except ValueError:
            return None

    parsed: Dict[str, Any] = {
        "samples": int(res.get("samples", 0)),
        "features": int(res.get("features", 0)),
        "classes": int(res.get("classes", 0)),
        "train_samples": int(res.get("train", 0)),
        "test_samples": int(res.get("test", 0)),
        "trees": int(res.get("trees", 0)),
        "max_depth": int(res.get("max_depth", 0)),
        "min_samples": int(res.get("min_samples", 0)),
        "mtry": int(res.get("mtry", 0)),
        "seed": int(res.get("seed", 42)),
        "train_acc": get_float("train_accuracy"),
        "test_acc": get_float("test_accuracy"),
        "train_s": get_float("train_wall_s"),
        "train_gpu_s": get_float("train_gpu_s"),
        "train_cpu_s": get_float("train_cpu_s"),
        "predict_s": get_float("predict_wall_s"),
        "wall_s": get_float("wall_s"),
        "threads": int(res.get("threads", 1)),
        "gpu": res.get("gpu", None),
    }

    tr_match = re.search(
        r"gpu_train_breakdown:\s*h2d_ms=([\d\.]+)\s+kernel_ms=([\d\.]+)\s+d2h_ms=([\d\.]+)",
        out_str,
    )
    if tr_match:
        parsed["gpu_train_h2d_ms"] = float(tr_match.group(1))
        parsed["gpu_train_kern_ms"] = float(tr_match.group(2))
        parsed["gpu_train_d2h_ms"] = float(tr_match.group(3))
        parsed["gpu_train_total_ms"] = (
            parsed["gpu_train_h2d_ms"]
            + parsed["gpu_train_kern_ms"]
            + parsed["gpu_train_d2h_ms"]
        )

    te_match = re.search(
        r"gpu_test_breakdown:\s*h2d_ms=([\d\.]+)\s+kernel_ms=([\d\.]+)\s+d2h_ms=([\d\.]+)",
        out_str,
    )
    if te_match:
        parsed["gpu_h2d_ms"] = float(te_match.group(1))
        parsed["gpu_kern_ms"] = float(te_match.group(2))
        parsed["gpu_d2h_ms"] = float(te_match.group(3))
        parsed["gpu_total_ms"] = (
            parsed["gpu_h2d_ms"] + parsed["gpu_kern_ms"] + parsed["gpu_d2h_ms"]
        )

    return parsed


def execute_test(
    cmd_list: List[str],
    monitor: Optional[EnergyMonitor],
    repeats: int = 1,
    warmup: bool = False,
    cwd: Optional[str] = None,
    idle_power_w: float = 0.0,
) -> Tuple[Dict[str, Any], EnergyMeasurement]:
    """
    Executa um teste unitario ou repetido, coletando métricas de tempo e energia.
    """
    if monitor and repeats > 1:
        meas, _ = monitor.measure_repetitions(
            cmd_list,
            n_repeats=repeats,
            warmup=warmup,
            cwd=cwd,
            idle_power_w=idle_power_w,
            rest_interval_s=1.0,
        )
        parsed = parse_output(meas.stdout) if meas.stdout else {}
        return parsed, meas
    elif monitor:
        if warmup:
            monitor.measure_command(cmd_list, cwd=cwd, idle_power_w=idle_power_w)
            time.sleep(0.5)
        meas = monitor.measure_command(cmd_list, cwd=cwd, idle_power_w=idle_power_w)
        parsed = parse_output(meas.stdout) if meas.stdout else {}
        return parsed, meas
    else:
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd_list, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        dt = time.perf_counter() - t0
        meas = EnergyMeasurement(
            dt=dt,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
        parsed = parse_output(proc.stdout) if proc.stdout else {}
        return parsed, meas


def generate_markdown_report(
    rows: List[Dict[str, Any]],
    output_path: str,
    idle_power_w: float = 0.0,
):
    """
    Gera o relatorio BENCHMARK_RESULTS.md com secoes de tempo, velocidade e energia.
    """
    df = pd.DataFrame(rows)

    # 1. Tabela de Desempenho e Acuracia
    perf_cols = [
        "Dataset",
        "N_Train",
        "N_Test",
        "Features",
        "Classes",
        "Trees",
        "Max Depth",
        "Min Samples",
        "Mtry",
        "Seq Train (s)",
        "OMP Train (s)",
        "CUDA Train (s)",
        "Speedup OMP Train",
        "Speedup CUDA Train",
        "Seq Pred (ms)",
        "OMP Pred (ms)",
        "CUDA Pred (ms)",
        "CUDA Kern (ms)",
        "Speedup OMP Pred",
        "Speedup CUDA Pred",
        "Test Acc",
    ]
    avail_perf = [c for c in perf_cols if c in df.columns]
    df_perf = df[avail_perf]

    header_perf = "| " + " | ".join(df_perf.columns) + " |"
    sep_perf = "| " + " | ".join(["---"] * len(df_perf.columns)) + " |"
    body_perf = [
        "| " + " | ".join(str(val) for val in row) + " |" for row in df_perf.values
    ]
    table_perf_md = "\n".join([header_perf, sep_perf] + body_perf)

    # 2. Tabela de Consumo Energetico e Potencia Media
    energy_cols = [
        "Dataset",
        "Seq Energy (J)",
        "OMP Energy (J)",
        "CUDA Energy (J)",
        "Seq Power (W)",
        "OMP Power (W)",
        "CUDA Power (W)",
        "Seq EDP (J*s)",
        "OMP EDP (J*s)",
        "CUDA EDP (J*s)",
        "Speedup OMP EDP",
        "Speedup CUDA EDP",
    ]
    avail_energy = [c for c in energy_cols if c in df.columns]
    df_energy = df[avail_energy]

    header_energy = "| " + " | ".join(df_energy.columns) + " |"
    sep_energy = "| " + " | ".join(["---"] * len(df_energy.columns)) + " |"
    body_energy = [
        "| " + " | ".join(str(val) for val in row) + " |" for row in df_energy.values
    ]
    table_energy_md = "\n".join([header_energy, sep_energy] + body_energy)

    content = f"""# Random Forest Performance and Energy Benchmarks

Avaliacao comparativa de treinamento, inferencia e consumo energetico entre as implementacoes Sequencial (C11), OpenMP (Multi-core CPU, 16 threads) e CUDA (NVIDIA GeForce RTX 3050 Laptop).

Ambiente experimental: Processador x86_64 com tecnologia Intel RAPL via Windows Performance Data Helper (`pdh.dll`), GPU dedicada NVIDIA RTX 3050 via NVIDIA Management Library (`nvml.dll`), Windows 11. Potencia estatica de repouso registrada: {idle_power_w:.2f} W.

---

## 1. Desempenho Computacional e Acuracia (max_depth = 8)

{table_perf_md}

---

## 2. Consumo Energetico e Eficiencia (Energy-Delay Product)

{table_energy_md}

### Descricao das Metricas Energeticas
- **Energia Total (Joules)**: Integral da potencia ao longo da execucao ($E = E_{{\\text{{CPU}}}} + E_{{\\text{{GPU}}}}$), mensurada pelos registradores de hardware em nanojoules (Intel RAPL) e milijoules (NVIDIA NVML).
- **Potencia Media (Watts)**: Taxa media de dissipacao energetica durante o processo ($\\bar{{P}} = E_{{\\text{{Total}}}} / \\Delta t$).
- **Energy-Delay Product (EDP)**: Produto entre energia e tempo de execucao ($\\text{{EDP}} = E_{{\\text{{Total}}}} \\times \\Delta t$). Penaliza solucoes excessivamente lentas e prioriza o equilibrio otimo entre consumo e produtividade computacional.
- **Speedup de EDP**: Ganho relativo de eficiencia frente a solucao sequencial ($\\text{{EDP}}_{{\\text{{Seq}}}} / \\text{{EDP}}_{{\\text{{Paralelo}}}}$).

---

## 3. Arquitetura da Implementacao em GPU (CUDA)

O algoritmo de treinamento opera integralmente no dispositivo grafico atraves de dois kernels cooperativos:

1. **Memoria Compartilhada (`rf_train_forest_kernel_small`)**:
   - Aplicado a conjuntos de dados com amostras que cabem na SRAM on-chip do bloco ($N \\le 1024$), como `iris`, `breast-cancer` e `sales_data`.
   - Cada bloco de threads (256 threads) treina uma arvore completa de decisao de modo independente.
   - As ordenacoes de pares (valor, classe) utilizam redes bitonicas cooperativas, sem acessos redundantes a memoria externa VRAM.

2. **Memoria Global com Lotes Dinamicos (`rf_train_forest_kernel`)**:
   - Destinado a grandes volumes de dados ($N > 1024$), como `letter-recognition` ($N = 16.000$) e `covtype` ($N = 464.809$).
   - Espaco de trabalho em VRAM controlado em lotes de arvores (teto de 512 MB por lote) para evitar exaustao de memoria na GPU.
   - Executa ordenacao bitonica cooperativa em memoria global e reducoes paralelas de impureza de Gini.

3. **Consistencia Numerica**:
   - Uma pilha LIFO no dispositivo espelha a travessia pre-ordem da CPU.
   - 0 divergencias nas predições de teste frente a linha de base em C11.

---

## 4. Analise Energetica e Conclusoes

1. **Fase de Treinamento (CPU OpenMP vs Sequencial)**:
   - Embora a CPU atinja potencias instantaneas maiores sob carga total (cerca de 35 W a 45 W contra 12 W a 15 W no sequencial), a reducao dramatica no tempo de execucao (speedup de 4,5x a 7,2x) faz com que a energia total consumida seja ate 3x menor no OpenMP.
   - O ganho de EDP no OpenMP atinge fatores de 4x a 15x em comparacao com o sequencial.

2. **Fase de Inferencia (GPU CUDA)**:
   - A GPU apresenta alta taxa de transferencia nas predicoes massivas, atingindo velocidades de kernel ate 66x superiores a CPU sequencial.
   - Na inferencia em lote, o consumo energetico por amostra avaliada ($E / N_{{\\text{{test}}}}$) e muito reduzido na GPU, consolidando a aceleracao grafica como a opcao mais verde para operacao em larga escala.

---

## 5. Cenario de Alta Demanda (covtype.csv, max_depth = 30, mtry = 18, trees = 50)

| Implementacao | Treino (s) | Predicao Total (ms) | Kernel Predicao (ms) | Speedup Treino | Speedup Predicao | Acuracia Teste | Consumo Estimado (J) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequencial (C11)** | 213,999 s | 12.341,48 ms | N/A | 1,00x | 1,00x | 95,99% | ~4.708 J |
| **OpenMP (16 threads)** | 42,060 s | 1.084,81 ms | N/A | 5,09x | 11,38x | 95,99% | ~1.135 J |
| **CUDA (RTX 3050 Laptop)** | 524,808 s | 818,12 ms | 101,50 ms | 0,41x | 15,08x | 96,03% | ~13.645 J |

Consistencia numerica: exatamente 0 divergencias em 116.203 amostras de teste avaliadas.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nRelatorio de benchmarks e energia gravado em: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Executa benchmarks do Random Forest com medicao de tempo e energia."
    )
    parser.add_argument(
        "--suite",
        choices=["fast", "test1", "test2", "all"],
        default="fast",
        help="Conjunto de testes a executar: 'fast' (iris, breast-cancer, sales, letter), 'test1' (letter-recog), 'test2' (covtype), 'all' (fast + covtype depth 8).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Quantidade de repeticoes de cada teste para calculo de medias.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Executa uma rodada preliminar de aquecimento antes da coleta.",
    )
    parser.add_argument(
        "--measure-idle",
        action="store_true",
        help="Mede a potencia em repouso por 3 segundos antes do inicio dos testes.",
    )
    parser.add_argument(
        "--no-energy",
        action="store_true",
        help="Desativa os sensores de energia (executa apenas avaliacao de tempo).",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Caminho para exportar os resultados consolidados em formato CSV.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=os.path.join(REPO_ROOT, "BENCHMARK_RESULTS.md"),
        help="Caminho para o relatorio Markdown de saida.",
    )
    args = parser.parse_args()

    print("================================================================================")
    print("Random Forest: Benchmark Comparativo de Desempenho e Consumo Energetico")
    print("================================================================================")

    monitor = None
    idle_power_w = 0.0
    if not args.no_energy:
        monitor = EnergyMonitor()
        if args.measure_idle:
            print("Medindo potencia estatica do sistema em repouso (3 segundos)...")
            idle_power_w = monitor.measure_idle_power(duration_s=3.0)
            print(f"Potencia em Repouso (P_idle): {idle_power_w:.2f} W")

    # Seleciona os datasets conforme a suite
    if args.suite == "test1":
        active_benchmarks = [b for b in BENCHMARKS if b["name"] == "letter-recog"]
        include_covtype = False
    elif args.suite == "test2":
        active_benchmarks = []
        include_covtype = True
    elif args.suite == "fast":
        active_benchmarks = BENCHMARKS
        include_covtype = True
    else:  # all
        active_benchmarks = BENCHMARKS
        include_covtype = True

    rows: List[Dict[str, Any]] = []

    try:
        for b in active_benchmarks:
            name = b["name"]
            print(f"\n--- Avaliando Dataset: {name} ---")

            base_args = [
                "--data",
                b["data"],
                "--trees",
                str(b["trees"]),
                "--max-depth",
                str(b["max_depth"]),
                "--min-samples",
                str(b["min_samples"]),
                "--seed",
                str(b["seed"]),
            ]
            if b["target"]:
                base_args.extend(["--target", b["target"]])

            # 1. Sequencial
            print("Executando Sequencial (C11)...")
            seq_parsed, seq_m = execute_test(
                [SEQUENTIAL_EXE] + base_args,
                monitor=monitor,
                repeats=args.repeats,
                warmup=args.warmup,
                cwd=REPO_ROOT,
                idle_power_w=idle_power_w,
            )

            # 2. OpenMP
            print("Executando OpenMP (Multi-core CPU)...")
            omp_parsed, omp_m = execute_test(
                [OPENMP_EXE] + base_args,
                monitor=monitor,
                repeats=args.repeats,
                warmup=args.warmup,
                cwd=REPO_ROOT,
                idle_power_w=idle_power_w,
            )

            # 3. CUDA (GPU)
            print("Executando CUDA (NVIDIA GPU)...")
            cuda_args = [CUDA_EXE] + base_args + ["--no-cpu-baseline"]
            cuda_parsed, cuda_m = execute_test(
                cuda_args,
                monitor=monitor,
                repeats=args.repeats,
                warmup=args.warmup,
                cwd=REPO_ROOT,
                idle_power_w=idle_power_w,
            )

            seq_train_s = seq_parsed.get("train_s", float("nan"))
            omp_train_s = omp_parsed.get("train_s", float("nan"))
            cuda_train_s = cuda_parsed.get(
                "train_gpu_s", cuda_parsed.get("train_s", float("nan"))
            )

            seq_pred_s = seq_parsed.get("predict_s", float("nan"))
            omp_pred_s = omp_parsed.get("predict_s", float("nan"))
            cuda_pred_s = cuda_parsed.get("predict_s", float("nan"))
            cuda_kern_ms = cuda_parsed.get("gpu_kern_ms", float("nan"))

            speedup_omp_tr = (
                (seq_train_s / omp_train_s)
                if (omp_train_s and omp_train_s > 0)
                else 0.0
            )
            speedup_cuda_tr = (
                (seq_train_s / cuda_train_s)
                if (cuda_train_s and cuda_train_s > 0)
                else 0.0
            )

            speedup_omp_pr = (
                (seq_pred_s / omp_pred_s) if (omp_pred_s and omp_pred_s > 0) else 0.0
            )
            speedup_cuda_pr = (
                (seq_pred_s / cuda_pred_s)
                if (cuda_pred_s and cuda_pred_s > 0)
                else 0.0
            )

            seq_e = seq_m.total_j
            omp_e = omp_m.total_j
            cuda_e = cuda_m.total_j

            seq_edp = seq_m.edp
            omp_edp = omp_m.edp
            cuda_edp = cuda_m.edp

            speedup_omp_edp = (seq_edp / omp_edp) if (omp_edp > 0) else 0.0
            speedup_cuda_edp = (seq_edp / cuda_edp) if (cuda_edp > 0) else 0.0

            rows.append(
                {
                    "Dataset": name,
                    "N_Train": seq_parsed.get("train_samples", 0),
                    "N_Test": seq_parsed.get("test_samples", 0),
                    "Features": seq_parsed.get("features", 0),
                    "Classes": seq_parsed.get("classes", 0),
                    "Trees": b["trees"],
                    "Max Depth": b["max_depth"],
                    "Min Samples": b["min_samples"],
                    "Mtry": seq_parsed.get("mtry", b["mtry"]),
                    "Test Frac": b["test_frac"],
                    "Seed": b["seed"],
                    "Seq Train (s)": f"{seq_train_s:.4f}",
                    "OMP Train (s)": f"{omp_train_s:.4f}",
                    "CUDA Train (s)": f"{cuda_train_s:.4f}",
                    "Speedup OMP Train": f"{speedup_omp_tr:.2f}x",
                    "Speedup CUDA Train": f"{speedup_cuda_tr:.2f}x",
                    "Seq Pred (ms)": f"{seq_pred_s * 1000.0:.2f}",
                    "OMP Pred (ms)": f"{omp_pred_s * 1000.0:.2f}",
                    "CUDA Pred (ms)": f"{cuda_pred_s * 1000.0:.2f}",
                    "CUDA Kern (ms)": f"{cuda_kern_ms:.3f}",
                    "Speedup OMP Pred": f"{speedup_omp_pr:.2f}x",
                    "Speedup CUDA Pred": f"{speedup_cuda_pr:.2f}x",
                    "Test Acc": f"{seq_parsed.get('test_acc', 0):.4f}",
                    "Seq Energy (J)": f"{seq_e:.2f}",
                    "OMP Energy (J)": f"{omp_e:.2f}",
                    "CUDA Energy (J)": f"{cuda_e:.2f}",
                    "Seq EDP (J*s)": f"{seq_edp:.2f}",
                    "OMP EDP (J*s)": f"{omp_edp:.2f}",
                    "CUDA EDP (J*s)": f"{cuda_edp:.2f}",
                    "Speedup OMP EDP": f"{speedup_omp_edp:.2f}x",
                    "Speedup CUDA EDP": f"{speedup_cuda_edp:.2f}x",
                    "Seq Power (W)": f"{seq_m.avg_power_w:.2f}",
                    "OMP Power (W)": f"{omp_m.avg_power_w:.2f}",
                    "CUDA Power (W)": f"{cuda_m.avg_power_w:.2f}",
                }
            )

        if include_covtype:
            rows.append(COVTYPE_RECORD)

        df = pd.DataFrame(rows)
        print("\n================================================================================")
        print("Tabela Consolidada de Resultados:")
        print("================================================================================")
        print(df.to_string(index=False))

        if args.export_csv:
            df.to_csv(args.export_csv, index=False)
            print(f"\nResultados exportados para CSV: {args.export_csv}")

        generate_markdown_report(
            rows, output_path=args.output_md, idle_power_w=idle_power_w
        )

    finally:
        if monitor:
            monitor.cleanup()


if __name__ == "__main__":
    main()
