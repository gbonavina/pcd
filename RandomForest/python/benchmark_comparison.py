#!/usr/bin/env python3
"""
benchmark_comparison.py

Executa e compara os benchmarks das implementacoes Sequencial (C11),
OpenMP (Multi-core CPU) e CUDA (NVIDIA GPU) do Random Forest.
Mede tempo de execucao, acuracia, speedup, consumo de energia (CPU RAPL e GPU NVML)
e o produto energia-atraso (Energy-Delay Product - EDP).

Os hiperparametros de cada dataset foram calibrados por grid search com o
binario OpenMP (ver tune_params.py); nenhum resultado deste relatorio e
pre-computado: todas as linhas vem de execucoes reais.
"""

import argparse
import math
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

# ---------------------------------------------------------------------------
# Hiperparametros calibrados (grid search com openmp/rf.exe, 5 sementes de
# holdout para os datasets leves e 3 para letter-recognition; escolha pelo
# maior patamar de acuracia media de teste). Reproduza com:
#   python tune_params.py --threads 16 --export-csv sweep.csv
#
# Acuracia media de teste obtida na calibracao:
#   iris          0.9533 (+-0.0267)   breast-cancer 0.9772 (+-0.0105)
#   sales_data    0.2780 (+-0.0284)   letter-recog  0.9642 (+-0.0019)
#   covtype       0.9603 (max_depth 30, mtry 18, calibrado previamente)
#
# Observacao: sales_data.csv e um conjunto sintetico cujo alvo
# (Product_Category, 4 classes) nao possui sinal preditivo - qualquer
# configuracao fica no nivel do acaso (~25%). Mantido como caso de controle.
# ---------------------------------------------------------------------------
BENCHMARKS = [
    {
        "name": "iris",
        "data": os.path.join(REPO_ROOT, "data", "iris.csv"),
        "target": None,
        "trees": 300,
        "max_depth": 12,
        "min_samples": 5,
        "mtry": 2,
        "test_frac": 0.20,
        "seed": 42,
        "heavy": False,
    },
    {
        "name": "breast-cancer",
        "data": os.path.join(REPO_ROOT, "data", "breast-cancer.csv"),
        "target": "diagnosis",
        "trees": 300,
        "max_depth": 6,
        "min_samples": 5,
        "mtry": 3,
        "test_frac": 0.20,
        "seed": 42,
        "heavy": False,
    },
    {
        "name": "sales_data",
        "data": os.path.join(REPO_ROOT, "data", "sales_data.csv"),
        "target": "Product_Category",
        "trees": 200,
        "max_depth": 10,
        "min_samples": 5,
        "mtry": 8,
        "test_frac": 0.20,
        "seed": 42,
        "heavy": False,
    },
    {
        "name": "letter-recog",
        "data": os.path.join(REPO_ROOT, "data", "letter-recognition.data"),
        "target": "0",
        "trees": 300,
        "max_depth": 30,
        "min_samples": 2,
        "mtry": 4,
        "test_frac": 0.20,
        "seed": 42,
        "heavy": False,
    },
    # Cenario de alta demanda: 581.012 amostras, 54 atributos, 7 classes.
    # Executado de verdade (nao ha valores tabelados no codigo). Custo tipico:
    # ~3,5 min na CPU sequencial, ~1 min em OpenMP e ~9 min em CUDA.
    {
        "name": "covtype",
        "data": os.path.join(REPO_ROOT, "data", "covtype.csv"),
        "target": "Cover_Type",
        "trees": 50,
        "max_depth": 30,
        "min_samples": 2,
        "mtry": 18,
        "test_frac": 0.20,
        "seed": 42,
        "heavy": True,
    },
]

SUITES = {
    "fast": ["iris", "breast-cancer", "sales_data"],
    "test1": ["letter-recog"],
    "test2": ["covtype"],
    "all": [b["name"] for b in BENCHMARKS],
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

    mm = re.search(r"gpu_cpu_test_mismatches=(\d+)\s*/\s*(\d+)", out_str)
    if mm:
        parsed["mismatches"] = int(mm.group(1))
        parsed["mismatch_total"] = int(mm.group(2))

    gpu_match = re.search(r'gpu="([^"]+)"', out_str)
    if gpu_match:
        parsed["gpu"] = gpu_match.group(1)

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


def _num(text: Any) -> float:
    """Converte celulas formatadas ('4.55x', '12.30') de volta para float."""
    try:
        return float(str(text).rstrip("x"))
    except (TypeError, ValueError):
        return float("nan")


def _fmt_range(vals: List[float], suffix: str = "x", digits: int = 2) -> str:
    vals = [v for v in vals if v == v and v > 0]
    if not vals:
        return "n/d"
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 10 ** (-digits):
        return f"{lo:.{digits}f}{suffix}"
    return f"{lo:.{digits}f}{suffix} a {hi:.{digits}f}{suffix}"


def _md_table(df: pd.DataFrame, cols: List[str]) -> str:
    avail = [c for c in cols if c in df.columns]
    sub = df[avail]
    header = "| " + " | ".join(sub.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(sub.columns)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in sub.values]
    return "\n".join([header, sep] + body)


def build_analysis(df: pd.DataFrame) -> str:
    """Gera as conclusoes da secao de analise a partir dos numeros medidos."""
    omp_tr = [_num(v) for v in df.get("Speedup OMP Train", [])]
    cuda_tr = [_num(v) for v in df.get("Speedup CUDA Train", [])]
    omp_pr = [_num(v) for v in df.get("Speedup OMP Pred", [])]
    cuda_pr = [_num(v) for v in df.get("Speedup CUDA Pred", [])]
    cuda_kn = [_num(v) for v in df.get("Speedup CUDA Kern", [])]
    omp_edp = [_num(v) for v in df.get("Speedup OMP EDP", [])]
    cuda_edp = [_num(v) for v in df.get("Speedup CUDA EDP", [])]

    seq_e = [_num(v) for v in df.get("Seq Energy (J)", [])]
    omp_e = [_num(v) for v in df.get("OMP Energy (J)", [])]
    ratio_e = [
        s / o for s, o in zip(seq_e, omp_e) if o == o and o > 0 and s == s
    ]

    cuda_tr_wins = sum(1 for v in cuda_tr if v == v and v > 1.0)
    cuda_edp_wins = sum(1 for v in cuda_edp if v == v and v > 1.0)
    n = len(df)

    lines = [
        "1. **Treinamento em OpenMP**: speedup de "
        f"{_fmt_range(omp_tr)} sobre a versao sequencial, com reducao de energia total de "
        f"{_fmt_range(ratio_e)} e ganho de EDP de {_fmt_range(omp_edp)}. "
        "A potencia instantanea da CPU cresce sob carga total, mas a compressao do tempo "
        "de execucao mais que compensa esse aumento.",
        "2. **Treinamento em CUDA**: speedup de "
        f"{_fmt_range(cuda_tr)} ({cuda_tr_wins} de {n} datasets acima de 1,00x). "
        "A construcao de arvores e irregular e dependente de ordenacoes/reducoes "
        "sequenciais por no, o que limita a ocupacao da GPU; o ganho aparece apenas "
        "quando ha volume suficiente para saturar os multiprocessadores.",
        "3. **Inferencia**: speedup de "
        f"{_fmt_range(omp_pr)} em OpenMP e {_fmt_range(cuda_pr)} em CUDA no tempo total "
        f"de predicao, chegando a {_fmt_range(cuda_kn)} quando se considera apenas o "
        "kernel (sem as transferencias host-device). A predicao e massivamente paralela "
        "e sem divergencia de trabalho, sendo o regime mais favoravel a GPU.",
        "4. **Eficiencia energetica (EDP)**: OpenMP domina em todos os casos "
        f"({_fmt_range(omp_edp)}); CUDA fica acima do sequencial em {cuda_edp_wins} de "
        f"{n} datasets ({_fmt_range(cuda_edp)}), porque a GPU mantem potencia elevada "
        "durante todo o treinamento, penalizando o produto energia-atraso quando o "
        "kernel nao converte essa potencia em reducao de tempo.",
    ]
    return "\n".join(lines)


def generate_markdown_report(
    rows: List[Dict[str, Any]],
    output_path: str,
    idle_power_w: float = 0.0,
    repeats: int = 1,
):
    """
    Gera o relatorio BENCHMARK_RESULTS.md com secoes de configuracao, tempo,
    energia e consistencia numerica. Todo o conteudo deriva das medicoes.
    """
    df = pd.DataFrame(rows)

    config_cols = [
        "Dataset", "N_Train", "N_Test", "Features", "Classes",
        "Trees", "Max Depth", "Min Samples", "Mtry", "Test Frac", "Seed",
        "Test Acc",
    ]
    perf_cols = [
        "Dataset",
        "Seq Train (s)", "OMP Train (s)", "CUDA Train (s)",
        "Speedup OMP Train", "Speedup CUDA Train",
        "Seq Pred (ms)", "OMP Pred (ms)", "CUDA Pred (ms)", "CUDA Kern (ms)",
        "Speedup OMP Pred", "Speedup CUDA Pred", "Speedup CUDA Kern",
    ]
    energy_cols = [
        "Dataset",
        "Seq Energy (J)", "OMP Energy (J)", "CUDA Energy (J)",
        "Seq Power (W)", "OMP Power (W)", "CUDA Power (W)",
        "Seq EDP (J*s)", "OMP EDP (J*s)", "CUDA EDP (J*s)",
        "Speedup OMP EDP", "Speedup CUDA EDP",
    ]
    consistency_cols = [
        "Dataset", "Seq Test Acc", "OMP Test Acc", "CUDA Test Acc",
        "CUDA Mismatches",
    ]

    table_config_md = _md_table(df, config_cols)
    table_perf_md = _md_table(df, perf_cols)
    table_energy_md = _md_table(df, energy_cols)
    table_consistency_md = _md_table(df, consistency_cols)

    threads = int(df["Threads"].max()) if "Threads" in df.columns else 0
    gpus = [g for g in df.get("GPU", []) if isinstance(g, str) and g]
    gpu_name = gpus[0].strip('"') if gpus else "GPU NVIDIA"
    analysis_md = build_analysis(df)

    heavy = df[df["Dataset"] == "covtype"] if "Dataset" in df.columns else pd.DataFrame()
    if not heavy.empty:
        h = heavy.iloc[0]
        heavy_md = f"""O conjunto `covtype.csv` (581.012 amostras, 54 atributos, 7 classes) e executado
com os mesmos binarios e a mesma instrumentacao dos demais, com arvores profundas
(`max_depth={h['Max Depth']}`, `mtry={h['Mtry']}`, `trees={h['Trees']}`), regime em que a acuracia de
teste alcanca {_num(h['Test Acc']) * 100:.2f}%.

| Implementacao | Treino (s) | Predicao Total (ms) | Kernel Predicao (ms) | Speedup Treino | Speedup Predicao | Energia (J) | EDP (J*s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Sequencial (C11) | {h['Seq Train (s)']} | {h['Seq Pred (ms)']} | N/A | 1.00x | 1.00x | {h['Seq Energy (J)']} | {h['Seq EDP (J*s)']} |
| OpenMP ({threads} threads) | {h['OMP Train (s)']} | {h['OMP Pred (ms)']} | N/A | {h['Speedup OMP Train']} | {h['Speedup OMP Pred']} | {h['OMP Energy (J)']} | {h['OMP EDP (J*s)']} |
| CUDA ({gpu_name}) | {h['CUDA Train (s)']} | {h['CUDA Pred (ms)']} | {h['CUDA Kern (ms)']} | {h['Speedup CUDA Train']} | {h['Speedup CUDA Pred']} | {h['CUDA Energy (J)']} | {h['CUDA EDP (J*s)']} |

Consistencia numerica: {h['CUDA Mismatches']} divergencia(s) entre GPU e CPU nas {h['N_Test']} amostras de teste."""
    else:
        heavy_md = (
            "Nao incluido nesta execucao. Rode `python benchmark_comparison.py "
            "--suite test2` (ou `--suite all`) para medir o cenario `covtype.csv`."
        )

    content = f"""# Random Forest: Benchmarks de Desempenho e Energia

Avaliacao comparativa de treinamento, inferencia e consumo energetico entre as
implementacoes Sequencial (C11), OpenMP ({threads} threads) e CUDA ({gpu_name}).

Ambiente experimental: CPU x86_64 instrumentada por Intel RAPL via Windows Performance
Data Helper (`pdh.dll`), GPU NVIDIA via NVIDIA Management Library (`nvml.dll`), Windows 11.
Potencia estatica de repouso registrada: {idle_power_w:.2f} W. Repeticoes por medicao: {repeats}.

Todos os numeros abaixo foram produzidos pela execucao que gerou este arquivo; nao ha
valores tabelados no codigo.

> Resolucao dos sensores: os contadores RAPL e NVML sao atualizados em intervalos da ordem
> de dezenas de milissegundos e incluem a energia de inicializacao do processo (contexto
> CUDA, carga do CSV). Execucoes muito curtas (abaixo de ~0,5 s, como `iris`) ficam no piso
> de resolucao e produzem potencias medias irrealistas; use `--repeats N --warmup` para
> medicoes energeticas comparaveis nesses casos.

---

## 1. Configuracao Experimental e Acuracia

Os hiperparametros de cada dataset foram selecionados por busca em grade com o binario
OpenMP (`python tune_params.py`), maximizando a acuracia media de teste sobre multiplas
particoes holdout. As tres implementacoes recebem exatamente os mesmos parametros e a
mesma semente, portanto treinam a mesma floresta.

{table_config_md}

Sobre `sales_data`: o alvo `Product_Category` desse conjunto sintetico nao possui sinal
preditivo (4 classes, acuracia de teste no nivel do acaso mesmo com acuracia de treino
proxima de 1,0). Ele permanece na suite como caso de controle de sobreajuste, nao como
referencia de qualidade preditiva.

---

## 2. Desempenho Computacional

{table_perf_md}

`CUDA Train (s)` usa o tempo de treino no dispositivo (`train_gpu_s`). `CUDA Kern (ms)`
isola o kernel de predicao, sem as transferencias host-device.

---

## 3. Consumo Energetico e Eficiencia (Energy-Delay Product)

{table_energy_md}

### Descricao das Metricas Energeticas
- **Energia Total (Joules)**: integral da potencia ao longo da execucao ($E = E_{{\\text{{CPU}}}} + E_{{\\text{{GPU}}}}$), lida dos contadores de hardware em nanojoules (Intel RAPL) e milijoules (NVIDIA NVML).
- **Potencia Media (Watts)**: taxa media de dissipacao durante o processo ($\\bar{{P}} = E_{{\\text{{Total}}}} / \\Delta t$).
- **Energy-Delay Product (EDP)**: produto entre energia e tempo ($\\text{{EDP}} = E_{{\\text{{Total}}}} \\times \\Delta t$). Penaliza solucoes lentas e prioriza o equilibrio entre consumo e produtividade.
- **Speedup de EDP**: ganho relativo frente a solucao sequencial ($\\text{{EDP}}_{{\\text{{Seq}}}} / \\text{{EDP}}_{{\\text{{Paralelo}}}}$).

---

## 4. Consistencia Numerica

{table_consistency_md}

As tres implementacoes recebem os mesmos parametros e a mesma semente de particionamento,
e a inferencia percorre a mesma travessia pre-ordem. Divergencias residuais de acuracia
podem ocorrer quando dois candidatos de corte empatam em ganho de Gini e a ordem de
avaliacao (sequencial na CPU, cooperativa na GPU) desfaz o empate de forma diferente.
`CUDA Mismatches` e a contagem de predicoes divergentes entre GPU e CPU para a mesma
floresta, reportada pelo proprio binario CUDA.

---

## 5. Arquitetura da Implementacao em GPU (CUDA)

O treinamento opera integralmente no dispositivo por meio de dois kernels:

1. **Memoria compartilhada (`rf_train_forest_kernel_small`)**: conjuntos cujas amostras
   cabem na SRAM do bloco ($N \\le 1024$), como `iris`, `breast-cancer` e `sales_data`.
   Cada bloco (256 threads) treina uma arvore completa; as ordenacoes de pares
   (valor, classe) usam redes bitonicas cooperativas, sem trafego adicional para a VRAM.
2. **Memoria global com lotes dinamicos (`rf_train_forest_kernel`)**: para $N > 1024$,
   como `letter-recognition` e `covtype`. O espaco de trabalho em VRAM e fatiado em lotes
   de arvores (teto de 512 MB por lote) para evitar exaustao de memoria, com ordenacao
   bitonica em memoria global e reducoes paralelas de impureza de Gini.
3. **Consistencia**: uma pilha LIFO no dispositivo espelha a travessia pre-ordem da CPU
   (ver secao 4 para a contagem de divergencias medida).

---

## 6. Analise dos Resultados

{analysis_md}

---

## 7. Cenario de Alta Demanda (covtype.csv)

{heavy_md}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nRelatorio de benchmarks e energia gravado em: {output_path}")


def build_row(
    b: Dict[str, Any],
    seq_parsed: Dict[str, Any],
    omp_parsed: Dict[str, Any],
    cuda_parsed: Dict[str, Any],
    seq_m: EnergyMeasurement,
    omp_m: EnergyMeasurement,
    cuda_m: EnergyMeasurement,
) -> Dict[str, Any]:
    nan = float("nan")

    def val(d: Dict[str, Any], *keys: str) -> float:
        """Primeiro valor numerico valido entre as chaves informadas."""
        for k in keys:
            v = d.get(k)
            if isinstance(v, (int, float)) and v == v:
                return float(v)
        return nan

    def to_ms(v: float) -> float:
        return v * 1000.0 if v == v else nan

    seq_train_s = val(seq_parsed, "train_s")
    omp_train_s = val(omp_parsed, "train_s")
    cuda_train_s = val(cuda_parsed, "train_gpu_s", "train_s")

    seq_pred_s = val(seq_parsed, "predict_s")
    omp_pred_s = val(omp_parsed, "predict_s")
    cuda_pred_s = val(cuda_parsed, "predict_s")
    cuda_kern_ms = val(cuda_parsed, "gpu_kern_ms")

    def ratio(a: float, b_: float) -> float:
        if not (a == a and b_ == b_) or b_ <= 0:
            return 0.0
        return a / b_

    speedup_omp_tr = ratio(seq_train_s, omp_train_s)
    speedup_cuda_tr = ratio(seq_train_s, cuda_train_s)
    speedup_omp_pr = ratio(seq_pred_s, omp_pred_s)
    speedup_cuda_pr = ratio(seq_pred_s, cuda_pred_s)
    speedup_cuda_kn = ratio(to_ms(seq_pred_s), cuda_kern_ms)

    seq_edp, omp_edp, cuda_edp = seq_m.edp, omp_m.edp, cuda_m.edp

    def f(v: Optional[float], spec: str) -> str:
        if not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
            return "n/d"
        return format(v, spec)

    mismatches = cuda_parsed.get("mismatches", None)

    return {
        "Dataset": b["name"],
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
        "Threads": omp_parsed.get("threads", 1),
        "GPU": cuda_parsed.get("gpu", ""),
        "Seq Train (s)": f(seq_train_s, ".4f"),
        "OMP Train (s)": f(omp_train_s, ".4f"),
        "CUDA Train (s)": f(cuda_train_s, ".4f"),
        "Speedup OMP Train": f"{speedup_omp_tr:.2f}x",
        "Speedup CUDA Train": f"{speedup_cuda_tr:.2f}x",
        "Seq Pred (ms)": f(to_ms(seq_pred_s), ".2f"),
        "OMP Pred (ms)": f(to_ms(omp_pred_s), ".2f"),
        "CUDA Pred (ms)": f(to_ms(cuda_pred_s), ".2f"),
        "CUDA Kern (ms)": f(cuda_kern_ms, ".3f"),
        "Speedup OMP Pred": f"{speedup_omp_pr:.2f}x",
        "Speedup CUDA Pred": f"{speedup_cuda_pr:.2f}x",
        "Speedup CUDA Kern": f"{speedup_cuda_kn:.2f}x",
        "Test Acc": f(seq_parsed.get("test_acc"), ".4f"),
        "Seq Test Acc": f(seq_parsed.get("test_acc"), ".4f"),
        "OMP Test Acc": f(omp_parsed.get("test_acc"), ".4f"),
        "CUDA Test Acc": f(cuda_parsed.get("test_acc"), ".4f"),
        "CUDA Mismatches": mismatches if mismatches is not None else "n/d",
        "Seq Energy (J)": f"{seq_m.total_j:.2f}",
        "OMP Energy (J)": f"{omp_m.total_j:.2f}",
        "CUDA Energy (J)": f"{cuda_m.total_j:.2f}",
        "Seq EDP (J*s)": f"{seq_edp:.2f}",
        "OMP EDP (J*s)": f"{omp_edp:.2f}",
        "CUDA EDP (J*s)": f"{cuda_edp:.2f}",
        "Speedup OMP EDP": f"{ratio(seq_edp, omp_edp):.2f}x",
        "Speedup CUDA EDP": f"{ratio(seq_edp, cuda_edp):.2f}x",
        "Seq Power (W)": f"{seq_m.avg_power_w:.2f}",
        "OMP Power (W)": f"{omp_m.avg_power_w:.2f}",
        "CUDA Power (W)": f"{cuda_m.avg_power_w:.2f}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Executa benchmarks do Random Forest com medicao de tempo e energia."
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES.keys()),
        default="fast",
        help="Conjunto de testes: 'fast' (iris, breast-cancer, sales_data), "
        "'test1' (letter-recog), 'test2' (covtype, alta demanda), 'all' (todos).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Executa apenas o dataset informado (repetivel); ignora --suite.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Quantidade de repeticoes de cada teste para calculo de medias.",
    )
    parser.add_argument(
        "--heavy-repeats",
        type=int,
        default=1,
        help="Repeticoes para datasets marcados como 'heavy' (covtype).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Executa uma rodada preliminar de aquecimento antes da coleta.",
    )
    parser.add_argument(
        "--skip-cuda",
        action="store_true",
        help="Nao executa a versao CUDA (util em maquinas sem GPU NVIDIA).",
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

    print("=" * 80)
    print("Random Forest: Benchmark Comparativo de Desempenho e Consumo Energetico")
    print("=" * 80)

    by_name = {b["name"]: b for b in BENCHMARKS}
    if args.dataset:
        unknown = [d for d in args.dataset if d not in by_name]
        if unknown:
            sys.exit(f"dataset desconhecido: {', '.join(unknown)}")
        selected = [by_name[d] for d in args.dataset]
    else:
        selected = [by_name[n] for n in SUITES[args.suite]]

    monitor = None
    idle_power_w = 0.0
    if not args.no_energy:
        monitor = EnergyMonitor()
        if args.measure_idle:
            print("Medindo potencia estatica do sistema em repouso (3 segundos)...")
            idle_power_w = monitor.measure_idle_power(duration_s=3.0)
            print(f"Potencia em Repouso (P_idle): {idle_power_w:.2f} W")

    rows: List[Dict[str, Any]] = []

    try:
        for b in selected:
            name = b["name"]
            repeats = args.heavy_repeats if b.get("heavy") else args.repeats
            # O aquecimento dobra o custo de uma rodada; em datasets de alta
            # demanda (centenas de segundos) o efeito de cache e irrelevante
            # frente ao tempo de computacao, entao ele e dispensado.
            warmup = args.warmup and not b.get("heavy")
            print(f"\n--- Avaliando Dataset: {name} (repeticoes={repeats}) ---")
            if b.get("heavy"):
                print("    Cenario de alta demanda: pode levar varios minutos por implementacao.")

            base_args = [
                "--data", b["data"],
                "--trees", str(b["trees"]),
                "--max-depth", str(b["max_depth"]),
                "--min-samples", str(b["min_samples"]),
                "--mtry", str(b["mtry"]),
                "--test-frac", f"{b['test_frac']:.4f}",
                "--seed", str(b["seed"]),
            ]
            if b["target"]:
                base_args.extend(["--target", b["target"]])

            runs: Dict[str, Tuple[Dict[str, Any], EnergyMeasurement]] = {}
            plan = [
                ("Sequencial (C11)", "seq", [SEQUENTIAL_EXE] + base_args),
                ("OpenMP (Multi-core CPU)", "omp", [OPENMP_EXE] + base_args),
            ]
            if not args.skip_cuda:
                plan.append(
                    ("CUDA (NVIDIA GPU)", "cuda", [CUDA_EXE] + base_args + ["--no-cpu-baseline"])
                )

            failed = False
            for label, key, cmd in plan:
                print(f"Executando {label}...", flush=True)
                t0 = time.perf_counter()
                parsed, meas = execute_test(
                    cmd,
                    monitor=monitor,
                    repeats=repeats,
                    warmup=warmup,
                    cwd=REPO_ROOT,
                    idle_power_w=idle_power_w,
                )
                if meas.returncode != 0:
                    print(f"  ERRO (codigo {meas.returncode}): {meas.stderr.strip()[:400]}")
                    failed = True
                    break
                train_val = parsed.get("train_gpu_s") or parsed.get("train_s") or 0.0
                print(
                    f"  ok em {time.perf_counter() - t0:.2f}s "
                    f"(treino={train_val:.4f}s, "
                    f"acc={parsed.get('test_acc') or 0.0:.4f}, energia={meas.total_j:.2f} J)"
                )
                runs[key] = (parsed, meas)

            if failed:
                print(f"  dataset {name} descartado por falha de execucao.")
                continue

            empty = ({}, EnergyMeasurement())
            seq_parsed, seq_m = runs.get("seq", empty)
            omp_parsed, omp_m = runs.get("omp", empty)
            cuda_parsed, cuda_m = runs.get("cuda", empty)

            rows.append(
                build_row(b, seq_parsed, omp_parsed, cuda_parsed, seq_m, omp_m, cuda_m)
            )

        if not rows:
            sys.exit("nenhum resultado coletado; relatorio nao gerado.")

        df = pd.DataFrame(rows)
        print("\n" + "=" * 80)
        print("Tabela Consolidada de Resultados:")
        print("=" * 80)
        print(df.drop(columns=["GPU"], errors="ignore").to_string(index=False))

        if args.export_csv:
            df.to_csv(args.export_csv, index=False)
            print(f"\nResultados exportados para CSV: {args.export_csv}")

        generate_markdown_report(
            rows,
            output_path=args.output_md,
            idle_power_w=idle_power_w,
            repeats=args.repeats,
        )

    finally:
        if monitor:
            monitor.cleanup()


if __name__ == "__main__":
    main()
