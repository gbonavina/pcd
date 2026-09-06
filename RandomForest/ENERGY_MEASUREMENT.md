# Metodologia de Medição de Consumo Energético para Random Forest

Este documento define a metodologia, os modelos matemáticos e as ferramentas práticas para mensurar o consumo de energia e a eficiência energética das três implementações do projeto Random Forest: Sequencial (C11), OpenMP (Multi-core CPU) e CUDA (NVIDIA GPU).

---

## 1. Fundamentação Teórica e Métricas

A energia elétrica consumida ($E$, em Joules) é a integral da potência instantânea ($P(t)$, em Watts) consumida pelo sistema ao longo do tempo de execução do algoritmo ($\Delta t$, em segundos):

$$E = \int_{t_0}^{t_1} P(t) \, dt \approx \sum_{i=1}^{M} P_i \times \Delta t_i \quad [\text{Joules}]$$

Para avaliar o equilíbrio entre desempenho computacional e eficiência energética, as seguintes métricas devem ser calculadas para cada cenário:

1. **Energia do Pacote da CPU ($E_{\text{CPU}}$)**: Energia consumida pelo processador (núcleos, caches e controlador de memória), expressa em Joules.
2. **Energia da GPU ($E_{\text{GPU}}$)**: Energia consumida pelo circuito integrado da GPU (SMs, memórias VRAM e barramentos internos), expressa em Joules.
3. **Energia Total do Processamento ($E_{\text{Total}}$)**:
   $$E_{\text{Total}} = E_{\text{CPU}} + E_{\text{GPU}} \quad [\text{Joules}]$$
4. **Potência Média ($\bar{P}$)**:
   $$\bar{P} = \frac{E_{\text{Total}}}{\Delta t} \quad [\text{Watts}]$$
5. **Eficiência Energética na Inferência ($\eta_{\text{pred}}$)**: Quantidade de amostras classificadas por unidade de energia consumida:
   $$\eta_{\text{pred}} = \frac{N_{\text{amostras}}}{E_{\text{pred}}} \quad \left[\frac{\text{amostras}}{\text{Joule}}\right]$$
6. **Eficiência Energética no Treinamento ($\eta_{\text{treino}}$)**: Quantidade de árvores induzidas por unidade de energia consumida:
   $$\eta_{\text{treino}} = \frac{N_{\text{árvores}}}{E_{\text{treino}}} \quad \left[\frac{\text{árvores}}{\text{Joule}}\right]$$
7. **Energy-Delay Product (EDP)**: Métrica canônica em computação de alto desempenho que penaliza soluções lentas, mesmo que operem sob baixa potência:
   $$\text{EDP} = E_{\text{Total}} \times \Delta t \quad [\text{Joules} \cdot \text{segundos}]$$

---

## 2. Sensores de Hardware e Mecanismos Nativos no Windows 11

A abordagem mais rigorosa e reproduzível baseia-se na leitura direta dos registradores acumuladores de hardware disponibilizados pelos fabricantes de processador e placa gráfica.

### A. Medição da CPU: Intel RAPL via Windows PDH
- **Tecnologia**: Intel RAPL (Running Average Power Limit).
- **Mapeamento no SO**: O Windows 11 mapeia os contadores MSR (Model-Specific Registers) da CPU no subsistema PDH (Performance Data Helper / `pdh.dll`).
- **Contador Utilizado**: `\Medidor de Energia(rapl_package0_pkg)\energia` (ou `\Energy Meter(rapl_package0_pkg)\Energy` em sistemas com idioma inglês).
- **Unidade**: O contador expõe o valor cumulativo em nanojoules ($10^{-9}$ Joules).
- **Cálculo da Energia**:
  $$E_{\text{CPU}} = \frac{C_{\text{fim}} - C_{\text{início}}}{10^9} \quad [\text{Joules}]$$

### B. Medição da GPU: NVIDIA NVML
- **Tecnologia**: NVIDIA Management Library (`nvml.dll`, nativa no diretório `System32` ao instalar o driver gráfico).
- **Função Principal**: `nvmlDeviceGetTotalEnergyConsumption(device, &energy_mJ)`. Retorna o valor cumulativo de energia registrado pelos sensores físicos de corrente e tensão da GPU em milijoules ($10^{-3}$ Joules).
- **Cálculo da Energia**:
  $$E_{\text{GPU}} = \frac{G_{\text{fim}} - G_{\text{início}}}{1000} \quad [\text{Joules}]$$
- **Potência Instantânea**: `nvmlDeviceGetPowerUsage(device, &power_mW)`, reportando a taxa instantânea em miliwatts.

---

## 3. Script de Medição Automatizada (`measure_energy.py`)

O script abaixo utiliza exclusivamente chamadas nativas de API via `ctypes` (sem necessidade de instalar bibliotecas de terceiros). Ele pode ser invocado diretamente da linha de comando para monitorar qualquer um dos três executáveis do projeto.

```python
#!/usr/bin/env python3
"""
measure_energy.py

Monitor de consumo energetico para CPU (Intel RAPL via Windows PDH)
e GPU (NVIDIA NVML) para comparacao entre Sequencial, OpenMP e CUDA.
"""

import ctypes
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# 1. Modulo CPU: Intel RAPL via Windows Performance Data Helper (pdh.dll)
# ---------------------------------------------------------------------------
pdh = ctypes.windll.pdh
h_query = ctypes.c_void_p()
pdh.PdhOpenQueryW(None, 0, ctypes.byref(h_query))

class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [
        ("CStatus", ctypes.c_uint32),
        ("doubleValue", ctypes.c_double),
    ]

# Tenta os nomes localizados (Portugues) e padrão (Ingles)
counter_paths = [
    r"\Medidor de Energia(rapl_package0_pkg)\energia",
    r"\Energy Meter(rapl_package0_pkg)\Energy",
]

h_cpu_counter = ctypes.c_void_p()
counter_active = False
for cp in counter_paths:
    ret = pdh.PdhAddCounterW(h_query, cp, 0, ctypes.byref(h_cpu_counter))
    if ret == 0:
        counter_active = True
        break

def read_cpu_energy_joules():
    if not counter_active:
        return 0.0
    pdh.PdhCollectQueryData(h_query)
    val = PDH_FMT_COUNTERVALUE()
    ret = pdh.PdhGetFormattedCounterValue(h_cpu_counter, 0x00000200, None, ctypes.byref(val))
    if ret == 0:
        return val.doubleValue / 1e9  # Converte de nanojoules para Joules
    return 0.0

# ---------------------------------------------------------------------------
# 2. Modulo GPU: NVIDIA NVML (nvml.dll)
# ---------------------------------------------------------------------------
nvml_active = False
nvml_dev = ctypes.c_void_p()
try:
    nvml = ctypes.CDLL("nvml.dll")
    if nvml.nvmlInit_v2() == 0:
        if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(nvml_dev)) == 0:
            nvml_active = True
except Exception:
    nvml_active = False

def read_gpu_energy_joules():
    if not nvml_active:
        return 0.0
    e_val = ctypes.c_ulonglong()
    ret = nvml.nvmlDeviceGetTotalEnergyConsumption(nvml_dev, ctypes.byref(e_val))
    if ret == 0:
        return e_val.value / 1000.0  # Converte de milijoules para Joules
    return 0.0

def read_gpu_power_watts():
    if not nvml_active:
        return 0.0
    p_val = ctypes.c_uint()
    ret = nvml.nvmlDeviceGetPowerUsage(nvml_dev, ctypes.byref(p_val))
    if ret == 0:
        return p_val.value / 1000.0  # Converte de miliwatts para Watts
    return 0.0

# ---------------------------------------------------------------------------
# 3. Execucao e Coleta
# ---------------------------------------------------------------------------
def run_and_measure(cmd_list, cwd=None):
    # Leituras iniciais
    cpu_e0 = read_cpu_energy_joules()
    gpu_e0 = read_gpu_energy_joules()
    t0 = time.perf_counter()

    proc = subprocess.run(cmd_list, cwd=cwd, capture_output=True, text=True)

    # Leituras finais
    t1 = time.perf_counter()
    cpu_e1 = read_cpu_energy_joules()
    gpu_e1 = read_gpu_energy_joules()

    dt = t1 - t0
    delta_cpu_j = max(0.0, cpu_e1 - cpu_e0)
    delta_gpu_j = max(0.0, gpu_e1 - gpu_e0)
    total_j = delta_cpu_j + delta_gpu_j
    avg_power_w = total_j / dt if dt > 0 else 0.0
    edp = total_j * dt

    print("=================================================================")
    print(" Relatorio de Consumo Energetico")
    print("=================================================================")
    print(f"Comando Executado      : {' '.join(cmd_list)}")
    print(f"Tempo de Parede (dt)   : {dt:.4f} s")
    print(f"Energia CPU (Package)  : {delta_cpu_j:.4f} Joules  (Potencia Media: {delta_cpu_j/dt:.2f} W)")
    print(f"Energia GPU (NVIDIA)   : {delta_gpu_j:.4f} Joules  (Potencia Media: {delta_gpu_j/dt:.2f} W)")
    print(f"Energia Total (CPU+GPU): {total_j:.4f} Joules  (Potencia Media: {avg_power_w:.2f} W)")
    print(f"Energy-Delay (EDP)     : {edp:.4f} J*s")
    print("=================================================================\n")

    return proc.stdout, proc.stderr, proc.returncode

def cleanup():
    if counter_active:
        pdh.PdhCloseQuery(h_query)
    if nvml_active:
        nvml.nvmlShutdown()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python measure_energy.py <comando> [argumentos...]")
        print("Exemplo: python measure_energy.py ../rf.exe --data ../data/iris.csv")
        cleanup()
        sys.exit(1)

    try:
        stdout, stderr, code = run_and_measure(sys.argv[1:])
        if stdout:
            print("--- Saida do Programa ---")
            print(stdout)
        if stderr:
            print("--- Erros / Avisos ---", file=sys.stderr)
            print(stderr, file=sys.stderr)
        sys.exit(code)
    finally:
        cleanup()
```

---

## 4. Protocolo Experimental Recomendado

Para garantir comparabilidade rigorosa entre Sequencial, OpenMP e CUDA, recomenda-se seguir os seguintes passos:

1. **Estado de Repouso (Baseline Idle Power)**:
   Antes de disparar os testes, medir a potência do sistema em repouso por 5 segundos ($P_{\text{idle}}$). Isso permite discriminar a energia dinâmica atribuível diretamente ao algoritmo da energia estática consumida pela máquina em espera:
   $$E_{\text{dinâmica}} = E_{\text{Total}} - (P_{\text{idle}} \times \Delta t)$$

2. **Fase de Aquecimento (Warm-up)**:
   Executar um teste rápido preliminar antes da coleta de dados. Isso preenche os caches do sistema de arquivos e carrega os módulos de driver da GPU na memória, evitando distorções no primeiro lote.

3. **Repetições e Estatística**:
   Executar cada configuração no mínimo 3 a 5 vezes consecutivas. Reportar a média aritmética e o desvio padrão da energia e do tempo de execução.

4. **Estabilidade Térmica**:
   Aguardar um intervalo de 5 a 10 segundos entre testes consecutivos para dissipação térmica da CPU e da GPU, prevenindo modulação forçada de frequência por temperatura (thermal throttling).

---

## 5. Exemplos de Execução Prática

### Teste 1: Comparação no Dataset de Médio Porte (`letter-recognition.data`)
Pode ser executado diretamente pelo Makefile ou via script individual:
```bash
# Execução automatizada comparando Sequencial, OpenMP e CUDA:
make energy-test1

# Ou invocando o script individualmente:
# 1. Sequencial
python python/measure_energy.py ./rf.exe --data data/letter-recognition.data --target 0 --trees 100 --max-depth 8 --mtry 4

# 2. OpenMP (16 threads)
python python/measure_energy.py ./openmp/rf.exe --data data/letter-recognition.data --target 0 --trees 100 --max-depth 8 --mtry 4

# 3. CUDA (GPU)
python python/measure_energy.py ./cuda/rf_cuda.exe --data data/letter-recognition.data --target 0 --trees 100 --max-depth 8 --mtry 4 --no-cpu-baseline
```

### Teste 2: Cenário de Alta Demanda (`covtype.csv`, max_depth = 30)
```bash
# Execução automatizada via Makefile:
make energy-test2

# Ou invocando os comandos individuais:
# 1. OpenMP
python python/measure_energy.py ./openmp/rf.exe --data ../data/covtype.csv --trees 50 --max-depth 30 --mtry 18 --seed 42

# 2. CUDA
python python/measure_energy.py ./cuda/rf_cuda.exe --data ../data/covtype.csv --trees 50 --max-depth 30 --mtry 18 --seed 42 --no-cpu-baseline
```

---

## 6. Interpretação dos Resultados Esperados

1. **Sequencial vs. OpenMP (Treinamento)**:
   Embora o OpenMP atinja potências instantâneas mais elevadas (por exemplo, 35 W a 45 W contra 12 W a 15 W do sequencial), a redução dramática no tempo de execução ($\Delta t$) faz com que a energia total consumida ($E = \bar{P} \times \Delta t$) seja substancialmente menor no OpenMP. O produto energia-atraso (EDP) do OpenMP é tipicamente muito superior.

2. **CPU vs. GPU (Inferência)**:
   A GPU apresenta uma potência ativa mais alta durante a execução dos kernels (25 W a 40 W na RTX 3050 Laptop). No entanto, como a predição é concluída em dezenas de milissegundos (versus centenas ou milhares de milissegundos na CPU), o consumo energético total por predição ($\text{Joules}/\text{amostra}$) na GPU é consideravelmente inferior, demonstrando alta eficiência energética.

