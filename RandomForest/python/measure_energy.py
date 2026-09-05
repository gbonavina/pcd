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
        return val.doubleValue / 1e9  # Nanojoules -> Joules
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
        return e_val.value / 1000.0  # Milijoules -> Joules
    return 0.0

def read_gpu_power_watts():
    if not nvml_active:
        return 0.0
    p_val = ctypes.c_uint()
    ret = nvml.nvmlDeviceGetPowerUsage(nvml_dev, ctypes.byref(p_val))
    if ret == 0:
        return p_val.value / 1000.0  # Miliwatts -> Watts
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
    print(f"Energia CPU (Package)  : {delta_cpu_j:.4f} Joules  (Potencia Media: {delta_cpu_j/dt if dt > 0 else 0:.2f} W)")
    print(f"Energia GPU (NVIDIA)   : {delta_gpu_j:.4f} Joules  (Potencia Media: {delta_gpu_j/dt if dt > 0 else 0:.2f} W)")
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

