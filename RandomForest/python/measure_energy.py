#!/usr/bin/env python3
"""
measure_energy.py

Monitor de consumo energetico para CPU (Intel RAPL via Windows PDH)
e GPU (NVIDIA NVML) para comparacao entre Sequencial, OpenMP e CUDA.
Oferece interface programatica (EnergyMonitor) e execucao via CLI.
"""

import ctypes
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class EnergyMeasurement:
    """Dados consolidados de uma medicao de energia e tempo."""
    dt: float = 0.0
    cpu_j: float = 0.0
    gpu_j: float = 0.0
    total_j: float = 0.0
    dynamic_j: float = 0.0
    cpu_power_w: float = 0.0
    gpu_power_w: float = 0.0
    avg_power_w: float = 0.0
    edp: float = 0.0
    dynamic_edp: float = 0.0
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [
        ("CStatus", ctypes.c_uint32),
        ("doubleValue", ctypes.c_double),
    ]


class EnergyMonitor:
    """
    Gerenciador de sensores de energia para processadores x86_64
    (Intel RAPL via Windows PDH) e GPUs NVIDIA (via NVML).
    """

    def __init__(self):
        self.pdh = None
        self.h_query = ctypes.c_void_p()
        self.h_cpu_counter = ctypes.c_void_p()
        self.cpu_active = False

        self.nvml = None
        self.nvml_dev = ctypes.c_void_p()
        self.nvml_active = False

        self.last_idle_power_w = 0.0
        self.init_sensors()

    def init_sensors(self):
        # 1. Inicializacao do Windows PDH (Intel RAPL)
        try:
            self.pdh = ctypes.windll.pdh
            if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.h_query)) == 0:
                counter_paths = [
                    r"\Medidor de Energia(rapl_package0_pkg)\energia",
                    r"\Energy Meter(rapl_package0_pkg)\Energy",
                ]
                for cp in counter_paths:
                    ret = self.pdh.PdhAddCounterW(
                        self.h_query, cp, 0, ctypes.byref(self.h_cpu_counter)
                    )
                    if ret == 0:
                        self.cpu_active = True
                        break
        except Exception:
            self.cpu_active = False

        # 2. Inicializacao da NVIDIA NVML
        try:
            self.nvml = ctypes.CDLL("nvml.dll")
            if self.nvml.nvmlInit_v2() == 0:
                if self.nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.nvml_dev)) == 0:
                    self.nvml_active = True
        except Exception:
            self.nvml_active = False

    def read_cpu_energy_joules(self) -> float:
        """Le o contador acumulativo de energia do pacote da CPU em Joules."""
        if not self.cpu_active or not self.pdh:
            return 0.0
        self.pdh.PdhCollectQueryData(self.h_query)
        val = PDH_FMT_COUNTERVALUE()
        ret = self.pdh.PdhGetFormattedCounterValue(
            self.h_cpu_counter, 0x00000200, None, ctypes.byref(val)
        )
        if ret == 0:
            return val.doubleValue / 1e9  # nanojoules -> Joules
        return 0.0

    def read_gpu_energy_joules(self) -> float:
        """Le o contador acumulativo de energia da GPU em Joules."""
        if not self.nvml_active or not self.nvml:
            return 0.0
        e_val = ctypes.c_ulonglong()
        ret = self.nvml.nvmlDeviceGetTotalEnergyConsumption(
            self.nvml_dev, ctypes.byref(e_val)
        )
        if ret == 0:
            return e_val.value / 1000.0  # milijoules -> Joules
        return 0.0

    def read_gpu_power_watts(self) -> float:
        """Le a potencia instantanea reportada pela GPU em Watts."""
        if not self.nvml_active or not self.nvml:
            return 0.0
        p_val = ctypes.c_uint()
        ret = self.nvml.nvmlDeviceGetPowerUsage(self.nvml_dev, ctypes.byref(p_val))
        if ret == 0:
            return p_val.value / 1000.0  # miliwatts -> Watts
        return 0.0

    def measure_idle_power(self, duration_s: float = 3.0) -> float:
        """
        Mede a potencia estatica de repouso (P_idle) do sistema ao longo de duration_s.
        Permite isolar a energia dinamica consumida especificamente pelo algoritmo.
        """
        cpu0 = self.read_cpu_energy_joules()
        gpu0 = self.read_gpu_energy_joules()
        t0 = time.perf_counter()

        time.sleep(duration_s)

        t1 = time.perf_counter()
        cpu1 = self.read_cpu_energy_joules()
        gpu1 = self.read_gpu_energy_joules()

        dt = t1 - t0
        d_cpu = max(0.0, cpu1 - cpu0)
        d_gpu = max(0.0, gpu1 - gpu0)
        idle_w = (d_cpu + d_gpu) / dt if dt > 0 else 0.0
        self.last_idle_power_w = idle_w
        return idle_w

    def measure_command(
        self,
        cmd_list: List[str],
        cwd: Optional[str] = None,
        idle_power_w: Optional[float] = None,
    ) -> EnergyMeasurement:
        """
        Executa um comando e mede simultaneamente tempo de parede e consumo energetico.
        """
        if idle_power_w is None:
            idle_power_w = self.last_idle_power_w

        cpu_e0 = self.read_cpu_energy_joules()
        gpu_e0 = self.read_gpu_energy_joules()
        t0 = time.perf_counter()

        proc = subprocess.run(
            cmd_list,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        t1 = time.perf_counter()
        cpu_e1 = self.read_cpu_energy_joules()
        gpu_e1 = self.read_gpu_energy_joules()

        dt = t1 - t0
        delta_cpu = max(0.0, cpu_e1 - cpu_e0)
        delta_gpu = max(0.0, gpu_e1 - gpu_e0)
        total_j = delta_cpu + delta_gpu
        dynamic_j = max(0.0, total_j - (idle_power_w * dt))

        cpu_pw = delta_cpu / dt if dt > 0 else 0.0
        gpu_pw = delta_gpu / dt if dt > 0 else 0.0
        avg_pw = total_j / dt if dt > 0 else 0.0
        edp = total_j * dt
        dynamic_edp = dynamic_j * dt

        return EnergyMeasurement(
            dt=dt,
            cpu_j=delta_cpu,
            gpu_j=delta_gpu,
            total_j=total_j,
            dynamic_j=dynamic_j,
            cpu_power_w=cpu_pw,
            gpu_power_w=gpu_pw,
            avg_power_w=avg_pw,
            edp=edp,
            dynamic_edp=dynamic_edp,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )

    def measure_repetitions(
        self,
        cmd_list: List[str],
        n_repeats: int = 3,
        warmup: bool = True,
        cwd: Optional[str] = None,
        idle_power_w: Optional[float] = None,
        rest_interval_s: float = 2.0,
    ) -> Tuple[EnergyMeasurement, Dict[str, float]]:
        """
        Executa repeticoes com aquecimento (warmup) e intervalo termico,
        retornando o resultado medio e o desvio padrao das metricas.
        """
        if warmup:
            self.measure_command(cmd_list, cwd=cwd, idle_power_w=idle_power_w)
            if rest_interval_s > 0:
                time.sleep(rest_interval_s)

        measurements: List[EnergyMeasurement] = []
        for i in range(n_repeats):
            m = self.measure_command(cmd_list, cwd=cwd, idle_power_w=idle_power_w)
            measurements.append(m)
            if i < n_repeats - 1 and rest_interval_s > 0:
                time.sleep(rest_interval_s)

        n = len(measurements)
        avg_m = EnergyMeasurement(
            dt=sum(m.dt for m in measurements) / n,
            cpu_j=sum(m.cpu_j for m in measurements) / n,
            gpu_j=sum(m.gpu_j for m in measurements) / n,
            total_j=sum(m.total_j for m in measurements) / n,
            dynamic_j=sum(m.dynamic_j for m in measurements) / n,
            cpu_power_w=sum(m.cpu_power_w for m in measurements) / n,
            gpu_power_w=sum(m.gpu_power_w for m in measurements) / n,
            avg_power_w=sum(m.avg_power_w for m in measurements) / n,
            edp=sum(m.edp for m in measurements) / n,
            dynamic_edp=sum(m.dynamic_edp for m in measurements) / n,
            stdout=measurements[0].stdout,
            stderr=measurements[0].stderr,
            returncode=measurements[0].returncode,
        )

        std_dict = {}
        for attr in ["dt", "cpu_j", "gpu_j", "total_j", "avg_power_w", "edp"]:
            vals = [getattr(m, attr) for m in measurements]
            mean_val = sum(vals) / n
            variance = sum((x - mean_val) ** 2 for x in vals) / (n if n > 1 else 1)
            std_dict[f"{attr}_std"] = variance ** 0.5

        return avg_m, std_dict

    def cleanup(self):
        """Libera os handles de conexao com PDH e NVML."""
        if self.cpu_active and self.pdh:
            try:
                self.pdh.PdhCloseQuery(self.h_query)
            except Exception:
                pass
            self.cpu_active = False

        if self.nvml_active and self.nvml:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
            self.nvml_active = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def run_and_measure(cmd_list: List[str], cwd: Optional[str] = None):
    """
    Funcao de conveniencia retrocompativel para execucao e exibicao do relatorio.
    """
    with EnergyMonitor() as monitor:
        m = monitor.measure_command(cmd_list, cwd=cwd)

        print("=================================================================")
        print(" Relatorio de Consumo Energetico")
        print("=================================================================")
        print(f"Comando Executado      : {' '.join(cmd_list)}")
        print(f"Tempo de Parede (dt)   : {m.dt:.4f} s")
        print(
            f"Energia CPU (Package)  : {m.cpu_j:.4f} Joules  (Potencia Media: {m.cpu_power_w:.2f} W)"
        )
        print(
            f"Energia GPU (NVIDIA)   : {m.gpu_j:.4f} Joules  (Potencia Media: {m.gpu_power_w:.2f} W)"
        )
        print(
            f"Energia Total (CPU+GPU): {m.total_j:.4f} Joules  (Potencia Media: {m.avg_power_w:.2f} W)"
        )
        print(f"Energy-Delay (EDP)     : {m.edp:.4f} J*s")
        print("=================================================================\n")

        return m.stdout, m.stderr, m.returncode


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python measure_energy.py <comando> [argumentos...]")
        print("Exemplo: python measure_energy.py ../rf.exe --data ../data/iris.csv")
        sys.exit(1)

    stdout, stderr, code = run_and_measure(sys.argv[1:])
    if stdout:
        print("--- Saida do Programa ---")
        print(stdout)
    if stderr:
        print("--- Erros / Avisos ---", file=sys.stderr)
        print(stderr, file=sys.stderr)
    sys.exit(code)

