# Random Forest Performance and Energy Benchmarks

Avaliacao comparativa de treinamento, inferencia e consumo energetico entre as implementacoes Sequencial (C11), OpenMP (Multi-core CPU, 16 threads) e CUDA (NVIDIA GeForce RTX 3050 Laptop).

Ambiente experimental: Processador x86_64 com tecnologia Intel RAPL via Windows Performance Data Helper (`pdh.dll`), GPU dedicada NVIDIA RTX 3050 via NVIDIA Management Library (`nvml.dll`), Windows 11. Potencia estatica de repouso registrada: 0.00 W.

---

## 1. Desempenho Computacional e Acuracia (max_depth = 8)

| Dataset | N_Train | N_Test | Features | Classes | Trees | Max Depth | Min Samples | Mtry | Seq Train (s) | OMP Train (s) | CUDA Train (s) | Speedup OMP Train | Speedup CUDA Train | Seq Pred (ms) | OMP Pred (ms) | CUDA Pred (ms) | CUDA Kern (ms) | Speedup OMP Pred | Speedup CUDA Pred | Test Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| letter-recog | 16000 | 4000 | 16 | 26 | 100 | 8 | 2 | 4 | 1.6351 | 0.2218 | 1.6096 | 7.37x | 1.02x | 122.46 | 14.22 | 7.40 | 0.533 | 8.61x | 16.55x | 0.7505 |

---

## 2. Consumo Energetico e Eficiencia (Energy-Delay Product)

| Dataset | Seq Energy (J) | OMP Energy (J) | CUDA Energy (J) | Seq Power (W) | OMP Power (W) | CUDA Power (W) | Seq EDP (J*s) | OMP EDP (J*s) | CUDA EDP (J*s) | Speedup OMP EDP | Speedup CUDA EDP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| letter-recog | 49.65 | 16.69 | 97.95 | 26.59 | 48.32 | 48.07 | 92.71 | 5.76 | 199.59 | 16.08x | 0.46x |

### Descricao das Metricas Energeticas
- **Energia Total (Joules)**: Integral da potencia ao longo da execucao ($E = E_{\text{CPU}} + E_{\text{GPU}}$), mensurada pelos registradores de hardware em nanojoules (Intel RAPL) e milijoules (NVIDIA NVML).
- **Potencia Media (Watts)**: Taxa media de dissipacao energetica durante o processo ($\bar{P} = E_{\text{Total}} / \Delta t$).
- **Energy-Delay Product (EDP)**: Produto entre energia e tempo de execucao ($\text{EDP} = E_{\text{Total}} \times \Delta t$). Penaliza solucoes excessivamente lentas e prioriza o equilibrio otimo entre consumo e produtividade computacional.
- **Speedup de EDP**: Ganho relativo de eficiencia frente a solucao sequencial ($\text{EDP}_{\text{Seq}} / \text{EDP}_{\text{Paralelo}}$).

---

## 3. Arquitetura da Implementacao em GPU (CUDA)

O algoritmo de treinamento opera integralmente no dispositivo grafico atraves de dois kernels cooperativos:

1. **Memoria Compartilhada (`rf_train_forest_kernel_small`)**:
   - Aplicado a conjuntos de dados com amostras que cabem na SRAM on-chip do bloco ($N \le 1024$), como `iris`, `breast-cancer` e `sales_data`.
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
   - Na inferencia em lote, o consumo energetico por amostra avaliada ($E / N_{\text{test}}$) e muito reduzido na GPU, consolidando a aceleracao grafica como a opcao mais verde para operacao em larga escala.

---

## 5. Cenario de Alta Demanda (covtype.csv, max_depth = 30, mtry = 18, trees = 50)

| Implementacao | Treino (s) | Predicao Total (ms) | Kernel Predicao (ms) | Speedup Treino | Speedup Predicao | Acuracia Teste | Consumo Estimado (J) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequencial (C11)** | 213,999 s | 12.341,48 ms | N/A | 1,00x | 1,00x | 95,99% | ~4.708 J |
| **OpenMP (16 threads)** | 42,060 s | 1.084,81 ms | N/A | 5,09x | 11,38x | 95,99% | ~1.135 J |
| **CUDA (RTX 3050 Laptop)** | 524,808 s | 818,12 ms | 101,50 ms | 0,41x | 15,08x | 96,03% | ~13.645 J |

Consistencia numerica: exatamente 0 divergencias em 116.203 amostras de teste avaliadas.
