# Random Forest em C e CUDA

Implementação do algoritmo de aprendizado de máquina Random Forest (árvores de decisão CART) desenvolvida em C11 e CUDA (C++17), comparando três abordagens de computação paralela:
1. **Sequencial (C11)**: Linha de base de referência em uma única thread com recursão e alocação dinâmica.
2. **OpenMP (Multi-core CPU)**: Paralelização em nível de árvore no treinamento e em nível de amostra na predição.
3. **CUDA (NVIDIA GPU)**: Aceleração em hardware gráfico tanto na inferência massiva quanto no treinamento do modelo diretamente na GPU.

---

## Estrutura do Repositório

- `src/`: Código-fonte da versão sequencial em C11 (`main.c`, `rf.c`, `rf.h`, `dataset.c`).
- `openmp/`: Versão paralela para CPU multi-core com diretivas OpenMP (`openmp/src/main.c`, `openmp/src/rf.c`).
- `cuda/`: Versão acelerada em GPU com CUDA Runtime (`cuda/src/main.cu`, `cuda/src/rf_gpu.cu`, `cuda/src/rf_host.c`, `cuda/src/rf_cuda.h`).
- `data/`: Conjuntos de dados tabulares utilizados para experimentação (`iris.csv`, `breast-cancer.csv`, `sales_data.csv`, `letter-recognition.data`, `covtype.csv`).
- `python/`: Scripts automatizados para execução de benchmarks comparativos (`python/benchmark_comparison.py`).
- `BENCHMARK_RESULTS.md`: Relatório técnico consolidado com métricas de tempo, speedup e acurácia.

---

## Métodos e Técnicas Implementadas

### 1. Algoritmo Fundamental (CART Random Forest)
- **Critério de Divisão**: Impureza de Gini, calculada como $I_G(p) = 1 - \sum_{k=1}^K p_k^2$.
- **Amostragem com Reposição (Bootstrap)**: Cada árvore seleciona $N$ amostras do conjunto de treinamento com reposição.
- **Amostragem de Atributos ($mtry$)**: Em cada nó candidato, sorteia-se um subconjunto de tamanho $mtry = \lfloor\sqrt{d}\rfloor$ atributos sem reposição.
- **Ponto de Corte**: Determinado pela média aritmética $0{,}5 \times (v_i + v_{i+1})$ entre valores contíguos ordenados com menor impureza ponderada.
- **Gerador Pseudoaleatório Determinístico**: Implementação de xorshift32 com inicialização independente por árvore ($seed_t = seed + t \times 0x9E3779B9 + 1$).
- **Critério de Classificação**: Votação majoritária entre todas as árvores da floresta.

### 2. Paralelização em CPU com OpenMP
- **Treinamento (`forest_train`)**: Paralelismo coarse-grained com `#pragma omp parallel for schedule(dynamic)`. Cada núcleo processa uma árvore independente com arena local de nós e estado RNG próprio, eliminando qualquer contenção de memória compartilhada.
- **Predição (`forest_predict`)**: Paralelismo fine-grained com `#pragma omp parallel for schedule(static)` sobre as linhas do conjunto de teste, com leitura concorrente das árvores já construídas.

### 3. Treinamento e Inferência em GPU com CUDA

#### A. Estratégia de Treinamento em Duplo Kernel
1. **Kernel Cooperativo em Memória Compartilhada (`rf_train_forest_kernel_small`)**:
   - Destinado a conjuntos de dados onde o número de amostras cabe na memória compartilhada on-chip da GPU ($N \le 1024$), como `iris`, `breast-cancer` e `sales_data`.
   - Cada bloco de threads CUDA (256 threads) treina uma árvore completa de decisão de forma independente.
   - Vetores de amostras, pares (valor, classe) e histogramas são mantidos em aproximadamente 26 KB de SRAM compartilhada por bloco.
   - A ordenação das amostras é realizada através de uma rede de ordenação bitônica cooperativa (`bitonic_sort_pairs_dev_shared`), eliminando acessos repetitivos à memória global externa (VRAM).
2. **Kernel em Memória Global com Lotes Dinâmicos (`rf_train_forest_kernel`)**:
   - Destinado a grandes volumes de dados ($N > 1024$), como `letter-recognition` ($N = 16.000$) e `covtype` ($N = 464.809$).
   - Espaço de trabalho em VRAM alocado sob demanda e controlado em lotes de árvores (espaço limitado a 512 MB por lote) para evitar esgotamento de memória em GPUs dedicadas de entrada (por exemplo, placas de 4 GB).
   - Executa ordenação bitônica paralela em memória global (`bitonic_sort_pairs_dev_global`) e reduções paralelas de Gini.
3. **Equivalência Algorítmica DFS LIFO**:
   - Utilização de pilha iterativa LIFO (Last-In, First-Out) no dispositivo, empilhando primeiro o filho direito e depois o esquerdo.
   - Reproduz estritamente a travessia pré-ordem da CPU, consumindo os números aleatórios na mesma sequência da referência recursiva.

#### B. Inferência Paralela Massiva (`rf_predict_kernel`)
- Cada thread da GPU avalia uma amostra inteira de teste contra todas as árvores da floresta de modo contíguo.
- As consultas às variáveis e aos nós das árvores aproveitam as caches L1/L2 da GPU, entregando latência na faixa de sub-milissegundos.

---

## Compilação e Execução

### Pré-requisitos
- Compilador C compatível com C11 (GCC / MinGW / Clang / MSVC).
- Suporte a OpenMP (`-fopenmp`).
- NVIDIA CUDA Toolkit 12 ou 13 com `nvcc` e host compiler C++17.

### 1. Versão Sequencial (C11)
```bash
# Na raiz do projeto RandomForest:
make
./rf --data data/iris.csv --trees 100 --max-depth 8 --min-samples 2 --seed 42
```

### 2. Versão Paralela OpenMP
```bash
cd openmp
make
./rf.exe --data ../data/covtype.csv --trees 50 --max-depth 8 --mtry 4 --seed 42
```

### 3. Versão Acelerada em CUDA (GPU)
```bash
cd cuda
make
./rf_cuda.exe --data ../data/letter-recognition.data --target 0 --trees 100 --max-depth 8 --mtry 4 --seed 42
```

### Argumentos de Linha de Comando Suportados
```text
  --data PATH         Caminho do arquivo CSV de dados (padrão: data/iris.csv)
  --target COL        Nome da coluna alvo ou índice zero-based (padrão: última coluna)
  --trees N           Quantidade de árvores na floresta (padrão: 100)
  --max-depth D       Profundidade máxima permitida por árvore (padrão: 8)
  --min-samples M     Quantidade mínima de amostras para tentar divisão (padrão: 2)
  --mtry K            Quantidade de variáveis sorteadas por nó (0 = floor(sqrt(p)))
  --test-frac F       Fração do conjunto reservada para teste holdout (padrão: 0.20)
  --seed S            Semente base inicial do gerador pseudoaleatório (padrão: 42)
  --dot FILE          Exporta a topologia de uma árvore para formato Graphviz DOT
  --no-cpu-baseline   Desabilita o treino redundante de CPU no executável CUDA (modo rápido)
```

---

## Resultados dos Benchmarks

Ambiente de teste: Processador x86_64 (16 threads lógicas), GPU NVIDIA GeForce RTX 3050 Laptop (4 GB VRAM, 16 SMs, CUDA Compute Capability 8.6), Windows 11.

### 1. Comparativo Geral com Profundidade Controlada (max_depth = 8)

| Dataset | Amostras Treino | Amostras Teste | Atributos | Classes | Árvores | Treino Seq (s) | Treino OMP (s) | Treino CUDA (s) | Speedup Treino OMP | Speedup Treino CUDA | Predição Seq (ms) | Predição OMP (ms) | Predição CUDA (ms) | Predição Kernel (ms) | Speedup Predição CUDA | Acurácia Teste |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **iris** | 120 | 30 | 4 | 3 | 100 | 0,0037 s | 0,0020 s | 0,0052 s | 1,87x | 0,71x | 0,14 ms | 0,45 ms | 1,68 ms | 0,175 ms | 0,08x | 0,9333 |
| **breast-cancer** | 455 | 114 | 30 | 2 | 100 | 0,0775 s | 0,0118 s | 0,0557 s | 6,57x | **1,39x** | 1,22 ms | 0,75 ms | 2,94 ms | 0,204 ms | 0,42x | 0,9825 |
| **sales_data** | 800 | 200 | 13 | 4 | 100 | 0,0937 s | 0,0141 s | 0,0875 s | 6,63x | **1,07x** | 6,86 ms | 1,93 ms | 2,98 ms | 0,246 ms | **2,31x** | 0,3000 |
| **letter-recog** | 16.000 | 4.000 | 16 | 26 | 100 | 1,7000 s | 0,2369 s | 1,5824 s | 7,18x | **1,07x** | 129,29 ms | 14,59 ms | 6,28 ms | 0,449 ms | **20,60x** | 0,7505 |
| **covtype** | 464.809 | 116.203 | 54 | 7 | 50 | 24,4033 s | 5,3620 s | 28,6819 s | 4,55x | 0,85x | 908,51 ms | 107,33 ms | 106,56 ms | 13,717 ms | **8,53x** | 0,6382 |

### 2. Comparativo em Alta Profundidade (covtype.csv, max_depth = 30, mtry = 18, trees = 50)

Cenário de estresse computacional com nós profundos (cerca de 48.000 nós por árvore individual):

| Métrica / Abordagem | Sequencial (C11, 1 thread) | OpenMP (16 threads CPU) | CUDA (NVIDIA RTX 3050 Laptop) |
| :--- | :--- | :--- | :--- |
| **Tempo de Treino** | 213,9990 s | **42,0602 s** | 524,8085 s |
| **Tempo de Predição Fim a Fim** | 12.341,48 ms | 1.084,81 ms | **818,12 ms** |
| **Tempo de Predição no Kernel** | N/A | N/A | **101,50 ms** |
| **Speedup no Treino** | 1,00x | **5,09x** | 0,41x |
| **Speedup na Predição (Fim a Fim)** | 1,00x | 11,38x | **15,08x** (vs. Seq) / **1,33x** (vs. OMP) |
| **Speedup na Predição (Kernel Puro)**| 1,00x | N/A | **121,59x** (vs. Seq) / **10,69x** (vs. OMP) |
| **Acurácia no Teste** | 0,9599 (95,99%) | 0,9599 (95,99%) | 0,9603 (96,03%) |
| **Divergências de Predição vs. CPU** | 0 | 0 | **0 / 116.203** |

---

## Análise Comparativa e Conclusões

1. **Fase de Treinamento**:
   - **OpenMP**: Apresenta a melhor eficiência geral na indução das árvores (speedup entre 4,5x e 7,2x). A recursão com quicksort local beneficia-se diretamente de caches L1/L2 dedicados por núcleo e predição de desvios da CPU.
   - **CUDA**: Mostra-se eficiente em conjuntos moderados onde os nós cabem em memória compartilhada (atingindo até 1,39x de aceleração sobre a CPU sequencial no dataset `breast-cancer`). Em árvores de profundidade extrema com milhões de amostras e dezenas de milhares de nós, a sobrecarga de sincronizações de barreira na ordenação bitônica reduz a eficiência relativa, mas mantém o modelo inteiramente residente na VRAM para consumo imediato.

2. **Fase de Inferência (Predição)**:
   - A GPU apresenta ganho massivo de taxa de transferência, atingindo até **20,6x de speedup fim a fim** no `letter-recognition` e avaliando mais de 116 mil amostras em apenas **101,5 ms de kernel** no `covtype`.
   - A avaliação de amostras é embaraçosamente paralela (uma thread por linha), ideal para o modelo de execução SIMT da GPU.

3. **Correção Numérica**:
   - As predições geradas pelos modelos construídos na GPU demonstraram consistência absoluta com a CPU de referência, registrando 0 divergências em todos os conjuntos de teste avaliados.

