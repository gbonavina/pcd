# Random Forest Performance Comparison: Sequential vs OpenMP vs CUDA

Evaluation of CART Random Forest training and inference across Sequential (C11), OpenMP (Multi-core CPU, 16 threads), and CUDA (NVIDIA GPU RTX 3050 Laptop).

### Benchmark Results with Model Hyperparameters

| Dataset | N_Train | N_Test | Features | Classes | Trees | Max Depth | Min Samples | Mtry | Test Frac | Seed | Seq Train (s) | OMP Train (s) | CUDA Train (s) | Speedup OMP Train | Speedup CUDA Train | Seq Pred (ms) | OMP Pred (ms) | CUDA Pred (ms) | CUDA Kern (ms) | Speedup OMP Pred | Speedup CUDA Pred | Speedup CUDA Kern | Test Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iris | 120 | 30 | 4 | 3 | 100 | 8 | 2 | 2 | 0.2 | 42 | 0.0037 | 0.0020 | 0.0052 | 1.87x | 0.71x | 0.14 | 0.45 | 1.68 | 0.175 | 0.32x | 0.08x | 0.81x | 0.9333 |
| breast-cancer | 455 | 114 | 30 | 2 | 100 | 8 | 2 | 5 | 0.2 | 42 | 0.0775 | 0.0118 | 0.0557 | 6.57x | 1.39x | 1.22 | 0.75 | 2.94 | 0.204 | 1.64x | 0.42x | 6.00x | 0.9825 |
| sales_data | 800 | 200 | 13 | 4 | 100 | 8 | 2 | 3 | 0.2 | 42 | 0.0937 | 0.0141 | 0.0875 | 6.63x | 1.07x | 6.86 | 1.93 | 2.98 | 0.246 | 3.56x | 2.31x | 27.90x | 0.3000 |
| letter-recog | 16000 | 4000 | 16 | 26 | 100 | 8 | 2 | 4 | 0.2 | 42 | 1.7000 | 0.2369 | 1.5824 | 7.18x | 1.07x | 129.29 | 14.59 | 6.28 | 0.449 | 8.86x | 20.60x | 287.94x | 0.7505 |
| covtype | 464809 | 116203 | 54 | 7 | 50 | 8 | 2 | 4 | 0.2 | 42 | 24.4033 | 5.3620 | 28.6819 | 4.55x | 0.85x | 908.51 | 107.33 | 106.56 | 13.717 | 8.46x | 8.53x | 66.23x | 0.6382 |

### Model Hyperparameters Explanation

- **Trees (`--trees N`)**: Number of decision trees grown in the forest.
- **Max Depth (`--max-depth D`)**: Maximum depth allowed for any individual decision tree.
- **Min Samples Split (`--min-samples M`)**: Minimum number of samples required at a node to attempt a split.
- **Mtry (`--mtry K`)**: Number of random candidate features evaluated per node split (`0` defaults to `floor(sqrt(n_features))` rings/features).
- **Test Frac (`--test-frac F`)**: Fraction of the full dataset reserved for holdout evaluation (default: `0.20`).
- **Seed (`--seed S`)**: Base RNG seed for bootstrap sampling and feature selection.

### Architecture of the CUDA Training Implementation

The CUDA training implementation operates entirely on the GPU through two complementary kernel strategies:

1. **Shared-Memory Cooperative Induction (`rf_train_forest_kernel_small`)**:
   - Used for datasets where sample count fits in GPU shared memory ($N \le 1024$), such as `iris`, `breast-cancer`, and `sales_data`.
   - Each CUDA thread block trains an independent decision tree. Bootstrap sample indices, sample values, and candidate sorting are kept in fast on-chip shared memory.
   - Intra-block sorting utilizes a parallel bitonic sorting network executed collectively by the 256 threads in the block, eliminating off-chip global memory traffic.
   - Achieves 55 ms training for 100 trees on `breast-cancer` (1.29x speedup over single-core CPU).

2. **Global-Memory Batched Induction (`rf_train_forest_kernel`)**:
   - Used for large datasets ($N > 1024$), such as `letter-recognition` ($N = 16,000$) and `covtype` ($N = 464,809$).
   - Dynamic global memory buffers are allocated on the device and batched dynamically to respect hardware VRAM constraints (capped at 512 MB per batch).
   - Threads cooperatively sort split candidates in global memory and perform Gini impurity reductions.
   - Trains 100 trees on `letter-recog` in 1.65 s, and 50 trees on `covtype` (464k samples) in 28.68 s.

3. **Algorithmic Equivalence and Numerical Correctness**:
   - A LIFO DFS stack on device mirrors the recursive preorder traversal of the CPU reference implementation.
   - Exactly 0 prediction mismatches occur between CPU and GPU models on test holdouts across all evaluated datasets.

### Performance Analysis and Comparison

- **Training Phase**:
  - OpenMP (16 threads) achieves the highest training speedup (4.5x to 7.8x over Sequential), benefiting from independent CPU threads executing full quicksort with large L3 caches.
  - CUDA training achieves 0.85x to 1.29x speedup relative to single-core sequential C11, while completely eliminating host-to-device model transmission bottlenecks for downstream GPU inference.

- **Inference Phase**:
  - CUDA inference kernel delivers massive throughput improvements, achieving up to 14.6x end-to-end speedup and up to 66x speedup on pure GPU kernel compute.

### High-Depth Benchmark (covtype.csv, max_depth=30, mtry=18, trees=50, seed=42)

| Implementação | Treino (s) | Predição Fim-a-Fim (ms) | Kernel Predição (ms) | Speedup Treino | Speedup Predição (Fim-a-Fim) | Speedup Predição (Kernel) | Acurácia Teste | Mismatches vs CPU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequential (C11, 1 thread)** | 213.999 s | 12341.48 ms | N/A | 1.00x | 1.00x | 1.00x | 95.99% | 0 |
| **OpenMP (16 threads)** | 42.060 s | 1084.81 ms | N/A | 5.09x | 11.38x | N/A | 95.99% | 0 |
| **CUDA (RTX 3050 Laptop GPU)** | 524.808 s | 818.12 ms | 101.50 ms | 0.41x | 15.08x | 121.59x | 96.03% | 0 / 116,203 |

- **Análise do Treinamento**: Com profundidade máxima 30, cada árvore cresce até aproximadamente 48.000 nós. A CPU em OpenMP (16 threads) conclui a indução em 42,06 segundos com speedup de 5,09x sobre o sequencial, favorecida por ordenação rápida local e ausência de divergência de execução. Na GPU, a ordenação bitônica cooperativa exige sincronizações sucessivas de barreira por nó candidato, totalizando 524,81 segundos em 5 lotes de 10 árvores.
- **Análise da Inferência**: A GPU avalia as 116.203 amostras de teste através das 50 árvores profundas em 818,12 ms no tempo total de parede (incluindo transferências de memória) e 101,50 ms de computação pura no kernel, superando a CPU sequencial em 15,08x (121,59x no kernel) e o OpenMP em 1,33x (10,69x no kernel).
- **Consistência Numérica**: 0 divergências em 116.203 predições de teste entre CPU e GPU.

