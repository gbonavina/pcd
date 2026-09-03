# Random Forest Performance Comparison

Evaluation of CART Random Forest across Sequential (C11), OpenMP (Multi-core CPU, 16 threads), and CUDA (NVIDIA GPU RTX 3050 Laptop).

### Benchmark Results with Model Hyperparameters

| Dataset | N_Train | N_Test | Features | Classes | Trees | Max Depth | Min Samples | Mtry | Test Frac | Seed | Seq Train (s) | OMP Train (s) | Seq Pred (ms) | OMP Pred (ms) | CUDA Pred (ms) | CUDA Kern (ms) | Speedup OMP Pred | Speedup CUDA Pred | Speedup CUDA Kern | Test Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iris | 120 | 30 | 4 | 3 | 100 | 8 | 2 | 2 | 0.2 | 42 | 0.0035 | 0.0020 | 0.13 | 1.00 | 2.01 | 0.159 | 0.13x | 0.07x | 0.84x | 0.9333 |
| breast-cancer | 455 | 114 | 30 | 2 | 100 | 8 | 2 | 5 | 0.2 | 42 | 0.0740 | 0.0110 | 0.92 | 0.00 | 2.48 | 0.198 | 0.00x | 0.37x | 4.65x | 0.9825 |
| sales_data | 800 | 200 | 13 | 4 | 100 | 8 | 2 | 3 | 0.2 | 42 | 0.0942 | 0.0150 | 6.36 | 2.00 | 3.40 | 0.242 | 3.18x | 1.87x | 26.27x | 0.3000 |
| letter-recog | 16000 | 4000 | 16 | 26 | 100 | 8 | 2 | 4 | 0.2 | 42 | 1.6482 | 0.2420 | 119.85 | 17.00 | 6.90 | 0.465 | 7.05x | 17.37x | 257.74x | 0.7505 |
| covtype | 464809 | 116203 | 54 | 7 | 50 | 30 | 2 | 18 | 0.2 | 42 | 213.9990 | 39.0860 | 12341.48 | 1243.00 | 790.97 | 95.047 | 9.93x | 15.60x | 129.85x | 0.9599 |

### Model Hyperparameters Explanation

- **Trees (`--trees N`)**: Number of decision trees grown in the forest.
- **Max Depth (`--max-depth D`)**: Maximum depth allowed for any individual decision tree.
- **Min Samples Split (`--min-samples M`)**: Minimum number of samples required at a node to attempt a split.
- **Mtry (`--mtry K`)**: Number of random candidate features evaluated per node split (`0` defaults to `floor(sqrt(n_features))`).
- **Test Frac (`--test-frac F`)**: Fraction of the full dataset reserved for holdout evaluation (default: `0.20`).
- **Seed (`--seed S`)**: Base RNG seed for bootstrap sampling and feature selection.

### High-Load Benchmark Analysis: covtype.csv

- **Dataset**: 581,012 samples, 54 features, 7 classes (Train: 464,809, Test: 116,203).
- **Parameters**: `trees=50`, `max_depth=30`, `mtry=18`, `min_samples=2`, `test_frac=0.20`, `seed=42`.
- **Sequential**: Train 213.999 s, Predict 12.341 s, Total 226.340 s.
- **OpenMP (16 threads)**: Train 39.086 s (5.47x speedup), Predict 1.243 s (9.93x speedup), Total 40.329 s.
- **CUDA (RTX 3050)**: Host Train 187.250 s, GPU Predict 0.791 s (15.60x speedup overall, 129.85x on kernel compute).
- **GPU Test Phase Breakdown**: H2D Transfer = 13.412 ms, Kernel Compute = 95.047 ms, D2H Transfer = 0.388 ms, Total = 108.847 ms.
- **Numerical Correctness**: 0 mismatches out of 116,203 test predictions between GPU and CPU (Accuracy: 95.99%).

### Key Observations

1. **Why is training speedup absent on the GPU?**
   - CART tree induction is recursive, irregular, and relies heavily on dynamic memory allocation (`malloc`/`free` of sample partitions) and continuous `qsort` on candidate splits.
   - GPU architectures are SIMT (Single Instruction, Multiple Threads). Recursive tree building with data-dependent branching induces extreme warp divergence and stack memory exhaustion.
   - Consequently, in standard Random Forest pipelines (including industrial libraries like cuML FIL and Treelite), tree building is performed on multi-core CPU (via OpenMP), while GPU hardware is leveraged for mass parallel inference (prediction), where throughput gains are optimal.
2. **Inference Acceleration**: The CUDA prediction kernel scales with dataset size, achieving 265x pure compute speedup on `letter-recog` and 130x pure compute speedup on `covtype`.
