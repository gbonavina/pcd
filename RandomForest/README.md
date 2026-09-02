# Random Forest in C

From-scratch CART classification forest: bootstrap samples, random `mtry` features
at each split, Gini impurity, majority vote. Sequential C11. OpenMP sites are
marked in `src/rf.c` with commented pragmas.

## Build and run

```bash
make
./rf --data data/iris.csv --trees 100 --max-depth 8 --min-samples 2 --seed 42
```

CSV: optional header. `--target` selects the class column by **name** or
**0-based index** (default: last column). Labels are remapped to `0 .. K-1`.
Other columns may be numeric, ISO dates (`YYYY-MM-DD`), or categories.

```text
./rf --data PATH --target COL --trees N --max-depth D --min-samples M --mtry K --test-frac F --seed S
```

Example: classify product category in the sales file:

```bash
./rf --data data/sales_data.csv --target Product_Category --trees 100 --seed 42
```

`mtry` defaults to `sqrt(n_features)`.

## OpenMP map

Compile flag when you uncomment pragmas: `gcc -fopenmp` (see `Makefile`).

| Site | File | Why | Schedule / caveats |
|---|---|---|---|
| Grow trees `t = 0 .. n_trees-1` | `forest_train` | Independent bootstrap + tree | **Best.** `schedule(dynamic)`. Per-tree RNG (`seed + t`), private node arena. Never share `rand()`. |
| Predict rows | `forest_predict` | Rows independent; trees read-only | **Strong.** `schedule(static)`. Same pattern for accuracy / OOB. |
| Votes over trees, one row | `forest_predict_one` | Trees independent | **Weaker.** Prefer the row loop. Array reduction or thread-local votes. |
| Split search over `mtry` features | `find_best_split` | Features independent given a node | **Usually not worth it.** Nested under tree-level parallelism saturates cores; high fork/join vs cheap Gini. Private pair buffers. |
| `qsort` one feature | `find_best_split` | Parallel sort | Almost never useful at node scale. |

Do not nest split-search OpenMP under tree-level `parallel for` without limiting active levels (`omp_set_max_active_levels(1)`).
