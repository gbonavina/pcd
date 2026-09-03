#ifndef RF_H
#define RF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double *x;      /* row-major: n_samples * n_features */
    int *y;
    int n_samples;
    int n_features;
    int n_classes;
    char *target_name; /* malloc'd; NULL if unnamed */
} Dataset;

typedef struct {
    int feature;    /* -1 if leaf */
    double threshold;
    int left;       /* node index, -1 if none */
    int right;
    int pred_class; /* majority class (leaves; unused on internals) */
} Node;

typedef struct {
    Node *nodes;
    int n_nodes;
    int cap_nodes;
    int n_classes;
} Tree;

typedef struct {
    Tree *trees;
    int n_trees;
    int n_features;
    int n_classes;
    int mtry;
    int max_depth;
    int min_samples_split;
} Forest;

typedef struct {
    int n_trees;
    int max_depth;
    int min_samples_split;
    int mtry;       /* 0 => sqrt(n_features) */
    uint32_t seed;
} ForestParams;

/* target: column name, 0-based index as text, or NULL/"" for last column.
 * A column named "id" is dropped (same as the sklearn notebook). */
int dataset_load_csv(const char *path, Dataset *ds, const char *target);
void dataset_free(Dataset *ds);

int forest_train(Forest *f, const Dataset *ds, const ForestParams *p);
int forest_predict_one(const Forest *f, const double *x);
void forest_predict(const Forest *f, const Dataset *ds, int *out);
double forest_accuracy(const Forest *f, const Dataset *ds);
void forest_free(Forest *f);
/* Graphviz DOT of trees[tree_index]. Returns 0 on success. */
int forest_write_dot(const Forest *f, int tree_index, const char *path);

void dataset_split_holdout(const Dataset *full, Dataset *train, Dataset *test,
                           double test_frac, uint32_t seed);

/* ---------- CUDA prediction (defined in rf_gpu.cu) ---------- */
/* Returns 0 on success. Non-zero => GPU unavailable/failed, use CPU fallback.
 * Breakdown times in ms (may be NULL). */
int forest_predict_gpu(const Forest *f, const Dataset *ds, int *out,
                       double *h2d_ms, double *kernel_ms, double *d2h_ms);
int cuda_device_available(void);
const char *cuda_device_name(void);

#ifdef __cplusplus
}
#endif

#endif
