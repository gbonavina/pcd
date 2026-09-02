#define _POSIX_C_SOURCE 200809L

#include "rf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s [--data PATH] [--target COL] [--trees N] [--max-depth D] "
            "[--min-samples M] [--mtry K] [--test-frac F] [--seed S]\n"
            "  --target  column name or 0-based index (default: last column)\n",
            argv0);
}

int main(int argc, char **argv) {
    const char *path = "data/iris.csv";
    const char *target = NULL;
    ForestParams p = {
        .n_trees = 100,
        .max_depth = 8,
        .min_samples_split = 2,
        .mtry = 0,
        .seed = 42,
    };
    double test_frac = 0.2;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i + 1 < argc)
            path = argv[++i];
        else if (!strcmp(argv[i], "--target") && i + 1 < argc)
            target = argv[++i];
        else if (!strcmp(argv[i], "--trees") && i + 1 < argc)
            p.n_trees = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-depth") && i + 1 < argc)
            p.max_depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-samples") && i + 1 < argc)
            p.min_samples_split = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mtry") && i + 1 < argc)
            p.mtry = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--test-frac") && i + 1 < argc)
            test_frac = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            p.seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    Dataset full = {0};
    if (dataset_load_csv(path, &full, target) != 0) {
        fprintf(stderr, "failed to load CSV: %s\n", path);
        return 1;
    }

    Dataset train = {0}, test = {0};
    dataset_split_holdout(&full, &train, &test, test_frac, p.seed);

    Forest forest = {0};
    double t0 = wall_now();
    if (forest_train(&forest, &train, &p) != 0) {
        fprintf(stderr, "training failed\n");
        dataset_free(&full);
        dataset_free(&train);
        dataset_free(&test);
        return 1;
    }
    double train_s = wall_now() - t0;

    t0 = wall_now();
    double acc_tr = forest_accuracy(&forest, &train);
    double acc_te = forest_accuracy(&forest, &test);
    double predict_s = wall_now() - t0;

    printf("samples=%d features=%d classes=%d target=%s\n",
           full.n_samples, full.n_features, full.n_classes,
           full.target_name ? full.target_name : "?");
    printf("train=%d test=%d trees=%d mtry=%d max_depth=%d min_samples=%d seed=%u\n",
           train.n_samples, test.n_samples, forest.n_trees, forest.mtry,
           forest.max_depth, forest.min_samples_split, p.seed);
    printf("train_accuracy=%.4f\n", acc_tr);
    printf("test_accuracy=%.4f\n", acc_te);
    printf("train_wall_s=%.6f predict_wall_s=%.6f wall_s=%.6f\n",
           train_s, predict_s, train_s + predict_s);

    forest_free(&forest);
    dataset_free(&full);
    dataset_free(&train);
    dataset_free(&test);
    return 0;
}
