#include "rf_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_now(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "Usage: %s [--data PATH] [--target COL] [--trees N] [--max-depth D] "
            "[--min-samples M] [--mtry K] [--test-frac F] [--seed S] "
            "[--dot FILE] [--dot-tree N]\n"
            "  --target    column name or 0-based index (default: last column)\n"
            "  --dot FILE  write Graphviz DOT of one tree (default: tree 0)\n"
            "  --dot-tree  which tree to export (0-based)\n",
            argv0);
}

int main(int argc, char **argv)
{
    if (cuda_device_available())
    {
        cudaFree(0); /* Initialize CUDA context upfront */
    }
    const char *path = "data/iris.csv";
    const char *target = NULL;
    ForestParams p;
    memset(&p, 0, sizeof(p));
    p.n_trees = 100;
    p.max_depth = 8;
    p.min_samples_split = 2;
    p.mtry = 0;
    p.seed = 42;
    double test_frac = 0.2;
    const char *dot_path = NULL;
    int dot_tree = 0;
    int cpu_baseline = 1;

    for (int i = 1; i < argc; i++)
    {
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
        else if (!strcmp(argv[i], "--dot") && i + 1 < argc)
            dot_path = argv[++i];
        else if (!strcmp(argv[i], "--dot-tree") && i + 1 < argc)
            dot_tree = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-cpu-baseline"))
            cpu_baseline = 0;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
        {
            usage(argv[0]);
            return 0;
        }
        else
        {
            usage(argv[0]);
            return 1;
        }
    }

    Dataset full = {0};
    if (dataset_load_csv(path, &full, target) != 0)
    {
        fprintf(stderr, "failed to load CSV: %s\n", path);
        return 1;
    }

    Dataset train = {0}, test = {0};
    dataset_split_holdout(&full, &train, &test, test_frac, p.seed);

    int has_gpu = cuda_device_available();
    const char *gpu_name = has_gpu ? cuda_device_name() : "none";

    Forest forest = {0};
    double train_gpu_s = 0.0;
    double h2d_tr_ms = 0.0, kern_tr_ms = 0.0, d2h_tr_ms = 0.0;
    int train_gpu_ok = 0;

    if (has_gpu)
    {
        double t0_tr = wall_now();
        train_gpu_ok = (forest_train_gpu(&forest, &train, &p, &h2d_tr_ms, &kern_tr_ms, &d2h_tr_ms) == 0);
        train_gpu_s = wall_now() - t0_tr;
    }

    /* CPU training baseline for comparison */
    double train_cpu_s = 0.0;
    if (!has_gpu || !train_gpu_ok)
    {
        double t0 = wall_now();
        if (forest_train(&forest, &train, &p) != 0)
        {
            fprintf(stderr, "training failed\n");
            dataset_free(&full);
            dataset_free(&train);
            dataset_free(&test);
            return 1;
        }
        train_cpu_s = wall_now() - t0;
    }
    else if (cpu_baseline)
    {
        Forest forest_cpu = {0};
        double t0 = wall_now();
        if (forest_train(&forest_cpu, &train, &p) == 0)
        {
            train_cpu_s = wall_now() - t0;
            forest_free(&forest_cpu);
        }
    }

    double active_train_s = (has_gpu && train_gpu_ok) ? train_gpu_s : train_cpu_s;

    /* CPU evaluation for baseline comparison and verification */
    double t0 = wall_now();
    int *cpu_pred_tr = (int *)malloc((size_t)train.n_samples * sizeof(int));
    int *cpu_pred_te = (int *)malloc((size_t)test.n_samples * sizeof(int));
    forest_predict(&forest, &train, cpu_pred_tr);
    forest_predict(&forest, &test, cpu_pred_te);
    double predict_cpu_s = wall_now() - t0;

    int cpu_tr_ok = 0, cpu_te_ok = 0;
    for (int i = 0; i < train.n_samples; i++)
        if (cpu_pred_tr[i] == train.y[i])
            cpu_tr_ok++;
    for (int i = 0; i < test.n_samples; i++)
        if (cpu_pred_te[i] == test.y[i])
            cpu_te_ok++;
    double acc_tr_cpu = (train.n_samples > 0) ? (double)cpu_tr_ok / train.n_samples : 0.0;
    double acc_te_cpu = (test.n_samples > 0) ? (double)cpu_te_ok / test.n_samples : 0.0;

    /* GPU evaluation */
    int *gpu_pred_tr = (int *)malloc((size_t)train.n_samples * sizeof(int));
    int *gpu_pred_te = (int *)malloc((size_t)test.n_samples * sizeof(int));
    double h2d_te = 0.0, kern_te = 0.0, d2h_te = 0.0;
    double h2d_tr_pred = 0.0, kern_tr_pred = 0.0, d2h_tr_pred = 0.0;
    double predict_gpu_s = 0.0;
    double acc_tr_gpu = 0.0, acc_te_gpu = 0.0;
    int mismatches_te = 0;

    if (has_gpu)
    {
        double t0_gpu = wall_now();
        int ok1 = forest_predict_gpu(&forest, &train, gpu_pred_tr, &h2d_tr_pred, &kern_tr_pred, &d2h_tr_pred);
        int ok2 = forest_predict_gpu(&forest, &test, gpu_pred_te, &h2d_te, &kern_te, &d2h_te);
        predict_gpu_s = wall_now() - t0_gpu;

        if (ok1 == 0 && ok2 == 0)
        {
            int gpu_tr_ok = 0, gpu_te_ok = 0;
            for (int i = 0; i < train.n_samples; i++)
                if (gpu_pred_tr[i] == train.y[i])
                    gpu_tr_ok++;
            for (int i = 0; i < test.n_samples; i++)
            {
                if (gpu_pred_te[i] == test.y[i])
                    gpu_te_ok++;
                if (gpu_pred_te[i] != cpu_pred_te[i])
                    mismatches_te++;
            }
            acc_tr_gpu = (train.n_samples > 0) ? (double)gpu_tr_ok / train.n_samples : 0.0;
            acc_te_gpu = (test.n_samples > 0) ? (double)gpu_te_ok / test.n_samples : 0.0;
        }
        else
        {
            fprintf(stderr, "warning: GPU prediction failed, falling back to CPU\n");
            acc_tr_gpu = acc_tr_cpu;
            acc_te_gpu = acc_te_cpu;
            predict_gpu_s = predict_cpu_s;
        }
    }
    else
    {
        acc_tr_gpu = acc_tr_cpu;
        acc_te_gpu = acc_te_cpu;
        predict_gpu_s = predict_cpu_s;
    }

    double active_acc_tr = has_gpu ? acc_tr_gpu : acc_tr_cpu;
    double active_acc_te = has_gpu ? acc_te_gpu : acc_te_cpu;
    double active_predict_s = has_gpu ? predict_gpu_s : predict_cpu_s;

    printf("samples=%d features=%d classes=%d target=%s\n",
           full.n_samples, full.n_features, full.n_classes,
           full.target_name ? full.target_name : "?");
    printf("train=%d test=%d trees=%d mtry=%d max_depth=%d min_samples=%d seed=%u gpu=\"%s\"\n",
           train.n_samples, test.n_samples, forest.n_trees, forest.mtry,
           forest.max_depth, forest.min_samples_split, p.seed, gpu_name);
    printf("train_accuracy=%.4f\n", active_acc_tr);
    printf("test_accuracy=%.4f\n", active_acc_te);
    printf("train_wall_s=%.6f predict_wall_s=%.6f wall_s=%.6f\n",
           active_train_s, active_predict_s, active_train_s + active_predict_s);

    if (has_gpu && train_gpu_ok)
    {
        double speedup_tr = (train_gpu_s > 0.0 && train_cpu_s > 0.0) ? (train_cpu_s / train_gpu_s) : 0.0;
        printf("train_cpu_s=%.6f train_gpu_s=%.6f speedup_train=%.2fx\n",
               train_cpu_s, train_gpu_s, speedup_tr);
        printf("gpu_train_breakdown: h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f total_train_ms=%.3f\n",
               h2d_tr_ms, kern_tr_ms, d2h_tr_ms, h2d_tr_ms + kern_tr_ms + d2h_tr_ms);
    }

    if (has_gpu)
    {
        double speedup = (predict_gpu_s > 0.0) ? (predict_cpu_s / predict_gpu_s) : 0.0;
        double speedup_kern = (kern_te > 0.0) ? ((predict_cpu_s * 1000.0) / (kern_tr_pred + kern_te)) : 0.0;
        printf("predict_cpu_s=%.6f predict_gpu_s=%.6f speedup_predict=%.2fx (kernel_only=%.2fx)\n",
               predict_cpu_s, predict_gpu_s, speedup, speedup_kern);
        printf("gpu_test_breakdown: h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f total_gpu_ms=%.3f\n",
               h2d_te, kern_te, d2h_te, h2d_te + kern_te + d2h_te);
        printf("gpu_cpu_test_mismatches=%d / %d\n", mismatches_te, test.n_samples);
    }

    if (dot_path)
    {
        if (forest_write_dot(&forest, dot_tree, dot_path) != 0)
        {
            fprintf(stderr, "failed to write DOT: %s (tree %d)\n", dot_path, dot_tree);
            forest_free(&forest);
            dataset_free(&full);
            dataset_free(&train);
            dataset_free(&test);
            free(cpu_pred_tr);
            free(cpu_pred_te);
            free(gpu_pred_tr);
            free(gpu_pred_te);
            return 1;
        }
        printf("wrote_dot=%s tree=%d nodes=%d\n", dot_path, dot_tree,
               forest.trees[dot_tree].n_nodes);
    }

    forest_free(&forest);
    dataset_free(&full);
    dataset_free(&train);
    dataset_free(&test);
    free(cpu_pred_tr);
    free(cpu_pred_te);
    free(gpu_pred_tr);
    free(gpu_pred_te);
    return 0;
}
