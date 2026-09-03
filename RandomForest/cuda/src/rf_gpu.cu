#include "rf_cuda.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CLASSES_STACK 64

#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess)                                            \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                              \
            return -1;                                                       \
        }                                                                    \
    } while (0)

#define CUDA_CHECK_CLEANUP(call)                                             \
    do                                                                       \
    {                                                                        \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess)                                            \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                              \
            goto cleanup;                                                    \
        }                                                                    \
    } while (0)

int cuda_device_available(void)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}

const char *cuda_device_name(void)
{
    static char name[256] = "No CUDA Device";
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device) == cudaSuccess)
        {
            strncpy(name, prop.name, sizeof(name) - 1);
            name[sizeof(name) - 1] = '\0';
        }
    }
    return name;
}

/*
 * CUDA kernel for Random Forest prediction.
 * Each CUDA thread evaluates one sample across all trees in the forest.
 */
__global__ void rf_predict_kernel(const Node *__restrict__ nodes,
                                  const int *__restrict__ tree_roots,
                                  int n_trees,
                                  const double *__restrict__ x,
                                  int n_samples,
                                  int n_features,
                                  int n_classes,
                                  int *__restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples)
        return;

    const double *sample_x = x + (size_t)idx * (size_t)n_features;
    int votes[MAX_CLASSES_STACK];
    for (int c = 0; c < n_classes && c < MAX_CLASSES_STACK; c++)
        votes[c] = 0;

    for (int t = 0; t < n_trees; t++)
    {
        int root_offset = tree_roots[t];
        int curr = 0;
        while (1)
        {
            const Node *nd = &nodes[root_offset + curr];
            if (nd->feature < 0 || nd->feature >= n_features)
            {
                int cls = nd->pred_class;
                if (cls >= 0 && cls < n_classes && cls < MAX_CLASSES_STACK)
                    votes[cls]++;
                break;
            }
            double v = sample_x[nd->feature];
            int next = (v <= nd->threshold) ? nd->left : nd->right;
            if (next < 0)
            {
                int cls = nd->pred_class;
                if (cls >= 0 && cls < n_classes && cls < MAX_CLASSES_STACK)
                    votes[cls]++;
                break;
            }
            curr = next;
        }
    }

    int best_c = 0;
    int max_v = -1;
    for (int c = 0; c < n_classes && c < MAX_CLASSES_STACK; c++)
    {
        if (votes[c] > max_v)
        {
            max_v = votes[c];
            best_c = c;
        }
    }
    out[idx] = best_c;
}

/* Fallback kernel for n_classes > MAX_CLASSES_STACK using global vote memory */
__global__ void rf_predict_kernel_large_classes(const Node *__restrict__ nodes,
                                                const int *__restrict__ tree_roots,
                                                int n_trees,
                                                const double *__restrict__ x,
                                                int n_samples,
                                                int n_features,
                                                int n_classes,
                                                int *__restrict__ global_votes,
                                                int *__restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples)
        return;

    const double *sample_x = x + (size_t)idx * (size_t)n_features;
    int *my_votes = global_votes + (size_t)idx * (size_t)n_classes;
    for (int c = 0; c < n_classes; c++)
        my_votes[c] = 0;

    for (int t = 0; t < n_trees; t++)
    {
        int root_offset = tree_roots[t];
        int curr = 0;
        while (1)
        {
            const Node *nd = &nodes[root_offset + curr];
            if (nd->feature < 0 || nd->feature >= n_features)
            {
                int cls = nd->pred_class;
                if (cls >= 0 && cls < n_classes)
                    my_votes[cls]++;
                break;
            }
            double v = sample_x[nd->feature];
            int next = (v <= nd->threshold) ? nd->left : nd->right;
            if (next < 0)
            {
                int cls = nd->pred_class;
                if (cls >= 0 && cls < n_classes)
                    my_votes[cls]++;
                break;
            }
            curr = next;
        }
    }

    int best_c = 0;
    int max_v = -1;
    for (int c = 0; c < n_classes; c++)
    {
        if (my_votes[c] > max_v)
        {
            max_v = my_votes[c];
            best_c = c;
        }
    }
    out[idx] = best_c;
}

int forest_predict_gpu(const Forest *f, const Dataset *ds, int *out,
                       double *h2d_ms, double *kernel_ms, double *d2h_ms)
{
    if (!f || !ds || !out || ds->n_samples <= 0 || f->n_trees <= 0)
        return -1;

    if (!cuda_device_available())
        return -1;

    /* 1. Flatten all trees into contiguous host arrays */
    int total_nodes = 0;
    for (int t = 0; t < f->n_trees; t++)
        total_nodes += f->trees[t].n_nodes;

    Node *flat_nodes = (Node *)malloc((size_t)total_nodes * sizeof(Node));
    int *tree_roots = (int *)malloc((size_t)f->n_trees * sizeof(int));
    if (!flat_nodes || !tree_roots)
    {
        free(flat_nodes);
        free(tree_roots);
        return -1;
    }

    int offset = 0;
    for (int t = 0; t < f->n_trees; t++)
    {
        tree_roots[t] = offset;
        memcpy(flat_nodes + offset, f->trees[t].nodes,
               (size_t)f->trees[t].n_nodes * sizeof(Node));
        offset += f->trees[t].n_nodes;
    }

    /* 2. Allocate device memory */
    Node *d_nodes = NULL;
    int *d_tree_roots = NULL;
    double *d_x = NULL;
    int *d_out = NULL;
    int *d_global_votes = NULL;

    size_t nodes_bytes = (size_t)total_nodes * sizeof(Node);
    size_t roots_bytes = (size_t)f->n_trees * sizeof(int);
    size_t x_bytes = (size_t)ds->n_samples * (size_t)ds->n_features * sizeof(double);
    size_t out_bytes = (size_t)ds->n_samples * sizeof(int);

    int status = 0;
    cudaEvent_t ev_h2d_start = NULL, ev_h2d_end = NULL;
    cudaEvent_t ev_kern_start = NULL, ev_kern_end = NULL;
    cudaEvent_t ev_d2h_start = NULL, ev_d2h_end = NULL;

    CUDA_CHECK_CLEANUP(cudaEventCreate(&ev_h2d_start));
    CUDA_CHECK_CLEANUP(cudaEventCreate(&ev_h2d_end));
    CUDA_CHECK_CLEANUP(cudaEventCreate(&ev_kern_start));
    CUDA_CHECK_CLEANUP(cudaEventCreate(&ev_kern_end));
    CUDA_CHECK_CLEANUP(cudaEventCreate(&ev_d2h_start));
    CUDA_CHECK_CLEANUP(cudaEventCreate(&ev_d2h_end));

    CUDA_CHECK_CLEANUP(cudaMalloc((void **)&d_nodes, nodes_bytes));
    CUDA_CHECK_CLEANUP(cudaMalloc((void **)&d_tree_roots, roots_bytes));
    CUDA_CHECK_CLEANUP(cudaMalloc((void **)&d_x, x_bytes));
    CUDA_CHECK_CLEANUP(cudaMalloc((void **)&d_out, out_bytes));

    if (ds->n_classes > MAX_CLASSES_STACK)
    {
        size_t votes_bytes = (size_t)ds->n_samples * (size_t)ds->n_classes * sizeof(int);
        CUDA_CHECK_CLEANUP(cudaMalloc((void **)&d_global_votes, votes_bytes));
    }

    /* 3. Host to Device transfer */
    CUDA_CHECK_CLEANUP(cudaEventRecord(ev_h2d_start, 0));
    CUDA_CHECK_CLEANUP(cudaMemcpyAsync(d_nodes, flat_nodes, nodes_bytes,
                                       cudaMemcpyHostToDevice, 0));
    CUDA_CHECK_CLEANUP(cudaMemcpyAsync(d_tree_roots, tree_roots, roots_bytes,
                                       cudaMemcpyHostToDevice, 0));
    CUDA_CHECK_CLEANUP(cudaMemcpyAsync(d_x, ds->x, x_bytes,
                                       cudaMemcpyHostToDevice, 0));
    CUDA_CHECK_CLEANUP(cudaEventRecord(ev_h2d_end, 0));

    /* 4. Launch kernel */
    {
        int block_size = 256;
        int grid_size = (ds->n_samples + block_size - 1) / block_size;

        CUDA_CHECK_CLEANUP(cudaEventRecord(ev_kern_start, 0));
        if (ds->n_classes <= MAX_CLASSES_STACK)
        {
            rf_predict_kernel<<<grid_size, block_size, 0, 0>>>(
                d_nodes, d_tree_roots, f->n_trees, d_x, ds->n_samples,
                ds->n_features, ds->n_classes, d_out);
        }
        else
        {
            rf_predict_kernel_large_classes<<<grid_size, block_size, 0, 0>>>(
                d_nodes, d_tree_roots, f->n_trees, d_x, ds->n_samples,
                ds->n_features, ds->n_classes, d_global_votes, d_out);
        }
        CUDA_CHECK_CLEANUP(cudaGetLastError());
        CUDA_CHECK_CLEANUP(cudaEventRecord(ev_kern_end, 0));
    }

    /* 5. Device to Host transfer */
    CUDA_CHECK_CLEANUP(cudaEventRecord(ev_d2h_start, 0));
    CUDA_CHECK_CLEANUP(cudaMemcpyAsync(out, d_out, out_bytes,
                                       cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK_CLEANUP(cudaEventRecord(ev_d2h_end, 0));
    CUDA_CHECK_CLEANUP(cudaEventSynchronize(ev_d2h_end));

    /* 6. Calculate breakdown times */
    {
        float t_h2d = 0.0f, t_kern = 0.0f, t_d2h = 0.0f;
        cudaEventElapsedTime(&t_h2d, ev_h2d_start, ev_h2d_end);
        cudaEventElapsedTime(&t_kern, ev_kern_start, ev_kern_end);
        cudaEventElapsedTime(&t_d2h, ev_d2h_start, ev_d2h_end);
        if (h2d_ms)
            *h2d_ms = (double)t_h2d;
        if (kernel_ms)
            *kernel_ms = (double)t_kern;
        if (d2h_ms)
            *d2h_ms = (double)t_d2h;
    }

    goto done;

cleanup:
    status = -1;

done:
    if (ev_h2d_start)
        cudaEventDestroy(ev_h2d_start);
    if (ev_h2d_end)
        cudaEventDestroy(ev_h2d_end);
    if (ev_kern_start)
        cudaEventDestroy(ev_kern_start);
    if (ev_kern_end)
        cudaEventDestroy(ev_kern_end);
    if (ev_d2h_start)
        cudaEventDestroy(ev_d2h_start);
    if (ev_d2h_end)
        cudaEventDestroy(ev_d2h_end);

    if (d_nodes)
        cudaFree(d_nodes);
    if (d_tree_roots)
        cudaFree(d_tree_roots);
    if (d_x)
        cudaFree(d_x);
    if (d_out)
        cudaFree(d_out);
    if (d_global_votes)
        cudaFree(d_global_votes);

    free(flat_nodes);
    free(tree_roots);

    return status;
}
