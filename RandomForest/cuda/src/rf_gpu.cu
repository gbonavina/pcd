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

/*
 * ============================================================================
 * CUDA Random Forest Training Implementation
 * ============================================================================
 */

#define MAX_CLASSES_TRAIN 64
#define MAX_STACK_TRAIN 128
#define MAX_PAIRS_SHARED 1024

typedef struct
{
    double v;
    int y;
} DevPair;

typedef struct
{
    int node_id;
    int start;
    int count;
    int depth;
} DevTask;

__device__ inline uint32_t rf_dev_rng_u32(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x ? x : 0x9E3779B9u;
    return *s;
}

__device__ inline int rf_dev_rng_int(uint32_t *s, int n)
{
    if (n <= 1)
        return 0;
    return (int)(rf_dev_rng_u32(s) % (uint32_t)n);
}

__device__ void bitonic_sort_pairs_dev_shared(DevPair *pairs, int n)
{
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int power2 = 1;
    while (power2 < n)
        power2 <<= 1;

    for (int k = 2; k <= power2; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            for (int i = tid; i < power2 / 2; i += bdim)
            {
                int l = 2 * (i - (i % j)) + (i % j);
                int r = l + j;
                if (r < n)
                {
                    bool dir = ((l & k) == 0);
                    if ((pairs[l].v > pairs[r].v) == dir)
                    {
                        DevPair tmp = pairs[l];
                        pairs[l] = pairs[r];
                        pairs[r] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__device__ void bitonic_sort_pairs_dev_global(DevPair *pairs, int n)
{
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int power2 = 1;
    while (power2 < n)
        power2 <<= 1;

    for (int k = 2; k <= power2; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            for (int i = tid; i < power2 / 2; i += bdim)
            {
                int l = 2 * (i - (i % j)) + (i % j);
                int r = l + j;
                if (r < n)
                {
                    bool dir = ((l & k) == 0);
                    if ((pairs[l].v > pairs[r].v) == dir)
                    {
                        DevPair tmp = pairs[l];
                        pairs[l] = pairs[r];
                        pairs[r] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__global__ void rf_train_forest_kernel_small(
    const double *__restrict__ x,
    const int *__restrict__ y,
    int n_samples,
    int n_features,
    int n_classes,
    int max_depth,
    int min_samples_split,
    int mtry,
    uint32_t base_seed,
    int tree_offset,
    Node *__restrict__ out_nodes,
    int *__restrict__ out_n_nodes,
    int max_nodes_per_tree)
{
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int local_t = blockIdx.x;
    int global_t = tree_offset + local_t;

    __shared__ int s_samples[MAX_PAIRS_SHARED];
    __shared__ int s_temp[MAX_PAIRS_SHARED];
    __shared__ DevPair s_pairs[MAX_PAIRS_SHARED];
    __shared__ int s_counts[MAX_CLASSES_TRAIN];
    __shared__ int s_left_c[MAX_CLASSES_TRAIN];
    __shared__ int s_right_c[MAX_CLASSES_TRAIN];
    __shared__ int s_feats[256];

    __shared__ DevTask s_stack[MAX_STACK_TRAIN];
    __shared__ int s_stail;
    __shared__ int s_node_count;
    __shared__ uint32_t s_rng;

    __shared__ double s_best_gini;
    __shared__ double s_best_thresh;
    __shared__ int s_best_feat;
    __shared__ int s_best_ok;

    __shared__ DevTask s_cur_task;
    __shared__ bool s_is_leaf;
    __shared__ int s_majority;

    if (tid == 0)
    {
        s_rng = base_seed + (uint32_t)global_t * 0x9E3779B9u + 1u;
        s_node_count = 1;
        s_stail = 0;

        for (int i = 0; i < n_samples; i++)
        {
            s_samples[i] = rf_dev_rng_int(&s_rng, n_samples);
        }

        s_stack[0].node_id = 0;
        s_stack[0].start = 0;
        s_stack[0].count = n_samples;
        s_stack[0].depth = 0;
        s_stail = 1;
    }
    __syncthreads();

    while (s_stail > 0)
    {
        DevTask task;
        if (tid == 0)
        {
            task = s_stack[--s_stail];
            s_cur_task = task;
        }
        __syncthreads();
        task = s_cur_task;

        /* 1. Class counts */
        for (int c = tid; c < n_classes && c < MAX_CLASSES_TRAIN; c += bdim)
        {
            s_counts[c] = 0;
        }
        __syncthreads();

        for (int i = tid; i < task.count; i += bdim)
        {
            int row = s_samples[task.start + i];
            int cls = y[row];
            if (cls >= 0 && cls < MAX_CLASSES_TRAIN)
                atomicAdd(&s_counts[cls], 1);
        }
        __syncthreads();

        /* 2. Majority class and purity check */
        if (tid == 0)
        {
            int max_c = -1;
            int maj = 0;
            for (int c = 0; c < n_classes && c < MAX_CLASSES_TRAIN; c++)
            {
                if (s_counts[c] > max_c)
                {
                    max_c = s_counts[c];
                    maj = c;
                }
            }
            s_majority = maj;
            s_is_leaf = (max_c == task.count || task.count < min_samples_split || task.depth >= max_depth);
        }
        __syncthreads();

        if (s_is_leaf)
        {
            if (tid == 0)
            {
                Node *nd = &out_nodes[(size_t)local_t * (size_t)max_nodes_per_tree + (size_t)task.node_id];
                nd->feature = -1;
                nd->threshold = 0.0;
                nd->left = -1;
                nd->right = -1;
                nd->pred_class = s_majority;
            }
            __syncthreads();
            continue;
        }

        /* 3. Pick mtry candidate features */
        if (tid == 0)
        {
            s_best_gini = 1e300;
            s_best_feat = -1;
            s_best_thresh = 0.0;
            s_best_ok = 0;

            int all_feats[256];
            int max_f = n_features < 256 ? n_features : 256;
            for (int i = 0; i < max_f; i++)
                all_feats[i] = i;
            for (int i = 0; i < mtry && i < max_f; i++)
            {
                int j = i + rf_dev_rng_int(&s_rng, max_f - i);
                int tmp = all_feats[i];
                all_feats[i] = all_feats[j];
                all_feats[j] = tmp;
                s_feats[i] = all_feats[i];
            }
        }
        __syncthreads();

        /* 4. Evaluate each candidate feature */
        for (int fi = 0; fi < mtry; fi++)
        {
            int f = s_feats[fi];

            for (int i = tid; i < task.count; i += bdim)
            {
                int row = s_samples[task.start + i];
                s_pairs[i].v = x[(size_t)row * (size_t)n_features + (size_t)f];
                s_pairs[i].y = y[row];
            }
            __syncthreads();

            bitonic_sort_pairs_dev_shared(s_pairs, task.count);
            __syncthreads();

            if (tid == 0)
            {
                for (int c = 0; c < n_classes && c < MAX_CLASSES_TRAIN; c++)
                {
                    s_left_c[c] = 0;
                    s_right_c[c] = s_counts[c];
                }
                int n_left = 0;
                int n_right = task.count;

                for (int i = 0; i < task.count - 1; i++)
                {
                    int cls = s_pairs[i].y;
                    if (cls >= 0 && cls < MAX_CLASSES_TRAIN)
                    {
                        s_left_c[cls]++;
                        s_right_c[cls]--;
                    }
                    n_left++;
                    n_right--;

                    double v0 = s_pairs[i].v;
                    double v1 = s_pairs[i + 1].v;
                    if (v0 == v1)
                        continue;

                    double gl = 0.0, gr = 0.0;
                    for (int c = 0; c < n_classes && c < MAX_CLASSES_TRAIN; c++)
                    {
                        double pl = (double)s_left_c[c] / (double)n_left;
                        double pr = (double)s_right_c[c] / (double)n_right;
                        gl += pl * pl;
                        gr += pr * pr;
                    }
                    gl = 1.0 - gl;
                    gr = 1.0 - gr;
                    double g = ((double)n_left * gl + (double)n_right * gr) / (double)task.count;

                    if (g < s_best_gini)
                    {
                        s_best_gini = g;
                        s_best_feat = f;
                        s_best_thresh = 0.5 * (v0 + v1);
                        s_best_ok = 1;
                    }
                }
            }
            __syncthreads();
        }

        if (tid == 0 && !s_best_ok)
        {
            Node *nd = &out_nodes[(size_t)local_t * (size_t)max_nodes_per_tree + (size_t)task.node_id];
            nd->feature = -1;
            nd->threshold = 0.0;
            nd->left = -1;
            nd->right = -1;
            nd->pred_class = s_majority;
        }
        __syncthreads();

        if (s_best_ok)
        {
            if (tid == 0)
            {
                int left_idx = 0;
                double thresh = s_best_thresh;
                int f = s_best_feat;

                for (int i = 0; i < task.count; i++)
                {
                    int row = s_samples[task.start + i];
                    double val = x[(size_t)row * (size_t)n_features + (size_t)f];
                    if (val <= thresh)
                    {
                        s_temp[left_idx++] = row;
                    }
                }
                int n_left = left_idx;
                for (int i = 0; i < task.count; i++)
                {
                    int row = s_samples[task.start + i];
                    double val = x[(size_t)row * (size_t)n_features + (size_t)f];
                    if (val > thresh)
                    {
                        s_temp[left_idx++] = row;
                    }
                }
                int n_right = task.count - n_left;

                if (n_left == 0 || n_right == 0 ||
                    s_node_count + 2 > max_nodes_per_tree ||
                    s_stail + 2 > MAX_STACK_TRAIN)
                {
                    Node *nd = &out_nodes[(size_t)local_t * (size_t)max_nodes_per_tree + (size_t)task.node_id];
                    nd->feature = -1;
                    nd->threshold = 0.0;
                    nd->left = -1;
                    nd->right = -1;
                    nd->pred_class = s_majority;
                }
                else
                {
                    for (int i = 0; i < task.count; i++)
                    {
                        s_samples[task.start + i] = s_temp[i];
                    }

                    int left_id = s_node_count++;
                    int right_id = s_node_count++;

                    Node *nd = &out_nodes[(size_t)local_t * (size_t)max_nodes_per_tree + (size_t)task.node_id];
                    nd->feature = s_best_feat;
                    nd->threshold = s_best_thresh;
                    nd->left = left_id;
                    nd->right = right_id;
                    nd->pred_class = s_majority;

                    s_stack[s_stail++] = {right_id, task.start + n_left, n_right, task.depth + 1};
                    s_stack[s_stail++] = {left_id, task.start, n_left, task.depth + 1};
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0)
    {
        out_n_nodes[local_t] = s_node_count;
    }
}

__global__ void rf_train_forest_kernel(
    const double *__restrict__ x,
    const int *__restrict__ y,
    int n_samples,
    int n_features,
    int n_classes,
    int max_depth,
    int min_samples_split,
    int mtry,
    uint32_t base_seed,
    int tree_offset,
    int *__restrict__ global_samples,
    int *__restrict__ global_temp,
    DevPair *__restrict__ global_pairs,
    Node *__restrict__ out_nodes,
    int *__restrict__ out_n_nodes,
    int max_nodes_per_tree)
{
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int local_t = blockIdx.x;
    int global_t = tree_offset + local_t;

    int *my_samples = global_samples + (size_t)local_t * (size_t)n_samples;
    int *my_temp = global_temp + (size_t)local_t * (size_t)n_samples;
    DevPair *my_pairs = global_pairs + (size_t)local_t * (size_t)n_samples;

    __shared__ int s_counts[MAX_CLASSES_TRAIN];
    __shared__ int s_left_c[MAX_CLASSES_TRAIN];
    __shared__ int s_right_c[MAX_CLASSES_TRAIN];
    __shared__ int s_feats[256];
    __shared__ DevPair s_pairs[MAX_PAIRS_SHARED];

    __shared__ DevTask s_stack[MAX_STACK_TRAIN];
    __shared__ int s_stail;
    __shared__ int s_node_count;
    __shared__ uint32_t s_rng;

    __shared__ double s_best_gini;
    __shared__ double s_best_thresh;
    __shared__ int s_best_feat;
    __shared__ int s_best_ok;

    __shared__ DevTask s_cur_task;
    __shared__ bool s_is_leaf;
    __shared__ int s_majority;

    if (tid == 0)
    {
        s_rng = base_seed + (uint32_t)global_t * 0x9E3779B9u + 1u;
        s_node_count = 1;
        s_stail = 0;

        for (int i = 0; i < n_samples; i++)
        {
            my_samples[i] = rf_dev_rng_int(&s_rng, n_samples);
        }

        s_stack[0].node_id = 0;
        s_stack[0].start = 0;
        s_stack[0].count = n_samples;
        s_stack[0].depth = 0;
        s_stail = 1;
    }
    __syncthreads();

    while (s_stail > 0)
    {
        DevTask task;
        if (tid == 0)
        {
            task = s_stack[--s_stail];
            s_cur_task = task;
        }
        __syncthreads();
        task = s_cur_task;

        /* 1. Class counts */
        for (int c = tid; c < n_classes && c < MAX_CLASSES_TRAIN; c += bdim)
        {
            s_counts[c] = 0;
        }
        __syncthreads();

        for (int i = tid; i < task.count; i += bdim)
        {
            int row = my_samples[task.start + i];
            int cls = y[row];
            if (cls >= 0 && cls < MAX_CLASSES_TRAIN)
                atomicAdd(&s_counts[cls], 1);
        }
        __syncthreads();

        /* 2. Majority class and purity check */
        if (tid == 0)
        {
            int max_c = -1;
            int maj = 0;
            for (int c = 0; c < n_classes && c < MAX_CLASSES_TRAIN; c++)
            {
                if (s_counts[c] > max_c)
                {
                    max_c = s_counts[c];
                    maj = c;
                }
            }
            s_majority = maj;
            s_is_leaf = (max_c == task.count || task.count < min_samples_split || task.depth >= max_depth);
        }
        __syncthreads();

        if (s_is_leaf)
        {
            if (tid == 0)
            {
                Node *nd = &out_nodes[(size_t)local_t * (size_t)max_nodes_per_tree + (size_t)task.node_id];
                nd->feature = -1;
                nd->threshold = 0.0;
                nd->left = -1;
                nd->right = -1;
                nd->pred_class = s_majority;
            }
            __syncthreads();
            continue;
        }

        /* 3. Pick mtry candidate features */
        if (tid == 0)
        {
            s_best_gini = 1e300;
            s_best_feat = -1;
            s_best_thresh = 0.0;
            s_best_ok = 0;

            int all_feats[256];
            int max_f = n_features < 256 ? n_features : 256;
            for (int i = 0; i < max_f; i++)
                all_feats[i] = i;
            for (int i = 0; i < mtry && i < max_f; i++)
            {
                int j = i + rf_dev_rng_int(&s_rng, max_f - i);
                int tmp = all_feats[i];
                all_feats[i] = all_feats[j];
                all_feats[j] = tmp;
                s_feats[i] = all_feats[i];
            }
        }
        __syncthreads();

        /* 4. Evaluate each candidate feature */
        for (int fi = 0; fi < mtry; fi++)
        {
            int f = s_feats[fi];

            if (task.count <= MAX_PAIRS_SHARED)
            {
                for (int i = tid; i < task.count; i += bdim)
                {
                    int row = my_samples[task.start + i];
                    s_pairs[i].v = x[(size_t)row * (size_t)n_features + (size_t)f];
                    s_pairs[i].y = y[row];
                }
                __syncthreads();

                bitonic_sort_pairs_dev_shared(s_pairs, task.count);
                __syncthreads();

                if (tid == 0)
                {
                    for (int c = 0; c < n_classes && c < MAX_CLASSES_TRAIN; c++)
                    {
                        s_left_c[c] = 0;
                        s_right_c[c] = s_counts[c];
                    }
                    int n_left = 0;
                    int n_right = task.count;

                    for (int i = 0; i < task.count - 1; i++)
                    {
                        int cls = s_pairs[i].y;
                        if (cls >= 0 && cls < MAX_CLASSES_TRAIN)
                        {
                            s_left_c[cls]++;
                            s_right_c[cls]--;
                        }
                        n_left++;
                        n_right--;

                        double v0 = s_pairs[i].v;
                        double v1 = s_pairs[i + 1].v;
                        if (v0 == v1)
                            continue;

                        double gl = 0.0, gr = 0.0;
                        for (int c = 0; c < n_classes && c < MAX_CLASSES_TRAIN; c++)
                        {
                            double pl = (double)s_left_c[c] / (double)n_left;
                            double pr = (double)s_right_c[c] / (double)n_right;
                            gl += pl * pl;
                            gr += pr * pr;
                        }
                        gl = 1.0 - gl;
                        gr = 1.0 - gr;
                        double g = ((double)n_left * gl + (double)n_right * gr) / (double)task.count;

                        if (g < s_best_gini)
                        {
                            s_best_gini = g;
                            s_best_feat = f;
                            s_best_thresh = 0.5 * (v0 + v1);
                            s_best_ok = 1;
                        }
                    }
                }
                __syncthreads();
            }
            else
            {
                for (int i = tid; i < task.count; i += bdim)
                {
                    int row = my_samples[task.start + i];
                    my_pairs[task.start + i].v = x[(size_t)row * (size_t)n_features + (size_t)f];
                    my_pairs[task.start + i].y = y[row];
                }
                __syncthreads();

                bitonic_sort_pairs_dev_global(my_pairs + task.start, task.count);
                __syncthreads();

                if (tid == 0)
                {
                    for (int c = 0; c < n_classes && c < MAX_CLASSES_TRAIN; c++)
                    {
                        s_left_c[c] = 0;
                        s_right_c[c] = s_counts[c];
                    }
                    int n_left = 0;
                    int n_right = task.count;

                    for (int i = 0; i < task.count - 1; i++)
                    {
                        int cls = my_pairs[task.start + i].y;
                        if (cls >= 0 && cls < MAX_CLASSES_TRAIN)
                        {
                            s_left_c[cls]++;
                            s_right_c[cls]--;
                        }
                        n_left++;
                        n_right--;

                        double v0 = my_pairs[task.start + i].v;
                        double v1 = my_pairs[task.start + i + 1].v;
                        if (v0 == v1)
                            continue;

                        double gl = 0.0, gr = 0.0;
                        for (int c = 0; c < n_classes && c < MAX_CLASSES_TRAIN; c++)
                        {
                            double pl = (double)s_left_c[c] / (double)n_left;
                            double pr = (double)s_right_c[c] / (double)n_right;
                            gl += pl * pl;
                            gr += pr * pr;
                        }
                        gl = 1.0 - gl;
                        gr = 1.0 - gr;
                        double g = ((double)n_left * gl + (double)n_right * gr) / (double)task.count;

                        if (g < s_best_gini)
                        {
                            s_best_gini = g;
                            s_best_feat = f;
                            s_best_thresh = 0.5 * (v0 + v1);
                            s_best_ok = 1;
                        }
                    }
                }
                __syncthreads();
            }
        }

        if (tid == 0 && !s_best_ok)
        {
            Node *nd = &out_nodes[(size_t)local_t * (size_t)max_nodes_per_tree + (size_t)task.node_id];
            nd->feature = -1;
            nd->threshold = 0.0;
            nd->left = -1;
            nd->right = -1;
            nd->pred_class = s_majority;
        }
        __syncthreads();

        if (s_best_ok)
        {
            if (tid == 0)
            {
                int left_idx = 0;
                double thresh = s_best_thresh;
                int f = s_best_feat;

                for (int i = 0; i < task.count; i++)
                {
                    int row = my_samples[task.start + i];
                    double val = x[(size_t)row * (size_t)n_features + (size_t)f];
                    if (val <= thresh)
                    {
                        my_temp[left_idx++] = row;
                    }
                }
                int n_left = left_idx;
                for (int i = 0; i < task.count; i++)
                {
                    int row = my_samples[task.start + i];
                    double val = x[(size_t)row * (size_t)n_features + (size_t)f];
                    if (val > thresh)
                    {
                        my_temp[left_idx++] = row;
                    }
                }
                int n_right = task.count - n_left;

                if (n_left == 0 || n_right == 0 ||
                    s_node_count + 2 > max_nodes_per_tree ||
                    s_stail + 2 > MAX_STACK_TRAIN)
                {
                    Node *nd = &out_nodes[(size_t)local_t * (size_t)max_nodes_per_tree + (size_t)task.node_id];
                    nd->feature = -1;
                    nd->threshold = 0.0;
                    nd->left = -1;
                    nd->right = -1;
                    nd->pred_class = s_majority;
                }
                else
                {
                    for (int i = 0; i < task.count; i++)
                    {
                        my_samples[task.start + i] = my_temp[i];
                    }

                    int left_id = s_node_count++;
                    int right_id = s_node_count++;

                    Node *nd = &out_nodes[(size_t)local_t * (size_t)max_nodes_per_tree + (size_t)task.node_id];
                    nd->feature = s_best_feat;
                    nd->threshold = s_best_thresh;
                    nd->left = left_id;
                    nd->right = right_id;
                    nd->pred_class = s_majority;

                    s_stack[s_stail++] = {right_id, task.start + n_left, n_right, task.depth + 1};
                    s_stack[s_stail++] = {left_id, task.start, n_left, task.depth + 1};
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0)
    {
        out_n_nodes[local_t] = s_node_count;
    }
}

int forest_train_gpu(Forest *f, const Dataset *ds, const ForestParams *p,
                     double *h2d_ms, double *kernel_ms, double *d2h_ms)
{
    if (!f || !ds || !p || ds->n_samples <= 0 || ds->n_features <= 0)
        return -1;

    if (!cuda_device_available())
    {
        fprintf(stderr, "warning: CUDA device not available, falling back to CPU train\n");
        return forest_train(f, ds, p);
    }

    memset(f, 0, sizeof(*f));
    f->n_trees = p->n_trees > 0 ? p->n_trees : 10;
    f->n_features = ds->n_features;
    f->n_classes = ds->n_classes;
    f->max_depth = p->max_depth > 0 ? p->max_depth : 16;
    f->min_samples_split = p->min_samples_split > 1 ? p->min_samples_split : 2;
    if (p->mtry > 0)
        f->mtry = p->mtry;
    else
    {
        int m = (int)sqrt((double)ds->n_features);
        f->mtry = m < 1 ? 1 : m;
    }
    if (f->mtry > ds->n_features)
        f->mtry = ds->n_features;

    f->trees = (Tree *)calloc((size_t)f->n_trees, sizeof(Tree));
    if (!f->trees)
        return -1;

    int max_nodes;
    if (f->max_depth <= 8)
        max_nodes = (1 << (f->max_depth + 1));
    else if (f->max_depth <= 16)
        max_nodes = 16384;
    else
        max_nodes = 65536;

    size_t x_bytes = (size_t)ds->n_samples * (size_t)ds->n_features * sizeof(double);
    size_t y_bytes = (size_t)ds->n_samples * sizeof(int);

    size_t per_tree_bytes = 0;
    if (ds->n_samples <= MAX_PAIRS_SHARED)
    {
        per_tree_bytes = (size_t)max_nodes * sizeof(Node) + sizeof(int);
    }
    else
    {
        per_tree_bytes = (size_t)ds->n_samples * sizeof(int) * 2 +
                         (size_t)ds->n_samples * sizeof(DevPair) +
                         (size_t)max_nodes * sizeof(Node) +
                         sizeof(int);
    }

    size_t max_workspace_vram = 512ULL * 1024ULL * 1024ULL;
    int batch_size = (int)(max_workspace_vram / per_tree_bytes);
    if (ds->n_samples > 100000 && batch_size > 10)
        batch_size = 10;
    if (batch_size < 1)
        batch_size = 1;
    if (batch_size > f->n_trees)
        batch_size = f->n_trees;

    size_t batch_samples_bytes = (size_t)batch_size * (size_t)ds->n_samples * sizeof(int);
    size_t batch_pairs_bytes = (size_t)batch_size * (size_t)ds->n_samples * sizeof(DevPair);
    size_t batch_nodes_bytes = (size_t)batch_size * (size_t)max_nodes * sizeof(Node);
    size_t batch_counts_bytes = (size_t)batch_size * sizeof(int);

    double *d_x = NULL;
    int *d_y = NULL;
    int *d_samples = NULL;
    int *d_temp = NULL;
    DevPair *d_pairs = NULL;
    Node *d_nodes = NULL;
    int *d_n_nodes = NULL;

    Node *h_batch_nodes = (Node *)malloc(batch_nodes_bytes);
    int *h_batch_counts = (int *)malloc(batch_counts_bytes);

    int status = 0;
    cudaEvent_t ev_h2d_start = NULL, ev_h2d_end = NULL;
    cudaEvent_t ev_kern_start = NULL, ev_kern_end = NULL;
    cudaEvent_t ev_d2h_start = NULL, ev_d2h_end = NULL;

    float t_h2d_total = 0.0f, t_kern_total = 0.0f, t_d2h_total = 0.0f;

    CUDA_CHECK_CLEANUP(cudaEventCreate(&ev_h2d_start));
    CUDA_CHECK_CLEANUP(cudaEventCreate(&ev_h2d_end));
    CUDA_CHECK_CLEANUP(cudaEventCreate(&ev_kern_start));
    CUDA_CHECK_CLEANUP(cudaEventCreate(&ev_kern_end));
    CUDA_CHECK_CLEANUP(cudaEventCreate(&ev_d2h_start));
    CUDA_CHECK_CLEANUP(cudaEventCreate(&ev_d2h_end));

    CUDA_CHECK_CLEANUP(cudaMalloc((void **)&d_x, x_bytes));
    CUDA_CHECK_CLEANUP(cudaMalloc((void **)&d_y, y_bytes));
    if (ds->n_samples > MAX_PAIRS_SHARED)
    {
        CUDA_CHECK_CLEANUP(cudaMalloc((void **)&d_samples, batch_samples_bytes));
        CUDA_CHECK_CLEANUP(cudaMalloc((void **)&d_temp, batch_samples_bytes));
        CUDA_CHECK_CLEANUP(cudaMalloc((void **)&d_pairs, batch_pairs_bytes));
    }
    CUDA_CHECK_CLEANUP(cudaMalloc((void **)&d_nodes, batch_nodes_bytes));
    CUDA_CHECK_CLEANUP(cudaMalloc((void **)&d_n_nodes, batch_counts_bytes));

    if (!h_batch_nodes || !h_batch_counts)
        goto cleanup;

    /* 1. Host to Device transfer */
    CUDA_CHECK_CLEANUP(cudaEventRecord(ev_h2d_start, 0));
    CUDA_CHECK_CLEANUP(cudaMemcpyAsync(d_x, ds->x, x_bytes, cudaMemcpyHostToDevice, 0));
    CUDA_CHECK_CLEANUP(cudaMemcpyAsync(d_y, ds->y, y_bytes, cudaMemcpyHostToDevice, 0));
    CUDA_CHECK_CLEANUP(cudaEventRecord(ev_h2d_end, 0));
    CUDA_CHECK_CLEANUP(cudaEventSynchronize(ev_h2d_end));
    cudaEventElapsedTime(&t_h2d_total, ev_h2d_start, ev_h2d_end);

    /* 2. Execute batches */
    for (int offset = 0; offset < f->n_trees; offset += batch_size)
    {
        int cur_batch = f->n_trees - offset;
        if (cur_batch > batch_size)
            cur_batch = batch_size;

        CUDA_CHECK_CLEANUP(cudaEventRecord(ev_kern_start, 0));
        if (ds->n_samples <= MAX_PAIRS_SHARED)
        {
            rf_train_forest_kernel_small<<<cur_batch, 256, 0, 0>>>(
                d_x, d_y, ds->n_samples, ds->n_features, ds->n_classes,
                f->max_depth, f->min_samples_split, f->mtry, p->seed,
                offset, d_nodes, d_n_nodes, max_nodes);
        }
        else
        {
            rf_train_forest_kernel<<<cur_batch, 256, 0, 0>>>(
                d_x, d_y, ds->n_samples, ds->n_features, ds->n_classes,
                f->max_depth, f->min_samples_split, f->mtry, p->seed,
                offset, d_samples, d_temp, d_pairs, d_nodes, d_n_nodes, max_nodes);
        }
        CUDA_CHECK_CLEANUP(cudaGetLastError());
        CUDA_CHECK_CLEANUP(cudaEventRecord(ev_kern_end, 0));
        CUDA_CHECK_CLEANUP(cudaEventSynchronize(ev_kern_end));

        float t_kern_batch = 0.0f;
        cudaEventElapsedTime(&t_kern_batch, ev_kern_start, ev_kern_end);
        t_kern_total += t_kern_batch;

        /* 3. Device to Host transfer of tree structures */
        CUDA_CHECK_CLEANUP(cudaEventRecord(ev_d2h_start, 0));
        CUDA_CHECK_CLEANUP(cudaMemcpyAsync(h_batch_counts, d_n_nodes,
                                           (size_t)cur_batch * sizeof(int),
                                           cudaMemcpyDeviceToHost, 0));
        CUDA_CHECK_CLEANUP(cudaMemcpyAsync(h_batch_nodes, d_nodes,
                                           (size_t)cur_batch * (size_t)max_nodes * sizeof(Node),
                                           cudaMemcpyDeviceToHost, 0));
        CUDA_CHECK_CLEANUP(cudaEventRecord(ev_d2h_end, 0));
        CUDA_CHECK_CLEANUP(cudaEventSynchronize(ev_d2h_end));

        float t_d2h_batch = 0.0f;
        cudaEventElapsedTime(&t_d2h_batch, ev_d2h_start, ev_d2h_end);
        t_d2h_total += t_d2h_batch;

        /* Unpack trees into Forest structure */
        for (int b = 0; b < cur_batch; b++)
        {
            int tree_idx = offset + b;
            int n_nds = h_batch_counts[b];
            if (n_nds <= 0)
                n_nds = 1;
            f->trees[tree_idx].n_nodes = n_nds;
            f->trees[tree_idx].cap_nodes = n_nds;
            f->trees[tree_idx].n_classes = ds->n_classes;
            f->trees[tree_idx].nodes = (Node *)malloc((size_t)n_nds * sizeof(Node));
            if (!f->trees[tree_idx].nodes)
                goto cleanup;
            memcpy(f->trees[tree_idx].nodes,
                   h_batch_nodes + (size_t)b * (size_t)max_nodes,
                   (size_t)n_nds * sizeof(Node));
        }
    }

    if (h2d_ms)
        *h2d_ms = (double)t_h2d_total;
    if (kernel_ms)
        *kernel_ms = (double)t_kern_total;
    if (d2h_ms)
        *d2h_ms = (double)t_d2h_total;

    goto done;

cleanup:
    status = -1;
    forest_free(f);

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

    if (d_x)
        cudaFree(d_x);
    if (d_y)
        cudaFree(d_y);
    if (d_samples)
        cudaFree(d_samples);
    if (d_temp)
        cudaFree(d_temp);
    if (d_pairs)
        cudaFree(d_pairs);
    if (d_nodes)
        cudaFree(d_nodes);
    if (d_n_nodes)
        cudaFree(d_n_nodes);

    free(h_batch_nodes);
    free(h_batch_counts);

    return status;
}
