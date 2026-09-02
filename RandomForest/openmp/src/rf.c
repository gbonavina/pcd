#include "rf.h"

#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- RNG (xorshift32): one state per tree so OpenMP can use seed+t) --- */

static uint32_t rng_u32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x ? x : 0x9E3779B9u;
    return *s;
}

static int rng_int(uint32_t *s, int n) {
    if (n <= 1)
        return 0;
    return (int)(rng_u32(s) % (uint32_t)n);
}

/* ---------- Dataset I/O ----------------------------------------------------- */

static int is_blank_line(const char *s) {
    while (*s) {
        if (!isspace((unsigned char)*s))
            return 0;
        s++;
    }
    return 1;
}

static void strip_nl(char *s) {
    size_t len = strlen(s);
    while (len && (s[len - 1] == '\n' || s[len - 1] == '\r'))
        s[--len] = '\0';
}

static const char *skip_ws(const char *s) {
    while (*s && isspace((unsigned char)*s))
        s++;
    return s;
}

/* Whole field is a C float (no leftover like the '-' in 2023-02-03). */
static int field_is_number(const char *s) {
    s = skip_ws(s);
    if (!*s)
        return 0;
    char *end = NULL;
    strtod(s, &end);
    if (end == s)
        return 0;
    end = (char *)skip_ws(end);
    return *end == '\0';
}

static int field_is_date(const char *s) {
    int y = 0, m = 0, d = 0;
    char extra;
    s = skip_ws(s);
    if (sscanf(s, "%d-%d-%d%c", &y, &m, &d, &extra) != 3)
        return 0;
    return y >= 1000 && m >= 1 && m <= 12 && d >= 1 && d <= 31;
}

static double field_date_value(const char *s) {
    int y = 0, m = 0, d = 0;
    sscanf(skip_ws(s), "%d-%d-%d", &y, &m, &d);
    return (double)(y * 10000 + m * 100 + d);
}

static int split_csv_fields(char *line, char **fields, int max_cols) {
    int n = 0;
    char *p = line;
    while (n < max_cols) {
        fields[n++] = p;
        char *c = strchr(p, ',');
        if (!c)
            break;
        *c = '\0';
        p = c + 1;
    }
    return n;
}

static int row_is_header(char **fields, int ncols) {
    for (int i = 0; i < ncols; i++) {
        if (!field_is_number(fields[i]) && !field_is_date(fields[i]))
            return 1;
    }
    return 0;
}

typedef struct {
    char **keys;
    int n;
    int cap;
} StrMap;

static void strmap_free(StrMap *m) {
    if (!m)
        return;
    for (int i = 0; i < m->n; i++)
        free(m->keys[i]);
    free(m->keys);
    m->keys = NULL;
    m->n = m->cap = 0;
}

static int strmap_intern(StrMap *m, const char *s) {
    for (int i = 0; i < m->n; i++)
        if (strcmp(m->keys[i], s) == 0)
            return i;
    if (m->n >= m->cap) {
        int cap = m->cap ? m->cap * 2 : 16;
        char **nk = realloc(m->keys, (size_t)cap * sizeof(char *));
        if (!nk)
            return -1;
        m->keys = nk;
        m->cap = cap;
    }
    size_t len = strlen(s);
    char *copy = malloc(len + 1);
    if (!copy)
        return -1;
    memcpy(copy, s, len + 1);
    m->keys[m->n] = copy;
    return m->n++;
}

enum { COL_NUM = 0, COL_DATE = 1, COL_CAT = 2 };

static int infer_col_kind(char **grid, int n_rows, int ncols, int col) {
    int saw = 0, all_num = 1, all_date = 1;
    for (int r = 0; r < n_rows; r++) {
        const char *s = grid[(size_t)r * (size_t)ncols + (size_t)col];
        if (!s || !s[0])
            continue;
        saw = 1;
        if (!field_is_number(s))
            all_num = 0;
        if (!field_is_date(s))
            all_date = 0;
    }
    if (!saw)
        return COL_NUM;
    if (all_date)
        return COL_DATE;
    if (all_num)
        return COL_NUM;
    return COL_CAT;
}

static char *dup_str(const char *s) {
    size_t len = strlen(s);
    char *copy = malloc(len + 1);
    if (!copy)
        return NULL;
    memcpy(copy, s, len + 1);
    return copy;
}

static void free_grid(char **grid, int n_cells) {
    if (!grid)
        return;
    for (int i = 0; i < n_cells; i++)
        free(grid[i]);
    free(grid);
}

static void free_names(char **names, int n) {
    if (!names)
        return;
    for (int i = 0; i < n; i++)
        free(names[i]);
    free(names);
}

/* Name match first, then 0-based integer index. NULL/empty => last column. */
static int resolve_target_col(const char *target, char **names, int ncols) {
    if (!target || !target[0])
        return ncols - 1;
    if (names) {
        for (int i = 0; i < ncols; i++)
            if (names[i] && strcmp(names[i], target) == 0)
                return i;
    }
    char *end = NULL;
    long v = strtol(target, &end, 10);
    if (end != target && *skip_ws(end) == '\0' && v >= 0 && v < ncols)
        return (int)v;
    return -1;
}

int dataset_load_csv(const char *path, Dataset *ds, const char *target) {
    memset(ds, 0, sizeof(*ds));
    FILE *fp = fopen(path, "r");
    if (!fp)
        return -1;

    char line[8192];
    int ncols = -1;
    int cap = 64;
    int n = 0;
    char **grid = NULL;
    char **colnames = NULL;
    int first = 1;

    while (fgets(line, sizeof line, fp)) {
        if (is_blank_line(line))
            continue;
        strip_nl(line);

        char *fields[256];
        char tmp[8192];
        memcpy(tmp, line, strlen(line) + 1);
        int nf = split_csv_fields(tmp, fields, 256);
        if (nf < 2) {
            fclose(fp);
            free_grid(grid, n * (ncols > 0 ? ncols : 0));
            free_names(colnames, ncols > 0 ? ncols : 0);
            return -1;
        }

        if (first) {
            ncols = nf;
            first = 0;
            if (row_is_header(fields, nf)) {
                colnames = calloc((size_t)ncols, sizeof(char *));
                if (!colnames) {
                    fclose(fp);
                    return -1;
                }
                for (int c = 0; c < ncols; c++) {
                    colnames[c] = dup_str(fields[c]);
                    if (!colnames[c]) {
                        fclose(fp);
                        free_names(colnames, ncols);
                        return -1;
                    }
                }
                continue;
            }
        }

        if (nf != ncols) {
            fclose(fp);
            free_grid(grid, n * ncols);
            free_names(colnames, ncols);
            return -1;
        }

        if (n == cap || !grid) {
            if (!grid)
                cap = 64;
            else
                cap *= 2;
            char **ng = realloc(grid, (size_t)cap * (size_t)ncols * sizeof(char *));
            if (!ng) {
                fclose(fp);
                free_grid(grid, n * ncols);
                free_names(colnames, ncols);
                return -1;
            }
            grid = ng;
        }

        for (int c = 0; c < ncols; c++) {
            size_t len = strlen(fields[c]);
            char *cell = malloc(len + 1);
            if (!cell) {
                fclose(fp);
                free_grid(grid, n * ncols + c);
                free_names(colnames, ncols);
                return -1;
            }
            memcpy(cell, fields[c], len + 1);
            grid[(size_t)n * (size_t)ncols + (size_t)c] = cell;
        }
        n++;
    }
    fclose(fp);

    if (n == 0 || ncols < 2) {
        free_grid(grid, n * (ncols > 0 ? ncols : 0));
        free_names(colnames, ncols > 0 ? ncols : 0);
        return -1;
    }

    int target_col = resolve_target_col(target, colnames, ncols);
    if (target_col < 0) {
        fprintf(stderr, "unknown target column: %s\n", target ? target : "");
        free_grid(grid, n * ncols);
        free_names(colnames, ncols);
        return -1;
    }

    /* Match the notebook: drop the identifier column, keep numeric features. */
    int n_features = 0;
    for (int c = 0; c < ncols; c++) {
        if (c == target_col)
            continue;
        if (colnames && colnames[c] && strcmp(colnames[c], "id") == 0)
            continue;
        n_features++;
    }
    if (n_features < 1) {
        free_grid(grid, n * ncols);
        free_names(colnames, ncols);
        return -1;
    }
    int *feat_src = malloc((size_t)n_features * sizeof(int));
    if (!feat_src) {
        free_grid(grid, n * ncols);
        free_names(colnames, ncols);
        return -1;
    }
    for (int c = 0, f = 0; c < ncols; c++) {
        if (c == target_col)
            continue;
        if (colnames && colnames[c] && strcmp(colnames[c], "id") == 0)
            continue;
        feat_src[f++] = c;
    }

    int *kind = malloc((size_t)n_features * sizeof(int));
    double *x = malloc((size_t)n * (size_t)n_features * sizeof(double));
    int *y = malloc((size_t)n * sizeof(int));
    if (!kind || !x || !y) {
        free(feat_src);
        free(kind);
        free(x);
        free(y);
        free_grid(grid, n * ncols);
        free_names(colnames, ncols);
        return -1;
    }

    StrMap *feat_maps = calloc((size_t)n_features, sizeof(StrMap));
    StrMap ymap = {0};
    if (!feat_maps) {
        free(feat_src);
        free(kind);
        free(x);
        free(y);
        free_grid(grid, n * ncols);
        free_names(colnames, ncols);
        return -1;
    }

    for (int f = 0; f < n_features; f++)
        kind[f] = infer_col_kind(grid, n, ncols, feat_src[f]);

    int ok = 1;
    for (int r = 0; r < n && ok; r++) {
        for (int f = 0; f < n_features && ok; f++) {
            const char *s = grid[(size_t)r * (size_t)ncols + (size_t)feat_src[f]];
            double v = 0.0;
            if (kind[f] == COL_DATE)
                v = field_date_value(s);
            else if (kind[f] == COL_NUM)
                v = strtod(skip_ws(s), NULL);
            else {
                int id = strmap_intern(&feat_maps[f], s);
                if (id < 0)
                    ok = 0;
                v = (double)id;
            }
            x[(size_t)r * (size_t)n_features + (size_t)f] = v;
        }
        int cls = strmap_intern(&ymap, grid[(size_t)r * (size_t)ncols + (size_t)target_col]);
        if (cls < 0)
            ok = 0;
        y[r] = cls;
    }

    for (int f = 0; f < n_features; f++)
        strmap_free(&feat_maps[f]);
    free(feat_maps);
    int n_classes = ymap.n;
    strmap_free(&ymap);
    free(kind);
    free(feat_src);
    free_grid(grid, n * ncols);

    if (!ok || n_classes <= 0) {
        free(x);
        free(y);
        free_names(colnames, ncols);
        return -1;
    }

    if (colnames && colnames[target_col])
        ds->target_name = dup_str(colnames[target_col]);
    else {
        char buf[32];
        snprintf(buf, sizeof buf, "%d", target_col);
        ds->target_name = dup_str(buf);
    }
    free_names(colnames, ncols);

    ds->x = x;
    ds->y = y;
    ds->n_samples = n;
    ds->n_features = n_features;
    ds->n_classes = n_classes;
    return 0;
}

void dataset_free(Dataset *ds) {
    if (!ds)
        return;
    free(ds->x);
    free(ds->y);
    free(ds->target_name);
    memset(ds, 0, sizeof(*ds));
}

/* sklearn.utils.extmath._approximate_mode — class quotas for a stratified split. */
static void approximate_mode(const int *class_counts, int n_classes, int n_draws,
                             int *out, uint32_t *rng) {
    int total = 0;
    for (int k = 0; k < n_classes; k++)
        total += class_counts[k];
    if (total <= 0 || n_draws <= 0) {
        memset(out, 0, (size_t)n_classes * sizeof(int));
        return;
    }

    double *cont = malloc((size_t)n_classes * sizeof(double));
    int *inds = malloc((size_t)n_classes * sizeof(int));
    if (!cont || !inds) {
        free(cont);
        free(inds);
        memset(out, 0, (size_t)n_classes * sizeof(int));
        return;
    }

    int sumf = 0;
    for (int k = 0; k < n_classes; k++) {
        cont[k] = (double)class_counts[k] / (double)total * (double)n_draws;
        out[k] = (int)floor(cont[k]);
        sumf += out[k];
    }
    int need = n_draws - sumf;
    while (need > 0) {
        double best = -1.0;
        for (int k = 0; k < n_classes; k++) {
            double rem = cont[k] - (double)out[k];
            if (rem > best)
                best = rem;
        }
        int ni = 0;
        for (int k = 0; k < n_classes; k++) {
            double rem = cont[k] - (double)out[k];
            if (fabs(rem - best) <= 1e-12)
                inds[ni++] = k;
        }
        if (ni <= 0)
            break;
        int add_now = need < ni ? need : ni;
        for (int i = 0; i < add_now; i++) {
            int j = i + rng_int(rng, ni - i);
            int tmp = inds[i];
            inds[i] = inds[j];
            inds[j] = tmp;
            out[inds[i]]++;
        }
        need -= add_now;
    }
    free(cont);
    free(inds);
}

static void copy_rows(const Dataset *full, Dataset *out, const int *rows, int n_rows) {
    int nf = full->n_features;
    out->n_samples = n_rows;
    out->n_features = nf;
    out->n_classes = full->n_classes;
    out->target_name = NULL;
    out->x = malloc((size_t)n_rows * (size_t)nf * sizeof(double));
    out->y = malloc((size_t)n_rows * sizeof(int));
    for (int i = 0; i < n_rows; i++) {
        int r = rows[i];
        memcpy(out->x + (size_t)i * (size_t)nf,
               full->x + (size_t)r * (size_t)nf,
               (size_t)nf * sizeof(double));
        out->y[i] = full->y[r];
    }
}

static void shuffle_ints(int *a, int n, uint32_t *rng) {
    for (int i = n - 1; i > 0; i--) {
        int j = rng_int(rng, i + 1);
        int tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
}

/* sklearn train_test_split(..., test_size=F, stratify=y, random_state=seed):
 * n_test = ceil(F * n), class counts via _approximate_mode. */
void dataset_split_holdout(const Dataset *full, Dataset *train, Dataset *test,
                           double test_frac, uint32_t seed) {
    int n = full->n_samples;
    int n_classes = full->n_classes;
    int n_test = (int)ceil(test_frac * (double)n);
    if (n_test < 1)
        n_test = 1;
    if (n_test >= n)
        n_test = n / 5;
    int n_train = n - n_test;

    int *counts = calloc((size_t)n_classes, sizeof(int));
    for (int i = 0; i < n; i++)
        counts[full->y[i]]++;

    int min_c = n;
    for (int k = 0; k < n_classes; k++)
        if (counts[k] < min_c)
            min_c = counts[k];

    uint32_t rng = seed ? seed : 1u;

    /* Fall back to a plain shuffle if a class is too small to stratify. */
    if (n_classes < 2 || min_c < 2 || n_train < n_classes || n_test < n_classes) {
        int *perm = malloc((size_t)n * sizeof(int));
        for (int i = 0; i < n; i++)
            perm[i] = i;
        shuffle_ints(perm, n, &rng);
        copy_rows(full, train, perm, n_train);
        copy_rows(full, test, perm + n_train, n_test);
        free(perm);
        free(counts);
        return;
    }

    int *n_i = malloc((size_t)n_classes * sizeof(int));
    int *t_i = malloc((size_t)n_classes * sizeof(int));
    int *remain = malloc((size_t)n_classes * sizeof(int));
    int **by_class = calloc((size_t)n_classes, sizeof(int *));
    int *fill = calloc((size_t)n_classes, sizeof(int));
    int *train_rows = malloc((size_t)n_train * sizeof(int));
    int *test_rows = malloc((size_t)n_test * sizeof(int));
    if (!n_i || !t_i || !remain || !by_class || !fill || !train_rows || !test_rows) {
        free(n_i);
        free(t_i);
        free(remain);
        free(by_class);
        free(fill);
        free(train_rows);
        free(test_rows);
        free(counts);
        memset(train, 0, sizeof(*train));
        memset(test, 0, sizeof(*test));
        return;
    }

    for (int k = 0; k < n_classes; k++) {
        by_class[k] = malloc((size_t)counts[k] * sizeof(int));
        if (!by_class[k]) {
            for (int j = 0; j < k; j++)
                free(by_class[j]);
            free(n_i);
            free(t_i);
            free(remain);
            free(by_class);
            free(fill);
            free(train_rows);
            free(test_rows);
            free(counts);
            memset(train, 0, sizeof(*train));
            memset(test, 0, sizeof(*test));
            return;
        }
    }
    for (int i = 0; i < n; i++) {
        int c = full->y[i];
        by_class[c][fill[c]++] = i;
    }

    approximate_mode(counts, n_classes, n_train, n_i, &rng);
    for (int k = 0; k < n_classes; k++)
        remain[k] = counts[k] - n_i[k];
    approximate_mode(remain, n_classes, n_test, t_i, &rng);

    int tr = 0, te = 0;
    for (int k = 0; k < n_classes; k++) {
        shuffle_ints(by_class[k], counts[k], &rng);
        int take_tr = n_i[k];
        int take_te = t_i[k];
        if (take_tr + take_te > counts[k])
            take_te = counts[k] - take_tr;
        for (int i = 0; i < take_tr; i++)
            train_rows[tr++] = by_class[k][i];
        for (int i = 0; i < take_te; i++)
            test_rows[te++] = by_class[k][take_tr + i];
    }
    n_train = tr;
    n_test = te;
    shuffle_ints(train_rows, n_train, &rng);
    shuffle_ints(test_rows, n_test, &rng);

    copy_rows(full, train, train_rows, n_train);
    copy_rows(full, test, test_rows, n_test);

    for (int k = 0; k < n_classes; k++)
        free(by_class[k]);
    free(by_class);
    free(fill);
    free(n_i);
    free(t_i);
    free(remain);
    free(train_rows);
    free(test_rows);
    free(counts);
}

/* ---------- Tree helpers ---------------------------------------------------- */

static int tree_add_node(Tree *t) {
    if (t->n_nodes >= t->cap_nodes) {
        int cap = t->cap_nodes ? t->cap_nodes * 2 : 16;
        Node *nn = realloc(t->nodes, (size_t)cap * sizeof(Node));
        if (!nn)
            return -1;
        t->nodes = nn;
        t->cap_nodes = cap;
    }
    int id = t->n_nodes++;
    t->nodes[id].feature = -1;
    t->nodes[id].threshold = 0.0;
    t->nodes[id].left = -1;
    t->nodes[id].right = -1;
    t->nodes[id].pred_class = 0;
    return id;
}

static int majority_class(const int *counts, int n_classes) {
    int best = 0;
    for (int k = 1; k < n_classes; k++)
        if (counts[k] > counts[best])
            best = k;
    return best;
}

static double gini_from_counts(const int *c, int n_classes, int n) {
    if (n <= 0)
        return 0.0;
    double g = 1.0;
    for (int k = 0; k < n_classes; k++) {
        double p = (double)c[k] / (double)n;
        g -= p * p;
    }
    return g;
}

typedef struct {
    double v;
    int y;
} Pair;

static int cmp_pair(const void *a, const void *b) {
    double da = ((const Pair *)a)->v;
    double db = ((const Pair *)b)->v;
    if (da < db)
        return -1;
    if (da > db)
        return 1;
    return 0;
}

static int pick_mtry_features(int *feats, int n_features, int mtry, uint32_t *rng) {
    /* Partial Fisher-Yates: first mtry of a shuffled index list */
    int *idx = malloc((size_t)n_features * sizeof(int));
    if (!idx)
        return -1;
    for (int i = 0; i < n_features; i++)
        idx[i] = i;
    for (int i = 0; i < mtry; i++) {
        int j = i + rng_int(rng, n_features - i);
        int tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
        feats[i] = idx[i];
    }
    free(idx);
    return 0;
}

typedef struct {
    int feature;
    double threshold;
    double gini;
    int ok;
} BestSplit;

/*
 * OPENMP (usually not worth it): the loop over mtry features below is a
 * candidate for nested parallelism while searching a split:
 *
 *   #pragma omp parallel for schedule(static) if(n >= 256)
 *
 * Each feature needs its own Pair buffer and running class counts (private).
 * Do NOT nest this under tree-level parallel for unless you set
 * omp_set_nested(0) or use omp_set_max_active_levels(1) — tree-level
 * parallelism already saturates cores. Parallel qsort of one feature is
 * almost never useful at node scale.
 */
static BestSplit find_best_split(const Dataset *ds, const int *idx, int n,
                                 int mtry, uint32_t *rng, int n_classes) {
    BestSplit best = {-1, 0.0, 1e300, 0};
    int *feats = malloc((size_t)mtry * sizeof(int));
    if (!feats || pick_mtry_features(feats, ds->n_features, mtry, rng) != 0) {
        free(feats);
        return best;
    }

    Pair *pairs = malloc((size_t)n * sizeof(Pair));
    int *left_c = calloc((size_t)n_classes, sizeof(int));
    int *right_c = calloc((size_t)n_classes, sizeof(int));
    if (!pairs || !left_c || !right_c) {
        free(feats);
        free(pairs);
        free(left_c);
        free(right_c);
        return best;
    }

    /* OPENMP-SPLIT: for (int fi = 0; fi < mtry; fi++)  — nested, high overhead */
    /* #pragma omp parallel for schedule(static)  -- example only, keep sequential */
    for (int fi = 0; fi < mtry; fi++) {
        int f = feats[fi];
        for (int i = 0; i < n; i++) {
            int row = idx[i];
            pairs[i].v = ds->x[(size_t)row * (size_t)ds->n_features + (size_t)f];
            pairs[i].y = ds->y[row];
        }
        qsort(pairs, (size_t)n, sizeof(Pair), cmp_pair);

        memset(left_c, 0, (size_t)n_classes * sizeof(int));
        memset(right_c, 0, (size_t)n_classes * sizeof(int));
        for (int i = 0; i < n; i++)
            right_c[pairs[i].y]++;

        int n_left = 0;
        int n_right = n;
        for (int i = 0; i < n - 1; i++) {
            int cls = pairs[i].y;
            left_c[cls]++;
            right_c[cls]--;
            n_left++;
            n_right--;
            if (pairs[i].v == pairs[i + 1].v)
                continue;
            double gl = gini_from_counts(left_c, n_classes, n_left);
            double gr = gini_from_counts(right_c, n_classes, n_right);
            double g = (n_left * gl + n_right * gr) / (double)n;
            if (g < best.gini) {
                best.gini = g;
                best.feature = f;
                best.threshold = 0.5 * (pairs[i].v + pairs[i + 1].v);
                best.ok = 1;
            }
        }
    }

    free(feats);
    free(pairs);
    free(left_c);
    free(right_c);
    return best;
}

static int grow_node(Tree *t, const Dataset *ds, int *idx, int n, int depth,
                     int max_depth, int min_samples_split, int mtry,
                     uint32_t *rng, int *scratch) {
    int id = tree_add_node(t);
    if (id < 0)
        return -1;

    int n_classes = t->n_classes;
    int *counts = scratch;
    memset(counts, 0, (size_t)n_classes * sizeof(int));
    for (int i = 0; i < n; i++)
        counts[ds->y[idx[i]]]++;
    t->nodes[id].pred_class = majority_class(counts, n_classes);

    int pure = 0;
    for (int k = 0; k < n_classes; k++) {
        if (counts[k] == n) {
            pure = 1;
            break;
        }
    }
    if (pure || n < min_samples_split || depth >= max_depth)
        return id;

    BestSplit sp = find_best_split(ds, idx, n, mtry, rng, n_classes);
    if (!sp.ok)
        return id;

    int n_left = 0;
    for (int i = 0; i < n; i++) {
        double v = ds->x[(size_t)idx[i] * (size_t)ds->n_features + (size_t)sp.feature];
        if (v <= sp.threshold)
            n_left++;
    }
    int n_right = n - n_left;
    if (n_left == 0 || n_right == 0)
        return id;

    int *left_idx = malloc((size_t)n_left * sizeof(int));
    int *right_idx = malloc((size_t)n_right * sizeof(int));
    if (!left_idx || !right_idx) {
        free(left_idx);
        free(right_idx);
        return id;
    }
    int li = 0, ri = 0;
    for (int i = 0; i < n; i++) {
        double v = ds->x[(size_t)idx[i] * (size_t)ds->n_features + (size_t)sp.feature];
        if (v <= sp.threshold)
            left_idx[li++] = idx[i];
        else
            right_idx[ri++] = idx[i];
    }

    /* Grow children first: tree_add_node may realloc t->nodes, so do not
     * take the address of t->nodes[id].left before the recursive calls. */
    int left = grow_node(t, ds, left_idx, n_left, depth + 1,
                         max_depth, min_samples_split, mtry, rng, scratch);
    int right = grow_node(t, ds, right_idx, n_right, depth + 1,
                          max_depth, min_samples_split, mtry, rng, scratch);
    t->nodes[id].feature = sp.feature;
    t->nodes[id].threshold = sp.threshold;
    t->nodes[id].left = left;
    t->nodes[id].right = right;
    free(left_idx);
    free(right_idx);
    return id;
}

static int tree_grow(Tree *t, const Dataset *ds, int max_depth,
                     int min_samples_split, int mtry, uint32_t seed) {
    memset(t, 0, sizeof(*t));
    t->n_classes = ds->n_classes;
    int n = ds->n_samples;
    int *boot = malloc((size_t)n * sizeof(int));
    int *scratch = calloc((size_t)ds->n_classes, sizeof(int));
    if (!boot || !scratch) {
        free(boot);
        free(scratch);
        return -1;
    }

    uint32_t rng = seed ? seed : 1u;
    for (int i = 0; i < n; i++)
        boot[i] = rng_int(&rng, n);

    int root = grow_node(t, ds, boot, n, 0, max_depth, min_samples_split, mtry,
                         &rng, scratch);
    free(boot);
    free(scratch);
    return root < 0 ? -1 : 0;
}

static int tree_predict_one(const Tree *t, const double *x, int n_features) {
    int i = 0;
    for (;;) {
        const Node *nd = &t->nodes[i];
        if (nd->feature < 0)
            return nd->pred_class;
        if (nd->feature >= n_features)
            return nd->pred_class;
        if (x[nd->feature] <= nd->threshold)
            i = nd->left;
        else
            i = nd->right;
        if (i < 0)
            return nd->pred_class;
    }
}

static void tree_free(Tree *t) {
    free(t->nodes);
    memset(t, 0, sizeof(*t));
}

/* ---------- Forest ---------------------------------------------------------- */

int forest_train(Forest *f, const Dataset *ds, const ForestParams *p) {
    memset(f, 0, sizeof(*f));
    f->n_trees = p->n_trees > 0 ? p->n_trees : 10;
    f->n_features = ds->n_features;
    f->n_classes = ds->n_classes;
    f->max_depth = p->max_depth > 0 ? p->max_depth : 16;
    f->min_samples_split = p->min_samples_split > 1 ? p->min_samples_split : 2;
    if (p->mtry > 0)
        f->mtry = p->mtry;
    else {
        int m = (int)sqrt((double)ds->n_features);
        f->mtry = m < 1 ? 1 : m;
    }
    if (f->mtry > ds->n_features)
        f->mtry = ds->n_features;

    f->trees = calloc((size_t)f->n_trees, sizeof(Tree));
    if (!f->trees)
        return -1;

    int fail = 0;
    /*
     * Trees are independent: per-tree bootstrap, RNG, and node arena.
     * Uneven tree sizes => schedule(dynamic). Do not share Tree.nodes.
     */
#pragma omp parallel for schedule(dynamic) default(none) shared(f, ds, p, fail)
    for (int t = 0; t < f->n_trees; t++) {
        uint32_t seed = p->seed + (uint32_t)t * 0x9E3779B9u + 1u;
        if (tree_grow(&f->trees[t], ds, f->max_depth, f->min_samples_split,
                      f->mtry, seed) != 0) {
#pragma omp atomic write
            fail = 1;
        }
    }
    if (fail) {
        forest_free(f);
        return -1;
    }
    return 0;
}

int forest_predict_one(const Forest *f, const double *x) {
    int *votes = calloc((size_t)f->n_classes, sizeof(int));
    if (!votes)
        return 0;

    /*
     * OPENMP (weaker): vote over trees for a *single* row. Prefer parallelizing
     * the outer sample loop in forest_predict instead. Example:
     *
     *   #pragma omp parallel for reduction(+:votes[:f->n_classes])
     *
     * (OpenMP 4.5 array reduction.) Or a thread-local vote buffer then atomic
     * add. Only useful if n_trees is huge and n_rows == 1.
     */
    /* #pragma omp parallel for reduction(+:votes[:f->n_classes]) */
    for (int t = 0; t < f->n_trees; t++) {
        int c = tree_predict_one(&f->trees[t], x, f->n_features);
        if (c >= 0 && c < f->n_classes)
            votes[c]++;
    }
    int pred = majority_class(votes, f->n_classes);
    free(votes);
    return pred;
}

void forest_predict(const Forest *f, const Dataset *ds, int *out) {
    /* Rows independent; trees read-only. */
#pragma omp parallel for schedule(static) default(none) shared(f, ds, out)
    for (int i = 0; i < ds->n_samples; i++)
        out[i] = forest_predict_one(f, ds->x + (size_t)i * (size_t)ds->n_features);
}

double forest_accuracy(const Forest *f, const Dataset *ds) {
    if (ds->n_samples <= 0)
        return 0.0;
    int *pred = malloc((size_t)ds->n_samples * sizeof(int));
    if (!pred)
        return 0.0;
    forest_predict(f, ds, pred);
    int ok = 0;
#pragma omp parallel for schedule(static) reduction(+:ok) default(none) shared(ds, pred)
    for (int i = 0; i < ds->n_samples; i++)
        if (pred[i] == ds->y[i])
            ok++;
    free(pred);
    return (double)ok / (double)ds->n_samples;
}

void forest_free(Forest *f) {
    if (!f)
        return;
    if (f->trees) {
        for (int t = 0; t < f->n_trees; t++)
            tree_free(&f->trees[t]);
        free(f->trees);
    }
    memset(f, 0, sizeof(*f));
}
