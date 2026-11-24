#include "rf.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define BATCH 256
static int agree(int fail) { int any; MPI_Allreduce(&fail, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); return any; }
static int predict(const Forest *f, const Dataset *d, int rank, int *local, int *sum) {
    int correct = 0, nc = d->n_classes;

    for (int start = 0; start < d->n_samples; start += BATCH) {
        int n = d->n_samples - start; if (n > BATCH) n = BATCH;
        
        for (int i = 0; i < n; i++)
            forest_count_votes(f, d->x + (size_t)(start+i)*d->n_features, local + (size_t)i*nc);
        
        MPI_Reduce(local, sum, n*nc, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (!rank) for (int i = 0; i < n; i++)
            correct += majority_class(sum + (size_t)i*nc, nc) == d->y[start+i];
    }

    return correct;
}

int main(int argc, char **argv) {
    int rank, np, fail = 0, dims[4] = {0}, rc = 1;
    const char *path = "data/iris.csv", *target = NULL, *dot = NULL;
    int dot_tree = 0;
    double frac = .2, elapsed[3], max_time[3], t;
    ForestParams p = {100, 8, 2, 0, 42};
    Dataset full = {0}, train = {0}, test = {0}; Forest forest = {0};
    int *local = NULL, *sum = NULL;
    MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { 
            if (!rank) puts("--data PATH --target COL --trees N --max-depth D --min-samples M --mtry K --test-frac F --seed S --dot FILE --dot-tree N"); 
            rc=0; 
            goto done; 
        }

        if (i+1 >= argc) { 
            fail=1;
            break; 
        }
        
        const char *key=argv[i], *v=argv[++i];
        if (!strcmp(key,"--data")) path=v; else if (!strcmp(key,"--target")) target=v;
        else if (!strcmp(key,"--trees")) p.n_trees=atoi(v); else if (!strcmp(key,"--max-depth")) p.max_depth=atoi(v);
        else if (!strcmp(key,"--min-samples")) p.min_samples_split=atoi(v); else if (!strcmp(key,"--mtry")) p.mtry=atoi(v);
        else if (!strcmp(key,"--test-frac")) frac=atof(v); else if (!strcmp(key,"--seed")) p.seed=(uint32_t)strtoul(v,NULL,10);
        else if (!strcmp(key,"--dot")) dot=v; else if (!strcmp(key,"--dot-tree")) dot_tree=atoi(v); 
        else fail=1;
    }

    if (p.n_trees <= 0 || (dot && (dot_tree < 0 || dot_tree >= p.n_trees))) fail=1;
    if (agree(fail)) { 
        if (!rank) fprintf(stderr,"invalid arguments\n"); 
        goto done; 
    }

    t=MPI_Wtime();

    if (!rank) {
        if (dataset_load_csv(path,&full,target)) fail=1;
        else { dataset_split_holdout(&full,&train,&test,frac,p.seed); if (!train.x || !train.y || !test.x || !test.y || !train.n_samples || !test.n_samples) fail=1; }
        dims[0]=train.n_samples; dims[1]=test.n_samples; dims[2]=full.n_features; dims[3]=full.n_classes;
        if ((long long)dims[0]*dims[2]>INT_MAX || (long long)dims[1]*dims[2]>INT_MAX || dims[3]>INT_MAX/BATCH) fail=1;
    }

    if (agree(fail)) { 
        if (!rank) fprintf(stderr,"dataset load or split failed\n"); 
        goto done; 
    }

    MPI_Bcast(dims,4,MPI_INT,0,MPI_COMM_WORLD);

    if (rank) {
        train.n_samples=dims[0]; test.n_samples=dims[1]; train.n_features=test.n_features=dims[2]; train.n_classes=test.n_classes=dims[3];
        train.x=malloc((size_t)dims[0]*dims[2]*sizeof(double)); train.y=malloc((size_t)dims[0]*sizeof(int));
        test.x=malloc((size_t)dims[1]*dims[2]*sizeof(double));
        if (!train.x || !train.y || !test.x) fail=1;
    }
    local=malloc((size_t)BATCH*dims[3]*sizeof(int)); sum=malloc((size_t)BATCH*dims[3]*sizeof(int));
    if (!local || !sum) fail=1;

    if (agree(fail)) { 
        if (!rank) fprintf(stderr,"allocation failed\n"); 
        goto done; 
    }

    MPI_Bcast(train.x,dims[0]*dims[2],MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(train.y,dims[0],MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(test.x,dims[1]*dims[2],MPI_DOUBLE,0,MPI_COMM_WORLD);
    elapsed[0]=MPI_Wtime()-t;
    t=MPI_Wtime(); fail=forest_train_partition(&forest,&train,&p,rank,np)!=0;

    if (agree(fail)) { 
        if (!rank) fprintf(stderr,"training failed\n"); 
        goto done; 
    }

    elapsed[1]=MPI_Wtime()-t;
    t=MPI_Wtime(); int tr=predict(&forest,&train,rank,local,sum), te=predict(&forest,&test,rank,local,sum);
    elapsed[2]=MPI_Wtime()-t;
    MPI_Reduce(elapsed,max_time,3,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if (!rank) printf("processes=%d\ntrain_accuracy=%.4f\ntest_accuracy=%.4f\ndistribution_wall_s=%.6f train_wall_s=%.6f predict_wall_s=%.6f\n",np,(double)tr/dims[0],(double)te/dims[1],max_time[0],max_time[1],max_time[2]);
    if (dot) {
        fail=0; if (rank==dot_tree%np) fail=forest_write_dot_global(&forest,dot_tree/np,dot_tree,dot)!=0;
        if (agree(fail)) { if (!rank) fprintf(stderr,"failed to write DOT: %s\n",dot); goto done; }
        if (!rank) printf("wrote_dot=%s tree=%d rank=%d\n",dot,dot_tree,dot_tree%np);
    }
    rc=0;
done:
    free(local); 
    free(sum); 
    forest_free(&forest); 
    dataset_free(&full); 
    dataset_free(&train); 
    dataset_free(&test); 
    MPI_Finalize(); 
    return rc;
}