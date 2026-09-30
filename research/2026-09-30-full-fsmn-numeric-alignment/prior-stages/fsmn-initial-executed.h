#ifndef DONOR_FSMN_H
#define DONOR_FSMN_H
#include <stddef.h>
#define DF_FLOATS 756933u
#define DF_CACHE_FLOATS 5632u
/* Cache layout matches Python [128][11][4]. Weights remain caller-owned. */
typedef struct { const float *weights; float cache[DF_CACHE_FLOATS]; } df_model;
int df_init(df_model*, const float*, size_t);
void df_reset(df_model*);
/* One 400D spliced row -> 2599 raw logits. Optional trace [21][2599],
   stage dimensions 140,250,250,(128,128,250,250)x4,140,2599. */
int df_step(df_model*, const float*, float*, float*);
#endif
