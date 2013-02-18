
/*****************************************************************

    Copyright (C) 2013 Stefan Grönroos

    Authors: Stefan Grönroos <stefan.gronroos@abo.fi>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

********************************************************************/


#ifndef DVBT2_CUDA_LDPC_H
#define DVBT2_CUDA_LDPC_H
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "helpers.h"

#define PARALLEL_FEC_BLOCKS 128 /* Process this many blocks in parallel */

/* If we use MEX, we want to redirect output to Matlab */
#ifdef USE_MEX
#include <mex.h>
#define DEBUG(...) (mexPrintf(__VA_ARGS__))
#else
#define DEBUG(...) (fprintf(stderr, __VA_ARGS__))
#endif

#define CODEWORDS 32
typedef char4 DLLR;
typedef uchar4 DBIT;

#define EDGESPERBLOCK 4 //8 for gtx570
#define MAX_BN_DEG 13
#define MAX_CN_DEG 22

#define FMAX(X,Y) (fmax(X,Y))
#define FMIN(X,Y) (fmin(X,Y))

#define IMAX(X,Y) (max(X,Y))
#define IMIN(X,Y) (min(X,Y))

#define CLAMP(VAL,MINV,MAXV) (IMIN(IMAX(VAL,MINV),MAXV))

#define F2C_FACTOR 8.0
#define CHARTOFLOAT_H(X) (((float)(X)) / F2C_FACTOR)
#define CHARTOFLOAT(X) (char2float[X+127])
#define FLOATTOCHAR(X) ((char) (FMAX(FMIN((X) * F2C_FACTOR, 127.0), -127.0)))

typedef struct gpu_edge {
	int i_next;
} GPUEDGE;

typedef struct gpu_message {
    char4 message[CODEWORDS];
} GPUMSG;

/* Context structure */
typedef struct {
    GPUEDGE *hc_dev;
    GPUEDGE *hb_dev;
    GPUMSG *edge_msg;
    DLLR *llr_d;
    DBIT *bitval_d;

    int *llr_map_d;
    int cuda_num_edges;
    int *unsatisfied_d;
    int *col_map_d;

    int *row_idx_d;
    int *col_idx_d;
} cuda_ldpc_ctx;

/* load structures onto GPU */
void cuda_ldpc_load(GPUEDGE *hb_host, GPUEDGE *hc_host, int numEdges, int M, int N, int* llr_map_h, int* col_map_h, int* gpu_row_idx, int *gpu_col_idx);

/* Launch GPU decoder */
void cuda_ldpc_d(DLLR *llr_h_words, int M, int N, DBIT *bitval_h_words, int max_iter);

/* Clean up GPU memory */
void cuda_ldpc_destroy();

#endif /* DVBT2_CUDA_LDPC_H */
