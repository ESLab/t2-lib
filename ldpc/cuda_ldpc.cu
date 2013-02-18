
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

#include "cuda_ldpc.h"


static cuda_ldpc_ctx h = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* CUDA PTX "assembler" instruction set contains a nice instruction for saturating conversion from one type to another,
   perfect for converting from short back to char */
__device__ char short_to_char(short value)
{
    int result;
    asm("cvt.sat.s8.s16 %0, %1;" : "=r" (result) : "h" (value));
    return (char)result;
}

void Check_CUDA_Error(const char *message)
{
    cudaError_t error = cudaGetLastError();

    if(error!=cudaSuccess) {
        DEBUG("ERROR: %s: %s\n", message, cudaGetErrorString(error) );
        exit(-1);
    }
}

__global__ void cuda_ldpc_init_messages(GPUMSG *emsg, const int num_edges) {
    int i = (blockIdx.x * blockDim.y + threadIdx.y);
    if(i < num_edges) {
        emsg[i].message[threadIdx.x].x = 0;
        emsg[i].message[threadIdx.x].y = 0;
        emsg[i].message[threadIdx.x].z = 0;
        emsg[i].message[threadIdx.x].w = 0;
    }
}


__global__ void cuda_ldpc_bn_update(GPUEDGE *hc, GPUEDGE *hb, DLLR* llr, int* col_map, int N, GPUMSG *emsg, int *col_idx) {
#define W threadIdx.x
#define MSG msg
    unsigned int i = (blockIdx.x * blockDim.y + threadIdx.y);
    int col_start;
    unsigned char counter = 0;
    short4 m;
    char4 msg;

    if(i < N)
    {
        col_start = col_idx[i];
        int index = col_start;
        i = (i<<5)+threadIdx.x;
        m.x = (short)llr[i].x;
        m.y = (short)llr[i].y;
        m.z = (short)llr[i].z;
        m.w = (short)llr[i].w;

        do
        {
            MSG = emsg[index].message[W];

#define BU1(X) \
    m.X = m.X + MSG.X;

            BU1(x);
            BU1(y);
            BU1(z);
            BU1(w);
#undef BU1
            index = hc[index].i_next;
            counter++;
        } while(index != col_start);

        counter = 0;
        do
        {
            MSG = emsg[index].message[W];

#define BU2(X) \
    MSG.X = short_to_char(m.X - MSG.X);

            BU2(x);
            BU2(y);
            BU2(z);
            BU2(w);
#undef BU2
            //Write back
            emsg[index].message[W] = MSG;

            counter++;
            index = hc[index].i_next;

        } while(index != col_start);

    }

}

__global__ void cuda_ldpc_bn_update_bitval(GPUEDGE *hc, GPUEDGE *hb, DBIT* bitval, DLLR* llr, int* col_map, int N, GPUMSG *emsg, int *col_idx) {
#define W threadIdx.x
#define MSG msg
    unsigned int i = (blockIdx.x * blockDim.y + threadIdx.y);
    int col_start;
    unsigned char counter = 0;
    short4 m;
    char4 msg;

    if(i < N)
    {
        col_start = col_idx[i];
        int index = col_start;
        i = (i<<5)+threadIdx.x;
        m.x = (short)llr[i].x;
        m.y = (short)llr[i].y;
        m.z = (short)llr[i].z;
        m.w = (short)llr[i].w;

        do
        {
            MSG = emsg[index].message[W];

#define BU1(X) \
    m.X += MSG.X;

            BU1(x);
            BU1(y);
            BU1(z);
            BU1(w);
#undef BU1

            index = hc[index].i_next;
            counter++;
        } while(index != col_start);

        counter = 0;
        do
        {
            MSG = emsg[index].message[W];
#define BU2(X) \
    MSG.X = short_to_char(m.X - MSG.X);

            BU2(x);
            BU2(y);
            BU2(z);
            BU2(w);
#undef BU2
            //Write back
            emsg[index].message[W] = MSG;

            counter++;
            index = hc[index].i_next;

        } while(index != col_start);

#define BITVAL_UPDATE(X) \
    bitval[i].X = (m.X >= 0 ? 0 : 1);

        BITVAL_UPDATE(x);
        BITVAL_UPDATE(y);
        BITVAL_UPDATE(z);
        BITVAL_UPDATE(w);

    }

}


__global__ void cuda_ldpc_cn_update(GPUEDGE *hc, GPUEDGE *hb, int M, GPUMSG *emsg, int *row_idx)
{

#define W threadIdx.x
#define MSG msg
    int i = (blockIdx.x * blockDim.y + threadIdx.y);

    int row_start;
    unsigned char counter = 0;
    unsigned char degree;
    uchar4 minMsg;
    char4 minLLR = {127, 127, 127, 127};
    char4 nMinLLR = {127, 127, 127, 127};
    char4 sign = {1,1,1,1};
    char4 msg;

    if (i < M) {
        row_start = row_idx[i]; //TODO: move to constant mem?
        int index = row_start;
        do {
            MSG = emsg[index].message[W];

#define CU1(X) \
    sign.X ^= MSG.X; \
    degree = fabsf(MSG.X); \
    nMinLLR.X = degree < minLLR.X ? minLLR.X : degree < nMinLLR.X ? degree : nMinLLR.X; \
    minLLR.X = degree < minLLR.X ? degree : minLLR.X; \
    minMsg.X = degree == minLLR.X ? counter : minMsg.X

            CU1(x);
            CU1(y);
            CU1(z);
            CU1(w);
#undef CU1
            index = hb[index].i_next;
            counter++;


        } while (index != row_start);

        counter = 0;

        do {
            MSG = emsg[index].message[W];
#define CU2(X) \
    MSG.X = (1-(((sign.X^MSG.X) & 0x80) >> 6)) * (counter != minMsg.X ? minLLR.X : nMinLLR.X)

            CU2(x);
            CU2(y);
            CU2(z);
            CU2(w);
#undef CU2

            //Write back
            emsg[index].message[W] = MSG;
            index = hb[index].i_next;
            counter++;
        } while(index != row_start);
    }
}


__global__ void cuda_ldpc_check_satisfied(GPUEDGE *hb, DBIT *bitval, int *llr_map, int *unsatisfied, int M, int *row_idx) {
    int i = (blockIdx.x * blockDim.y + threadIdx.y);
    if (i == 0) unsatisfied[0] = 0;
    __syncthreads();
    if (i < M) {
        int row_start = row_idx[i];
        int index = row_start;
        short sum = 0;
        do {
            int tmp = llr_map[index]*CODEWORDS+threadIdx.x;
            sum ^= bitval[tmp].x;
            sum ^= bitval[tmp].y;
            sum ^= bitval[tmp].z;
            sum ^= bitval[tmp].w;

            index = hb[index].i_next;
        } while (index != row_start);
        if (sum==1) atomicAdd(unsatisfied, 1);
    }
}

void cuda_ldpc_load(GPUEDGE *hb_host, GPUEDGE *hc_host, int numEdges, int M, int N, int* llr_map_h, int* col_map_h, int* row_idx_h, int* col_idx_h) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&h.hc_dev, numEdges*sizeof(GPUEDGE)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&h.hb_dev, numEdges*sizeof(GPUEDGE)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&h.llr_map_d, numEdges*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&h.edge_msg, numEdges*sizeof(GPUMSG)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&h.row_idx_d, M*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&h.col_idx_d, N*sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&h.llr_d, CODEWORDS*N*sizeof(DLLR))); //might as well allocate space for llrs already
    CUDA_SAFE_CALL(cudaMalloc((void**)&h.bitval_d, CODEWORDS*N*sizeof(DBIT)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&h.unsatisfied_d, 1*sizeof(int)));

    CUDA_SAFE_CALL(cudaMemcpy(h.hb_dev, hb_host, numEdges*sizeof(GPUEDGE), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(h.hc_dev, hc_host, numEdges*sizeof(GPUEDGE), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(h.llr_map_d, llr_map_h, numEdges*sizeof(int), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(h.row_idx_d, row_idx_h, M*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(h.col_idx_d, col_idx_h, N*sizeof(int), cudaMemcpyHostToDevice));

    h.cuda_num_edges = numEdges;
    if (h.llr_d)
        DEBUG("Cuda Load done\n");
}


/* A decoder function using the kernels above.
  Input LLRs are in llr_h_words, and are assumed to be ordered such that
  llr_h_words[b*PARALLEL_FEC_BLOCKS/4 + c] contains bit b of codeword c.

  Returns the result to the (assumedly already allocated) bitval_h_words
  in the same memory layout for further reordering in calling function, if necessary
*/

void cuda_ldpc_d(DLLR *llr_h_words, int M, int N, DBIT *bitval_h_words, int max_iter) {
    int unsatisfied = -1;
    if (max_iter < 1) max_iter = 1;
    START_CLOCK(2)
            CUDA_SAFE_CALL(cudaMemcpy(h.llr_d, llr_h_words, CODEWORDS*N*sizeof(DLLR), cudaMemcpyHostToDevice));

    dim3 blockSize(CODEWORDS,EDGESPERBLOCK);
    dim3 blockSizeCN(CODEWORDS,EDGESPERBLOCK);
    int gridSizeBN = ceil((float)N / EDGESPERBLOCK);
    int gridSizeCN = ceil((float)M / EDGESPERBLOCK);
    int gridSizeInit = ceil((float)h.cuda_num_edges / EDGESPERBLOCK);

    cuda_ldpc_init_messages<<<gridSizeInit, blockSize>>>(h.edge_msg, h.cuda_num_edges);

    /* We hardly use any shared memory, so maximize L1 cache size instead */
    cudaFuncSetCacheConfig(cuda_ldpc_bn_update, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(cuda_ldpc_cn_update, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(cuda_ldpc_check_satisfied, cudaFuncCachePreferL1);


    for (int i=0; i < max_iter; i++) {
        Check_CUDA_Error("Before BN update");

        if(i % 10 != 0 || i == 0) {
            cuda_ldpc_bn_update<<<gridSizeBN, blockSize>>>(h.hc_dev, h.hb_dev, h.llr_d, h.col_map_d, N, h.edge_msg, h.col_idx_d);
        } else {
            cuda_ldpc_bn_update_bitval<<<gridSizeBN, blockSize>>>(h.hc_dev, h.hb_dev, h.bitval_d, h.llr_d, h.col_map_d, N, h.edge_msg, h.col_idx_d);
            cuda_ldpc_check_satisfied<<<gridSizeCN, blockSize>>>(h.hb_dev, h.bitval_d, h.llr_map_d, h.unsatisfied_d, M, h.row_idx_d);
            CUDA_SAFE_CALL(cudaMemcpy(&unsatisfied, h.unsatisfied_d, 1*sizeof(int), cudaMemcpyDeviceToHost));
            if (unsatisfied == 0) {DEBUG("CUDA done already after %i iterations!\n", i); break;}
        }

        cuda_ldpc_cn_update<<<gridSizeCN, blockSizeCN>>>(h.hc_dev, h.hb_dev, M, h.edge_msg, h.row_idx_d);
    }
    cuda_ldpc_bn_update_bitval<<<gridSizeBN, blockSize>>>(h.hc_dev, h.hb_dev, h.bitval_d, h.llr_d, h.col_map_d, N, h.edge_msg, h.col_idx_d);

    CUDA_SAFE_CALL(cudaMemcpy(bitval_h_words, h.bitval_d, CODEWORDS*N*sizeof(DBIT), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    BENCHMARK_NOW(2, "CUDA ldpc_d")

}


/* Clear device memory */

void cuda_ldpc_destroy() {
    DEBUG("Freeing CUDA memory\n");
    if (h.hc_dev) CUDA_SAFE_CALL(cudaFree(h.hc_dev));
    if (h.hb_dev) CUDA_SAFE_CALL(cudaFree(h.hb_dev));
    if (h.llr_d) CUDA_SAFE_CALL(cudaFree(h.llr_d));
    if (h.bitval_d) CUDA_SAFE_CALL(cudaFree(h.bitval_d));
    if (h.llr_map_d) CUDA_SAFE_CALL(cudaFree(h.llr_map_d));
    if (h.unsatisfied_d) CUDA_SAFE_CALL(cudaFree(h.unsatisfied_d));
    if (h.edge_msg) CUDA_SAFE_CALL(cudaFree(h.edge_msg));

    if (h.row_idx_d) CUDA_SAFE_CALL(cudaFree(h.row_idx_d));
    if (h.col_idx_d) CUDA_SAFE_CALL(cudaFree(h.col_idx_d));

    memset(&h, 0, sizeof(cuda_ldpc_ctx)); /* NULL the context */
}
