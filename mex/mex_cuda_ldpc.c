/*****************************************************************
    MEX Function for interfacing with the CUDA LDPC decoder

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

#include <math.h>
#include <matrix.h>
#include <mex.h>
#include "../cuda_ldpc/cuda_ldpc.h"

void mexFunction_decode(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void mexFunction_free(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void mexFunction_init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void clean_up();

#define INIT_FUNC 0
#define DECODE_FUNC 1
#define FREE_FUNC 2

/* This is the master function which simply checks which slave function to run,
  based on the first argument to the function
   0 = init
   1 = decode
   2 = free
   */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mexAtExit(&clean_up);
    unsigned char *func_ptr;
    int *m_ptr;

    func_ptr = (unsigned char *)mxGetData(prhs[0]);
    //m_ptr = (int *)mxGetData(prhs[4]);
    //mexPrintf("M: %d\n", *m_ptr);


    switch(*func_ptr) {
    case INIT_FUNC:
        mexFunction_init(nlhs, plhs, nrhs-1, prhs+1);
        break;
    case DECODE_FUNC:
        mexFunction_decode(nlhs, plhs, nrhs-1, prhs+1);
        break;
    case FREE_FUNC:
        mexFunction_free(nlhs, plhs, nrhs-1, prhs+1);
        break;
    default:
        break;
    }

    return;
}

/* bitval (uint8) = function(1 (uint8), llr (single), M (int32), N (int32)) */
void mexFunction_decode(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    mexPrintf("Decoder starting...\n");
    mxArray *bitval_a;
    unsigned char *bitval;
    int *n_ptr, *m_ptr;
    int N ,M;
    float *llr;

    llr = (float *)mxGetData(prhs[0]);
    m_ptr = (int *)mxGetData(prhs[1]);
    n_ptr = (int *)mxGetData(prhs[2]);
    M = *m_ptr;
    N = *n_ptr;

    /* Create output matrix
       This matrix will contain hard decisions, and will contain
       128 rows times N columns of bit values in uint8 format
    */
    bitval_a = plhs[0] = mxCreateNumericMatrix(128, N, mxUINT8_CLASS, mxREAL);
    bitval = (unsigned char *) mxGetData(bitval_a);

    /* Allocate host-side GPU llr and bitval arrays */
    DLLR *llr_words = (DLLR *)malloc((PARALLEL_FEC_BLOCKS/4)*N*sizeof(DLLR));
    DBIT *bitval_words = (DBIT *)malloc((PARALLEL_FEC_BLOCKS/4)*N*sizeof(DBIT));

    /* Reorganize the data for the GPU */
    for(int i = 0; i < N;i++) {
       for (int n=0;n<(PARALLEL_FEC_BLOCKS/4);n++)  {
           llr_words[i*(PARALLEL_FEC_BLOCKS/4)+n].x = FLOATTOCHAR(llr[i*PARALLEL_FEC_BLOCKS+n*4]);
           llr_words[i*(PARALLEL_FEC_BLOCKS/4)+n].y = FLOATTOCHAR(llr[i*PARALLEL_FEC_BLOCKS+n*4+1]);
           llr_words[i*(PARALLEL_FEC_BLOCKS/4)+n].z = FLOATTOCHAR(llr[i*PARALLEL_FEC_BLOCKS+n*4+2]);
           llr_words[i*(PARALLEL_FEC_BLOCKS/4)+n].w = FLOATTOCHAR(llr[i*PARALLEL_FEC_BLOCKS+n*4+3]);
       }
   }


    mexPrintf("Launching CUDA decoder...\n");

    cuda_ldpc_d(llr_words, M, N, bitval_words, 50);

    /* Order back into MATLAB-friendly column-major matrix */

    for (int n=0;n<N;n++) {
        for(int i=0;i<(PARALLEL_FEC_BLOCKS/4);i++) {
            bitval[n*PARALLEL_FEC_BLOCKS+i*4] = (unsigned char) bitval_words[n*(PARALLEL_FEC_BLOCKS/4) + i].x;
            bitval[n*PARALLEL_FEC_BLOCKS+i*4+1] = (unsigned char) bitval_words[n*(PARALLEL_FEC_BLOCKS/4) + i].y;
            bitval[n*PARALLEL_FEC_BLOCKS+i*4+2] = (unsigned char) bitval_words[n*(PARALLEL_FEC_BLOCKS/4) + i].z;
            bitval[n*PARALLEL_FEC_BLOCKS+i*4+3] = (unsigned char) bitval_words[n*(PARALLEL_FEC_BLOCKS/4) + i].w;
        }
    }


    /* Free up memory */
    free(llr_words);
    free(bitval_words);

    return;


}

void mexFunction_free(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  mexPrintf("Hello World!\n");
  cuda_ldpc_destroy();

  return;

}

/* called as i = function(2 (uint8),Hvn, Hcn, num_edges, M, N, llr_map, row_idx, col_idx) */
void mexFunction_init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  mexPrintf("Hello World!\n");

  int *num_edges_ptr;
  int num_edges;
  int M, N, K;
  int *m_ptr, *n_ptr;
  int *llr_map;
  int *hb;
  int *hc;
  int *row_idx;
  int *col_idx;

  hb = (int *)mxGetData(prhs[0]);
  hc = (int *)mxGetData(prhs[1]);
  num_edges_ptr = (int *)mxGetData(prhs[2]);
  m_ptr = (int *)mxGetData(prhs[3]);
  n_ptr = (int *)mxGetData(prhs[4]);
  llr_map = (int *)mxGetData(prhs[5]);
  row_idx = (int *)mxGetData(prhs[6]);
  col_idx = (int *)mxGetData(prhs[7]);

  M = *m_ptr;
  N = *n_ptr;
  K = N-M;
  num_edges = *num_edges_ptr;



  mexPrintf("M: %d, N: %d, K: %d. Edges: %d\n", M, N, K, num_edges);
  mexPrintf("vn: %d, cn: %d\n", hb[11132], hc [11132]);

  mexPrintf("Loading...\n");
  cuda_ldpc_load((GPUEDGE *)hb, (GPUEDGE *)hc, num_edges, M, N, llr_map, NULL, row_idx, col_idx);

  mexPrintf("Done\n");


  /* Clean up */

}

void clean_up() {
    cuda_ldpc_destroy();

    /* MATLAB seems to crash without a cudaDeviceReset... */
    cudaDeviceReset();
    return;
}



