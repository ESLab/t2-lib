/*****************************************************************
    Simple LDPC encoder (CPU), written mostly to test the GPU decoder

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
#include <string.h>
#include "../cuda_ldpc/cuda_ldpc.h"

#define INPUT(x,y) (input[(y)*n_cw+(x)])
#define OUTPUT(x,y) (output[(y)*n_cw+(x)])


/* Input
  out = function(Hcn (int32), llr_map (int32), M (int32), N (int32), num_edges (int32), input (uint8), codewords (int32))
  */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int *hb;
    int *llr_map;
    unsigned char *input;

    mxArray *output_a;
    unsigned char *output;

    int *n_cw_ptr;
    int *n_ptr;
    int *m_ptr;
    int *num_edges_ptr;

    int N, M, K ,n_cw, num_edges;
    int edge = 0;
    int row = 0;
    int idx = 0;
    int row_start = 0;


    mexPrintf("Encoder starting...");

    hb = (int *)mxGetData(prhs[0]);
    llr_map = (int *)mxGetData(prhs[1]);
    m_ptr = (int *)mxGetData(prhs[2]);
    n_ptr = (int *)mxGetData(prhs[3]);
    num_edges_ptr = (int *)mxGetData(prhs[4]);

    input = (unsigned char *) mxGetData(prhs[5]);
    n_cw_ptr = (int *) mxGetData(prhs[6]);

    M = *m_ptr;
    N = *n_ptr;
    n_cw = *n_cw_ptr;
    num_edges = *num_edges_ptr;

    K = N-M;

    mexPrintf("M %d, N %d, nume %d, input[10] %d, ncw %d\n", M, N, num_edges, input[10], n_cw);


    output_a = plhs[0] = mxCreateNumericMatrix(n_cw, N, mxUINT8_CLASS, mxREAL);
    output = (unsigned char *) mxGetData(output_a);

    memset(output, 0, n_cw*N*sizeof(unsigned char));

    for (int i=0; i < K; i++) {
        for (int cw = 0; cw < n_cw; cw++) {
            OUTPUT(cw, i) = INPUT(cw, i);
        }
    }

    for (int cw = 0; cw < n_cw; cw++) {

        idx = 0;
        row = 0;
        while(idx < num_edges) {
            row_start = idx;
            OUTPUT(cw, K+row) = 0;
            while(hb[idx] != row_start) {
                OUTPUT(cw, K+row) ^= OUTPUT(cw, llr_map[idx]);
                idx++;
            }
            idx++;
            row++;
        }
    }

    mexPrintf("Done\n");
    return;
}
