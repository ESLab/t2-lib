#This Makefile is a bit messy right now
#Try to change some of the paths to match your environment, and you should be OK :)

NVCC=nvcc
MEXCC=mex

NVARCH=compute_11
CUDALIB=-L/usr/local/cuda/lib64
LINK=-lcudart
HEADERS=cuda_ldpc/cuda_ldpc.h cuda_ldpc/helpers.h
MATLAB_INCLUDE=-I/opt/matlab_R2007a/extern/include/
CUDA_INCLUDE=-I$(HOME)/NVIDIA_GPU_Computing_SDK/C/common/inc/
INCLUDE=-I/usr/local/cuda/include $(MATLAB_INCLUDE) $(CUDA_INCLUDE)
CUSRC=cuda_ldpc/cuda_ldpc.cu
CUOBJ=$(CUSRC:%.cu=%.o)
DEFINES=-DUSE_MEX

MEXSRC=mex/mex_cuda_ldpc.c mex/mex_ldpc_enc.c
MEXOBJ=$(MEXSRC:%.c=%.mexa64)

CSRC=cuda_ldpc/helpers.c

%.o: %.cu $(HEADERS)
	$(NVCC) -o $@ $< -c -arch $(NVARCH) $(CUDALIB) $(DEFINES) $(INCLUDE) -lcuda -Xcompiler -fpic

%.mexa64: %.c $(HEADERS) $(CUOBJ)
	$(MEXCC) -cxx CC=g++ $(CUDALIB) $(DEFINES) $(LINK) $(INCLUDE) $(CUOBJ) $< $(CSRC) -o $@

all: $(CUOBJ) $(MEXOBJ)

clean:
	rm *.o cuda_ldpc/*.o mex/*.mexa64 mex/*.o	
