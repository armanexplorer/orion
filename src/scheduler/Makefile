CC=g++
NVCC=/usr/local/cuda/bin/nvcc
CFLAGS=-O3
CUDAINCLUDE=/usr/local/cuda/include/
CUDALIB=/usr/local/cuda/lib64
HOME_DIR=$(shell echo $$HOME)
CUDNNINCLUDE=$(HOME_DIR)/.local/lib/python3.10/site-packages/nvidia/cudnn/include
CUBLASINCLUDE=$(HOME_DIR)/.local/lib/python3.10/site-packages/nvidia/cublas/include
CUDNNLIB=$(HOME_DIR)/.local/lib/python3.10/site-packages/nvidia/cudnn/lib
CUBLASLIB=$(HOME_DIR)/.local/lib/python3.10/site-packages/nvidia/cublas/lib

utils_sched.o: utils_sched.cpp
	$(NVCC) $(CFLAGS) -Xcompiler  -fPIC -x cu -shared -c utils_sched.cpp -I$(CUDAINCLUDE) -I$(CUDNNINCLUDE) -I$(CUBLASINCLUDE) -L$(CUDALIB) -L$(CUDNNLIB) -L$(CUBLASLIB) -lcudart -l:libcudnn.so.9 -l:libcublas.so.12 -l:libcublasLt.so.12 -lpthread

scheduler.o: scheduler.cpp
	$(NVCC) $(CFLAGS) -Xcompiler  -fPIC  -x cu -shared -c scheduler.cpp -I$(CUDAINCLUDE) -I$(CUDNNINCLUDE) -I$(CUBLASINCLUDE) -L$(CUDALIB) -L$(CUDNNLIB) -L$(CUBLASLIB) -lcudart -l:libcudnn.so.9 -l:libcublas.so.12 -l:libcublasLt.so.12 -lpthread

scheduler_eval.o: scheduler_eval.cpp
	$(NVCC) $(CFLAGS) -Xcompiler  -fPIC  -x cu -shared -c scheduler_eval.cpp -I$(CUDAINCLUDE) -I$(CUDNNINCLUDE) -I$(CUBLASINCLUDE) -L$(CUDALIB) -L$(CUDNNLIB) -L$(CUBLASLIB) -lcudart -l:libcudnn.so.9 -l:libcublas.so.12 -l:libcublasLt.so.12 -lpthread

scheduler_eval.so: scheduler_eval.o utils_sched.o
	 $(CC) $(CFLAGS) -fPIC -shared utils_sched.o scheduler_eval.o -I$(CUDAINCLUDE) -I$(CUDNNINCLUDE) -I$(CUBLASINCLUDE) -L$(CUDALIB) -L$(CUDNNLIB) -L$(CUBLASLIB) -lcudart -lpthread -l:libcudnn.so.9 -l:libcublas.so.12 -l:libcublasLt.so.12 -o scheduler_eval.so

all:
	make scheduler_eval.so

clean:
	rm -rf *.o *.so
