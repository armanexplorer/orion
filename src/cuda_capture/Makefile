CC=g++
CUDAINCLUDE=/usr/local/cuda/include/
CUDALIB=/usr/local/cuda/lib64
HOME_DIR=$(shell echo $$HOME)
CUDNNINCLUDE=$(HOME_DIR)/.local/lib/python3.10/site-packages/nvidia/cudnn/include
CUBLASINCLUDE=$(HOME_DIR)/.local/lib/python3.10/site-packages/nvidia/cublas/include
CUDNNLIB=$(HOME_DIR)/.local/lib/python3.10/site-packages/nvidia/cudnn/lib
CUBLASLIB=$(HOME_DIR)/.local/lib/python3.10/site-packages/nvidia/cublas/lib

libinttemp.so: utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp
	$(CC) -O3 -fPIC -shared utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp -I$(CUDAINCLUDE) -I$(CUDNNINCLUDE) -I$(CUBLASINCLUDE) -L$(CUDALIB) -L$(CUDNNLIB) -L$(CUBLASLIB) -lcudart -l:libcudnn.so.9 -l:libcublas.so.12 -l:libcublasLt.so.12 -lpthread -o libinttemp.so

all:
	make libinttemp.so

clean:
	rm -rf *.o libinttemp.so
