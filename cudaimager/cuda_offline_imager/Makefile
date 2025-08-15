cuda_imager: cuda_imager.cu cuda_imager.h
	nvcc cuda_imager.cu -o cuda_imager -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -lcufft


clean:
	rm -f cuda_imager

all: cuda_imager




