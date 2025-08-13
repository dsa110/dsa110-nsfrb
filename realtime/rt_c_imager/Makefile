# This is set up for the CORR containers

CC=gcc
CFLAGS1 = -g -O3 -Wall -pthread -march=native -I/usr/local/include -I/usr/local/include/src -I/usr/local/cfitsio-3.47/include/ -I/home/ubuntu/proj/dsa110-shell/dsa110-sigproc
CDEPS1=dsaX_def.h dsaX_nsfrbimage.h #dsaX_capture_manythread.h
CDEPS2=dsaX_def.h dsaX_capture.h
LIBS = -L/usr/local/lib -lpsrdada -ldedisp -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran -L/usr/local/cuda/lib64 -lcudart -lcublas -lm -L/usr/local/cfitsio-3.47/lib -lcfitsio -lxgpu -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lfftw3f -I/usr/local/include/ -L/usr/local/lib/ -lfftw3 -L/usr/lib/x86_64-linux-gnu/ -I/usr/include/ -llapack -llapacke -lblas
#LIBS = -L/usr/local/lib -lpsrdada -ldedisp -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran -L/usr/local/cuda/lib64 -lcudart -lcublas -lm -L/usr/local/cfitsio-3.47/lib -lcfitsio -lsigproc -lxgpu -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc

#LIBS2 = -L/home/ubuntu/PF_RING/userland/libpcap-1.9.1 -lpcap
#CDEPS3=dsaX_def.h dsaX_capture_pcap.h

CCU=/usr/local/cuda/bin/nvcc -D CUDA -ccbin=g++
CFLAGS2 = -I/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src -I/home/ubuntu/proj/dsa110-shell/dsa110-xGPU/src -I/usr/local/include -I/usr/local/include/src -I/usr/local/cfitsio-3.47/include -I/home/ubuntu/proj/dsa110-shell/dsa110-sigproc -arch=sm_75 -O3 -Xcompiler="-pthread" -DMATRIX_ORDER_TRIANGULAR -std=c++14 -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I/usr/local/thrust-1.8.1


.DEFAULT_GOAL := all


dsaX_nsfrbimage.o: dsaX_nsfrbimage.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

dsaX_nsfrbimage: dsaX_nsfrbimage.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

.PHONY: clean all

clean:
	rm -f *.o *~ dsaX_nsfrbimage dsaX_capture 

all: dsaX_nsfrbimage




