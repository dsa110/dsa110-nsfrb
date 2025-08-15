# This is set up for the CORR containers

CC=gcc
CFLAGS1 = -g -O3 -Wall -pthread -march=native -I/usr/local/include -I/usr/local/include/src -I/usr/local/cfitsio-3.47/include/ -I/home/ubuntu/proj/dsa110-shell/dsa110-sigproc
CDEPS1=dsaX_def.h offlinefbank_nsfrbimage.h #dsaX_capture_manythread.h
LIBS = -L/usr/local/lib -lpsrdada -ldedisp -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran -L/usr/local/cuda/lib64 -lcudart -lcublas -lm -L/usr/local/cfitsio-3.47/lib -lcfitsio -lxgpu -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lfftw3f -I/usr/local/include/ -L/usr/local/lib/ -lfftw3 -L/usr/lib/x86_64-linux-gnu/ -I/usr/include/ -llapack -llapacke -lblas -lc



.DEFAULT_GOAL := all


offlinefbank_nsfrbimage.o: offlinefbank_nsfrbimage.c $(CDEPS1)
	$(CC) -c -o $@ $< $(CFLAGS1)

offlinefbank_nsfrbimage: offlinefbank_nsfrbimage.o
	$(CC) -o $@ $^ $(CFLAGS1) $(LIBS)

.PHONY: clean all

clean:
	rm -f *.o *~ offlinefbank_nsfrbimage

all: offlinefbank_nsfrbimage




