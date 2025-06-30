CC=gcc
CFLAGS1= -g -O3 -Wall -pthread -march=native -I/usr/local/include -I/usr/local/include/src -I/usr/local/cfitsio-3.47/include/ -I/home/ubuntu/proj/dsa110-shell/dsa110-sigproc
LIBS =  -L/usr/local/lib -lpsrdada -L/usr/lib/gcc/x86_64-linux-gnu/5 -lgfortran -L/usr/local/cuda/lib64 -lcudart -lcublas -lm -L/usr/local/cfitsio-3.47/lib -lcfitsio -lsigproc -lxgpu
dsaX_nsfrb_test: dsaX_nsfrb_test.c $(CDEPS1)
	$(CC) -o $@ $< $(CFLAGS1) $(LIBS)
	
dada_setup:
	#dada_db -k caba -b 14899200 -n 8
	dada_db -k caea -b 28606464 -n 300
	#dada_db -k caea -b 143032320 -n 300
	#dada_db -k cafa -b 28606464 -n 300
dada_shutdown:
	#dada_db -k caba -d
	dada_db -k caea -d
	#dada_db -k cafa -d

clean:
	rm dsaX_nsfrb_test
	
	

