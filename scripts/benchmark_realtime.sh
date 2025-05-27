#!/bin/bash
#create psrdada buffer
#dada_junkdb -b 14899200 -k caba -r 4.440 -t 3.355 -g -m 0 -s 10 ../realtime/hdrtest.txt
while true
do 
	dada_junkdb -b 14899200 -k caba -r 4.440 -t 3.355 -g -m 0 -s 0.1 ../realtime/hdrtest.txt 
done
