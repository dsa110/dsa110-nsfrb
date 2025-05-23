#!/bin/bash
#create psrdada buffer

while true
do 
	dada_junkdb -b 14899200 -k caba -r 4.440 -t 3.355 ../realtime/hdrtest.txt
done
