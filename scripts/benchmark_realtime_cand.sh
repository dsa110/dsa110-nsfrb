#!/bin/bash
#create psrdada buffer

while true
do 
	dada_junkdb -b 144961600 -k cada -r 43.2 -t 3.355 ../realtime/hdrtest.txt -g -m 0 -s 10
	dada_junkdb -b 28992320 -k caea -r 8.64 -t 3.355 ../realtime/hdrtest.txt -g -m 0 -s 10
	dada_junkdb -b 28992320 -k cafa -r 8.64 -t 3.355 ../realtime/hdrtest.txt -g -m 0 -s 10
done
