#!/bin/bash
#create psrdada buffer

while true
do 
	dada_junkdb -b 14899200 -k caba -r 4.58 -t 3.25 ../realtime/hdrtest.txt
done
