#!/bin/bash

while true
do
	for i in $(seq 1 10); do
		python noise_monitor.py
	done

	ls -t /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-noise/*long_term_noise* | tail -5 | xargs -n 1 rm 
done
