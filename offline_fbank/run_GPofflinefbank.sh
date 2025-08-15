#!/bin/bash

#seq 0 15 | xargs -I {} taskset --cpu-list 5-11 python GPofflinefbank_nsfrbimage.py --sb {}    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 &
echo ${@}

#imaging fbank
taskset --cpu-list 5 python GPofflinefbank_nsfrbimage.py --sb 0    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 6 python GPofflinefbank_nsfrbimage.py --sb 1    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 7 python GPofflinefbank_nsfrbimage.py --sb 2    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 8 python GPofflinefbank_nsfrbimage.py --sb 3    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 9 python GPofflinefbank_nsfrbimage.py --sb 4    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 10 python GPofflinefbank_nsfrbimage.py --sb 5    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} & 
taskset --cpu-list 11 python GPofflinefbank_nsfrbimage.py --sb 6    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 12 python GPofflinefbank_nsfrbimage.py --sb 7    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 13 python GPofflinefbank_nsfrbimage.py --sb 8    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 14 python GPofflinefbank_nsfrbimage.py --sb 9    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 15 python GPofflinefbank_nsfrbimage.py --sb 10    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 16 python GPofflinefbank_nsfrbimage.py --sb 11    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 17 python GPofflinefbank_nsfrbimage.py --sb 12    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 18 python GPofflinefbank_nsfrbimage.py --sb 13    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 19 python GPofflinefbank_nsfrbimage.py --sb 14    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &
taskset --cpu-list 20 python GPofflinefbank_nsfrbimage.py --sb 15    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 ${@} &


taskset --cpu-list 3 python GPofflinefbank_nsfrbsearch.py    --verbose --num_time_samples 25 --gridsize 301 --nchans_per_node 8 --briggs --robust -2.0 --spacefilter --usefft --samenoise --cuda --usejax --SNRthresh 3 --noiseth 3 --initframes --initnoise --initnoisezero --cleanup --lockdev 0 ${@}
#pgrep -f GPofflinefbank_nsfrb | xargs -n 1 kill -9
