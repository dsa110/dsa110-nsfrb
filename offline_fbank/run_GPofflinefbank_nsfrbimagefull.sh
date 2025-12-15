#!/bin/bash

cat GP_allnames2.txt | xargs -n 1 basename | xargs -n 1 taskset --cpu-list 5-35 python GPofflinefbank_nsfrbimagefull.py --num_time_samples 25 --gridsize 175 --nchans_per_node 8 --briggs --robust -2.0 --spacefilter --usefft --samenoise --cuda --usejax --SNRthresh 6 --noiseth 3 --initframes --initnoise --initnoisezero --cleanup --gulp_offset 0 --num_gulps 90 "$@" --mixedimage --GPdir

#cat GP_allnames2.txt | xargs -n 1 basename | xargs -n 1 taskset --cpu-list 5-20 python GPofflinefbank_nsfrbimagefull.py --num_time_samples 25 --gridsize 175 --nchans_per_node 8 --briggs --robust -2.0 --spacefilter --usefft --samenoise --cuda --usejax --SNRthresh 6 --noiseth 3 --initframes --initnoise --initnoisezero --cleanup --gulp_offset 0 --num_gulps 90 "$@" --mixedimage --GPdir

#cat GP_allnames2.txt | xargs -n 1 basename | xargs -n 1 taskset --cpu-list 5-20 python GPofflinefbank_nsfrbimagefull.py --num_time_samples 25 --gridsize 175 --nchans_per_node 8 --briggs --robust -2.0 --spacefilter --usefft --samenoise --cuda --usejax --SNRthresh 8.5 --noiseth 3 --initframes --initnoise --initnoisezero --cleanup --gulp_offset 0 --num_gulps 90 "$@" --overwritenoise --mixedimage --makeimage --GPdir

#taskset --cpu-list 5-20 python GPofflinefbank_nsfrbimagefull.py --num_time_samples 25 --gridsize 175 --nchans_per_node 8 --briggs --robust -2.0 --spacefilter --usefft --samenoise --cuda --usejax --SNRthresh 8.5 --noiseth 3 --initframes --initnoise --initnoisezero --cleanup --lockdev 1 --gulp_offset 0 --num_gulps 90 --mixedimage "$@" --GPdir GP_observations_2025-02-18T01:34:55.247 --makeimage

