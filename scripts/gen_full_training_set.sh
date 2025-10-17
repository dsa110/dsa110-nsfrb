#!/bin/bash

for i in $(seq 1 3); do
	taskset --cpu-list 8-10 python gen_training_set.py --N 50 --verbose --gridsize 175 --briggs --robust -2 --bmin 20 --dec 54.0 --outputdir /dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/
	taskset --cpu-list 11-13 python gen_training_set.py --N 50 --verbose --gridsize 175 --briggs --robust -2 --bmin 20 --dec 21.0 --outputdir /dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/
	taskset --cpu-list 14-16 python gen_training_set.py --N 50 --verbose --gridsize 175 --briggs --robust -2 --bmin 20 --dec 0.0 --outputdir /dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/
	taskset --cpu-list 5-7 python gen_training_set.py --N 50 --verbose --gridsize 175 --briggs --robust -2 --bmin 20 --dec 71.6 --outputdir /dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/
done

#python gen_training_set.py --N 100 --verbose --gridsize 175 --briggs --robust -2 --bmin 20 --dec 71.6 --outputdir /dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/ --RFI
#python gen_training_set.py --N 100 --verbose --gridsize 175 --briggs --robust -2 --bmin 20 --dec 54.0 --outputdir /dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/ --RFI
#python gen_training_set.py --N 100 --verbose --gridsize 175 --briggs --robust -2 --bmin 20 --dec 21.0 --outputdir /dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/ --RFI
#python gen_training_set.py --N 100 --verbose --gridsize 175 --briggs --robust -2 --bmin 20 --dec 0.0 --outputdir /dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/ --RFI


