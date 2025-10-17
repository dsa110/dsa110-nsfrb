#!/bin/bash
ss=$(seq 1 23 | shuf)

echo $ss
taskset --cpu-list 5-19 python run_training_test.py 0 1
for i in $ss; do
	#10 5 4 3 1 11 0 12 9 6 2 7 8; do
	#1 3 12 5 4 6 8 11 10 0 2 7; do
	#7 8 9 1 5 10 12 2 0 4 11 3 6; do
	echo $i
	ls /dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/model_weights_3Drev11_????-??-??T??:??:??.???.pth | xargs -n 1 taskset --cpu-list 5-19 python run_training_test.py $i 0 
done
