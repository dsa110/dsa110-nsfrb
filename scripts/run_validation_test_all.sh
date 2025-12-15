#!/bin/bash
ls /dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/model_weights_3Drev12_????-??-??T??:??:??.???.pth | xargs -n 1 taskset --cpu-list 5-19 python run_validation_test.py 
