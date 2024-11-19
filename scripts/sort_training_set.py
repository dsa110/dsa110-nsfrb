import csv
import os
import sys
import numpy as np

"""
This file sorts through training data and moves labelled data to the correct 
datasets.
"""
training_dir = os.environ['NSFRBDATA'] + "/dsa110-nsfrb-training/"
simulated_dir = training_dir + "simulated/"
data_dir = training_dir + "data/"
labels_simulated = training_dir + "simulated/labels.csv"
labels_data = training_dir + "data/labels.csv"


#sim data
with open(labels_simulated,"r") as csvfile:
    rdr = csv.reader(csvfile,delimiter=',')
    rem_idx = ""
    i = 1
    for row in rdr:
        name,label,end = row
        if label != '-1':
            #check label
            label = int(label)
            target_dir = training_dir + "/dataset/train_" + str("falsepositive/" if label==1 else "src/")
            print(name,label,target_dir)

            #move 
            os.system("mv " + simulated_dir + "*" + name + "* " + target_dir)

            #remove from label file
            rem_idx += str(i) + "d;"
        i += 1
if len(rem_idx) > 0:
    print(rem_idx[:-1])
    os.system("sed -i.bak -e '" + rem_idx[:-1] + "' " + labels_simulated)
        

            

