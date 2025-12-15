import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
torch.set_default_device("cpu")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from nsfrb import classifying_with_time
from nsfrb import classifying_two_stage


#load training set

import glob

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []



suff = "3Drev12"
cutsize = 50
maxset = 80
if len(sys.argv)>1:
    modelfile = sys.argv[1]
    print("Using previous model file:",modelfile)
else:
    modelfile = ""

fset1_fs = np.genfromtxt("train_shuffled.txt",dtype=str)[-maxset:]
#fset1_fs = np.sort(glob.glob("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/dataset_*/*.npy"))[maxset*setidx:maxset*(setidx+1)]

vsetidxs = np.arange(len(fset1_fs)) #np.array(list(set(np.arange(len(fset1_fs),dtype=int)) - set(tsetidxs)))
vset_size = len(vsetidxs)

vset1 = np.zeros((vset_size,16,25,cutsize,cutsize),dtype=np.float32)
for i in range(vset_size):
    d = np.load(fset1_fs[vsetidxs[i]])
    mx,my= np.unravel_index(np.argmax(np.nanmean(d - np.nanmedian(d,2,keepdims=True),(2,3))),d.shape[:2])
    minx = max(0,mx-cutsize//2)
    maxx = min(minx+cutsize,d.shape[0])
    minx -= (cutsize- (maxx-minx))
    miny = max(0,my-cutsize//2)
    maxy = min(miny+cutsize,d.shape[1])
    miny -= (cutsize - (maxy-miny))
    print(minx,maxx,miny,maxy)
    vset1[i,:,:,:,:] = (d[minx:maxx,miny:maxy,:,:] - np.nanmedian(d[minx:maxx,miny:maxy,:,:],2,keepdims=True)).transpose((3,2,0,1))#classifying_with_time.resize_cube(d - np.nanmedian(d,2,keepdims=True),size=(cutsize,cutsize)).transpose((3,2,0,1))
    vset1[i,:,:,:,:] -= np.nanmean(vset1[i,:,:,:,:])
    vset1[i,:,:,:,:] /= np.nanstd(vset1[i,:,:,:,:])


vset1 -= np.nanmin(vset1)
vset1 *= (2/np.nanmax(vset1))
vset1 -= 1

vset1[np.isnan(vset1)] = 0

print("Validation Set: ",vset_size)
np.save("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/full_validation_reduced.npy",vset1)
"""
vset1=np.load("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/full_validation_reduced.npy")
vset1_labels= np.load("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/full_validation_labels.npy")
print(vset1.shape,vset1_labels.shape)
vset_size = len(vset1_labels)
"""
vset1_labels = []
for i in range(vset_size):
    vset1_labels.append("RFI" in fset1_fs[vsetidxs[i]])
vset1_labels = np.array(vset1_labels)

np.save("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/full_validation_labels.npy",vset1_labels)

print("Normalizing...")
for i in range(vset_size):
    vset1[i,:,:,:,:] -= np.nanmean(vset1[i,:,:,:,:])
    vset1[i,:,:,:,:] /= (np.nanstd(vset1[i,:,:,:,:]) + 1e-8)
print("done")

#train --> adapted from https://github.com/dsa110/dsa110-nsfrb/blob/main/simulations_and_classifications/rfi_classification_pytorch.ipynb
model = classifying_with_time.Enhanced3DCNN()#classifying_two_stage.CombinedCNN()
if len(modelfile)>0:
    model.load_state_dict(torch.load(modelfile, weights_only=True, map_location=torch.device('cpu')))
criterion = nn.BCEWithLogitsLoss()



vset1_bsize = vset_size #226
vsetloader = torch.utils.data.DataLoader(vset1,batch_size=vset1_bsize)
#validation set

model.eval()
val_running_loss = 0.0
correct_val = 0
total_val = 0
with torch.no_grad():
    for i,data in enumerate(vsetloader,0):#range(tset_size):
        print(i)
        inputs,labels = data,torch.from_numpy(vset1_labels[i*vset1_bsize:(i+1)*vset1_bsize])
        outputs = torch.sigmoid(model(inputs).squeeze())
        loss = criterion(outputs, labels.float())
        val_running_loss += loss.item()


        predicted = (outputs > 0.5).float()
        print(outputs)
        print(predicted)
        print(labels)
        correct_val += (predicted == labels.float()).sum().item()
        total_val += labels.size(0)




accuracy_val = 100 * correct_val / total_val
print(f"Validation - Full Set; Loss: {val_running_loss/len(vsetloader)}, Accuracy: {accuracy_val}%")
val_losses.append(val_running_loss/len(vsetloader))
val_accuracies.append(accuracy_val)

print('Finished Training')

np.save("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/full_validation_lossses_"+suff+".npy",val_losses)
np.save("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/full_validation_accuracies_"+suff+".npy",val_accuracies)
