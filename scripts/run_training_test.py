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
#fset1_fs=glob.glob("/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-images/2025-08-31_PSRB0329+54/2025-08-31T12:4?:??.???_input.npy")
suff = "3Drev12"
cutsize = 50
maxset = 113
setidx = int(sys.argv[1])
init_weights = bool(sys.argv[2])
if len(sys.argv)>3:
    modelfile = sys.argv[3]
    print("Using previous model file:",modelfile)
else:
    modelfile = ""


#use pre-shuffled training/validation set
fset1_fs = np.genfromtxt("train_shuffled.txt",dtype=str)[maxset*setidx:maxset*(setidx+1)]
#fset1_fs = np.sort(glob.glob("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/dataset_*/*.npy"))[maxset*setidx:maxset*(setidx+1)]
#fset1_fs = np.sort(glob.glob("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/dataset_fpdata*/*.npy"))[maxset*setidx:maxset*(setidx+1)]
"""
tsetidxs = np.random.choice(np.arange(len(fset1_fs),dtype=int),3*len(fset1_fs)//4,replace=False)
tset_size = len(tsetidxs)
tset1 = np.zeros((tset_size,16,25,cutsize,cutsize),dtype=np.float32)
for i in range(tset_size):
    d = np.load(fset1_fs[tsetidxs[i]])
    mx,my= np.unravel_index(np.argmax(np.nanmean(d - np.nanmedian(d,2,keepdims=True),(2,3))),d.shape[:2])
    minx = max(0,mx-cutsize//2)
    maxx = min(minx+cutsize,d.shape[0])
    minx -= (cutsize- (maxx-minx))
    miny = max(0,my-cutsize//2)
    maxy = min(miny+cutsize,d.shape[1])
    miny -= (cutsize - (maxy-miny))
    print(minx,maxx,miny,maxy)
    tset1[i,:,:,:,:] = (d[minx:maxx,miny:maxy,:,:] - np.nanmedian(d[minx:maxx,miny:maxy,:,:],2,keepdims=True)).transpose((3,2,0,1)) #classifying_with_time.resize_cube(d - np.nanmedian(d,2,keepdims=True),size=(cutsize,cutsize)).transpose((3,2,0,1))
    tset1[i,:,:,:,:] -= np.nanmean(tset1[i,:,:,:,:])
    tset1[i,:,:,:,:] /= np.nanstd(tset1[i,:,:,:,:])
vsetidxs = np.array(list(set(np.arange(len(fset1_fs),dtype=int)) - set(tsetidxs)))
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

tset1 -= np.nanmin(tset1)
tset1 *= (2/np.nanmax(tset1))
tset1 -= 1

vset1 -= np.nanmin(vset1)
vset1 *= (2/np.nanmax(vset1))
vset1 -= 1

tset1[np.isnan(tset1)] = 0
vset1[np.isnan(vset1)] = 0

print("Training Set: ",tset_size)
print("Validation Set: ",vset_size)
np.save("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/training_reduced_"+str(setidx)+".npy",tset1)
np.save("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/validation_reduced_"+str(setidx)+".npy",vset1)

tset1_labels = []
vset1_labels = []
for i in range(tset_size):
    tset1_labels.append("RFI" in fset1_fs[tsetidxs[i]])
for i in range(vset_size):
    vset1_labels.append("RFI" in fset1_fs[vsetidxs[i]])
tset1_labels = np.array(tset1_labels)
vset1_labels = np.array(vset1_labels)

np.save("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/training_labels_"+str(setidx)+".npy",tset1_labels)
np.save("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/validation_labels_"+str(setidx)+".npy",vset1_labels)
"""
tset1 = np.load("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/training_reduced_"+str(setidx)+".npy")
vset1 = np.load("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/validation_reduced_"+str(setidx)+".npy")
tset1_labels = np.load("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/training_labels_"+str(setidx)+".npy")
vset1_labels = np.load("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/validation_labels_"+str(setidx)+".npy")
tset_size = tset1.shape[0]
vset_size = vset1.shape[0]
print(tset_size,vset_size)


#normalize
print("Normalizing...")
for i in range(tset_size):
    tset1[i,:,:,:,:] -= np.nanmean(tset1[i,:,:,:,:])
    tset1[i,:,:,:,:] /= (np.nanstd(tset1[i,:,:,:,:]) + 1e-8)
for i in range(vset_size):
    vset1[i,:,:,:,:] -= np.nanmean(vset1[i,:,:,:,:])
    vset1[i,:,:,:,:] /= (np.nanstd(vset1[i,:,:,:,:]) + 1e-8)
print("done")

#train --> adapted from https://github.com/dsa110/dsa110-nsfrb/blob/main/simulations_and_classifications/rfi_classification_pytorch.ipynb
import torch.optim as optim
nepochs = 40
model = classifying_with_time.Enhanced3DCNN()#classifying_two_stage.CombinedCNN()
if len(modelfile)>0:
    model.load_state_dict(torch.load(modelfile, weights_only=True))
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)
#optimizer1 = optim.Adam(model.spatcnn.parameters(), lr=0.001)
#optimizer2 = optim.Adam(model.speccnn.parameters(), lr=0.001)


#initialize weights
if init_weights:
    print("initializing weights")
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            print(m.weight)


train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

tset1_bsize = 4#tset_size
vset1_bsize = vset_size #226
tsetloader = torch.utils.data.DataLoader(tset1,batch_size=tset1_bsize)
vsetloader = torch.utils.data.DataLoader(vset1,batch_size=vset1_bsize)
for epoch in range(nepochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    tsetorder=np.random.choice(np.arange(tset1_bsize),tset1_bsize,replace=True)
    print(tsetorder)
    for i,data in enumerate(tsetloader,0):#range(tset_size):
        print(i)
        inputs,labels = data,torch.from_numpy(tset1_labels[i*tset1_bsize:(i+1)*tset1_bsize])
        
        inputs = inputs[tsetorder,...]
        labels=labels[tsetorder]

        #inputs = tset1[i:i+1,:,:,:,:]
        #labels = tset1_labels[i:i+1]
        optimizer.zero_grad()
        #optimizer1.zero_grad()
        #optimizer2.zero_grad()
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs.squeeze())


        print(torch.min(outputs),torch.max(outputs))
        loss= criterion(outputs,labels.float())
        loss.backward()


        #  Gradient Clipping
        totalnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        print("EPOCH ",epoch,"NORM-->",totalnorm)
        print([p.grad for p in model.parameters() if p.grad is not None])

        optimizer.step()
        #optimizer1.step()
        #optimizer2.step()
        running_loss += loss.item()


        predicted = (outputs > 0.5).float()
        correct_train += (predicted==labels.float()).sum().item()
        total_train += labels.size(0)

    accuracy_train = 100 * correct_train / total_train
    print(f"Training - Epoch {epoch+1}, Loss: {running_loss/len(tsetloader)}, Accuracy: {accuracy_train}%")
    train_losses.append(running_loss/len(tsetloader))
    train_accuracies.append(accuracy_train)


    #validation set

    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    vsetorder = np.random.choice(np.arange(vset1_bsize),vset1_bsize,replace=True)
    print(vsetorder)
    with torch.no_grad():
        for i,data in enumerate(vsetloader,0):#range(tset_size):
            print(i)
            inputs,labels = data,torch.from_numpy(vset1_labels[i*vset1_bsize:(i+1)*vset1_bsize])

            inputs = inputs[vsetorder,...]
            labels = labels[vsetorder]
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs.squeeze())

            print(torch.min(outputs),torch.max(outputs))
            loss = criterion(outputs, labels.float())
            val_running_loss += loss.item()


            predicted = (outputs > 0.5).float()
            correct_val += (predicted == labels.float()).sum().item()
            total_val += labels.size(0)

    accuracy_val = 100 * correct_val / total_val
    print(f"Validation - Epoch {epoch+1}, Loss: {val_running_loss/len(vsetloader)}, Accuracy: {accuracy_val}%")
    val_losses.append(val_running_loss/len(vsetloader))
    val_accuracies.append(accuracy_val)

print('Finished Training')

np.save("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/training_losses_"+str(setidx)+"_"+suff+".npy",train_losses)
np.save("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/validation_lossses_"+str(setidx)+"_"+suff+".npy",val_losses)
np.save("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/training_accuracies_"+str(setidx)+"_"+suff+".npy",train_accuracies)
np.save("/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/validation_accuracies_"+str(setidx)+"_"+suff+".npy",val_accuracies)

from astropy.time import Time

if len(modelfile)>0:
    torch.save(model.state_dict(),modelfile)
    print("Model weights saved: ",modelfile)
else:
    tstamp = Time.now().isot
    torch.save(model.state_dict(),"/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/model_weights_"+suff+"_"+tstamp+".pth")
    print("Model weights saved: ","/dataz/dsa110/nsfrb/dsa110-nsfrb-training/final_commissioning_trainingset/model_weights_"+suff+"_"+tstamp+".pth")
