from PIL import Image,ImageOps
import numpy as np
from scipy.stats import norm


import os


# This script converts png test images to np arrays and saves them to npy files after adding Gaussian noise #


dirname = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/simulations_and_classifications/src_examples/observation_1/images/"
fs = os.listdir(dirname)
for f in fs:
	if ".png" in f:
		#load data
		x = np.asarray(ImageOps.grayscale(Image.open(dirname + f)))
		
		#add axes
		y = np.zeros((32,32,25),dtype=np.float64)
		y[:,:,15] += x[25-16:25+16,25-16:25+16]
		
		#normalize to -128- 128 range
		y[:,:,15] -= 128

		#add noise
		y += norm.rvs(loc=0,scale=1,size=y.shape)

		#write to npy file
		np.save(dirname + f[:-4] + "_ID0000.npy",y)

		
