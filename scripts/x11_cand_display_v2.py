import time
import os

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


file_path = "x11display.png"

def main():
    last_modified = os.path.getmtime(file_path)
    while True:
        current_modified = os.path.getmtime(file_path)
        if current_modified != last_modified:
            print("File has changed!")
            last_modified = current_modified

            f = open("x11size.txt","r")
            scale = float(f.read())
            f.close()
            
            figsize=(40,40*scale)
            img = Image.open(file_path)
            plt.figure(figsize=figsize)
            imgplot = plt.imshow(img)
            plt.show()
        time.sleep(10)


if __name__=="__main__":
    main()
