import time
import os

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


file_path = "x11display.png"
def close_event():
    plt.close()
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
            fig=plt.figure(figsize=figsize)
            timer = fig.canvas.new_timer(interval = 60000) #creating a timer object and setting an interval of 3000 milliseconds
            timer.add_callback(close_event)

            imgplot = plt.imshow(img)
            timer.start()
            plt.show()
        time.sleep(10)


if __name__=="__main__":
    main()
