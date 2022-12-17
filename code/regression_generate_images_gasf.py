import time
from tensorflow.keras.preprocessing import sequence as seqq

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField

import numpy as np
import sys
import os
from shutil import copyfile, move

# path to the files containing the anomalous diffusing trajectories, created using the andi-datasets python package
task2 = 'task1.txt' 
ref2 = 'ref1.txt' 

with open(task1, "r") as f:
    task = [np.array(x.split(";"), dtype="float32") for x in f.readlines()]
    task = np.array([x[1:] for x in task]) 
    task = task[:,]

max_length = 50
task = seqq.pad_sequences(task, maxlen=max_length, dtype="float32")

gasf = GramianAngularField(image_size=48, method='summation')
X_gasf = gasf.fit_transform(task)

len_task = len(task)

if not os.path.isdir('regression_images_gasf'):
    os.makedirs('regression_images_gasf', exist_ok=True)

for i in range(len_task):
    print(f'\r{i}/{len_task} - {round(i/len_task*100, 2)}%', end='')
    plt.imshow(X_gasf[i], extent=[0, 1, 0, 1], cmap = 'coolwarm', aspect = 'auto',vmax=abs(X_gasf[i]).max(), vmin=-abs(X_gasf[i]).max())
    plt.savefig('regression_images_gasf/X_gasf_{}.jpg'.format(i), bbox_inches='tight')
    plt.close("all")

print("Program finished!")
