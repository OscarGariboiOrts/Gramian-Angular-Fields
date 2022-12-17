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
task2 = 'task2.txt' 
ref2 = 'ref2.txt' 

with open(task2, "r") as f:
    task = [np.array(x.split(";"), dtype="float32") for x in f.readlines()]
    task = np.array([x[1:] for x in task]) 
    task = task[:,]

max_length = 50
task = seqq.pad_sequences(task, maxlen=max_length, dtype="float32")

with open(ref2, 'r') as f:
    ref = [float(x.split(';')[1].strip()) for x in f.readlines()[:]]
    ref_cat = to_categorical(ref)

gasf = GramianAngularField(image_size=48, method='summation')
X_gasf = gasf.fit_transform(task)
#gadf = GramianAngularField(image_size=48, method='difference')
#X_gadf = gadf.fit_transform(task)

len_task = len(task)

if not os.path.isdir('images_gasf'):
    os.makedirs('images_gasf', exist_ok=True)

for i in range(len_task):
    print(f'\r{i}/{len_task} - {round(i/len_task*100, 2)}%', end='')
    plt.imshow(X_gasf[i], extent=[0, 1, 0, 1], cmap = 'coolwarm', aspect = 'auto',vmax=abs(X_gasf[i]).max(), vmin=-abs(X_gasf[i]).max())
    plt.savefig('images_gasf/X_gasf_{}.jpg'.format(i), bbox_inches='tight')
    plt.close("all")


dataset_home = 'images_gasf/'
labeldirs = ['class1/', 'class2/', 'class3/', 'class4/', 'class5/']
for labldir in labeldirs:
    if not os.path.isdir(labldir):
        newdir = dataset_home + labldir
        os.makedirs(newdir, exist_ok=True)

src_directory = 'images_gasf/'
for file in [x for x in os.listdir(src_directory) if not x.startswith("class")]:
    src = src_directory + '/' + file
    dst_dir = 'images_gasf/'
    file_number_id = int(file.split(".")[0].split("_")[-1])
    file_number = int(ref[file_number_id])
    if file_number == 0:
        dst = dst_dir + "class1/" + file
        move(src, dst)
    elif file_number == 1:
        dst = dst_dir + "class2/" + file
        move(src, dst)
    elif file_number == 2:
        dst = dst_dir + "class3/" + file
        move(src, dst)
    elif file_number == 3:
        dst = dst_dir + "class4/" + file
        move(src, dst)
    elif file_number == 4:
        dst = dst_dir + "class5/" + file
        move(src, dst)

print("Program finished!")
