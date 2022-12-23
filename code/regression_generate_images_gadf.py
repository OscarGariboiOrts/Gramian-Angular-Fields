import numpy as np
import sys
import os
from shutil import copyfile, move

# path to the files containing the anomalous diffusing trajectories, created using the andi-datasets python package
task2 = 'task1.txt' 
ref2 = 'ref1.txt' 

# we read the file and convert to a numpy array
with open(task1, "r") as f:
    task = [np.array(x.split(";"), dtype="float32") for x in f.readlines()]
    task = np.array([x[1:] for x in task]) 
    task = task[:,]

# left padding with 0 to make all trajectories of same length (50)
max_length = 50
task = seqq.pad_sequences(task, maxlen=max_length, dtype="float32")

# instantiate a Gramian Angular Field with the difference method to convert the trajectories into images
gadf = GramianAngularField(image_size=48, method='difference')
X_gadf = gadf.fit_transform(task)

len_task = len(task)

# check that the directory to save the images exists or create it
if not os.path.isdir('regression_images_gadf'):
    os.makedirs('regression_images_gadf', exist_ok=True)

# iterate over the trajectories to create the images
for i in range(len_task):
    print(f'\r{i}/{len_task} - {round(i/len_task*100, 2)}%', end='')
    plt.imshow(X_gadf[i], extent=[0, 1, 0, 1], cmap = 'coolwarm', aspect = 'auto',vmax=abs(X_gadf[i]).max(), vmin=-abs(X_gadf[i]).max())
    plt.savefig('regression_images_gadf/X_gadf_{}.jpg'.format(i), bbox_inches='tight')
    plt.close("all")

print("Program finished!")
