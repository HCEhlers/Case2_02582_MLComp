import sys
import pandas as pd
import numpy as np
import scipy
import scipy.misc
from PIL import Image
import imageio as iio
from pathlib import Path
import string
import glob
import os
from sklearn import decomposition
from joblib import dump, load

jid = sys.argv[1]

labels = pd.read_csv('/work3/s200770/data/labels.csv', header = None)
labels = labels.values
print(labels.shape)
print(labels)

landmarks = pd.read_csv('/work3/s200770/data/labels_and_landmarks.csv')
landmarks_only = landmarks[[str(i) for i in range(1,137)]]
print(landmarks)

print("labels of ages: {}".format(np.unique(labels[:, 0])))
print("labels of genders: {}".format(np.unique(labels[:, 1])))
print("labels of races: {}".format(np.unique(labels[:, -1])))

# read images from Faces folder
images = list()

filelist = glob.glob('/work3/s200770/data/Faces/*.jpg')
for file in sorted(filelist, key=lambda s: int(s[20:].strip(string.ascii_letters + "./"))):
    im = iio.imread(file)
    images.append(im)
images = np.array(images)
print(images.shape)

print("Finished reading raw images")

LANDMARK_SIZE = 15

def get_landmarks(X, imgNum, landmarks):
    pieces = []
    for i in range(len(landmarks)//2):
        lx = landmarks[2*i]
        ly = landmarks[2*i+1]
        if lx < LANDMARK_SIZE:
            lx = LANDMARK_SIZE
        if 200-LANDMARK_SIZE < lx:
            lx = 200-LANDMARK_SIZE
        if ly < LANDMARK_SIZE:
            ly = LANDMARK_SIZE
        if 200-LANDMARK_SIZE < ly:
            ly = 200-LANDMARK_SIZE
        piece = X[imgNum,lx-LANDMARK_SIZE:lx+LANDMARK_SIZE,ly-LANDMARK_SIZE:ly+LANDMARK_SIZE,:]
        pieces.append(piece)
    return np.array(pieces)

#def flattened_landmarks(X, imgNum, landmarks):
#    return np.array(get_landmarks(X, imgNum, landmarks)).flatten()

def landmarks_all(X, landmarks):
    #data = np.memmap("landmark_memory.dat", dtype='float64', mode="w+", shape=(len(X),len(landmarks[0])//2,2*LANDMARK_SIZE,2*LANDMARK_SIZE,3))
    data = np.empty(dtype='float64', shape=(len(X),len(landmarks[0])//2,2*LANDMARK_SIZE,2*LANDMARK_SIZE,3))
    for i in range(len(X)):
        lm = get_landmarks(X, i, landmarks[i,:])
        if lm is not None:
            np.copyto(data[i], lm)
        #if i%100 == 0:
        #    data.flush()
    return data

def landmarks_all_reuse(X, landmarks):
    return np.memmap("landmark_memory.dat", dtype='float64', mode="r", shape=(len(X),len(landmarks[0])//2,2*LANDMARK_SIZE,2*LANDMARK_SIZE,3))

image_all_pieces = landmarks_all(images, landmarks_only.to_numpy())
print(np.shape(image_all_pieces))

print("Starting ICA")

# Fit ICA
n_components = 1
model = decomposition.FastICA(n_components=n_components,algorithm='deflation')
ICAimgs = model.fit_transform(np.reshape(image_all_pieces, (23705,68*30*30*3)))

# Write to file
dump(model, 'ICA_' + jid + '.joblib')
np.savetxt('ICA_' + jid + '.csv', ICAimgs, delimiter=",")
