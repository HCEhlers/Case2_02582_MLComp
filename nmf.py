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
from sklearn.model_selection import train_test_split
from joblib import dump, load

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

def landmarks_all(X, landmarks, memmap):
    if memmap is None:
        data = np.empty(dtype='float64', shape=(len(X),len(landmarks[0])//2,2*LANDMARK_SIZE,2*LANDMARK_SIZE,3))
    else:
        data = np.memmap(memmap, dtype='float64', mode="w+", shape=(len(X),len(landmarks[0])//2,2*LANDMARK_SIZE,2*LANDMARK_SIZE,3))
    for i in range(len(X)):
        lm = get_landmarks(X, i, landmarks[i,:])
        if lm is not None:
            np.copyto(data[i], lm)
        if memmap is not None and i%1000 == 0:
            data.flush()
    return data

def load_image_data(use_landmarks, on_hpc, subset):
    labelsUrl = '/work3/s200770/data/labels.csv' if on_hpc else './data/labels.csv'
    labels = pd.read_csv(labelsUrl, header = None)
    labels = labels.values

    # read images from Faces folder
    images = list()

    if on_hpc:
        filelist = glob.glob('/work3/s200770/data/Faces/*.jpg')
        filelist = [file[20:] for file in filelist]
    else:
        filelist = glob.glob('data/Faces/*.jpg')

    nimg = len(filelist)
    if subset:
        nimg = 200
    train, test = train_test_split([i for i in range(nimg)], test_size=0.20, random_state=3872324)

    for file in sorted(filelist, key=lambda s: int(s.strip(string.ascii_letters + "./"))):
        im = iio.imread(file)
        images.append(im)

    imgTrain = np.array([images[i] for i in train])
    imgTest = np.array([images[i] for i in test])

    if use_landmarks:
        landmarksUrl = '/work3/s200770/data/labels_and_landmarks.csv' if on_hpc else './labels_and_landmarks.csv'
        landmarks = pd.read_csv(landmarksUrl)
        landmarks_only = landmarks[[str(i) for i in range(1,137)]].to_numpy()

        imgTrain = landmarks_all(imgTrain, landmarks_only[train], None if on_hpc else 'mem_train.dat')
        imgTest = landmarks_all(imgTest, landmarks_only[test], None if on_hpc else 'mem_test.dat')

        imgTrain = np.reshape(imgTrain, (len(train), -1))
        imgTest = np.reshape(imgTest, (len(test), -1))
        return imgTrain, labels[train], imgTest, labels[test]
    else:
        imgTrain = np.reshape(imgTrain, (len(train), -1))
        imgTest = np.reshape(imgTest, (len(test), -1))
        return imgTrain, labels[train], imgTest, labels[test]

jid = 'LAPTOP'
use_landmarks = False
on_hpc = False
subset = False
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if arg == '--landmarks':
            use_landmarks = True
        elif arg == '--hpc':
            on_hpc = True
        elif arg == '--subset':
            subset = True
        else:
            jid = int(arg)

print("Loading images for job ", jid)

Xtrain, ytrain, Xtest, ytest = load_image_data(use_landmarks, on_hpc, subset)

print(np.shape(Xtrain), np.shape(ytrain), np.shape(Xtest), np.shape(ytest))

tol = 1e-4
if subset:
    tol = 1e-2

print("Starting NMF with tolerance", tol)

model = decomposition.NMF(n_components=128, init='random', random_state=3879324, tol=1e-2, max_iter=5000, verbose=2)
model.fit(Xtrain)

print("Done fitting")

name = jid + '_RAW_'
if use_landmarks:
    name = jid + '_LANDMARK_'

XtrainNMF = model.transform(Xtrain)
np.savetxt('NMF_' + name + '_TRAIN.csv', XtrainNMF, delimiter=",")
XtrainNMF = None

XtestNMF = model.transform(Xtest)
np.savetxt('NMF_' + name + '_TEST.csv', XtestNMF, delimiter=",")
