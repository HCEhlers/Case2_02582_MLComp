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
    if memmap:
        data = np.memmap("landmark_memory.dat", dtype='float64', mode="w+", shape=(len(X),len(landmarks[0])//2,2*LANDMARK_SIZE,2*LANDMARK_SIZE,3))
    else:
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

def load_image_data(use_landmarks, on_hpc):
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
    for file in sorted(filelist, key=lambda s: int(s.strip(string.ascii_letters + "./"))):
        im = iio.imread(file)
        images.append(im)
    nimg = len(images)
    images = np.array(images)

    if use_landmarks:
        landmarksUrl = '/work3/s200770/data/labels_and_landmarks.csv' if on_hpc else './labels_and_landmarks.csv'
        landmarks = pd.read_csv(landmarksUrl)
        landmarks_only = landmarks[[str(i) for i in range(1,137)]].to_numpy()

        image_all_pieces = landmarks_all(images, landmarks_only, not on_hpc)
        return np.reshape(image_all_pieces, (nimg, -1)), labels
    else:
        return np.reshape(images, (nimg, -1)), labels

jid = 'LAPTOP'
use_landmarks = False
on_hpc = False
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if arg == '--landmarks':
            use_landmarks = True
        elif arg == '--hpc':
            on_hpc = True
        else:
            jid = int(arg)

data, labels = load_image_data(use_landmarks, on_hpc)

print(np.shape(data))
#
#print("Starting NMF")
#
#model = decomposition.NMF(n_components=128, init='random', random_state=0, tol=1e-4, max_iter=5000, verbose=2)
#NMFimgs = model.fit_transform(np.reshape(image_all_pieces, (23705,68*30*30*3)))
#
#dump(model, 'NMF_' + jid + '.joblib')
#np.savetxt('NMF_' + jid + '.csv', NMFimgs, delimiter=",")
