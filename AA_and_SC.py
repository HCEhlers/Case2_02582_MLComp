#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import py_pcha
from joblib import dump, load


# In[2]:


############
###Labels###
#age: from 1 to 116
#gender: 0-male; 1-female
#race: 0-white; 1-black; 2-Asian; 3-Indian; 4-others(like Hispanic, Latino, Middle Eastern)
############

labels = pd.read_csv('./data/labels.csv', header = None)
labels = labels.values
print(labels.shape)
print(labels)

landmarks = pd.read_csv('labels_and_landmarks.csv')
landmarks_only = landmarks[[str(i) for i in range(1,137)]]
print(landmarks)


# In[3]:


print("labels of ages: {}".format(np.unique(labels[:, 0])))
print("labels of genders: {}".format(np.unique(labels[:, 1])))
print("labels of races: {}".format(np.unique(labels[:, -1])))


# In[4]:


# read images from Faces folder
images = list()

filelist = glob.glob('./data/Faces/*.jpg')
for file in sorted(filelist, key=lambda s: int(s.strip(string.ascii_letters + "\\./"))):
    im = iio.imread(file)
    images.append(im)
images = np.array(images)
print(images.shape)


# In[5]:


LANDMARK_SIZE = 20

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
    data = np.memmap("landmark_memory.dat", dtype='float64', mode="w+", shape=(len(X),len(landmarks[0])//2,2*LANDMARK_SIZE,2*LANDMARK_SIZE,3))
    for i in range(len(X)):
        lm = get_landmarks(X, i, landmarks[i,:])
        if lm is not None:
            np.copyto(data[i], lm)
        if i%100 == 0:
            data.flush()
    return data


# In[6]:


def landmarks_all_reuse(X, landmarks):
    return np.memmap("landmark_memory.dat", dtype='float64', mode="r", shape=(len(X),len(landmarks[0])//2,2*LANDMARK_SIZE,2*LANDMARK_SIZE,3))


# In[7]:


imgs = []
for piece in get_landmarks(images, 50, [40,60,80,100,170,20]):
    print(np.shape(piece))
    imgs.append(Image.fromarray(piece))

display(*imgs)


# In[8]:


# # USE ONLY IF the file "landmark_memory.dat" already exists
# image_all_pieces = landmarks_all_reuse(images, landmarks_only.to_numpy())
# print(np.shape(image_all_pieces))


# In[64]:


image_all_pieces = landmarks_all(images, landmarks_only.to_numpy())
print(np.shape(image_all_pieces))


# In[9]:


# reshape 
X = image_all_pieces.reshape(image_all_pieces.shape[0], -1)


# ### Archetypical Analysis

# In[14]:


n_components = 25 # num components


# In[15]:


XC, S, C, SSE, varexpl = py_pcha.PCHA(X.T, noc=n_components, delta=0.1)
'''
    Output
    ------
    XC : numpy.2darray
        I x noc feature matrix (i.e. XC=X[:,I]*C forming the archetypes)

    S : numpy.2darray
        noc X n matrix, S>=0 |S_j|_1=1

    C : numpy.2darray
        x x noc matrix, C>=0 |c_j|_1=1

    SSE : float
        Sum of Squared Errors
'''
X_hat = X.T @ C @ S
L = 0.5*np.linalg.norm(X.T-X_hat)**2


# In[28]:


np.savetxt("AA_output_XC.csv", XC, delimiter=",")
np.savetxt("AA_output_S.csv", S, delimiter=",")
np.savetxt("AA_output_C.csv", C, delimiter=",")
print('Sum of Squared Errors (SSE) for AA:', SSE)


# In[29]:


lbda = .01 # L1 regularization strength for the methods SC and NSC


# In[31]:


model = decomposition.DictionaryLearning(n_components=n_components, alpha=lbda, transform_alpha=lbda, max_iter=100, transform_max_iter=100, fit_algorithm='cd', transform_algorithm='lasso_cd')
X_transformed = model.fit_transform(X)
X_hat = X_transformed @ model.components_
L = 0.5*np.linalg.norm(X-X_hat)**2
components = model.components_


# In[3]:


np.savetxt("SC_output.csv", components, delimiter=",")
dump(model, 'sc.joblib')


# In[34]:


# Variance explained
SST = np.sum(X**2)
VE_AA = 1-2*L/SST 

print('AA variance explained:', VE_AA)


# In[35]:


# Variance explained
VE_SC = 1-2*L/SST 

print('SC variance explained:', VE_SC)

