###################################################################################################################
#Function that takes as input the output from dimensionality reduction method, trains and SMV and returns accuracy#
###################################################################################################################

# Load libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import sys, os, re
import imageio as iio

if len(sys.argv) != 2:
    print("Usage: SMV.py <labels.csv>");
    sys.exit(1)

# Order files correctly
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def train_SMV():
    
    # Define yy => labels
    y_file = sys.argv[1]    
    
    # Make labels to correct format
    labels = pd.read_csv(y_file, header = None)
    
    # Get gender
    labels = labels.values[:,1].ravel()
    
    # Read images from Faces folder
    images = list()
    cwd = os.getcwd()
    filelist = os.listdir(cwd + "/data/Faces")
    filelist = sorted_alphanumeric(filelist)
    filelist = filelist
    nimg = len(filelist)
 
    train, test = train_test_split([i for i in range(nimg)], test_size=0.20, random_state=3872324)
    
    for file in filelist:
         im = iio.imread(cwd + "/data/Faces/" + file)
         images.append(im)
    
    imgTrain = np.array([images[i] for i in train])
    imgTest = np.array([images[i] for i in test])
    imgTrain = np.reshape(imgTrain, (len(train), -1))
    imgTest = np.reshape(imgTest, (len(test), -1))    
    
    # Load input file which is the output from dimensionality reduction method.
    # SVM take as input two arrays: an array X of shape (n_samples, n_features) 
    # holding the training samples, and an array y of class labels (strings or
    # integers), of shape (n_samples).
    clf = SVC()
    clf.fit(imgTrain, labels[train])
    y_pred = clf.predict(imgTest)
    return labels[test], y_pred

y_test, y_pred = train_SMV()

# Get accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
