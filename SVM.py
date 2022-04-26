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

if len(sys.argv) != 4:
    print("Usage: SMV.py <train.csv> <test.csv> <labels.csv>");
    sys.exit(1)

# Order files correctly
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def train_SMV():
    
    # Define x => output and y => labels
    X_file_train = sys.argv[1]
    X_file_test = sys.argv[2]
    y_file = sys.argv[3]    
    
    # Make labels to correct format
    labels = pd.read_csv(y_file, header = None)
    
    # Get gender
    labels = labels.values[:,1].ravel()
    
    # Read images from Faces folder
    images = list()
    cwd = os.getcwd()
    filelist = os.listdir(cwd + "/data/Faces")
    filelist = sorted_alphanumeric(filelist)
    nimg = len(filelist)
    
    # Outcomment below if not on subset
    nimg = 200
    
    train, test = train_test_split([i for i in range(nimg)], test_size=0.20, random_state=3872324)
   
    # Make input to correct format
    X_train = pd.read_csv(X_file_train, header = None)
    X_train = X_train.values
    X_test = pd.read_csv(X_file_test, header = None)
    X_test = X_test.values
    
    # Load input file which is the output from dimensionality reduction method.
    # SVM take as input two arrays: an array X of shape (n_samples, n_features) 
    # holding the training samples, and an array y of class labels (strings or
    # integers), of shape (n_samples).
    clf = SVC()
    clf.fit(X_train, labels[train])
    y_pred = clf.predict(X_test)
    return labels[test], y_pred

y_test, y_pred = train_SMV()

# Get accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
