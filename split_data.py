#######################################
#Split data into train and test folder#
#######################################

# Load libraries
import os, shutil, re 
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

# Get data directory with images
cwd = os.getcwd()
data_path = cwd + "/data/Faces"
data = os.listdir(data_path)

# Order files correctly
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

data = sorted_alphanumeric(data)
#data = np.asarray(data)
# Get labels
labels = genfromtxt(cwd + '/data/labels.csv', delimiter=',')
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1, shuffle=False)

print(X_train[0:10], y_train[0:10])
print(X_test[0:10], y_test[0:10])
# Save labels to csv file as y_train.csv and y_test.csv
np.savetxt("data/y_train.csv", y_train, fmt='%i', delimiter=",")
np.savetxt("data/y_test.csv", y_test, fmt='%i', delimiter=",")

# Create test and train directory
train_path = cwd + "/data/" + "train/"
test_path = cwd + "/data/" + "test/"

try:
   os.mkdir(train_path)
   os.mkdir(test_path)
except FileExistsError:
   pass


# Move train files to train folder
for file_name in X_train:
   shutil.copy(os.path.join(data_path, file_name), train_path)


# Move test files to test folder
for file_name in X_test:
   shutil.copy(os.path.join(data_path, file_name), test_path)

print("Size train data: ", len(X_train), "\nSize test data: ", len(X_test))
print(len(data))
