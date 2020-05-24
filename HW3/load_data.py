
import gzip
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pickle


#import data from online mnist database;
import ssl
print("hello")
ssl._create_default_https_context = ssl._create_unverified_context
(X_train,y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

def ploting_some_examples():
    for i in range(3):
        img = X_train[i]
        plt.imshow(img, cmap="Greys")
        plt.show()

num_train = 20000
num_test = 2000

X_train = X_train[:num_train]
X_test = X_train[:num_test]
y_train = y_train[:num_train]
y_test = y_test[:num_test]

y_train = np.resize(y_train,(num_train,1))
y_test = np.resize(y_test,(num_test,1))

img_train = X_train[0].flatten()
for i in range(1, len(X_train)):
    img = X_train[i].flatten()
    img_train = np.vstack((img_train,img))

img_test = X_test[0].flatten()
for i in range(1, len(X_test)):
    img = X_test[i].flatten()
    img_test = np.vstack((img_test,img))
# add 1 as bias
img_train = np.insert(img_train, 0, 1, axis=1)
img_test = np.insert(img_test, 0, 1, axis=1)
# normalization and get rid of 1s and 0s
fac = 0.99/255
img_train_N = np.asfarray(img_train[:, 1:]) * fac + 0.01
img_test_N = np.asfarray(img_test[:, 1:]) * fac + 0.01

train_labels = np.asfarray(img_train_N[:, :1])  #testing

img_train_N = np.insert(img_train_N, 0, 1, axis=1)
img_test_N = np.insert(img_test_N, 0, 1, axis=1)

#one-hot representation for the labels
lr = np.arange(10)
for label in range(10):
    one_hot = (lr==label).astype(np.int)
    print("label: ", label, "fs ", one_hot)

train_labels_one_hot = (lr== y_train).astype(np.float)
test_labels_one_hot = (lr== y_test).astype(np.float)
# get rid of 1s and zeros with 0.01 and 0.99
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99


with open("data.pkl", "bw") as fh:
    data = (img_train_N,
            img_test_N,
            y_train,
            y_test,
            train_labels_one_hot,
            test_labels_one_hot,
            img_train,
            img_test)
    pickle.dump(data, fh)

# with gzip.open("t10k-images-idx3-ubyte.gz", 'rb') as f:
#     images_test = f.read()
#
# with gzip.open("train-images-idx3-ubyte.gz", 'rb') as f:
#     images_train = f.read()
#
# with gzip.open("t10k-labels-idx1-ubyte.gz", 'rb') as f:
#     labels_test = f.read()
#
# with gzip.open("train-labels-idx1-ubyte.gz", 'rb') as f:
#     labels_train = f.read()

