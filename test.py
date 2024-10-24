import tensorflow
from keras.applications.resnet50 import ResNet50, preprocess_input 
from keras.layers import MaxPooling2D 
import cv2
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from sklearn.neighbors import NearestNeighbors

feature_list = np.array(pickle.load(open("featurevector.pkl","rb")))
filename = pickle.load(open("filenames.pkl", "rb"))

model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([model, MaxPooling2D()])
model.summary()

img = cv2.imread(r'D:\Real Estate\E_commerce_Product\Dataset\1636.jpg')
img = cv2.resize(img, (224, 224))
img = np.array(img)
expand_img = np.expand_dims(img, axis = 0)
pre_img = preprocess_input(expand_img)
result = model.predict(pre_img).flatten()
normalized = result/norm(result)
normalized

Neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
Neighbors.fit(feature_list)

distanc, indices = Neighbors.kneighbors([normalized])

print(indices)

for file in indices[0][1:6]:
    # print(filename[file])
    imgName = cv2.imread(filename[file])
    cv2.imshow("Frame", cv2.resize(imgName, (640, 480)))
    cv2.waitKey(0)


