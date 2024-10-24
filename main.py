import streamlit as st
from PIL import Image

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