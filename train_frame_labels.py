import numpy as np
import warnings

import keras
import get_data

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, merge, Lambda
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.regularizers import l2,l1
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K
from keras.utils import np_utils

from keras.activations import relu
from functools import partial


from keras.utils import np_utils
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint

import threading
import os
from keras.callbacks import ReduceLROnPlateau

from sklearn import cross_validation
warnings.filterwarnings("ignore")

feature_dir = '/home/ye/Works/pain'
label_dir = '/home/ye/Works/pain/Sequence_Labels'
feature_name = 'feature_from_verification_model.mat'
label_name = 'OPR'

def frame_labels_classification(
           n_classes,
           max_len):

  input = Input(shape=(max_len,1))
  model = input
  print 'step1: ',model.shape
  flatten = Flatten()(model)
  # dense = Dense(output_dim=max_len,
  #       init="normal")(model)
  # print 'step2: ',dense.shape

  dense = Dense(output_dim=n_classes,
        init="he_normal",
        activation="softmax")(flatten)
  print 'step3: ',dense.shape
  # dense = Dense(output_dim=1,
  #     init="he_normal",
  #     activation="softmax")(dense)
  model = Model(input=input, output=dense)
  optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True) 
  model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['categorical_accuracy'])
  return model

def to_vector(mat):
	out = np.zeros((mat.shape[0],mat.shape[1]))
	out2 = np.zeros((mat.shape[0]))
	for i in range(mat.shape[0]):
		for n, j in enumerate(mat[i]):
			if j == np.amax(mat[i]):
				out[i][n] = 1
				out2[i] = n

	return out2

def train_frame_model(model, y_categorical,max_len):
	if y_categorical == True:
		x = get_data.get_frame_labels(feature_dir,feature_name,max_len)
		y = get_data.get_labels(label_dir, label_name)
		y = np.array(y)
		y_cat = np_utils.to_categorical(y)
		print x.shape, y_cat.shape
		x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y_cat,test_size=0.1, random_state=1)
	elif y_categorical == False:
		x = get_data.get_frame_labels(feature_dir,feature_name,max_len)
		y = get_data.get_labels(label_dir, label_name)
		y = np.array(y)
		print x.shape, y.shape
		x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.1, random_state=1)
	else:
		print 'Error'
	print x_train.shape, x_test.shape

	if model == frame_labels_classification:
		model = frame_labels_classification(n_classes=6, max_len=max_len)
		model.summary()
		model.fit(x_train, y_train, validation_data=[x_test,y_test])
		loss_and_metrics = model.evaluate(x_test, y_test)
		classes = model.predict(x_test)
	elif model == TK_TCN_regression:
		model = TK_TCN_regression(n_classes=6, feat_dim=512, max_len=max_len)
		# model.summary()
		model.fit(x_train, y_train, validation_data=[x_test,y_test])
	return classes, y_test


max_len = 400
classes, y_test = train_frame_model(frame_labels_classification, y_categorical=True, max_len=max_len)
print to_vector(classes)
print to_vector(y_test)
