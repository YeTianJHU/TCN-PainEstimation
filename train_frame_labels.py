import numpy as np
import warnings
from functools import partial
import threading
import os
import sklearn
from sklearn import cross_validation
import matplotlib.pyplot as plt

import keras
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
from keras.utils import np_utils
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import get_data
import train
warnings.filterwarnings("ignore")

feature_dir = '/home/ye/Works/pain'
label_dir = '/home/ye/Works/pain/Sequence_Labels'
feature_name = 'feature_from_verification_model.mat'
label_name = 'OPR'

def frame_labels_classification(
           n_classes,
           max_len):
  """frame_labels_classification model. Use frame-level labels to classify video-level labels.
  Args:
    n_classes: number of classes for this kind of label.
    max_len: the number of frames for each video.
  Returns:
    model: compiled model."""

  input = Input(shape=(max_len,1))
  model = input
  print 'step1: ',model.shape
  flatten = Flatten()(model)
  dense = Dense(output_dim=max_len,
        init="normal")(model)
  print 'step2: ',dense.shape
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

def train_frame_model(model, y_categorical,max_len, get_cross_validation):
	"""Load data, compile, fit, evaluate model, and predict labels.
	Args:
		model: model name.
		y_categorical: whether to use the original label or one-hot label. True for classification models. False for regression models.
		max_len: the number of frames for each video. 
		get_cross_validation: whether to cross validate.
	Returns:
		classes: predications. Predication for all the videos is using cross validation.
		y_test: test ground truth. Equal to all labels if using cross validation."""
	
	x = get_data.get_frame_labels(feature_dir,feature_name,max_len)
	y = get_data.get_labels(label_dir, label_name)
	y = np.array(y)
	if y_categorical == True:
		y = np_utils.to_categorical(y)
		print x.shape, y.shape

	model = frame_labels_classification(6,max_len)
	if get_cross_validation==True:
		loss = np.zeros((4))
		acc = np.zeros((4))
		classes = np.zeros((200, 6))
		x_train_cro, y_train_cro, x_test_cro, y_test_cro = train.set_cross_validation(x, y)
		for i in range(3):
			model.fit(x_train_cro[i], y_train_cro[i], validation_data=[x_test_cro[i],y_test_cro[i]], epochs=5)
			loss_and_metrics = model.evaluate(x_test_cro[i], y_test_cro[i])	
			loss[i] = loss_and_metrics[0]
			acc[i]  = loss_and_metrics[1]
			classes[i*50:(i+1)*50] = model.predict(x_test_cro[i])
		loss_mean = np.mean(loss)
		acc_mean = np.mean(acc)
		y_test = y
	elif get_cross_validation==False:
		x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2, random_state=1)
		model.fit(x_train, y_train, validation_data=[x_test,y_test], epochs=5)
		loss_mean, acc_mean = model.evaluate(x_test,y_test)
		classes = model.predict(x_test)

	return classes, y_test


max_len = 100
classes, y_test = train_frame_model(frame_labels_classification, y_categorical=True, max_len=max_len, get_cross_validation=True)
print '======================'
classes = train.to_vector(classes)
y_test =  train.to_vector(y_test)
cnf_matrix = sklearn.metrics.confusion_matrix(y_test, classes)
plt.figure()
train.plot_confusion_matrix(cnf_matrix,classes=[0,1,2,3,4,5], normalize=True,
                      title='Confusion matrix, without normalization')
plt.show()