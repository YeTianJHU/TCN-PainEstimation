import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, merge, Lambda
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.regularizers import l2,l1
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K
from keras.layers.core import Reshape
from keras.activations import relu
from functools import partial

import get_data

import numpy as np
import warnings

import keras
import get_data

from keras.utils import np_utils
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import threading
import os
import sklearn
from sklearn import cross_validation
import matplotlib.pyplot as plt
import itertools
warnings.filterwarnings("ignore")

feature_dir = '/home/ye/Works/pain'
label_dir = '/home/ye/Works/pain/Sequence_Labels'
feature_name = 'feature_from_verification_model.mat'
label_name = 'OPR'

def max_filter(x):
    # Max over the best filter score (like ICRA paper)
    max_values = K.max(x, 2, keepdims=True)
    max_flag = tf.greater_equal(x, max_values)
    out = x * tf.cast(max_flag, tf.float32)
    return out

def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True)+1e-5
    out = x / max_values
    return out

def WaveNet_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)  
    return Merge(mode='mul')([tanh_out, sigm_out])

def ED_TCN(n_nodes, pool_sizes, conv_lens, n_classes, n_feat, max_len, 
      loss='categorical_crossentropy', causal=False, 
      optimizer="rmsprop", activation='norm_relu',
      compile_model=True):
  n_layers = len(n_nodes)

  inputs = Input(shape=(max_len,n_feat))
  model = inputs

  # ---- Encoder ----
  for i in range(n_layers):
    # Pad beginning of sequence to prevent usage of future data
    if causal: model = ZeroPadding1D((conv_lens[i]//2,0))(model)
    model = Convolution1D(n_nodes[i], conv_lens[i], border_mode='same')(model)
    if causal: model = Cropping1D((0,conv_lens[i]//2))(model)

    model = SpatialDropout1D(0.3)(model)
    
    if activation=='norm_relu': 
      model = Activation('relu')(model)            
      model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
    elif activation=='wavenet': 
      model = WaveNet_activation(model) 
    else:
      model = Activation(activation)(model)            
    
    model = MaxPooling1D(pool_sizes[i])(model)

  # ---- Decoder ----
  for i in range(n_layers):
    model = UpSampling1D(pool_sizes[-i-1])(model)
    if causal: model = ZeroPadding1D((conv_lens[-i-1]//2,0))(model)
    print n_nodes[-i-1], conv_lens[-i-1]
    model = Convolution1D(n_nodes[-i-1], conv_lens[-i-1], border_mode='same')(model)
    if causal: model = Cropping1D((0,conv_lens[-i-1]//2))(model)

    model = SpatialDropout1D(0.3)(model)

    if activation=='norm_relu': 
      model = Activation('relu')(model)
      model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
    elif activation=='wavenet': 
      model = WaveNet_activation(model) 
    else:
      model = Activation(activation)(model)

  # Output FC layer
  model = TimeDistributed(Dense(n_classes, activation="softmax" ))(model)
  
  model = Model(input=inputs, output=model)

  if compile_model:
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal", metrics=['accuracy'])

  return model

  # Output FC layer
  model = TimeDistributed(Dense(n_classes, activation="softmax" ))(model)

  model = Model(input=inputs, output=model)
  model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal", metrics=['categorical_accuracy'])
#########################################################################
def get_cross_validation(x,y):

  # x_train_1 = x[50:]
  # y_train_1 = y[50:]
  # x_test_1 = x[:49]
  # y_test_1 = y[:49]
  # x_train_2 = np.concatenate((x[:49], x[100:]),axis=0)
  # y_train_2 = np.concatenate((y[:49], y[100:]),axis=0)
  # x_test_2 = x[50:99]
  # y_test_2 = y[50:99]
  # x_train_3 = np.concatenate((x[:99], x[150:]),axis=0)
  # y_train_3 = np.concatenate((y[:99], y[150:]),axis=0)
  # x_test_3 = x[100:149]
  # y_test_3 = y[100:149]
  # x_train_4 = x[:149]
  # y_train_4 = y[:149]
  # x_test_4 = x[150:]
  # y_test_4 = y[150:]
  x_train_1 = x[50:]
  y_train_1 = y[50:]
  x_test_1 = x[:50]
  y_test_1 = y[:50]
  x_train_2 = np.concatenate((x[:50], x[100:]),axis=0)
  y_train_2 = np.concatenate((y[:50], y[100:]),axis=0)
  x_test_2 = x[50:100]
  y_test_2 = y[50:100]
  x_train_3 = np.concatenate((x[:100], x[150:]),axis=0)
  y_train_3 = np.concatenate((y[:100], y[150:]),axis=0)
  x_test_3 = x[100:150]
  y_test_3 = y[100:150]
  x_train_4 = x[:150]
  y_train_4 = y[:150]
  x_test_4 = x[150:]
  y_test_4 = y[150:]

  x_train = [x_train_1,x_train_2,x_train_3,x_train_4]
  y_train = [y_train_1,y_train_2,y_train_3,y_train_4]
  x_test = [x_test_1,x_test_2,x_test_3,x_test_4]
  y_test = [y_test_1,y_test_2,y_test_3,y_test_4]
  print 'cross val shapes',  x_test_1.shape, x_test_2.shape, x_test_3.shape,x_test_4.shape
  return x_train, y_train, x_test, y_test

def to_vector(mat):
  out = np.zeros((mat.shape[0],mat.shape[1]))
  out2 = np.zeros((mat.shape[0]))
  for i in range(mat.shape[0]):
    for n, j in enumerate(mat[i]):
      if j == np.amax(mat[i]):
        out[i][n] = 1
        out2[i] = n
  return out2

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#####################################################################
def train_model(model, max_len, set_cross_validation=False, non_zero=False):
  x = get_data.get_feature_tensor(feature_dir,feature_name,max_len)
  y = get_data.get_frame_01_labels(feature_dir,feature_name,max_len)
  np.set_printoptions(threshold='nan')
  print y.T
  y_video = get_data.get_labels(label_dir, label_name)


  if model == ED_TCN:
    n_nodes = [512, 512]  #, 1024]
    pool_sizes = [2, 2]  #, 2]
    conv_lens = [10, 10]  #, 10]

    causal = False

    model = ED_TCN(n_nodes, pool_sizes, conv_lens, 2, 512, max_len, 
      causal=causal, activation='norm_relu', optimizer='rmsprop')
    model.summary()

  loss = np.zeros((4))
  acc = np.zeros((4))
  classes = np.zeros((200,max_len, 2))
  if set_cross_validation == False:
    if non_zero == True:
      x,labels_new, y = get_data.non_zero_data(x,y_video,max_len, y, use_y_frame=True)
    y_cat = np_utils.to_categorical(y,num_classes=2)
    y_cat = np.reshape(y_cat, (-1, max_len, 2))
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y_cat,test_size=0.2, random_state=1)
    model.fit(x_train,y_train, epochs=10)
    loss_and_metrics = model.evaluate(x_test,y_test)
    loss_mean = loss_and_metrics[0]
    acc_mean  = loss_and_metrics[1]
    classes = model.predict(x_test)
  elif non_zero == False:
    y_cat = np_utils.to_categorical(y,num_classes=2)
    y_cat = np.reshape(y_cat, (200, max_len, 2))
    x_train_cro, y_train_cro, x_test_cro, y_test_cro = get_cross_validation(x, y_cat)
    for i in range(4):
      print i
      model.fit(x_train_cro[i], y_train_cro[i],batch_size=20, epochs=5)
      loss_and_metrics = model.evaluate(x_test_cro[i], y_test_cro[i]) 
      loss[i] = loss_and_metrics[0]
      acc[i]  = loss_and_metrics[1]
      classes[i*50:(i+1)*50] = model.predict(x_test_cro[i])
    loss_mean = np.mean(loss)
    acc_mean = np.mean(acc)
    y_test = y_cat
  print 'loss_mean: ', loss_mean, ' ', 'acc_mean: ', acc_mean
  return classes, y_test

classes, y_cat = train_model(ED_TCN, 48, set_cross_validation=False, non_zero=True)
y_cat = np.reshape(y_cat, (y_cat.shape[0]*y_cat.shape[1], y_cat.shape[2]))
classes = np.reshape(classes, (classes.shape[0]*classes.shape[1], classes.shape[2]))
y_test = to_vector(y_cat)
classes = to_vector(classes)
print y_test[:100], classes[:100]

cnf_matrix = sklearn.metrics.confusion_matrix(y_test, classes)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=[0,1], normalize=False,
                      title='Confusion matrix, without normalization')
plt.show()