# -*- coding: utf-8 -*-


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import pickle
import numpy as np
import scipy.io as scio 
import h5py

import math
import pdb
import shutil
import os


from Dual_surrogate import Dual_surrogate



# Load data
main_path = os.path.dirname(__file__)
grid_size = (100, 100, 1)
n_s = 1200
ts_features = [50,8]

# Randomly generate data for verification. The user should prepare their own data using 
# reservoir simulation software or other methods
x = np.random.rand(1000,10000)
y = np.random.rand(1000,400)


print(x.shape)
print(y.shape)

# Prepare data
num_train = 100

train_x = x[0:num_train,:]
train_y = y[0:num_train,:]

val_x = x[num_train:int(num_train+200),:]
val_y = y[num_train:int(num_train+200),:]

test_x = x[int(num_train+200):int(num_train+400):,:]
test_y = y[int(num_train+200):int(num_train+400):,:]



# Training

batch_size = 32
latent_dim = 100
num_lstm = 1

mid_features = []
for j in range(num_lstm):
    mid_features += [100]

lr = 0.001
dropout = 0.2
n_filters = [64,128,256,512]
epochs = 1
validation_split = 0.25

surro = Dual_surrogate(train_x, train_y, val_x,val_y,test_x, test_y,\
        grid_size, ts_features, path=main_path,)

surro.hyper_para(epochs=epochs, loss='mse', batch_size=batch_size,\
    num_lstm=num_lstm, mid_features=mid_features, \
    dropout=dropout,\
    n_filters=n_filters, lr=lr, latent_dim=latent_dim,\
    validation_split=validation_split,\
    Bidirectional=False, rnn='LSTM',)

surro.hyper_para2(
    epochs=epochs, 
    lr=0.001, 
    batch_size=64,
    )

surro.train()




