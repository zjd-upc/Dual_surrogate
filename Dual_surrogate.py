
# %%
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers

from tensorflow.keras.models import load_model,Model
from tensorflow.keras.layers import Dense, Dropout,LSTM, Input, Dense, RepeatVector,\
    TimeDistributed, GlobalAveragePooling2D, MaxPooling2D,Bidirectional,GRU

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers.experimental.preprocessing import Resizing


import numpy as np
import matplotlib.pyplot as plt
import math
import datetime


from resnet import residual_block
from process import minmax_ts, minmax, grid_reshape




class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.rmse = []
        self.r2 = []
        self.loss = []
    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, rmse, r2  = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, r2: {}\n'.format(loss, r2))
        self.rmse += [rmse]
        self.loss += [loss]
        self.r2 += [r2]

def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



def rmse(y_true, y_pred):
    mse = K.mean(K.square(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)
    rmse = K.sqrt(mse)
    return rmse



class Dual_surrogate():
    '''
    Dual-surrogate framework
    '''
    def __init__(self, 
                 train_x, 
                 train_y, 
                 val_x, 
                 val_y,\
                 test_x, test_y, 
                 grid_size=(40,120,20),\
                 ts_features=[50,8], 
                 path=None, 
                 error_mean=None,
                ):
        '''
        Args: 
            train_x: training data for input
            train_y: training data for output
            val_x: validation data for input
            val_y: validation data for output
            test_x: test data for input
            test_y: test data for output
            grid_size: (x,y,z)
            ts_features: list, [timesteps, features]
            path: model save path
            error_mean: the mean of the prediction error for the 
                two surrogate models
        '''
        self.train_x = train_x
        self.train_y = train_y

        self.val_x = val_x
        self.val_y = val_y

        self.test_x = test_x
        self.test_y = test_y

        self.x_dim = train_x.shape[1]
        self.ts_features = ts_features
        self.path = path
        self.grid_size = grid_size

        self.error_mean = error_mean


    def hyper_para(self, 
                   num_lstm=2, 
                   mid_features=[10,10], 
                   epochs=200,
                   lr=0.001, 
                   batch_size=8, 
                   validation_split=0.25,
                   loss='mae',
                   dropout=0.5,
                   latent_dim=100,
                   n_filters=[64,128,256,512,1024,1024],
                   Bidirectional=False, 
                   rnn='LSTM',
                   preac=True,
                   ):
        ''' Other hyper-parameters for Model 1
        Args: 
            num_lstm: the number of lstm layers
            mid_features: the number of features in the lstm layers
            epochs: the number of epochs
            lr: learning rate
            batch_size: batch size
            validation_split: validation split
            loss: loss function
            dropout: dropout rate
            latent_dim: the number of features in the latent space
            n_filters: the number of filters in the cnn layers
            Bidirectional: whether to use Bidirectional lstm
            rnn: the type of rnn
            preac: whether to use pre-activation
        '''
        self.num_lstm = num_lstm
        self.mid_features = mid_features
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.loss = loss
        self.dropout = dropout

        self.latent_dim = latent_dim
        self.n_filters = n_filters

        self.Bidirectional = Bidirectional
        self.rnn = rnn
        self.preac=True # resnet v2


    def hyper_para2(self,
                    epochs=200, 
                    lr=0.001, 
                    batch_size=8, 
                    temp_size=64,
                    n_filters=[64,128,256,512],
                    ):
        ''' Other hyper-parameters for Model 2
        Args: 
            epochs: the number of epochs
            lr: learning rate
            batch_size: batch size
            temp_size: the size of the resized data
            n_filters: the number of filters in the cnn layers
        '''
        self.batch_size2 = batch_size
        self.lr2 = lr
        self.epochs2 = epochs
        self.temp_size = temp_size
        self.n_filters2 = n_filters


    def process_x(self):
        '''
        Process the input data
        '''
        mn_x = minmax(self.train_x)
        self.mn_x = mn_x
        train_xn = mn_x.fit_transform(self.train_x)
        train_xn = train_xn.reshape(-1, \
            self.grid_size[2],self.grid_size[1],self.grid_size[0])
        train_xn = np.transpose(train_xn,(0,2,3,1))
        return train_xn


    def process_x_s(self, data):
        test_xn = self.mn_x.fit_transform(data)
        test_xn = test_xn.reshape(-1, \
            self.grid_size[2],self.grid_size[1],self.grid_size[0])
        test_xn = np.transpose(test_xn,(0,2,3,1))
        return test_xn


    def process_y(self):
        '''
        Process the output data
        '''
        timesteps, out_features = self.ts_features[0], self.ts_features[1]
        train_y = self.train_y.copy()
        train_y = train_y.reshape(train_y.shape[0],out_features,timesteps)
        train_y = train_y.swapaxes(1,2)
        mn = minmax_ts(train_y, self.ts_features)
        train_yn = mn.fit_transform(train_y)
        self.mn = mn
        return train_yn


    def process_y_s(self, test_y):
        timesteps, out_features = self.ts_features[0], self.ts_features[1]
        test_y_ = test_y.reshape(test_y.shape[0],out_features,timesteps)
        test_y_ = test_y_.swapaxes(1,2)
        test_yn = self.mn.fit_transform(test_y_)
        return test_yn

        
    def cnn_module(self, input_layer):
        '''
        CNN module for Model 1
        '''
        x = Resizing(self.y1, self.x1)(input_layer)
        n_filters = self.n_filters[:]
        f_n = int(n_filters[-1]/self.ts_features[0]+1)
        f_n = int(f_n*self.ts_features[0])

        if self.preac:
            for i in range(len(n_filters)):
                x = residual_block(x, n_filters[i], version=2) 
                if i != len(n_filters)-1:
                    x = MaxPooling2D()(x)

            x = layers.BatchNormalization()(x) 
            x = layers.LeakyReLU()(x)
        else:
            for i in range(len(n_filters)):
                x = residual_block(x, n_filters[i], version=1) 
                if i != len(n_filters)-1:
                    x = MaxPooling2D()(x) 

        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.dropout)(x)
        x = Dense(self.latent_dim, activation='relu')(x)
        x = RepeatVector(self.ts_features[0])(x)

        return x


    def rnn_module(self, x):
        '''
        RNN module for Model 2
        '''
        for i in range(self.num_lstm):

            if self.Bidirectional:
                if self.rnn=='LSTM':
                    x = Bidirectional(LSTM(self.mid_features[i], \
                        return_sequences=True, dropout=self.dropout))(x)

                elif self.rnn=='GRU':
                    x = Bidirectional(GRU(self.mid_features[i], \
                        return_sequences=True, dropout=self.dropout))(x)  
            else:
                if self.rnn=='LSTM':
                    x = LSTM(self.mid_features[i], \
                        return_sequences=True, dropout=self.dropout)(x)
                elif self.rnn=='GRU':
                    x = GRU(self.mid_features[i], \
                        return_sequences=True, dropout=self.dropout)(x) 

        x = layers.ReLU()(x) 
        x = TimeDistributed(Dense(self.ts_features[1], 
            activation='sigmoid'))(x)

        return x


    def get_model(self):
        '''
        Define Model 1
        '''
        input_layer = Input(shape=(self.grid_size[1],\
            self.grid_size[0],self.grid_size[2]))
        z = self.cnn_module(input_layer)
        x = self.rnn_module(z)

        model = Model(inputs=input_layer, outputs=x)
        opt = Adam(learning_rate=self.lr)
        model.compile(loss=self.loss, optimizer=opt, metrics=[rmse, r_square])
        
        print(model.summary())  
        
        return model







    def fcnn(self, input_layer):
        '''
        Fully convolutionary neural network for Model 2
        '''
        x_size = self.temp_size
        y_size = self.temp_size

        # 1. resize module
        x = Resizing(x_size, y_size)(input_layer)

        # 2. encoder
        n_filters = self.n_filters2.copy()


        for i in range(len(n_filters)):
            x = residual_block(x, n_filters[i],version=2) 
            x = MaxPooling2D()(x)
        
        x = layers.BatchNormalization()(x) 
        x = layers.LeakyReLU()(x)

        x = Dropout(0.5)(x)
        
        # 3. decoder
        filter_num_d = n_filters.copy()
        filter_num_d.reverse()
        for i in range(len(filter_num_d)):
            x = residual_block(x, filter_num_d[i],version=2) 
            x = layers.UpSampling2D()(x)

        x = layers.BatchNormalization()(x) 
        x = layers.LeakyReLU()(x)

        # 4. resize module
        x = Resizing(self.ts_features[0], self.ts_features[1])(x)

        # 5. transition
        final_filter = [n_filters[0], 32, 1]
        for i in range(3):
            x = layers.Conv2D(final_filter[i],\
                 (3, 3),  
                 padding="same")(x)
            x = layers.BatchNormalization()(x)
            if i!=2:
                x = layers.Activation('relu')(x)
            else:
                x = layers.Activation('sigmoid')(x)
                output = K.squeeze(x, axis=3)
        
        return output


    def get_model2(self):
        '''
        Define Model 2
        '''
        input_layer = Input(shape=(self.grid_size[1],\
            self.grid_size[0],self.grid_size[2]))

        x = self.fcnn(input_layer)

        model = Model(inputs=input_layer, outputs=x)
        opt = Adam(learning_rate=self.lr2)
        model.compile(loss=self.loss, optimizer=opt, metrics=[rmse, r_square])

        print(model.summary())  
        
        return model


    def train(self):
        
        '''
        Args: 
            datax: (n,:)
            datay: (n,:)
            timesteps: int
            out_features: int
            path: str
        '''

        # Model 1
        train_x = self.process_x()
        train_y = self.process_y()
        self.x1, self.y1, self.z1 = grid_reshape(self.grid_size[0], \
            self.grid_size[1], self.grid_size[2])

        model = self.get_model()

        val_x = self.process_x_s(self.val_x)
        val_y = self.process_y_s(self.val_y)
        test_x = self.process_x_s(self.test_x)
        test_y = self.process_y_s(self.test_y)

        test_call = TestCallback((test_x, test_y))
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                  mode='min', min_delta=0.0001, cooldown=0, min_lr=0.0001)
        
        history = model.fit(train_x, train_y,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            # validation_split=self.validation_split,
                            validation_data= (val_x, val_y),
                            callbacks=[test_call, reduce_lr],
                            # callbacks=[test_call, reduce_lr,tensorboard_callback],
                            # shuffle= True,
                            )

        self.model = model 

        # Model 2
        train_y_pre = model.predict(train_x)
        val_y_pre = model.predict(val_x)
        test_y_pre = model.predict(test_x)

        train_y_e = train_y-train_y_pre
        val_y_e = val_y-val_y_pre
        test_y_e = test_y-test_y_pre


        mn_e = minmax(train_y_e)

        self.mn_e = mn_e
        train_y_en = mn_e.fit_transform(train_y_e)
        test_y_en = mn_e.fit_transform(test_y_e)
        val_y_en = mn_e.fit_transform(val_y_e)


        test_call1 = TestCallback((test_x, test_y_en))
        reduce_lr1 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                  mode='min', min_delta=0.0001, cooldown=0, min_lr=0.0001)

        model_e = self.get_model2()

        history1 = model_e.fit(train_x, train_y_en,
                            epochs=self.epochs2,
                            batch_size=self.batch_size2,
                            validation_data= (val_x, val_y_en),
                            callbacks=[test_call1, reduce_lr1],
                            )


        self.model_e = model_e




    def predict(self, datax):
        '''
        Args:
            datax: (1,:)
        Returns:
            out: (1,:)
        '''
        datax = self.process_x_s(datax)
        
        # Model 1
        y_ = self.model(datax).numpy()  # (n, timesteps, out_features)
        
        # Model 2
        y_e_ = self.model_e(datax).numpy()
        y_e_ = self.mn_e.inverse_transform(y_e_)

        y_ += y_e_
        y_[y_<0]=0
        y_[y_>1]=1

        y_ = self.mn.inverse_transform(y_)
        y_ = y_.swapaxes(1,2)
        y_ = y_.reshape(y_.shape[0],-1)

        # Add the error mean
        if self.error_mean is not None:
            y_ += self.error_mean

        return y_
