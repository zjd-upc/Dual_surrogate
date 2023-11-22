

import numpy as np
import math

class minmax():
    def __init__(self,data,if_tanh=False):
        '''
        data : (n,:,)
        '''
        if len(data.shape)==1:
            self.min = np.min(data,axis=0).reshape(1,-1)
            self.max = np.max(data,axis=0).reshape(1,-1)        
        else:
            self.min = np.min(data,axis=0)
            self.max = np.max(data,axis=0)
        
        self.if_tanh = if_tanh

    def fit_transform(self,data):
        data1 = (data-self.min)/(self.max-self.min)
        data1[data1<0] = 0
        data1[data1>1] = 1

        data1 = np.nan_to_num(data1)

        if self.if_tanh:
            data1 = data1*2-1

        return data1

    def inverse_transform(self,data):

        if self.if_tanh:
            data1 = (data+1)/2*(self.max-self.min)+self.min
        else:
            data1 = data*(self.max-self.min)+self.min
            
        return data1

class minmax_ts():
    def __init__(self, data, ts):
        '''
        data : (n,:,:,:)
        '''
        self.min = np.min(data,axis=0)
        self.max = np.max(data,axis=0)

        for i in range(ts[1]):
            self.min[:,i] = self.min[:,i].min()
            self.max[:,i] = self.max[:,i].max()

    def fit_transform(self,data):
        data1 = (data-self.min)/(self.max-self.min)
        
        data1[data1<0] = 0
        data1[data1>1] = 1

        data1 = np.nan_to_num(data1)
        return data1

    def inverse_transform(self,data):
        data1 = data*(self.max-self.min)+self.min
        return data1


def grid_reshape(x, y, z):
    '''
    reshape the model grids for training.
    Input, return: the number of grids in the x,y direction.
    '''
    log_int_x = int(math.log(x, 2))+1
    x1 = int(math.pow(2, log_int_x))

    log_int_y = int(math.log(y, 2))+1
    y1 = int(math.pow(2, log_int_y))

    return x1, y1, z