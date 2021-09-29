#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging

import numpy as np
import numpy.random as rng
# import pickle
# import gzip
from tqdm import tqdm


from .base import ModelOptimimizer

from .lrate import ConstLRA


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class EarlyStopping:

    def __init__( self, min_delta, patience):
        self._min_delta=min_delta
        self._patience=patience

    def estimate(self,loss):
        if len(loss.history)<(self._patience): return False
        return (loss.history[-1]-loss.history[-self._patience])<self._min_delta

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class GD(ModelOptimimizer):

    def __init__(self,loss,lra=ConstLRA(.1),es=None):
        super().__init__(loss=loss)
        self._lra = lra
        self._es = es

    def _adjust_weigth(self,data,lr):
        x,t = data
        d_loss = self._loss.gradient(x,t)
        self._loss.model.weight  = self._loss.model.weight - lr*d_loss
        return self
    
    def _estimate_epoch(self,data):
        self._loss.estimate(data[0],data[1])

    def _fit_epoch(self,data_train,data_val,): 
        lr = self._lra.next()
        self._adjust_weigth(data_train,lr) # обучаем модель
        self._estimate_epoch(data_val)
        return self

    def fit(self,data_train,data_val=None,n_epoch=2): 
        data_val = data_train if data_val is None else data_val
        epoch = tqdm(range(n_epoch))
        for _ in epoch:
            self._fit_epoch(data_train,data_val)
            epoch.set_postfix({'loss':self._loss.history[-1], 'lr':self._lra.history[-1],})
            if not self._es is None: 
                if self._es.estimate(self._loss):
                    break

        return self._loss.history,self._lra.history

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class SGD(GD):
    
    def __init__(self,loss,lra=ConstLRA(.1),es=None):
        super().__init__(loss=loss,lra=lra,es=es)
        self._batch_size=0
        self._target_is_indices=False
        self._batch_transform = lambda d: d
    
    def _select_data(self,data,idx):
        if self._target_is_indices : return data[0],data[1][idx,:] 
        return data[0][idx,:],data[1][idx,:]

    def _get_batch(self,data): 
        n_samples = data[1].shape[0] # количество учебных пар
        batch_count = np.ceil( n_samples/self._batch_size).astype(int) # количество батчей
        # перемешиваем учебный набор и режем его на батчи
        for idx in np.array_split(rng.permutation(n_samples), batch_count ):
            yield self._select_data(data,idx) 

    def _adjust_weigth(self,data,lr):
        for batch_data in self._get_batch(data):
            super()._adjust_weigth( self._batch_transform(batch_data),lr) # обучаем модель на батче
        return self
    
    # FIXME: нужно получить общую оценку эпохи
    def _estimate_epoch(self,data): 
        for batch_data in self._get_batch(data):
            super()._estimate_epoch( self._batch_transform(batch_data) )
        return self

    def fit(self,data_train,batch_size, data_val=None,n_epoch=2,target_is_indices=False,batch_transform=None):
        assert (batch_size>0), 'batch_size less zero'
        # assert (batch_size<data_tarin[1]), 'batch_size more than target len'
        self._batch_size=batch_size
        self._target_is_indices=target_is_indices
        if hasattr(batch_transform, '__call__'): self._batch_transform = batch_transform
        return super().fit(data_train=data_train, data_val=data_val, n_epoch=n_epoch)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

