#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging

import numpy as np
import numpy.random as rng
from tqdm import tqdm

from .base import ModelOptimimizer
from .lrate import ConstLRA
from .regularizator import Regularization
# from .regularizator import RegularizationL1
# from .breaker import EarlyStopping
from .breaker import FitBreakException
# from .breaker import Breaking

from ..loss.base import EmptyLoss


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class BaseGD(ModelOptimimizer):

    def __init__(self,loss,loss_val=EmptyLoss(),lra=ConstLRA(.1),breaker=[],breaker_val=[]):
        super().__init__(loss=loss)
        self._lra = lra
        self._breaker = breaker
        assert (loss!=loss_val), 'train loss and validation loss is same object'
        self._loss_val = loss_val
        self._breaker_val = breaker_val

    def _adjust_weigth(self,data,lr):
        x,t = data
        d_loss = self._loss.gradient(x,t)
        self._loss.model.weight  = self._loss.model.weight - lr*d_loss
        return self
    
    def _estimate_epoch(self,data_train,data_val):
        self._loss.estimate(data_train[0],data_train[1])
        self._loss_val.estimate(data_val[0],data_val[1])
        return self

    def _fit_epoch(self,data_train,data_val,): 
        lr = self._lra.next()
        self._adjust_weigth(data_train,lr) # обучаем модель
        self._estimate_epoch(data_train,data_val)
        return self
    
    def _check_loss(self):
        for b in self._breaker: b.check(self._loss)
        for b in self._breaker_val: b.check(self._loss_val)
        return self    

    def _fit(self,data_train,data_val,n_epoch): 
        data_val = data_train if data_val is None else data_val
        epoch = tqdm(range(n_epoch))
        for _ in epoch:
            self._fit_epoch(data_train,data_val)
            epoch.set_postfix({'loss':self._loss.history[-1], 'lr':self._lra.history[-1],})
            self._check_loss()
        return self

    def fit(self,data_train,data_val=None,n_epoch=2): 
        try:
            self._fit(data_train=data_train,data_val=data_val,n_epoch=n_epoch) 
        except FitBreakException as break_reason:
            logging.info(break_reason)
        except Exception as err:
            logging.error(err)

        return self._loss.model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class GD(BaseGD):

    def __init__(self,loss,loss_val=EmptyLoss(),lra=ConstLRA(.1),breaker=Breaking(),regul=Regularization(1.),momentum=0.):
        super().__init__(loss=loss,loss_val=loss_val,lra=lra,breaker=breaker)
        self._regularizator = regul # регуляризатор
        self._dweight = 0. # значения изменения весов на пред. шаге для расчёта момента
        self._momentum = momentum # коэффициент момента
    
    def _adjust_weigth(self,data,lr):
        x,t = data
        d_loss = self._loss.gradient(x,t) # значение градиента ф-ции потери
        dweight =(
                lr*( 
                    d_loss # значение градиента ф-ции потери
                    * self._regularizator.transform(self._loss.model.weight) # добавка регуляризатора
                    ) 
                    + self._momentum * self._dweight # добавка момента
                )
        self._loss.model.weight  = ( self._loss.model.weight - dweight )
        self._dweight = dweight

        return self

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class SGD(GD):
    
    def __init__(self,loss,loss_val=EmptyLoss(),lra=ConstLRA(.1),breaker=Breaking(),regul=Regularization(1.),momentum=0.):
        super().__init__(loss=loss,loss_val=loss_val,lra=lra,breaker=breaker,regul=regul,momentum=momentum)
        self._batch_size=0
        self._target_is_indices=False
    
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
            super()._adjust_weigth( batch_data,lr) # обучаем модель на батче
        return self
  
    def fit(self, data_train, batch_size, data_val=None, n_epoch=2, target_is_indices=False ):
        assert (batch_size>0), 'batch_size less zero'
        # assert (batch_size<data_tarin[1]), 'batch_size more than target len'
        self._batch_size=batch_size
        self._target_is_indices=target_is_indices
        return super().fit(data_train=data_train, data_val=data_val, n_epoch=n_epoch)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

