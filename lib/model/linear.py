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
# from tqdm import tqdm

from .base import MLModel


class LinearModel(MLModel): # линейная модель

    def __init__(self,initiator): # ,n_features=0,n_out=0):
        super().__init__(initiator)

    def _reset(self):
        self._weight = self._initiator.get()
        return self    

    def _state(self,x): 
        return x.dot(self._weight)
    
    def _predict(self,x): 
        return self._act( self._state(x) )    

    def _partial(self,x): 
        return self._act_derivative( self._state(x) )*x 
        # return self._act_derivative( self._state(x) )[:,np.newaxis]*x   
     
    @staticmethod
    def _act(s): return s # linear
    
    @staticmethod
    def _act_derivative(s): return np.array([1])


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class LinearRegression(LinearModel): # линейная регрессия
   
    def _reset(self):
        w = self._initiator.get()
        assert w.shape[1]==1, f'size output incorrect - {w.shape}'
        self.weight = w
        return self

#     @property
#     def weight(self): 
#         return super().weight
# 
#     @weight.setter
#     def weight(self, value): 
#         assert value.shape[1]==1, f'size output incorrect - {value.shape}'
#         super().weight = value


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class LogisticRegression(LinearRegression): # логистическая регрессия

    @staticmethod
    def _act(s): 
        return 1./(1.+np.exp(-s) ) # sigmoid
    
    @classmethod
    def _act_derivative(cls,s): # sigmoid derivative
        o = cls._act(s)
        return o*(1.-o)



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class Softmax(LinearModel): 

    def _reset(self):
        w =  self._initiator.get()
        assert (w.shape[1]>1),f'softmax size output less 2 - {w.shape}'
        self.weight = w
        return self

#     @property
#     def weight(self): 
#         return super().weight
# 
#     @weight.setter
#     def weight(self, value): 
#         assert (value.shape[1]>1),'softmax size output less 2'
#         super().weight = value

    @staticmethod
    def _act(s): 
        es = np.exp(s)
        ess = es.sum(axis=1)[:,np.newaxis]
        with np.errstate(invalid='ignore',divide='ignore'):
            o = np.where( ess!=0., es/ess, 0. )
        return o
    
    @classmethod
    def _act_derivative(cls,s): # sigmoid derivative
        o = cls._act(s)
        return o*(1.-o)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

