#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging

import numpy as np
import numpy.random as rng

from .base import MLModel

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class LinearModel(MLModel): # линейная модель

    def __init__(self,initiator):
        super().__init__(initiator) # инициализируем параметры с помощью процедуры initiator

    def _reset(self): # процедура инициализации генерирует начальные веса модели
        self._weight = self._initiator.get()
        return self    
    
    def _score(self,x): # генерируем выход модели
        return self._act( self._state(x) )    

    def _state(self,x): # состояние модели
        return x.dot(self._weight)
   
    def _partial(self,x): # частные производные по параметрам модели для применения градиентных методов 
        act_d = self._act_derivative( self._state(x) )[:,np.newaxis]
        state_d = self._state_derivative(x)[:,np.newaxis]
        state_d = np.swapaxes(state_d,1,2)
        return np.matmul( state_d, act_d )

    def _state_derivative(self,x): # производная функции состояния по параметрам модели
        return x
     
    @staticmethod
    def _act(s): # функция активации состояния
        return s # линейная 
    
    @staticmethod
    def _act_derivative(s): # производная функция активации по её аргументу
        return np.ones(s.shape)

    @property
    def shape(self): return self._weight.shape


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class LinearClassifier(LinearModel): # линейный классификатор

    def __init__(self,initiator):
        super().__init__(initiator) # инициализируем параметры с помощью процедуры initiator
        self._score_threshold = None
    
    def _solve_score(self,o):
        return (o>self._score_threshold).astype(int) if (o.shape[1]==1) else np.argmax(o,axis=1)

    def _predict(self,x):
        return self._solve_score( self._score(x) ) 

    def _save(self): # пакуем параметры модели
        return {'weight':self._weight,'score_threshold':self._score_threshold,}

    def _load(self,data): # распаковываем считанные параметры модели
        self._weight = data['weight'] 
        self._score_threshold = data['score_threshold'] 
        return self    

    @property
    def score_threshold(self): return self._score_threshold

    @score_threshold.setter
    def score_threshold(self,value): self._score_threshold = value

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class SLP(LinearClassifier): # однослойная нейросеть

    @staticmethod
    def _act(s): 
        return 1./(1.+np.exp(-s) ) # sigmoid
    
    @classmethod
    def _act_derivative(cls,s): # sigmoid derivative
        o = cls._act(s)
        return o*(1.-o)



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class Softmax(LinearClassifier): 

    def _reset(self):
        w =  self._initiator.get() # размер выхода softmax должен быть 2 или больше
        assert (w.shape[1]>1),f'softmax size output less 2 - {w.shape}'
        self.weight = w
        return self

    @staticmethod
    def _act(s): # вычисляем softmax
        es = np.exp(s)
        ess = es.sum(axis=1)[:,np.newaxis]
        with np.errstate(invalid='ignore',divide='ignore'):
            o = np.where( ess!=0., es/ess, 0. )
        return o
    
    @classmethod
    def _act_derivative(cls,s): # производная softmax
        o = cls._act(s)
        return o*(1.-o) 
        
        # https://peterroelants.github.io/posts/cross-entropy-softmax/

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# class LogisticRegression(LinearModel): # логистическая регрессия
# 
#     def _reset(self):
#         w = self._initiator.get()
#         assert w.shape[1]==1, f'size output incorrect - {w.shape}'
#         self.weight = w
#         return self
# 
#     @staticmethod
#     def _act(s): 
#         return 1./(1.+np.exp(-s) ) # sigmoid
#     
#     @classmethod
#     def _act_derivative(cls,s): # sigmoid derivative
#         o = cls._act(s)
#         return o*(1.-o)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# class LinearRegression(LinearModel): # линейная регрессия
#    
#     def _reset(self):
#         w = self._initiator.get() 
#         assert w.shape[1]==1, f'size output incorrect - {w.shape}'
#         self.weight = w
#         return self


