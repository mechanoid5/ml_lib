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
class MLP(MLModel): 

    def __init__(self,initiator):
        super().__init__(initiator) # инициализируем параметры с помощью процедуры initiator

    def _reset(self): # процедура инициализации генерирует начальные веса модели
        self._weight = [ i.get() for i in self._initiator ]
        return self    
        
    def _score(self,x): # генерируем выход модели
        o = x
        for w in self._weight:
            o = self._act(o.dot(w))
        return o
    
    @staticmethod
    def _act(s): return np.tanh(s)
    
    @staticmethod
    def _act_derivative(s): 
        return 1./np.square(np.cosh(s))

    def _state(self,x):
        o,s = x, []
        for w in self._weight:
            s.append( o.dot(w) )
            o = self._act(s[-1])
        return s
    
    def _score_derivative(self,x):
        o = [x]
        for w in self._weight:
            o.append( self._act_derivative(o[-1].dot(w)) )
        return o[1:]
    
    def _partial(self,x):
        return [ s*d for s,d in zip( self._state(x), self._score_derivative(x) ) ]
    


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)


