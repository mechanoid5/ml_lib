#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging

import numpy as np
# import numpy.random as rng


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class EmptyLoss:
    
    # def __init__(self,model=None,normalize_gradient=True): pass

    def estimate(self,input_data,target): return .0
    
    def gradient(self,input_data,target): return .0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class Loss(EmptyLoss):

    def __init__(self,model,normalize_gradient=True):
        super().__init__()
        assert not(model is None), 'try estimate empty model'
        self._model = model
        self._history = []
        self._normalize_gradient = normalize_gradient

    def estimate(self,input_data,target):
        s = self._estimate(self._model.score(input_data), target)
        self._history.append(s)
        return s
        
    def _estimate(self,output,target): pass    
    
    def gradient(self,input_data,target): 
        return self._gradient(input_data,target)
    
    def _gradient(self,input_data): pass

    @property
    def model(self): 
        return self._model

    @property
    def history(self): 
        return self._history

    def _norm(self,x):
        if not self._normalize_gradient: return x
        amax = np.abs(x).max()
        return x if amax==0. else x/amax




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

