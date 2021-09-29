#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging

import numpy as np
# import numpy.random as rng

from .base import Loss


class PairRankerLoss(Loss):
    
    def __init__(self,model,sigma=.95):
        super().__init__(model=model)
        self._sigma = sigma
   
    def estimate(self,data,pair):
        first,second = pair[:,0],pair[:,1]
        s = self._estimate(
            self._model.predict(data[first,:]), 
            self._model.predict(data[second,:]), 
        )    
        self._history.append(s)
        return s

    def _estimate(self,first_score,second_score): 
        # минимизация количества отрицательный "отступов", 
        # т.е. пар у которых скор второго больше скора первого
        margin = first_score - second_score
        s = 1. + np.exp( -self._sigma * margin )
        with np.errstate(divide='ignore',invalid='ignore',):
            r = np.where( s>0, np.log(s), 0. )
        return r.sum()/len(r)

    def gradient(self,data,pair):
        first,second = pair[:,0],pair[:,1]
        return self._gradient( data[first,:], data[second,:], )
            

    def _gradient(self,x_first,x_second):
        margin = self._model.predict(x_first) - self._model.predict(x_second)
        dmargin = self._model._partial(x_first) - self._model._partial(x_second)
        
        d = 1. + np.exp( self._sigma * margin ) 
        with np.errstate(divide='ignore',invalid='ignore',):
            g = np.where( d!=0, (-self._sigma/d)*dmargin, 0. )
          
        return self._norm( g.sum(axis=0)/len(g)) [:,np.newaxis]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

