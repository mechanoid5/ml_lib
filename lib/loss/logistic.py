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

class LogisticLoss(Loss):  # для бинарной класcификации {-1,1}

    def __init__(self,model,normalize_gradient=True,sigma=.95):
        super().__init__(model=model,normalize_gradient=normalize_gradient)
        self._sigma = sigma

    def _estimate(self,output,target):
        margin = target - output
        s = 1. + np.exp( -self._sigma * margin )
        with np.errstate(divide='ignore',invalid='ignore',):
            r = np.where( s>0, np.log(s), 0. )
        return r.sum()/len(r)

    def _gradient(self,input_data,target):
        margin = target - self._model.score(input_data)
        dmargin = target[:,np.newaxis] - self._model._partial(input_data) 
        d = 1. + np.exp( self._sigma * margin )[:,np.newaxis]
        with np.errstate(divide='ignore',invalid='ignore',):
            g = np.where( d!=0, (-self._sigma/d)*dmargin, 0. )
        return self._norm( g.sum(axis=0)/len(g))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

