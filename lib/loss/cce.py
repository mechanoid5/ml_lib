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


# CCE: t log(s)
class CCE(Loss): # CategoricalCrossEntropy
    
    def _estimate(self,output,target):
        o = output.flatten()
        t = target.flatten()
        with np.errstate(divide='ignore',invalid='ignore'):
            lg = np.where(o>0., np.log(o), 0.)
        return -t.dot( lg.T )/len(target)

    def _gradient(self,input_data,target): 
        p = self._model._partial(input_data)
        o = self._model.score(input_data)[:,np.newaxis,:]
        t = target[:,np.newaxis,:]
        with np.errstate(divide='ignore',invalid='ignore'):
            d = np.where(o!=0.,(1./o)*p , 1. )
        g = (d*t).sum(axis=0)/len(target)
        return self._norm(g)
 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)



