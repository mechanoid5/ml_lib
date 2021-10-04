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

class HingeLoss(Loss):  # для бинарной класcификации {-1,1}

    def _estimate(self,output,target):
        l = 1. - target*output
        return np.where( l>0., l, .0 ).sum()/len(target)

    def _gradient(self,input_data,target):
        o = self._model.score(input_data)
        p = self._model._partial(input_data)
        g = (np.where( target*o<1, -target*input_data, 0, )[:,:,np.newaxis])*p
        g = (g.sum(axis=0)/len(g))
        return self._norm(g)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

