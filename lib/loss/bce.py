#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging

import numpy as np
# import numpy.random as rng

from ..base import Loss

# LBCE: t log(p) - (1-t)log(1-p)
class BCE(Loss): # BinaryCrossEntropy, для бинарной классификации {0,1}
    
    def _estimate(self,output,target): 
        assert (output.shape==target.shape),f'incompatible shape output {output.shape} and target {target.shape}'
        t = target.flatten()
        o = output.flatten()
        with np.errstate(divide='ignore',invalid='ignore'):
            lg1 = np.where( o>0, np.log(o), 0. )
            lg2 = np.where( (1.-o)>0, np.log(1.-o), 0. )
        ce = -( t*lg1+(1.-t)*lg2 )
        return ce.sum() /len(ce)


    def _gradient(self,input_data,target): 
        p = self._model._partial(input_data)
        o = self._model.score(input_data)[:,np.newaxis,:]
        t = target[:,np.newaxis,:]
        d = o*(1.-o)
        with np.errstate(divide='ignore',invalid='ignore'):
            d = np.where( d!=0, (o-t)/d, 0. )
        g = (p*d).sum(axis=0)/len(target)
        return self._norm(g)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)



