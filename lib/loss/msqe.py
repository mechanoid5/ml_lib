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

class MSQE(Loss):
       
    def _estimate(self,output,target): 
        assert (output.shape==target.shape),f'incompatible shape output {output.shape} and target {target.shape}'
        d = (output-target).flatten()
        return d.dot(d.T)/len(output)    
    
    def _gradient(self,input_data,target): 
        d = (self._model.score(input_data) - target)[:,np.newaxis,:]
        p = self._model._partial(input_data)
        g = 2.*(p*d).sum(axis=0)/len(d)
        return self._norm(g)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

