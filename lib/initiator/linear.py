#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging

import numpy as np
import numpy.random as rng

from .base import InitiatorModel

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class InitiatorLinearModel(InitiatorModel):

    def __init__(self,input_size,output_size=1):
        self._size = (input_size, output_size)      

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class UniformInitiatorLinearModel(InitiatorLinearModel): 

    def __init__(self,input_size,output_size=1,high=.1):
        super().__init__(input_size=input_size, output_size=output_size)
        assert (high>0.), 'UniformInitiatorLinearModel: incorrect high'
        self._high = high

    def get(self): return rng.uniform(size=self._size,high=self._high)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class NormalInitiatorLinearModel(InitiatorLinearModel):

    def __init__(self,input_size,output_size=1,scale=.1):
        super().__init__(input_size=input_size, output_size=output_size)
        assert (scale>0.), 'NormalInitiatorLinearModel: incorrect scale'
        self._scale = scale

    def get(self): return rng.normal(size=self._size,scale=self._scale,)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

