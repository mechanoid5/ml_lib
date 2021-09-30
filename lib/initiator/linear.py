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

class InitiatorLinearModel(InitiatorModel):

    def __init__(self,input_size,output_size=1):
        self._size = (input_size, output_size)      

class UniformInitiatorLinearModel(InitiatorLinearModel):

    def get(self):  return rng.uniform( size=self._size, high=.01, )

class NormalInitiatorLinearModel(InitiatorLinearModel):

    def get(self):  return rng.normal( size=self._size, scale=.01, )


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

