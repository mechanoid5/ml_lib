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
class Regularization:

    def __init__( self, rho ):
        assert rho!=0.,'zero regularization parameter'
        self._rho=rho

    def transform(self,weight): 
        return weight
        

class RegularizationL1(Regularization):

    def transform(self,weight): 
        return self._rho*np.abs(weight)

        
class RegularizationL2(Regularization):

    def transform(self,weight): 
        return self._rho*np.square(weight)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

