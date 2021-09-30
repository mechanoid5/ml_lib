#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging
import numpy as np
import numpy.random as rng

class LearningRateAdjuster:
    
    def __init__(self,value):
        assert value!=0, 'try set zero learning rate'
        self._value=value
        self._history=[value]
        
    def adjust(self): pass

    def next(self):
        self._history.append(self._value)
        self._adjust()
        return self._history[-1]
    
    @property
    def history(self): return self._history


class ConstLRA(LearningRateAdjuster):
    
    def next(self):
        return self._value
    

class FactorLRA(LearningRateAdjuster):

    def __init__(self,start,bound,factor):
        super().__init__(start)
        assert factor!=0, 'try set zero learning rate factor'
        self._bound = bound
        self._factor = factor
    
    def _adjust(self):
        self._value = max( self._value*self._factor, self._bound )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

