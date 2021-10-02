#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging

import numpy as np
# import numpy.random as rng



class FitBreakException(Exception): pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class Breaking: # Bad
    
    def check(self,loss): 
        return self
  
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class ThresholdBreaking(Breaking): # прерывание по достижению порога значения ф-ции потери

    def __init__( self, value ):
        self._value = value # порог значений ф-ции потери для срабатывания прерывателя

    def check(self,loss):
        if (loss.history[-1]<self._value) # достигнут порог значения ф-ции потери 
            raise FitBreakException('ThresholdBreaking: loss min value has been reached') 
        return self

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class GrowthBreaking(Breaking): # прерывание при росте ф-ции потери

    def __init__( self, patience=2, delta=0. ):
        assert patience>1,'patience must be greater than 1'
        self._patience = patience # глубина истории значений ф-ции потери для сравнения с текущим показателем
        self._delta = delta # допустимый прирост ф-ции потери

    def check(self,loss): # проверяем рост ф-ции потери
        if ( len(loss.history) < self._patience ): return self
        if ( (loss.history[-1]-loss.history[-self._patience]) > self._delta ):
            raise FitBreakException('GrowthBreaking: significant increase in the loss function has been detected')
        return self

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class DifferenceBreaking(Breaking): # прерывание при отсутвии существенной разницы в занчениях ф-ции потери

    def __init__( self, patience=2, delta=0. ):
        assert patience>1,'patience must be greater than 1'
        self._patience = patience # глубина истории значений ф-ции потери для сравнения с текущим показателем
        self._delta = delta # минимальное допустимое изменение ф-ции потери

    def check(self,loss):# проверяем изменения значений ф-ции потери
        if ( len(loss.history) < self._patience ): return self
        if ( np.abs(loss.history[-1]-loss.history[-self._patience]) < self._delta ):
            raise FitBreakException('DifferenceBreaking: loss values do not change')
        return self



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

