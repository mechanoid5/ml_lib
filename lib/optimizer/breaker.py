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
    
    def check(self,loss): return self

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class EarlyStopping(Breaking): # прерыватель цикла градиентного спуска

    def __init__( self, bound=None, min_delta=0., max_delta=None, patience=2 ):
        self._min_delta=min_delta # минимальное отклонение потери для срабатывания прерывателя
        self._max_delta=max_delta # максимально допустимый рост потери для срабатывания прерывателя
        assert patience>1,'patience must be greater than 1'
        self._patience=patience # глубина истории значений ф-ции потери для сравнения с текущим показателем
        self._bound = bound # порог значений ф-ции потери для срабатывания прерывателя

    def _check_bound(self,loss): # проверка на достижение порога минимального значения ф-ции потери
        return False if (self._bound is None) else (loss.history[-1]<self._bound)

    def _check_delta(self,loss): # проверка на достижение минимального изменения значений ф-ции потери
        return False if (len(loss.history)<self._patience) else ( np.abs(loss.history[-1]-loss.history[-self._patience])<self._min_delta )

    def _check_increase(self,loss): # проверка на возрастание значений ф-ции потери
        return (
            False 
            if ( (len(loss.history)<self._patience) or (self._max_delta is None) )
            else ( (loss.history[-1]-loss.history[-self._patience])>self._max_delta )
        )

    def check(self,loss):
        if self._check_bound(loss): # достигнут порог значения ф-ции потери 
            raise FitBreakException('EarlyStopping: loss min value bound has been reached') 
        if self._check_delta(loss): # изменения значений ф-ции потери минимально 
            raise FitBreakException('EarlyStopping: value loss min difference has been detected') 
        if self._check_increase(loss): # рост ф-ции потери
            raise FitBreakException('EarlyStopping: significant increase in the loss function has been detected') 
        return self
 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

