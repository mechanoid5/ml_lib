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
class FitBreakException(Exception):

   def __init__( self, message, weight ):
       super().__init__(message)
       # при срабатывании условия остановки обучения 
       # возвращаем веса сохранённые на предыдущем шаге 
       self._weight = weight 

   @property
   def weight(self): return self._weight

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class Breaking: # Bad

    def __init__(self, patience=1,):
       name = type(self).__name__
       assert patience>0,'patience must be greater than 0'
       self._patience = patience # глубина истории значений ф-ции потери для сравнения с текущим показателем
       self._weight = [] # история значений параметров системы глубины patience

    def check(self,loss): 
        if ( len(loss.history) > self._patience ): self._check(loss)
        self._push_weight(loss)
        return self

    def _check(self,loss): 
      if self._condition(loss): # проверяем условие
          self._raise(loss) # условие выполняется, прерываем цикл обучения
      return self

    def _raise(self,loss): 
        # условие выполняется, прерываем цикл обучения, возвращаем старые веса
        name = type(self).__name__
        msg = self._message() 
        raise FitBreakException(f'{name}: {msg}',self._weight[0]) # GD._init_breakers()
        #w = self._weight[0] if len(self._weight)>0 else loss.model.weight 
        #raise FitBreakException(f'{name}: {msg}',w) 

    def _condition(self,loss): return False

    def _message(self): return ' '

    def _push_weight(self,loss): # обновляем кэш параметров модели
        self._weight.append( loss.model.weight.copy() )
        if len(self._weight)>self._patience: self._weight = self._weight[1:]
        return self

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class ThresholdBreaking(Breaking): # прерывание по достижению порога значения ф-ции потери

    def __init__( self, value ):
        super().__init__()
        self._value = value # порог значений ф-ции потери для срабатывания прерывателя

    def _condition(self,loss): 
        return (loss.history[-1]<self._value) # достигнут порог значения ф-ции потери

    def _message(self): return 'loss min value has been reached'


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class GrowthBreaking(Breaking): # прерывание при росте ф-ции потери

    def __init__( self, patience=2, delta=0. ):
        super().__init__(patience)
        self._delta = delta # допустимый прирост ф-ции потери

    def _condition(self,loss): # проверяем рост ф-ции потери
        return ( (loss.history[-1]-loss.history[-self._patience]) > self._delta )

    def _message(self): return 'significant increase in the loss function has been detected'


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class DifferenceBreaking(Breaking): # прерывание при отсутствии существенной разницы в значениях ф-ции потери

    def __init__( self, patience=2, delta=0. ):
        super().__init__(patience)
        self._delta = delta # минимальное допустимое изменение ф-ции потери

    def _message(self): return 'loss values do not change'
 
    def _condition(self,loss): # проверяем рост ф-ции потери
        return ( np.abs(loss.history[-1]-loss.history[-self._patience]) < self._delta )


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

