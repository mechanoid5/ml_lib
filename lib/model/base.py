#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging

import numpy as np
import numpy.random as rng
# import pickle
# import gzip
# from tqdm import tqdm


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class MLModel:
        
    def __init__(self,initiator):
        self._weight = None # параметры модели
        self._initiator = initiator # процедура инициализации параметров модели
        self._reset() # инициализируем параметры модели

    @property
    def weight(self): return self._weight

    @weight.setter
    def weight(self, value): self._weight = value         

    def _reset(self): pass # инициализируем параметры модели

    def reset(self):
        self._reset()
        return self

    def predict(self,x): 
        assert not(self._weight is None), 'try predict with empty weight'
        return self._predict(x)

    def _predict(self,x): # интерпретация выхода модели 
        return self._score(x)

    def _score(self,x): pass # генерируем выход модели

    def _save(self): # пакуем параметры модели
        return {'weight':self._weight,}

    def save(self,file): # сохраняем параметры модели
        assert not(self._weight is None), 'try save empty weight'
        with open(file,'wb') as f: pickle.dump( self._save(), f )
        return self    
      
    def _load(self,data): # распаковываем считанные параметры модели
        self._weight = data['weight'] 
        return self    

    def load(self,fname): # считываем сохранённые параметры модели
        assert len(fname)>0, 'load model file name is empty'
        fopen = gzip.open if fname[-3:]=='.gz' else open
        with fopen(file,'rb') as f: data = pickle.load(f)
        self._load(data)
        return self
    
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

