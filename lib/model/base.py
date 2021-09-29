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
        
    def __init__(self):
        self._weight = None # вектор весов

    def predict(self,x):
        assert not(self._weight is None), 'try predict with empty weight'
        return self._predict(x)

    def _predict(self,x):  pass
    
    def save(self,file): 
        assert not(self._weight is None), 'try save empty weight'
        with open(file,'wb') as f: 
            pickle.dump( {'weight':self._weight,}, f )
        return self    
      
    def load(self,fname):
        assert len(fname)>0, 'load model file name is empty'
        fopen = gzip.open if fname[-3:]=='.gz' else open
        with fopen(file,'rb') as f: data = pickle.load(f)
        self._weight = data['weight']   
        return self
    
    def __partial(self,x): pass # вектор частных производных по параметрам модели
    


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

