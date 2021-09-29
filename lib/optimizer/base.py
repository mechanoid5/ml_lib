#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging

# import numpy as np
# import numpy.random as rng
# import pickle
# import gzip
# from tqdm import tqdm


class ModelOptimimizer:
    
    def __init__(self,loss):
        assert not(loss is None), 'try optimize empty loss'
        self._loss = loss
               
    def fit(self,data,n_epoch=2): pass
    


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

