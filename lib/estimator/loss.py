#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging

import numpy as np
import matplotlib.pyplot as plt

   
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class LossPlot:

    def __init__(self,loss_train,loss_val=None):
        self._loss_train = loss_train
        self._loss_val = loss_val
    
    def plot(self):
        plt.plot( self._loss_train.history,label='loss train')
        if not (self._loss_val is None):
            plt.plot( self._loss_val.history,label='loss validation')
        plt.grid(True)
        plt.legend(loc='upper right')
        return self
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)


