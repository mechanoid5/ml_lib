#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging

import numpy as np
import numpy.random as rng


class Activation:
    
    @staticmethod
    def transform(s): pass 
    
    @staticmethod
    def derivative(s): pass 
    

class BiSigmoid(Activation):
    
    @staticmethod
    def transform(s): return np.tanh(s) 
    
    @staticmethod
    def derivative(s): return 1./np.square(np.cosh(s))

    
    
class Sigmoid(Activation):
    
    @staticmethod
    def transform(s): 
        return 1./(1.+np.exp(-s) )
    
    @classmethod
    def derivative(cls,s): 
        o = cls.transform(s)
        return o*(1.-o)

    
    
class Softmax(Activation):
    
    @staticmethod
    def transform(s): 
        assert s.shape[1]>1,'incorrect activation argument size'
        es = np.exp(s)
        ess = es.sum(axis=1)[:,np.newaxis]
        with np.errstate(invalid='ignore',divide='ignore'):
            o = np.where( ess!=0., es/ess, 0. )
        return o
    
    @classmethod
    def derivative(cls,s): 
        o = cls.transform(s)
        return o*(1.-o) 
        
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)


