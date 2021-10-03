#!/usr/bin/env python
# coding: utf-8

# модели машинного обучения
# 
# Евгений Борисов  <esborisov@sevsu.ru>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class BinnaryClassifierScoreThreshold:
   
    def __init__(self,model):
        self._model = model

    @staticmethod
    def _optimal_threshold(tpr,fpr,thresholds):
        return thresholds[ np.argmax( np.abs(tpr-fpr) )  ]

    def fit(self,X,target):
        s = self._model.score(X)
        fpr, tpr, thresholds = roc_curve( y_true=target, y_score=s )
        self._model.score_threshold = self._optimal_threshold(tpr,fpr,thresholds)
        return self._model


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class ClassifierEstimator:
    
    def __init__(self,model):
        self._model = model

    def estimate(self,X,target,figsize=(12,5)):
        s = self._model.score(X)
        p = self._model._solve_score(s)

        if self._model.shape[1]>1: 
            print('predict is argmax output score vector')
        else:
            print(f'score threshold to predict is {self._model.score_threshold}')
        print(classification_report(target, p ))

        self._roc_auc(s,target,figsize=figsize)
        return self
        
    
    def _roc_auc(self,s,target,figsize):
        out_size = s.shape[1]
        fig,ax = plt.subplots(1,out_size,figsize=figsize)
        if out_size==1:
            self._plot_roc_auc(ax,target,y_score=s[:,0])
            ax.set_title(f'class 1')
        else:
            for c in range(out_size):
                self._plot_roc_auc(ax[c],target=(target==c).astype(int),y_score=s[:,c])
                ax[c].set_title(f'output {c}')

        plt.show()
 
#     def _roc_auc(self,s,target,figsize):
#         out_size = s.shape[1]
#         if out_size==1:
#             fig,ax = plt.figure(figsize=figsize)
#         else:
#             fig,ax = plt.subplots(1,out_size,figsize=figsize)
# 
#         for c in range(out_size):
#             self._plot_roc_auc(ax[c],target=(target==c).astype(int),y_score=s[:,c])
#             ax[c].set_title(f'output {c}')
    

   
    @staticmethod
    def _plot_roc_auc(ax,target,y_score):
        fpr, tpr, thresholds = roc_curve( y_true=target, y_score=y_score )
        roc_auc = auc(fpr,tpr)
        optimal_threshold = BinnaryClassifierScoreThreshold._optimal_threshold(tpr,fpr,thresholds)
        label='ROC AUC = %0.2f\nOptimal Threshold = %0.3f' %( roc_auc,optimal_threshold)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=label)
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.grid()
        return optimal_threshold

      


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)
