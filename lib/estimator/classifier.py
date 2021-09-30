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

class ClassifierEstimator:
    
    def __init__(self,target,predict_score,loss_history):
        self._target = target
        self._predict_score = predict_score
        self._loss_history = loss_history
        
    def _roc_auc(self):
        fpr, tpr, thresholds = roc_curve( self._target, self._predict_score )
        roc_auc = auc(fpr,tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return fpr, tpr, roc_auc, optimal_threshold


    @staticmethod
    def _plot_roc_auc(ax,fpr,tpr,roc_auc):
        ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.grid()

    def _plot_loss_history(self,ax):
        ax.plot( self._loss_history,label='loss history')
        ax.grid(True)
        ax.legend(loc='upper right')
        return self
    
    def estimate(self,figsize=(12,5)): 
        fpr, tpr, roc_auc, optimal_threshold = self._roc_auc()
        print(f'Threshold value is:{optimal_threshold}\n')
        o =  (self._predict_score>optimal_threshold).astype(int)
        print(classification_report(self._target,o))
        fig,(ax0,ax1) = plt.subplots(1,2,figsize=figsize)
        self._plot_loss_history(ax0)
        self._plot_roc_auc(ax1,fpr,tpr,roc_auc)
        plt.show()
        

# ClassifierEstimator(
#         target=y,
#         predict_score=p,
#         loss_history=lsh,
#     ).estimate(figsize=(12,5))  


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)

