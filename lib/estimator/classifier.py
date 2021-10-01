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

    def fit(self,X,target):
        s = self._model.score(X)
        fpr, tpr, thresholds = roc_curve( y_true=target, y_score=s )
        optimal_threshold_idx = np.argmax( np.abs(tpr-fpr) )
        self._model.score_threshold = thresholds[optimal_threshold_idx]
        return self._model


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
class ClassifierEstimator:
    
    def __init__(self,model):
        self._model = model

    def estimate(self,X,target,figsize=(12,5)):
        p = self._model.predict(X)
        print(classification_report(target, p ))
        self._roc_auc(X,target,figsize=figsize)
        return self
        
    def _roc_auc(self,X,target,figsize):
        s = self._model.score(X)
        out_size = s.shape[1]

        fig,ax = plt.subplots(1,out_size,figsize=figsize)
        ax = ax if out_size>1 else [ax,]
        for c in range(out_size):
            fpr, tpr, thresholds = roc_curve( y_true=target, y_score=s[:,c] )
            roc_auc = auc(fpr,tpr)
            self._plot_roc_auc(ax[c],fpr,tpr,roc_auc)
            ax[c].set_title(f'class {c}')

        plt.show()
 
   
    @staticmethod
    def _plot_roc_auc(ax,fpr,tpr,roc_auc):
        # if roc_auc<.5: roc_auc,fpr,tpr = 1.-roc_auc,tpr,fpr 
        ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.grid()

      


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)




#     def _roc_auc(self):
#         p = self._predict_score if (self._predict_score.shape[1]==1) else self._predict_score[:,1]
#         fpr, tpr, thresholds = roc_curve( self._target, p )
#         roc_auc = auc(fpr,tpr)
#         optimal_idx = np.argmax(tpr - fpr)
#         optimal_threshold = thresholds[optimal_idx]
#         return fpr, tpr, roc_auc, optimal_threshold
# 

#     ,predict_score,loss_train_history,loss_val_history=None):

#    
#     def _answer(self,optimal_threshold):
#         if  (self._predict_score.shape[1]==1): 
#             print(f'Threshold value is:{optimal_threshold}\n')
#             return (self._predict_score>optimal_threshold).astype(int) 
# 
#         print(f'For predict class argmax of output was used\n')
#         label = np.sort(np.unique(self._target))
#         return label[np.argmax(self._predict_score,axis=1)]
# 

#  def _roc_auc(self):
#         p = self._predict_score if (self._predict_score.shape[1]==1) else self._predict_score[:,1]
#         fpr, tpr, thresholds = roc_curve( self._target, p )
#         roc_auc = auc(fpr,tpr)
#         optimal_idx = np.argmax(tpr - fpr)
#         optimal_threshold = thresholds[optimal_idx]
#         return fpr, tpr, roc_auc, optimal_threshold

#    def _plot_loss_history(self,ax):
#         ax.plot( self._loss_train_history,label='loss train')
#         if not (self._loss_val_history is None):
#             ax.plot( self._loss_val_history,label='loss validation')
#         ax.grid(True)
#         ax.legend(loc='upper right')
#         return self
#     

