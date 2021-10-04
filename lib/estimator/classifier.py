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




class ClassifierEstimator:
    
    def __init__(self,model):
        self._model = model

    def estimate(self,X,target_score,plot_width=4):
        score = self._model.score(X)
        n_out = score.shape[1]
        if n_out>1: 
            self._report_multi(target_score, score, plot_width=plot_width)
            return self
        self._report_binary(target_score, score, plot_width=plot_width)
        return self

    @classmethod
    def _report_multi(cls, target_score, score, plot_width=4):
        labels = np.array(sorted(set(target_score.flatten())))
        n_classes = target_score.shape[1]
        print( '- - - - - - - - - - - - - - - - - - - - - -')    
        print( 'all classes:' )
        print( 'predict select argmax scores\n' )
        print( classification_report( np.argmax(target_score,axis=1), np.argmax(score,axis=1) ) )
        plot_height = plot_width
        figsize = ((plot_width+1)*n_classes,plot_height)
        fig,ax = plt.subplots(1,n_classes,figsize=figsize)   
        for nc in range(n_classes):
            t = target_score[:,nc]
            s = score[:,nc]
            fpr, tpr, thresholds = roc_curve( y_true=t, y_score=s )
            roc_auc = auc(fpr,tpr)
            opt_trs = BinnaryClassifierScoreThreshold._optimal_threshold(tpr,fpr,thresholds)
            p = labels[ (s>opt_trs).astype(int) ]
            print( '- - - - - - - - - - - - - - - - - - - - - -')
            print( f'\nclass {nc}:')
            print( f'score threshold:{opt_trs}\n')
            print( classification_report(t,p))
            cls._plot_roc_auc(ax[nc],fpr,tpr,roc_auc,opt_trs,nc)
        plt.show()


    @classmethod
    def _report_binary(cls,target_score, score, plot_width=4):
        labels = np.array(sorted(set(target_score.flatten())))
        plot_height = plot_width
        figsize = (plot_width,plot_height)
        fig,ax = plt.subplots(figsize=figsize)   
        fpr, tpr, thresholds = roc_curve( y_true=target_score, y_score=score )
        roc_auc = auc(fpr,tpr)
        opt_trs = BinnaryClassifierScoreThreshold._optimal_threshold(tpr,fpr,thresholds)
        p = labels[ (score>opt_trs).astype(int) ]
        print( f'score threshold:{opt_trs}\n')
        print( classification_report(target_score,p))
        cls._plot_roc_auc(ax,fpr,tpr,roc_auc,opt_trs,1)


    @staticmethod
    def _plot_roc_auc(ax,fpr,tpr,roc_auc,opt_trs,class_num):
        label='ROC AUC = %0.2f\nOptimal Threshold = %0.3f' %( roc_auc,opt_trs)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=label)
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.set_title(f'class {class_num}')
        ax.grid()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)
