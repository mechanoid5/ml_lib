#!/usr/bin/env python
# coding: utf-8
# 
# кластеризатор MeanShift (ФорЭл)
#
# Евгений Борисов esborisov@sevsu.ru
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import logging
import sys 
import numpy as np
from numpy import random as rng
from tqdm import tqdm


# 1.задаём радиус кластера r
# 2.помечаем все точки как некластеризированые
# 3.выбираем рандомно точку из некластеризированых как центроид
# 4.собираем кластер радиуса r из некластеризированых точек вокруг центроида
# 5.пересчитываем центроид как среднее по кластеру
# 6.если позиция центроида изменилась то переход на п.4 иначе переход на след.п.
# 7.запоминаем центроид помечаем точки центройда как кластеризированые
# 8.если есть ещё некластеризированые точки то переход на п.3. иначе переход на след.п.
# 9.конец работы
 


from sklearn.metrics.pairwise import cosine_distances as metric
# from sklearn.metrics.pairwise import euclidean_distances as metric

class MeanShift:
    
    def __init__(self,metric=metric,centroid=None):
        self._distance = metric  # метрика, ф-ция расстояния
        self._centroid = [] if (centroid is None) else centroid # центроиды  
        
    def predict(self,X):
        assert len(self._centroid)>0, 'empty centroids, run fit() first'
        return np.argsort(self._distance(X,self._centroid))[:,0] 
    
    def fit(self,X,radius,delta=1e-5,verbose=True): 
        assert len(np.unique(X,axis=0))>1, 'dataset too small'

        # print('dataset:',X.shape)

        # radius - радиус кластера
        # delta  - минимально допустимый сдвиг центройда
        n_samples = X.shape[0] # количество точек
        idx_free = np.array(range(n_samples)) # номера некластеризированых точек
        n_free = len(idx_free)
        # пока есть некластеризированые точки формируем из них кластеры
        if verbose: pbar = tqdm(total = n_samples )
        while len(idx_free)>0: 
            idx_free = self._fit(X,idx_free,radius,delta)
            if verbose: pbar.update(  n_free-len(idx_free) )
            n_free = len(idx_free)
        if verbose: pbar.close()

        # print('centroid:', len(self._centroid) )

        self._centroid = np.vstack(self._centroid)            
        return self
                    
    
    @property
    def centroid(self): return self._centroid

    
    @staticmethod
    def sqe(x1,x2): # квадратичное отклонение
        d=x1-x2
        return d.dot(d.T).flatten()[0]
    
    
    def _fit(self,X,idx_free,radius,delta): 
        # сформировать кластер    
        centroid, idx_cluster = self._build_cluster(X, idx_free.copy(),radius,delta )
        # сохранить центроид и номера точек кластера
        self._centroid.append(centroid) 
        # выкинуть кластеризировнные точки из общего списка необработанных
        return  np.setdiff1d(idx_free,idx_cluster)
    
    
    
    def _build_cluster(self,X,idx_cluster,radius,delta,max_iter=100000):      
        # radius - радиус кластера
        # delta  - минимально допустимый сдвиг центройда
        # idx_cluster = self._idx_free.copy() # кандидаты в кластер
        centroid = X[[rng.choice(idx_cluster)]] # выбираем рандомно точку из некластеризированых как центроид

        for _ in range(max_iter):
            # растояния от центройда до всех точек-кандидатов
            d = self._distance(centroid,X[idx_cluster]) 
            # собираем кластер радиуса r из некластеризированых точек вокруг центроида
            idx_cluster = idx_cluster[ np.where(d<radius)[1] ]
            # пересчитываем центроид
            centroid_old = centroid.copy() 
            centroid = X[ idx_cluster ].mean(axis=0,keepdims=True)
            if self.sqe(centroid_old,centroid)<delta: break

        return centroid, idx_cluster # центроид и номера точек кластера       
    
            
 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__': sys.exit(0)   

