библиотека линейных методов машинного обучения

Евгений Борисов <esborisov@sevsu.ru>


### модели

- lib.model.base.MLModel - базовый класс
- lib.model.linear.LinearModel - линейная модель
- lib.model.linear.LinearClassifier - линейный классификатор
- lib.model.linear.SLP - однослойная нейросеть (сигмойда)
- lib.model.linear.Softmax 


### ф-ции потери

- lib.loss.base.Loss - базовый класс
- lib.loss.msqe.MSQE - среднее квадратичное отклонение
- lib.loss.cce.CCE - категориальная кроссэнтропия
- lib.loss.logistic.LogisticLoss 
- lib.loss.bce.BCE - кроссэнтропия
- lib.loss.hinge.HingeLoss  
- lib.loss.ranker.PairRankerLogisticLoss - оценка разницы скоров для задачи парного ранжирования


### инициализация параметров моделей

- lib.initiator.base.InitiatorModel - базовый класс
- lib.initiator.linear.InitiatorLinearModel - инициализация линейной модели
- lib.initiator.linear.UniformInitiatorLinearModel - случайные малые значения равномерно распределенные 
- lib.initiator.linear.NormalInitiatorLinearModel-  случайные малые значения нормально распределенные 



### методы оптимизации ф-ций потери

[http://mechanoid.su/neural-net-backprop2.html](http://mechanoid.su/neural-net-backprop2.html)

- lib.optimizer.base.ModelOptimimizer - базовый класс
- lib.optimizer.gd.BaseGD - "ванильный" градиентный спуск
- lib.optimizer.gd.GD - градиентный спуск с регуляризацией и моментом
- lib.optimizer.gd.SGD - стохастический градиентный спуск
- lib.optimizer.gd.Adam 


### регуляризаторы для градиентного спуска

- lib.optimizer.regularizator.Regularization - базовый класс
- lib.optimizer.regularizator.RegularizationL1 - L1 регуляризатор
- lib.optimizer.regularizator.RegularizationL2 - L2 регуляризатор


### изменение скорости обучения в процессе градиентного спуска

- lib.optimizer.lrate.LearningRateAdjuster - базовый класс
- lib.optimizer.lrate.ConstLRA - постоянная скорость обучения
- lib.optimizer.lrate.FactorLRA - убывающая скорость обучения


### прерывание цикла обучения при выполнении условия

- lib.optimizer.breaker.Breaking - базовый класс
- lib.optimizer.breaker.FitBreakException - исключение прерывания цикла обучения
- lib.optimizer.breaker.ThresholdBreaking - прерывание по достижению порога значения потери
- lib.optimizer.breaker.GrowthBreaking - прерывание при росте ф-ции потери
- lib.optimizer.breaker.DifferenceBreaking -  прерывание при отсутвии существенной разницы в занчениях ф-ции потери


### оценка результатов моделей 

- lib.estimator.classifier.ClassifierEstimator - оценка результатов классификатора
- lib.estimator.classifier.BinnaryClassifierScoreThreshold - подбор оптимального порога скора модели для разделения двух классов
- lib.estimator.loss.LossPlot

