библиотека базовых методов машинного обучения

Евгений Борисов <esborisov@sevsu.ru>

### модели

- lib.model.base.MLModel - базовый класс
- lib.model.linear.LinearModel - линейная модель
- lib.model.linear.LinearRegression - линейная регрессия
- lib.model.linear.LogisticRegression - логистическакя регрессия
- lib.model.linear.Softmax 


### ф-ции потери

- lib.loss.base.Loss - базовый класс
- lib.loss.msqe.MSQE - среднее квадратичное отклонение
- lib.loss.crossentropy.BCE - кроссэнтропия
- lib.loss.crossentropy.CCE - категориальная кроссэнтропия
- lib.loss.ranker.PairRankerLoss - оценка разницы скоров для задачи парного ранжирования


### инициализация параметров моделей

- lib.initiator.base.InitiatorModel - базовый класс
- lib.initiator.linear.InitiatorLinearModel - инициализация линейной модели
- lib.initiator.linear.UniformInitiatorLinearModel - случайные малые значения равномерно распределенные на ( 0, 0.1 )
- lib.initiator.linear.NormalInitiatorLinearModel-  случайные малые значения нормально распределенные на ( -0.1 , 0.1 )



### методы оптимизации ф-ций потери

- lib.optimizer.base.ModelOptimimizer - базовый класс
- lib.optimizer.gd.BaseGD - "ванильный" градиентный спуск
- lib.optimizer.gd.GD - градиентный спуск с регуляризацией и моментом
- lib.optimizer.gd.SGD - стохастический градиентный спуск


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
- lib.optimizer.breaker.EarlyStopping - прерывание по достижению порога значения потери
- lib.optimizer.breaker.FitBreakException - исключение прерывания цикла обучения



### оценка результатов моделей 

- lib.estimator.classifier.ClassifierEstimator - оценка результатов классификатора
