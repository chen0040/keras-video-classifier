# DataSet

* [UCF101 - Action Recognition Data Set](http://crcv.ucf.edu/data/UCF101.php)
* [CV Datasets on the web](http://www.cvpapers.com/datasets.html)
* [Deep Learning Datasets](http://deeplearning.net/datasets/)

# Install Some Libraries

To install opencv3 on Windows with Anaconda:

If you are using python 3.5 and below, install opencv3 using the following command:

```bash
conda install -c menpo opencv3
```

If you are using python 3.6, install opencv using the following command:

```bash
pip install opencv-python
```

# Split Data for Training and Test

```python

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold

seed = 7
np.random.seed(seed)

X = []
Y = [] 

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(64, input_dim=12, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.fit(X[train], Y[train], epochs=10, verbose=1)

```

# Model Evaluation

### Classification Model Evaluation

* The confusion matrix, which is a breakdown of predictions into a table showing correct predictions and the types of incorrect predictions made. Ideally, you will only see numbers in the diagonal, which means that all your predictions were correct!
* Precision is a measure of a classifier’s exactness. The higher the precision, the more accurate the classifier.
* Recall is a measure of a classifier’s completeness. The higher the recall, the more cases the classifier covers.
* The F1 Score or F-score is a weighted average of precision and recall.
* The Kappa or Cohen’s kappa is the classification accuracy normalized by the imbalance of the classes in the data.

```python

# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

y_test = [[0, 0, 1], [0, 1, 0]]
y_pred = [[1, 0, 0], [0, 1, 0]]

# Confusion matrix
confusion_matrix(y_test, y_pred)

# Precision 
precision_score(y_test, y_pred)

# Recall
recall_score(y_test, y_pred)

# F1 score
f1_score(y_test,y_pred)

# Cohen's kappa
cohen_kappa_score(y_test, y_pred)
```

### Regression Model Evaluation

* R2
* MSE: mean squared error
* MAE: mean absolute error

```python
from sklearn.metrics import r2_score

y_test = [[0, 0, 1], [0, 1, 0]]
y_pred = [[1, 0, 0], [0, 1, 0]]

r2_score(y_test, y_pred)

```

