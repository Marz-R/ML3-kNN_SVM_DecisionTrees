import numpy as np
from sklearn.base import BaseEstimator
from collections import Counter

class KNearestNeighborsClassifier(BaseEstimator):
  def __init__(self, k=1):
    self.k = k

  def fit(self, X, y):
    # TODO IMPLEMENT ME
    # store X and y
    self.y_train = y #output
    self.X_train = X #input
    return self

  def score(self, X, y):
    y_pred = self.predict(X)
    return np.mean(y_pred == y)

  def predict(self, X):
    # TODO: assign class labels
    # useful numpy methods: np.argsort, np.unique, np.argmax, np.count_nonzero
    # pay close attention to the `axis` parameter of these methods
    # broadcasting is really useful for this task!
    # See https://numpy.org/doc/stable/user/basics.broadcasting.html
    
    # find distances
    y_pred = np.empty(X.shape[0])

    for i in range(X.shape[0]):
      distance = [] #np.empty([X.shape[0], X_train.shape[0]])
      votes = []

      for j in range(self.X_train.shape[0]):
        temp = np.linalg.norm(X[i] - self.X_train[j])
        distance.append([temp, j])

      # get k neighbors
      distance = np.argsort(distance, axis=0)
      distance = distance[0:self.k]

      # predict
      for c, d in distance:
        votes.append(self.y_train[d])

      ans = Counter(votes).most_common(1)[0][0]

      for e in range(y_pred.shape[0]):
        y_pred[e] = ans

    return y_pred #class 0 or 1
