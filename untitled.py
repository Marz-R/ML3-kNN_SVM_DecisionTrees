import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split

with open('data/dataset1.pkl', 'rb') as f:
    X, y = pkl.load(f)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


print(X[0].shape[0]) #(51, 2)
print(X[0])
