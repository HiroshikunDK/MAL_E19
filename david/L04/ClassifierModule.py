#!/usr/bin/env python
# coding: utf-8

# ### Welcome to Hiro's Calculator
# 
# This calculator will contain simple function for mathematical operations

# In[4]:

from sklearn.base import BaseEstimator
import sklearn.metrics
import numpy as np
import sklearn.metrics
from sklearn.datasets import fetch_openml


class DummyClassifier(BaseEstimator):
	def fit(self, X, y=None):
		pass
	def predict(self, X):
		return np.zeros((len(X), 1), dtype=bool)

fig_cnt = 0

def MNIST_PlotDigit(data):
	import matplotlib
	import matplotlib.pyplot as plt
	global fig_cnt
	plt.figure(fig_cnt)
	fig_cnt += 1
	image = data.reshape(28, 28)
	plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
	plt.axis("off")
	plt.show()

def MNIST_GetDataSet():
	# Load data from https://www.openml.org/d/554
	X, y = fetch_openml('mnist_784', return_X_y=1) # needs to return X, y, replace '??' with suitable parameters! 
	#MNIST_PlotDigit(X[6])
	print(f"X.shape={X.shape}") # print X.shape= (70000, 28, 28)
	if X.ndim==3:
		print("reshaping X..")
		assert y.ndim==1
		X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
	assert X.ndim==2
	print(f"X.shape={X.shape}") # X.shape= (70000, 784)
	return (X, y)
	# Convert at scale (not always needed)
	#X = X / 255.
	
	
