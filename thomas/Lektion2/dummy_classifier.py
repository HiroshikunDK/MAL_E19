#!/usr/bin/env python
# coding: utf-8

# # ITMAL Exercise
# 
# REVISIONS||
# ---------||
# 2018-1219| CEF, initial.                  
# 2018-0206| CEF, updated and spell checked. 
# 2018-0208| CEF, minor text update.
# 2018-0305| CEF, updated with SHN comments.
# 2019-0902| CEF, updated for ITMAL v2.
# 2019-0904| CEF, updated and added conclusion Q.
# 
# ## Implementing a dummy classifier with fit-predict interface
# 
# We begin with the MNIST data-set and will reuse the data loader from Scikit-learn. Next we create a dummy classifier, and compare the results of the SGD and dummy classifiers using the MNIST data...
# 
# #### Qb  Load and display the MNIST data
# 
# There is a `sklearn.datasets.fetch_openml` dataloader interface in Scikit-learn. You can load MNIST data like 
# 
# ```python
# from sklearn.datasets import fetch_openml
# # Load data from https://www.openml.org/d/554
# X, y = fetch_openml('mnist_784',??) # needs to return X, y, replace '??' with suitable parameters! 
# # Convert at scale (not always needed)
# #X = X / 255.
# ```
# 
# but you need to set parameters like `return_X_y` and `cache` if the default values are not suitable! 
# 
# Check out the documentation for the `fetch_openml` MNIST loader, try it out by loading a (X,y) MNIST data set, and plot a single digit via the `MNIST_PlotDigit` function here (input data is a 28x28 NMIST subimage)
# 
# ```python
# %matplotlib inline
# def MNIST_PlotDigit(data):
#     import matplotlib
#     import matplotlib.pyplot as plt
#     image = data.reshape(28, 28)
#     plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
#     plt.axis("off")
# ```
# 
# Finally, put the MNIST loader into a single function called `MNIST_GetDataSet()` so you can resuse it later.

# In[4]:


fig_cnt = 0
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import fetch_openml

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
    return (X, y)
    # Convert at scale (not always needed)
    #X = X / 255.

X, y = MNIST_GetDataSet()
MNIST_PlotDigit(X[6])
MNIST_PlotDigit(X[7])


# #### Qb  Add a Stochastic Gradient Decent [SGD] Classifier
# 
# Create a train-test data-set for MNIST and then add the `SGDClassifier` as done in [HOLM], p82.
# 
# Split your data and run the fit-predict for the classifier using the MNIST data.(We will be looking at cross-validation instead of the simple fit-predict in a later exercise.)
# 
# Notice that you have to reshape the MNIST X-data to be able to use the classifier. It may be a 3D array, consisting of 70000 (28 x 28) images, or just a 2D array consisting of 70000 elements of size 784.
# 
# A simple `reshape()` could fix this on-the-fly:
# ```python
# X, y = MNIST_GetDataSet()
# 
# print(f"X.shape={X.shape}") # print X.shape= (70000, 28, 28)
# if X.ndim==3:
#     print("reshaping X..")
#     assert y.ndim==1
#     X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
# assert X.ndim==2
# print(f"X.shape={X.shape}") # X.shape= (70000, 784)
# ```
# 
# Remember to use the category-5 y inputs
# 
# ```python
# y_train_5 = (y_train == '5')    
# y_test_5  = (y_test == '5')
# ```
# instead of the `y`'s you are getting out of the dataloader...
# 
# Test your model on using the test data, and try to plot numbers that have been categorized correctly. Then also find and plots some misclassified numbers.

# In[5]:



from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)

#X, y = MNIST_GetDataSet()

print(f"X.shape={X.shape}") # print X.shape= (70000, 28, 28)
if X.ndim==3:
    print("reshaping X..")
    assert y.ndim==1
    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
assert X.ndim==2
print(f"X.shape={X.shape}") # X.shape= (70000, 784)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == '5')    
y_test_5  = (y_test == '5')

sgd_clf.fit(X_train, y_train_5)

range_var = 1000
plots = 5
cnt = 0
print("Some correct predictions:")
test = sgd_clf.predict(X_test[0:range_var]) == y_test_5[0:range_var]
for n in range(range_var):
    if test[n] == True and cnt < 5:
        MNIST_PlotDigit(X_test[n])
        cnt += 1
    
cnt = 0   
print("Some incorrect predictions:")

for n in range(range_var):
    if test[n] != True and cnt < 5:
        cnt += 1
        MNIST_PlotDigit(X_test[n])

#print(sgd_clf.predict(X_test[0:10]))
#print(y_test_5[0:10])
#print(test)
#print(X_train.shape)
print("Accuracy score of SGDClassifier on training set: ", sklearn.metrics.accuracy_score(y_train_5, sgd_clf.predict(X_train)))
print("Accuracy score of SGDClassifier on test set: ", sklearn.metrics.accuracy_score(y_test_5, sgd_clf.predict(X_test)))


# #### Qc Implement a dummy binary classifier
# 
# Now we will try to create a Scikit-learn compatible estimator implemented via a python class. Follow the code found in [HOML], p84, but name you estimator `DummyClassifier` instead of `Never5Classifyer`.
# 
# Here our Python class knowledge comes into play. The estimator class hierarchy looks like
# 
# <img src="https://itundervisning.ase.au.dk/E19_itmal/L02/Figs/class_base_estimator.png" style="width:500px">
# 
# All Scikit-learn classifiers inherit from `BaseEstimator` (and possibly also `ClassifierMixin`), and they must have a `fit-predict` function pair (strangely not in the base class!) and you can actually find the `sklearn.base.BaseEstimator` and `sklearn.base.ClassifierMixin` python source code somewhere in you anaconda install dir, if you should have the nerves to go to such interesting details.
# 
# But surprisingly you may just want to implement a class that contains the `fit-predict` functions, ___without inheriting___ from the `BaseEstimator`, things still work due to the pythonic 'duck-typing': you just need to have the class implement the needed interfaces, obviously `fit()` and `predict()` but also the more obscure `get_params()` etc....then the class 'looks like' a `BaseEstimator`...and if it looks like an estimator, it _is_ an estimator (aka. duck typing).
# 
# Templates in C++ also allow the language to use compile-time duck typing!
# 
# > https://en.wikipedia.org/wiki/Duck_typing
# 
# Call the fit-predict on a newly instantiated `DummyClassifier` object, and find a way to extract the accuracy `score` from the test data. You may implement an accuracy function yourself or just use the `sklearn.metrics.accuracy_score` function. 
# 
# Finally, compare the accuracy score from your `DummyClassifier` with the scores found in [HOML] "Measuring Accuracy Using Cross-Validation", p.83. Are they comparable? 

# In[ ]:


from sklearn.base import BaseEstimator
import sklearn.metrics
import numpy as np

class DummyClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train,y_train_5)
print("Accuracy score of dummy classifier: ", sklearn.metrics.accuracy_score(y_train_5, dummy_clf.predict(X_train)))


# ####Discussion
# The accuracy scores in [HOML] are [0.91125, 0.90855, 0.90915] and are therefore comparable to the accuracy score with the dummy classifier, which we found to be 0.90965.

# ### Qd Conclusion
# 
# Now, conclude on all the exercise above. 
# 
# Write a short textual conclusion (max. 10- to 20-lines) that extract the _essence_ of the exercises: why did you think it was important to look at these particular ML concepts, and what was our overall learning outcome of the exercises (in broad terms).

# In the exercise we implemented a dummy classifier with the basic interface expected of a classifier in SKLearn including including inheriting from the   `BaseEstimator` and implementing the necessary `fit()` and `predict()` functions.
# 
# Having knowledge about how classifiers and models in SKLearn are implemented is imperative if we are to implement our own machine learning models using SKLearn and furthermore beneficial for understanding the interface used in SKLearn in general.
# 
# In the exercise we also tried to use the `DummyClassifier` for classifying the well-known MNIST dataset containing hand-written digits as well as using a stochastic gradient descent classifier from SKLearn. We tried to apply pre-processing to the dataset to be able to use it with the `DummyClassifier`. The goal was to predict whether a number was a 5 or not. 
# The pre-processing very important when we are going to work with datasets in the rest of the course.
# 
# An important part of the exercise is also to highlight that a high accuracy score doesn't necessarily mean that the classifier is doing well when the dataset is skewed, which is the case for the MNIST dataset when our goal is to predict whether a hand-written number is a 5 as roughly 10% of the dataset consists of the digit 5.  
