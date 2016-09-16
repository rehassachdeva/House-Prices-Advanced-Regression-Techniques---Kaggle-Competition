

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets
from sklearn import tree
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB

#1. Load dataset
iris = datasets.load_iris()

#2.Select features
X = iris.data[:, 0:2] 
y = iris.target
print 
# Explore features visually
#plt.plot(y, X[:,2], 'ro')
#plt.ylabel('')
#plt.xlabel('class label')
#plt.show()

n_features = X.shape[1]

#print X[60],y[60]
C=1.0
#3. Choose classifier
classifier = GaussianNB();
#classifier = tree.DecisionTreeClassifier()
#classifier = SVC(kernel='linear', C=C, probability=True, random_state=0);

#4. Train the classifier
classifier = classifier.fit(X, y)

#5. Test the classifier
y_pred = classifier.predict(X)

classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
print("classif_rate for classifier: %f " % (classif_rate))

#print classifier.predict([5.2,2.5])

#Split dataset
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
#clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
#print "score on split dataset:",clf.score(X_test, y_test)  

#cross validation
#clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
#scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
#print "Scores on cross_validation: ",scores

