#import libraries
# python version
import sys
# scipy
import scipy
# numpy
import numpy
# matplotlib
import matplotlib
# pandas
import pandas
# scikit-learn
import sklearn

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load iris data set from git hub using pandas
fileName = "Machine Learning\iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(fileName, names=names)

#summarize data sheet
    #get dimesions of data set
print(dataset.shape)
    #show first 20 lines of data sheet
print(dataset.head(20))
    #stastical summary of of dataset attributes
print(dataset.describe())
    #number of instances of each class(type of iris)
print(dataset.groupby('class').size())

#data visualization
    #univariate plots
    #better understand each attribute
        #box and whisker plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
        #histograms
dataset.hist()
pyplot.show()

    #mulitvariates plots
    #better understand relationships between attributes
        #scatter-plot matrix
scatter_matrix(dataset)
pyplot.show()

#evaluate algorithms
    #create validation dataset
        #data that algorithems can't see to test how accurate algoritems are
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

    #build models
        #test 6 different algorithms
            #spot check algorithms
            #append each algoritem to models array
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
            #evaluate each model
results = []
names = []
for name, model in models: #loop through array of models
    #perform k-fold cross-validation
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    #add results and names to approiate arrays
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    #select best model
        #compare algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

#make predictions
    #make predictions using most accurate model determined above
    # Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

    #evaluate predictions made above
    #compare predictions to expected results
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
