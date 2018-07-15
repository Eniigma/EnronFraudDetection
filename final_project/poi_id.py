#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
import pylab

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 
				'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 
				'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'other', 'from_this_person_to_poi', 'director_fees', 
				'deferred_income', 'long_term_incentive', 'from_poi_to_this_person'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#print data_dict['METTS MARK']

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#print features
#Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
#k best features:1.shared_receipt_with_poi, 2.from_poi_to_this_person, 3.loan_advances, 4.from_this_person_to_poi', 5.director_fees
selector = SelectKBest(f_classif, k=4)
features_new = selector.fit_transform(features, labels)
print features_new
#print features_list[1:], features_new[0]
#print selector.scores_, selector.get_support()

plt.figure(1)
plt.subplot(331)
plt.scatter(features_new[:,0],features_new[:,1],c=labels)
plt.subplot(332)
plt.scatter(features_new[:,0],features_new[:,2],c=labels)
plt.show()

#For visualizing the feature's relation between each other
for i in range(2):#range(len(features[0])):
	arr = [row[i] for row in features]
	for j in range(2):#range(i, len(features[0])):
		arr2 = [row1[j] for row1 in features]
		print arr
		print arr2
		pylab.scatter(arr,arr2, c=labels)
		plt.legend()
		plt.xlabel("feature: "+ features_list[i+1])
		plt.ylabel("feature: "+ features_list[j+1])
		plt.savefig('./plots/'+ features_list[i+1] + ' vs ' + features_list[j+1] +'.png')

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#classifiers: NaiveBayes, SVM, Decision Trees, KNeighbours, RandomForest, Adaboost, 
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
#clf = GaussianNB() #accuracy: 0.81,Precision: 0.15621,	Recall: 0.83450
clf = svm.SVC()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features_new, labels, test_size=0.3, random_state=42)

# My Training Code
#clf = clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#print clf.score(features_train, labels_train)
#print clf.score(features_test, labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
