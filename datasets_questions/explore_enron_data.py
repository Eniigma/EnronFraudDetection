#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

features = enron_data["COLWELL WESLEY"].keys()

#count = 0
#for i in enron_data:
	#print enron_data[i].keys()
	#break
	#if enron_data[i]['poi'] == True:
	#	count +=1
#print count

arr = featureFormat(enron_data, features)
print arr[0]