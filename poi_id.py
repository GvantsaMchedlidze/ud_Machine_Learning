#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
features_list = ['poi','bonus','salary','total_payments','deferral_payments','deferred_income',
                 'expenses','long_term_incentive',
                 'restricted_stock','restricted_stock_deferred','total_stock_value',
                 'shared_receipt_with_poi','fraction_from_poi','fraction_to_poi','director_fees']
                 
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
data_dict.pop('LOCKHART EUGENE E',0)

### Task 3: Create new feature(s)

### comput fraction from from_poi_to_this_person and from_this_person_to_poi
def computeFraction( poi_messages, all_messages ):

    fraction = 0.
    
    if poi_messages!="NaN":
        if all_messages!="NaN":
            fraction = (poi_messages*1.0)/(all_messages*1.0)

        else:
            return 0
    else:
        return 0

    return fraction

### create dictionary of new variables
submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

    
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi
    
    
#####################
### add new features to datat dict

my_data = {}
my_data.update(submit_dict)  # Modifies my_data, not data
my_data.update(data_dict) 

### Store to my_dataset for easy export below.
my_dataset = my_data

### Extract features and labels from dataset for local testing
#   NaN values will be replaced with 0 and features with all 0 values will be removed

data = featureFormat(my_dataset, features_list, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# Stratified ShuffleSplit cross-validator. 
# Provides train/test indices to split data in train/test sets.
# This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, 
# which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class.

# NaiveBayes
from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()

# SVM
from sklearn.svm import SVC

svm_clf = SVC()

# DecisionTree
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()

# RandomForest
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=25)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

ab_clf = AdaBoostClassifier()

# PCA
from sklearn.decomposition import PCA

pca=PCA()

# KBest
from sklearn.feature_selection import SelectKBest

skb = SelectKBest(k=5)

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features=scaler.fit_transform(features)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split, cross_val_score

sk_fold = StratifiedShuffleSplit(labels, 100, random_state = 42)

# Having handle to validation scores
from sklearn.metrics import (precision_score, recall_score,f1_score)

# pipelines and grid serch
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

### Define function which returns best algorithm with tuned parameters
def grid_search():
    
    #FeatureUnion
    from sklearn.pipeline import FeatureUnion
    
    combined_features = FeatureUnion([("kbest", skb), ("pca", pca)])
    
   # NaiveBayes
    pipeline0 = Pipeline(steps=[('scaling',scaler),('reduce_dim', pca), ("nb", nb_clf)])
    
    parameters0 = {
    'reduce_dim': [skb],
    'reduce_dim__k': range(5,15),
    }
    
    # SVM
    pipeline1 = Pipeline(steps=[('scaling',scaler),('reduce_dim', pca),('svm', svm_clf)])
    
    parameters1 = {
    'svm__C': [1., 10, 100, 10000],
    'svm__kernel': ['rbf', 'poly'],
    'reduce_dim': [skb],
    'reduce_dim__k': range(5,15),
    }
    
    #DecisionTree
    pipeline2 = Pipeline(steps=[('scaling',scaler),('reduce_dim', pca), ('dt', dt_clf)])
    
    
    parameters2 = {
    'dt__criterion': ['entropy'],
    'dt__min_samples_split': [2, 3, 4, 5],
    'dt__max_depth': [None, 2],
    'reduce_dim': [skb],
    'reduce_dim__k': range(8,13),
    }
    
    
    #AdaBoost
    pipeline4 = Pipeline(steps=[('scaling',scaler),('reduce_dim', pca), ('ab', ab_clf)])
    
    parameters4 = {
    'ab__n_estimators': [25, 50, 100],
    'reduce_dim': [skb],
    'reduce_dim__k': range(5,15),
    }
    
    
    # array of pipliens and parameters
    pars = [parameters0, parameters1, parameters2, parameters4]
    pips = [pipeline0, pipeline1, pipeline2,  pipeline4]
    
    
    # loop of pips and pars to get best algorithm 
    print ("starting Gridsearch")
    
    ind = 0. #index for best algorithm
    gs = {}  # array for best estimator algorithm
    #for i in range(0,len(pars)):
    for i in range(2,3):
      
        gs[i] = GridSearchCV(pips[i], pars[i], verbose=1, cv=sk_fold, scoring = 'f1')
        gs[i].fit((features), (labels))
        
        print (gs[i].best_estimator_)
        # save index of best algorithm
        #if i!=0. and gs[i].best_score_>gs[i-1].best_score_ :
        #  ind = i
        #else:
        #  ind = ind
            
    print ("finished Gridsearch")
    #print (gs[ind].best_estimator_)
    
    #return gs[ind].best_estimator_
    return gs[i].best_estimator_


clf = grid_search()

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
