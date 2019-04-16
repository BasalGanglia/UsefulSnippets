# exploratory data analysis with SeaBorn
import seaborn as sns

sns.factorplot('income', 'capital-gain', hue='sex', data=data, kind='bar', col='race', row='relationship')

# Lambda expression for feature engineering
income = income_raw.apply(lambda x: 0 if x == "<=50K" else 1)

# Notes about optimizing hyperparameters:
"""
One option for a smarter implementation of hyperparameter tuning is to combine random search and grid search:

Use random search with a large hyperparameter grid
Use the results of random search to build a focused hyperparameter grid around the best performing hyperparameter values.
Run grid search on the reduced hyperparameter grid.
Repeat grid search on more focused grids until maximum computational/time budget is exceeded.
Or could even look into using hyperopt. Here might be a good example of how to use this bayesian optimization technique in python.
"""

from sklearn.metrics import fbeta_score, accuracy_score


#Implementation - Creating a Training and Predicting Pipeline
#To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section. In the code block below, you will need to implement the following:

#Import fbeta_score and accuracy_score from sklearn.metrics.
#Fit the learner to the sampled training data and record the training time.
#Perform predictions on the test data X_test, and also on the first 300 training points X_train[:300].
#Record the total prediction time.
#Calculate the accuracy score for both the training subset and testing set.
#Calculate the F-score for both the training subset and testing set.
#Make sure that you set the beta parameter!

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    # using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end-start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end-start
            
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
        
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results


# TODO: Import the three supervised learning models from sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
# TODO: Initialize the three models
# As per request by the reviewer, I set the random states (to 42 for obvious reasons)
clf_A = GradientBoostingClassifier(random_state=42)
clf_B = LogisticRegression(random_state=42)
clf_C = GaussianNB()

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = len(y_train)
samples_10 = int(len(y_train)*0.1)
samples_1 = int(len(y_train)*0.01)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
            train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)