###################################################################
###FOURTH SCRIPT - RANDOM FORESTS ON RESTRICTED SET OF VARIABLES###
###################################################################

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.inspection import permutation_importance

np.random.seed(1123581321)

'''
COMMENTS

This is the fourth script of the Restricted Set producing the results in 
"Machine Learning in the Prediction of Human
Wellbeing", joint first authorship by Oparina, E., Kaiser, C., 
and Gentile, N., et al.    

Aim of this script is to fit and predict via Random Forests
on the restricted set of variables, cross-sectionally on the 
different years.

All the intermediate results along the hyperparameter optimization
path are not reported for the sake of readabilty. Please refer
to the paper for the specification of the final values of the
hyperparameters.

Also in this case, as for the Linear Regressions, the Permutation
Importances are computed considering only 2013, and performing it
separatedly for Age and Age**2 (since permuted jointly).
'''

scaler = StandardScaler()

def RandomForest(X_train, y_train, if_bootstrap,
                 optim, n_trees, n_max_feats, 
                 n_max_depth, n_min_sample_leaf, 
                 n_cv, X_test = None,
                 y_test = None):
        
    if optim == True:
        
        rf = RandomForestRegressor(bootstrap = if_bootstrap)

        pruning_dict = {'n_estimators':n_trees,
                'max_features': n_max_feats,
                'max_depth':n_max_depth,
                'min_samples_leaf':n_min_sample_leaf
                }
        
        rf_regr_optim = GridSearchCV(rf, 
                        pruning_dict, 
                        cv = n_cv, 
                        n_jobs = -1,
                        scoring = 'neg_mean_squared_error')
        
    else:
        
        rf_regr_optim = RandomForestRegressor(n_estimators = n_trees[0],
                                              max_features = n_max_feats[0],
                                              max_depth = n_max_depth[0])
        
    rf_regr_fitted = rf_regr_optim.fit(X_train, y_train)
        
    best_rf = rf_regr_fitted.best_estimator_
    
    yhat_train = best_rf.predict(X_train)

    Train_MSE = ((yhat_train - y_train)**2).mean() 
    
    results_from_cv = rf_regr_fitted.cv_results_
    
    if X_test is None and y_test is None:
        
        print('No out of sample accuracy was computed')
    
    else:
        
        yhat_test = best_rf.predict(X_test)

        Test_MSE = ((yhat_test - y_test)**2).mean() 
        
        Train_R2 = r2_score(y_train, yhat_train)
        
        Test_R2 = r2_score(y_test, yhat_test)
        
    list_of_results = [rf_regr_fitted, best_rf, results_from_cv, Test_MSE, Train_MSE, Test_R2, Train_R2]
    
    return list_of_results

years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]

path = 'C:\\Some\\Local\\Path\\'

yearly_dsets_train = []

for i in years:
    
    j = str(i)
    
    import_path = path + 'train_ohed_nomostpop' + j + '.csv'

    yearly_dsets_train.append(pd.read_csv(import_path))

yearly_dsets_test = []        

for i in years:
    
    j = str(i)
    
    import_path = path + 'test_ohed_nomostpop' + j + '.csv'

    yearly_dsets_test.append(pd.read_csv(import_path))
        
start_time = time.time()
    
######################
##RANDOM FOREST 2010##
######################

X_train_10 = yearly_dsets_train[0].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                         'State of Residence', 'hid','Religion','Month of Interview',
                                         'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
X_test_10 = yearly_dsets_test[0].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                       'State of Residence', 'hid','Religion','Month of Interview',
                                       'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
y_train_10 = yearly_dsets_train[0]['Life Satisfaction']
    
y_test_10 = yearly_dsets_test[0]['Life Satisfaction']

X_train_10_stand = pd.DataFrame(scaler.fit_transform(X_train_10), index = y_train_10.index)

X_test_10_stand = pd.DataFrame(scaler.transform(X_test_10), index = y_test_10.index)

X_train_10_stand.columns = list(X_train_10)

X_test_10_stand.columns = list(X_test_10)

RF_2010 = RandomForest(X_train = X_train_10_stand, 
                       y_train = y_train_10, 
                       if_bootstrap = True,
                       optim = True, 
                       n_trees = [1000], 
                       n_max_feats = [9], 
                       n_max_depth = [19], 
                       n_min_sample_leaf = [1], 
                       n_cv = 4, 
                       X_test = X_test_10_stand,
                       y_test = y_test_10)

best_rf_2010 = RF_2010[1]

test_mse_2010 = RF_2010[3]

#2.51

train_mse_2010 = RF_2010[4]

#0.93

test_r2_2010 = RF_2010[5]

#0.16

train_r2_2010 = RF_2010[6]

#0.70

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

######################
##RANDOM FOREST 2011##
######################

X_train_11 = yearly_dsets_train[1].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                         'State of Residence', 'hid','Religion','Month of Interview',
                                         'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
X_test_11 = yearly_dsets_test[1].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                       'State of Residence', 'hid','Religion','Month of Interview',
                                       'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
y_train_11 = yearly_dsets_train[1]['Life Satisfaction']
    
y_test_11 = yearly_dsets_test[1]['Life Satisfaction']

X_train_11_stand = pd.DataFrame(scaler.fit_transform(X_train_11), index = y_train_11.index)

X_test_11_stand = pd.DataFrame(scaler.transform(X_test_11), index = y_test_11.index)

X_train_11_stand.columns = list(X_train_11)

X_test_11_stand.columns = list(X_test_11)

RF_2011 = RandomForest(X_train = X_train_11_stand, 
                       y_train = y_train_11, 
                       if_bootstrap = True,
                       optim = True, 
                       n_trees = [1000], 
                       n_max_feats = [7], 
                       n_max_depth = [20], 
                       n_min_sample_leaf = [1], 
                       n_cv = 4, 
                       X_test = X_test_11_stand,
                       y_test = y_test_11)

best_rf_2011 = RF_2011[1]

test_mse_2011 = RF_2011[3]

#2.47

train_mse_2011 = RF_2011[4]

#0.89

test_r2_2011 = RF_2011[5]

#0.16

train_r2_2011 = RF_2011[6]

#0.71

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

######################
##RANDOM FOREST 2012##
######################

X_train_12 = yearly_dsets_train[2].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                         'State of Residence', 'hid','Religion','Month of Interview',
                                         'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
X_test_12 = yearly_dsets_test[2].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                       'State of Residence', 'hid','Religion','Month of Interview',
                                       'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
y_train_12 = yearly_dsets_train[2]['Life Satisfaction']
    
y_test_12 = yearly_dsets_test[2]['Life Satisfaction']

X_train_12_stand = pd.DataFrame(scaler.fit_transform(X_train_12), index = y_train_12.index)

X_test_12_stand = pd.DataFrame(scaler.transform(X_test_12), index = y_test_12.index)

X_train_12_stand.columns = list(X_train_12)

X_test_12_stand.columns = list(X_test_12)

RF_2012 = RandomForest(X_train = X_train_12_stand, 
                       y_train = y_train_12, 
                       if_bootstrap = True,
                       optim = True, 
                       n_trees = [1000], 
                       n_max_feats = [9], 
                       n_max_depth = [19], 
                       n_min_sample_leaf = [1], 
                       n_cv = 4, 
                       X_test = X_test_12_stand,
                       y_test = y_test_12)

best_rf_2012 = RF_2012[1]

test_mse_2012 = RF_2012[3]

#2.52

train_mse_2012 = RF_2012[4]

#0.91

test_r2_2012 = RF_2012[5]

#0.16

train_r2_2012 = RF_2012[6]

#0.70

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')


######################
##RANDOM FOREST 2013##
######################

X_train_13 = yearly_dsets_train[3].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                         'State of Residence', 'hid','Religion','Month of Interview',
                                         'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
X_test_13 = yearly_dsets_test[3].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                       'State of Residence', 'hid','Religion','Month of Interview',
                                       'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
y_train_13 = yearly_dsets_train[3]['Life Satisfaction']
    
y_test_13 = yearly_dsets_test[3]['Life Satisfaction']

#09/12/2021. 13:15. Adding standardization. 

X_train_13_stand = pd.DataFrame(scaler.fit_transform(X_train_13), index = y_train_13.index)

X_test_13_stand = pd.DataFrame(scaler.transform(X_test_13), index = y_test_13.index)

X_train_13_stand.columns = list(X_train_13)

X_test_13_stand.columns = list(X_test_13)

RF_2013 = RandomForest(X_train = X_train_13_stand, 
                       y_train = y_train_13, 
                       if_bootstrap = True,
                       optim = True, 
                       n_trees = [1000], 
                       n_max_feats = [8], 
                       n_max_depth = [19], 
                       n_min_sample_leaf = [1], 
                       n_cv = 4, 
                       X_test = X_test_13_stand,
                       y_test = y_test_13)

best_rf_2013 = RF_2013[1]

test_mse_2013 = RF_2013[3]

#2.71

train_mse_2013 = RF_2013[4]

#1.11

test_r2_2013 = RF_2013[5]

#0.12

train_r2_2013 = RF_2013[6]

#0.64

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

######################
##RANDOM FOREST 2014##
######################

X_train_14 = yearly_dsets_train[4].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                         'State of Residence', 'hid','Religion','Month of Interview',
                                         'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
X_test_14 = yearly_dsets_test[4].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                       'State of Residence', 'hid','Religion','Month of Interview',
                                       'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
y_train_14 = yearly_dsets_train[4]['Life Satisfaction']
    
y_test_14 = yearly_dsets_test[4]['Life Satisfaction']

X_train_14_stand = pd.DataFrame(scaler.fit_transform(X_train_14), index = y_train_14.index)

X_test_14_stand = pd.DataFrame(scaler.transform(X_test_14), index = y_test_14.index)

X_train_14_stand.columns = list(X_train_14)

X_test_14_stand.columns = list(X_test_14)

RF_2014 = RandomForest(X_train = X_train_14, 
                       y_train = y_train_14, 
                       if_bootstrap = True,
                       optim = True, 
                       n_trees = [1000], 
                       n_max_feats = [8], 
                       n_max_depth = [20], 
                       n_min_sample_leaf = [1], 
                       n_cv = 4, 
                       X_test = X_test_14,
                       y_test = y_test_14)

best_rf_2014 = RF_2014[1]

test_mse_2014 = RF_2014[3]

#2.59

train_mse_2014 = RF_2014[4]

#0.93

test_r2_2014 = RF_2014[5]

#0.13

train_r2_2014 = RF_2014[6]

#0.69

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

######################
##RANDOM FOREST 2015##
######################

X_train_15 = yearly_dsets_train[5].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                         'State of Residence', 'hid','Religion','Month of Interview',
                                         'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
X_test_15 = yearly_dsets_test[5].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                       'State of Residence', 'hid','Religion','Month of Interview',
                                       'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
y_train_15 = yearly_dsets_train[5]['Life Satisfaction']
    
y_test_15 = yearly_dsets_test[5]['Life Satisfaction']

X_train_15_stand = pd.DataFrame(scaler.fit_transform(X_train_15), index = y_train_15.index)

X_test_15_stand = pd.DataFrame(scaler.transform(X_test_15), index = y_test_15.index)

X_train_15_stand.columns = list(X_train_15)

X_test_15_stand.columns = list(X_test_15)

RF_2015 = RandomForest(X_train = X_train_15_stand, 
                       y_train = y_train_15, 
                       if_bootstrap = True,
                       optim = True, 
                       n_trees = [1000], 
                       n_max_feats = [7], 
                       n_max_depth = [20],  
                       n_min_sample_leaf = [1], 
                       n_cv = 4, 
                       X_test = X_test_15_stand,
                       y_test = y_test_15)

best_rf_2015 = RF_2015[1]

test_mse_2015 = RF_2015[3]

#2.62

train_mse_2015 = RF_2015[4]

#0.96

test_r2_2015 = RF_2015[5]

#0.12

train_r2_2015 = RF_2015[6]

#0.68

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')
 
######################
##RANDOM FOREST 2016##
######################

X_train_16 = yearly_dsets_train[6].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                         'State of Residence', 'hid','Religion','Month of Interview',
                                         'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
X_test_16 = yearly_dsets_test[6].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                       'State of Residence', 'hid','Religion','Month of Interview',
                                       'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
y_train_16 = yearly_dsets_train[6]['Life Satisfaction']
    
y_test_16 = yearly_dsets_test[6]['Life Satisfaction']

X_train_16_stand = pd.DataFrame(scaler.fit_transform(X_train_16), index = y_train_16.index)

X_test_16_stand = pd.DataFrame(scaler.transform(X_test_16), index = y_test_16.index)

X_train_16_stand.columns = list(X_train_16)

X_test_16_stand.columns = list(X_test_16)

RF_2016 = RandomForest(X_train = X_train_16_stand, 
                       y_train = y_train_16, 
                       if_bootstrap = True,
                       optim = True, 
                       n_trees = [1000], 
                       n_max_feats = [5], 
                       n_max_depth = [20], 
                       n_min_sample_leaf = [1], 
                       n_cv = 4, 
                       X_test = X_test_16_stand,
                       y_test = y_test_16)

best_rf_2016 = RF_2016[1]

test_mse_2016 = RF_2016[3]

#2.78

train_mse_2016 = RF_2016[4]

#1.31

test_r2_2016 = RF_2016[5]

#0.11

train_r2_2016 = RF_2016[6]

#0.60

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

######################
##RANDOM FOREST 2017##
######################

X_train_17 = yearly_dsets_train[7].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                         'State of Residence', 'hid','Religion','Month of Interview',
                                         'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
X_test_17 = yearly_dsets_test[7].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                       'State of Residence', 'hid','Religion','Month of Interview',
                                       'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
y_train_17 = yearly_dsets_train[7]['Life Satisfaction']
    
y_test_17 = yearly_dsets_test[7]['Life Satisfaction']

#09/12/2021. 13:20. Adding standardization. 

X_train_17_stand = pd.DataFrame(scaler.fit_transform(X_train_17), index = y_train_17.index)

X_test_17_stand = pd.DataFrame(scaler.transform(X_test_17), index = y_test_17.index)

X_train_17_stand.columns = list(X_train_17)

X_test_17_stand.columns = list(X_test_17)

RF_2017 = RandomForest(X_train = X_train_17_stand, 
                       y_train = y_train_17, 
                       if_bootstrap = True,
                       optim = True, 
                       n_trees = [1000], 
                       n_max_feats = [7], 
                       n_max_depth = [20],  
                       n_min_sample_leaf = [1], 
                       n_cv = 4, 
                       X_test = X_test_17_stand,
                       y_test = y_test_17)

best_rf_2017 = RF_2017[1]

test_mse_2017 = RF_2017[3]

#2.75

train_mse_2017 = RF_2017[4]

#1.21

test_r2_2017 = RF_2017[5]

#0.10

train_r2_2017 = RF_2017[6]

#0.62

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

######################
##RANDOM FOREST 2018##
######################

X_train_18 = yearly_dsets_train[8].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                         'State of Residence', 'hid','Religion','Month of Interview',
                                         'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
X_test_18 = yearly_dsets_test[8].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                       'State of Residence', 'hid','Religion','Month of Interview',
                                       'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
y_train_18 = yearly_dsets_train[8]['Life Satisfaction']
    
y_test_18 = yearly_dsets_test[8]['Life Satisfaction']

X_train_18_stand = pd.DataFrame(scaler.fit_transform(X_train_18), index = y_train_18.index)

X_test_18_stand = pd.DataFrame(scaler.transform(X_test_18), index = y_test_18.index)

X_train_18_stand.columns = list(X_train_18)

X_test_18_stand.columns = list(X_test_18)

RF_2018 = RandomForest(X_train = X_train_18_stand, 
                       y_train = y_train_18, 
                       if_bootstrap = True,
                       optim = True, 
                       n_trees = [1000], 
                       n_max_feats = [5], 
                       n_max_depth = [22], 
                       n_min_sample_leaf = [1], 
                       n_cv = 4, 
                       X_test = X_test_18_stand,
                       y_test = y_test_18)

best_rf_2018 = RF_2018[1]

test_mse_2018 = RF_2018[3]

#2.72

train_mse_2018 = RF_2018[4]

#1.06

test_r2_2018 = RF_2018[5]

#0.11

train_r2_2018 = RF_2018[6]

#0.66

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

####################
##PI MSE RF - 2013##
####################

start_time = time.time()

PI_13_mse = permutation_importance(estimator = RF_2013[1], 
                                   X = X_test_13_stand,
                                   y = y_test_13,
                                   n_jobs = 1,
                                   n_repeats = 10,
                                   scoring = 'neg_mean_squared_error')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_mse = PI_13_mse.importances_mean.argsort()[::-1]

PI_13_mse_list = []

for i in perm_sorted_idx_mse:
    
    PI_13_mse_list.append([list(X_test_13_stand)[i], PI_13_mse.importances_mean[i], PI_13_mse.importances_std[i]])
    
PI_13_mse_df = pd.DataFrame(PI_13_mse_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
PI_13_mse_df.to_csv('C:\\Some\\Local\\Path\\PI_13_mse_rf_with_joined_ages.csv')

##Permutation importance for age and age square jointly

age_pis = np.empty(10)

for i in range(10):
    
    age_vars_permuted = X_test_13_stand[['Age','Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_13_stand.copy()
    
    X_test_stand_permuted['Age'] = age_vars_permuted['Age']
    
    X_test_stand_permuted['Age^2'] = age_vars_permuted['Age^2']
    
    MSE_permuted = ((RF_2013[1].predict(X_test_stand_permuted) - y_test_13)**2).mean()  
    
    age_pis[i] = MSE_permuted - RF_2013[3]
    
print(age_pis.mean())

print(age_pis.std())

print(age_pis)
