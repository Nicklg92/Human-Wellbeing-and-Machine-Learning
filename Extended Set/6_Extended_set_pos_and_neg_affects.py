##########################################################
###SIXTH SCRIPT - POSITIVE AND NEGATIVE AFFECT ANALYSIS###
##########################################################

import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyreadstat
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

np.random.seed(1123581321)

'''
COMMENTS:
    
This is the sixth script regarding the Extended Set 
producing the results in  "Machine Learning in the 
Prediction of Human Wellbeing", joint first authorship by Oparina, E., Kaiser, C., 
and Gentile, N., and et al.

Aim of this script is to use Affect Variables as target instead
of life satisfaction on the Post-LASSO Extended dataset.

Affect variables can be both positive and negative. What are they?

1) Positive Affect Target Variable

Using bdp0203 = Frequency of being happy in the last 4 weeks

2) Negative Affect Target Variable

A new variable being the average of bdp0201, bdp0202 and bdp0204 for
each individual. If any of the three is missing, the value of the new
dependent is missing. In particular:

bdp0201 = Frequency of Being Angry in the Last 4 Weeks
bdp0202 = Frequency of Being Worried in the Last 4 Weeks
bdp0204 = Frequency of Being Sad in the Last 4 Weeks

On top of Machine Learning estimations, we also 
compute the Permutation Importances.
'''

def linreg_train_test(X_train, y_train, X_test, y_test):
    
    lineareg = LinearRegression()
    
    X_const_train = sm.add_constant(X_train, has_constant = 'add')
    
    X_const_test = sm.add_constant(X_test, has_constant = 'add')
    
    lineareg_fitted = lineareg.fit(X_const_train, y_train)
    
    lineareg_yhat_test = lineareg_fitted.predict(X_const_test)

    Mse_lineareg_test = ((lineareg_yhat_test - y_test)**2).mean()
    
    lineareg_yhat_train = lineareg_fitted.predict(X_const_train)

    Mse_lineareg_train = ((lineareg_yhat_train - y_train)**2).mean()  
    
    lineareg_yhat_train_round = np.round(lineareg_yhat_train)
        
    Test_R2 = r2_score(y_test, lineareg_yhat_test)
    
    Train_R2 = r2_score(y_train, lineareg_yhat_train)

    list_of_results = [Mse_lineareg_test, Mse_lineareg_train, Test_R2, Train_R2, lineareg_fitted]
    
    return list_of_results

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

def GradBoostReg(X_train, y_train, 
                 lr,
                 n_iters,  
                 max_depth,  
                 subsample_frac,
                 max_feats,
                 n_cv, 
                 X_test = None,
                 y_test = None):
        
    gb = GradientBoostingRegressor(verbose = 1)

    optim_dict = {'n_estimators': n_iters,
                  'learning_rate': lr,
                  'max_depth': max_depth,
                  'subsample': subsample_frac,
                  'max_features': max_feats}
        
    gb_regr_optim = GridSearchCV(gb, 
                                 optim_dict, 
                                 cv = n_cv, 
                                 n_jobs = -1,  
                                 scoring = 'neg_mean_squared_error')

    gb_regr_fitted = gb_regr_optim.fit(X_train, y_train)
        
    best_gb = gb_regr_fitted.best_estimator_
    
    yhat_train = best_gb.predict(X_train)

    Train_MSE = ((yhat_train - y_train)**2).mean() 
    
    results_from_cv = gb_regr_fitted.cv_results_
    
    if X_test is None and y_test is None:
        
        print('No out of sample accuracy was computed')
    
    else:
        
        yhat_test = best_gb.predict(X_test)

        Test_MSE = ((yhat_test - y_test)**2).mean() 
        
        Train_R2 = r2_score(y_train, yhat_train)
        
        Test_R2 = r2_score(y_test, yhat_test)

    list_of_results = [gb_regr_fitted, best_gb, results_from_cv, Test_MSE, Train_MSE, Test_R2, Train_R2]
    
    return list_of_results

read_path = 'C:\\Some\\Local\\Path\\'

X_train_plks = pd.read_csv(read_path + 'X_train_plks_stand.csv') 

X_test_plks = pd.read_csv(read_path + 'X_test_plks_stand.csv')

#Differently from when importing the whole Extended set,
#in this case a "Unnamed: 0" column with the row indexes is 
#created, hence we drop it.

X_train_plks.drop(["Unnamed: 0"], axis = 1, inplace = True)

X_test_plks.drop(["Unnamed: 0"], axis = 1, inplace = True)

#We also need to import the (Full) Extended train and 
#test sets as in those there is also life satisfaction and the
#pids, which will be useful.

train_ks = pd.read_csv(read_path + 'train_ks_stand.csv') 

test_ks = pd.read_csv(read_path + 'test_ks_stand.csv')

pids_train_plks = train_ks['pid']

pids_test_plks = test_ks['pid']

del train_ks, test_ks

X_train_plks["pid"] = pids_train_plks

X_test_plks["pid"] = pids_test_plks

del pids_train_plks, pids_test_plks

################################
###IMPORTING AFFECT VARIABLES###
################################

bdp_path = 'C:\\Some\\Local\\Path\\'

vars_bdp = ['syear', 'pid', 'bdp0201', 'bdp0202', 'bdp0203', 'bdp0204']

bdp_obj = pyreadstat.read_dta(bdp_path + 'bdp.dta', usecols = vars_bdp)

bdp_data = bdp_obj[0]

bdp_data.shape

#Notice that:

#(30956, 6)

#bdp_data_13 = bdp_data[bdp_data['syear'] == 2013]

#bdp_data_13.shape

#(30956, 6)

#(bdp_data_13 != bdp_data).sum().sum()

#0

#Since bdp_data['syear'].unique()

#array([2013], dtype=int64)

#Missing and negative values in bdp?

for i in list(bdp_data):
    
    print([i, "Negatives in " + i + ": " + str(np.sum(bdp_data[i] < 0)), "NaNs in " + i + ": " + str(bdp_data[i].isna().sum())])
    

#['pid', 'Negatives in pid: 0', 'NaNs in pid: 0']
#['syear', 'Negatives in syear: 0', 'NaNs in syear: 0']
#['bdp0201', 'Negatives in bdp0201: 5037', 'NaNs in bdp0201: 0']
#['bdp0202', 'Negatives in bdp0202: 5061', 'NaNs in bdp0202: 0']
#['bdp0203', 'Negatives in bdp0203: 5047', 'NaNs in bdp0203: 0']
#['bdp0204', 'Negatives in bdp0204: 5035', 'NaNs in bdp0204: 0']

#and since we know that negative values are another way
#to label missingness

bdp_data_nomiss = bdp_data[(bdp_data >= 0).all(1)]

#bdp_data_nomiss.shape

#(25861, 6)

X_train_plks_bdp = pd.merge(X_train_plks, bdp_data_nomiss, on = 'pid', how = 'inner')

X_test_plks_bdp = pd.merge(X_test_plks, bdp_data_nomiss, on = 'pid', how = 'inner')

const_in_test = []

for i in list(X_test_plks_bdp):
        
    if X_test_plks_bdp[i].nunique() == 1:
        
        const_in_test.append(i)
            
        X_train_plks_bdp.drop(i, axis = 1, inplace = True)
            
        X_test_plks_bdp.drop(i, axis = 1, inplace = True)
        
len(const_in_test)

const_in_train = []

for i in list(X_train_plks_bdp):
        
    if X_train_plks_bdp[i].nunique() == 1:
        
        const_in_train.append(i)
            
        X_train_plks_bdp.drop(i, axis = 1, inplace = True)
            
        X_test_plks_bdp.drop(i, axis = 1, inplace = True)
        
len(const_in_train)

###################
##NEGATIVE AFFECT##
###################

X_train_plks_bdp['neg_affect'] = 1/3 * X_train_plks_bdp['bdp0201'] + 1/3 * X_train_plks_bdp['bdp0202'] +  1/3 * X_train_plks_bdp['bdp0204']

X_test_plks_bdp['neg_affect'] = 1/3 * X_test_plks_bdp['bdp0201'] + 1/3 * X_test_plks_bdp['bdp0202'] +  1/3 * X_test_plks_bdp['bdp0204']

#####################
##LINEAR REGRESSION##
#####################

X_train_plks_bdp_neg = X_train_plks_bdp.drop(['neg_affect', 'bdp0201', 'bdp0202', 'bdp0204', 'bdp0203', 'pid'], axis = 1)

y_train_plks_bdp_neg = X_train_plks_bdp['neg_affect']

X_test_plks_bdp_neg = X_test_plks_bdp.drop(['neg_affect', 'bdp0201', 'bdp0202', 'bdp0204', 'bdp0203', 'pid'], axis = 1)

y_test_plks_bdp_neg = X_test_plks_bdp['neg_affect']

linreg_plks_bdp_neg = linreg_train_test(X_train = X_train_plks_bdp_neg, 
                                        y_train = y_train_plks_bdp_neg, 
                                        X_test = X_test_plks_bdp_neg, 
                                        y_test = y_test_plks_bdp_neg)

linreg_plks_bdp_neg[0] 

#Test MSE = 0.42

linreg_plks_bdp_neg[1] 

#Train MSE =  0.41

linreg_plks_bdp_neg[2] 

#Test R2 = 0.29

linreg_plks_bdp_neg[3] 

#Train R2 = 0.32

##################
###RANDOM FOREST##
##################

start_time = time.time()

RF_plksbdp_neg = RandomForest(X_train = X_train_plks_bdp_neg, 
                              y_train = y_train_plks_bdp_neg, 
                              if_bootstrap = True,
                              optim = True, 
                              n_trees = [1000], 
                              n_max_feats = [60], 
                              n_max_depth = [60], 
                              n_min_sample_leaf = [1], 
                              n_cv = 4, 
                              X_test = X_test_plks_bdp_neg,
                              y_test = y_test_plks_bdp_neg)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

#Attempts

RF_plksbdp_neg[1]


RF_plksbdp_neg[3]

#Test MSE = 0.43

RF_plksbdp_neg[4]

#Train MSE = 0.06

RF_plksbdp_neg[5]

#Test R2 = 0.29

RF_plksbdp_neg[6]

#Train R2 = 0.91

######################
###GRADIENT BOOSTING##
######################

start_time = time.time()

GB_plksbdp_neg = GradBoostReg(X_train = X_train_plks_bdp_neg, 
                              y_train = y_train_plks_bdp_neg, 
                              lr = [0.01],
                              n_iters = [2000],
                              max_depth = [6], 
                              subsample_frac = [0.75],
                              max_feats = [25], 
                              n_cv = 4, 
                              X_test = X_test_plks_bdp_neg,
                              y_test = y_test_plks_bdp_neg)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

#Attempts

#Runtime was 1762.0474042892456 seconds

GB_plksbdp_neg[1]


GB_plksbdp_neg[3]

#Test MSE = 0.41

GB_plksbdp_neg[4]

#Train MSE = 0.25

GB_plksbdp_neg[5]

#Train R2 = 0.31

GB_plksbdp_neg[6]

#Test R2 = 0.58

###################
##POSITIVE AFFECT##
###################

#Positive affect is simply bdp0203.

#####################
##LINEAR REGRESSION##
#####################

X_train_plks_bdp_pos = X_train_plks_bdp.drop(['bdp0201', 'bdp0202', 'bdp0204', 'bdp0203', 'pid', 'neg_affect'], axis = 1)

y_train_plks_bdp_pos = X_train_plks_bdp['bdp0203']

X_test_plks_bdp_pos = X_test_plks_bdp.drop(['bdp0201', 'bdp0202', 'bdp0204', 'bdp0203', 'pid', 'neg_affect'], axis = 1)

y_test_plks_bdp_pos = X_test_plks_bdp['bdp0203']

linreg_plks_bdp_pos = linreg_train_test(X_train = X_train_plks_bdp_pos, 
                                        y_train = y_train_plks_bdp_pos, 
                                        X_test = X_test_plks_bdp_pos, 
                                        y_test = y_test_plks_bdp_pos)

linreg_plks_bdp_pos[0] 

#Test MSE = 0.57

linreg_plks_bdp_pos[1] 

#Train MSE = 0.55

linreg_plks_bdp_pos[2] 

#Test R2 = 0.21

linreg_plks_bdp_pos[3] 

#Train R2 = 0.22

##################
###RANDOM FOREST##
##################

start_time = time.time()

RF_plksbdp_pos = RandomForest(X_train = X_train_plks_bdp_pos, 
                              y_train = y_train_plks_bdp_pos, 
                              if_bootstrap = True,
                              optim = True, 
                              n_trees = [1000], 
                              n_max_feats = [70], 
                              n_max_depth = [60], 
                              n_min_sample_leaf = [1],
                              n_cv = 4, 
                              X_test = X_test_plks_bdp_pos,
                              y_test = y_test_plks_bdp_pos)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

#Attempts

RF_plksbdp_pos[1]


RF_plksbdp_pos[3]

#Test MSE = 0.57

RF_plksbdp_pos[4]

#Train MSE = 0.08

RF_plksbdp_pos[5]

#Test R2 = 0.21

RF_plksbdp_pos[6]

#Train R2 = 0.89

######################
###GRADIENT BOOSTING##
######################


start_time = time.time()

GB_plksbdp_pos = GradBoostReg(X_train = X_train_plks_bdp_pos, 
                              y_train = y_train_plks_bdp_pos, 
                              lr = [0.005],
                              n_iters = [2500],
                              max_depth = [9], 
                              subsample_frac = [0.75],
                              max_feats = [25], 
                              n_cv = 4, 
                              X_test = X_test_plks_bdp_pos,
                              y_test = y_test_plks_bdp_pos)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

#Attempts

GB_plksbdp_pos[1]

GB_plksbdp_pos[3]

#Test MSE = 0.55

GB_plksbdp_pos[4]

#Train MSE = 0.18

GB_plksbdp_pos[5]

#Train R2 = 0.24

GB_plksbdp_pos[6]

#Test R2 = 0.75


#############################
####PI OLS NEGATIVE AFFECT###
#############################

X_test_plks_bdp_neg_const = sm.add_constant(X_test_plks_bdp_neg, has_constant = 'add')

start_time = time.time()

PI_neg_OLS = permutation_importance(estimator = linreg_plks_bdp_neg[-1], 
                                    X = X_test_plks_bdp_neg_const,
                                    y = y_test_plks_bdp_neg,
                                    n_jobs = 1,
                                    n_repeats = 10,
                                    scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_neg_OLS.importances_mean.argsort()[::-1]

PI_neg_OLS_list = []

for i in perm_sorted_idx_r2:
    
    PI_neg_OLS_list.append([list(X_test_plks_bdp_neg_const)[i], PI_neg_OLS.importances_mean[i], PI_neg_OLS.importances_std[i]])
    
PI_neg_OLS_df = pd.DataFrame(PI_neg_OLS_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
PI_neg_OLS_df.to_csv('C:\\Some\\Local\\Path\\PI_neg_OLS_df.csv')

#############################
####PI OLS POSITIVE AFFECT###
#############################

X_test_plks_bdp_pos_const = sm.add_constant(X_test_plks_bdp_pos, has_constant = 'add')

start_time = time.time()

PI_pos_OLS = permutation_importance(estimator = linreg_plks_bdp_pos[-1], 
                                    X = X_test_plks_bdp_pos_const,
                                    y = y_test_plks_bdp_pos,
                                    n_jobs = 1,
                                    n_repeats = 10,
                                    scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_pos_OLS.importances_mean.argsort()[::-1]

PI_pos_OLS_list = []

for i in perm_sorted_idx_r2:
    
    PI_pos_OLS_list.append([list(X_test_plks_bdp_pos_const)[i], PI_pos_OLS.importances_mean[i], PI_pos_OLS.importances_std[i]])
    
PI_pos_OLS_df = pd.DataFrame(PI_pos_OLS_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
PI_pos_OLS_df.to_csv('C:\\Some\\Local\\Path\\PI_pos_OLS_df.csv')

#############################
####PI RF NEGATIVE AFFECT###
#############################

start_time = time.time()

PI_neg_RF = permutation_importance(estimator = RF_plksbdp_neg[1], 
                                    X = X_test_plks_bdp_neg,
                                    y = y_test_plks_bdp_neg,
                                    n_jobs = 1,
                                    n_repeats = 10,
                                    scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_neg_RF.importances_mean.argsort()[::-1]

PI_neg_RF_list = []

for i in perm_sorted_idx_r2:
    
    PI_neg_RF_list.append([list(X_test_plks_bdp_neg)[i], PI_neg_RF.importances_mean[i], PI_neg_RF.importances_std[i]])
    
PI_neg_RF_df = pd.DataFrame(PI_neg_RF_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
PI_neg_RF_df.to_csv('C:\\Some\\Local\\Path\\PI_neg_RF_df.csv')

#############################
####PI RF POSITIVE AFFECT###
#############################

start_time = time.time()

PI_pos_RF = permutation_importance(estimator = RF_plksbdp_pos[1], 
                                    X = X_test_plks_bdp_pos,
                                    y = y_test_plks_bdp_pos,
                                    n_jobs = 1,
                                    n_repeats = 10,
                                    scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_pos_RF.importances_mean.argsort()[::-1]

PI_pos_RF_list = []

for i in perm_sorted_idx_r2:
    
    PI_pos_RF_list.append([list(X_test_plks_bdp_pos)[i], PI_pos_RF.importances_mean[i], PI_pos_RF.importances_std[i]])
    
PI_pos_RF_df = pd.DataFrame(PI_pos_RF_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
PI_pos_RF_df.to_csv('C:\\Some\\Local\\Path\\PI_pos_RF_df.csv')

#############################
####PI GB NEGATIVE AFFECT###
#############################

start_time = time.time()

PI_neg_GB = permutation_importance(estimator = GB_plksbdp_neg[1], 
                                    X = X_test_plks_bdp_neg,
                                    y = y_test_plks_bdp_neg,
                                    n_jobs = 1,
                                    n_repeats = 10,
                                    scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_neg_GB.importances_mean.argsort()[::-1]

PI_neg_GB_list = []

for i in perm_sorted_idx_r2:
    
    PI_neg_GB_list.append([list(X_test_plks_bdp_neg)[i], PI_neg_GB.importances_mean[i], PI_neg_GB.importances_std[i]])
    
PI_neg_GB_df = pd.DataFrame(PI_neg_GB_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
PI_neg_GB_df.to_csv('C:\\Some\\Local\\Path\\PI_neg_GB_df.csv')

#############################
####PI GB POSITIVE AFFECT###
############################

start_time = time.time()

PI_pos_GB = permutation_importance(estimator = GB_plksbdp_pos[1], 
                                    X = X_test_plks_bdp_pos,
                                    y = y_test_plks_bdp_pos,
                                    n_jobs = 1,
                                    n_repeats = 10,
                                    scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_pos_GB.importances_mean.argsort()[::-1]

PI_pos_GB_list = []

for i in perm_sorted_idx_r2:
    
    PI_pos_GB_list.append([list(X_test_plks_bdp_pos)[i], PI_pos_GB.importances_mean[i], PI_pos_GB.importances_std[i]])
    
PI_pos_GB_df = pd.DataFrame(PI_pos_GB_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
PI_pos_GB_df.to_csv('C:\\Some\\Local\\Path\\PI_pos_GB_df.csv')





