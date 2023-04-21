################################################################
###FIFTH SCRIPT - MACHINE LEARNING ON POST-LASSO EXTENDED SET###
################################################################

import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

np.random.seed(1123581321)

'''
COMMENTS

This is the fifth script regarding the Extended Set 
producing the results in  "Machine Learning in the 
Prediction of Human Wellbeing", joint first authorship by Oparina, E., Kaiser, C., 
and Gentile, N., and et al.

Aim of this script is to fit and predict with Machine
Learning algorithms on the Post-LASSO Extended set.

Moreover, we also do the analysis for Table 1, 
comparing OLS' performance including or not health variables.
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

read_path = 'C:\\Users\\niccolo.gentile\\Desktop\\Joined_paper\\New_cleared_codes_to_share_230323\\New_KS_codes\\New_scripts_and_results\\New_dsets_17042023\\'

X_train_plks = pd.read_csv(read_path + 'X_train_plks_stand.csv') 

X_test_plks = pd.read_csv(read_path + 'X_test_plks_stand.csv')

#Differently from when importing the whole Extended set,
#in this case a "Unnamed: 0" column with the row indexes is 
#created, hence we drop it.

X_train_plks.drop(["Unnamed: 0"], axis = 1, inplace = True)

X_test_plks.drop(["Unnamed: 0"], axis = 1, inplace = True)

#We also need to import the (Full) Extended train and 
#test sets as in those there is also life satisfaction.

train_ks = pd.read_csv(read_path + 'train_ks_stand.csv') 

test_ks = pd.read_csv(read_path + 'test_ks_stand.csv')

y_train_plks = train_ks['lsat']

y_test_plks = test_ks['lsat']

del train_ks, test_ks

#X_train_plks.shape

#23454 x 224

#X_test_plks.shape

#5864 x 224

###########################
###FITTING THE ALGORTHMS###
###########################

#############################
###MULTICOLLINEARITY CHECK###
#############################

#X_train_plks_const = sm.add_constant(X_train_plks, has_constant = 'add')

#model = sm.OLS(y_train_plks, X_train_plks_const)

#results = model.fit()

#print(results.summary())

#In this case, no multicollinearity nor numerical issues: since the datset
#itself derives from a LASSO regression, the Norm 1 
#penalization already took care of automatically deleting
#correlated features.

##############################################
##LINEAR REGRESSION WITH NO HEALTH VARIABLES##
##############################################

#health_to_drop = ['ple0053','m11102','m11105','m11108','m11109','m11110','m11116',
#                  'doctorvisits','ple0030','ple0031','ple0035','ple0009_1.0',
#                  'ple0009_2.0','ple0009_nan','plb0024_h_1.0',
#                  'ple0005_2.0','ple0005_nan','ple0004_1.0','ple0004_nan']

#X_train_plks_no_health = X_train_plks.drop(health_to_drop, axis = 1)

#X_test_plks_no_health = X_test_plks.drop(health_to_drop, axis = 1)

#linreg_nohealth_plks = linreg_train_test(X_train = X_train_plks_no_health, 
#                                         y_train = y_train_plks, 
#                                         X_test = X_test_plks_no_health, 
#                                         y_test = y_test_plks)

#linreg_nohealth_plks[0] 

#Test MSE = 2.36

#linreg_nohealth_plks[1] 

#Train MSE = 2.24

#linreg_nohealth_plks[2] 

#Test R2 = 0.24

#linreg_nohealth_plks[3]

#Train R2 = 0.27

#######################
###LINEAR REGRESSION###
#######################

linreg_plks = linreg_train_test(X_train = X_train_plks, 
                                y_train = y_train_plks, 
                                X_test = X_test_plks, 
                                y_test = y_test_plks)



linreg_plks[0] 

#Test MSE = 2.23

linreg_plks[1] 

#Train MSE = 2.12

linreg_plks[2] 

#Test R2 = 0.28

linreg_plks[3]

#Train R2 = 0.31

#################
##RANDOM FOREST##
#################

start_time = time.time()

RF_plks = RandomForest(X_train = X_train_plks, 
                       y_train = y_train_plks, 
                       if_bootstrap = True,
                       optim = True, 
                       n_trees = [1000], 
                       n_max_feats = [65], 
                       n_max_depth = [70], 
                       n_min_sample_leaf = [1], 
                       n_cv = 4, 
                       X_test = X_test_plks,
                       y_test = y_test_plks)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_plks[1]

RF_plks[3]

#Test MSE = 2.25

RF_plks[4]

#Train MSE = 0.30

RF_plks[5]

#Test R2 = 0.28

RF_plks[6]

#Train R2 = 0.90

#####################
##GRADIENT BOOSTING##
#####################

GB_plks = GradBoostReg(X_train = X_train_plks, 
                       y_train = y_train_plks, 
                       lr = [0.01],
                       n_iters = [2000],
                       max_depth = [8], 
                       subsample_frac = [0.75],
                       max_feats = [30], 
                       n_cv = 4, 
                       X_test = X_test_plks,
                       y_test = y_test_plks)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

GB_plks[3]

#Test MSE = 2.15

GB_plks[4]

#Train MSE = 0.71

GB_plks[5]

#Test R2 = 0.31

GB_plks[6]

#Train R2 = 0.77


#############################
###PERMUTATION IMPORTANCES###
#############################

#############
##PI R2 OLS##
#############

X_test_plks_const = sm.add_constant(X_test_plks, has_constant = 'add')

start_time = time.time()

PI_plks_r2_ols = permutation_importance(estimator = linreg_plks[-1], 
                                        X = X_test_plks,
                                        y = y_test_plks,
                                        n_jobs = 1,
                                        n_repeats = 10,
                                        scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_plks_r2_ols.importances_mean.argsort()[::-1]

PI_plks_r2_ols_list = []

for i in perm_sorted_idx_r2:
    
    PI_plks_r2_ols_list.append([list(X_test_plks_const)[i], PI_plks_r2_ols.importances_mean[i], PI_plks_r2_ols.importances_std[i]])
    
PI_plks_r2_ols_df = pd.DataFrame(PI_plks_r2_ols_list, columns = ['Variable', 'Average PI as of r2 in 10 reps', 'SD PI as of r2 in 10 reps'])
    
PI_plks_r2_ols_df.to_csv('C:\\Some\\Local\\Path\\PI_plks_r2_ols.csv')

############
##PI R2 RF##
############

start_time = time.time()

PI_plks_r2_rf = permutation_importance(estimator = RF_plks[1], 
                                       X = X_test_plks,
                                       y = y_test_plks,
                                       n_jobs = 1,
                                       n_repeats = 10,
                                       scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_plks_r2_rf.importances_mean.argsort()[::-1]

PI_plks_r2_rf_list = []

for i in perm_sorted_idx_r2:
    
    PI_plks_r2_rf_list.append([list(X_test_plks)[i], PI_plks_r2_rf.importances_mean[i], PI_plks_r2_rf.importances_std[i]])
    
PI_plks_r2_rf_df = pd.DataFrame(PI_plks_r2_rf_list, columns = ['Variable', 'Average PI as of r2 in 10 reps', 'SD PI as of r2 in 10 reps'])
    
PI_plks_r2_rf_df.to_csv('C:\\Some\\Local\\Path\\PI_plks_r2_rf.csv')

############
##PI R2 GB##
############

start_time = time.time()

PI_plks_r2 = permutation_importance(estimator = GB_plks[1], 
                                   X = X_test_plks,
                                   y = y_test_plks,
                                   n_jobs = 1,
                                   n_repeats = 10,
                                   scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_plks_r2.importances_mean.argsort()[::-1]

PI_plks_r2_list = []

for i in perm_sorted_idx_r2:
    
    PI_plks_r2_list.append([list(X_test_plks)[i], PI_plks_r2.importances_mean[i], PI_plks_r2.importances_std[i]])
    
PI_plks_r2_df = pd.DataFrame(PI_plks_r2_list, columns = ['Variable', 'Average PI as of r2 in 10 reps', 'SD PI as of r2 in 10 reps'])
    
PI_plks_r2_df.to_csv('C:\\Some\\Local\\Path\\PI_rks_r2_gb_df.csv')



