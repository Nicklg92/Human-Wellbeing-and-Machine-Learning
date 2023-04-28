####################################################################################
##SEVENTH SCRIPT - VARIABLE DEMEANING AND MACHINE LEARNING ON PANEL RESTRICTED SET##
####################################################################################

import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance


np.random.seed(1123581321)

'''
COMMENT

This is the seventh script producing of the Restricted Set producing the results in 
"Human Wellbeing and Machine Learning" by Ekaterina Oparina* (r) Caspar Kaiser* (r) Niccolò Gentile*; 
Alexandre Tkatchenko, Andrew E. Clark, Jan-Emmanuel De Neve 
and Conchita D'Ambrosio (* equal first authors in random order).  

Defining "means of independent variables" as:
    
average over T periods of the variable j for individual i 
(also called "group-mean" in the Econometric literature)

and "demeaned" as:
    
value of variable j for individual i at time t - average over the T 
periods of variable j for individual i
(also called "group-mean deviations" in the Econometric literature) 

the aim of this script is to fit and predict two specifications:
    
1) life satisfaction = f(demeaned independent variables, means of independent variables)
    
2) demeaned life satisfaction = f(demeaned independent variables, means of independent variables)
     
The M[D], P[D] notation is borrowed from "Econometric Analysis", 2016, Giovanni Bruno.

The notation relates to the mathematical concept of projection
matrices. 

With "x_PD", in this script we refer to the vector including the
"group-mean", for each individual, of the associated variable x.

For instance, in "y_train_PD" there will be the "group-mean"
of life satisfaction for each inidividual in the training set.

In "y_train_MD" there will be the "group-mean deviations" of 
life satisfaction for each inidividual in the training set.

Similarly are defined also for the independent variables. 
'''

import_path = 'C:\\Some\\Loal\\Path\\'

train_panel = pd.read_csv(import_path + 'at_least_two_df_train.csv')

test_panel = pd.read_csv(import_path + 'at_least_two_df_test.csv')

train_panel.drop(['Unnamed: 0'], axis = 1, inplace = True)

test_panel.drop(['Unnamed: 0'], axis = 1, inplace = True)

train_panel.drop(['index_compilers_1'], axis = 1, inplace = True)

test_panel.drop(['index_compilers_1'], axis = 1, inplace = True)

######################################################
###CREATING GROUP-MEAN AND GROUP-DEMEANED VARIABLES###
######################################################

y_train = train_panel[['pid','Life Satisfaction']]

X_train = train_panel.drop(['Life Satisfaction'], axis = 1)

y_test = test_panel[['pid','Life Satisfaction']]

X_test = test_panel.drop(['Life Satisfaction'], axis = 1)

#Computing P[D]y_train and M[D]y_train

y_train_PD = pd.DataFrame(y_train.groupby(['pid']).mean())

y_train_PD['pid'] = y_train_PD.index

y_train_PD.reset_index(inplace = True, drop = True)

y_train_PD_y = y_train.merge(y_train_PD, how = 'inner', on = 'pid')

y_train_PD_y.columns = ['pid', 'Life Satisfaction', 'Group Mean Life Satisfaction']

y_train_MD = y_train_PD_y['Life Satisfaction'] - y_train_PD_y['Group Mean Life Satisfaction']

#Computing P[D]y_test and M[D]y_test

y_test_PD = pd.DataFrame(y_test.groupby(['pid']).mean())

y_test_PD['pid'] = y_test_PD.index

y_test_PD.reset_index(inplace = True, drop = True)

y_test_PD_y = y_test.merge(y_test_PD, how = 'inner', on = 'pid')

y_test_PD_y.columns = ['pid', 'Life Satisfaction', 'Group Mean Life Satisfaction']

y_test_MD = y_test_PD_y['Life Satisfaction'] - y_test_PD_y['Group Mean Life Satisfaction']

#Computing P[D]X_train and M[D]X_train
#In this case, the group-mean deviations' versions of the
#variables are themselves included in "X_train_MD".

X_train_MD = X_train.copy()

for i in list(X_train):
    
    if i in ['pid', 'year', 'hid']:
        
        print('No within transformation for pid, year and hid')
    
    else:
    
        var_1_PD = pd.DataFrame(X_train_MD[['pid', i]].groupby(['pid']).mean())
    
        var_1_PD['pid'] = var_1_PD.index
    
        var_1_PD.reset_index(inplace = True, drop = True)
        
        X_train_MD = X_train_MD.merge(var_1_PD, how = 'inner', on = 'pid')
        
        #At this point, in X_train_MD, we have both the variable 
        #in its original form, as well as in group-means.
        #Python automatically changes the name of the former adding
        #an "_x" to its name, and a "_y" to the latter. 
        #E.g., "Age_x" contains the age for each individual, and
        #"Age_y" the average value of age for each individual over
        #the t periods in which replied. We change the names using
        #the aforementioned PD and MD for greater clarity down the
        #road. Same thing on the test set.

        cols_to_change = [i + '_x', i + '_y']
        
        PD_x = 'P[D]_' + i
        
        X_train_MD.rename(columns = {cols_to_change[0]: i,
                                     cols_to_change[1]: PD_x}, inplace = True)
        
        MD_x = 'M[D]_' + i
    
        X_train_MD[MD_x] =  X_train_MD[i] - X_train_MD[PD_x]
                
#Computing P[D]X_test and M[D]X_test
        
X_test_MD = X_test.copy()

for i in list(X_test):
    
    if i in ['pid', 'year', 'hid']:
        
        print('No within transformation for pid, year and hid')
    
    else:
    
        var_1_PD = pd.DataFrame(X_test_MD[['pid', i]].groupby(['pid']).mean())
    
        var_1_PD['pid'] = var_1_PD.index
    
        var_1_PD.reset_index(inplace = True, drop = True)
        
        X_test_MD = X_test_MD.merge(var_1_PD, how = 'inner', on = 'pid')
                
        cols_to_change = [i + '_x', i + '_y']
        
        PD_x = 'P[D]_' + i
        
        X_test_MD.rename(columns = {cols_to_change[0]: i,
                                    cols_to_change[1]: PD_x}, inplace = True)
        
        MD_x = 'M[D]_' + i
    
        X_test_MD[MD_x] =  X_test_MD[i] - X_test_MD[PD_x]
        
#We now drop variables having 0 M[D], that is variables 
#whose value was constant across the years for everyone.
        
for j in list(X_train_MD):
    
    if np.sum(X_train_MD[j] == 0) == len(X_train_MD):
        
        X_train_MD.drop([j], axis = 1, inplace = True)
        
        X_test_MD.drop([j], axis = 1, inplace = True)

#Quick check on no issues with misalignements in the rows:

#np.sum(y_train_PD_y['pid'] != X_train_MD['pid'])
#0
        
#np.sum(y_test_PD_y['pid'] != X_test_MD['pid'])
#0        

#and exclusion of the variables in their original forms.

vars_no_PD_MD = [x for x in list(X_train_MD) if x[0:4] != 'P[D]' and x[0:4] != 'M[D]']

X_train_MD.drop(vars_no_PD_MD, axis = 1, inplace = True)

X_test_MD.drop(vars_no_PD_MD, axis = 1, inplace = True)

#Finally, we also delete both the group-means and group-means
#deviations of the categorical variables, as we have the 
#both versions for each of the derived dummies. 
    
X_train_MD.drop(['P[D]_Marital Status','M[D]_Marital Status', 
                 'P[D]_State of Residence','M[D]_State of Residence',
                 'P[D]_Religion', 'M[D]_Religion',
                 'P[D]_Month of Interview', 'M[D]_Month of Interview', 
                 'P[D]_Housing ownership status', 'M[D]_Housing ownership status',
                 'P[D]_Household yearly disposable income','M[D]_Household yearly disposable income',
                 'P[D]_Ethnicity'], axis = 1, inplace = True)
    
#M[D]_Ethnicity already deleted in the previous for loop since constant. 
#Same for 'M[D]_Eth_2' and 'M[D]_Eth_3'.
    
X_test_MD.drop(['P[D]_Marital Status','M[D]_Marital Status', 
                 'P[D]_State of Residence','M[D]_State of Residence',
                 'P[D]_Religion', 'M[D]_Religion',
                 'P[D]_Month of Interview', 'M[D]_Month of Interview', 
                 'P[D]_Housing ownership status', 'M[D]_Housing ownership status',
                 'P[D]_Household yearly disposable income','M[D]_Household yearly disposable income',
                 'P[D]_Ethnicity'], axis = 1, inplace = True)

###################################
####MACHINE LEARNING ALGORITHMS####
###################################

scaler = StandardScaler()

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

##################################################
##LINEAR REGRESSIONS: TARGET = LIFE SATISFACTION##
##################################################

X_train_MD_stand = pd.DataFrame(scaler.fit_transform(X_train_MD), index = y_train_PD_y['Life Satisfaction'].index)

X_test_MD_stand = pd.DataFrame(scaler.transform(X_test_MD), index = y_test_PD_y['Life Satisfaction'].index)

X_train_MD_stand.columns = list(X_train_MD)

X_test_MD_stand.columns = list(X_test_MD)

linreg_MD_PD = linreg_train_test(X_train = X_train_MD_stand, 
                                 y_train = y_train_PD_y['Life Satisfaction'], 
                                 X_test = X_test_MD_stand, 
                                 y_test = y_test_PD_y['Life Satisfaction'])


linreg_MD_PD[2] 

#Test R2 = 0.12

linreg_MD_PD[3] 

#Train R2 = 0.11

linreg_MD_PD[0] 

#Test MSE = 2.67 

linreg_MD_PD[1] 

#Train MSE = 2.70

################################################
###RANDOM FORESTS: TARGET = LIFE SATISFACTION###
################################################ 

start_time = time.time()

RF_MD_PD = RandomForest(X_train = X_train_MD_stand, 
                        y_train = y_train_PD_y['Life Satisfaction'], 
                        if_bootstrap = True,
                        optim = True, 
                        n_trees = [1000], 
                        n_max_feats = [18], 
                        n_max_depth = [23], 
                        n_min_sample_leaf = [1], 
                        n_cv = 4, 
                        X_test = X_test_MD_stand,
                        y_test = y_test_PD_y['Life Satisfaction'])

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

RF_MD_PD[1]

RF_MD_PD[5]

#Test R2 = 0.15

RF_MD_PD[6]

#Train R2 = 0.72


RF_MD_PD[3]

#Test MSE = 2.58

RF_MD_PD[4]

#Train MSE = 0.85

###################################################
###GRADIENT BOOSTING: TARGET = LIFE SATISFACTION###
###################################################

start_time = time.time()

GB_MD_PD = GradBoostReg(X_train = X_train_MD_stand, 
                       y_train = y_train_PD_y['Life Satisfaction'], 
                       lr = [0.02],
                       n_iters = [350],
                       max_depth = [12], 
                       subsample_frac = [0.75],
                       max_feats = [8], 
                       n_cv = 4, 
                       X_test = X_test_MD_stand,
                       y_test = y_test_PD_y['Life Satisfaction'])

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

#Attempts

GB_MD_PD[1]

GB_MD_PD[5]

#Test R2 = 0.16

GB_MD_PD[6]

#Train R2 = 0.43

GB_MD_PD[3]

#Test MSE = 2.56

GB_MD_PD[4]

#Train MSE = 1.73


###########################################################
##LINEAR REGRESSIONS: TARGET = DEMEANED LIFE SATISFACTION##
###########################################################
                    
linreg_MD_PD_1 = linreg_train_test(X_train = X_train_MD_stand, 
                                   y_train = y_train_MD, 
                                   X_test = X_test_MD_stand, 
                                   y_test = y_test_MD)


linreg_MD_PD_1[2] 

#Test R2 = 0.01

linreg_MD_PD_1[3] 

#Train R2 = 0.01

linreg_MD_PD_1[0] 

#Test MSE = 1.17

linreg_MD_PD_1[1] 

#Train MSE = 1.17

#########################################################
###RANDOM FORESTS: TARGET = DEMEANED LIFE SATISFACTION###
#########################################################

start_time = time.time()

RF_MD_PD_1 = RandomForest(X_train = X_train_MD_stand, 
                       y_train = y_train_MD, 
                       if_bootstrap = True,
                       optim = True, 
                       n_trees = [1000], 
                       n_max_feats = [21], 
                       n_max_depth = [21], 
                       n_min_sample_leaf = [1], 
                       n_cv = 4, 
                       X_test = X_test_MD_stand,
                       y_test = y_test_MD)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

#Attempts


RF_MD_PD_1[1]

RF_MD_PD_1[5]

#Test R2 = 0.015

RF_MD_PD_1[6]

#Train R2 = 0.18315027523171612

RF_MD_PD_1[3]

#Test MSE = 1.16

RF_MD_PD_1[4]

#Train MSE = 0.96

############################################################
###GRADIENT BOOSTING: TARGET = DEMEANED LIFE SATISFACTION###
############################################################

start_time = time.time()

GB_MD_PD_1 = GradBoostReg(X_train = X_train_MD_stand, 
                       y_train = y_train_MD, 
                       lr = [0.02],
                       n_iters = [400],
                       max_depth = [8], 
                       subsample_frac = [0.75],
                       max_feats = [6], 
                       n_cv = 4, 
                       X_test = X_test_MD_stand,
                       y_test = y_test_MD)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

GB_MD_PD_1[1]

GB_MD_PD_1[5]

#Test R2 = 0.014

GB_MD_PD_1[6]

#Train R2 = 0.06

GB_MD_PD_1[3]

#Test MSE = 1.16

GB_MD_PD_1[4]

#Train MSE = 1.11


#######################################
####PI OLS ON Y = LIFE SATISFACTION####
#######################################

X_test_MD_stand_const = sm.add_constant(X_test_MD_stand, has_constant = 'add')

start_time = time.time()

PI_pooled_OLS = permutation_importance(estimator = linreg_MD_PD[-1], 
                                       X = X_test_MD_stand_const,
                                       y = y_test_PD_y['Life Satisfaction'],
                                       n_jobs = 1,
                                       n_repeats = 10,
                                       scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_pooled_OLS.importances_mean.argsort()[::-1]

PI_pooled_OLS_list = []

for i in perm_sorted_idx_r2:
    
    PI_pooled_OLS_list.append([list(X_test_MD_stand_const)[i], PI_pooled_OLS.importances_mean[i], PI_pooled_OLS.importances_std[i]])
    
PI_pooled_OLS_df = pd.DataFrame(PI_pooled_OLS_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
PI_pooled_OLS_df.to_csv('C:\\Some\\Local\\Path\\Results_pooled_OLS_ls.csv')

##Permutation importance for age and age square jointly
#Permuting jointly M[D]_Age and M[D]_Age^2, as well as 
#P[D]_Age and P[D]_Age^2

#P[D]_Age and P[D]_Age^2

age_pis_PD_ols_pooled = np.empty(10)

for i in range(10):
        
    age_vars_permuted = X_test_MD_stand_const[['P[D]_Age','P[D]_Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_MD_stand_const.copy()
    
    X_test_stand_permuted['P[D]_Age'] = age_vars_permuted['P[D]_Age']
    
    X_test_stand_permuted['P[D]_Age^2'] = age_vars_permuted['P[D]_Age^2']
    
    MSE_permuted = ((linreg_MD_PD[-1].predict(X_test_stand_permuted) - y_test_PD_y['Life Satisfaction'])**2).mean()  
    
    age_pis_PD_ols_pooled[i] = MSE_permuted - linreg_MD_PD[0] 
    
print(age_pis_PD_ols_pooled.mean())

#0.08

print(age_pis_PD_ols_pooled.std())

#0.003

print(age_pis_PD_ols_pooled)

#M[D]_Age and M[D]_Age^2

age_pis_MD_ols_pooled = np.empty(10)

for i in range(10):
        
    age_vars_permuted = X_test_MD_stand_const[['M[D]_Age','M[D]_Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_MD_stand_const.copy()
    
    X_test_stand_permuted['M[D]_Age'] = age_vars_permuted['M[D]_Age']
    
    X_test_stand_permuted['M[D]_Age^2'] = age_vars_permuted['M[D]_Age^2']
    
    MSE_permuted = ((linreg_MD_PD[-1].predict(X_test_stand_permuted) - y_test_PD_y['Life Satisfaction'])**2).mean()  
    
    age_pis_MD_ols_pooled[i] = MSE_permuted - linreg_MD_PD[0] 
    
print(age_pis_MD_ols_pooled.mean())

#0.001

print(age_pis_MD_ols_pooled.std())

#0.00033

print(age_pis_MD_ols_pooled)

################################################
####PI OLS ON Y = DEMEANED LIFE SATISFACTION####
################################################

X_test_MD_stand_const = sm.add_constant(X_test_MD_stand, has_constant = 'add')

start_time = time.time()

PI_demeaned_OLS = permutation_importance(estimator = linreg_MD_PD_1[-1], 
                                       X = X_test_MD_stand_const,
                                       y = y_test_MD,
                                       n_jobs = 1,
                                       n_repeats = 10,
                                       scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_demeaned_OLS.importances_mean.argsort()[::-1]

PI_demeaned_OLS_list = []

for i in perm_sorted_idx_r2:
    
    PI_demeaned_OLS_list.append([list(X_test_MD_stand_const)[i], PI_demeaned_OLS.importances_mean[i], PI_demeaned_OLS.importances_std[i]])
    
PI_demeaned_OLS_df = pd.DataFrame(PI_demeaned_OLS_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
PI_demeaned_OLS_df.to_csv('C:\\Some\\Local\\Path\\Results_demeaned_OLS_ls.csv')

##Permutation importance for age and age square jointly
#Permuting jointly M[D]_Age and M[D]_Age^2, as well as
#P[D]_Age and P[D]_Age^2

#P[D]_Age and P[D]_Age^2

age_pis_PD_ols_demeaned = np.empty(10)

for i in range(10):
        
    age_vars_permuted = X_test_MD_stand_const[['P[D]_Age','P[D]_Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_MD_stand_const.copy()
    
    X_test_stand_permuted['P[D]_Age'] = age_vars_permuted['P[D]_Age']
    
    X_test_stand_permuted['P[D]_Age^2'] = age_vars_permuted['P[D]_Age^2']
    
    MSE_permuted = ((linreg_MD_PD_1[-1].predict(X_test_stand_permuted) - y_test_MD)**2).mean()  
    
    age_pis_PD_ols_demeaned[i] = MSE_permuted - linreg_MD_PD_1[0] 
    
print(age_pis_PD_ols_demeaned.mean())

#0

print(age_pis_PD_ols_demeaned.std())

#0

print(age_pis_PD_ols_demeaned)

#0

#M[D]_Age and M[D]_Age^2

age_pis_MD_ols_demeaned = np.empty(10)

for i in range(10):
        
    age_vars_permuted = X_test_MD_stand_const[['M[D]_Age','M[D]_Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_MD_stand_const.copy()
    
    X_test_stand_permuted['M[D]_Age'] = age_vars_permuted['M[D]_Age']
    
    X_test_stand_permuted['M[D]_Age^2'] = age_vars_permuted['M[D]_Age^2']
    
    MSE_permuted = ((linreg_MD_PD_1[-1].predict(X_test_stand_permuted) - y_test_MD)**2).mean()  
    
    age_pis_MD_ols_demeaned[i] = MSE_permuted - linreg_MD_PD_1[0] 
    
print(age_pis_MD_ols_demeaned.mean())

#0.001

print(age_pis_MD_ols_demeaned.std())

#0.0002

print(age_pis_MD_ols_demeaned)

######################################
####PI RF ON Y = LIFE SATISFACTION####
######################################

start_time = time.time()

PI_pooled_RF = permutation_importance(estimator = RF_MD_PD[1], 
                                       X = X_test_MD_stand,
                                       y = y_test_PD_y['Life Satisfaction'],
                                       n_jobs = 1,
                                       n_repeats = 10,
                                       scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_pooled_RF.importances_mean.argsort()[::-1]

PI_pooled_RF_list = []

for i in perm_sorted_idx_r2:
    
    PI_pooled_RF_list.append([list(X_test_MD_stand)[i], PI_pooled_RF.importances_mean[i], PI_pooled_RF.importances_std[i]])
    
PI_pooled_RF_df = pd.DataFrame(PI_pooled_RF_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
PI_pooled_RF_df.to_csv('C:\\Some\\Local\\Path\\Results_pooled_RF_ls.csv')

##Permutation importance for age and age square jointly
#Permuting jointly M[D]_Age and M[D]_Age^2, as well as
#P[D]_Age and P[D]_Age^2

#P[D]_Age and P[D]_Age^2

age_pis_PD_RF_pooled = np.empty(10)

for i in range(10):
        
    age_vars_permuted = X_test_MD_stand[['P[D]_Age','P[D]_Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_MD_stand.copy()
    
    X_test_stand_permuted['P[D]_Age'] = age_vars_permuted['P[D]_Age']
    
    X_test_stand_permuted['P[D]_Age^2'] = age_vars_permuted['P[D]_Age^2']
    
    MSE_permuted = ((RF_MD_PD[1].predict(X_test_stand_permuted) - y_test_PD_y['Life Satisfaction'])**2).mean()  
    
    age_pis_PD_RF_pooled[i] = MSE_permuted - RF_MD_PD[3] 
    
print(age_pis_PD_RF_pooled.mean())

#0.13

print(age_pis_PD_RF_pooled.std())

#0.004

print(age_pis_PD_RF_pooled)

#M[D]_Age and M[D]_Age^2

age_pis_MD_RF_pooled = np.empty(10)

for i in range(10):
        
    age_vars_permuted = X_test_MD_stand[['M[D]_Age','M[D]_Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_MD_stand.copy()
    
    X_test_stand_permuted['M[D]_Age'] = age_vars_permuted['M[D]_Age']
    
    X_test_stand_permuted['M[D]_Age^2'] = age_vars_permuted['M[D]_Age^2']
    
    MSE_permuted = ((RF_MD_PD[1].predict(X_test_stand_permuted) - y_test_PD_y['Life Satisfaction'])**2).mean()  
    
    age_pis_MD_RF_pooled[i] = MSE_permuted - RF_MD_PD[3] 
    
print(age_pis_MD_RF_pooled.mean())

#0.008

print(age_pis_MD_RF_pooled.std())

#0.001

print(age_pis_MD_RF_pooled)

################################################
####PI RF ON Y = DEMEANED LIFE SATISFACTION####
################################################

start_time = time.time()

PI_demeaned_RF = permutation_importance(estimator = RF_MD_PD_1[1], 
                                       X = X_test_MD_stand,
                                       y = y_test_MD,
                                       n_jobs = 1,
                                       n_repeats = 10,
                                       scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_demeaned_RF.importances_mean.argsort()[::-1]

PI_demeaned_RF_list = []

for i in perm_sorted_idx_r2:
    
    PI_demeaned_RF_list.append([list(X_test_MD_stand)[i], PI_demeaned_RF.importances_mean[i], PI_demeaned_RF.importances_std[i]])
    
PI_demeaned_RF_df = pd.DataFrame(PI_demeaned_RF_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
PI_demeaned_RF_df.to_csv('C:\\Some\\Local\\Path\\Results_demeaned_RF_ls.csv')

##Permutation importance for age and age square jointly
#Permuting jointly M[D]_Age and M[D]_Age^2, as well as
#P[D]_Age and P[D]_Age^2

#P[D]_Age and P[D]_Age^2

age_pis_PD_RF_demeaned = np.empty(10)

for i in range(10):
        
    age_vars_permuted = X_test_MD_stand[['P[D]_Age','P[D]_Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_MD_stand.copy()
    
    X_test_stand_permuted['P[D]_Age'] = age_vars_permuted['P[D]_Age']
    
    X_test_stand_permuted['P[D]_Age^2'] = age_vars_permuted['P[D]_Age^2']
    
    MSE_permuted = ((RF_MD_PD_1[1].predict(X_test_stand_permuted) - y_test_MD)**2).mean()  
    
    age_pis_PD_RF_demeaned[i] = MSE_permuted - RF_MD_PD_1[3] 
    
print(age_pis_PD_RF_demeaned.mean())

#0.002

print(age_pis_PD_RF_demeaned.std())

#0.0002

print(age_pis_PD_RF_demeaned)

#M[D]_Age and M[D]_Age^2

age_pis_MD_RF_demeaned = np.empty(10)

for i in range(10):
        
    age_vars_permuted = X_test_MD_stand[['M[D]_Age','M[D]_Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_MD_stand.copy()
    
    X_test_stand_permuted['M[D]_Age'] = age_vars_permuted['M[D]_Age']
    
    X_test_stand_permuted['M[D]_Age^2'] = age_vars_permuted['M[D]_Age^2']
    
    MSE_permuted = ((RF_MD_PD_1[1].predict(X_test_stand_permuted) - y_test_MD)**2).mean()  
    
    age_pis_MD_RF_demeaned[i] = MSE_permuted - RF_MD_PD_1[3] 
    
print(age_pis_MD_RF_demeaned.mean())

#0.007

print(age_pis_MD_RF_demeaned.std())

#0.0004

print(age_pis_MD_RF_demeaned)

######################################
####PI GB ON Y = LIFE SATISFACTION####
######################################

start_time = time.time()

PI_pooled_GB = permutation_importance(estimator = GB_MD_PD[1], 
                                       X = X_test_MD_stand,
                                       y = y_test_PD_y['Life Satisfaction'],
                                       n_jobs = 1,
                                       n_repeats = 10,
                                       scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_pooled_GB.importances_mean.argsort()[::-1]

PI_pooled_GB_list = []

for i in perm_sorted_idx_r2:
    
    PI_pooled_GB_list.append([list(X_test_MD_stand)[i], PI_pooled_GB.importances_mean[i], PI_pooled_GB.importances_std[i]])
    
PI_pooled_GB_df = pd.DataFrame(PI_pooled_GB_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
PI_pooled_GB_df.to_csv('C:\\Some\\Local\\Path\\Results_pooled_GB_ls.csv')

#Permutation importance for age and age square jointly
#Permuting jointly M[D]_Age and M[D]_Age^2, as well as
#P[D]_Age and P[D]_Age^2

#P[D]_Age and P[D]_Age^2

age_pis_PD_GB_pooled = np.empty(10)

for i in range(10):
        
    age_vars_permuted = X_test_MD_stand[['P[D]_Age','P[D]_Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_MD_stand.copy()
    
    X_test_stand_permuted['P[D]_Age'] = age_vars_permuted['P[D]_Age']
    
    X_test_stand_permuted['P[D]_Age^2'] = age_vars_permuted['P[D]_Age^2']
    
    MSE_permuted = ((GB_MD_PD[1].predict(X_test_stand_permuted) - y_test_PD_y['Life Satisfaction'])**2).mean()  
    
    age_pis_PD_GB_pooled[i] = MSE_permuted - GB_MD_PD[3] 
    
print(age_pis_PD_GB_pooled.mean())

#0.123

print(age_pis_PD_GB_pooled.std())

#0.003

print(age_pis_PD_GB_pooled)

#M[D]_Age and M[D]_Age^2

age_pis_MD_GB_pooled = np.empty(10)

for i in range(10):
        
    age_vars_permuted = X_test_MD_stand[['M[D]_Age','M[D]_Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_MD_stand.copy()
    
    X_test_stand_permuted['M[D]_Age'] = age_vars_permuted['M[D]_Age']
    
    X_test_stand_permuted['M[D]_Age^2'] = age_vars_permuted['M[D]_Age^2']
    
    MSE_permuted = ((GB_MD_PD[1].predict(X_test_stand_permuted) - y_test_PD_y['Life Satisfaction'])**2).mean()  
    
    age_pis_MD_GB_pooled[i] = MSE_permuted - GB_MD_PD[3] 
    
print(age_pis_MD_GB_pooled.mean())

#0.009

print(age_pis_MD_GB_pooled.std())

#0.001

print(age_pis_MD_GB_pooled)

################################################
####PI GB ON Y = DEMEANED LIFE SATISFACTION####
################################################

start_time = time.time()

PI_demeaned_GB = permutation_importance(estimator = GB_MD_PD_1[1], 
                                       X = X_test_MD_stand,
                                       y = y_test_MD,
                                       n_jobs = 1,
                                       n_repeats = 10,
                                       scoring = 'r2')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_r2 = PI_demeaned_GB.importances_mean.argsort()[::-1]

PI_demeaned_GB_list = []

for i in perm_sorted_idx_r2:
    
    PI_demeaned_GB_list.append([list(X_test_MD_stand)[i], PI_demeaned_GB.importances_mean[i], PI_demeaned_GB.importances_std[i]])
    
PI_demeaned_GB_df = pd.DataFrame(PI_demeaned_GB_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
PI_demeaned_GB_df.to_csv('C:\\Some\\Local\\Path\\Results_demeaned_GB_ls.csv')

##Permutation importance for age and age square jointly
#Permuting jointly M[D]_Age and M[D]_Age^2, as well as
#P[D]_Age and P[D]_Age^2

#P[D]_Age and P[D]_Age^2

age_pis_PD_GB_demeaned = np.empty(10)

for i in range(10):
        
    age_vars_permuted = X_test_MD_stand[['P[D]_Age','P[D]_Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_MD_stand.copy()
    
    X_test_stand_permuted['P[D]_Age'] = age_vars_permuted['P[D]_Age']
    
    X_test_stand_permuted['P[D]_Age^2'] = age_vars_permuted['P[D]_Age^2']
    
    MSE_permuted = ((GB_MD_PD_1[1].predict(X_test_stand_permuted) - y_test_MD)**2).mean()  
    
    age_pis_PD_GB_demeaned[i] = MSE_permuted - GB_MD_PD_1[3] 
    
print(age_pis_PD_GB_demeaned.mean())

#0.002

print(age_pis_PD_GB_demeaned.std())

#0.0003

print(age_pis_PD_GB_demeaned)

#M[D]_Age and M[D]_Age^2

age_pis_MD_GB_demeaned = np.empty(10)

for i in range(10):
        
    age_vars_permuted = X_test_MD_stand[['M[D]_Age','M[D]_Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_MD_stand.copy()
    
    X_test_stand_permuted['M[D]_Age'] = age_vars_permuted['M[D]_Age']
    
    X_test_stand_permuted['M[D]_Age^2'] = age_vars_permuted['M[D]_Age^2']
    
    MSE_permuted = ((GB_MD_PD_1[1].predict(X_test_stand_permuted) - y_test_MD)**2).mean()  
    
    age_pis_MD_GB_demeaned[i] = MSE_permuted - GB_MD_PD_1[3] 
    
print(age_pis_MD_GB_demeaned.mean())

#0.006

print(age_pis_MD_GB_demeaned.std())

#0.0003

print(age_pis_MD_GB_demeaned)






