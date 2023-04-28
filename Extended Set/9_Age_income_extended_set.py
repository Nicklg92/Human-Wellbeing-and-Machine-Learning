######################################################################
###NINTH SCRIPT - AVERAGE EFFECTS OF INCOME AND AGE ON EXTENDED SET###
######################################################################

import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

np.random.seed(1123581321)

'''
COMMENTS

This is the ninth script and final script regarding the Extended Set 
producing the results in "Machine Learning in the 
Prediction of Human Wellbeing", joint first authorship by Oparina, E., Kaiser, C., 
and Gentile, N., and et al.

Aim of this script is to compute the average effects of income
and age in the Extended Set 2013.

Since in the Post-LASSO Extended Set Age is not available,
here we use the entire Extended Set.
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

train_ks = pd.read_csv(read_path + 'train_ks_stand.csv') 

test_ks = pd.read_csv(read_path + 'test_ks_stand.csv')

y_train_ks = train_ks['lsat']

y_test_ks = test_ks['lsat']

X_train_ks = train_ks.drop(['lsat'], axis = 1)

X_test_ks = test_ks.drop(['lsat'], axis = 1)
        
const_in_train = []

for i in list(X_train_ks):
        
    if X_train_ks[i].nunique() == 1:
        
        const_in_train.append(i)
            
        X_train_ks.drop(i, axis = 1, inplace = True)
            
        X_test_ks.drop(i, axis = 1, inplace = True)
        
len(const_in_train)

#0

const_in_test = []

for i in list(X_test_ks):
        
    if X_test_ks[i].nunique() == 1:
        
        const_in_test.append(i)
            
        X_train_ks.drop(i, axis = 1, inplace = True)
            
        X_test_ks.drop(i, axis = 1, inplace = True)
        
len(const_in_test)

#10

#X_train_ks.shape

#23454 x 542

#X_test_ks.shape

#5864 x 542

multicoll = ["i11112", "m11101", "m11122", "m11123", 
             "plb0097_-2.0", "hlf0011_h_nan", "ple0004_2.0",
             "ple0164_-2.0", "plb0040_nan", "plj0022_-2.0",
             "e11103_2.0", "plb0022_h_5.0", "plb0022_h_9.0",
             "plb0035_nan", "plj0116_h_-2.0", "e11106_nan",
             "plb0041_-2.0", "hlf0073_-2.0", "plb0031_h_-2.0",
             "hlf0092_h_nan", "plb0103_-2.0", "plb0156_v1_nan"]
             
X_train_ks.drop(multicoll, axis = 1, inplace = True)

X_test_ks.drop(multicoll, axis = 1, inplace = True)

X_train_ks.drop(["pid"], axis = 1, inplace = True)

X_test_ks.drop(["pid"], axis = 1, inplace = True)

saving_pred_path = 'C:\\Some\\Local\\Path\\'

#In this case, X_train_ks is already standardized, hence we
#cannot simply extract the means and sd of lnhhinc, age,
#and age^2. Nonetheless, these variables are defined identically
#in the Restricted set, hence we can directly take 
#the values as observed in Eighth_script_age_income_restricted_set.py

lnmu = 9.920028439394548

lnsi = 0.5750788029849921

agemu = 46.94726396370854

agesi = 17.04424512942103

agesqmu = 2494.5401190813723

agesqsi = 1726.3032558273792

#Hence, standardizing the three variables each time
#will be possible. The only difference is in the naming:
    
#"Adjusted Income" in Restricted = "lnhhinc" in Extended
#"Age^2" in Restricted = "agesquared" in Extended
#"Age" in Restricted = "age" in Extended
  
#########
###OLS###
#########

linreg_ks = linreg_train_test(X_train = X_train_ks, 
                              y_train = y_train_ks, 
                              X_test = X_test_ks, 
                              y_test = y_test_ks)

##########
##INCOME##
##########

tmp_data = sm.add_constant(X_test_ks.copy(), has_constant = 'add')

max_int = 700000

num_bands = 2000

inc_predictions = pd.DataFrame(range(1, max_int, num_bands), columns=['income'])

tmp = np.empty(int(max_int / num_bands)) #350 = round(700000/2000) 

n=0

for i in range(1, max_int, num_bands):
    
    print(i)  
    
    tmp_data['lnhhinc'] = (np.log(i) - lnmu)/lnsi
    
    tmp[n] = linreg_ks[-1].predict(tmp_data).mean()
    
    n += 1
    
inc_predictions['lshat_ols'] = tmp

inc_predictions.to_csv(saving_pred_path + 'inc_predictions_OLS_ks.csv', index = False)

del tmp

del tmp_data

#########
###AGE###
#########

tmp_data = sm.add_constant(X_test_ks.copy(), has_constant = 'add')

age_predictions = pd.DataFrame(range(16, 96, 1), columns=['age'])

tmp = np.empty(80) #this time, simply 96 - 16

n = 0

for i in range(16, 96, 1):
    
    tmp_data['age'] = (i - agemu)/agesi
    
    tmp_data['agesquared'] = ((i**2) - agesqmu)/agesqsi
    
    tmp[n] = linreg_ks[-1].predict(tmp_data).mean()
    
    n += 1
    
age_predictions['lshat_ols'] = tmp

age_predictions.to_csv(saving_pred_path + 'age_predictions_OLS_ks.csv',index=False)

#################
##RANDOM FOREST##
#################

start_time = time.time()

RF_ks = RandomForest(X_train = X_train_ks, 
                       y_train = y_train_ks, 
                       if_bootstrap = True,
                       optim = True, 
                       n_trees = [1000], 
                       n_max_feats = [225], 
                       n_max_depth = [96], 
                       n_min_sample_leaf = [1], 
                       n_cv = 4, 
                       X_test = X_test_ks,
                       y_test = y_test_ks)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

tmp_data = X_test_ks.copy()

max_int = 700000

num_bands = 2000

tmp = np.empty(int(max_int / num_bands)) #350 = round(700000/2000) 

n=0

for i in range(1, max_int, num_bands):
    
    print(i)  
    
    tmp_data['lnhhinc'] = (np.log(i) - lnmu)/lnsi
    
    tmp[n] = RF_ks[1].predict(tmp_data).mean()
    
    n += 1
    
inc_predictions['lshat_rf']=tmp

inc_predictions.to_csv(saving_pred_path + 'inc_predictions_RF_ks.csv', index = False)

del tmp

del tmp_data

#########
###AGE###
#########

tmp_data = X_test_ks.copy()

tmp = np.empty(80) #this time, simply 96 - 16 

n = 0

for i in range(16, 96, 1):
    
    tmp_data['age'] = (i - agemu)/agesi
    
    tmp_data['agesquared'] = ((i**2) - agesqmu)/agesqsi
    
    tmp[n] = RF_ks[1].predict(tmp_data).mean()
    
    n += 1
    
age_predictions['lshat_rf'] = tmp

age_predictions.to_csv(saving_pred_path + 'age_predictions_RF_ks.csv',index=False)

del tmp

del tmp_data

#####################
##GRADIENT BOOSTING##
#####################

GB_ks = GradBoostReg(X_train = X_train_ks, 
                     y_train = y_train_ks, 
                     lr = [0.01],
                     n_iters = [6000],
                     max_depth = [8], 
                     subsample_frac = [0.75],
                     max_feats = [75], 
                     n_cv = 4, 
                     X_test = X_test_ks,
                     y_test = y_test_ks)

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

############
###INCOME###
############

tmp_data = X_test_ks.copy()

max_int = 700000

num_bands = 2000

tmp = np.empty(int(max_int / num_bands)) #350 = round(700000/2000) 

n=0

for i in range(1, max_int, num_bands):
    
    print(i)  
    
    tmp_data['lnhhinc'] = (np.log(i) - lnmu)/lnsi
    
    tmp[n] = GB_ks[1].predict(tmp_data).mean()
    
    n += 1
    
inc_predictions['lshat_gb']=tmp

inc_predictions.to_csv(saving_pred_path + 'inc_predictions_GB_ks.csv', index = False)

del tmp

del tmp_data

#########
###AGE###
#########

tmp_data = X_test_ks.copy()

tmp = np.empty(80) #this time, simply 96 - 16 

n = 0

for i in range(16, 96, 1):
    
    tmp_data['age'] = (i - agemu)/agesi
    
    tmp_data['agesquared'] = ((i**2) - agesqmu)/agesqsi
    
    tmp[n] = GB_ks[1].predict(tmp_data).mean()
    
    n += 1
    
age_predictions['lshat_gb']=tmp

age_predictions.to_csv(saving_pred_path + 'age_predictions_GB_ks.csv',index=False)

del tmp

del tmp_data


saving_graph_path = 'C:\\Some\\Local\\Path\\'

####################
##MAKING THE PLOTS##
####################

print(list(age_predictions))

#['Age', 'lshat_ols', 'lshat_rf', 'lshat_gb']

print(list(inc_predictions))

#['income', 'lshat_ols', 'lshat_rf', 'lshat_gb']

####################
##MAKING THE PLOTS##
####################

#########
###AGE###
#########
plt.plot(age_predictions['age'][age_predictions['age']<97],age_predictions['lshat_ols'][age_predictions['age']<97], label='Predicted OLS')
plt.plot(age_predictions['age'][age_predictions['age']<97],age_predictions['lshat_rf'][age_predictions['age']<97], label='Predicted RF', color = 'red', linestyle = 'dotted')
plt.plot(age_predictions['age'][age_predictions['age']<97],age_predictions['lshat_gb'][age_predictions['age']<97], label='Predicted GB', color = 'green', linestyle = 'dashed')
plt.ylim(6.5, 8.3)
plt.xlim(10, 100)
plt.yticks(np.arange(6.5, 8.3, 0.2)) 
plt.xlabel('Age')
plt.ylabel('Wellbeing')
plt.savefig(saving_graph_path + "fig_a3_2.png", dpi = 1200)
plt.show()

############
###INCOME###
############

plt.plot(inc_predictions['income']/1000,inc_predictions['lshat_ols'], label='Predicted OLS')
plt.plot(inc_predictions['income']/1000,inc_predictions['lshat_rf'], label='Predicted RF', color = 'red', linestyle = 'dotted')
plt.plot(inc_predictions['income']/1000,inc_predictions['lshat_gb'], label='Predicted GB', color = 'green', linestyle = 'dashed')
plt.xlabel('Equiv. HH Income, annual (000)')
plt.ylabel('Wellbeing')
plt.xlim(-2, 172) 
plt.ylim(6.5, 8.3)
plt.yticks(np.arange(6.5, 8.3, 0.2)) 
plt.savefig(saving_graph_path + "fig_a3_1.png", dpi = 1200)
plt.show()



