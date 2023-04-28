##########################################################################
###EIGHTH SCRIPT - AVERAGE EFFECTS OF INCOME AND AGE IN RESTRICTED SET####
##########################################################################

import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor


np.random.seed(1123581321)

'''
COMMENT

This is the eighth and final script of the Restricted Set producing the results in 
"Machine Learning in the Prediction of Human
Wellbeing", joint first authorship by Oparina, E., Kaiser, C., 
and Gentile, N., et al.  

Aim of this script is to compute the average effects of income
and age in the Restricted Set 2013.
'''

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

###########################################
###AVERAGE EFFECTS ON PROTECTED SET 2013###
###########################################
    
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
    
X_train_13 = yearly_dsets_train[3].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                         'State of Residence', 'hid','Religion','Month of Interview',
                                         'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
X_test_13 = yearly_dsets_test[3].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                       'State of Residence', 'hid','Religion','Month of Interview',
                                       'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
y_train_13 = yearly_dsets_train[3]['Life Satisfaction']
    
y_test_13 = yearly_dsets_test[3]['Life Satisfaction']

X_train_13_stand = pd.DataFrame(scaler.fit_transform(X_train_13), index = y_train_13.index)

X_test_13_stand = pd.DataFrame(scaler.transform(X_test_13), index = y_test_13.index)

X_train_13_stand.columns = list(X_train_13)

X_test_13_stand.columns = list(X_test_13)

del yearly_dsets_train, yearly_dsets_test

saving_pred_path = 'C:\\Some\\Local\\Path\\'

#We define the values to standardize income and age each time
#we predict the average life satisfaction fixing a different
#age or income level.

lnmu = X_train_13['Adjusted Income'].mean() #9.920028439394548

lnsi = X_train_13['Adjusted Income'].std() #0.5750788029849921

agemu = X_train_13['Age'].mean() #46.94726396370854

agesi = X_train_13['Age'].std() #17.04424512942103

agesqmu = X_train_13['Age^2'].mean() #2494.5401190813723

agesqsi = X_train_13['Age^2'].std() #1726.3032558273792

#########
###OLS###
#########

Linreg_13 = linreg_train_test(X_train = X_train_13_stand, 
                              y_train = y_train_13, 
                              X_test = X_test_13_stand, 
                              y_test = y_test_13)

##########
##INCOME##
##########

tmp_data = sm.add_constant(X_test_13_stand.copy(), has_constant = 'add')

#We update the upper level of the income range with 
#a value close to max of exp(Adj_Income).

Adj_income_nolog = np.exp(X_test_13['Adjusted Income'])

Adj_income_nolog.max()

#776104.4324676052

max_int = 700000

num_bands = 2000

inc_predictions = pd.DataFrame(range(1, max_int, num_bands), columns=['income'])

tmp = np.empty(int(max_int / num_bands)) #350 = 700000/2000

n = 0

for i in range(1, max_int, num_bands):
    
    print(i)  
    
    tmp_data['Adjusted Income'] = (np.log(i) - lnmu)/lnsi
    
    tmp[n] = Linreg_13[-1].predict(tmp_data).mean()
    
    n += 1
    
inc_predictions['lshat_ols'] = tmp

inc_predictions.to_csv(saving_pred_path + 'inc_predictions_OLS_restricted.csv', index = False)

del tmp

del tmp_data


#########
###AGE###
#########

tmp_data = sm.add_constant(X_test_13_stand.copy(), has_constant = 'add')

age_predictions = pd.DataFrame(range(16, 96, 1), columns=['Age'])

tmp = np.empty(80) #this time, simply 96 - 16

n = 0

for i in range(16, 96, 1):
    
    tmp_data['Age'] = (i - agemu)/agesi
    
    tmp_data['Age^2'] = ((i**2) - agesqmu)/agesqsi
    
    tmp[n] = Linreg_13[-1].predict(tmp_data).mean()
    
    n += 1
    
age_predictions['lshat_ols'] = tmp

age_predictions.to_csv(path + 'age_predictions_OLS_restricted.csv',index=False)

del tmp

del tmp_data

###################
###RANDOM FOREST###
###################

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

############
###INCOME###
############

tmp_data = X_test_13_stand.copy()

tmp = np.empty(int(max_int / num_bands)) #350 = round(700000/2000) 

n=0

for i in range(1, max_int, num_bands):
    
    print(i)  
    
    tmp_data['Adjusted Income'] = (np.log(i) - lnmu)/lnsi
    
    tmp[n] = RF_2013[1].predict(tmp_data).mean()
    
    n += 1
    
inc_predictions['lshat_rf'] = tmp

inc_predictions.to_csv(saving_pred_path + 'inc_predictions_RF_restricted.csv', index = False)

del tmp

del tmp_data

#########
###AGE###
#########

tmp_data = X_test_13_stand.copy()

tmp = np.empty(80) #this time, simply 96 - 16 

n = 0

for i in range(16, 96, 1):
    
    tmp_data['Age'] = (i - agemu)/agesi
    
    tmp_data['Age^2'] = ((i**2) - agesqmu)/agesqsi
    
    tmp[n] = RF_2013[1].predict(tmp_data).mean()
    
    n += 1
    
age_predictions['lshat_rf']=tmp

age_predictions.to_csv(saving_pred_path + 'age_predictions_RF_restricted.csv',index=False)

del tmp

del tmp_data

#######################
###GRADIENT BOOSTING###
#######################

GB_2013 = GradBoostReg(X_train = X_train_13_stand, 
                       y_train = y_train_13, 
                       lr = [0.003],
                       n_iters = [3500],  
                       max_depth = [7],  
                       subsample_frac = [0.5],
                       max_feats = [6],
                       n_cv = 4, 
                       X_test = X_test_13_stand,
                       y_test = y_test_13)


############
###INCOME###
############

tmp_data = X_test_13_stand.copy()

tmp = np.empty(int(max_int / num_bands)) #350 = round(700000/2000) 

n=0

for i in range(1, max_int, num_bands):
    
    print(i)  
    
    tmp_data['Adjusted Income'] = (np.log(i) - lnmu)/lnsi
    
    tmp[n] = GB_2013[1].predict(tmp_data).mean()
    
    n += 1
    
inc_predictions['lshat_gb']=tmp

inc_predictions.to_csv(saving_pred_path + 'inc_predictions_GB_restricted.csv', index = False)

del tmp

del tmp_data

#########
###AGE###
#########

tmp_data = X_test_13_stand.copy()

tmp = np.empty(80) #this time, simply 96 - 16 

n = 0

for i in range(16, 96, 1):
    
    tmp_data['Age'] = (i - agemu)/agesi
    
    tmp_data['Age^2'] = ((i**2) - agesqmu)/agesqsi
    
    tmp[n] = GB_2013[1].predict(tmp_data).mean()
    
    n += 1
    
age_predictions['lshat_gb'] = tmp

age_predictions.to_csv(saving_pred_path + 'age_predictions_GB_restricted.csv',index=False)

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

#########
###AGE###
#########

plt.plot(age_predictions['Age'][age_predictions['Age']<97],age_predictions['lshat_ols'][age_predictions['Age']<97], label='Predicted OLS')
plt.plot(age_predictions['Age'][age_predictions['Age']<97],age_predictions['lshat_rf'][age_predictions['Age']<97], label='Predicted RF', color = 'red', linestyle = 'dotted')
plt.plot(age_predictions['Age'][age_predictions['Age']<97],age_predictions['lshat_gb'][age_predictions['Age']<97], label='Predicted GB', color = 'green', linestyle = 'dashed')
plt.xlim(10, 100)
plt.ylim(6.5, 8.1) 
plt.yticks(np.arange(6.5, 8.3, 0.2)) 
plt.xlabel('Age')
plt.ylabel('Wellbeing')
plt.savefig(saving_graph_path + "fig4_2.png", dpi = 1200)
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
plt.ylim(6.5, 8.1) #
plt.yticks(np.arange(6.5, 8.3, 0.2)) 
plt.savefig(saving_graph_path + "fig4_1.png", dpi = 1200)
plt.show()

