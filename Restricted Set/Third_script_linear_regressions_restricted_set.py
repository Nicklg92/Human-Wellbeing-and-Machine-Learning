######################################################################
###THIRD SCRIPT - LINEAR REGRESSIONS ON RESTRICTED SET OF VARIABLES###
######################################################################

import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.inspection import permutation_importance

np.random.seed(1123581321)

'''
COMMENTS:

This is the third script of the Restricted Set producing the results in 
"Machine Learning in the Prediction of Human
Wellbeing", joint first authorship by Oparina, E., Kaiser, C., 
and Gentile, N., et al.    

Aim of this script is to run the Linear Regressions 
on the restricted set of variables, cross-sectionally on the 
different years.

In particular, beside simply fitting and predicting with a 
hyperplane:
    
-- Potential multicollinearity issues are taken into account,
by dropping perfectly (multi)collinear variables 
(e.g., columns of dummies consisting of only 0s in the training
sets, easily collinear since linear combination of 0 times any
variable).

-- The comparison in predictive performance ablating or not
health variables variables is performed. These are the values 
of Table 1 in the paper.

-- The Permutation Feature Importances are  computed, 
uniquely for the year 2013 - so to be compared with the values
of the Extended Set.

-- For the particular case of Age, since there is also Age**2
among the predictors, the Permutation Importance is computed
manually, by permuting Age and Age**2 together.

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

##############################
###LINEAR REGRESSION MODELS###
##############################

linreg_results = []

for i in range(0, len(yearly_dsets_train)):
    
    year = list(yearly_dsets_train[i]['year'].unique())[0]
    
    #We delete the categorical variables that have indeed been 
    #one-hot-encoded. Also, we delete extraneous meta-data life pid, hid,
    #as well as redundant information like the (constant) year, and
    #Household yearly disposable income (we have Adjusted Income).
    
    X_train_i = yearly_dsets_train[i].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                            'State of Residence', 'hid','Religion','Month of Interview',
                                            'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
    X_test_i = yearly_dsets_test[i].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                           'State of Residence', 'hid','Religion','Month of Interview',
                                           'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 

    y_train_i = yearly_dsets_train[i]['Life Satisfaction']
    
    y_test_i = yearly_dsets_test[i]['Life Satisfaction']
    
    #We further drop potentially one-value-only, perfectly
    #multicollinear variables.
    
    for i in list(X_train_i):
        
        if X_train_i[i].nunique() == 1:
            
            X_train_i.drop(i, axis = 1, inplace = True)
            
            X_test_i.drop(i, axis = 1, inplace = True)
    
    results_i = linreg_train_test(X_train = X_train_i, 
                                  y_train = y_train_i, 
                                  X_test = X_test_i, 
                                  y_test = y_test_i)

    results_i.append(year)
    
    linreg_results.append(results_i)
    
linreg_results_pd = pd.DataFrame(linreg_results, columns = ['Test MSE', 'Train MSE', 'Test_R2', 'Train_R2', 'Model', 'Year']) 

linreg_results_pd.drop(["Model"], axis = 1, inplace = True)

linreg_results_pd.loc['Averages'] = linreg_results_pd.mean()

avgs = linreg_results_pd.loc['Averages']

#linreg_results_pd.to_csv('C:\\Some\\Local\\Path\\Linreg_results_restricted.csv')

##############################################
###LINEAR REGRESSION ON 2013 WITHOUT HEALTH###
##############################################

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

#X_train_13_stand.drop(['Disability Status', 'Number of doctor visits in previous year'], axis = 1, inplace = True)

#X_test_13_stand.drop(['Disability Status', 'Number of doctor visits in previous year'], axis = 1, inplace = True)

Linreg_13 = linreg_train_test(X_train = X_train_13_stand, 
                              y_train = y_train_13, 
                              X_test = X_test_13_stand, 
                              y_test = y_test_13)

Linreg_13[0]

#Test MSE = 2.78
#Test MSE without health = 2.84

Linreg_13[1]

#Train MSE = 2.73
#Train MSE without health = 2.81

Linreg_13[2]

#Test R2 = 0.09
#Test R2 without health = 0.07

Linreg_13[3]

#Train R2 = 0.11
#Train R2 without health = 0.08

#####################
##PI MSE OLS - 2013##
#####################

#In the below, we chose to not parallelize the task (n_jobs = 1 instead of
#-1) so to not paralyze the computer and be able to perform other tasks at the
#same time.

X_test_13_stand_const = sm.add_constant(X_test_13_stand, has_constant = 'add')

start_time = time.time()

PI_13_mse = permutation_importance(estimator = Linreg_13[-1], 
                                   X = X_test_13_stand_const,
                                   y = y_test_13,
                                   n_jobs = 1,
                                   n_repeats = 10,
                                   scoring = 'neg_mean_squared_error')

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

perm_sorted_idx_mse = PI_13_mse.importances_mean.argsort()[::-1]

PI_13_mse_list = []

for i in perm_sorted_idx_mse:
    
    PI_13_mse_list.append([list(X_test_13_stand_const)[i], PI_13_mse.importances_mean[i], PI_13_mse.importances_std[i]])
    
PI_13_mse_df = pd.DataFrame(PI_13_mse_list, columns = ['Variable', 'Average PI as of MSE in 10 reps', 'SD PI as of MSE in 10 reps'])
    
#PI_13_mse_df.to_csv('C:\\Some\\Local\\Path\\PIs_13_OLS.csv')

##Permutation importance for age and age square jointly, done separately

age_pis = np.empty(10)

for i in range(10):
    
    age_vars_permuted = X_test_13_stand[['Age','Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_13_stand.copy()
    
    X_test_stand_permuted['Age'] = age_vars_permuted['Age']
    
    X_test_stand_permuted['Age^2'] = age_vars_permuted['Age^2']
    
    MSE_permuted = ((Linreg_13[-1].predict(sm.add_constant(X_test_stand_permuted, has_constant = 'add')) - y_test_13)**2).mean()  
    
    age_pis[i] = MSE_permuted - Linreg_13[0] 
    
print(age_pis.mean())

print(age_pis.std())

print(age_pis)

#del X_test_stand_permuted
#del age_vars_permuted
#del age_pis

