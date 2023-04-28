#################################################################
##FIFTH SCRIPT - GRADIENT BOOSTING TREE REGRESSOR, SQUARED LOSS##
#################################################################

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,r2_score
from sklearn.inspection import permutation_importance

scaler = StandardScaler()

'''
COMMENTS

This is the fifth script of the Restricted Set producing the results in 
"Human Wellbeing and Machine Learning" by Ekaterina Oparina* (r) Caspar Kaiser* (r) Niccolò Gentile*; 
Alexandre Tkatchenko, Andrew E. Clark, Jan-Emmanuel De Neve 
and Conchita D'Ambrosio (* equal first authors in random order).   

Aim of this script is to fit and predict via Gradient Boosting
on the Restricted set of variables, cross-sectionally on the 
different years.

All the intermediate results along the hyperparameter optimization
path are not reported for the sake of readabilty. Please refer
to the paper for the specification of the final values of the
hyperparameters.

Also in this case, as for the Linear Regressions, the Permutation
Importances are computed considering only 2013, and performing
separatedly for Age and Age**2 (since permuted jointly).

The characterstic of GB is that using loss functions typical of
regression problmes (squared loss, absolute loss or Huber loss), the derivative of
the loss is equal to the residuals in the squared loss function, to their signs 
in the absolute loss and to the pseudo-residuals for Huber loss.

Moreover, with squared loss, we know that the optimal initial guess is the
mean.

Hence, also for coherence with the previous algorithms, we use the squared loss.

Differently from Random Forests, beside the tree-specific hyperparameters, there are
also three other hyperparameters to optimize over, namely: 
    
-- The amount of iterations (trees) to fit.
-- The learning rate
-- The fraction of the sample to be randomly extracted and used to fit at each new iteration.

Differently from Random Forests, in this case too many iterations (trees) can lead to
overfitting. Indeed, in this case, each tree is based on the estimated residuals (with squared
loss, in general on the gradient of the loss) at the previous iterations.

Too many iterations may lead back to an algorithm that perfectly fits the
training set. 

Similarly, the 0 < learning rate < 1 reduces the "greediness" of the procedure, by reducing
the length of optimization step induced by the gradient of the loss, 
since moving in the steepest direction.  

Clearly, the learning rate and the number of iterations compete:
the greater the learning rate, the less the iterations needed, and vice versa.                                                                            

In The Elements of Statistical Learning, it is however suggested to keep the learning rate
around 0.1 (default in sklearn, not by chance!) and keep high number of iterations.
The authors also talk about the possibility of using early stopping, but we avoid it.

The fraction of the sample to be randomly extract to build onto the successive
iteration should have the same role of nonparametrically bootstrapping in Random Forests, 
namely decorrelate the trees. If the sumbsampling fraction is smaller than 1
we talk about Stochastic Gradient Boosting (indeed our case).
'''

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
    
np.random.seed(1123581321)

################################
##GRADIENT BOOSTING TREES 2010##
################################

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

GB_2010 = GradBoostReg(X_train = X_train_10_stand, 
                       y_train = y_train_10, 
                       lr = [0.003],
                       n_iters = [4000],  
                       max_depth = [7],  
                       subsample_frac = [0.5],
                       max_feats = [5],
                       n_cv = 4, 
                       X_test = X_test_10_stand,
                       y_test = y_test_10)

best_gb_2010 = GB_2010[1]

test_mse_2010 = GB_2010[3]

#2.48

train_mse_2010 = GB_2010[4]

#2.03

test_r2_2010 = GB_2010[5]

#0.17

train_r2_2010 = GB_2010[6]

#0.33

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

################################
##GRADIENT BOOSTING TREES 2011##
################################

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

GB_2011 = GradBoostReg(X_train = X_train_11_stand, 
                       y_train = y_train_11, 
                       lr = [0.003],
                       n_iters = [3500],  
                       max_depth = [7],  
                       subsample_frac = [0.5],
                       max_feats = [6],
                       n_cv = 4, 
                       X_test = X_test_11_stand,
                       y_test = y_test_11)

best_gb_2011 = GB_2011[1]

test_mse_2011 = GB_2011[3]

#2.45

train_mse_2011 = GB_2011[4]

#2.05

test_r2_2011 = GB_2011[5]

#0.17

train_r2_2011 = GB_2011[6]

#0.33

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

################################
##GRADIENT BOOSTING TREES 2012##
################################

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

GB_2012 = GradBoostReg(X_train = X_train_12_stand, 
                       y_train = y_train_12, 
                       lr = [0.003],
                       n_iters = [3500],  
                       max_depth = [7],  
                       subsample_frac = [0.5],
                       max_feats = [6],
                       n_cv = 4, 
                       X_test = X_test_12_stand,
                       y_test = y_test_12)

best_gb_2012 = GB_2012[1]

test_mse_2012 = GB_2012[3]

#2.50

train_mse_2012 = GB_2012[4]

#2.05

test_r2_2012 = GB_2012[5]

#0.17

train_r2_2012 = GB_2012[6]

#0.32

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

################################
##GRADIENT BOOSTING TREES 2013##
################################

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

best_gb_2013 = GB_2013[1]

test_mse_2013 = GB_2013[3]

#2.70

train_mse_2013 = GB_2013[4]

#2.18

test_r2_2013 = GB_2013[5]

#0.12

train_r2_2013 = GB_2013[6]

#0.29

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

################################
##GRADIENT BOOSTING TREES 2014##
################################

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

GB_2014 = GradBoostReg(X_train = X_train_14_stand, 
                       y_train = y_train_14, 
                       lr = [0.003],
                       n_iters = [3500],  
                       max_depth = [7],  
                       subsample_frac = [0.5],
                       max_feats = [6],
                       n_cv = 4, 
                       X_test = X_test_14_stand,
                       y_test = y_test_14)

best_gb_2014 = GB_2014[1]

test_mse_2014 = GB_2014[3]

#2.59

train_mse_2014 = GB_2014[4]

#2.10

test_r2_2014 = GB_2014[5]

#0.13

train_r2_2014 = GB_2014[6]

#0.30

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

################################
##GRADIENT BOOSTING TREES 2015##
################################

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

GB_2015 = GradBoostReg(X_train = X_train_15_stand, 
                       y_train = y_train_15, 
                       lr = [0.002],
                       n_iters = [3500],  
                       max_depth = [10],  
                       subsample_frac = [0.5],
                       max_feats =  [4],
                       n_cv = 4, 
                       X_test = X_test_15_stand,
                       y_test = y_test_15)

best_gb_2015 = GB_2015[1]

test_mse_2015 = GB_2015[3]

#2.59

train_mse_2015 = GB_2015[4]

#1.74

test_r2_2015 = GB_2015[5]

#0.13

train_r2_2015 = GB_2015[6]

#0.42

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

################################
##GRADIENT BOOSTING TREES 2016##
################################

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

GB_2016 = GradBoostReg(X_train = X_train_16_stand, 
                       y_train = y_train_16, 
                       lr = [0.002],
                       n_iters = [4000],  
                       max_depth = [8],  
                       subsample_frac = [0.5],
                       max_feats = [4],
                       n_cv = 4, 
                       X_test = X_test_16_stand,
                       y_test = y_test_16)

best_gb_2016 = GB_2016[1]

test_mse_2016 = GB_2016[3]

#2.75

train_mse_2016 = GB_2016[4]

#2.35

test_r2_2016 = GB_2016[5]

#0.12

train_r2_2016 = GB_2016[6]

#0.29

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

################################
##GRADIENT BOOSTING TREES 2017##
################################

X_train_17 = yearly_dsets_train[7].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                         'State of Residence', 'hid','Religion','Month of Interview',
                                         'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
X_test_17 = yearly_dsets_test[7].drop(['Life Satisfaction', 'year', 'pid', 'Household yearly disposable income',
                                       'State of Residence', 'hid','Religion','Month of Interview',
                                       'Housing ownership status','Ethnicity', 'Marital Status'], axis = 1) 
    
y_train_17 = yearly_dsets_train[7]['Life Satisfaction']
    
y_test_17 = yearly_dsets_test[7]['Life Satisfaction']

X_train_17_stand = pd.DataFrame(scaler.fit_transform(X_train_17), index = y_train_17.index)

X_test_17_stand = pd.DataFrame(scaler.transform(X_test_17), index = y_test_17.index)

X_train_17_stand.columns = list(X_train_17)

X_test_17_stand.columns = list(X_test_17)

GB_2017 = GradBoostReg(X_train = X_train_17_stand, 
                       y_train = y_train_17, 
                       lr = [0.003],
                       n_iters = [4000],  
                       max_depth = [7],  
                       subsample_frac = [0.5],
                       max_feats = [4],
                       n_cv = 4, 
                       X_test = X_test_17_stand,
                       y_test = y_test_17)

best_gb_2017 = GB_2017[1]

test_mse_2017 = GB_2017[3]

#2.73

train_mse_2017 = GB_2017[4]

#2.33

test_r2_2017 = GB_2017[5]

#0.11

train_r2_2017 = GB_2017[6]

#0.27

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')

################################
##GRADIENT BOOSTING TREES 2018##
################################

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

GB_2018 = GradBoostReg(X_train = X_train_18_stand, 
                       y_train = y_train_18, 
                       lr = [0.003],
                       n_iters = [3500],  
                       max_depth = [6],  
                       subsample_frac = [0.5],
                       max_feats = [5],
                       n_cv = 4, 
                       X_test = X_test_18_stand,
                       y_test = y_test_18)

best_gb_2018 = GB_2018[1]

test_mse_2018 = GB_2018[3]

#2.72

train_mse_2018 = GB_2018[4]

#2.44

test_r2_2018 = GB_2018[5]

#0.11

train_r2_2018 = GB_2018[6]

#0.21

end_time = time.time()

print('Runtime was ' + str(end_time - start_time) + ' seconds')


####################
##PI MSE GB - 2013##
####################

start_time = time.time()

PI_13_mse = permutation_importance(estimator = GB_2013[1], 
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
    
PI_13_mse_df.to_csv('C:\\Some\\Local\\Path\\PI_13_mse_gb_with_joined_ages.csv')

##Permutation importance for age and age square jointly

age_pis = np.empty(10)

for i in range(10):
    
    age_vars_permuted = X_test_13_stand[['Age','Age^2']].sample(frac=1).reset_index(drop=True)
    
    X_test_stand_permuted = X_test_13_stand.copy()
    
    X_test_stand_permuted['Age'] = age_vars_permuted['Age']
    
    X_test_stand_permuted['Age^2'] = age_vars_permuted['Age^2']
    
    MSE_permuted = ((GB_2013[1].predict(X_test_stand_permuted) - y_test_13)**2).mean()  
    
    age_pis[i] = MSE_permuted - GB_2013[3]
    
print(age_pis.mean())

print(age_pis.std())

print(age_pis)
