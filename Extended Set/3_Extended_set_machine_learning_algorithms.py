################################################################
###THIRD SCRIPT - MACHINE LEARNING ALGORITHMS ON EXTENDED SET###
################################################################

import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(1123581321)

'''
COMMENTS

This is the third script regarding the Extended Set 
producing the results in  "Human Wellbeing and Machine Learning" by Ekaterina Oparina* (r) Caspar Kaiser* (r) Niccolò Gentile*; 
Alexandre Tkatchenko, Andrew E. Clark, Jan-Emmanuel De Neve 
and Conchita D'Ambrosio (* equal first authors in random order).

Aim of this script is to run ML algorithms on the Extended set.
Since one year only, differently from the Restricted set, all 
algorithms are fit in one script.

Upon running the linear regressions, however, strong 
multicollinearities issues had been observed.

In R, the lm() function fitting the linear regression, automatically
identifies the perfectly multicollinear variables. 

The results can be observed in the R script Miscellanea_untangling_issues_with_OLS,
in which this same diagnostic check is performed on the OLS also
in the Post-LASSO Extended set case and Panel analysis.

Moreover, the Variance Inflation Factor(s) were computed to further
deep dive.

This issue is described in-depth in the script below.

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

###########################
###FITTING THE ALGORTHMS###
###########################

#############################
###MULTICOLLINEARITY CHECK###
#############################

#X_train_ks_const = sm.add_constant(X_train_ks, has_constant = 'add')

#X_train_ks_const.drop(['pid'], axis = 1, inplace = True)

#model = sm.OLS(y_train_ks, X_train_ks_const)

#results = model.fit()

#print(results.summary())

#has indeed multicollinearity issues. One good way to start
#analyzing them is via the Variance Inflation Factor.

##############################
###VARIANCE INFLATION FACTOR##
############################## 

#from statsmodels.stats.outliers_influence import variance_inflation_factor

#Calculate the VIF for each independent variable

#vif_data = pd.DataFrame()

#vif_data["Variable"] = X_train_ks_const.columns

#vif_data["VIF"] = [variance_inflation_factor(X_train_ks_const.values, i) for i in range(X_train_ks_const.shape[1])]

#vif_sorted = vif_data.sort_values(by = ['VIF'], ascending = False)

#vif_sorted.to_csv("C:\\Some\\Local\\Path\\vif_sorted_after_multicoll_clean.csv")

#184 of them have a VIF larger than 10. However, not that
#many are needed to be dropped to solve the perfect 
#multicollinearity.

#Therefore, we also investigate the multicollinearity issues in 
#the R script Miscellanea_untangling_issues_with_OLS

#Hence why we manually drop here the following 22 variables,
#already sufficient to resolve the perfect multicollinearity:
    
multicoll = ["i11112", "m11101", "m11122", "m11123", 
             "plb0097_-2.0", "hlf0011_h_nan", "ple0004_2.0",
             "ple0164_-2.0", "plb0040_nan", "plj0022_-2.0",
             "e11103_2.0", "plb0022_h_5.0", "plb0022_h_9.0",
             "plb0035_nan", "plj0116_h_-2.0", "e11106_nan",
             "plb0041_-2.0", "hlf0073_-2.0", "plb0031_h_-2.0",
             "hlf0092_h_nan", "plb0103_-2.0", "plb0156_v1_nan"]
             
X_train_ks.drop(multicoll, axis = 1, inplace = True)

X_test_ks.drop(multicoll, axis = 1, inplace = True)

#After rerunning the above, can be seen that the condition
#number (ratio of largest to smallest eigenvalue of X_train_ks) has moved from 
#approaching infinity (with smallest eigenvalue 2.87 * 10**28)
#to a Condition Number 4 * 10**6, and that the
#warning about the design matrix being singular is no longer
#displayed.

#After re-running the VIF on X_train_ks without the 
#multicollinear (lines 262 - 274), further deleting:
    
#"i11110", "ijob1", "iself", "i11103", "idemy", "hhinc",
#"i11109", "i11104", "i11108", "ijob2", "i11107",
#"i11117", "iothy", "i13ly", "ixmas", "i11106", "iholy",
#"i14ly", "itray", "imilt", "plb0036_h", "plb0193_1.0",
#"numhh", "plb0193_2.0", "plb0193_3.0", "h11110", "ioldy",
#"h11112", "h11109", "h11105", "h11108", "h11106", "h11104",
#"h11107", "igrv1", "iciv1", "ple0164_nan", "iwidy", "h11103_1.0"

#lowers the Condition Number to 143.

#However, the obtained MSEs and R2 are reasonable numbers already
#simply deleting the variables suggested by R's lm() listed in multicoll,
#implying that by dropping those we do not have anymore perfect multicollinearity
#issues.

#Therefore, the remaining reasons for the high
#condition number are simply numerical.

#Nonetheless, for this reason, as specified in the paper, 
#the presented results for the SOEP Extended dataset are those
#coming from the Post-LASSO Extended set, where even these 
#final numerical issues are not there.

#Only exception is for the average effects of income and
#age, simply because the LASSO penalty deletes Age
#from the equation.


X_train_ks.drop(["pid"], axis = 1, inplace = True)

X_test_ks.drop(["pid"], axis = 1, inplace = True)

linreg_ks = linreg_train_test(X_train = X_train_ks, 
                              y_train = y_train_ks, 
                              X_test = X_test_ks, 
                              y_test = y_test_ks)



linreg_ks[0] 

#Test MSE = 2.23

linreg_ks[1] 

#Train MSE = 2.10

linreg_ks[2] 

#Test R2 = 0.28

linreg_ks[3]

#Train R2 = 0.31


#################
##RANDOM FOREST##
#################

#Ideas for hyperparameters optimization were based on EOSL (book).
#The many considered experiments are omitted for readability, and only
#the final values of the hyperparametric space are reported.

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

RF_ks[1]

RF_ks[3]

#Test MSE = 2.27

RF_ks[4]

#Train MSE = 0.30

RF_ks[5]

#Test R2 = 0.27

RF_ks[6]

#Train R2 = 0.90

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

GB_ks[3]

#Test MSE = 2.13

GB_ks[4]

#Train MSE = 0.08

GB_ks[5]

#Test R2 = 0.31

GB_ks[6]

#Train R2 = 0.97

#Measures of variables' importances are computed directly on the
#Post-LASSO Extended set of variables.
