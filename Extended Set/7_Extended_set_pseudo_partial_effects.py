#################################################################################
##SEVENTH SCRIPT - PSEUDO PARTIAL EFFECTS FOR LIFE SATISFACTION ON EXTENDED SET##
#################################################################################

import re 
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(1123581321)

'''
COMMENTS

This is the seventh script regarding the Extended Set 
producing the results in "Human Wellbeing and Machine Learning" by Ekaterina Oparina* (r) Caspar Kaiser* (r) Niccolò Gentile*; 
Alexandre Tkatchenko, Andrew E. Clark, Jan-Emmanuel De Neve 
and Conchita D'Ambrosio (* equal first authors in random order).

Aim of this script is to compute the pseudo partial effects
for life satisfaction.

Id est, we identify which variables are positively related with
life satisfaction and which aren't.
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

############################
###FITTING AND PREDICTING###
############################

#########
###OLS###
#########

linreg_plks = linreg_train_test(X_train = X_train_plks, 
                                y_train = y_train_plks, 
                                X_test = X_test_plks, 
                                y_test = y_test_plks)

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

##########################################
###COMPUTING THE PSEUDO PARTIAL EFFECTS###
##########################################

#######
##OLS##
#######

#Get the top 10 variables

varlist = pd.read_csv(path + 'PI_plks_r2_ols_df.csv', nrows = 11)

varlist = varlist['Variable'].tolist() #only keep the list of variables

#Check whether variable is binary or which ordinal/continous, then compute the difference

differences_list = []

for i in varlist:  
    
    #Save the data to be manipulated in a temporary file
            
    tmp = X_test_rks_stand.copy() 
    
    tmp_const = sm.add_constant(tmp, has_constant = 'add')

    if tmp_const[i].nunique() == 2: #Binary
        
        print('# ' + i + ' is binary')
              
        stub = re.split("_", i)[0]
      
        print(stub)
        
        #Get the full varlist

        fullvarlist = pd.read_csv(path + 'PI_plks_r2_ols_df.csv')
        
        fullvarlist = fullvarlist['Variable'].tolist() 
        
        #Get a varlist with all instances that need to be set to (standardized) 0

        tomanipulate = []
        
        for j in fullvarlist:
            
            #The second condition (j != i) in the following conditional
            #is to ensure that to the list tomanipulate we only 
            #add the dummies associated to some categorical 
            #nonordered, and not natural dummies (e.g. "plb0021").
            
            #If j == i, this means that i is a proper dummy.
            
            if (re.search(stub, j) != None) & (j != i) :
                
                tomanipulate.append(j)
                
                #In other words, in tomanipulate we have the comparison
                #categories to put to a standardized 0 of the respective categorical nonordinal.
                #Set all elements in tomanipulate to (standardized) 0
        
        for j in tomanipulate:
            
            tmp_const[j] = tmp_const[j].min()  #equivalent to 0 in the original data.                    
                
        del tomanipulate         
        
        #Find standardized 0 and 1 for variable i
        
        #i is either a natural dummy or the class of the
        #categorical nonordinal we are making the estimation over (chosen since
        #among the top 10 most important).
        
        low = tmp_const[i].min() #equivalent to 0 in the original data. 
        
        high = tmp_const[i].max() #equivalent to 1 in the original data. 
        
    else:  #Ordinal/continous      
        
        print('# ' + i + ' is ordinal/continous')
        
        low = tmp_const[i].describe().loc['25%'] # 25th pctile.
        
        high = tmp_const[i].describe().loc['75%'] # 75th pctile.
        
    #Find predicted ls with i set to low

    tmp_const[i] = low
    
    avg_yhat_low = linreg_plks[-1].predict(tmp_const).mean()
        
    tmp_const[i] = high
               
    avg_yhat_high = linreg_plks[-1].predict(tmp_const).mean()
    
    #Compute difference

    diff = avg_yhat_high - avg_yhat_low
         
    differences_list.append(diff)
        
    print(diff)
        
    del tmp_const
    
#Add result as a new column to the PI csv file
 
PI_OLS_plks_sorted = pd.read_csv(path + 'PI_plks_r2_ols_df.csv')

not_computed = ['not computed'] * (PI_OLS_plks_sorted.shape[0] - 11)  

differences_list.extend(not_computed)

PI_OLS_plks_sorted['Pseudo effect'] = differences_list

PI_OLS_plks_sorted.to_csv(path + 'PI_plks_r2_ols_df.csv', index = False)

#################
##RANDOM FOREST##
#################

varlist = pd.read_csv(path + 'PI_plks_r2_rf_df.csv', nrows = 11)

varlist = varlist['Variable'].tolist() #only keep the list of variables

differences_list = []

for i in varlist:  
    
    tmp = X_test_plks_stand.copy() 
      
    if tmp[i].nunique() == 2: #Binary
        
        print('# ' + i + ' is binary')
        
        stub = re.split("_", i)[0]
      
        print(stub)
                                
        fullvarlist = pd.read_csv(path + 'PI_plks_r2_rf_df.csv')
        
        fullvarlist = fullvarlist['Variable'].tolist() #only keep the list of variables
                
        tomanipulate = []
        
        for j in fullvarlist:
            
            if (re.search(stub, j) != None) & (j != i) :
                
                tomanipulate.append(j)
        
        for j in tomanipulate:
            
            tmp[j] = tmp[j].min()                      
                
        del tomanipulate         
        
        low = tmp[i].min() 
        
        high = tmp[i].max() 
          
    else:  #Ordinal/continous      
        
        print('# ' + i + ' is ordinal/continous')
        
        low = tmp[i].describe().loc['25%'] # 25th pctile.
        
        high = tmp[i].describe().loc['75%'] # 75th pctile.
    
    #Find predicted ls with i set to low
    
    tmp[i] = low
                
    avg_yhat_low = RF_plks[1].predict(tmp).mean()
        
    #Find predicted ls with i set to high
    
    tmp[i] = high
               
    avg_yhat_high = RF_plks[1].predict(tmp).mean()

    #Compute difference
    
    diff = avg_yhat_high - avg_yhat_low
         
    differences_list.append(diff)
        
    print(diff)
        
    del tmp
 
#Add result as a new column to the PI csv file

PI_RF_plks_sorted = pd.read_csv(path + 'PI_plks_r2_rf_df.csv')

not_computed = ['not computed'] * (PI_RF_plks_sorted.shape[0] - 11)  

differences_list.extend(not_computed)

PI_RF_plks_sorted['Pseudo effect'] = differences_list

PI_RF_plks_sorted.to_csv(path + 'PI_plks_r2_rf_df.csv', index = False)

#####################
##GRADIENT BOOSTING##
#####################

varlist = pd.read_csv(path + 'PI_plks_r2_gb_df.csv', nrows = 11)

varlist = varlist['Variable'].tolist() #only keep the list of variables

differences_list = []

for i in varlist:  
    
    tmp = X_test_rks_stand.copy() 
      
    if tmp[i].nunique() == 2: #Binary
        
        print('# ' + i + ' is binary')
              
        stub = re.split("_", i)[0]
      
        print(stub)
                
        fullvarlist = pd.read_csv(path + 'PI_plks_r2_gb_df.csv')
        
        fullvarlist = fullvarlist['Variable'].tolist() 
                
        tomanipulate = []
        
        for j in fullvarlist:
            
            if (re.search(stub, j) != None) & (j != i) :
                
                tomanipulate.append(j)
        
        for j in tomanipulate:
            
            tmp[j] = tmp[j].min()                     
                
        del tomanipulate         
        
        low = tmp[i].min() 
        
        high = tmp[i].max() 
        
    else:  #Ordinal/continous      
        
        print('# ' + i + ' is ordinal/continous')
        
        low = tmp[i].describe().loc['25%'] # 25th pctile.
        
        high = tmp[i].describe().loc['75%'] # 75th pctile.
      
    tmp[i] = low
    
    avg_yhat_low = GB_plks[1].predict(tmp).mean()
        
    tmp[i] = high
               
    avg_yhat_high = GB_plks[1].predict(tmp).mean()
    
    diff = avg_yhat_high - avg_yhat_low
         
    differences_list.append(diff)
        
    print(diff)
        
    del tmp
 
PI_GB_plks_sorted = pd.read_csv(path + 'PI_plks_r2_gb_df.csv')

not_computed = ['not computed'] * (PI_GB_plks_sorted.shape[0] - 11)  

differences_list.extend(not_computed)

PI_GB_plks_sorted['Pseudo effect'] = differences_list

PI_GB_plks_sorted.to_csv(path + 'PI_plks_r2_gb_df.csv', index = False)

