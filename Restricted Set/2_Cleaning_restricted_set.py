#############################################
###SECOND SCRIPT - CLEANING RESTRICTED SET###
#############################################

#import os
import pandas as pd
import numpy as np
from random import randint
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 2000)

pd.options.display.float_format = '{:.4f}'.format

np.random.seed(123)

'''
COMMENTS:

This is the second script of the Restricted Set producing the results in 
"Machine Learning in the Prediction of Human
Wellbeing", joint first authorship by Oparina, E., Kaiser, C., 
and Gentile, N., et al.

In this one, in particular, the aim is to clean the Restricted set of variables 
in SOEP. Scripts for are Gallup and UKHLS are similar.

More in details, in what follows:
    
1) Missing values are analyzed and treated. The treatment depends
on the nature of the variable (dep. vs. indep., numerical vs. 
categorical).

2) Income is OECD equivalized, and Age squared is added at the 
regressors.
    
3) One-hot-encoding of the categorical variables and drop of the 
reference categories are performed.

4) Train-test splits are performed, and the resulting datasets
saved.

In this case, the performed operations are less straightforward
than in the previous script. Therefore, in order to make them
more intelligible also to non-Python users, they are heavily
commented.

You can skip these comments if you have good understanding 
of the language.
'''

path = 'C:\\Some\\Local\\Path\\'

dset = pd.read_csv(path + 'Name_of_temporary_dset_1.csv')

dset.drop(['Unnamed: 0'], axis = 1, inplace = True)

dset.rename(columns = {'ethnicity': 'Ethnicity'}, inplace = True)

##########################
##STEP 1: MISSING VALUES##
##########################

dset.shape

#260357 x 21

#As usual, we simply drop all those instances of 
#missing life satisfaction, our dependent variable.

#In these survey, missingness can be encoded in two ways:
    
#1) Negative value, if the variable can assume only positive ones
#(which is indeed the case for life satisfaction).

#2) Simply a Pythonic NaN. 

#We check for all the three kinds.

dset = dset[dset['Life Satisfaction'] >= 0]

dset.shape

#255931 x 21

#Are there missing values that are NaNs?

dset.isna().sum()

#dset.isnull().sum()

#No, all NaNs are as such negative values or different labels.

np.sum(dset < 0)

#First, let's see which variables are characters.

for i in list(dset):
    
    print([i, dset[i].head()])

#All of them are numbers.

#BMI we know is tracked only in even years, hence the high
#missingness is expected. Let's see the percentages:

for i in list(dset):
    
    print([i, str(round((np.sum(dset[i] < 0) / len(dset))*100, 4)) + '%'])
    
#The only other problematic are Religion (76.73%) and Number of doctor visits (12.05%).
#Let's investigate year by year.
    
for i in ['Number of doctor visits in previous year', 'BMI', 'Religion']:
    
    for j in list(dset['year'].unique()):
        
        dset_y = dset[dset['year'] == j]
        
        print([i, j, str(round((np.sum(dset_y[i] < 0) / len(dset_y))*100, 2)) + '%'])
        
        
#Number of doctor visits missing are overall evenly distributed, and
#BMI behaves as expected. Religion is almost always missing in even years,
#and apart from 2015 and 2011 (missigness at 0.55% and 24.11% respectively)
#in the other years is always over 84%.
        
#Imputation strategy:
        
#1) For all variables but 'BMI', and 'Religion':
        
        #1.1) Cross-sectional mean if numeric (including dummies, then rounded).
        #1.2) Cross-sectional mode if categorical.
        #1.3) Round the dummies.
        
#2) For BMI:
        
        #2.1) For each person, impute odd years with even ones.
        #2.2) Check how much missigness remaining, and behave accordingly.
        
#3) For Religion:
        
        #3.1) For each person, check if they ever replied the question. If yes, impute with it. 
        #3.2) Check how much missingness still there, and see accordingly.
        
#To simplify all the operations down the road, we simply transform
#each negative value in a Pythonic NaN.

dset[dset < 0] = np.nan

############################################
##IMPUTING EVERYTHING BUT BMI AND RELIGION##
############################################
     
#We use a list comprehension to keep track of all those variables 
#that are not BMI and Religion and have at least one NaN.

vars_ok_to_impute = [x for x in list(dset) if x not in ['BMI', 'Religion'] and dset[x].isna().sum() >= 1]
        
dset[vars_ok_to_impute].isna().sum()  

for i in vars_ok_to_impute:
    
    if i in ['Marital Status', 'Month of Interview', 'Housing ownership status']:
        
        #1.2) Using cross-sectional mode to impute if categorical.
        
        dset[i].fillna(dset[i].mode()[0], inplace = True)
        
    else: 
        
        #1.1) Using cross-sectional mean to impute if numeric (including dummies).
        
        dset[i].fillna(dset[i].mean(), inplace = True)

dset[vars_ok_to_impute].isna().sum()

for j in ['Age', 'Employment status dummies', 'Disability Status']:
    
    #1.3) Round the dummies (and Age, which can be only an integer).
    
    dset[j] = round(dset[j], 0)

dset.isna().sum()  

####################
####IMPUTING BMI####
####################

#In order to imput odd years with even ones, we need to interpolate
#using the "backward" direction.

#More precisely, the strategy is the following:

#A) Split the dataset in 4 subsets: 2011-2012, 2013-2014, 2015-2016, 2017-2018.

#B) In each of them 4, if the inidivudal (pid) is present in both years, 
#interpolate, and impute, else skip the person.
#For them, we'll use the resulting cross-sectional mean from A).

dset_1112 = dset[dset['year'].isin([2011, 2012])]

dset_1314 = dset[dset['year'].isin([2013, 2014])]

dset_1516 = dset[dset['year'].isin([2015, 2016])]

dset_1718 = dset[dset['year'].isin([2017, 2018])]

for i in [dset_1112, dset_1314, dset_1516, dset_1718]:
        
    i.sort_values(by = ['pid', 'year'], inplace = True)
    
    i.reset_index(drop = True, inplace = True)

#To review:
    
#dset_1112[['pid', 'year', 'Marital Status', 'Gender', 'Age', 'BMI']].head(20)

#dset_1314[['pid', 'year', 'Marital Status', 'Gender', 'Age', 'BMI']].head(20)
    
#For those pids available in one year only - the odd one - and as such missing BMI,
#the risk woud be that of imputing, using backward interpolation, someone else's BMI.

#If I were to use 'unbounded' backward imputation, I may end up imputing 
#someone else's BMI. The issue is solved by specifying that only one
#value back has to be imputed, via limit = 1.
    
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html

dset_1112['BMI'].interpolate(method = 'bfill', limit_direction = 'backward', limit = 1, inplace = True)
    
dset_1314['BMI'].interpolate(method = 'bfill', limit_direction = 'backward', limit = 1, inplace = True)    

#dset_1314[['pid', 'year', 'Marital Status', 'Gender', 'Age', 'BMI']].head(20)
    
#It indeed works as desired. 

dset_1516['BMI'].interpolate(method = 'bfill', limit_direction = 'backward', limit = 1, inplace = True)    

dset_1718['BMI'].interpolate(method = 'bfill', limit_direction = 'backward', limit = 1, inplace = True)    

#At this point, I can check how many missing still in each year:

for i in [dset_1112, dset_1314, dset_1516, dset_1718]:
    
    for j in list(dset['year'].unique()):
        
        if j in list(i['year'].unique()):
        
            dset_y = i[i['year'] == j]

            print([j, (dset_y['BMI'].isna().sum() / len(i))*100])

#At this point, I can split the datasets on year basis, and proceed with mean
#imputation. Let's take into account also 2010, as later I'll concatenate
#them all altogether.
            
dset_10 = dset[dset['year'] == 2010]
            
dset_11 = dset_1112[dset_1112['year'] == 2011]

dset_12 = dset_1112[dset_1112['year'] == 2012]

dset_13 = dset_1314[dset_1314['year'] == 2013]

dset_14 = dset_1314[dset_1314['year'] == 2014]

dset_15 = dset_1516[dset_1516['year'] == 2015]

dset_16 = dset_1516[dset_1516['year'] == 2016]

dset_17 = dset_1718[dset_1718['year'] == 2017]

dset_18 = dset_1718[dset_1718['year'] == 2018]

for i in [dset_10, dset_11, dset_12, dset_13, dset_14, dset_15,
          dset_16, dset_17, dset_18]:
    
    i['BMI'].fillna(i['BMI'].mean(), inplace = True)
    
#And now we can vertically concatenate all the datasets together.
    
dset_1 = pd.concat([dset_10, dset_11, dset_12, dset_13, dset_14, dset_15,
                    dset_16, dset_17, dset_18], ignore_index = True)
    
#Let's clean the environment before proceeding 
#with the imputation of religion.

del dset, i, j, dset_y, vars_ok_to_impute, dset_1112, dset_1314, dset_1516, dset_1718
del dset_10, dset_11, dset_12, dset_13, dset_14, dset_15, dset_16, dset_17, dset_18


#######################
###IMPUTING RELIGION###
#######################

#How to proceed?

#Solution: use fillna() grouping by pid!
 
#The values to impute will be obtained using individualdset.dropna().unique(), which
#should have one only values.

print(dset_1['Religion'].isna().sum())

dset_1['Religion'] = dset_1['Religion'].fillna(dset_1.groupby('pid')['Religion'].transform(lambda x: x.value_counts(dropna = False).index[0] if (x.value_counts(dropna = False).index[0] in [1,2,3,4,5,6,7,8,9,10] or len(x.value_counts(dropna = False)) == 1) else x.value_counts(dropna = False).index[1]))

#Some explanation of the above messy function:

#dset_1['Religion'].fillna(dset_1.groupby('pid')['Religion'].transform(...))

#We start by telling to Python that we want to fill the missing values in the 'Religion' column.

#How we want to do that? We are grouping obseravations by pid, that is we are working 
#on an individual level across the years.

#More in detail: for each of the subdatasets defined at an individual level (e.g., pid n appears
#only in three years, hence a three row dataset, an alike for the other people),
#we do the following:

#1) Impute the most frequently reported religion across the years by this person
#(using x.value_counts(dropna = False).index[0]) if it is indeed a possible 
#religion (using x.value_counts(dropna = False).index[0] in [1,2,3,4,5,6,7,8,9,10]).

#Hence, x.value_counts(dropna = False).index[0] is imputed.

#2) Leave the nan if this person has never said his/her religion,
#len(x.value_counts(dropna = False)) == 1, implying that value_counts() only has
#one value, nan.

#Hence, x.value_counts(dropna = False).index[0] is imputed.

#3) Otherwise, if the most frequently reported value is not a religion, that is nan (implying that
#x.value_counts(dropna = False).index[0] in [1,2,3,4,5,6,7,8,9,10] is False)
#but at some point the person has also said a proper religion (implying that
#len(x.value_counts(dropna = False)) > 1), impute using the second most frequent reply,
#indeed a religion.

#Hence, x.value_counts(dropna = False).index[1].

#Can now see how many more missing Religion has:

print(dset_1['Religion'].isna().sum())

#The individual imputation has reduced the missingness from
#(196376 / 255931) * 100 = 76/73%
#to
#(31330 / 255931) * 100 = 12.24%

#At this point mode imputation and that's it.

dset_1['Religion'].fillna(dset_1['Religion'].mode()[0], inplace = True)

print(dset_1['Religion'].isna().sum())

#0

for i in list(dset_1['Religion'].unique()):
    
    print(i, np.sum(dset_1['Religion'] == i), round(np.sum(dset_1['Religion'] == i) / len(dset_1), 2))

dset_1.isna().sum()

#All 0s, we can move over.

#############################################
###STEP 2: ADAPTATION OF INCOME AND AGE**2###
#############################################


dset_1['Adjusted Income'] = np.log(1 + dset_1['Household yearly disposable income'] / np.sqrt(dset_1['Number of people in the household']))

dset_1['Age^2'] = dset_1['Age'] ** 2

##################################################################
####STEP 3: CREATING THE YEARLY DATASETS AND TRAIN-TEST SPLITS####
##################################################################

#Here, the following:

#1) We save dset_1 as it is now.

#2) On top of 1), we do the train-test splits, and save them.

#3) On top of 1), we do the one-hot-encoding and dropping of most populous class, and save it.

#4) On top of 3), we do the train-test sets split, and save them. 



#############################
##1: SAVING DSET_1 AS IT IS##
#############################

dset_1.to_csv('C:\\Users\\niccolo.gentile\\Desktop\\Joined_paper\\New_protected_from_24112021\\Final_protected_no_ohe_with_missing_indicators.csv')

######################################################
##2: SAVING TRAIN - TEST SPLITS FROM DSET_1 AS IT IS##
######################################################

dest_path = 'C:\\Some\\local\\path\\'    

for i in list(dset_1['year'].unique()):
    
    X_i = dset_1[dset_1['year'] == i]
    
    y_i = X_i['Life Satisfaction']  
    
    X_i.drop(['Life Satisfaction'], axis = 1, inplace = True)
  
    seed = randint(0, 1000)    
    
    X_train, X_test, y_train, y_test = train_test_split(X_i, 
                                                        y_i,
                                                        test_size = 0.20,
                                                        random_state = seed) 
    
    train = pd.concat([y_train, X_train], axis = 1)
    
    test = pd.concat([y_test, X_test], axis = 1)
        
    save_train = dest_path + '\\train_noohe' + str(i) + '.csv'
    
    save_test = dest_path + '\\test_noohe' + str(i) + '.csv'
    
    train.to_csv(save_train, index = False)
    
    test.to_csv(save_test, index = False)

    
############################################################
###3: ONE-HOT-ENCODING AND DROP OF MOST POPULOUS ON DSET_1##
############################################################
    
#One - hot - encoding of Marital Status, State of Residence, Religion, Month of Interview,
#Housing ownership status, and Ethincity.

########################
###OHE MARITAL STATUS###
########################
   
dset_1['Marital Status'].unique()  

dset_1_ms = pd.get_dummies(dset_1['Marital Status'], prefix = 'MS')

dset_1_ms.reset_index(drop = True, inplace = True)

dset_1 = pd.concat([dset_1, dset_1_ms], axis = 1)

############################
###OHE STATE OF RESIDENCE###
############################

dset_1['State of Residence'].unique()  

dset_1_sor = pd.get_dummies(dset_1['State of Residence'], prefix = 'SoR')

dset_1_sor.reset_index(drop = True, inplace = True)

dset_1 = pd.concat([dset_1, dset_1_sor], axis = 1)    

##################
###OHE RELIGION###
##################
    
dset_1['Religion'].unique()  

dset_1_rel = pd.get_dummies(dset_1['Religion'], prefix = 'Rel')

dset_1_rel.reset_index(drop = True, inplace = True)

dset_1 = pd.concat([dset_1, dset_1_rel], axis = 1) 

############################
###OHE MONTH OF INTERVIEW###
############################
    
dset_1['Month of Interview'].unique()  

dset_1_moi = pd.get_dummies(dset_1['Month of Interview'], prefix = 'MoI')

dset_1_moi.reset_index(drop = True, inplace = True)

dset_1 = pd.concat([dset_1, dset_1_moi], axis = 1) 

##################################
###OHE HOUSING OWNERSHIP STATUS###
##################################
    
dset_1['Housing ownership status'].unique()  

dset_1_hos = pd.get_dummies(dset_1['Housing ownership status'], prefix = 'HoS')

dset_1_hos.reset_index(drop = True, inplace = True)

dset_1 = pd.concat([dset_1, dset_1_hos], axis = 1) 

###################
###OHE ETHNICITY###
###################
    
dset_1['Ethnicity'].unique()  

dset_1_eth = pd.get_dummies(dset_1['Ethnicity'], prefix = 'Eth')

dset_1_eth.reset_index(drop = True, inplace = True)

dset_1 = pd.concat([dset_1, dset_1_eth], axis = 1)

#########################################################
####DROPPING MOST POPULOUS CLASS (REFERENCE CATEGORY)####
#########################################################

#Most populous Marital Status
    
vars_ms = [x for x in list(dset_1) if 'MS_' in x]

dset_i_ms = dset_1[vars_ms]
    
dset_i_ms_sum = dset_i_ms.sum().sort_values(ascending = False)
    
to_drop_ms = dset_i_ms_sum.idxmax()
    
dset_1.drop([to_drop_ms], axis = 1, inplace = True)
    
#Most populous State of Resience

vars_sor = [x for x in list(dset_1) if 'SoR_' in x]
    
dset_i_sor = dset_1[vars_sor]
    
dset_i_sor_sum = dset_i_sor.sum().sort_values(ascending = False)
    
to_drop_sor = dset_i_sor_sum.idxmax()
    
dset_1.drop([to_drop_sor], axis = 1, inplace = True)
 
#Most populous Religion
    
vars_rel = [x for x in list(dset_1) if 'Rel_' in x]
    
dset_i_rel = dset_1[vars_rel]
    
dset_i_rel_sum = dset_i_rel.sum().sort_values(ascending = False)
    
to_drop_rel = dset_i_rel_sum.idxmax()
    
dset_1.drop([to_drop_rel], axis = 1, inplace = True)
    
#Most populous Month of Interview
    
vars_moi = [x for x in list(dset_1) if 'MoI_' in x]
    
dset_i_moi = dset_1[vars_moi]
    
dset_i_moi_sum = dset_i_moi.sum().sort_values(ascending = False)
    
to_drop_moi = dset_i_moi_sum.idxmax()
    
dset_1.drop([to_drop_moi], axis = 1, inplace = True)
    
#Most populous House ownership status
    
vars_hos = [x for x in list(dset_1) if 'HoS_' in x]
    
dset_i_hos = dset_1[vars_hos]
    
dset_i_hos_sum = dset_i_hos.sum().sort_values(ascending = False)
    
to_drop_hos = dset_i_hos_sum.idxmax()
    
dset_1.drop([to_drop_hos], axis = 1, inplace = True)
      
#Most populous Ethicity
    
vars_eth = [x for x in list(dset_1) if 'Eth_' in x]
    
dset_i_eth = dset_1[vars_eth]
    
dset_i_eth_sum = dset_i_eth.sum().sort_values(ascending = False)
    
to_drop_eth = dset_i_eth_sum.idxmax()
    
dset_1.drop([to_drop_eth], axis = 1, inplace = True)

#########################################################################
##4: SAVING TRAIN - TEST SPLITS FROM DSET_1 OHED AND NO MOST POP. CLASS##
#########################################################################
   
dest_path = 'C:\\Some\\Local\\Path\\'    

for i in list(dset_1['year'].unique()):
    
    X_i = dset_1[dset_1['year'] == i]
    
    y_i = X_i['Life Satisfaction']  
    
    X_i.drop(['Life Satisfaction'], axis = 1, inplace = True)
  
    seed = randint(0, 1000)    
    
    X_train, X_test, y_train, y_test = train_test_split(X_i, 
                                                        y_i,
                                                        test_size = 0.20,
                                                        random_state = seed) 
    
    train = pd.concat([y_train, X_train], axis = 1)
    
    test = pd.concat([y_test, X_test], axis = 1)
        
    save_train = dest_path + '\\train_ohed_nomostpop' + str(i) + '.csv'
    
    save_test = dest_path + '\\test_ohed_nomostpop' + str(i) + '.csv'
    
    train.to_csv(save_train, index = False)
    
    test.to_csv(save_test, index = False)


