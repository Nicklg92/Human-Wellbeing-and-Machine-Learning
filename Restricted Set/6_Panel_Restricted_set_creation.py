######################################################
###SIXTH SCRIPT - RESTRICTED PANEL DATASET CREATION###
######################################################

import pandas as pd
import numpy as np

'''
COMMENTS

This is the sixth script in the Restricted Set producing the results in 
"Human Wellbeing and Machine Learning" by Ekaterina Oparina* (r) Caspar Kaiser* (r) Niccolò Gentile*; 
Alexandre Tkatchenko, Andrew E. Clark, Jan-Emmanuel De Neve 
and Conchita D'Ambrosio (* equal first authors in random order).

Aim of this script is to start creating the Mundlak-corrected 
datasets, used in the Panel data analysis in Appendix A2.

In particular, here we stack the datasets and perform the
train-test split properly given the panel structure of the
data.

The only requirement to include an individual is that they have 
replied at least two times.
'''

years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]

path = 'C:\\Users\\Some\\Local\\Path\\'

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
    
############################
###BINDING BACK ALL YEARS###
############################    
    
train_set_all_years = pd.concat([yearly_dsets_train[0], yearly_dsets_train[1],
                                 yearly_dsets_train[2], yearly_dsets_train[3],
                                 yearly_dsets_train[4], yearly_dsets_train[5],
                                 yearly_dsets_train[6], yearly_dsets_train[7],
                                 yearly_dsets_train[8]], axis = 0, ignore_index = True)
    
#train_set_all_years.shape
#(204742, 67)
    
test_set_all_years = pd.concat([yearly_dsets_test[0], yearly_dsets_test[1],
                                 yearly_dsets_test[2], yearly_dsets_test[3],
                                 yearly_dsets_test[4], yearly_dsets_test[5],
                                 yearly_dsets_test[6], yearly_dsets_test[7],
                                 yearly_dsets_test[8]], axis = 0, ignore_index = True)
    
#test_set_all_years.shape
#(51189, 67)

#And finally all the data back together

all_years = pd.concat([train_set_all_years, test_set_all_years], axis = 0, ignore_index = True)

#all_years.shape
#(255931, 67)

#The only constraint is that a person has 
#replied at least two times.

at_least_two = np.sum(all_years.pid.value_counts() > 1)

#len(at_least_two)

#46392

#We subset considering only these individuals.

all_years['index_compilers_1'] = all_years.groupby(by = 'pid')['pid'].transform('count')

at_least_two_df = all_years[all_years['index_compilers_1'] > 1]

#We save the entire dataset as well.

at_least_two_df.to_csv('C:\\Some\\Local\\Path\\at_least_two_df.csv')

#########################
##TRAIN AND TEST SPLITS##
#########################

#First, let's manually sample 80% and 20% of the pids.

#Why? The training and test set split has to be done at the 
#individual level, not row level! The two coincide on cross-sectional
#data, but not on panel ones.

#For instance, consider an individual whose person identifier 
#pid is "A5", and has replied 3 times (t1, t2, and t3). If we 
#were to use the custom train-test split function in sklearn, we might have
#individual A5 indepvars values in t1 and t3 in the training set, and 
#her/his/their indepvar values in t2 in the test set, leading
#to potential leakeage!

#By instead doing it manually (creating the lists of pids 
#train_pids and test_pids as below), we ensure that if individual
#A5 is in the training set, so do all the values in the three periods, and vice 
#versa if is in the test set.

train_pids = list(pd.Series(at_least_two_df['pid'].unique()).sample(frac = 0.8, random_state = 1123581321))

test_pids = [x for x in list(at_least_two_df['pid'].unique()) if x not in train_pids]

at_least_two_df_train = at_least_two_df[at_least_two_df['pid'].isin(train_pids)]

at_least_two_df_test = at_least_two_df[at_least_two_df['pid'].isin(test_pids)]

at_least_two_df_train.to_csv('C:\\Users\\Some\\Local\\Path\\at_least_two_df_train.csv')

at_least_two_df_test.to_csv('C:\\Users\\Some\\Local\\Path\at_least_two_df_test.csv')

