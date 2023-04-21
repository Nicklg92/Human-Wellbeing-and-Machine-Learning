###############################################################################
###MISCELLANEA SCRIT - POOLING ALL YEARS FOR LIFE SATISFACTION'S DESCRIPTION###
###############################################################################

import pandas as pd
import numpy as np

'''
COMMENTS

This is a miscellanea script: we pool all the yearly datasets - 
and the train and test individuals - to pbserve the entire 
distribution of life satisfaction.
'''

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
    
###############################################
####POOLING BACK TRAIN AND TEST INDIVIDUALS####
###############################################
    
dset_2010 = pd.concat([yearly_dsets_train[0], yearly_dsets_test[0]], ignore_index = True)

dset_2011 = pd.concat([yearly_dsets_train[1], yearly_dsets_test[1]], ignore_index = True)

dset_2012 = pd.concat([yearly_dsets_train[2], yearly_dsets_test[2]], ignore_index = True)

dset_2013 = pd.concat([yearly_dsets_train[3], yearly_dsets_test[3]], ignore_index = True)

dset_2014 = pd.concat([yearly_dsets_train[4], yearly_dsets_test[4]], ignore_index = True)

dset_2015 = pd.concat([yearly_dsets_train[5], yearly_dsets_test[5]], ignore_index = True)

dset_2016 = pd.concat([yearly_dsets_train[6], yearly_dsets_test[6]], ignore_index = True)

dset_2017 = pd.concat([yearly_dsets_train[7], yearly_dsets_test[7]], ignore_index = True)

dset_2018 = pd.concat([yearly_dsets_train[8], yearly_dsets_test[8]], ignore_index = True)

dset = pd.concat([dset_2010, dset_2011, dset_2012, dset_2013, dset_2014,
                  dset_2015, dset_2016, dset_2017, dset_2018], ignore_index = True)
    
for i in [dset_2010, dset_2011, dset_2012, dset_2013, dset_2014,
                  dset_2015, dset_2016, dset_2017, dset_2018]:
    
    print(len(i))
    
ls_pooled = dset['Life Satisfaction']

#ls_pooled.to_csv('C:\\Some\\Local\\Path\\ls_all_years_SOEP.csv')