# -*- coding: utf-8 -*-
"""
@author: Alex Tselikov
"""

from mers_utils import *


t0 = time()
train = pd.read_csv(INIT_FOLD + 'train_orig.csv')
print (np.shape(train))

print ('correct outlier:')
train.loc[train.y>260, 'y'] = 91

test = pd.read_csv(INIT_FOLD + 'test_orig.csv')
print (np.shape(test))
#train = train[train.y<200]
test[TARGET]=np.nan
df = pd.concat([train, test])
print (np.shape(df))

features = ['X236', 'X127', 'X267', 'X261', 'X383', 'X275', 'X311', 'X189', 'X328',
            'X104', 'X240', 'X152', 'X265', 'X276', 'X162', 'X238', 'X52', 'X117', 'X342',
            'X126', 'X316', 'X339', 'X312', 'X244', 'X77', 'X340', 'X115', 'X38', 'X341',
            'X206', 'X75', 'X203', 'X292', 'X65', 'X221', 'X151', 'X345', 'X198', 'X73',
            'X327', 'X113', 'X196', 'X310']

df[df[TARGET].notnull()][features].to_csv(FFOLD+'forumkn_cont.train.csv',index=False)
df[df[TARGET].isnull()][features].to_csv(FFOLD+'forumkn_cont.test.csv',index=False)

np.savetxt(FFOLD+'forumkn_cont_arr.train.csv',df[df[TARGET].notnull()][features].values, fmt='%10.0f')
np.savetxt(FFOLD+'forumkn_cont_arr.test.csv',df[df[TARGET].isnull()][features].values, fmt='%10.0f')

print ('DONE in %s min' % np.round((time() - t0) / 60. ,3))