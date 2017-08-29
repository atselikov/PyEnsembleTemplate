# -*- coding: utf-8 -*-
"""
@author: Alex Tselikov
"""

from mers_utils import *
import warnings
warnings.filterwarnings('ignore')


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




CorrKoef = df.corr()
CorField985 = []
print ('----- Correlations')
print ('TOP > .985')
for i in CorrKoef:
    for j in CorrKoef.index[CorrKoef[i] > 0.99]:
        if i != j and j not in CorField985 and i not in CorField985:
            CorField985.append(j)
            print ("%s-->%s: r^2=%f" % (i,j, CorrKoef[i][CorrKoef.index==j].values[0]))
print 
print ('CorField985', len(CorField985))

print 
CorSingleValue = []
for col in df.columns: 
    if df[col].nunique()<2:
        CorSingleValue.append(col)
        print (col, df[col].nunique())
print ('Constant featuries: %s' %len(CorSingleValue))


print ('%s columns total' % df.shape[1])
print ('drop CorField985 + CorSingleValue')
to_drop_cor = list(set().union(CorField985, CorSingleValue))
df = df.drop(to_drop_cor, 1)
print ('%s columns left' % df.shape[1])

print ('------ remove constant features from train')

train = train.drop(to_drop_cor, 1)
CorSingleValue = []
for col in train.columns: 
    if train[col].nunique()<2:
        CorSingleValue.append(col)
        print (col, train[col].nunique())
print ('Constant featuries: %s' %len(CorSingleValue))

print ('%s columns total' % df.shape[1])
print ('drop CorSingleValue')
df = df.drop(CorSingleValue, 1)
print ('%s columns left' % df.shape[1])

print ('------ clean not common categories')
print (df.drop(TARGET,1).isnull().sum().sum())
for column in list(df.select_dtypes(include=['object']).columns):
    if train[column].nunique() != test[column].nunique():
        set_train = set(train[column].unique())
        set_test = set(test[column].unique())
        remove_train = set_train - set_test
        remove_test = set_test - set_train
        remove = remove_train.union(remove_test)
        print (column, ' : ', len(remove_test), len(remove_train))
        print (column, ' : ', len(remove), remove)

        def filter_cat(x):
            if x in remove:
                return np.nan
            return x

        df[column] = df[column].apply(lambda x: filter_cat(x), 1)
print (df.drop(TARGET,1).isnull().sum().sum())
print ('------ saving')

cat_cols = df.select_dtypes(include=['object']).columns
cont_cols = df.select_dtypes(exclude=['object']).columns
cont_cols = list( set(cont_cols)-set(['ID'])-set([TARGET]) )

df[df[TARGET].notnull()][cont_cols].to_csv(FFOLD+'cleared_cont.train.csv',index=False)
df[df[TARGET].isnull()][cont_cols].to_csv(FFOLD+'cleared_cont.test.csv',index=False)

np.savetxt(FFOLD+'cleared_cont_arr.train.csv',df[df[TARGET].notnull()][cont_cols].values, fmt='%10.0f')
np.savetxt(FFOLD+'cleared_cont_arr.test.csv',df[df[TARGET].isnull()][cont_cols].values, fmt='%10.0f')

df[df[TARGET].notnull()][cat_cols].to_csv(FFOLD+'cleared_cats.train.csv',index=False)
df[df[TARGET].isnull()][cat_cols].to_csv(FFOLD+'cleared_cats.test.csv',index=False)

train[TARGET].to_csv(FFOLD+'target.csv',index=False)
train['ID'].to_csv(FFOLD+'ids.train.csv',index=False)
test['ID'].to_csv(FFOLD+'ids.test.csv',index=False)

print ('DONE in %s min' % np.round((time() - t0) / 60. ,3))