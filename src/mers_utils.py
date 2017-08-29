from __future__ import division
import numpy as np
import pandas as pd
from time import time
from sklearn.cross_validation import KFold

RANDOM_STATE = 2017
NFOLDS = 5
INIT_FOLD='../data/'
FFOLD='features/'
L1FOLD='level1/'
L2FOLD='level2/'
TARGET='y'
train = pd.read_csv(INIT_FOLD + 'train_orig.csv')
SKFOLDS = KFold(len(train), n_folds=NFOLDS, shuffle=True, random_state=RANDOM_STATE)

def get_ctr_features(data, test, y, ctr_cols, dctr, num):
        data["target"] = y
        dcols = set(test.columns)
        kf = cross_validation.StratifiedKFold(y, n_folds=4, shuffle=True, random_state=11)
        tr = np.zeros((data.shape[0], len(ctr_cols)))
        for kfold, (itr, icv) in enumerate(kf):
            data_tr = data.iloc[itr]
            data_te = data.iloc[icv]
            for t, col in enumerate(ctr_cols):
                if col not in dcols:
                    continue
                ctr_df = data_tr[[col, "target"]].groupby(col).agg(["count", "sum"])
                ctr_dict = ctr_df.apply(lambda x: calc_ctr(x, num), axis=1).to_dict()
                tr[icv, t] = data_te[col].apply(lambda x: ctr_dict.get(x, dctr))

        te = np.zeros((test.shape[0], len(ctr_cols)))
        for t, col in enumerate(ctr_cols):
            if col not in dcols:
                    continue
            ctr_df = data[[col, "target"]].groupby(col).agg(["count", "sum"])
            ctr_dict = ctr_df.apply(lambda x: calc_ctr(x, num), axis=1).to_dict()
            te[:, t] = test[col].apply(lambda x: ctr_dict.get(x, dctr))
        del data["target"]
        return tr, te  

class CategoricalMeanEncoded(object):

    requirements = ['categorical']

    def __init__(self, C=100, loo=False, noisy=True, noise_std=None, random_state=11, combinations=[]):
        self.random_state = np.random.RandomState(random_state)
        self.C = C
        self.loo = loo
        self.noisy = noisy
        self.noise_std = noise_std
        self.combinations = [map(categoricals.index, comb) for comb in combinations]

    def fit_transform(self, ds):
        train_cat = ds['categorical']
        train_target = pd.Series(np.log(ds['loss'] + 100))
        train_res = np.zeros((train_cat.shape[0], len(categoricals) + len(self.combinations)), dtype=np.float32)

        self.global_target_mean = train_target.mean()
        self.global_target_std = train_target.std() if self.noise_std is None else self.noise_std

        self.target_sums = {}
        self.target_cnts = {}

        for col in xrange(len(categoricals)):
            train_res[:, col] = self.fit_transform_column(col, train_target, pd.Series(train_cat[:, col]))

        for idx, comb in enumerate(self.combinations):
            col = idx + len(categoricals)
            train_res[:, col] = self.fit_transform_column(col, train_target, pd.Series(map(''.join, train_cat[:, comb])))

        return train_res




    def transform(self, ds):
        test_cat = ds['categorical']
        test_res = np.zeros((test_cat.shape[0], len(categoricals) + len(self.combinations)), dtype=np.float32)

        for col in xrange(len(categoricals)):
            test_res[:, col] = self.transform_column(col, pd.Series(test_cat[:, col]))

        for idx, comb in enumerate(self.combinations):
            col = idx + len(categoricals)
            test_res[:, col] = self.transform_column(col, pd.Series(map(''.join, test_cat[:, comb])))

        return test_res

    def fit_transform_column(self, col, train_target, train_series):
        self.target_sums[col] = train_target.groupby(train_series).sum()
        self.target_cnts[col] = train_target.groupby(train_series).count()

        if self.noisy:
            train_res_reg = self.random_state.normal(
                loc=self.global_target_mean * self.C,
                scale=self.global_target_std * np.sqrt(self.C),
                size=len(train_series)
            )
        else:
            train_res_reg = self.global_target_mean * self.C

        train_res_num = train_series.map(self.target_sums[col]) + train_res_reg
        train_res_den = train_series.map(self.target_cnts[col]) + self.C

        if self.loo:  # Leave-one-out mode, exclude current observation
            train_res_num -= train_target
            train_res_den -= 1

        return np.exp(train_res_num / train_res_den).values

    def transform_column(self, col, test_series):
        test_res_num = test_series.map(self.target_sums[col]).fillna(0.0) + self.global_target_mean * self.C
        test_res_den = test_series.map(self.target_cnts[col]).fillna(0.0) + self.C

        return np.exp(test_res_num / test_res_den).values

    def get_feature_names(self):
        return categoricals + ['_'.join(categoricals[c] for c in comb) for comb in self.combinations]



def add_likelihood_feature(fname, train_likeli, test_likeli, flist):
    tt_likeli = pd.DataFrame()
    np.random.seed(1232345)
    skf = StratifiedKFold(train_likeli[TARGET].values, n_folds=5, shuffle=True, random_state=21387)
    for train_index, test_index in skf:
        ids = train_likeli['ID'].values[train_index]
        train_fold = train_likeli.loc[train_likeli['ID'].isin(ids)].copy()
        test_fold = train_likeli.loc[~train_likeli['ID'].isin(ids)].copy()
        global_avg = np.mean(train_fold[TARGET].values)
        feats_likeli = train_fold.groupby(fname)[TARGET].agg({'sum': np.sum, 'count': len}).reset_index()
        feats_likeli[fname + '_likeli'] = (feats_likeli['sum'] + 30.0*global_avg)/(feats_likeli['count']+30.0)
        test_fold = pd.merge(test_fold, feats_likeli[[fname, fname + '_likeli']], on=fname, how='left')
        test_fold[fname + '_likeli'] = test_fold[fname + '_likeli'].fillna(global_avg)
        tt_likeli = tt_likeli.append(test_fold[['ID', fname + '_likeli']], ignore_index=True)
    train_likeli = pd.merge(train_likeli, tt_likeli, on='ID', how='left')
    
    global_avg = np.mean(train_likeli[TARGET].values)
    feats_likeli = train_likeli.groupby(fname)[TARGET].agg({'sum': np.sum, 'count': len}).reset_index()
    feats_likeli[fname + '_likeli'] = (feats_likeli['sum'] + 30.0*global_avg)/(feats_likeli['count']+30.0)
    test_likeli = pd.merge(test_likeli, feats_likeli[[fname, fname + '_likeli']], on=fname, how='left')
    test_likeli[fname + '_likeli'] = test_likeli[fname + '_likeli'].fillna(global_avg)
    return train_likeli, test_likeli, flist + [fname + '_likeli']