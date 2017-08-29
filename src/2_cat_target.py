# -*- coding: utf-8 -*-
"""
@author: Alex Tselikov
"""

from mers_utils import *
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import argparse

if __name__ == "__main__":
    t0 = time()
    parser = argparse.ArgumentParser(description='create pca features',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='training data')
    args = parser.parse_args()
    
    iinput=args.input



    ## READ
    train = pd.read_csv(FFOLD+iinput+'.train.csv')
    y = np.loadtxt(FFOLD+'target.csv')
    train = pd.concat([train, pd.Series(y, name='y')], axis=1)
    test = pd.read_csv(FFOLD+iinput+'.test.csv')
    df = pd.concat([train, test])

    ## PROCESS
    cat_cols=train.select_dtypes(include=['object']).columns

    for feat in cat_cols:
       m = train.groupby([feat])['y'].mean()
       train[feat].replace(m, inplace=True)
       test[feat].replace(m, inplace=True)
    
    #df = df.fillna(-1)
    ### SAVE
    np.savetxt(FFOLD+'2cat_target_'+iinput+'.train.csv',train[cat_cols].values, fmt='%10.5f')
    np.savetxt(FFOLD+'2cat_target_'+iinput+'.test.csv',test[cat_cols].values, fmt='%10.5f')
    
    print ('DONE in %s min' % np.round((time() - t0) / 60. ,3))
