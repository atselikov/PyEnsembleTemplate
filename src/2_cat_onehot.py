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
    test = pd.read_csv(FFOLD+iinput+'.test.csv')
    df = pd.concat([train, test])

    ## PROCESS
    cat_cols=train.select_dtypes(include=['object']).columns
    df = df[cat_cols]
    df = pd.get_dummies(df)
    
    #df = df.fillna(-1)
    ### SAVE
    np.savetxt(FFOLD+'2cat_onehot_'+iinput+'.train.csv',df[:len(train)].values, fmt='%10.0f')
    np.savetxt(FFOLD+'2cat_onehot_'+iinput+'.test.csv',df[len(train):].values, fmt='%10.0f')
    
    print ('DONE in %s min' % np.round((time() - t0) / 60. ,3))
