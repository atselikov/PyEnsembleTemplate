# -*- coding: utf-8 -*-
"""
@author: Alex Tselikov
"""

from mers_utils import *
from sklearn.feature_extraction import DictVectorizer
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
    
    ## PROCESS

    train = train.T.reset_index(drop=True).to_dict().values()
    test = test.T.reset_index(drop=True).to_dict().values()

    vec = DictVectorizer(sparse=False)
    train = vec.fit_transform(train)
    test = vec.transform(test)

    ## SAVE
    np.savetxt(FFOLD+'2cat_dictvect_'+iinput+'.train.csv',train.values, fmt='%10.0f')
    np.savetxt(FFOLD+'2cat_dictvect_'+iinput+'.test.csv',test.values, fmt='%10.0f')
    print ('DONE in %s min' % np.round((time() - t0) / 60. ,3))
