# -*- coding: utf-8 -*-
"""
@author: Alex Tselikov
"""

from mers_utils import *
from sklearn.preprocessing import StandardScaler
import argparse

if __name__ == "__main__":
    t0 = time()
    parser = argparse.ArgumentParser(description='create pca features',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('nclusters', type=int, help='n_clusters')
    parser.add_argument('input', type=str, help='training data')
    #parser.add_argument('output', type=str, help='processed data')
    args = parser.parse_args()
    
    iinput=args.input
    #in_clusters=args.nclusters 

    ## READ
    train = pd.read_csv(FFOLD+iinput+'.train.csv')
    test = pd.read_csv(FFOLD+iinput+'.test.csv')
    
    ## PROCESS
    cls = StandardScaler()
    cls_train = cls.fit_transform(train)
    cls_test = cls.transform(test)

    ## SAVE
    np.savetxt(FFOLD+'2scale_'+iinput+'.train.csv',cls_train)
    np.savetxt(FFOLD+'2scale_'+iinput+'.test.csv',cls_test)
    print ('DONE in %s min' % np.round((time() - t0) / 60. ,3))
