# -*- coding: utf-8 -*-
"""
@author: Alex Tselikov
"""

from mers_utils import *
from sklearn.feature_extraction.text import TfidfTransformer
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
    #train = np.loadtxt(FFOLD+iinput+'.train.csv')
    #test =  np.loadtxt(FFOLD+iinput+'.test.csv')
    train = pd.read_csv(FFOLD+iinput+'.train.csv')
    test = pd.read_csv(FFOLD+iinput+'.test.csv')
    
    ## PROCESS
    cls = TfidfTransformer()
    tsneded = cls.fit_transform(np.vstack((train, test))).astype(float).toarray()  
    print (np.shape(tsneded))

    ## SAVE
    np.savetxt(FFOLD+'2tfidf_'+iinput+'.train.csv',tsneded[:len(train)])
    np.savetxt(FFOLD+'2tfidf_'+iinput+'.test.csv',tsneded[len(train):])
    print ('DONE in %s min' % np.round((time() - t0) / 60. ,3))
