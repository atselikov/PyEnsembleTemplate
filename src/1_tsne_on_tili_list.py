# -*- coding: utf-8 -*-
"""
@author: Alex Tselikov
"""

from mers_utils import *
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import argparse

if __name__ == "__main__":
    t0 = time()
    parser = argparse.ArgumentParser(description='create pca features',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('nclusters', type=int, help='n_clusters')
    parser.add_argument('flist', type=int, help='flist')
    parser.add_argument('input', type=str, help='training data')
    #parser.add_argument('output', type=str, help='processed data')
    args = parser.parse_args()
    
    iinput=args.input
    in_clusters=args.nclusters 

    ## READ
    train = pd.read_csv(FFOLD+iinput+'.train.csv')
    test = pd.read_csv(FFOLD+iinput+'.test.csv')

    if args.flist>1:
        features = ['X118','X127','X47','X315','X311','X179','X314','X232','X232','X136','X261']
    else:
        features = ['X118', 'X127', 'X47', 'X315', 'X311', 'X179', 'X314', 'X261']
    

    ## PROCESS
    cls = TSNE(n_components=in_clusters, random_state=RANDOM_STATE)
    tsneded = cls.fit_transform(np.vstack((train[features], test[features])))
    
    ## SAVE
    np.savetxt(FFOLD+'tsne_tili_'+str(args.flist)+'_'+str(in_clusters)+'_'+iinput+'.train.csv',tsneded[:len(train)])
    np.savetxt(FFOLD+'tsne_tili_'+str(args.flist)+'_'+str(in_clusters)+'_'+iinput+'.test.csv',tsneded[len(train):])
    print ('DONE in %s min' % np.round((time() - t0) / 60. ,3))
