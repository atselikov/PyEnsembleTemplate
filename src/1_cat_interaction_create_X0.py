# -*- coding: utf-8 -*-
"""
@author: Alex Tselikov
"""

from mers_utils import *
from itertools import combinations
import argparse

if __name__ == "__main__":
    t0 = time()
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ninteracts', type=int, help='training data')
    parser.add_argument('input', type=str, help='training data')
    args = parser.parse_args()
    
    iinput=args.input
    
    ## READ
    train = pd.read_csv(FFOLD+iinput+'.train.csv')
    test = pd.read_csv(FFOLD+iinput+'.test.csv')
    #df = pd.concat([train, test])

    ## PROCESS

    cat_cols=train.drop('X0',1).columns

    for i in range(1,args.ninteracts+1):
        combi = list(combinations(cat_cols, i))
        for comb in combi:
                #print (comb)
                for i in range(0, len(comb)):
                    if i==0:
                        feat = 'X0'+"_"+comb[i]
                    else:
                        feat = feat + "_" + comb[i]
                        
                for i in range(0, len(comb)):
                    if i==0:
                        train[feat] = train['X0'] + train[comb[i]]
                        test[feat] = test['X0'] + test[comb[i]]
                    else:
                        train[feat] = train[feat] + train[comb[i]]
                        test[feat] = test[feat] + test[comb[i]]
                        
    ### SAVE
    train = train.drop(cat_cols,1)
    test = test.drop(cat_cols,1)
    train = train.drop('X0',1)
    test = test.drop('X0',1)

    train.to_csv(FFOLD+'1cat_X0_interats'+str(args.ninteracts)+iinput+'.train.csv',index=False)
    test.to_csv(FFOLD+'1cat_X0_interats'+str(args.ninteracts)+iinput+'.test.csv',index=False)

    print ('DONE in %s min' % np.round((time() - t0) / 60. ,3))
