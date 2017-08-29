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

    cat_cols=train.select_dtypes(include=['object']).columns
    combi = list(combinations(cat_cols, args.ninteracts))

    for comb in combi:
        #print (comb)
        for i in range(0, len(comb)):
            if i==0:
                feat = comb[i]
            else:
                feat = feat + "_" + comb[i]
                
        for i in range(0, len(comb)):
            if i==0:
                train[feat] = train[comb[i]]
                test[feat] = test[comb[i]]
            else:
                train[feat] = train[feat] + train[comb[i]]
                test[feat] = test[feat] + test[comb[i]]
   
    #df = df.fillna(-1)
    ### SAVE
    train = train.drop(cat_cols,1)
    test = test.drop(cat_cols,1)

    train.to_csv(FFOLD+'1cat_interats'+str(args.ninteracts)+iinput+'.train.csv',index=False)
    test.to_csv(FFOLD+'1cat_interats'+str(args.ninteracts)+iinput+'.test.csv',index=False)

    print ('DONE in %s min' % np.round((time() - t0) / 60. ,3))
