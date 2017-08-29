# -*- coding: utf-8 -*-
"""
@author: Alex Tselikov
"""
import warnings
warnings.filterwarnings('ignore')
from mers_utils import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn.metrics import r2_score

if __name__ == "__main__":
    t0 = time()
    ### parse args
    ## ex: python 3_xgb_level1.py 1 3 50 0.1 5 0.7 0.7 7 1 1 1 cleared_cont kmeans_10_cleared_cont ids
    parser = argparse.ArgumentParser(description='train xgb model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('alg_id', type=int, help='param_id')
    parser.add_argument('xgb_bags', type=int, help='xgb_bags')
    parser.add_argument('n_neighbors', type=int, help='n_neighbors')
    parser.add_argument('weights', type=str, help='weights')
    parser.add_argument('add_sum_zeros', type=int, help='add_sum_zeros')
    parser.add_argument('scaled', type=int, help='scaled')
    parser.add_argument('input', nargs='+', type=str, help='training data')    
    #parser.add_argument('output', type=str, help='predicted data')
    #parser.add_argument_group('input files to concat')
    #parser.add_argument('-i', 'input', help='Input file names', required=True)
    #parser.add_argument('--nargs', nargs='+')

    args = parser.parse_args()

    j = args.alg_id
    BAGS = args.xgb_bags
    
    y = np.loadtxt(FFOLD+'target.csv')
    ## READ
    print ('---load datas')    
    k=0
    for ifile in args.input:
        print (ifile)
        if ifile=='REM':
            break
        if k==0:
            #train = pd.read_csv(FFOLD+ifile+'.train.csv').values
            #test = pd.read_csv(FFOLD+ifile+'.test.csv').values
            train = np.loadtxt(FFOLD+ifile+'.train.csv')
            test =  np.loadtxt(FFOLD+ifile+'.test.csv')
            k+=1
        else:
            if ifile=='ids':
                train = np.hstack((train, np.expand_dims(np.loadtxt(FFOLD+ifile+'.train.csv'), axis=1)     ))
                test =  np.hstack((test,  np.expand_dims(np.loadtxt(FFOLD+ifile+'.test.csv'), axis=1)      ))
            else:
                train = np.hstack((train, np.loadtxt(FFOLD+ifile+'.train.csv')))
                test = np.hstack((test, np.loadtxt(FFOLD+ifile+'.test.csv')))


    #add sum zeros
    if args.add_sum_zeros>0:
        test = np.hstack((test,   np.expand_dims((test  == 0).astype(int).sum(axis=1), axis=1)))
        train = np.hstack((train, np.expand_dims((train == 0).astype(int).sum(axis=1), axis=1)))

    print (np.shape(train), np.shape(test))
    if args.scaled == 1:
        cls = StandardScaler()    
        train = cls.fit_transform(train)
        test = cls.transform(test)

    ## RUN
    print ('---run knn')    
    
    blend_train = np.zeros(len(y))
    blend_test = np.zeros(len(test))
    cv_results = np.zeros(len(SKFOLDS))
    blend_test_j = np.zeros((len(test), len(SKFOLDS)))
    R2_list=[]
    for i, (train_index, valid_index) in enumerate(SKFOLDS):
        
        dtrain = train[train_index]
        ytrain=y[train_index]
        dvalid = train[valid_index]
        #deval =  train[valid_index]
        yeval=y[valid_index]
    
    
        pred_bag = np.zeros(len(valid_index))
        pred_bag_test = np.zeros(len(test))
        
        #bagging
        for bag in range(BAGS):
            #paramz = xgb_params
            #paramz['random_state'] = (i+1)*1000+(bag+1)*100+j
            #paramz['seed'] = (i+1)*1000+(bag+1)*100+j
            clf = KNeighborsRegressor(n_neighbors=args.n_neighbors, weights = args.weights, n_jobs =-1, p=2)
            clf.fit(dtrain, ytrain)
            pred_bag += clf.predict(dvalid)
            pred_bag_test += clf.predict(test)
            if BAGS>1:
                print ('        Bag [%s] R2 = %0.5f' % (bag, r2_score(y[valid_index], pred_bag/(bag+1))) ,np.round((time() - t0) / 60. ,3),'mins' )
        
        pred_bag /= bag+1
        pred_bag_test /= bag+1
        
        fold_R2 = r2_score(y[valid_index], pred_bag)
        R2_list.append(fold_R2)
        
        blend_train[valid_index] = pred_bag
        cv_results[i] = fold_R2
        print ('    Fold [%s] R2 = %0.5f' % (i, fold_R2) ,np.round((time() - t0) / 60. ,3),'mins')
        
        blend_test_j[:, i] = pred_bag_test
        
    blend_test = blend_test_j.mean(1) #get mean for test by folds
    print ('!!!Clf_%d Mean R2 = %0.5f (%0.5f)' % (j, cv_results.mean(), cv_results.std()))
    #print (np.shape(blend_test), np.shape(blend_train))
    #print (type(blend_test), type(blend_train))

    ## SAVE
    np.savetxt(L1FOLD+str((100000*cv_results.mean()).astype(int))+'_'+'knn_1l_param'+str(j)+'.train.csv',blend_train)
    np.savetxt(L1FOLD+str((100000*cv_results.mean()).astype(int))+'_'+'knn_1l_param'+str(j)+'.test.csv',blend_test)

    print ('DONE in %s min' % np.round((time() - t0) / 60. ,3))