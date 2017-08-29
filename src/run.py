python 0_cleaning1.py #cleared_cont cleared_cats
python 0_cleaning0.py #cont cats
python 0_get_top_cont.py #top cont

#cat interactions
python 1_cat_interaction_create.py 2 cleared_cats 		#1cat_interats2cleared_cats
python 1_cat_interaction_create.py 3 cats 				#1cat_interats3cats
python 1_cat_interaction_create.py 4 cleared_cats 		#1cat_interats4cleared_cats
python 1_cat_interaction_create_X0.py 2 cleared_cats 	#1cat_X0_interats2cleared_cats
python 1_cat_interaction_create_X0.py 3 cleared_cats 	#1cat_X0_interats3cleared_cats
python 1_cat_interaction_create_X0.py 4 cleared_cats 	#1cat_X0_interats4cleared_cats


#cat encode
python 2_cat_label.py cleared_cats #2cat_labeled_cleared_cats
python 2_cat_onehot.py cleared_cats #2cat_onehot_cleared_cats
python 2_cat_catEncoder.py 25 cleared_cats #2cat_catEncoded25_cleared_cats
python 2_cat_catEncoder.py 50 cleared_cats #2cat_catEncoded50_cleared_cats
python 2_cat_lexico.py cleared_cats 	   #2cat_lexico_cleared_cats
python 2_cat_label.py cats                 #2cat_labeled_cats
python 2_cat_onehot.py cats                #2cat_onehot_cats
python 2_cat_catEncoder.py 15 cats         #2cat_catEncoded15_cats
python 2_cat_catEncoder.py 30 cats         #2cat_catEncoded30_cats
python 2_cat_lexico.py cats                #2cat_lexico_cats
#cat encode target
#python 2_cat_target.py cats
#python 2_cat_target.py cleared_cats

#cat encode interactions
python 2_cat_label.py 1cat_interats2cleared_cats 		#2cat_labeled_1cat_interats2cleared_cats
python 2_cat_onehot.py 1cat_interats2cleared_cats 		#2cat_onehot_1cat_interats2cleared_cats
python 2_cat_catEncoder.py 25 1cat_interats3cats 		#2cat_catEncoded25_1cat_interats3cats
python 2_cat_catEncoder.py 50 1cat_interats4cleared_cats#2cat_catEncoded50_1cat_interats4cleared_cats
python 2_cat_lexico.py 1cat_interats2cleared_cats 	    #2cat_lexico_1cat_interats2cleared_cats
python 2_cat_label.py 1cat_X0_interats2cleared_cats                 #2cat_labeled_1cat_X0_interats2cleared_cats
python 2_cat_catEncoder.py 20 1cat_X0_interats3cleared_cats         #2cat_catEncoded20_1cat_X0_interats3cleared_cats
python 2_cat_catEncoder.py 40 1cat_X0_interats4cleared_cats         #2cat_catEncoded40_1cat_X0_interats4cleared_cats
python 2_cat_lexico.py 1cat_X0_interats2cleared_cats                #2cat_lexico_1cat_X0_interats2cleared_cats

#cat interacts encode target
#python 2_cat_target.py 1cat_interats2cleared_cats
#python 2_cat_target.py 1cat_interats3cats
#python 2_cat_target.py 1cat_interats4cleared_cats
#python 2_cat_target.py 1cat_X0_interats2cleared_cats
#python 2_cat_target.py 1cat_X0_interats3cleared_cats
#python 2_cat_target.py 1cat_X0_interats4cleared_cats

#dim red create
python 1_kmeans.py 5  cleared_cont             #kmeans_5_cleared_cont
python 1_kmeans.py 10 2cat_onehot_cleared_cats #kmeans_10_2cat_onehot_cleared_cats
python 1_kmeans.py 15 2cat_onehot_cats 		   #kmeans_15_2cat_onehot_cats
python 1_pca.py 5     2cat_onehot_cats         #pca_5_2cat_onehot_cats
python 1_pca.py 10    cleared_cont             #pca_10_cleared_cont
python 1_pca.py 15    2cat_onehot_cleared_cats #pca_15_2cat_onehot_cleared_cats
python 1_svd.py 5     cleared_cont             #svd_5_cleared_cont
python 1_svd.py 10    2cat_onehot_cleared_cats #svd_10_2cat_onehot_cleared_cats
python 1_svd.py 15    2cat_onehot_cats         #svd_15_2cat_onehot_cats
python 1_grp.py 5     2cat_onehot_cats 		   #grp_5_2cat_onehot_cats
python 1_grp.py 10    cleared_cont 			   #grp_10_cleared_cont
python 1_grp.py 15    2cat_onehot_cleared_cats #grp_10_2cat_onehot_cleared_cats
python 1_srp.py 5     cleared_cont 				#srp_5_cleared_cont
python 1_srp.py 10    2cat_onehot_cleared_cats #srp_10_2cat_onehot_cleared_cats
python 1_srp.py 15    2cat_onehot_cats  		#srp_10_2cat_onehot_cats
python 1_ica.py 5     2cat_onehot_cats 			#ica_5_2cat_onehot_cats
python 1_ica.py 10    cleared_cont 				#ica_10_cleared_cont
python 1_ica.py 15    2cat_onehot_cleared_cats  #ica_15_2cat_onehot_cleared_cats
python 1_tsne.py 2                cleared_cont  #tsne_2_cleared_cont
python 1_tsne_on_tili_list.py 2 1 cleared_cont  #tsne_tili_1_2_cleared_cont
python 1_tsne_on_tili_list.py 3 2 cleared_cont  #tsne_tili_3_2_cleared_cont

python 2_log1p.py cleared_cont #2log1p_cleared_cont
python 2_scale.py cleared_cont #2scale_cleared_cont
python 2_tfidf.py cleared_cont #2tfidf_cleared_cont

#XGB bag help
#id bag esr eta max_depth subsample colsample_bytree min_child_weight gamma alpha add_sum_zeros? datas (main_data first ...)
python 3_xgb_level1.py 01 2  50 0.11 5  0.7 0.5 4  1.0 1.0 1 cleared_cont_arr kmeans_10_cleared_cont ids #0.58437
python 3_xgb_level1.py 02 2  99 0.11 7  0.8 0.6 3  2.0 1.5 0 forumkn_cont_arr pca_15_2cat_onehot_cleared_cats ids #0.54926
python 3_xgb_level1.py 03 2  50 0.11 9  0.9 0.7 5  1.5 2.0 1 forumkn_cont_arr kmeans_15_2cat_onehot_cats tsne_2_cleared_cont ids #0.55800
python 3_xgb_level1.py 04 2 99 0.05 10 0.7 0.5 9  1.2 0.5 0 cleared_cont_arr kmeans_10_2cat_onehot_cleared_cats tsne_2_cleared_cont ids #0.56766
python 3_xgb_level1.py 05 2 50 0.05 12 0.8 0.6 8  0.6 2.0 1 2tfidf_cleared_cont tsne_tili_1_2_cleared_cont 2cat_labeled_1cat_interats2cleared_cats ids #0.56805
python 3_xgb_level1.py 06 2 99 0.05 15 0.9 0.7 6  0.9 1.2 0 forumkn_cont_arr tsne_tili_3_2_cleared_cont 2cat_lexico_1cat_X0_interats2cleared_cats ids #0.56053
python 3_xgb_level1.py 07 2 50 0.05 15 0.6 0.5 7  0.4 0.1 1 forumkn_cont_arr tsne_2_cleared_cont 2cat_catEncoded25_1cat_interats3cats ids #0.54510
python 3_xgb_level1.py 08 2 99 0.01 15 0.8 0.2 10 1.2 0.1 0 cleared_cont_arr tsne_tili_1_2_cleared_cont 2cat_catEncoded50_1cat_interats4cleared_cats ids #0.57533
python 3_xgb_level1.py 09 2 50 0.04 20 0.7 0.4 4  2.0 1.2 1 cleared_cont_arr svd_10_2cat_onehot_cleared_cats 2cat_lexico_1cat_interats2cleared_cats ids #0.56589
python 3_xgb_level1.py 10 2 99 0.02 25 0.9 0.3 15 1.5 1.5 0 forumkn_cont_arr ica_10_2cat_onehot_cleared_cats 2cat_labeled_1cat_X0_interats2cleared_cats ids #0.56283
python 3_xgb_level1.py 11 2 50 0.09 15 0.6 0.5 5 0.5 1.9 1 forumkn_cont_arr srp_5_cleared_cont 2cat_catEncoded20_1cat_X0_interats3cleared_cats ids #0.54486
python 3_xgb_level1.py 12 2 99 0.03 10 0.4 0.6 7 0.7 1.2 1 cleared_cont_arr svd_15_2cat_onehot_cats 2cat_catEncoded40_1cat_X0_interats4cleared_cats ids #0.56640
python 3_xgb_level1.py 13 2 50 0.05 20 0.8 0.2 10 0.3 0.8 1 cleared_cont_arr tsne_tili_3_2_cleared_cont kmeans_10_cleared_cont 2cat_catEncoded50_1cat_interats4cleared_cats ids #0.56502

#KNN bag not help
	#id bag nn weights add_sum_zeros? scaled? datas (main_data first ...)
python 3_knn_level1.py 1 1 48 uniform 0 cleared_cont_arr
python 3_knn_level1.py 2 1 80 uniform 0 0 pca_50_cleared_cont svd_10_2cat_onehot_cleared_cats #0.44844
python 3_knn_level1.py 3 1 70 distance 0 0 pca_50_cleared_cont ica_5_2cat_onehot_cats #0.43533
python 3_knn_level1.py 3 1 25 distance 0 0 svd_25_cleared_cont kmeans_15_2cat_onehot_cats #0.46789
python 3_knn_level1.py 3 1 70 uniform 0 0 svd_25_cleared_cont kmeans_15_2cat_onehot_cats #0.47038
python 3_knn_level1.py 3 1 50 uniform 0 0 ica_25_cleared_cont kmeans_15_2cat_onehot_cleared_cats #0.23990
python 3_knn_level1.py 3 1 50 uniform 0 0 srp_25_cleared_cont kmeans_15_2cat_onehot_cleared_cats #0.45287
python 3_knn_level1.py 3 1 50 uniform 0 0 srp_25_cleared_cont svd_15_2cat_onehot_cats #0.45682
python 3_knn_level1.py 3 1 70 uniform 0 0 forumkn_cont_arr svd_15_2cat_onehot_cats #0.56972

	#forum models
python 3_knn_level1_forum.py 1000 1 48 uniform 0 0 cleared_cont #0.58052
python 3_knn_level1_forum.py 1001 1 80 uniform 0 0 cleared_cont ica_10_2cat_onehot_cleared_cats #0.58253
python 3_knn_level1_forum.py 1002 1 75 uniform 0 0 cleared_cont kmeans_15_2cat_onehot_cats
python 3_knn_level1_forum.py 1003 1 75 uniform 0 0 2cat_onehot_cleared_cats svd_15_2cat_onehot_cats
python 3_knn_level1_forum.py 1004 1 25 uniform 0 1 cleared_cont 2cat_catEncoded50_cleared_cats #0.52172 

#GBR
   #id bag est max_depth max_features scaled? add_sum_zeros? datas (main_data first ...)
python 3_GBR_level1.py 1 3 300 5 0.6 1 1 forumkn_cont_arr kmeans_10_cleared_cont #0.55819 
python 3_GBR_level1.py 2 3 400 6 0.5 1 1 forumkn_cont_arr tsne_tili_1_2_cleared_cont 2cat_catEncoded40_1cat_X0_interats4cleared_cats ica_10_2cat_onehot_cleared_cats #0.53942
python 3_GBR_level1.py 3 3 300 15 0.8 1 1 forumkn_cont_arr 2cat_labeled_cleared_cats pca_5_2cat_onehot_cats ids
python 3_GBR_level1.py 4 3 400 10 0.9 1 1 forumkn_cont_arr 2cat_catEncoded25_cleared_cats grp_10_2cat_onehot_cleared_cats ids #
python 3_GBR_level1.py 5 3 500 7 0.6 1 1 forumkn_cont_arr 2cat_lexico_cats pca_10_cleared_cont ids #

#ADA not  help!
   #id bag est lr scaled? add_sum_zeros? datas (main_data first ...)
python 3_ADA_level1.py 0 3 300 0.005 0 1 forumkn_cont_arr kmeans_5_cleared_cont 2cat_catEncoded50_cleared_cats ids  # 0.58049
python 3_ADA_level1.py 1 1 300 0.001 0 1 forumkn_cont_arr kmeans_10_cleared_cont # 0.58778
python 3_ADA_level1.py 2 3 500 0.003 0 0 forumkn_cont_arr pca_10_cleared_cont 2cat_catEncoded30_cats ids #0.58299
python 3_ADA_level1.py 3 10 400 0.001 1 1 forumkn_cont_arr svd_5_cleared_cont 2cat_labeled_cats #0.56754

#extra bag not help
	#id bag est max_f scaled? add_sum_zeros? datas (main_data first ...)
python 3_Extra_level1.py 1 1 500 0.7 15 0 1 forumkn_cont_arr kmeans_15_2cat_onehot_cats 2cat_lexico_cats ids #0.55528
python 3_Extra_level1.py 2 1 300 0.6 25 0 1 forumkn_cont_arr grp_10_2cat_onehot_cleared_cats 2cat_catEncoded40_1cat_X0_interats4cleared_cats ids #0.50436
python 3_Extra_level1.py 3 1 400 0.8 30 0 1 forumkn_cont_arr tsne_tili_3_2_cleared_cont 2cat_labeled_cleared_cats ids # 0.52844
python 3_Extra_level1.py 4 1 500 0.9 10 1 1 pca_50_cleared_cont pca_15_2cat_onehot_cleared_cats 2cat_catEncoded25_cleared_cats ids #0.53808
python 3_Extra_level1.py 5 1 500 0.7 15 0 1 cleared_cont_arr ica_10_cleared_cont 2cat_lexico_cleared_cats ids #0.53235

#RF bag not help
	#id bag est max_f scaled? add_sum_zeros? datas (main_data first ...)
python 3_RF_level1.py 1 1 500 0.9 10 0 1 pca_50_cleared_cont pca_15_2cat_onehot_cleared_cats 2cat_catEncoded25_cleared_cats ids #0.53331
python 3_RF_level1.py 2 2 300 0.7 15 0 1 forumkn_cont_arr tsne_tili_3_2_cleared_cont kmeans_15_2cat_onehot_cats 2cat_lexico_cleared_cats ids #0.57331
python 3_RF_level1.py 3 3 400 0.8 20 0 1 forumkn_cont_arr tsne_tili_1_2_cleared_cont svd_15_2cat_onehot_cats 2cat_labeled_cats ids #0.56134

#LR
	#id scaled? add_sum_zeros? datas (main_data first ...)
python 3_LR_level1.py 2 0 1 cleared_cont_arr svd_15_2cat_onehot_cats #0.55270
python 3_LR_level1.py 3 1 1 pca_50_cleared_cont forumkn_cont_arr kmeans_15_2cat_onehot_cats grp_10_2cat_onehot_cleared_cats #0.57668
python 3_LR_level1.py 4 0 0 pca_50_cleared_cont forumkn_cont_arr pca_15_2cat_onehot_cleared_cats ica_10_2cat_onehot_cleared_cats svd_15_2cat_onehot_cats #0.58203
python 3_LR_level1.py 5 0 0 pca_50_cleared_cont forumkn_cont_arr tsne_tili_3_2_cleared_cont srp_5_cleared_cont kmeans_15_2cat_onehot_cats ica_5_2cat_onehot_cats #0.58835
python 3_LR_level1.py 6 0 0 pca_50_cleared_cont forumkn_cont_arr tsne_2_cleared_cont srp_10_2cat_onehot_cats 2cat_onehot_cats ica_15_2cat_onehot_cleared_cats #0.57406
