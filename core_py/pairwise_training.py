from genericpath import isfile
import json
from warnings import catch_warnings
import matplotlib.pyplot as plt
import pprint
from random import sample
import os
import numpy as np
import random
from numpy.core.fromnumeric import sort
from numpy.core.shape_base import _block_check_depths_match
from numpy.lib.npyio import _save_dispatcher
from numpy.lib.polynomial import _roots_dispatcher, roots
from numpy.testing._private.utils import HAS_LAPACK64
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
import random
import lightgbm as lgb
import pickle

with open("../AnalyzeDV/continous_batch_0.json", 'r', encoding='UTF-8') as f:
    dvinfos = json.load(f)

def change_both_feature(is_range, feature, with_header=0):
        if is_range == True:
            range_bit = np.array([1])
            global_bit = np.array([0])
            if with_header == 0:
                global_feature = np.zeros(7)
            elif with_header == 1:
                global_feature = np.zeros(8)
            elif with_header == 2:
                global_feature = np.zeros(9)
                
            result = np.concatenate((range_bit, np.array(feature), global_bit, global_feature))
            # print('range feature:', len(list(feature)))
            # print('global feature:', len(list(global_feature)))
            # print('range:', len(list(result)))
        else:
            range_bit = np.array([0])
            global_bit = np.array([1])
            range_feature = np.zeros(8)
            result = np.concatenate((range_bit, range_feature, global_bit, np.array(feature)))
            # print('range feature:', len(list(range_feature)))
            # print('global feature:', len(list(feature)))
            # print('global:', len(list(result)))
        return list(result)

def pairwise_training(with_header, load_data=False, classifier=1,model=0):
    def train_one_batch(feature_dict_1, label_dict_1, feature_dict_2, label_dict_2, feature_dict_3, label_dict_3):
        X_train = []
        y_train = []

        for feature, label in [(feature_dict_1, label_dict_1), (feature_dict_2, label_dict_2), (feature_dict_3, label_dict_3)]:
            for dvid in feature.keys():
                for candid in feature[dvid]:
                    if dvid in label:
                        if candid in label[dvid]:
                            X_train.append(feature[dvid][candid])
                            y_train.append(label[dvid][candid])

        print(np.array(X_train).shape)
        print(np.array(y_train).shape)
        if model==0:
            if classifier == 1:
                clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
                    np.array(X_train), y_train)
            elif classifier == 0:
                clf = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
                    np.array(X_train), y_train)
        elif model==1:
            if classifier == 1:
                clf = XGBClassifier(objective = 'binary:logistic', n_estimators=100, random_state=0).fit(
                    np.array(X_train), y_train)
            elif classifier == 0:
                clf = XGBRegressor(objective = 'binary:logistic', n_estimators=100, random_state=0).fit(
                    np.array(X_train), y_train)
        elif model==2:
            params = {
                'task': 'train',
                'boosting_type': 'gbdt',  # 设置提升类型
                'metric': {'l2', 'auc'},  # 评估函数
                'num_leaves': 31,   # 叶子节点数
                'learning_rate': 0.05,  # 学习速率
                'feature_fraction': 0.9, # 建树的特征选择比例
                'bagging_fraction': 0.8, # 建树的样本采样比例
                'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
                'verbose': 1, # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
                }
            train_data = lgb.Dataset(data=X_train,label=y_train)
            if classifier==0:
                params['objective'] = 'regression'
                clf = lgb.train(params,train_data,num_boost_round=20)
            elif classifier == 1:
                params['objective'] = 'binary'
                clf = lgb.train(params,train_data,num_boost_round=20)

        return clf

    if with_header == 1:
        root_path = "with_header"
    elif with_header == 0:
        root_path = "without_header"
    elif with_header == 2:
        root_path = "with_header_sname"
    
    if load_data == False:
        range_feature_dict_1 = np.load(
            root_path+"/range_feature_dict_1.npy", allow_pickle=True).item()
        range_feature_dict_2 = np.load(
            root_path+"/range_feature_dict_2.npy", allow_pickle=True).item()
        range_feature_dict_3 = np.load(
            root_path+"/range_feature_dict_3.npy", allow_pickle=True).item()
        range_feature_dict_4 = np.load(
            root_path+"/range_feature_dict_4.npy", allow_pickle=True).item()

        global_feature_dict_1 = np.load(
            root_path+"/global_feature_dict_1.npy", allow_pickle=True).item()
        global_feature_dict_2 = np.load(
            root_path+"/global_feature_dict_2.npy", allow_pickle=True).item()
        global_feature_dict_3 = np.load(
            root_path+"/global_feature_dict_3.npy", allow_pickle=True).item()
        global_feature_dict_4 = np.load(
            root_path+"/global_feature_dict_4.npy", allow_pickle=True).item()

        train_range_label_dict_1 = np.load(
            "without_header/range_0_global_0/label/range_label_dict_range_0_global_0_1.npy", allow_pickle=True).item()
        train_range_label_dict_2 = np.load(
            "without_header/range_0_global_0/label/range_label_dict_range_0_global_0_2.npy", allow_pickle=True).item()
        train_range_label_dict_3 = np.load(
            "without_header/range_0_global_0/label/range_label_dict_range_0_global_0_3.npy", allow_pickle=True).item()
        train_range_label_dict_4 = np.load(
            "without_header/range_0_global_0/label/range_label_dict_range_0_global_0_4.npy", allow_pickle=True).item()

        train_global_label_dict_1 = np.load(
            "without_header/range_0_global_0/label/global_label_dict_range_0_global_0_1.npy", allow_pickle=True).item()
        train_global_label_dict_2 = np.load(
            "without_header/range_0_global_0/label/global_label_dict_range_0_global_0_2.npy", allow_pickle=True).item()
        train_global_label_dict_3 = np.load(
            "without_header/range_0_global_0/label/global_label_dict_range_0_global_0_3.npy", allow_pickle=True).item()
        train_global_label_dict_4 = np.load(
            "without_header/range_0_global_0/label/global_label_dict_range_0_global_0_4.npy", allow_pickle=True).item()

        pair_train_feature_dict_1 = {}
        pair_train_feature_dict_2 = {}
        pair_train_feature_dict_3 = {}
        pair_train_feature_dict_4 = {}

        pair_train_label_dict_1 = {}
        pair_train_label_dict_2 = {}
        pair_train_label_dict_3 = {}
        pair_train_label_dict_4 = {}

        train_findex_2_cindex_dict = {}

        feature_dict_1 = {}
        feature_dict_2 = {}
        feature_dict_3 = {}
        feature_dict_4 = {}
        label_dict_1 = {}
        label_dict_2 = {}
        label_dict_3 = {}
        label_dict_4 = {}

        train_feature_dict_list = [feature_dict_1,feature_dict_2,feature_dict_3,feature_dict_4]
        train_label_dict_list = [label_dict_1,label_dict_2,label_dict_3,label_dict_4]
        train_range_feature_dict_list = [range_feature_dict_1, range_feature_dict_2, range_feature_dict_3, range_feature_dict_4, global_feature_dict_1, global_feature_dict_2, global_feature_dict_3, global_feature_dict_4]
        train_range_label_dict_list = [train_range_label_dict_1, train_range_label_dict_2, train_range_label_dict_3, train_range_label_dict_4, train_global_label_dict_1, train_global_label_dict_2, train_global_label_dict_3, train_global_label_dict_4]
        pair_train_feature_dict_list = [pair_train_feature_dict_1, pair_train_feature_dict_2, pair_train_feature_dict_3,pair_train_feature_dict_4]
        pair_train_label_dict_list = [pair_train_label_dict_1,pair_train_label_dict_2, pair_train_label_dict_3, pair_train_label_dict_4]

        for index, label_dict in enumerate(train_label_dict_list):
            feature_dict = train_feature_dict_list[index]
            range_feature_dict = train_range_feature_dict_list[index]
            global_feature_dict = train_range_feature_dict_list[index+4]
            range_label_dict = train_range_label_dict_list[index]
            global_label_dict = train_range_label_dict_list[index+4]

            for dvid in list(set(range_feature_dict.keys()) | set(global_feature_dict.keys())):
                feature_dict[dvid] = {}
                label_dict[dvid] = {}
                if dvid in range_feature_dict and dvid in range_label_dict:
                    for cand_index in range_feature_dict[dvid]:
                        if cand_index in range_label_dict[dvid]:
                            new_feature = change_both_feature(True, range_feature_dict[dvid][cand_index], with_header)
                            feature_dict[dvid][cand_index] = new_feature
                            label_dict[dvid][cand_index] = range_label_dict[dvid][cand_index]
                if dvid in global_feature_dict and dvid in global_label_dict:
                    for cand_index in global_feature_dict[dvid]:
                        if cand_index in global_label_dict[dvid]:
                            new_feature = change_both_feature(False, global_feature_dict[dvid][cand_index], with_header)
                            feature_dict[dvid][cand_index] = new_feature
                            label_dict[dvid][cand_index] = global_label_dict[dvid][cand_index]
                    

        for index, train_label_dict in enumerate(train_label_dict_list):
            pair_train_feature_dict = pair_train_feature_dict_list[index]
            train_feature_dict = train_feature_dict_list[index]
            pair_train_label_dict = pair_train_label_dict_list[index]
            for dvid in train_label_dict:
                has_positive_label = False
                positive_cand_index = 0
                findex = 0
                for cand_index in train_label_dict[dvid]:
                    if train_label_dict[dvid][cand_index] == 1:
                        has_positive_label = True
                        positive_cand_index = cand_index
                        break
                # print(train_label_dict[dvid])
                if has_positive_label == False:
                    continue
                # print("has positive")
                pair_train_feature_dict[dvid] = {}
                train_findex_2_cindex_dict[dvid] = {}
                pair_train_label_dict[dvid] = {}
                # print(train_label_dict[dvid])
                for cand_index in train_label_dict[dvid]:
                    if cand_index == positive_cand_index:
                        continue
                    # print("train_feature_dict[dvid][positive_cand_IN]:", train_feature_dict[dvid][positive_cand_index])
                    # print("train_feature_dict[dvid][cand_index][char]:", train_feature_dict[dvid][cand_index])
                    # print("len(train_feature_dict[dvid][positive_cand_index])",len(train_feature_dict[dvid][positive_cand_index]))
                    # print("len(train_feature_dict[dvid][cand_index])",len(train_feature_dict[dvid][cand_index]))
                    positive_feature = [train_feature_dict[dvid][positive_cand_index][char] - train_feature_dict[dvid][cand_index][char] for char in range(0,len(train_feature_dict[dvid][positive_cand_index]))]
                    # print("positive_feature", positive_feature)
                    # positive_feature = train_feature_dict[dvid][positive_cand_index] - train_feature_dict[dvid][cand_index]
                    positive_label = 1
                    pair_train_feature_dict[dvid][findex] = positive_feature    
                    pair_train_label_dict[dvid][findex] = positive_label
                    train_findex_2_cindex_dict[dvid][findex] = (positive_cand_index, cand_index)
                
                    findex += 1
                    negative_feature = [train_feature_dict[dvid][cand_index][char] - train_feature_dict[dvid][positive_cand_index][char] for char in range(0,len(train_feature_dict[dvid][positive_cand_index]))]
                    # negative_feature = train_feature_dict[dvid][cand_index] - train_feature_dict[dvid][positive_cand_index]
                    negative_label = 0
                    pair_train_feature_dict[dvid][findex] = negative_feature
                    pair_train_label_dict[dvid][findex] = negative_label
                    train_findex_2_cindex_dict[dvid][findex] = (cand_index, positive_cand_index)
                    # print("dvid:", dvid, ", findex:", findex, ", pfeature:", positive_feature, ', nfeature:', negative_feature)
                    findex += 1

  
        np.save(root_path + "/" + "pair_train_feature_dict_1", pair_train_feature_dict_1)
        np.save(root_path + "/" + "pair_train_feature_dict_2", pair_train_feature_dict_2)
        np.save(root_path + "/" + "pair_train_feature_dict_3", pair_train_feature_dict_3)
        np.save(root_path + "/" + "pair_train_feature_dict_4", pair_train_feature_dict_4)
        np.save(root_path + "/" + "pair_train_label_dict_1", pair_train_label_dict_1)
        np.save(root_path + "/" + "pair_train_label_dict_2", pair_train_label_dict_2)
        np.save(root_path + "/" + "pair_train_label_dict_3", pair_train_label_dict_3)
        np.save(root_path + "/" + "pair_train_label_dict_4", pair_train_label_dict_4)

        np.save(root_path + "/" + "pair_train_findex_2_cindex_dict", train_findex_2_cindex_dict)

    else:
        pair_train_feature_dict_1 = np.load(root_path + "/" + "pair_train_feature_dict_1.npy")
        pair_train_feature_dict_2 = np.load(root_path + "/" + "pair_train_feature_dict_2.npy")
        pair_train_feature_dict_3 = np.load(root_path + "/" + "pair_train_feature_dict_3.npy")
        pair_train_feature_dict_4 = np.load(root_path + "/" + "pair_train_feature_dict_4.npy")
        pair_train_label_dict_1 = np.load(root_path + "/" + "pair_train_label_dict_1.npy")
        pair_train_label_dict_2 = np.load(root_path + "/" + "pair_train_label_dict_2.npy")
        pair_train_label_dict_3 = np.load(root_path + "/" + "pair_train_label_dict_3.npy")
        pair_train_label_dict_4 = np.load(root_path + "/" + "pair_train_label_dict_4.npy")

        train_findex_2_cindex_dict = np.load(root_path + "/" + "pair_train_findex_2_cindex_dict.npy")

    # 1,2,3
    clf1 = train_one_batch(pair_train_feature_dict_1, pair_train_label_dict_1, pair_train_feature_dict_2, pair_train_label_dict_2, pair_train_feature_dict_3, pair_train_label_dict_3)
    # 1,2,4
    clf2 = train_one_batch(pair_train_feature_dict_1, pair_train_label_dict_1, pair_train_feature_dict_2, pair_train_label_dict_2, pair_train_feature_dict_4, pair_train_label_dict_4)
    # 1,3,4
    clf3 = train_one_batch(pair_train_feature_dict_1, pair_train_label_dict_1, pair_train_feature_dict_3, pair_train_label_dict_3, pair_train_feature_dict_4, pair_train_label_dict_4)
    # 2,3,4
    clf4 = train_one_batch(pair_train_feature_dict_3, pair_train_label_dict_3, pair_train_feature_dict_2, pair_train_label_dict_2, pair_train_feature_dict_4, pair_train_label_dict_4)
    del pair_train_feature_dict_1,pair_train_feature_dict_2,pair_train_feature_dict_3,pair_train_feature_dict_4,pair_train_label_dict_1,pair_train_label_dict_2,pair_train_label_dict_3,pair_train_label_dict_4,feature_dict_1,feature_dict_2,feature_dict_3,feature_dict_4,range_feature_dict_1,range_feature_dict_2,range_feature_dict_3,range_feature_dict_4,global_feature_dict_1,global_feature_dict_2,global_feature_dict_3,global_feature_dict_4
    del train_range_label_dict_1,train_range_label_dict_2,train_range_label_dict_3,train_range_label_dict_4,train_global_label_dict_1,train_global_label_dict_2,train_global_label_dict_3,train_global_label_dict_4
    pickle.dump(clf1, open("xgb_clf1_r.pickle.dat", "wb"))
    pickle.dump(clf2, open("xgb_clf2_r.pickle.dat", "wb"))
    pickle.dump(clf3, open("xgb_clf3_r.pickle.dat", "wb"))
    pickle.dump(clf4, open("xgb_clf4_r.pickle.dat", "wb"))
    return clf1, clf2, clf3, clf4

def pairwise_test(clf1, clf2, clf3, clf4, with_header, evaluate_range, evaluate_global, batch_id, load_data=False, precision=1, classifier=1,model=0):
    
    best_cand_index_dict = {}
    best_cand_index_dict_record = {}
    def test_one_batch(clf, feature_dict, range_label_dict, global_label_dict, batch_id, precision=precision):
        
        sort_cand_dict = {}

        X_test = []
        test_id_dict_1 = {}
        test_id_dict = {}
        _id = 0
        # print(feature_dict)
        for dvid in feature_dict.keys():
            for findex in feature_dict[dvid]:
                if dvid not in test_id_dict_1:
                    test_id_dict_1[dvid] = {}
                test_id_dict_1[dvid][findex] = _id
                test_id_dict[_id] = [dvid, findex]
                _id += 1
                X_test.append(feature_dict[dvid][findex])
        # print(X_test)
        # np.save('with_header_sname/pairwise_range_1_global_1/x_test_4', X_test)
        # np.save('with_header_sname/pairwise_range_1_global_1/test_id_dict_1_4',test_id_dict_1)
        # np.save('with_header_sname/pairwise_range_1_global_1/test_id_dict_4',test_id_dict)

        y_pred = clf.predict(np.array(X_test))


        print("len y_pred", len(y_pred))
        for index, pred_res in enumerate(y_pred):
            pred_dvid, pred_findex = test_id_dict[index]
            pred_cand1, pred_cand2 = test_findex_2_cindex_dict[pred_dvid][pred_findex]
            print("dvid", pred_dvid, "cand1", pred_cand1, "cand2", pred_cand2, 'pred_res', pred_res)
            if classifier == 1:
                if pred_res > 0.5: # pred_cand1 better than pred_cand2
                    better_cand = pred_cand1
                    worse_cand = pred_cand2
                else:
                    better_cand = pred_cand2
                    worse_cand = pred_cand1
            elif classifier == 0:
                if pred_res > 0.5: # pred_cand1 better than pred_cand2
                    better_cand = pred_cand1
                    worse_cand = pred_cand2
                else:
                    better_cand = pred_cand2
                    worse_cand = pred_cand1
            if pred_dvid not in sort_cand_dict:
                sort_cand_dict[pred_dvid] = {}
            if better_cand not in sort_cand_dict[pred_dvid]:
                sort_cand_dict[pred_dvid][better_cand] = 0
            if worse_cand not in sort_cand_dict[pred_dvid]:
                sort_cand_dict[pred_dvid][worse_cand] = 0
            sort_cand_dict[pred_dvid][better_cand] += 1
            # if pred_dvid not in best_cand_index_dict:
            #     best_cand_index_dict[pred_dvid] = set()
            # else:
            #     if worse_cand in best_cand_index_dict[pred_dvid]:
            #         print('remove', worse_cand)
            #         best_cand_index_dict[pred_dvid].remove(worse_cand)
            # best_cand_index_dict[pred_dvid].add(better_cand)
            # print(pred_dvid, best_cand_index_dict[pred_dvid])
        # print(sort_cand_dict)
        for dvid in sort_cand_dict:
            sort_list = sorted(sort_cand_dict[dvid].items(), key=lambda x: x[1], reverse=True)
            best_cand_index_dict[dvid] = []
            best_cand_index_dict_record[dvid] = {}
            for index in range(0, min(precision, len(sort_list))):
                best_cand_index_dict[dvid].append(sort_list[index][0])
                best_cand_index_dict_record[dvid]["sort_list"] = sort_list
                if dvid in range_label_dict:
                    best_cand_index_dict_record[dvid]["range_label"] = range_label_dict[dvid]
                if dvid in global_label_dict:
                    best_cand_index_dict_record[dvid]["global_label"] = global_label_dict[dvid]

        hitnum = 0
        suc_dvid = []
        # for dvid in best_cand_index_dict:
        #     if len(list(best_cand_index_dict[dvid])) != 1:
        #         # print(best_cand_index_dict[dvid])
        #         if len(list(best_cand_index_dict[dvid])) > 1:
        #             print("best large than 1")
        #         elif len(list(best_cand_index_dict[dvid])) == 0:
        #             print("best is 0")
        #         continue
        #     best_cand_index = list(best_cand_index_dict[dvid])[0]
        #     if best_cand_index in range_label_dict:
        #         if range_label_dict[best_cand_index] == 1:
        #             suc_dvid.append(dvid)
        #             hitnum += 1
        #             continue
        #     if best_cand_index in global_label_dict:
        #         if global_label_dict[best_cand_index] == 1:
        #             suc_dvid.append(dvid)
        #             hitnum += 1
        #             continue
        range_true = 0
        # print(range_label_dict)
        for dvid in best_cand_index_dict:
            # if len(list(best_cand_index_dict[dvid])) != 1:
            #     if len(list(best_cand_index_dict[dvid])) > 1:
            #         print("best large than 1")
            #     elif len(list(best_cand_index_dict[dvid])) == 0:
            #         print("best is 0")
            #     continue
            # print('best_cand_index_dict[dvid]:', best_cand_index_dict[dvid])
            # is_in_label = False
            if dvid in range_label_dict:
                print('in range')
                for cad in range_label_dict[dvid]:
                    if range_label_dict[dvid][cad] == 1:
                        is_in_label = True
                        range_true += 1
                for best_cand_index in best_cand_index_dict[dvid]:
                        print('best_cand_index', best_cand_index)
                        print(range_label_dict[dvid])
                        if best_cand_index in range_label_dict[dvid]:
                            if classifier == 1:
                                if range_label_dict[dvid][best_cand_index] == 1:
                                    suc_dvid.append(dvid)
                                    hitnum += 1
                                    break
                            elif classifier==0:
                                if range_label_dict[dvid][best_cand_index] > 0.5:
                                    suc_dvid.append(dvid)
                                    hitnum += 1
                                    break
            if dvid in global_label_dict:
                for best_cand_index in best_cand_index_dict[dvid]: 
                    if best_cand_index in global_label_dict[dvid]:
                        # print("global:", global_label_dict[dvid])
                        # print('best_cand_index', best_cand_index)
                        if classifier == 1:
                            for cad in global_label_dict[dvid]:
                                if global_label_dict[dvid][cad] == 1:
                                    print("######################################################")
                                    print(global_label_dict[dvid])
                                    print(best_cand_index)
                                    print("global_label_dict[dvid][best_cand_index]", global_label_dict[dvid][best_cand_index])
                                    print("global_label_dict[dvid][best_cand_index] == 1", global_label_dict[dvid][best_cand_index] == 1)
                                    break
                            # print("global_label_dict[dvid][best_cand_index]", global_label_dict[dvid][best_cand_index])
                            if global_label_dict[dvid][best_cand_index] > 0.5:
                                print("global add")
                                suc_dvid.append(dvid)
                                hitnum += 1
                                break
                        elif classifier==0:
                            for cad in global_label_dict[dvid]:
                                if global_label_dict[dvid][cad] == 1:
                                    print(global_label_dict[dvid])
                                    print(best_cand_index)
                                    print("global_label_dict[dvid][best_cand_index]", global_label_dict[dvid][best_cand_index])
                                    break
                            if global_label_dict[dvid][best_cand_index] > 0.5:
                                suc_dvid.append(dvid)
                                hitnum += 1
                                break
        print('range_true', range_true)
        print(len(suc_dvid))
        return hitnum, suc_dvid

    if with_header == 1:
        root_path = "with_header"
    elif with_header == 0:
        root_path = "without_header"
    elif with_header == 2:
        root_path = "with_header_sname"
    
    label_path = "without_header" + "/" + "range_"+str(evaluate_range).replace("-","n").replace('-','n')+"_global_"+str(evaluate_global) + '/label'
    res_path = root_path + "/" + "pairwise_range_"+str(evaluate_range).replace("-","n").replace('-','n')+"_global_"+str(evaluate_global)
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    if load_data == False:
        # range_feature_dict_1 = np.load(
        #     root_path+"/range_feature_dict_1.npy", allow_pickle=True).item()
        # range_feature_dict_2 = np.load(
        #     root_path+"/range_feature_dict_2.npy", allow_pickle=True).item()
        # range_feature_dict_3 = np.load(
        #     root_path+"/range_feature_dict_3.npy", allow_pickle=True).item()
        # range_feature_dict_4 = np.load(
        #     root_path+"/range_feature_dict_4.npy", allow_pickle=True).item()

        # global_feature_dict_1 = np.load(
        #     root_path+"/global_feature_dict_1.npy", allow_pickle=True).item()
        # global_feature_dict_2 = np.load(
        #     root_path+"/global_feature_dict_2.npy", allow_pickle=True).item()
        # global_feature_dict_3 = np.load(
        #     root_path+"/global_feature_dict_3.npy", allow_pickle=True).item()
        # global_feature_dict_4 = np.load(
        #     root_path+"/global_feature_dict_4.npy", allow_pickle=True).item()
            
        test_feature_dict_1 = {}
        test_feature_dict_2 = {}
        test_feature_dict_3 = {}
        test_feature_dict_4 = {}

        pair_test_feature_dict_1 = {}
        pair_test_feature_dict_2 = {}
        pair_test_feature_dict_3 = {}
        pair_test_feature_dict_4 = {}
        test_findex_2_cindex_dict ={}
        # test_findex_2_cindex_dict = np.load(res_path+'/test_findex_2_cindex_dict.json', allow_pickle=True).item()

        # test_feature_dict_list = [test_feature_dict_1, test_feature_dict_2, test_feature_dict_3,test_feature_dict_4]
        # pair_test_feature_dict_list = [pair_test_feature_dict_1, pair_test_feature_dict_2, pair_test_feature_dict_3, pair_test_feature_dict_4]
        # test_range_feature_dict_list = [range_feature_dict_1, range_feature_dict_2, range_feature_dict_3, range_feature_dict_4, global_feature_dict_1, global_feature_dict_2, global_feature_dict_3, global_feature_dict_4]
        # print('start_get_feature.....')
        # for index, feature_dict in enumerate(test_feature_dict_list):
        #     range_feature_dict = test_range_feature_dict_list[index]
        #     global_feature_dict = test_range_feature_dict_list[index+4]
        #     for dvid in list(set(range_feature_dict.keys()) | set(global_feature_dict.keys())):
        #         if dvid in range_feature_dict:
        #             if dvid not in feature_dict:
        #                 feature_dict[dvid] = {}
        #             for cand_index in range_feature_dict[dvid]:
                        
        #                 new_feature = change_both_feature(True, range_feature_dict[dvid][cand_index], with_header)
        #                 feature_dict[dvid][cand_index] = new_feature
        #         if dvid in global_feature_dict:
        #             if dvid not in feature_dict:
        #                 feature_dict[dvid] = {}
        #             for cand_index in global_feature_dict[dvid]:
        #                 new_feature = change_both_feature(False, global_feature_dict[dvid][cand_index], with_header)
        #                 feature_dict[dvid][cand_index] = new_feature

        # np.save(root_path + "/" + "test_feature_dict_1", test_feature_dict_1)
        # np.save(root_path + "/" + "test_feature_dict_2", test_feature_dict_2)
        # np.save(root_path + "/" + "test_feature_dict_3", test_feature_dict_3)
        # np.save(root_path + "/" + "test_feature_dict_4", test_feature_dict_4)
        if batch_id==4:
            test_feature_dict_1 = np.load(root_path + "/" + "test_feature_dict_1.npy", allow_pickle=True).item()
            test_feature_dict = test_feature_dict_1
        elif batch_id==3:
            test_feature_dict_2 = np.load(root_path + "/" + "test_feature_dict_2.npy", allow_pickle=True).item()
            test_feature_dict = test_feature_dict_2
        elif batch_id==2:
            test_feature_dict_3 = np.load(root_path + "/" + "test_feature_dict_3.npy", allow_pickle=True).item()
            test_feature_dict = test_feature_dict_3
        else:
            test_feature_dict_4 = np.load(root_path + "/" + "test_feature_dict_4.npy", allow_pickle=True).item()
            test_feature_dict = test_feature_dict_4
        test_feature_dict_list = [test_feature_dict_1, test_feature_dict_2, test_feature_dict_3,test_feature_dict_4]
        pair_test_feature_dict_list = [pair_test_feature_dict_1, pair_test_feature_dict_2, pair_test_feature_dict_3, pair_test_feature_dict_4]
        print('start_get_pair_feature.....')
        # for index, test_feature_dict in enumerate(test_feature_dict_list):
        # pair_test_feature_dict = pair_test_feature_dict_list[3]

        batch = 0
        pair_test_feature_dict = {}
        add_dvid=[]
        for dvid in test_feature_dict:
            print(dvid)
            if dvid == 9632:
                print('continue')
                continue
            # print(dvid in add_dvid)
            # add_dvid.append(dvid)
            # print(len(pair_test_feature_dict))
            # if len(pair_test_feature_dict) == 500:
            #     # np.save(root_path + "/" + "pair_test_feature_dict_1_"+str(batch), pair_test_feature_dict_1)
            #     # b = json.dumps(test_findex_2_cindex_dict)
            #     # f2 = open(root_path + "/" + "pair_test_feature_dict_1_"+str(batch)+'.json', 'w')
            #     # f2.write(b)
            #     # f2.close()
            #     batch+=1
            #     # pair_test_feature_dict = {}
            findex = 0
            pair_test_feature_dict[dvid] = {}
            # if index <= 3:
            #     test_findex_2_cindex_dict[dvid] = {}
            # else:
            test_findex_2_cindex_dict[dvid] = {}
            for cand_index1 in test_feature_dict[dvid]:
                for cand_index2 in test_feature_dict[dvid]:
                    if cand_index1 >= cand_index2:
                        continue
                    feature_1_2 = [test_feature_dict[dvid][cand_index1][char] - test_feature_dict[dvid][cand_index2][char] for char in range(0,len(test_feature_dict[dvid][cand_index2]))]
                    # feature_1_2 = test_feature_dict[dvid][cand_index1] - test_feature_dict[dvid][cand_index2]
                    pair_test_feature_dict[dvid][findex] = feature_1_2
                    test_findex_2_cindex_dict[dvid][findex] = (cand_index1, cand_index2)
                    findex += 1

                    feature_2_1 = [test_feature_dict[dvid][cand_index2][char] - test_feature_dict[dvid][cand_index1][char] for char in range(0,len(test_feature_dict[dvid][cand_index2]))]
                    # feature_2_1 = test_feature_dict[dvid][cand_index2] - test_feature_dict[dvid][cand_index1]
                    
                    pair_test_feature_dict[dvid][findex] = feature_2_1
                    test_findex_2_cindex_dict[dvid][findex] = (cand_index2, cand_index1)
                    findex += 1

        
        # np.save(root_path + "/" + "pair_test_feature_dict_2", pair_test_feature_dict_2)
        # np.save(root_path + "/" + "pair_test_feature_dict_3", pair_test_feature_dict_3)
        # np.save(root_path + "/" + "pair_test_feature_dict_4", pair_test_feature_dict_4)

        # np.save(root_path + "/" + "pair_test_findex_2_cindex_dict", test_findex_2_cindex_dict)
        if batch_id==1:
            pair_test_feature_dict_4 = pair_test_feature_dict
        if batch_id==2:
            pair_test_feature_dict_3 = pair_test_feature_dict
        if batch_id==3:
            pair_test_feature_dict_2 = pair_test_feature_dict
        if batch_id==4:
            pair_test_feature_dict_1 = pair_test_feature_dict
        # b = json.dumps(test_findex_2_cindex_dict)
        # f2 = open(res_path+'/test_findex_2_cindex_dict.json', 'w')
        # f2.write(b)
        # f2.close()

    else:
        pair_test_feature_dict_1 = np.load(root_path + "/" + "pair_test_feature_dict_1.npy")
        pair_test_feature_dict_2 = np.load(root_path + "/" + "pair_test_feature_dict_2.npy")
        pair_test_feature_dict_3 = np.load(root_path + "/" + "pair_test_feature_dict_3.npy")
        pair_test_feature_dict_4 = np.load(root_path + "/" + "pair_test_feature_dict_4.npy")

        test_findex_2_cindex_dict = np.load(root_path + "/" + "pair_test_findex_2_cindex_dict.npy")

    if batch_id==1:
        test_range_label_dict_4 = np.load(
            label_path+"/range_label_dict_range_"+str(evaluate_range).replace("-","n")+"_global_"+str(evaluate_global)+"_4.npy", allow_pickle=True).item()
        test_global_label_dict_4 = np.load(
            label_path+"/global_label_dict_range_"+str(evaluate_range).replace("-","n")+"_global_"+str(evaluate_global)+"_4.npy", allow_pickle=True).item()
        hitnum_1, suc_dvid1 = test_one_batch(clf1, pair_test_feature_dict_4, test_range_label_dict_4, test_global_label_dict_4, batch_id=batch_id)
        b = json.dumps(suc_dvid1)
        f2 = open(res_path+'/suc_dvid1_'+str(precision) + "_"+str(classifier)+"_xgb.json", 'w')
        f2.write(b)
        f2.close()
    elif batch_id==2:
        test_range_label_dict_3 = np.load(
            label_path+"/range_label_dict_range_"+str(evaluate_range).replace("-","n")+"_global_"+str(evaluate_global)+"_3.npy", allow_pickle=True).item()
        test_global_label_dict_3 = np.load(
            label_path+"/global_label_dict_range_"+str(evaluate_range).replace("-","n")+"_global_"+str(evaluate_global)+"_3.npy", allow_pickle=True).item()
        hitnum_2, suc_dvid2 = test_one_batch(clf2, pair_test_feature_dict_3, test_range_label_dict_3, test_global_label_dict_3, batch_id=batch_id)
        b = json.dumps(suc_dvid2)
        f2 = open(res_path+'/suc_dvid2_'+str(precision) + "_"+str(classifier)+"_xgb.json", 'w')
        f2.write(b)
        f2.close()
    elif batch_id==3:
        test_range_label_dict_2 = np.load(
            label_path+"/range_label_dict_range_"+str(evaluate_range).replace("-","n")+"_global_"+str(evaluate_global)+"_2.npy", allow_pickle=True).item()
        test_global_label_dict_2 = np.load(
            label_path+"/global_label_dict_range_"+str(evaluate_range).replace("-","n")+"_global_"+str(evaluate_global)+"_2.npy", allow_pickle=True).item()
        hitnum_3, suc_dvid3 = test_one_batch(clf3, pair_test_feature_dict_2, test_range_label_dict_2, test_global_label_dict_2, batch_id=batch_id)
        b = json.dumps(suc_dvid3)
        f2 = open(res_path+'/suc_dvid3_'+str(precision) + "_"+str(classifier)+"_xgb.json", 'w')
        f2.write(b)
        f2.close()
    else:
        test_range_label_dict_1 = np.load(
            label_path+"/range_label_dict_range_"+str(evaluate_range).replace("-","n")+"_global_"+str(evaluate_global)+"_1.npy", allow_pickle=True).item()
        test_global_label_dict_1 = np.load(
            label_path+"/global_label_dict_range_"+str(evaluate_range).replace("-","n")+"_global_"+str(evaluate_global)+"_1.npy", allow_pickle=True).item()
        hitnum_4, suc_dvid4 = test_one_batch(clf4, pair_test_feature_dict_1, test_range_label_dict_1, test_global_label_dict_1, batch_id=batch_id)
        b = json.dumps(suc_dvid4)
        f2 = open(res_path+'/suc_dvid4_'+str(precision) + "_"+str(classifier)+"_xgb.json", 'w')
        f2.write(b)
        f2.close()

    b = json.dumps(best_cand_index_dict_record)
    f2 = open(res_path+'/best_cand_index_dict_2_'+str(precision)+'.json', 'w')
    f2.write(b)
    f2.close()
    # hitnum_3, suc_dvid3 = test_one_batch(clf3, pair_test_feature_dict_2, test_range_label_dict_2, test_global_label_dict_2)
    # hitnum_4, suc_dvid4 = test_one_batch(clf4, pair_test_feature_dict_1, test_range_label_dict_1, test_global_label_dict_1)

    # b = json.dumps(best_cand_index_dict)
    # f2 = open(res_path+'/best_cand_index_dict_'+str(precision)+'.json', 'w')
    # f2.write(b)
    # f2.close()

    print('len(suc_dvid1):', len(set(suc_dvid1)))
    print('len(suc_dvid2):', len(set(suc_dvid2)))
    print('len(suc_dvid3):', len(set(suc_dvid3)))
    print('len(suc_dvid4):', len(set(suc_dvid4)))
    # print(suc_dvid1)
    # print(suc_dvid2)
    # print(suc_dvid3)
    # print(suc_dvid4)
    print('len(pair_test_feature_dict_1):', len(pair_test_feature_dict_1))
    print('len(pair_test_feature_dict_2):', len(pair_test_feature_dict_2))
    print('len(pair_test_feature_dict_3):', len(pair_test_feature_dict_3))
    print('len(pair_test_feature_dict_4):', len(pair_test_feature_dict_4))

    print("len(list(set(pair_test_feature_dict_1) | set(pair_test_feature_dict_2) | set(pair_test_feature_dict_3) | set(pair_test_feature_dict_4)))",len(list(set(pair_test_feature_dict_1) | set(pair_test_feature_dict_2) | set(pair_test_feature_dict_3) | set(pair_test_feature_dict_4))))
    suc_dvid = list(set(suc_dvid1) | set(suc_dvid2) | set(suc_dvid3) | set(suc_dvid4))
    
    print('len(suc_dvid', len(suc_dvid))
    print("len inter suc dvid", len(list(set(suc_dvid1) & set(suc_dvid2))))
    print("len inter suc dvid", len(list(set(suc_dvid1) & set(suc_dvid3))))
    print("len inter suc dvid", len(list(set(suc_dvid1) & set(suc_dvid4))))
    print("len inter suc dvid", len(list(set(suc_dvid2) & set(suc_dvid3))))
    print("len inter suc dvid", len(list(set(suc_dvid2) & set(suc_dvid4))))
    print("len inter suc dvid", len(list(set(suc_dvid3) & set(suc_dvid4))))

    print("len all suc dvid", len(list(set(suc_dvid1) | set(suc_dvid2))))
    print("len all suc dvid", len(list(set(suc_dvid1) | set(suc_dvid3))))
    print("len all suc dvid", len(list(set(suc_dvid1) | set(suc_dvid4))))
    print("len all suc dvid", len(list(set(suc_dvid2) | set(suc_dvid3))))
    print("len all suc dvid", len(list(set(suc_dvid2) | set(suc_dvid4))))
    print("len all suc dvid", len(list(set(suc_dvid3) | set(suc_dvid4))))


    
    print('len all suc dvid', len(list(set(suc_dvid1) | set(suc_dvid2) | set(suc_dvid3) | set(suc_dvid4))))
    global_list = []
    range_list = []
    rank_range_suc = []
    rank_range_fail = []
    rank_global_suc = []
    rank_global_fail = []
    all_ = 0

    for dvid in list(set(pair_test_feature_dict_1) | set(pair_test_feature_dict_2) | set(pair_test_feature_dict_3) | set(pair_test_feature_dict_4)):
        # if dvid in search_fail:
        #     continue
        # try:
        #     with open("../PredictDV/evaluates1/delete_o_strip_relax_eval_1/"+str(dvid)+".json",'r', encoding='UTF-8') as f:
        #         one_result = json.load(f)
        # except:
        #     continue
        all_ += 1
        for k in dvinfos:
            if k["ID"] == dvid:

                if "," in k["Value"]:
                    global_list.append(dvid)
                    if dvid in suc_dvid1 or dvid in suc_dvid2 or dvid in suc_dvid3 or dvid in suc_dvid4:
                        rank_global_suc.append(dvid)
                    else:
                        rank_global_fail.append(dvid)
                else:
                    range_list.append(dvid)
                    if dvid in suc_dvid1 or dvid in suc_dvid2 or dvid in suc_dvid3 or dvid in suc_dvid4:
                        rank_range_suc.append(dvid)
                    else:
                        rank_range_fail.append(dvid)

    print('all_', all_)
    result = {}
    result["len_range_suc"] = len(rank_range_suc)
    result["len_range_fail"] = len(rank_range_fail)
    result["len_global_suc"] = len(rank_global_suc)
    result["len_global_fail"] = len(rank_global_fail)
    result["len_global_list"] = len(global_list)
    result["len_range_list"] = len(range_list)
    result["hit@"+str(precision)] = (len(rank_range_suc) + len(rank_global_suc))/(len(global_list) + len(range_list))
    result["range hit@"+str(precision)] = len(rank_range_suc) / len(range_list)
    result["global hit@"+str(precision)] = len(rank_global_suc) / len(global_list)

    print('len_range_suc', len(rank_range_suc))
    print('len_range_fail', len(rank_range_fail))
    print('len_global_suc', len(rank_global_suc))
    print('len_global_fail', len(rank_global_fail))
    print('len_global', len(global_list))
    print('len_range', len(range_list))
    

    print("acc_1:", (len(rank_range_suc) + len(rank_global_suc))/(len(global_list) + len(range_list)))
    print("rank_range_acc_1:", len(rank_range_suc) / len(range_list))
    print("rank_global_acc_1:", len(rank_global_suc) / len(global_list))
    print("merge_range_acc_1:", len(rank_range_suc) / len(range_list))
    print("merge_global_acc_1:", len(rank_global_suc) / len(global_list))

    if model == 0:
        b = json.dumps(result)
        f2 = open(res_path+'/evaluation_'+str(precision) + "_"+str(classifier)+"_xgbC.json", 'w')
        f2.write(b)
        f2.close()
    elif model == 1:
        b = json.dumps(result)
        f2 = open(res_path+'/evaluation_'+str(precision) +"_"+str(classifier)+ "_xgbC.json", 'w')
        f2.write(b)
        f2.close()
    elif model == 2:
        b = json.dumps(result)
        f2 = open(res_path+'/evaluation_'+str(precision) +"_"+str(classifier)+ "_xgbC.json", 'w')
        f2.write(b)
        f2.close()

def look_res(root_path="with_header_sname", evaluate_range=1, evaluate_global=1, precision=1, classifier=1,model=1):
    res_path = root_path + "/" + "pairwise_range_"+str(evaluate_range).replace("-","n").replace('-','n')+"_global_"+str(evaluate_global)
    
    with open(res_path+'/suc_dvid1_'+str(precision) + "_"+str(classifier)+"_xgb.json", 'r', encoding='UTF-8') as f:
        suc_dvid1 = json.load(f)

    with open(res_path+'/suc_dvid2_'+str(precision) + "_"+str(classifier)+"_xgb.json", 'r', encoding='UTF-8') as f:
        suc_dvid2 = json.load(f)

    with open(res_path+'/suc_dvid3_'+str(precision) + "_"+str(classifier)+"_xgb.json", 'r', encoding='UTF-8') as f:
        suc_dvid3 = json.load(f)

    with open(res_path+'/suc_dvid4_'+str(precision) + "_"+str(classifier)+"_xgb.json", 'r', encoding='UTF-8') as f:
        suc_dvid4 = json.load(f)

    suc_dvid = list(set(suc_dvid1) | set(suc_dvid2) | set(suc_dvid3) | set(suc_dvid4))
    
    print(suc_dvid)
    print('len(suc_dvid', len(suc_dvid))
    print("len inter suc dvid", len(list(set(suc_dvid1) & set(suc_dvid2))))
    print("len inter suc dvid", len(list(set(suc_dvid1) & set(suc_dvid3))))
    print("len inter suc dvid", len(list(set(suc_dvid1) & set(suc_dvid4))))
    print("len inter suc dvid", len(list(set(suc_dvid2) & set(suc_dvid3))))
    print("len inter suc dvid", len(list(set(suc_dvid2) & set(suc_dvid4))))
    print("len inter suc dvid", len(list(set(suc_dvid3) & set(suc_dvid4))))

    print("len all suc dvid", len(list(set(suc_dvid1) | set(suc_dvid2))))
    print("len all suc dvid", len(list(set(suc_dvid1) | set(suc_dvid3))))
    print("len all suc dvid", len(list(set(suc_dvid1) | set(suc_dvid4))))
    print("len all suc dvid", len(list(set(suc_dvid2) | set(suc_dvid3))))
    print("len all suc dvid", len(list(set(suc_dvid2) | set(suc_dvid4))))
    print("len all suc dvid", len(list(set(suc_dvid3) | set(suc_dvid4))))


    
    print('len all suc dvid', len(list(set(suc_dvid1) | set(suc_dvid2) | set(suc_dvid3) | set(suc_dvid4))))
    global_list = []
    range_list = []
    rank_range_suc = []
    rank_range_fail = []
    rank_global_suc = []
    rank_global_fail = []
    all_ = 0

    test_feature_dict_1 = np.load(root_path + "/" + "test_feature_dict_1.npy", allow_pickle=True).item()
    test_feature_dict_2 = np.load(root_path + "/" + "test_feature_dict_2.npy", allow_pickle=True).item()
    test_feature_dict_3 = np.load(root_path + "/" + "test_feature_dict_3.npy", allow_pickle=True).item()
    test_feature_dict_4 = np.load(root_path + "/" + "test_feature_dict_4.npy", allow_pickle=True).item()
    for dvid in list(set(test_feature_dict_1) | set(test_feature_dict_2) | set(test_feature_dict_3) | set(test_feature_dict_4)):
        # if dvid in search_fail:
        #     continue
        # try:
        #     with open("../PredictDV/evaluates1/delete_o_strip_relax_eval_1/"+str(dvid)+".json",'r', encoding='UTF-8') as f:
        #         one_result = json.load(f)
        # except:
        #     continue
        all_ += 1
        for k in dvinfos:
            if k["ID"] == dvid:

                if "," in k["Value"]:
                    global_list.append(dvid)
                    if dvid in suc_dvid1 or dvid in suc_dvid2 or dvid in suc_dvid3 or dvid in suc_dvid4:
                        rank_global_suc.append(dvid)
                    else:
                        rank_global_fail.append(dvid)
                else:
                    range_list.append(dvid)
                    if dvid in suc_dvid1 or dvid in suc_dvid2 or dvid in suc_dvid3 or dvid in suc_dvid4:
                        rank_range_suc.append(dvid)
                    else:
                        rank_range_fail.append(dvid)

    print('all_', all_)
    result = {}
    result["len_range_suc"] = len(rank_range_suc)
    result["len_range_fail"] = len(rank_range_fail)
    result["len_global_suc"] = len(rank_global_suc)
    result["len_global_fail"] = len(rank_global_fail)
    result["len_global_list"] = len(global_list)
    result["len_range_list"] = len(range_list)
    result["hit@"+str(precision)] = (len(rank_range_suc) + len(rank_global_suc))/(len(global_list) + len(range_list))
    result["range hit@"+str(precision)] = len(rank_range_suc) / len(range_list)
    result["global hit@"+str(precision)] = len(rank_global_suc) / len(global_list)

    print('len_range_suc', len(rank_range_suc))
    print('len_range_fail', len(rank_range_fail))
    print('len_global_suc', len(rank_global_suc))
    print('len_global_fail', len(rank_global_fail))
    print(rank_range_fail)
    print('len_global', len(global_list))
    print('len_range', len(range_list))
    

    print("acc_1:", (len(rank_range_suc) + len(rank_global_suc))/(len(global_list) + len(range_list)))
    print("rank_range_acc_1:", len(rank_range_suc) / len(range_list))
    print("rank_global_acc_1:", len(rank_global_suc) / len(global_list))
    print("merge_range_acc_1:", len(rank_range_suc) / len(range_list))
    print("merge_global_acc_1:", len(rank_global_suc) / len(global_list))

    if model == 0:
        b = json.dumps(result)
        f2 = open(res_path+'/evaluation_'+str(precision) + "_"+str(classifier)+"_xgbC.json", 'w')
        f2.write(b)
        f2.close()
    elif model == 1:
        b = json.dumps(result)
        f2 = open(res_path+'/evaluation_'+str(precision) +"_"+str(classifier)+ "_xgbC.json", 'w')
        f2.write(b)
        f2.close()
    elif model == 2:
        b = json.dumps(result)
        f2 = open(res_path+'/evaluation_'+str(precision) +"_"+str(classifier)+ "_xgbC.json", 'w')
        f2.write(b)
        f2.close()
   

def pairwise_run(with_header, classifier, model, evaluate_range, evaluate_global, precision, batch_id, need_train=True):
    look_res(evaluate_range=evaluate_range, evaluate_global=evaluate_global, precision=precision, classifier=classifier,model=model)
    # if need_train:
    #     clf1, clf2, clf3, clf4 = pairwise_training(with_header=with_header, classifier=classifier,model=model)
    # else:
    #     clf1 = pickle.load(open("xgb_clf1_r.pickle.dat", "rb"))
    #     clf2 = pickle.load(open("xgb_clf2_r.pickle.dat", "rb"))
    #     clf3 = pickle.load(open("xgb_clf3_r.pickle.dat", "rb"))
    #     clf4 = pickle.load(open("xgb_clf4_r.pickle.dat", "rb"))
    # pairwise_test(clf1, clf2, clf3, clf4,with_header=with_header, evaluate_range=evaluate_range, evaluate_global=evaluate_global, precision=precision,classifier=classifier,model=model, batch_id=batch_id)

if __name__ == "__main__":
    pairwise_run(with_header = 2, evaluate_range = 1, evaluate_global=1, precision=1, classifier = 1, model = 1, need_train=False, batch_id=3)
    # test_1 = np.load( "with_header_sname/pair_test_feature_dict_1_0.npy", allow_pickle=True).item()
    # test_2 = np.load( "with_header_sname/pair_test_feature_dict_1_1.npy", allow_pickle=True).item()

    # print(len(set(test_1.keys())&set(test_2.keys())))
    # # print(set(test_2.keys())-set(test_1.keys()))
    # # print(len(test_1.keys()))
    # # print(len(test_2.keys()))