from genericpath import isfile
import json
from warnings import catch_warnings
import matplotlib.pyplot as plt
import pprint
from random import sample
import os
import numpy as np
import random
from numpy.lib.npyio import _save_dispatcher
from numpy.lib.polynomial import _roots_dispatcher, roots
from numpy.testing._private.utils import HAS_LAPACK64
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pandas as pd
import random
from xgboost import DMatrix,train

range_search_faile = []
global_search_faile = []
with open("../AnalyzeDV/continous_batch_0.json", 'r', encoding='UTF-8') as f:
    dvinfos = json.load(f)
global_pred_result = {}
range_pred_result = {}
#         one_result = json.load(f)

def change_both_feature(is_range, feature, with_header=0):
        if is_range == True:
            range_bit = np.array([1])
            global_bit = np.array([0])
            if with_header == 0:
                global_feature = np.zeros(7)
            elif with_header == 1:
                global_feature = np.zeros(8)
            elif with_header == 2 or with_header==3:
                global_feature = np.zeros(9)
                
            result = np.concatenate((range_bit, np.array(feature), global_bit, global_feature))
            # print('range feature:', len(list(feature)))
            # print('global feature:', len(list(global_feature)))
            # print('range:', len(list(result)))
        else:
            range_bit = np.array([0])
            global_bit = np.array([1])
            if with_header == 3:
                range_feature = np.zeros(15)
            else:
                range_feature = np.zeros(5)
            result = np.concatenate((range_bit, range_feature, global_bit, np.array(feature)))
            # print('range feature:', len(list(range_feature)))
            # print('global feature:', len(list(feature)))
            # print('global:', len(list(result)))
        return list(result)

def devide_training(precision=1, start_index_1=2628, start_index_2=5256, start_index_3=7884, need_load_features=False, need_load_labels=False, evaluate_range=-1, evaluate_global=0, with_header=True, classifier=0, model=0):
    if with_header == 1:
        root_path = "with_header"
    elif with_header == 0:
        root_path = "without_header"
    elif with_header == 2:
        root_path = "with_header_sname"
    elif with_header == 3:
        root_path = "with_header_sname_style"
    res_path = root_path + "/" + "range_" + \
        str(evaluate_range).replace("-", "n").replace('-', 'n') + \
            "_global_"+str(evaluate_global)
    label_path = "without_header" + "/" + "range_"+str(evaluate_range).replace(
        "-", "n").replace('-', 'n')+"_global_"+str(evaluate_global) + '/label'
    # with open("../test-table-understanding/test-table-understanding/bin/debug/c#_both_features_1000_4.json", 'r', encoding='UTF-8') as f:
    #     features = json.load(f)
    # with open("../PredictDV/ListDV/c#_both_features_1000_3_1.json", 'r', encoding='UTF-8') as f:
    #     features = json.load(f)
    with open("../PredictDV/ListDV/c#_complete_both_features_10000_1.json", 'r', encoding='UTF-8') as f:
        features = json.load(f)
    # new_features = []
    # for feature in features:
    #     for feature1 in features1:
    #         if feature1['dvid'] == feature['dvid']:
    #             new_features.append(feature)
    #             break
    # features = new_features
    # if not os.path.exists(res_path):
    #     os.mkdir(res_path)
    # if not os.path.exists(label_path):
    #     os.mkdir(label_path)
    # with open("../test-table-understanding/test-table-understanding/bin/debug/c#_both_features_1000_3_01.json",'r', encoding='UTF-8') as f:
    #     features = json.load(f)

    if need_load_features == False:

        range_feature_dict_1 = {}
        range_feature_dict_2 = {}
        range_feature_dict_3 = {}
        range_feature_dict_4 = {}

        global_feature_dict_1 = {}
        global_feature_dict_2 = {}
        global_feature_dict_3 = {}
        global_feature_dict_4 = {}

        id_dict = {}

        # feature = {}
        # label = {}
        dvid_list = []
        start_1 = 0
        start_2 = 0
        start_3 = 0

        id_dict_1 = {}
        one_info = {}
        for index, i in enumerate(features):
            # print(len(features))
            if len(dvid_list) == start_index_1:
                start_1 = index
                print(start_1)
            if len(dvid_list) == start_index_2:
                start_2 = index
            if len(dvid_list) == start_index_3:
                start_3 = index

            if len(dvid_list) <= start_index_1 and index >= 0:
                if i["type"] == 1 or i["type"] == 0:
                    if i["dvid"] not in range_feature_dict_1:
                        range_feature_dict_1[i["dvid"]] = {}
                    if i["dvid"] not in id_dict_1:
                        id_dict_1[i["dvid"]] = {}
                    if i["cand_index"] not in range_feature_dict_1[i["dvid"]]:
                        range_feature_dict_1[i["dvid"]][i["cand_index"]] = []

                    id_dict[index] = [i["dvid"], i["cand_index"]]
                    id_dict_1[i["dvid"]][i["cand_index"]] = index
                    range_feature_dict_1[i["dvid"]
                                         ][i["cand_index"]].append(i["d_char"])
                    range_feature_dict_1[i["dvid"]
                                         ][i["cand_index"]].append(i["d_len"])
                    range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                        i["emptiness"])

                    range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                        i["distinctness"])
                    # range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                    #         i["type_ratio"])
                    # if(type(i["distinctness"]).__name__ != 'float'):
                    #     print(type(i["distinctness"]))
                    #     print(i["distinctness"])
                    range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                        i["completeness"])
                    if with_header==3:
                        range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                            i["leftness"])
                        range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                            i["closeness"])
                        range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                            i["column_num"])
                        range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                            i["orientation_type"])
                        range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                            i["location_type"])
                        range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                            i["fill_color"])
                        range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                            i["font_color"])
                        range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                            i["width"])
                        range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                            i["height"])
                        range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                            i["is_hidden"])
                        
                    # print('len(range_feaure', len(range_feature_dict_1[i["dvid"]][i["cand_index"]]))
                    # range_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                    #     i["local_pupolarity"])
             
                # # feature_dict_1[i["dvid"]][i["cand_index"]].append(i["type_ratio"])
                #         range_feature_dict_1[i["dvid"]][i["cand_index"]].append(i["popularity"])
                #         range_feature_dict_1[i["dvid"]][i["cand_index"]].append(i["type"])
                else:
                    if i["dvid"] not in global_feature_dict_1:
                        global_feature_dict_1[i["dvid"]] = {}
                    if i["dvid"] not in id_dict_1:
                        id_dict_1[i["dvid"]] = {}
                    if i["cand_index"] not in global_feature_dict_1[i["dvid"]]:
                        global_feature_dict_1[i["dvid"]][i["cand_index"]] = []

                    id_dict[index] = [i["dvid"], i["cand_index"]]
                    id_dict_1[i["dvid"]][i["cand_index"]] = index
                    global_feature_dict_1[i["dvid"]
                                          ][i["cand_index"]].append(i["d_char"])
                    global_feature_dict_1[i["dvid"]
                                          ][i["cand_index"]].append(i["d_len"])
                    global_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                        i["emptiness"])
                    global_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                        i["distinctness"])
                    global_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                        i["completeness"])
                    if with_header:
                        global_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                            i["header_jaccard"])
                    if with_header >= 2:
                        global_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                            i["sheet_ratio"])
                # feature_dict_1[i["dvid"]][i["cand_index"]].append(i["type_ratio"])

                    global_feature_dict_1[i["dvid"]][i["cand_index"]].append(
                        i["popularity"])
                    global_feature_dict_1[i["dvid"]
                                          ][i["cand_index"]].append(i["type"])

            elif len(dvid_list) <= start_index_2 and len(dvid_list) > start_index_1:
                if i["type"] == 1 or i["type"] == 0:
                    if i["dvid"] not in range_feature_dict_2:
                        range_feature_dict_2[i["dvid"]] = {}
                    if i["dvid"] not in id_dict_1:
                        id_dict_1[i["dvid"]] = {}
                    if i["cand_index"] not in range_feature_dict_2[i["dvid"]]:
                        range_feature_dict_2[i["dvid"]][i["cand_index"]] = []

                    id_dict[index] = [i["dvid"], i["cand_index"]]
                    id_dict_1[i["dvid"]][i["cand_index"]] = index
                    range_feature_dict_2[i["dvid"]
                                         ][i["cand_index"]].append(i["d_char"])
                    range_feature_dict_2[i["dvid"]
                                         ][i["cand_index"]].append(i["d_len"])
                    range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                        i["emptiness"])

                    range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                        i["distinctness"])

                    range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                        i["completeness"])
                    # range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                    #         i["type_ratio"])
                    if with_header == 3:
                        range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                            i["leftness"])
                        range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                            i["closeness"])
                        range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                            i["column_num"])
                        
                        range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                            i["orientation_type"])
                        range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                            i["location_type"])
                        range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                            i["fill_color"])
                        range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                            i["font_color"])
                        range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                            i["width"])
                        range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                            i["height"])
                        range_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                            i["is_hidden"])
                    # print('len(range_feaure', len(range_feature_dict_2[i["dvid"]][i["cand_index"]]))
                else:
                    if i["dvid"] not in global_feature_dict_2:
                        global_feature_dict_2[i["dvid"]] = {}
                    if i["dvid"] not in id_dict_1:
                        id_dict_1[i["dvid"]] = {}
                    if i["cand_index"] not in global_feature_dict_2[i["dvid"]]:
                        global_feature_dict_2[i["dvid"]][i["cand_index"]] = []

                    id_dict[index] = [i["dvid"], i["cand_index"]]
                    id_dict_1[i["dvid"]][i["cand_index"]] = index
                    global_feature_dict_2[i["dvid"]
                                          ][i["cand_index"]].append(i["d_char"])
                    global_feature_dict_2[i["dvid"]
                                          ][i["cand_index"]].append(i["d_len"])
                    global_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                        i["emptiness"])
                    global_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                        i["distinctness"])
                    global_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                        i["completeness"])
                    if with_header:
                        global_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                            i["header_jaccard"])
                    if with_header >= 2:
                        global_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                            i["sheet_ratio"])
                    global_feature_dict_2[i["dvid"]][i["cand_index"]].append(
                        i["popularity"])
                    global_feature_dict_2[i["dvid"]
                                          ][i["cand_index"]].append(i["type"])
            elif len(dvid_list) <= start_index_3 and len(dvid_list) > start_index_2:
                if i["type"] == 1 or i["type"] == 0:
                    if i["dvid"] not in range_feature_dict_3:
                        range_feature_dict_3[i["dvid"]] = {}
                    if i["dvid"] not in id_dict_1:
                        id_dict_1[i["dvid"]] = {}
                    if i["cand_index"] not in range_feature_dict_3[i["dvid"]]:
                        range_feature_dict_3[i["dvid"]][i["cand_index"]] = []

                    id_dict[index] = [i["dvid"], i["cand_index"]]
                    id_dict_1[i["dvid"]][i["cand_index"]] = index
                    range_feature_dict_3[i["dvid"]
                                         ][i["cand_index"]].append(i["d_char"])
                    range_feature_dict_3[i["dvid"]
                                         ][i["cand_index"]].append(i["d_len"])
                    range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                        i["emptiness"])

                    range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                        i["distinctness"])

                    range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                        i["completeness"])
                    # range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                    #         i["type_ratio"])
                    if with_header == 3:
                        range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                            i["leftness"])
                        range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                            i["closeness"])
                        range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                            i["column_num"])
                        
                        range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                            i["orientation_type"])
                        range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                            i["location_type"])
                        range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                            i["fill_color"])
                        range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                            i["font_color"])
                        range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                            i["width"])
                        range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                            i["height"])
                        range_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                            i["is_hidden"])
                    # print('len(range_feaure', len(range_feature_dict_3[i["dvid"]][i["cand_index"]]))
                else:
                    if i["dvid"] not in global_feature_dict_3:
                        global_feature_dict_3[i["dvid"]] = {}
                    if i["dvid"] not in id_dict_1:
                        id_dict_1[i["dvid"]] = {}
                    if i["cand_index"] not in global_feature_dict_3[i["dvid"]]:
                        global_feature_dict_3[i["dvid"]][i["cand_index"]] = []

                    id_dict[index] = [i["dvid"], i["cand_index"]]
                    id_dict_1[i["dvid"]][i["cand_index"]] = index
                    global_feature_dict_3[i["dvid"]
                                          ][i["cand_index"]].append(i["d_char"])
                    global_feature_dict_3[i["dvid"]
                                          ][i["cand_index"]].append(i["d_len"])
                    global_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                        i["emptiness"])
                    global_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                        i["distinctness"])
                    global_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                        i["completeness"])
                    if with_header:
                        global_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                            i["header_jaccard"])
                    if with_header >= 2:
                        global_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                            i["sheet_ratio"])
                    global_feature_dict_3[i["dvid"]][i["cand_index"]].append(
                        i["popularity"])
                    global_feature_dict_3[i["dvid"]
                                          ][i["cand_index"]].append(i["type"])
            elif len(dvid_list) > start_index_3:
                if i["type"] == 1 or i["type"] == 0:
                    if i["dvid"] not in range_feature_dict_4:
                        range_feature_dict_4[i["dvid"]] = {}
                    if i["dvid"] not in id_dict_1:
                        id_dict_1[i["dvid"]] = {}
                    if i["cand_index"] not in range_feature_dict_4[i["dvid"]]:
                        range_feature_dict_4[i["dvid"]][i["cand_index"]] = []

                    id_dict[index] = [i["dvid"], i["cand_index"]]

                    id_dict_1[i["dvid"]][i["cand_index"]] = index
                    range_feature_dict_4[i["dvid"]
                                         ][i["cand_index"]].append(i["d_char"])
                    range_feature_dict_4[i["dvid"]
                                         ][i["cand_index"]].append(i["d_len"])
                    range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                        i["emptiness"])

                    range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                        i["distinctness"])
                    # if(type(i["distinctness"]).__name__ != 'float'):
                    #     print(type(i["distinctness"]))
                    #     print(i["distinctness"])
                    range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                        i["completeness"])
                    # range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                    #         i["type_ratio"])
                    if with_header == 3:
                        range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                            i["leftness"])
                        range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                            i["closeness"])
                        range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                            i["column_num"])
                        
                        range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                            i["orientation_type"])
                        range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                            i["location_type"])
                        range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                            i["fill_color"])
                        range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                            i["font_color"])
                        range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                            i["width"])
                        range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                            i["height"])
                        range_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                            i["is_hidden"])
                    # print('len(range_feaure', len(range_feature_dict_4[i["dvid"]][i["cand_index"]]))
                # # feature_dict_1[i["dvid"]][i["cand_index"]].append(i["type_ratio"])
                #         range_feature_dict_1[i["dvid"]][i["cand_index"]].append(i["popularity"])
                #         range_feature_dict_1[i["dvid"]][i["cand_index"]].append(i["type"])
                else:
                    if i["dvid"] not in global_feature_dict_4:
                        global_feature_dict_4[i["dvid"]] = {}
                    if i["dvid"] not in id_dict_1:
                        id_dict_1[i["dvid"]] = {}
                    if i["cand_index"] not in global_feature_dict_4[i["dvid"]]:
                        global_feature_dict_4[i["dvid"]][i["cand_index"]] = []

                    id_dict[index] = [i["dvid"], i["cand_index"]]
                    id_dict_1[i["dvid"]][i["cand_index"]] = index
                    global_feature_dict_4[i["dvid"]
                                          ][i["cand_index"]].append(i["d_char"])
                    global_feature_dict_4[i["dvid"]
                                          ][i["cand_index"]].append(i["d_len"])
                    global_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                        i["emptiness"])
                    global_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                        i["distinctness"])
                    global_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                        i["completeness"])
                    if with_header:
                        global_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                            i["header_jaccard"])
                # feature_dict_1[i["dvid"]][i["cand_index"]].append(i["type_ratio"])
                    if with_header >= 2:
                        global_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                            i["sheet_ratio"])
                    global_feature_dict_4[i["dvid"]][i["cand_index"]].append(
                        i["popularity"])
                    global_feature_dict_4[i["dvid"]
                                          ][i["cand_index"]].append(i["type"])
            if i["dvid"] not in dvid_list:
                dvid_list.append(i["dvid"])

        print('len(dvid)', len(dvid_list))
        with open('range_feature_a_1_keys.json', 'w') as f:
            json.dump(list(range_feature_dict_1.keys()), f)
        with open('range_feature_a_2_keys.json', 'w') as f:
            json.dump(list(range_feature_dict_2.keys()), f)
        with open('range_feature_a_3_keys.json', 'w') as f:
            json.dump(list(range_feature_dict_3.keys()), f)
        with open('range_feature_a_4_keys.json', 'w') as f:
            json.dump(list(range_feature_dict_4.keys()), f)
        np.save(root_path+"/range_feature_dict_1",
                range_feature_dict_1)
        np.save(root_path+"/range_feature_dict_2",
                range_feature_dict_2)
        np.save(root_path+"/range_feature_dict_3",
                range_feature_dict_3)
        np.save(root_path+"/range_feature_dict_4",
                range_feature_dict_4)

        np.save(root_path+"/global_feature_dict_1",
                global_feature_dict_1)
        np.save(root_path+"/global_feature_dict_2",
                global_feature_dict_2)
        np.save(root_path+"/global_feature_dict_3",
                global_feature_dict_3)
        np.save(root_path+"/global_feature_dict_4",
                global_feature_dict_4)

        with open('range_feature_dict_1.json', 'w') as f:
            json.dump(range_feature_dict_1, f)
        with open('range_feature_dict_2.json', 'w') as f:
            json.dump(range_feature_dict_2, f)
        with open('range_feature_dict_3.json', 'w') as f:
            json.dump(range_feature_dict_3, f)
        with open('range_feature_dict_4.json', 'w') as f:
            json.dump(range_feature_dict_4, f)

        with open('global_feature_dict_1.json', 'w') as f:
            json.dump(global_feature_dict_1, f)
        with open('global_feature_dict_2.json', 'w') as f:
            json.dump(global_feature_dict_2, f)
        with open('global_feature_dict_3.json', 'w') as f:
            json.dump(global_feature_dict_3, f)
        with open('global_feature_dict_4.json', 'w') as f:
            json.dump(global_feature_dict_4, f)


        one_info['id_dict'] = id_dict
        one_info['id_dict_1'] = id_dict_1
        one_info['dvid_list'] = dvid_list
        one_info['start_1'] = start_1
        one_info['start_2'] = start_2
        one_info['start_3'] = start_3
        np.save(root_path+"/one_info", one_info)
    else:
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

        one_info = np.load(root_path+"/one_info.npy",
                           allow_pickle=True).item()
        id_dict = one_info['id_dict']

        dvid_list = one_info["dvid_list"]
        start_1 = one_info["start_1"]
        start_2 = one_info["start_2"]
        start_3 = one_info["start_3"]
        id_dict_1 = one_info["id_dict_1"]

    if not need_load_labels:
        try:
            if evaluate_range == -1:
                print("both_label_10000.json")
                with open("both_label_10000.json", 'r', encoding='UTF-8') as f:
                    labels = json.load(f)
            else:
                print('both_label_range_'+str(evaluate_range).replace("-",
                      "n") + "_global_" + str(evaluate_global)+'_10000.json')
                with open('both_label_range_'+str(evaluate_range).replace("-", "n") + "_global_" + str(evaluate_global)+'_10000.json', 'r', encoding='UTF-8') as f:
                    labels = json.load(f)
        except:
            print("no_such_label_file:")
            return
        print(len(labels))
        test_range_label_dict_1 = {}
        test_range_label_dict_2 = {}
        test_range_label_dict_3 = {}
        test_range_label_dict_4 = {}

        test_global_label_dict_1 = {}
        test_global_label_dict_2 = {}
        test_global_label_dict_3 = {}
        test_global_label_dict_4 = {}

        count = 0

        for index, i in enumerate(labels):
            if i["dvid"] in range_feature_dict_1 or i["dvid"] in global_feature_dict_1:
                if i["dvid"] in range_feature_dict_1:
                    if i['cand_index'] in range_feature_dict_1[i['dvid']]:
                        if i["dvid"] not in test_range_label_dict_1:
                            test_range_label_dict_1[i["dvid"]] = {}
                        test_range_label_dict_1[i["dvid"]
                            ][i["cand_index"]] = i["label"]
                if i["dvid"] in global_feature_dict_1:
                    if i['cand_index'] in global_feature_dict_1[i['dvid']]:
                        if i["dvid"] not in test_global_label_dict_1:
                            test_global_label_dict_1[i["dvid"]] = {}
                        test_global_label_dict_1[i["dvid"]
                            ][i["cand_index"]] = i["label"]
            elif i["dvid"] in range_feature_dict_2 or i["dvid"] in global_feature_dict_2:
                if i["dvid"] in range_feature_dict_2:
                    if i['cand_index'] in range_feature_dict_2[i['dvid']]:
                        if i["dvid"] == 31531 or i["dvid"] == '31531':
                            print(i, i["cand_index"])
                        if i["dvid"] not in test_range_label_dict_2:
                            test_range_label_dict_2[i["dvid"]] = {}
                        test_range_label_dict_2[i["dvid"]
                            ][i["cand_index"]] = i["label"]
                if i["dvid"] in global_feature_dict_2:
                    if i['cand_index'] in global_feature_dict_2[i['dvid']]:
                        if i["dvid"] not in test_global_label_dict_2:
                            test_global_label_dict_2[i["dvid"]] = {}
                        test_global_label_dict_2[i["dvid"]
                            ][i["cand_index"]] = i["label"]
            elif i["dvid"] in range_feature_dict_3 or i["dvid"] in global_feature_dict_3:
                if i["dvid"] in range_feature_dict_3:
                    if i['cand_index'] in range_feature_dict_3[i['dvid']]:
                        if i["dvid"] not in test_range_label_dict_3:
                            test_range_label_dict_3[i["dvid"]] = {}
                        test_range_label_dict_3[i["dvid"]
                            ][i["cand_index"]] = i["label"]
                if i["dvid"] in global_feature_dict_3:
                    if i['cand_index'] in global_feature_dict_3[i['dvid']]:
                        if i["dvid"] not in test_global_label_dict_3:
                            test_global_label_dict_3[i["dvid"]] = {}
                        test_global_label_dict_3[i["dvid"]
                            ][i["cand_index"]] = i["label"]
            elif i["dvid"] in range_feature_dict_4 or i["dvid"] in global_feature_dict_4:
                if i["dvid"] in range_feature_dict_4:
                    if i['cand_index'] in range_feature_dict_4[i['dvid']]:
                        if i["dvid"] not in test_range_label_dict_4:
                            test_range_label_dict_4[i["dvid"]] = {}
                        test_range_label_dict_4[i["dvid"]
                            ][i["cand_index"]] = i["label"]
                if i["dvid"] in global_feature_dict_4:
                    if i['cand_index'] in global_feature_dict_4[i['dvid']]:
                        if i["dvid"] not in test_global_label_dict_4:
                            test_global_label_dict_4[i["dvid"]] = {}
                        test_global_label_dict_4[i["dvid"]
                            ][i["cand_index"]] = i["label"]

        np.save(label_path+"/range_label_dict_range_"+str(evaluate_range).replace("-",
                "n")+"_global_"+str(evaluate_global)+"_1", test_range_label_dict_1)
        np.save(label_path+"/range_label_dict_range_"+str(evaluate_range).replace("-",
                "n")+"_global_"+str(evaluate_global)+"_2", test_range_label_dict_2)
        np.save(label_path+"/range_label_dict_range_"+str(evaluate_range).replace("-",
                "n")+"_global_"+str(evaluate_global)+"_3", test_range_label_dict_3)
        np.save(label_path+"/range_label_dict_range_"+str(evaluate_range).replace("-",
                "n")+"_global_"+str(evaluate_global)+"_4", test_range_label_dict_4)

        np.save(label_path+"/global_label_dict_range_"+str(evaluate_range).replace("-",
                "n")+"_global_"+str(evaluate_global)+"_1", test_global_label_dict_1)
        np.save(label_path+"/global_label_dict_range_"+str(evaluate_range).replace("-",
                "n")+"_global_"+str(evaluate_global)+"_2", test_global_label_dict_2)
        np.save(label_path+"/global_label_dict_range_"+str(evaluate_range).replace("-",
                "n")+"_global_"+str(evaluate_global)+"_3", test_global_label_dict_3)
        np.save(label_path+"/global_label_dict_range_"+str(evaluate_range).replace("-",
                "n")+"_global_"+str(evaluate_global)+"_4", test_global_label_dict_4)

        b = json.dumps(test_range_label_dict_1)
        f2 = open(res_path+'/range_label_dict_1.json', 'w')
        f2.write(b)
        f2.close()

        b = json.dumps(test_range_label_dict_2)
        f2 = open(res_path+'/range_label_dict_2.json', 'w')
        f2.write(b)
        f2.close()

        b = json.dumps(test_range_label_dict_3)
        f2 = open(res_path+'/range_label_dict_3.json', 'w')
        f2.write(b)
        f2.close()

        b = json.dumps(test_range_label_dict_4)
        f2 = open(res_path+'/range_label_dict_4.json', 'w')
        f2.write(b)
        f2.close()

        b = json.dumps(test_global_label_dict_1)
        f2 = open(res_path+'/global_label_dict_1.json', 'w')
        f2.write(b)
        f2.close()

        b = json.dumps(test_global_label_dict_2)
        f2 = open(res_path+'/global_label_dict_2.json', 'w')
        f2.write(b)
        f2.close()

        b = json.dumps(test_global_label_dict_3)
        f2 = open(res_path+'/global_label_dict_3.json', 'w')
        f2.write(b)
        f2.close()

        b = json.dumps(test_global_label_dict_4)
        f2 = open(res_path+'/global_label_dict_4.json', 'w')
        f2.write(b)
        f2.close()

    else:
        test_range_label_dict_1 = np.load(
            label_path+"/range_label_dict_range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"_1.npy", allow_pickle=True).item()
        test_range_label_dict_2 = np.load(
            label_path+"/range_label_dict_range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"_2.npy", allow_pickle=True).item()
        test_range_label_dict_3 = np.load(
            label_path+"/range_label_dict_range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"_3.npy", allow_pickle=True).item()
        test_range_label_dict_4 = np.load(
            label_path+"/range_label_dict_range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"_4.npy", allow_pickle=True).item()

        test_global_label_dict_1 = np.load(
            label_path+"/global_label_dict_range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"_1.npy", allow_pickle=True).item()
        test_global_label_dict_2 = np.load(
            label_path+"/global_label_dict_range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"_2.npy", allow_pickle=True).item()
        test_global_label_dict_3 = np.load(
            label_path+"/global_label_dict_range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"_3.npy", allow_pickle=True).item()
        test_global_label_dict_4 = np.load(
            label_path+"/global_label_dict_range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"_4.npy", allow_pickle=True).item()

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

    if classifier in [3,4,5]:
        rank_suc_1, rank_fail_1, global_list_1, range_list_1 = train_pair_rank_one_batch(
            range_feature_dict_1, train_range_label_dict_1, test_range_label_dict_1, range_feature_dict_2, train_range_label_dict_2, test_range_label_dict_2, range_feature_dict_3, train_range_label_dict_3, test_range_label_dict_3, range_feature_dict_4, train_range_label_dict_4, test_range_label_dict_4, global_feature_dict_1, train_global_label_dict_1, test_global_label_dict_1, global_feature_dict_2, train_global_label_dict_2, test_global_label_dict_2, global_feature_dict_3, train_global_label_dict_3, test_global_label_dict_3, global_feature_dict_4, train_global_label_dict_4, test_global_label_dict_4, precision=precision, classifier=classifier, model=model, with_header=with_header)
        rank_suc_2, rank_fail_2, global_list_2, range_list_2 = train_pair_rank_one_batch(
            range_feature_dict_4, train_range_label_dict_4, test_range_label_dict_4, range_feature_dict_2, train_range_label_dict_2, test_range_label_dict_2, range_feature_dict_3, train_range_label_dict_3, test_range_label_dict_3, range_feature_dict_1, train_range_label_dict_1, test_range_label_dict_1, global_feature_dict_4, train_global_label_dict_4, test_global_label_dict_4, global_feature_dict_2, train_global_label_dict_2, test_global_label_dict_2, global_feature_dict_3, train_global_label_dict_3, test_global_label_dict_3, global_feature_dict_1, train_global_label_dict_1, test_global_label_dict_1, precision=precision, classifier=classifier, model=model, with_header=with_header)
        rank_suc_3, rank_fail_3, global_list_3, range_list_3 = train_pair_rank_one_batch(
            range_feature_dict_4, train_range_label_dict_4, test_range_label_dict_4, range_feature_dict_1, train_range_label_dict_1, test_range_label_dict_1, range_feature_dict_3, train_range_label_dict_3, test_range_label_dict_3, range_feature_dict_2, train_range_label_dict_2, test_range_label_dict_2, global_feature_dict_4, train_global_label_dict_4, test_global_label_dict_4, global_feature_dict_1, train_global_label_dict_1, test_global_label_dict_1, global_feature_dict_3, train_global_label_dict_3, test_global_label_dict_3, global_feature_dict_2, train_global_label_dict_2, test_global_label_dict_2, precision=precision, classifier=classifier, model=model, with_header=with_header)
        rank_suc_4, rank_fail_4, global_list_4, range_list_4 = train_pair_rank_one_batch(
            range_feature_dict_4, train_range_label_dict_4, test_range_label_dict_4, range_feature_dict_2, train_range_label_dict_2, test_range_label_dict_2, range_feature_dict_1, train_range_label_dict_1, test_range_label_dict_1, range_feature_dict_3, train_range_label_dict_3, test_range_label_dict_3, global_feature_dict_4, train_global_label_dict_4, test_global_label_dict_4, global_feature_dict_2, train_global_label_dict_2, test_global_label_dict_2, global_feature_dict_1, train_global_label_dict_1, test_global_label_dict_1, global_feature_dict_3, train_global_label_dict_3, test_global_label_dict_3, precision=precision, classifier=classifier, model=model, with_header=with_header)
    else:
        merge_range_suc_1, merge_range_fail_1, merge_global_suc_1, merge_global_fail_1, rank_range_suc_1, rank_range_fail_1, rank_global_suc_1, rank_global_fail_1, range_list_1, global_list_1 = train_one_batch(
            range_feature_dict_1, train_range_label_dict_1, test_range_label_dict_1, range_feature_dict_2, train_range_label_dict_2, test_range_label_dict_2, range_feature_dict_3, train_range_label_dict_3, test_range_label_dict_3, range_feature_dict_4, train_range_label_dict_4, test_range_label_dict_4, global_feature_dict_1, train_global_label_dict_1, test_global_label_dict_1, global_feature_dict_2, train_global_label_dict_2, test_global_label_dict_2, global_feature_dict_3, train_global_label_dict_3, test_global_label_dict_3, global_feature_dict_4, train_global_label_dict_4, test_global_label_dict_4, precision=precision, classifier=classifier, model=model)
        merge_range_suc_2, merge_range_fail_2, merge_global_suc_2, merge_global_fail_2, rank_range_suc_2, rank_range_fail_2, rank_global_suc_2, rank_global_fail_2, range_list_2, global_list_2 = train_one_batch(
            range_feature_dict_4, train_range_label_dict_4, test_range_label_dict_4, range_feature_dict_2, train_range_label_dict_2, test_range_label_dict_2, range_feature_dict_3, train_range_label_dict_3, test_range_label_dict_3, range_feature_dict_1, train_range_label_dict_1, test_range_label_dict_1, global_feature_dict_4, train_global_label_dict_4, test_global_label_dict_4, global_feature_dict_2, train_global_label_dict_2, test_global_label_dict_2, global_feature_dict_3, train_global_label_dict_3, test_global_label_dict_3, global_feature_dict_1, train_global_label_dict_1, test_global_label_dict_1, precision=precision, classifier=classifier, model=model)
        merge_range_suc_3, merge_range_fail_3, merge_global_suc_3, merge_global_fail_3, rank_range_suc_3, rank_range_fail_3, rank_global_suc_3, rank_global_fail_3, range_list_3, global_list_3 = train_one_batch(
            range_feature_dict_4, train_range_label_dict_4, test_range_label_dict_4, range_feature_dict_1, train_range_label_dict_1, test_range_label_dict_1, range_feature_dict_3, train_range_label_dict_3, test_range_label_dict_3, range_feature_dict_2, train_range_label_dict_2, test_range_label_dict_2, global_feature_dict_4, train_global_label_dict_4, test_global_label_dict_4, global_feature_dict_1, train_global_label_dict_1, test_global_label_dict_1, global_feature_dict_3, train_global_label_dict_3, test_global_label_dict_3, global_feature_dict_2, train_global_label_dict_2, test_global_label_dict_2, precision=precision, classifier=classifier, model=model)
        merge_range_suc_4, merge_range_fail_4, merge_global_suc_4, merge_global_fail_4, rank_range_suc_4, rank_range_fail_4, rank_global_suc_4, rank_global_fail_4, range_list_4, global_list_4 = train_one_batch(
            range_feature_dict_4, train_range_label_dict_4, test_range_label_dict_4, range_feature_dict_2, train_range_label_dict_2, test_range_label_dict_2, range_feature_dict_1, train_range_label_dict_1, test_range_label_dict_1, range_feature_dict_3, train_range_label_dict_3, test_range_label_dict_3, global_feature_dict_4, train_global_label_dict_4, test_global_label_dict_4, global_feature_dict_2, train_global_label_dict_2, test_global_label_dict_2, global_feature_dict_1, train_global_label_dict_1, test_global_label_dict_1, global_feature_dict_3, train_global_label_dict_3, test_global_label_dict_3, precision=precision, classifier=classifier, model=model)

    if classifier in [3,4,5]:
        rank_suc = set(rank_suc_1) | set(
            rank_suc_2) | set(rank_suc_3) | set(rank_suc_4)
        rank_fail = set(rank_fail_1) | set(
            rank_fail_2) | set(rank_fail_3) | set(rank_fail_4)

        b = json.dumps(list(rank_suc))
        f2 = open(res_path+'/xgbrank_rank_suc_2_'+str(precision) +
                  '_'+str(classifier)+str(model)+'.json', 'w')
        f2.write(b)
        f2.close()

        b = json.dumps(list(rank_fail))
        f2 = open(res_path+'/xgbrank_rank_fail_'+str(precision) +
                  '_'+str(classifier)+str(model)+'.json', 'w')
        f2.write(b)
        f2.close()

        all_global_list = set(global_list_1) | set(
            global_list_2) | set(global_list_3) | set(global_list_4)
        all_range_list = set(range_list_1) | set(
            range_list_2) | set(range_list_3) | set(range_list_4)
        result = {}
        result["len_suc"] = len(rank_suc)
        result["len_fail"] = len(rank_fail)
        result["len_global_list"] = len(all_global_list)
        result["len_range_list"] = len(all_range_list)
        result["hit@"+str(precision)] = len(rank_suc) / \
                          len(all_global_list | all_range_list)
        result["rank range hit@"+str(precision)] = len(set(rank_suc)
                                     & all_range_list) / len(all_range_list)
        result["rank global hit@"+str(precision)] = len(set(rank_suc)
                                      & all_global_list) / len(all_global_list)

        print('len_suc', len(rank_suc))
        print('len_fail', len(rank_fail))
        print('len_global_list', len(all_global_list))
        print('len_range_list', len(all_range_list))

        print("acc_1:", len(rank_suc)/len(all_global_list | all_range_list))
        print("rank_range_acc_1:", len(set(rank_suc) & all_range_list) / len(all_range_list))
        print("rank_global_acc_1:", len(set(rank_suc) & all_global_list) / len(all_global_list))


        b=json.dumps(range_pred_result)
        f2=open(res_path+'/style_pair_rank_pred_result_' +
                str(precision)+"_"+str(classifier)+'.json', 'w')
        f2.write(b)
        f2.close()

        if model == 0:
            b=json.dumps(result)
            f2=open(res_path+'/style_evaluation_'+str(precision) +
                    "_"+str(classifier)+'_GB.json', 'w')
            f2.write(b)
            f2.close()
        if model == 1:
            b=json.dumps(result)
            f2=open(res_path+'/style_evaluation_'+str(precision) +
                    "_"+str(classifier)+'_XGB.json', 'w')
            f2.write(b)
            f2.close()
        if model == 2:
            b=json.dumps(result)
            f2=open(res_path+'/style_evaluation_'+str(precision) +
                    "_"+str(classifier)+'_LGB.json', 'w')
            f2.write(b)
            f2.close()
    else:
        merge_range_suc=set(merge_range_suc_1) | set(
            merge_range_suc_2) | set(merge_range_suc_3) | set(merge_range_suc_4)
        merge_range_fail=set(merge_range_fail_1) | set(
            merge_range_fail_2) | set(merge_range_fail_3) | set(merge_range_fail_4)

        merge_global_suc=set(merge_global_suc_1) | set(
            merge_global_suc_2) | set(merge_global_suc_3) | set(merge_global_suc_4)
        merge_global_fail=set(merge_global_fail_1) | set(
            merge_global_fail_2) | set(merge_global_fail_3) | set(merge_global_fail_4)

        rank_range_suc=set(rank_range_suc_1) | set(
            rank_range_suc_2) | set(rank_range_suc_3) | set(rank_range_suc_4)
        rank_range_fail=set(rank_range_fail_1) | set(
            rank_range_fail_2) | set(rank_range_fail_3) | set(rank_range_fail_4)
        rank_global_suc=set(rank_global_suc_1) | set(
            rank_global_suc_2) | set(rank_global_suc_3) | set(rank_global_suc_4)
        rank_global_fail=set(rank_global_fail_1) | set(
            rank_global_fail_2) | set(rank_global_fail_3) | set(rank_global_fail_4)


        b=json.dumps(list(merge_range_suc))
        f2=open(res_path+'/style_merge_range_suc_'+str(precision) +
                '_'+str(classifier)+str(model)+'.json', 'w')
        f2.write(b)
        f2.close()

        b=json.dumps(list(merge_global_suc))
        f2=open(res_path+'/style_merge_global_suc_'+str(precision) +
                '_'+str(classifier)+str(model)+'.json', 'w')
        f2.write(b)
        f2.close()

        b=json.dumps(list(rank_range_suc))
        f2=open(res_path+'/style_rank_range_suc_'+str(precision) +
                '_'+str(classifier)+str(model)+'.json', 'w')
        f2.write(b)
        f2.close()

        b=json.dumps(list(rank_range_fail))
        f2=open(res_path+'/style_rank_range_fail_'+str(precision) +
                '_'+str(classifier)+str(model)+'.json', 'w')
        f2.write(b)
        f2.close()

        b=json.dumps(list(rank_global_suc))
        f2=open(res_path+'/style_rank_global_suc_'+str(precision) +
                '_'+str(classifier)+str(model)+'.json', 'w')
        f2.write(b)
        f2.close()

        b=json.dumps(list(rank_global_fail))
        f2=open(res_path+'/style_rank_global_fail_'+str(precision) +
                '_'+str(classifier)+str(model)+'.json', 'w')
        f2.write(b)
        f2.close()


        all_global_list=set(global_list_1) | set(
            global_list_2) | set(global_list_3) | set(global_list_4)
        all_range_list=set(range_list_1) | set(
            range_list_2) | set(range_list_3) | set(range_list_4)
        result={}
        result["global search fail"]=list(
            (all_global_list) & set(global_search_faile))
        result["global search fail number"]=len(result["global search fail"])
        result["len_merge_range_suc"]=len(merge_range_suc)
        result["len_merge_range_fail"]=len(merge_range_fail)
        result["len_merge_global_suc"]=len(merge_global_suc)
        result["len_merge_global_fail"]=len(merge_global_fail)
        result["len_rank_range_suc"]=len(rank_range_suc)
        result["len_rank_range_fail"]=len(rank_range_fail)
        result["len_rank_global_suc"]=len(rank_global_suc)
        result["len_rank_global_fail"]=len(rank_global_fail)
        result["len_global_list"]=len(all_global_list)
        result["len_range_list"]=len(all_range_list)
        result["hit@"+str(precision)]=(len(merge_range_suc) +
                          len(merge_global_suc))/(len(all_global_list) + len(all_range_list))
        result["rank range hit@" +
            str(precision)]=len(rank_range_suc) / len(all_range_list)
        result["rank global hit@" +
            str(precision)]=len(rank_global_suc) / len(all_global_list)
        result["merge range hit@" +
            str(precision)]=len(merge_range_suc) / len(all_range_list)
        result["merge global hit@" +
            str(precision)]=len(merge_global_suc) / len(all_global_list)

        print("global search fail", len(
            all_global_list & set(global_search_faile)))
        print('len_merge_range_suc', len(merge_range_suc))
        print('len_merge_range_fail', len(merge_range_fail))
        print('len_merge_global_suc', len(merge_global_suc))
        print('len_merge_global_fail', len(merge_global_fail))

        print('len_rank_range_suc', len(rank_range_suc))
        # print('len_rank_range_fail',len(range_list) - len(rank_range_suc))
        print('len_rank_range_fail', len(rank_range_fail))
        print('len_rank_global_suc', len(rank_global_suc))

        #
        print('len_rank_global_fail', len(rank_global_fail))
        print('len_rank_global', len(all_global_list))


        print("acc_1:", (len(merge_range_suc) + len(merge_global_suc)) / \
              (len(all_global_list) + len(all_range_list)))
        print("rank_range_acc_1:", len(rank_range_suc) / len(all_range_list))
        print("rank_global_acc_1:", len(rank_global_suc) / len(all_global_list))
        print("merge_range_acc_1:", len(merge_range_suc) / len(all_range_list))
        print("merge_global_acc_1:", len(
            merge_global_suc) / len(all_global_list))

        # print("global_pred_result", global_pred_result)
        for dvid in global_pred_result:
            for key in global_pred_result[dvid]:
                for index, point in enumerate(global_pred_result[dvid][key]):
                    global_pred_result[dvid][key][index]=float(point)

       

        if model == 0:
            b=json.dumps(result)
            f2=open(res_path+'/evaluation_'+str(precision) + \
                    "_"+str(classifier)+'_GB.json', 'w')
            f2.write(b)
            f2.close()
            b=json.dumps(global_pred_result)
            f2=open(res_path+'/global_pred_result_' + \
                    str(precision)+"_"+str(classifier)+"_"+str(model)+'.json', 'w')
            f2.write(b)
            f2.close()

            b=json.dumps(global_pred_result)
            f2=open(res_path+'/range_pred_result_' + \
                    str(precision)+"_"+str(classifier)+"_"+str(model)+'.json', 'w')
            f2.write(b)
            f2.close()
        if model == 1:
            b=json.dumps(result)
            f2=open(res_path+'/style_evaluation_'+str(precision) + \
                    "_"+str(classifier)+'_XGB.json', 'w')
            f2.write(b)
            f2.close()
            b=json.dumps(global_pred_result)
            f2=open(res_path+'/style_global_pred_result_' + \
                    str(precision)+"_"+str(classifier)+"_"+str(model)+'.json', 'w')
            f2.write(b)
            f2.close()

            b=json.dumps(global_pred_result)
            f2=open(res_path+'/style_range_pred_result_' + \
                    str(precision)+"_"+str(classifier)+"_"+str(model)+'.json', 'w')
            f2.write(b)
            f2.close()
        if model == 2:
            b=json.dumps(result)
            f2=open(res_path+'/evaluation_'+str(precision) + \
                    "_"+str(classifier)+'_LGB.json', 'w')
            f2.write(b)
            f2.close()
            b=json.dumps(global_pred_result)
            f2=open(res_path+'/global_pred_result_' + \
                    str(precision)+"_"+str(classifier)+"_"+str(model)+'.json', 'w')
            f2.write(b)
            f2.close()

            b=json.dumps(global_pred_result)
            f2=open(res_path+'/range_pred_result_' + \
                    str(precision)+"_"+str(classifier)+"_"+str(model)+'.json', 'w')
            f2.write(b)
            f2.close()


def train_one_batch(range_feature_dict_1, train_range_label_dict_1, test_range_label_dict_1, range_feature_dict_2, train_range_label_dict_2, test_range_label_dict_2, range_feature_dict_3, train_range_label_dict_3, test_range_label_dict_3, range_feature_dict_4, train_range_label_dict_4, test_range_label_dict_4, global_feature_dict_1, train_global_label_dict_1, test_global_label_dict_1, global_feature_dict_2, train_global_label_dict_2, test_global_label_dict_2, global_feature_dict_3, train_global_label_dict_3, test_global_label_dict_3, global_feature_dict_4, train_global_label_dict_4, test_global_label_dict_4, precision=1, classifier=0, model=0):
    range_X_train=[]
    range_y_train=[]

    range_X_test=[]
    range_y_test=[]

    # print(range_feature_dict_1)
    for feature, label in [(range_feature_dict_1, train_range_label_dict_1), (range_feature_dict_2, train_range_label_dict_2), (range_feature_dict_3, train_range_label_dict_3)]:
        for dvid in feature.keys():
            for candid in feature[dvid]:
                if dvid in label:
                    if candid in label[dvid]:
                        # if(len(feature[dvid][candid]) == 20):
                        #     print(dvid, candid)
                        range_X_train.append(feature[dvid][candid])
                        range_y_train.append(label[dvid][candid])

    test_id_dict_1={}
    test_id_dict={}
    _id=0
    for dvid in range_feature_dict_4.keys():
        for candid in range_feature_dict_4[dvid]:
            if dvid in test_range_label_dict_4:
                if candid in test_range_label_dict_4[dvid]:

                    if dvid not in test_id_dict_1:
                        test_id_dict_1[dvid]={}
                    test_id_dict_1[dvid][candid]=_id
                    test_id_dict[_id]=[dvid, candid]
                    _id += 1
                    range_X_test.append(range_feature_dict_4[dvid][candid])
                    range_y_test.append(test_range_label_dict_4[dvid][candid])

    print(np.array(range_X_train).shape)
    test_range=[]
    for i in range_X_test:
        test_range.append(i[0:8])
    range_X_test=test_range
    if model == 0:
        if classifier == 0:
            range_clf=GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
                np.array(range_X_train), range_y_train)
            range_y_pred=range_clf.predict(range_X_test)
        elif classifier == 1:
            range_clf=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
                np.array(range_X_train), range_y_train)
            range_y_pred=range_clf.predict(range_X_test)
        elif classifier == 2:
            range_clf=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
                np.array(range_X_train), range_y_train)
            range_y_pred_1=range_clf.predict_proba(range_X_test)
            range_y_pred=range_y_pred_1[:, 1]
    elif model == 1:
        if classifier == 0:
            range_clf=XGBRegressor(objective='binary:logistic', n_estimators=100, random_state=0).fit(
                np.array(range_X_train), range_y_train)
            range_y_pred=range_clf.predict(np.array(range_X_test))
        elif classifier == 1:
            range_clf=XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=0).fit(
                np.array(range_X_train), range_y_train)
            range_y_pred=range_clf.predict(np.array(range_X_test))
        elif classifier == 2:
            range_clf=XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=0).fit(
                np.array(range_X_train), range_y_train)
            range_y_pred_1=range_clf.predict_proba(np.array(range_X_test))
            range_y_pred=range_y_pred_1[:, 1]
    elif model == 2:
        params={
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'metric': {'l2', 'auc'},  # 评估函数
            'num_leaves': 31,   # 叶子节点数
            'learning_rate': 0.05,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.8,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
        train_data=lgb.Dataset(data=range_X_train, label=range_y_train)
        if classifier == 0:
            params['objective']='regression'
            range_clf=lgb.train(params, train_data, num_boost_round=20)
            range_y_pred=range_clf.predict(range_X_test)
        elif classifier == 1:
            params['objective']='binary'
            range_clf=lgb.train(params, train_data, num_boost_round=20)
            range_y_pred=range_clf.predict(range_X_test)
        elif classifier == 2:
            params['objective']='binary'
            range_clf=lgb.train(params, train_data, num_boost_round=20)
            range_y_pred_1=range_clf.predict_proba(np.array(range_X_test))
            range_y_pred=range_y_pred_1[:, 1]

    print(np.array(range_X_test).shape)
    print(np.array(range_y_test).shape)

    dvid_list=[]
    pred_list=[]
    test_list=[]

    range_rank_suc=[]
    range_rank_fail=[]

    range_search_fail=[]
    top_range={}
    ran_list=[]
    glo_list=[]

    for index, i in enumerate(range_y_test):
        dvid=test_id_dict[index][0]

        for k in dvinfos:
            if k["ID"] == dvid:
                if "," in k["Value"]:
                    glo_list.append(dvid)

                else:
                    ran_list.append(dvid)
    print("precision:", precision)
    test_list_dict={}
    pred_list_dict={}

    for index, i in enumerate(range_y_test):
        if test_id_dict[index][0] not in test_list_dict:
            test_list_dict[test_id_dict[index][0]]=[]
            pred_list_dict[test_id_dict[index][0]]=[]
        pred_list_dict[test_id_dict[index][0]].append(
            float(range_y_pred[index]))
        test_list_dict[test_id_dict[index][0]].append(
            float(range_y_test[index]))

    for dvid in test_list_dict:
        test_list=test_list_dict[dvid]
        pred_list=pred_list_dict[dvid]
        range_pred_result[dvid]={"test": test_list, "pred": pred_list}

        prec=0
        temp=pred_list.copy()
        while(prec < precision):
            top_index=np.array(temp).argmax()
            if dvid not in top_range:
                top_range[dvid]=[]
            top_range[dvid].append(
                (top_index, pred_list[top_index], test_list[top_index]))
            temp[top_index]=-1
            prec += 1

        if dvid in glo_list:
            top1=np.array(pred_list).argmax()
        if 1 not in test_list:
            range_rank_fail.append(dvid)
            range_search_fail.append(dvid)
            range_search_faile.append(dvid)
        else:
            prec=0
            temp=pred_list.copy()
            top_index_list=[]
            while prec < precision:
                top_index=np.array(temp).argmax()
                top_index_list.append(top_index)
                temp[top_index]=-1
                prec += 1

            is_suc=False
            for top_index in top_index_list:
                if test_list[top_index] != 0:
                    is_suc=True
                    break
            if not is_suc:
                range_rank_fail.append(dvid)
            else:
                range_rank_suc.append(dvid)

    print('range_rank_suc', len(range_rank_suc))
    print('range_rank_fail', len(range_rank_fail))
    print('range_search_fail', len(range_search_fail))

    global_X_train=[]
    global_y_train=[]

    global_X_test=[]
    global_y_test=[]
    # print(global_feature_dict_4)
    for feature, label in [(global_feature_dict_1, train_global_label_dict_1), (global_feature_dict_2, train_global_label_dict_2), (global_feature_dict_3, train_global_label_dict_3)]:
        for dvid in feature.keys():
            for candid in feature[dvid]:
                if dvid in label:
                    if candid in label[dvid]:
                        global_X_train.append(feature[dvid][candid])
                        global_y_train.append(label[dvid][candid])

    test_id_dict_1={}
    test_id_dict={}
    _id=0
    for dvid in global_feature_dict_4.keys():
        for candid in global_feature_dict_4[dvid]:
            if dvid in test_global_label_dict_4:
                # print("lllllllllllll")
                if candid in test_global_label_dict_4[dvid]:
                    # print("xxxxxxxxxxxxxxxxxx")
                    if dvid not in test_id_dict_1:
                        test_id_dict_1[dvid]={}
                    test_id_dict_1[dvid][candid]=_id
                    test_id_dict[_id]=[dvid, candid]
                    _id += 1
                    global_X_test.append(global_feature_dict_4[dvid][candid])
                    global_y_test.append(
                        test_global_label_dict_4[dvid][candid])
    print(np.array(global_X_train).shape)
    print(np.array(global_y_train).shape)
    if classifier == 0:
        global_clf=GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
            np.array(global_X_train), global_y_train)
        global_y_pred=global_clf.predict(global_X_test)
    elif classifier == 1:
        global_clf=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
            np.array(global_X_train), global_y_train)
        global_y_pred=global_clf.predict(global_X_test)
    elif classifier == 2:
        global_clf=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
            np.array(global_X_train), global_y_train)
        global_y_pred_1=global_clf.predict_proba(global_X_test)
        global_y_pred=global_y_pred_1[:, 1]

    dvid_list=[]
    pred_list=[]
    test_list=[]

    global_rank_suc=[]
    global_rank_fail=[]

    global_search_fail=[]

    top_global={}

    for index, i in enumerate(global_y_test):
        if len(dvid_list) == 0:
            dvid_list.append(test_id_dict[index][0])
        else:
            if test_id_dict[index][0] not in dvid_list:
                global_pred_result[dvid_list[-1]
                            ]={"test": test_list, "pred": pred_list}
                prec=0
                temp=pred_list.copy()
                while(prec < precision):
                    top_index=np.array(temp).argmax()
                    if dvid_list[-1] not in top_global:
                        top_global[dvid_list[-1]]=[]
                    top_global[dvid_list[-1]].append(
                        (top_index, pred_list[top_index], test_list[top_index]))
                    temp[top_index]=-1
                    prec += 1

                if 1 not in test_list:
                    global_rank_fail.append(dvid_list[-1])
                    global_search_fail.append(dvid_list[-1])
                    global_search_faile.append(dvid_list[-1])
                else:
                    prec=0
                    temp=pred_list.copy()
                    top_index_list=[]
                    while prec < precision:
                        top_index=np.array(temp).argmax()
                        top_index_list.append(top_index)
                        temp[top_index]=-1
                        prec += 1

                    is_suc=False
                    for top_index in top_index_list:
                        if test_list[top_index] != 0:
                            is_suc=True
                            break
                    if not is_suc:
                        global_rank_fail.append(dvid_list[-1])
                    else:
                        global_rank_suc.append(dvid_list[-1])
                pred_list=[]
                test_list=[]
                dvid_list.append(test_id_dict[index][0])

        pred_list.append(global_y_pred[index])
        test_list.append(global_y_test[index])

    all_=0
    hardcode=0

    for dvid in list(set(global_rank_suc) | set(global_rank_fail) | set(range_rank_suc) | set(range_rank_fail)):
        for k in dvinfos:
            if k["ID"] == dvid:
                if "," in k["Value"]:
                    hardcode += 1

        all_ += 1

    range_list=[]
    global_list=[]

    for dvid in list(set(global_rank_suc) | set(global_rank_fail) | set(range_rank_suc) | set(range_rank_fail)):
        for k in dvinfos:
            if k["ID"] == dvid:
                if "," in k["Value"]:
                    global_list.append(dvid)

                else:
                    range_list.append(dvid)

    print('global_rank_suc', len(global_rank_suc))
    print('global_rank_fail', len(global_rank_fail))
    print('global_search_fail', len(global_search_fail))


    merge_global_suc=[]
    merge_global_fail=[]
    merge_range_suc=[]
    merge_range_fail=[]



    for dvid in list(set(global_rank_suc) | set(global_rank_fail) | set(range_rank_suc) | set(range_rank_fail)):
        if dvid not in top_global:
            if dvid in global_list:
                merge_global_fail.append(dvid)
            else:
                if dvid not in top_range:
                    merge_range_fail.append(dvid)
                    continue
                prec=0
                is_suc=False
                while prec < precision:
                    if top_range[dvid][prec][2] != 0:
                        is_suc=True
                        break
                    prec += 1
                if not is_suc:
                    merge_range_fail.append(dvid)
                else:
                    merge_range_suc.append(dvid)
        elif dvid not in top_range:
            if dvid in range_list:
                merge_range_fail.append(dvid)
            else:
                if dvid not in top_global:
                    merge_global_fail.append(dvid)
                    continue
                prec=0
                is_suc=False
                while prec < precision:
                    if top_global[dvid][prec][2] != 0:
                        is_suc=True
                        break
                    prec += 1
                if not is_suc:
                    merge_global_fail.append(dvid)
                else:
                    merge_global_suc.append(dvid)
        else:
            prec=0
            top_merge_label_list=[]
            range_index=0
            global_index=0
            while prec < precision:
                if top_global[dvid][global_index][1] < top_range[dvid][range_index][1]:  # range add
                    top_merge_label_list.append(
                        top_range[dvid][range_index][2])
                    range_index += 1
                else:
                    top_merge_label_list.append(
                        top_global[dvid][global_index][2])
                    global_index += 1
                prec += 1
            is_suc=False
            for label in top_merge_label_list:
                if label != 0:
                    is_suc=True
                    break
                prec += 1
            if dvid in global_list:
                if not is_suc:
                    merge_global_fail.append(dvid)
                else:
                    merge_global_suc.append(dvid)
            else:
                if not is_suc:
                    merge_range_fail.append(dvid)
                else:
                    merge_range_suc.append(dvid)
    print('global_merge_suc', len(merge_global_suc))

    return merge_range_suc, list(set(range_list)-set(merge_range_suc)), merge_global_suc,  list(set(global_list)-set(merge_global_suc)), range_rank_suc, list(set(range_list)-set(range_rank_suc)), global_rank_suc, list(set(global_list)-set(global_rank_suc)), range_list, global_list

def train_pair_rank_one_batch(range_feature_dict_1, train_range_label_dict_1, test_range_label_dict_1, range_feature_dict_2, train_range_label_dict_2, test_range_label_dict_2, range_feature_dict_3, train_range_label_dict_3, test_range_label_dict_3, range_feature_dict_4, train_range_label_dict_4, test_range_label_dict_4, global_feature_dict_1, train_global_label_dict_1, test_global_label_dict_1, global_feature_dict_2, train_global_label_dict_2, test_global_label_dict_2, global_feature_dict_3, train_global_label_dict_3, test_global_label_dict_3, global_feature_dict_4, train_global_label_dict_4, test_global_label_dict_4, with_header, precision=1, classifier=0, model=0):
    X_train=[]
    y_train=[]

    X_test=[]
    y_test=[]

    train_group=[]
    for range_feature, global_feature, range_label, global_label in [(range_feature_dict_1, global_feature_dict_1, train_range_label_dict_1, train_global_label_dict_1), (range_feature_dict_2, global_feature_dict_2, train_range_label_dict_2, train_global_label_dict_2), (range_feature_dict_3, global_feature_dict_3, train_range_label_dict_3, train_global_label_dict_3)]:
        for dvid in list(set(range_feature.keys()) | set(global_feature.keys())):
            group_num=0
            if dvid in range_feature:
                for candid in range_feature[dvid]:
                    if dvid in range_label:
                        if candid in range_label[dvid]:
                            # if(len(feature[dvid][candid]) == 20):
                            #     print(dvid, candid)
                            X_train.append(change_both_feature(True, range_feature[dvid][candid], with_header=with_header))
                            y_train.append(range_label[dvid][candid])
                            group_num += 1
            if dvid in global_feature:
                for candid in global_feature[dvid]:
                    if dvid in global_label:
                        if candid in global_label[dvid]:
                            # if(len(feature[dvid][candid]) == 20):
                            #     print(dvid, candid)
                            X_train.append(change_both_feature(False, global_feature[dvid][candid], with_header=with_header))
                            y_train.append(global_label[dvid][candid])
                            group_num += 1
            train_group.append(group_num)

    test_id_dict_1={}
    test_id_dict={}
    _id=0
    test_group=[]
    for dvid in list(set(range_feature_dict_4.keys()) | set(global_feature_dict_4.keys())):
        group_num=0
        if dvid in range_feature_dict_4:
            for candid in range_feature_dict_4[dvid]:
                if dvid in test_range_label_dict_4:
                    if candid in test_range_label_dict_4[dvid]:

                        if dvid not in test_id_dict_1:
                            test_id_dict_1[dvid]={}
                        test_id_dict_1[dvid][candid]=_id
                        test_id_dict[_id]=[dvid, candid]
                        _id += 1
                        
                        X_test.append(change_both_feature(True, range_feature_dict_4[dvid][candid], with_header=with_header))
                        y_test.append(test_range_label_dict_4[dvid][candid])
                        group_num += 1
        if dvid in global_feature_dict_4:
            for candid in global_feature_dict_4[dvid]:
                if dvid in test_global_label_dict_4:
                    if candid in test_global_label_dict_4[dvid]:

                        if dvid not in test_id_dict_1:
                            test_id_dict_1[dvid]={}
                        test_id_dict_1[dvid][candid]=_id
                        test_id_dict[_id]=[dvid, candid]
                        _id += 1
                        # print('global_feature_dict_4[dvid][candid]', len(global_feature_dict_4[dvid][candid]))
                        X_test.append(change_both_feature(False, global_feature_dict_4[dvid][candid], with_header=with_header))
                        y_test.append(test_global_label_dict_4[dvid][candid])
                        group_num += 1
        test_group.append(group_num)

    print(np.array(X_train).shape)

    if model == 0:
        if classifier == 0:
            range_clf=GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
                np.array(range_X_train), range_y_train)
            range_y_pred=range_clf.predict(range_X_test)
        elif classifier == 1:
            range_clf=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
                np.array(range_X_train), range_y_train)
            range_y_pred=range_clf.predict(range_X_test)
        elif classifier == 2:
            range_clf=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
                np.array(range_X_train), range_y_train)
            range_y_pred_1=range_clf.predict_proba(range_X_test)
            range_y_pred=range_y_pred_1[:, 1]
    elif model == 1:
        if classifier == 0:
            range_clf=XGBRegressor(objective='binary:logistic', n_estimators=100, random_state=0).fit(
                np.array(range_X_train), range_y_train)
            range_y_pred=range_clf.predict(np.array(range_X_test))
        elif classifier == 1:
            range_clf=XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=0).fit(
                np.array(range_X_train), range_y_train)
            range_y_pred=range_clf.predict(np.array(range_X_test))
        elif classifier == 2:
            range_clf=XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=0).fit(
                np.array(range_X_train), range_y_train)
            range_y_pred_1=range_clf.predict_proba(np.array(range_X_test))
            range_y_pred=range_y_pred_1[:, 1]
        elif classifier == 3:
            xgb_rank_params={
                'bst:max_depth': 1,
                'bst:eta': 1, 'silent': 1,
                'objective': 'rank:pairwise',
                'nthread': 4,
                'n_estimators': 100,
                'random_state': 0,
            }
            # print(X_train)
            # for i in X_train:
            #     print(len(i))
            X_train=DMatrix(np.array(X_train), label=y_train)
            X_train.set_group(np.array(train_group))
            rankModel=train(xgb_rank_params, X_train, num_boost_round=20)
            # for i in X_test:
            #     print(len(i))
            X_test=DMatrix(np.array(X_test))
            X_test.set_group(np.array(test_group))
            y_pred=rankModel.predict(X_test)
        elif classifier == 4:
            xgb_rank_params={
                'bst:max_depth': 1,
                'bst:eta': 1, 'silent': 1,
                'objective': 'rank:ndcg',
                'nthread': 4,
                'n_estimators': 100,
                'random_state': 0,
            }
            X_train=DMatrix(np.array(X_train), label=y_train)
            X_train.set_group(np.array(train_group))
            rankModel=train(xgb_rank_params, X_train, num_boost_round=20)

            X_test=DMatrix(np.array(X_test))
            X_test.set_group(np.array(test_group))
            y_pred=rankModel.predict(X_test)
        elif classifier == 5:
            xgb_rank_params={
                'bst:max_depth': 1,
                'bst:eta': 1, 'silent': 1,
                'objective': 'rank:map',
                'nthread': 4,
                'n_estimators': 100,
                'random_state': 0,
            }
            X_train=DMatrix(np.array(X_train), label=y_train)
            X_train.set_group(np.array(train_group))
            rankModel=train(xgb_rank_params, X_train, num_boost_round=20)

            X_test=DMatrix(np.array(X_test))
            X_test.set_group(np.array(test_group))
            y_pred=rankModel.predict(X_test)

    elif model == 2:
        params={
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'metric': {'l2', 'auc'},  # 评估函数
            'num_leaves': 31,   # 叶子节点数
            'learning_rate': 0.05,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.8,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
        train_data=lgb.Dataset(data=range_X_train, label=range_y_train)
        if classifier == 0:
            params['objective']='regression'
            range_clf=lgb.train(params, train_data, num_boost_round=20)
            range_y_pred=range_clf.predict(range_X_test)
        elif classifier == 1:
            params['objective']='binary'
            range_clf=lgb.train(params, train_data, num_boost_round=20)
            range_y_pred=range_clf.predict(range_X_test)
        elif classifier == 2:
            params['objective']='binary'
            range_clf=lgb.train(params, train_data, num_boost_round=20)
            range_y_pred_1=range_clf.predict_proba(np.array(range_X_test))
            range_y_pred=range_y_pred_1[:, 1]

    print(np.array(X_test).shape)
    print(np.array(y_test).shape)
    dvid_list=[]
    pred_list=[]
    test_list=[]

    rank_suc=[]
    rank_fail=[]

    top_={}
    ran_list=[]
    glo_list=[]

    for index, i in enumerate(y_test):
        # if dvid in search_fail:
        #     continue
        dvid=test_id_dict[index][0]

        for k in dvinfos:
            if k["ID"] == dvid:
                if "," in k["Value"]:
                    glo_list.append(dvid)

                else:
                    ran_list.append(dvid)

    print("precision:", precision)
    test_list_dict={}
    pred_list_dict={}

    for index, i in enumerate(y_test):
        if test_id_dict[index][0] not in test_list_dict:
            test_list_dict[test_id_dict[index][0]]=[]
            pred_list_dict[test_id_dict[index][0]]=[]
        pred_list_dict[test_id_dict[index][0]].append(float(y_pred[index]))
        test_list_dict[test_id_dict[index][0]].append(float(y_test[index]))

    for dvid in test_list_dict:
        test_list=test_list_dict[dvid]
        pred_list=pred_list_dict[dvid]
        range_pred_result[dvid] = {"test": test_list, "pred": pred_list}
        if 1 not in test_list:
            rank_fail.append(dvid)
        else:
            prec=0
            temp=pred_list.copy()
            top_index_list=[]
            while prec < precision:
                top_index=np.array(temp).argmax()
                top_index_list.append(top_index)
                temp[top_index]=-1
                prec += 1

            is_suc=False
            for top_index in top_index_list:
                if test_list[top_index] != 0:
                    is_suc=True
                    break
            if not is_suc:
                rank_fail.append(dvid)
            else:
                rank_suc.append(dvid)

    print('rank_suc', len(rank_suc))
    print('rank_fail', len(rank_fail))

    range_list=[]
    global_list=[]

    for dvid in list(set(range_feature_dict_1) | set(range_feature_dict_2) | set(range_feature_dict_3) | set(range_feature_dict_4) | set(global_feature_dict_1) | set(global_feature_dict_2) | set(global_feature_dict_3) | set(global_feature_dict_4)):
        for k in dvinfos:
            if k["ID"] == dvid:
                if "," in k["Value"]:
                    global_list.append(dvid)

                else:
                    range_list.append(dvid)
    return rank_suc, rank_fail, global_list, range_list

def look_rank_fail(evaluate_range, evaluate_global, precision):
    with open("with_header/range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"/evaluation_"+str(precision)+".json", 'r', encoding='UTF-8') as f:
        evaluation_5=json.load(f)
    with open("with_header/range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"/rank_global_fail_"+str(precision)+".json", 'r', encoding='UTF-8') as f:
        rank_global_fail_5=json.load(f)
    with open("with_header/range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"/full_header_pred_result_"+str(precision)+".json", 'r', encoding='UTF-8') as f:
        full_header_pred_result_5=json.load(f)
    with open("with_header/range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"/rank_global_suc_"+str(precision)+".json", 'r', encoding='UTF-8') as f:
        rank_global_suc_5=json.load(f)

    no_search=[]
    rank_fail=[]
    count_id=[]
    for i in list(set(rank_global_fail_5) | set(rank_global_suc_5)):
        count_id.append(i)
        if str(i) not in full_header_pred_result_5 and i not in full_header_pred_result_5:
            no_search.append(i)

        else:

            if 1 not in full_header_pred_result_5[str(i)]['test']:
                no_search.append(i)
            else:
                if i in rank_global_fail_5:
                # print(full_header_pred_result_5[str(i)]['test'])
                    rank_fail.append(i)
    print("fail before rank:", len(no_search))
    print(len(count_id)-len(no_search))
    print((len(count_id)-len(no_search))/len(count_id))
    print(len(count_id))
    print(rank_fail)

def look_range_rank_fail(evaluate_range, evaluate_global, precision):
    with open("with_header/range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"/evaluation_"+str(precision)+".json", 'r', encoding='UTF-8') as f:
        evaluation_5=json.load(f)
    with open("with_header/range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"/rank_range_fail_"+str(precision)+".json", 'r', encoding='UTF-8') as f:
        rank_global_fail_5=json.load(f)
    with open("with_header/range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"/full_header_range_pred_result_"+str(precision)+".json", 'r', encoding='UTF-8') as f:
        full_header_pred_result_5=json.load(f)
    with open("with_header/range_"+str(evaluate_range).replace("-", "n")+"_global_"+str(evaluate_global)+"/rank_range_suc_"+str(precision)+".json", 'r', encoding='UTF-8') as f:
        rank_global_suc_5=json.load(f)

    # print(len(rank_global_fail_5))
    no_search=[]
    rank_fail=[]
    count_id=[]
    for i in list(set(rank_global_fail_5) | set(rank_global_suc_5)):
        count_id.append(i)
        if str(i) not in full_header_pred_result_5:
            no_search.append(i)

        else:
            if 1 not in full_header_pred_result_5[str(i)]['test']:
                no_search.append(i)
            else:
                # print(full_header_pred_result_5[str(i)]['test'])
                rank_fail.append(i)
    print("fail before rank:", len(no_search))
    print(len(count_id)-len(no_search))
    print((len(count_id)-len(no_search))/len(count_id))

def look_fail():
    evaluate_range=2
    evaluate_global=2
    precision=1
    look_rank_fail(evaluate_range, evaluate_global, precision)
    look_range_rank_fail(evaluate_range, evaluate_global, precision)
    # with open("with_header/range_0_global_0/evaluation_1.json", 'r', encoding='UTF-8') as f:
    #     evaluation_1 = json.load(f)

    # with open("with_header/range_n1_global_0/evaluation_1.json", 'r', encoding='UTF-8') as f:
    #     evaluation_2 = json.load(f)
    # print(set(evaluation_1["global search fail"])-set(evaluation_2["global search fail"]))
    # print(rank_fail)
if __name__ == '__main__':
    devide_training(precision=5, need_load_features=False, need_load_labels=False,
                    evaluate_range=1, evaluate_global=1, with_header=2, classifier=0, model=1)
    # look_fail()
