from genericpath import isfile
import json 
import matplotlib.pyplot as plt
import pprint
from random import sample
import os
import numpy as np
import random
from numpy.core.numeric import ones_like
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score
import pandas as pd

def check_sorted_values():
    sort_result = []
    def check_s_2_b(i):
        refer_item_before = str(i["candidate_list"][0]["Value"])
        is_first = True
        for refer_item in i["candidate_list"]:
            if is_first == True:
                is_first = False
            else:
                if refer_item_before >= str(refer_item["Value"]):
                    return False
                refer_item_before = str(refer_item["Value"])
        return True

    def check_b_2_s(i):
        refer_item_before = str(i["candidate_list"][0]["Value"])
        is_first = True
        for refer_item in i["candidate_list"]:
            if is_first == True:
                is_first = False
            else:
                if refer_item_before <= str(refer_item["Value"]):
                    return False
                refer_item_before = str(refer_item["Value"])
        return True

    file_path = "../PredictDV/ListDV/continous_batch_0_1.json"
    
    all_result = 0
    with open(file_path, 'r', encoding='UTF-8') as f:
        listdvfindelete_o_relax_strip_list = json.load(f)
    
    sorted_num = 0
    for i in listdvfindelete_o_relax_strip_list:
        if not os.path.exists("../PredictDV/delete_row_single_o_baseline/" + str(i["ID"]) + "_search_result.json"):
            continue
        with open("../PredictDV/delete_row_single_o_baseline/" + str(i["ID"]) + "_search_result.json", 'r', encoding='UTF-8') as f:
            candidate_list = json.load(f)

        cand_index = 0
        for j in candidate_list:
            
            # if not (j["left_top_x"] == j["right_bottom_x"] and j["left_top_y"] == j["right_bottom_y"]):
            if check_b_2_s(j) or check_s_2_b(j):
                sort_result.append({"dvid": i["ID"], "cand_index": cand_index, "sortness": True})
            else:
                sort_result.append({"dvid": i["ID"], "cand_index": cand_index, "sortness": False})
            cand_index += 1

    
    b = json.dumps(sort_result)
    f2 = open('sortness.json', 'w')
    f2.write(b)
    f2.close()
    print(all_result)
    print(sorted_num)

def get_sematic_distance():
    word2vec = {}
    with open("data/glove.840B.300d.txt", 'r',encoding="utf-8") as f:
        
        for line in f:
            index = line.find(' ')
            word = line[0:index]
            vec = line[index+1:].split(' ')
            num_vec = []
            for i in vec:
                num_vec.append(float(i))
            word2vec[word] = num_vec
            print(line)

    file_path = "../PredictDV/ListDV/continous_batch_0_1.json"
    sematic_result = []
    all_result = 0
    with open(file_path, 'r', encoding='UTF-8') as f:
        listdvfindelete_o_relax_strip_list = json.load(f)
    
    sorted_num = 0
    for i in listdvfindelete_o_relax_strip_list:
        if not os.path.exists("../PredictDV/delete_row_single_o_baseline/" + str(i["ID"]) + "_search_result.json"):
            continue
        with open("../PredictDV/delete_row_single_o_baseline/" + str(i["ID"]) + "_search_result.json", 'r', encoding='UTF-8') as f:
            candidate_list = json.load(f)

        cand_index = 0
        for j in candidate_list:
            in_number = 0
            res = np.zeros(300)
            for k in j["candidate_list"]:
                if str(k["Value"]) in word2vec.keys():
                    in_number += 1
                    res += word2vec[str(k["Value"])]
            res /= in_number
            cand_index += 1
            sematic_result.append({"dvid": i["ID"], "cand_index": cand_index, "sematic_distance": list(res)})
    b = json.dumps(sematic_result)
    f2 = open('sematic_result.json', 'w')
    f2.write(b)
    f2.close()
    print(all_result)
    print(sorted_num)
    

def analyze_gt():
    file_path = "../PredictDV/ListDV/continous_batch_0_1.json"
    column_row_dic = {}
    orientation_dic = {}
    isempty_dic = {}
    hasdistinct_dic = {}
    hashead_dic = {}
    hastail_dic = {}
    single_cell_list = []
        
    with open(file_path, 'r', encoding='UTF-8') as f:
        listdvfindelete_o_relax_strip_list = json.load(f)
    all_num = 0

    row_num = 0
    sample_50 = []
    random.shuffle(listdvfindelete_o_relax_strip_list)

    one_value_content_nubmer = 0
    range_num= 0
    for i in listdvfindelete_o_relax_strip_list:
        all_num += 1
        content_list = set()
        for k in i["content"]:
            content_list.add(k["Value"])
        if len(content_list) == 1:
            one_value_content_nubmer += 1
        if  i["GTOrientationType"] == 3:
            single_cell_list.append(i["ID"])
        if i["GTOrientationType"] == 0:
            continue
        range_num += 1
        if i["GTOrientationType"] not in column_row_dic:
            column_row_dic[i["GTOrientationType"]] = 0
        column_row_dic[i["GTOrientationType"]] += 1
        if(i["GTOrientationType"] == 2):
            if len(sample_50) < 50:
                sample_50.append(i["ID"])
            # print(i["ID"])
            row_num += 1

        if i["GTLocationType"] not in orientation_dic:
            orientation_dic[i["GTLocationType"]] = 0
        orientation_dic[i["GTLocationType"]] += 1

        if i["GTHasEmptyCell"] not in isempty_dic:
            isempty_dic[i["GTHasEmptyCell"]] = 0
        isempty_dic[i["GTHasEmptyCell"]] += 1

        if i["GTHasDistinctValues"] not in hasdistinct_dic:
            hasdistinct_dic[i["GTHasDistinctValues"]] = 0
        hasdistinct_dic[i["GTHasDistinctValues"]] += 1

        if i["GTHasTail"]:
            print(i["ID"])
        if i["GTHasHead"] not in hashead_dic:
            hashead_dic[i["GTHasHead"]] = 0
        hashead_dic[i["GTHasHead"]] += 1

        if i["GTHasTail"] not in hastail_dic:
            hastail_dic[i["GTHasTail"]] = 0
        hastail_dic[i["GTHasTail"]] += 1

    print(column_row_dic)
    print(orientation_dic)
    print(isempty_dic)
    print(hasdistinct_dic)
    print(hashead_dic)
    print(hastail_dic)
    print("range_num", range_num)
    print("all_num", all_num)
    print("one content number ", one_value_content_nubmer)

    print(row_num)
    print(sample_50)
    print(single_cell_list)
    print(len(single_cell_list))
    np.save("single_cell_list.npy",single_cell_list)


# def get_global_dict_header():



def get_leftness():
    file_list = os.listdir("../PredictDV/evaluates1/u_eval/")
    file_list.sort()
    # with open("../PredictDV/ListDV/continous_batch_0_1.json",'r', encoding='UTF-8') as f:
    #     listdvinfdelete_o_relax_strip_list = json.load(f)
    all_num = 0
    result = []
    x = []
    bincount = {}
    for filename in file_list:
        dvid = int(filename.split('.')[0])
        with open("../PredictDV/evaluates1/u_eval/"+filename,'r', encoding='UTF-8') as f:
            one_result = json.load(f)
        with open("../test-table-understanding/test-table-understanding/data/understanding_1000_batches/"+str(dvid)+"_tableinfo.json",'r', encoding='UTF-8') as f:
            tableinfos = json.load(f)
            if ',' in one_result["refer_value"]:
                continue
            if len(tableinfos) == 0:
                continue

            temp = one_result["refer_last_row"]
            one_result["refer_last_row"] = one_result["refer_last_column"]
            one_result["refer_last_column"] = temp
            if not one_result["is_range_superset"] or one_result["refer_first_column"] != one_result["refer_last_column"]:
                continue
            all_num += 1
            found_tableinfo = tableinfos[0]
            is_found = False
            for tableinfo in tableinfos:

                if tableinfo["left_top_x"] <= one_result["refer_first_row"] and tableinfo["left_top_y"] <= one_result["refer_first_column"] and tableinfo["right_bottom_x"] >= one_result["refer_last_row"] and tableinfo["right_bottom_y"] >= one_result["refer_last_column"]:
                    found_tableinfo = tableinfo
                    is_found = True
                    break
            if is_found == False:
                print(dvid)
                continue
            
            leftness = (one_result["refer_first_column"] - tableinfo["left_top_y"] + 1 - 1) / ( tableinfo["right_bottom_y"] - tableinfo["left_top_y"] + 1 - 1)
            print("xxxxx")
            print(one_result["refer_first_column"])
            print(one_result["refer_last_column"])
            print(tableinfo["left_top_y"])
            print(tableinfo["right_bottom_y"])
            if leftness not in bincount:
                bincount[leftness] = 0
            bincount[leftness] += 1
            # if leftness>1:
            #     print(leftness)
        x.append(all_num)
        result.append(leftness)
        
    print(all_num)
    print(np.array(result).mean())
    print(np.array(result).std())
    # print(np.bincount(np.array(result)))
    print(np.median(np.array(result)))
    print(bincount)
    plt.scatter(x,result)
    # plt.ylim(0,10)
    plt.savefig("../PredictDV/evaluates1/leftness.png")


def check_header():
    file_list_1 = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval/")
    file_list_2 = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval_rmh/")
    file_list_1.sort()
    file_list_2.sort()
    all_result = 0
    
    u_cover_number = 0
    urmh_cover_number = 0
    u_cover_rmh_not_cover = 0

    urmh_full_fit = 0
    u_full_fit = 0
    urmh_full_fit_u_not = 0
    c = 0
    hard_code_number = 0
    for filename in file_list_1:
        with open("../PredictDV/evaluates1/delete_o_relax_strip_eval/"+filename,'r', encoding='UTF-8') as f:
            u_one_result = json.load(f)
        if filename not in file_list_1 or filename not in file_list_2:
            continue
        with open("../PredictDV/evaluates1/delete_o_relax_strip_eval_rmh/"+filename,'r', encoding='UTF-8') as f:
            urmh_one_result = json.load(f)
      
      
        all_result += 1
        if "," in u_one_result["refer_value"]:
            hard_code_number += 1
            continue

        if(u_one_result["is_range_superset"]):
            u_cover_number += 1
            if(not urmh_one_result["is_range_superset"]):
                u_cover_rmh_not_cover += 1
            else:
                print(filename)
                # print(filename)
        if urmh_one_result["is_range_superset"]:
            urmh_cover_number += 1

        if(urmh_one_result["is_range_extract_match"]):
            urmh_full_fit += 1
            if(not u_one_result["is_range_extract_match"]):
                urmh_full_fit_u_not += 1
            
        if u_one_result["is_range_extract_match"]:
            u_full_fit += 1

            
    print(urmh_cover_number)
    print(u_cover_number)
    print(u_cover_rmh_not_cover)

    print(u_full_fit)
    print(urmh_full_fit)
    print(urmh_full_fit_u_not)

def re_stat_head_len():
    with open('../PredictDV/tail_len_dic.json', 'r') as f:
        head_len_dic = json.load(f)

    new_dic = {}
    all_num = 0
    for key in head_len_dic:
        all_num += head_len_dic[key]

    for key in head_len_dic:
        if key in ["0","1","2"]:
            new_dic[key] = head_len_dic[key]
        else:
            if -1 not in new_dic:
                new_dic[-1] = 0
            new_dic[-1] += head_len_dic[key]

    print(new_dic)


def re_stat_evalutaion():
    file_list = os.listdir("../PredictDV/evaluates1/delete_o_strip_relax_eval/")
    file_list_1 = os.listdir("../PredictDV/evaluates1/delete_o_strip_relax_eval/")
    file_list.sort()
    all_result = 0
    full_hit_number = 0
    cover_hit_number = 0
    content_hit_number = 0
    content_superset_number = 0
    content_subset_number = 0
    range_superset_number = 0
    range_subset_number = 0

    candidate_number = {}
    multiple_candidate_number = {}
    cover_candidate_number = {}

    full_hit_number_dic = {}
    range_subset_number_dic = {}
    range_superset_number_dic = {}
    range_subset_candidates_number_dic = {}
    range_superset_candidates_number_dic = {}

    success_min_redundancy = {}
    success_mean_redundancy = []
    success_number_list = {}

    content_extract_match_number_dic = {}
    content_superset_number_dic = {}
    content_subset_number_dic = {}
    content_superset_candidates_number_dic = {}
    content_subset_candidates_number_dic = {}
    content_hit_number_dic = {}
    hard_code_number = 0
    suc_div_cn = []
    fail_number = 0


    filenamelist = os.listdir("../PredictDV/evaluates1/delete_o_strip_relax_eval/")
    filenamelist1 = os.listdir("../PredictDV/evaluates1/delete_o_strip_relax_eval")
    c = 0
    hard_code_has_range = 0
    for filename in file_list:
        with open("../PredictDV/evaluates1/delete_o_strip_relax_eval/"+filename,'r', encoding='UTF-8') as f:
            one_result = json.load(f)
        # with open("../test-table-understanding/test-table-understanding/data/final_boundary_orientation_r097_1000_batches/"+filename,'r', encoding='UTF-8') as f:
        #     cand_result = json.load(f)

        if filename not in filenamelist or filename not in filenamelist1:
            continue
        with open("../PredictDV/evaluates1/delete_o_strip_relax_eval/"+filename,'r', encoding='UTF-8') as f:
            delete_o_relax_strip_one_result = json.load(f)
      
        all_result += 1
        if "," in one_result["refer_value"]:
            hard_code_number += 1
            if(len(one_result["cover_number_list"]) > 0):
                hard_code_has_range += 1
                print(filename)
            continue

        
        if(one_result["is_range_extract_match"]):
            full_hit_number += 1
            if one_result["candidates_number"] not in full_hit_number_dic:
                full_hit_number_dic[one_result["candidates_number"]] = 0
            full_hit_number_dic[one_result["candidates_number"]] += 1
        # else:
        #     if delete_o_relax_strip_one_result["is_range_extract_match"]:
        #         print(filename)
        # else:
        #     if delete_o_relax_strip_one_result["is_range_extract_match"]:
        #         # if cand_result[0]["sheet_name"] not in ['Org structure', 'Parameters', 'Prop Time']:
        #         print(filename)
        #         print(cand_result[0]["sheet_name"])
        
        if(one_result["is_range_superset"]):
            cover_hit_number += 1
            if one_result["candidates_number"] not in cover_candidate_number:
                cover_candidate_number[one_result["candidates_number"]] = 0
            cover_candidate_number[one_result["candidates_number"]] += 1
            # if one_result["candidates_number"] < delete_o_relax_strip_one_result["candidates_number"]:
            #     print(filename)
            # if one_result["refer_first_row"] == one_result["refer_last_row"]:
            #     print(filename)
            # if not delete_o_relax_strip_one_result["is_range_superset"]:
            #     print(filename)
        else:
            # if delete_o_relax_strip_one_result["refer_first_row"] == delete_o_relax_strip_one_result["refer_last_row"]:
            #     print(filename)
            if delete_o_relax_strip_one_result["is_range_superset"]:
                # print(filename)
                c+=1
            # if one_result["candidates_number"] < delete_o_relax_strip_one_result["candidates_number"]:
            #     print(filename)
       
            # if delete_o_relax_strip_one_result["candidates_number"] > 1:
            #     print(filename)
        # else:
        #     if delete_o_relax_strip_one_result["is_range_superset"]:
        #         print(filename)
            # print(filename)
        # else:
        #     print(filename)
            # print(cand_result[0]["sheet_name"])


            
        if(one_result["is_content_extract_match"]):
            content_hit_number += 1
            if(one_result["candidates_number"] not in content_hit_number_dic):
                content_hit_number_dic[one_result["candidates_number"]] = 0
            content_hit_number_dic[one_result["candidates_number"]] += 1

        if(one_result["content_superset_number"] > 0):
            content_superset_number += 1
            if(one_result["candidates_number"] not in content_superset_candidates_number_dic):
                content_superset_candidates_number_dic[one_result["candidates_number"]] = 0
            content_superset_candidates_number_dic[one_result["candidates_number"]] += 1
        

        if(one_result["content_subset_number"] > 0):
            content_subset_number += 1
            if(one_result["candidates_number"] not in content_subset_candidates_number_dic):
                content_subset_candidates_number_dic[one_result["candidates_number"]] = 0
            content_subset_candidates_number_dic[one_result["candidates_number"]] += 1

        if(one_result["range_superset_number"]>0):
            range_superset_number += 1
            if(one_result["candidates_number"] not in range_superset_candidates_number_dic):
                range_superset_candidates_number_dic[one_result["candidates_number"]] = 0
            range_superset_candidates_number_dic[one_result["candidates_number"]] += 1
            # if one_result["candidates_number"] > 0:
            #     print(filename)
        # else:
            # if delete_o_relax_strip_one_result["range_superset_number"] > 0:
            #     print(filename)
            # if candidate_number > 0:
            #     print(filename)

        if(one_result["range_subset_number"] > 0):
            range_subset_number += 1
            if(one_result["candidates_number"] not in range_subset_candidates_number_dic):
                range_subset_candidates_number_dic[one_result["candidates_number"]] = 0
            range_subset_candidates_number_dic[one_result["candidates_number"]] += 1


        if(one_result["content_extract_match_number"] not in content_extract_match_number_dic):
            content_extract_match_number_dic[one_result["content_extract_match_number"]] = 0
        content_extract_match_number_dic[one_result["content_extract_match_number"]] += 1


        if one_result["candidates_number"] not in candidate_number:
            candidate_number[one_result["candidates_number"]] = 0
        candidate_number[one_result["candidates_number"]] += 1
        



        if one_result["content_superset_number"] not in content_superset_number_dic:
            content_superset_number_dic[one_result["content_superset_number"]] = 0
        content_superset_number_dic[one_result["content_superset_number"]] += 1


        if one_result["content_subset_number"] not in content_subset_number_dic:
            content_subset_number_dic[one_result["content_subset_number"]] = 0
        content_subset_number_dic[one_result["content_subset_number"]] += 1

        if one_result["range_superset_number"] not in range_superset_number_dic:
            range_superset_number_dic[one_result["range_superset_number"]] = 0
        range_superset_number_dic[one_result["range_superset_number"]] += 1

        if one_result["range_subset_number"] not in range_subset_number_dic:
            range_subset_number_dic[one_result["range_subset_number"]] = 0
        range_subset_number_dic[one_result["range_subset_number"]] += 1

        if one_result["multiple_cells_candidates_number"] not in multiple_candidate_number:
            multiple_candidate_number[one_result["multiple_cells_candidates_number"]] = 0
        multiple_candidate_number[one_result["multiple_cells_candidates_number"]] += 1

 
    print(c)
    print(all_result)
    print("hard_code_number:", hard_code_number)
    print(all_result - hard_code_number)
    print("range exact:",full_hit_number/(all_result-hard_code_number))
    print("range superset:",cover_hit_number/(all_result-hard_code_number))
    print("content exact:",content_hit_number/(all_result-hard_code_number))
    print(range_superset_number)
    print(full_hit_number)
    print("range subset:", range_subset_number/(all_result-hard_code_number))
    print("content subset:", content_subset_number/(all_result-hard_code_number))
    print("content superset:", content_superset_number/(all_result-hard_code_number))
    print("hard code has range:", hard_code_has_range)


    # plt.scatter(full_hit_number_dic.keys(), full_hit_number_dic.values())
    # x = list(full_hit_number_dic.keys())
    # y = list(full_hit_number_dic.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_full_hit_number_dic.png")
    # plt.cla()

    # plt.scatter(candidate_number.keys(), candidate_number.values())
    # x = list(candidate_number.keys())
    # y = list(candidate_number.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_candidate_number.png")
    # plt.cla()

    # plt.scatter(content_extract_match_number_dic.keys(), content_extract_match_number_dic.values())
    # x = list(content_extract_match_number_dic.keys())
    # y = list(content_extract_match_number_dic.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_content_extract_match_number_dic.png")
    # plt.cla()

    # plt.scatter(content_superset_number_dic.keys(), content_superset_number_dic.values())
    # x = list(content_superset_number_dic.keys())
    # y = list(content_superset_number_dic.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_content_superset_number_dic.png")
    # plt.cla()

    # plt.scatter(range_superset_number_dic.keys(), range_superset_number_dic.values())
    # x = list(range_superset_number_dic.keys())
    # y = list(range_superset_number_dic.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_range_superset_number_dic.png")
    # plt.cla()

    # plt.scatter(content_subset_number_dic.keys(), content_subset_number_dic.values())
    # x = list(content_subset_number_dic.keys())
    # y = list(content_subset_number_dic.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_content_subset_number_dic.png")
    # plt.cla()

    # plt.scatter(range_subset_number_dic.keys(), range_subset_number_dic.values())
    # x = list(range_subset_number_dic.keys())
    # y = list(range_subset_number_dic.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_range_subset_number_dic.png")
    # plt.cla()
    # #print(candidate_number)


    # plt.scatter(multiple_candidate_number.keys(), multiple_candidate_number.values())
    # x = list(multiple_candidate_number.keys())
    # y = list(multiple_candidate_number.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_multiple_candidate_number.png")
    # plt.cla()

    # plt.scatter(content_hit_number_dic.keys(), content_hit_number_dic.values())
    # x = list(content_hit_number_dic.keys())
    # y = list(content_hit_number_dic.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_content_hit_number_dic.png")
    # plt.cla()

    # plt.scatter(content_superset_candidates_number_dic.keys(), content_superset_candidates_number_dic.values())
    # x = list(content_superset_candidates_number_dic.keys())
    # y = list(content_superset_candidates_number_dic.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_content_superset_candidates_number_dic.png")
    # plt.cla()

    # plt.scatter(range_superset_candidates_number_dic.keys(), range_superset_candidates_number_dic.values())
    # x = list(range_superset_candidates_number_dic.keys())
    # y = list(range_superset_candidates_number_dic.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_range_superset_candidates_number_dic.png")
    # plt.cla()

    # plt.scatter(content_subset_candidates_number_dic.keys(), content_subset_candidates_number_dic.values())
    # x = list(content_subset_candidates_number_dic.keys())
    # y = list(content_subset_candidates_number_dic.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_content_subset_candidates_number_dic.png")
    # plt.cla()

    # plt.scatter(range_subset_candidates_number_dic.keys(), range_subset_candidates_number_dic.values())
    # x = list(range_subset_candidates_number_dic.keys())
    # y = list(range_subset_candidates_number_dic.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_range_subset_candidates_number_dic.png")
    # plt.cla()
    #print(multiple_candidate_number)

    # plt.scatter(cover_candidate_number.keys(), cover_candidate_number.values())
    # x = list(cover_candidate_number.keys())
    # y = list(cover_candidate_number.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # # plt.xlim(0,60)
    # plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_cover_candidate_number.png")
    # plt.cla()
    # print(cover_candidate_number)

def delete_o_relax_strip_stat_evaluation():
    file_list = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval/")
    file_list.sort()
    all_result = 0
    full_hit_number = 0
    cover_hit_number = 0
    content_hit_number = 0


    candidate_number = {}
    multiple_candidate_number = {}
    cover_candidate_number = {}

    success_min_redundancy = {}
    success_mean_redundancy = []
    success_number_list = {}

    content_hit_number_dic = {}
    hard_code_number = 0
    suc_div_cn = []
    fail_number = 0


    filenamelist = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval/")
    filenamelist1 = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval")
    c = 0
    for filename in file_list:
        with open("../PredictDV/evaluates1/delete_o_relax_strip_eval/"+filename,'r', encoding='UTF-8') as f:
            one_result = json.load(f)
        if filename not in filenamelist or filename not in filenamelist1:
            continue
      
        all_result += 1
        if "," in one_result["refer_value"]:
            hard_code_number += 1
            continue

        if(one_result["is_range_extract_match"]):
            full_hit_number += 1
        
        if(one_result["is_range_superset"]):
            cover_hit_number += 1
            if one_result["candidates_number"] not in cover_candidate_number:
                cover_candidate_number[one_result["candidates_number"]] = 0
            cover_candidate_number[one_result["candidates_number"]] += 1
            # print(filename)

            
        if(one_result["is_content_extract_match"]):
            content_hit_number += 1
            if(one_result["candidates_number"] not in content_hit_number_dic):
                content_hit_number_dic[one_result["candidates_number"]] = 0
            content_hit_number_dic[one_result["candidates_number"]] += 1


        if one_result["candidates_number"] not in candidate_number:
            candidate_number[one_result["candidates_number"]] = 0
        candidate_number[one_result["candidates_number"]] += 1

        if one_result["multiple_cells_candidates_number"] not in multiple_candidate_number:
            multiple_candidate_number[one_result["multiple_cells_candidates_number"]] = 0
        multiple_candidate_number[one_result["multiple_cells_candidates_number"]] += 1

 
    print(c)
    print(all_result)
    print(hard_code_number)
    print(all_result - hard_code_number)
    print(full_hit_number/(all_result-hard_code_number))
    print(cover_hit_number/(all_result-hard_code_number))
    print(content_hit_number/(all_result-hard_code_number))

    print(full_hit_number)


    plt.scatter(candidate_number.keys(), candidate_number.values())
    x = list(candidate_number.keys())
    y = list(candidate_number.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_candidate_number.png")
    plt.cla()
    #print(candidate_number)


    plt.scatter(multiple_candidate_number.keys(), multiple_candidate_number.values())
    x = list(multiple_candidate_number.keys())
    y = list(multiple_candidate_number.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_multiple_candidate_number.png")
    plt.cla()

    plt.scatter(content_hit_number_dic.keys(), content_hit_number_dic.values())
    x = list(content_hit_number_dic.keys())
    y = list(content_hit_number_dic.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_content_hit_number_dic.png")
    plt.cla()
    #print(multiple_candidate_number)

    plt.scatter(cover_candidate_number.keys(), cover_candidate_number.values())
    x = list(cover_candidate_number.keys())
    y = list(cover_candidate_number.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_cover_candidate_number.png")
    plt.cla()
    print(cover_candidate_number)


def u_rmh_stat_evalutaion():
    file_list = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval/")
    file_list.sort()
    all_result = 0
    full_hit_number = 0
    cover_hit_number = 0
    content_hit_number = 0


    candidate_number = {}
    multiple_candidate_number = {}
    cover_candidate_number = {}

    success_min_redundancy = {}
    success_mean_redundancy = []
    success_number_list = {}

    content_hit_number_dic = {}
    hard_code_number = 0
    suc_div_cn = []
    fail_number = 0


    filenamelist = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval/")
    filenamelist1 = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval")
    c = 0
    for filename in file_list:
        with open("../PredictDV/evaluates1/delete_o_relax_strip_eval/"+filename,'r', encoding='UTF-8') as f:
            one_result = json.load(f)
        if filename not in filenamelist or filename not in filenamelist1:
            continue
      
        all_result += 1
        if "," in one_result["refer_value"]:
            hard_code_number += 1
            continue

        if(one_result["is_range_extract_match"]):
            full_hit_number += 1
        
        if(one_result["is_range_superset"]):
            cover_hit_number += 1
            if one_result["candidates_number"] not in cover_candidate_number:
                cover_candidate_number[one_result["candidates_number"]] = 0
            cover_candidate_number[one_result["candidates_number"]] += 1
            # print(filename)

            
        if(one_result["is_content_extract_match"]):
            content_hit_number += 1
            if(one_result["candidates_number"] not in content_hit_number_dic):
                content_hit_number_dic[one_result["candidates_number"]] = 0
            content_hit_number_dic[one_result["candidates_number"]] += 1


        if one_result["candidates_number"] not in candidate_number:
            candidate_number[one_result["candidates_number"]] = 0
        candidate_number[one_result["candidates_number"]] += 1

        if one_result["multiple_cells_candidates_number"] not in multiple_candidate_number:
            multiple_candidate_number[one_result["multiple_cells_candidates_number"]] = 0
        multiple_candidate_number[one_result["multiple_cells_candidates_number"]] += 1

 
    print(c)
    print(all_result)
    print(hard_code_number)
    print(all_result - hard_code_number)
    print(full_hit_number/(all_result-hard_code_number))
    print(cover_hit_number/(all_result-hard_code_number))
    print(content_hit_number/(all_result-hard_code_number))

    print(full_hit_number)


    plt.scatter(candidate_number.keys(), candidate_number.values())
    x = list(candidate_number.keys())
    y = list(candidate_number.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_candidate_number.png")
    plt.cla()
    #print(candidate_number)


    plt.scatter(multiple_candidate_number.keys(), multiple_candidate_number.values())
    x = list(multiple_candidate_number.keys())
    y = list(multiple_candidate_number.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_multiple_candidate_number.png")
    plt.cla()

    plt.scatter(content_hit_number_dic.keys(), content_hit_number_dic.values())
    x = list(content_hit_number_dic.keys())
    y = list(content_hit_number_dic.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_content_hit_number_dic.png")
    plt.cla()
    #print(multiple_candidate_number)

    plt.scatter(cover_candidate_number.keys(), cover_candidate_number.values())
    x = list(cover_candidate_number.keys())
    y = list(cover_candidate_number.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_cover_candidate_number.png")
    plt.cla()
    print(cover_candidate_number)

def delete_o_relax_strip_rmh_stat_evalutaion():
    file_list = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval/")
    file_list.sort()
    all_result = 0
    full_hit_number = 0
    cover_hit_number = 0
    content_hit_number = 0


    candidate_number = {}
    multiple_candidate_number = {}
    cover_candidate_number = {}

    success_min_redundancy = {}
    success_mean_redundancy = []
    success_number_list = {}

    content_hit_number_dic = {}
    hard_code_number = 0
    suc_div_cn = []
    fail_number = 0


    filenamelist = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval/")
    filenamelist1 = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval")
    c = 0
    for filename in file_list:
        with open("../PredictDV/evaluates1/delete_o_relax_strip_eval/"+filename,'r', encoding='UTF-8') as f:
            one_result = json.load(f)
        if filename not in filenamelist or filename not in filenamelist1:
            continue
      
        all_result += 1
        if "," in one_result["refer_value"]:
            hard_code_number += 1
            continue

        if(one_result["is_range_extract_match"]):
            full_hit_number += 1
        
        if(one_result["is_range_superset"]):
            cover_hit_number += 1
            if one_result["candidates_number"] not in cover_candidate_number:
                cover_candidate_number[one_result["candidates_number"]] = 0
            cover_candidate_number[one_result["candidates_number"]] += 1
            # print(filename)

            
        if(one_result["is_content_extract_match"]):
            content_hit_number += 1
            if(one_result["candidates_number"] not in content_hit_number_dic):
                content_hit_number_dic[one_result["candidates_number"]] = 0
            content_hit_number_dic[one_result["candidates_number"]] += 1


        if one_result["candidates_number"] not in candidate_number:
            candidate_number[one_result["candidates_number"]] = 0
        candidate_number[one_result["candidates_number"]] += 1

        if one_result["multiple_cells_candidates_number"] not in multiple_candidate_number:
            multiple_candidate_number[one_result["multiple_cells_candidates_number"]] = 0
        multiple_candidate_number[one_result["multiple_cells_candidates_number"]] += 1

 
    print(c)
    print(all_result)
    print(hard_code_number)
    print(all_result - hard_code_number)
    print(full_hit_number/(all_result-hard_code_number))
    print(cover_hit_number/(all_result-hard_code_number))
    print(content_hit_number/(all_result-hard_code_number))

    print(full_hit_number)


    plt.scatter(candidate_number.keys(), candidate_number.values())
    x = list(candidate_number.keys())
    y = list(candidate_number.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_candidate_number.png")
    plt.cla()
    #print(candidate_number)


    plt.scatter(multiple_candidate_number.keys(), multiple_candidate_number.values())
    x = list(multiple_candidate_number.keys())
    y = list(multiple_candidate_number.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_multiple_candidate_number.png")
    plt.cla()

    plt.scatter(content_hit_number_dic.keys(), content_hit_number_dic.values())
    x = list(content_hit_number_dic.keys())
    y = list(content_hit_number_dic.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_content_hit_number_dic.png")
    plt.cla()
    #print(multiple_candidate_number)

    plt.scatter(cover_candidate_number.keys(), cover_candidate_number.values())
    x = list(cover_candidate_number.keys())
    y = list(cover_candidate_number.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_cover_candidate_number.png")
    plt.cla()
    print(cover_candidate_number)

def check_orientation_2_row_major(threshold):
    file_list = os.listdir("../PredictDV/evaluates1/u_eval/")
    file_list.sort()
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    all_number = 0
    not_column_major = 0
    pred_not_num = 0
    row_number = 0
    for filename in file_list:
        dvid = int(filename.split('.')[0])
        with open("../PredictDV/evaluates1/u_eval/"+filename,'r', encoding='UTF-8') as f:
            one_result = json.load(f)
        with open("../test-table-understanding/test-table-understanding/data/understanding_1000_batches/"+str(dvid)+"_tableinfo.json",'r', encoding='UTF-8') as f:
            tableinfos = json.load(f)
        if len(tableinfos) == 0:
            continue
        if ',' in one_result["refer_value"]:
            continue

        temp = one_result["refer_last_row"]
        one_result["refer_last_row"] = one_result["refer_last_column"]
        one_result["refer_last_column"] = temp

        if one_result["refer_first_column"] != one_result["refer_last_column"] and one_result["refer_first_row"] != one_result["refer_last_row"]:
            continue
    
        if one_result["refer_first_column"] == one_result["refer_last_column"] and one_result["refer_first_row"] == one_result["refer_last_row"]:
            continue



        
        is_found = False
        found_tableinfo = tableinfos[0]
        for tableinfo in tableinfos:
            if tableinfo["left_top_x"] <= one_result["refer_first_row"] and tableinfo["left_top_y"] <= one_result["refer_first_column"] and tableinfo["right_bottom_x"] >= one_result["refer_last_row"] and tableinfo["right_bottom_y"] >= one_result["refer_last_column"]:
                found_tableinfo = tableinfo
                is_found = True
                break
        if not is_found:
            continue
            
        # print(filename)
        if len(found_tableinfo["RowsScore"]) == 0:
            continue 
        # print(type(found_tableinfo["RowsScore"][0]).__name__)
        if type(found_tableinfo["RowsScore"][0]).__name__ == "dict":
            continue

        if found_tableinfo["ColumnsTagsScore"] >= threshold:
            pred_row_major = 1
        else:
            pred_row_major = 0
            # if found_tableinfo["SheetName"] not in ['Org structure']:
            # print(filename)
            # print(found_tableinfo["ColumnsTagsScore"])
            # print(found_tableinfo["SheetName"])
            pred_not_num +=1
                
        data_tag_num = 0
        header_tag_num = 0
        column_tag = "Data" 

        
        if one_result["refer_first_row"] == one_result["refer_last_row"]:
            for one_column in found_tableinfo["RowsScore"]:
                # print(one_column[one_result["refer_first_row"]-tableinfo["left_top_x"]]["Tag"])
                if one_column[one_result["refer_first_row"]-tableinfo["left_top_x"]]["Tag"] == "Header":
                    header_tag_num += 1
                if one_column[one_result["refer_first_row"]-tableinfo["left_top_x"]]["Tag"] == "Data":
                    data_tag_num += 1


            gt_orientation = 'row'

        else:
            for cell in found_tableinfo["RowsScore"][one_result["refer_first_column"]-tableinfo["left_top_y"]]:
                if cell["Tag"] == "Header":
                    header_tag_num += 1
                if cell["Tag"] == "Data":
                    data_tag_num += 1
            gt_orientation = 'column'
        # if gt_orientation == 'row':
        #     print(gt_orientation)
        #     print(header_tag_num, data_tag_num )
        #     print(filename)
        if data_tag_num < header_tag_num:
            column_tag = "Header"

        if found_tableinfo["ColumnsTagsScore"] < 0.5 and found_tableinfo["RowsTagsScore"] < 0.5:
            print(filename)
            print(found_tableinfo["SheetName"])
            print(found_tableinfo["ColumnsTagsScore"])
            print(found_tableinfo["RowsTagsScore"])


        if one_result["refer_first_column"] == one_result["refer_last_column"]: # 如果gt是一个列
            gt_row_major = 0
                
        if one_result["refer_first_column"] != one_result["refer_last_column"]: #如果gt是一个行
            row_number += 1
            if column_tag == "Data":
                gt_row_major = 1
            else:
                gt_row_major = 0
                

        all_number += 1
        
        if pred_row_major == gt_row_major:
            if pred_row_major == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred_row_major == 1:
                fp += 1
            else:
                fn += 1
    print("  c1  c0 label")
    print("p1  "+str(tp)+'  '+str(fp))
    print("p0  "+str(fn)+'  '+str(tn))
    print("pred")
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    print("row_major:", threshold)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', 2*precision*recall/(precision+recall))
    # print('unknown', unknown_number)
    # print('all', all_number)
def check_orientation_2_column_major(threshold):
    file_list = os.listdir("../PredictDV/evaluates1/u_eval/")
    file_list.sort()
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    all_number = 0
    not_column_major = 0
    pred_not_num = 0
    row_number = 0
    for filename in file_list:
        dvid = int(filename.split('.')[0])
        with open("../PredictDV/evaluates1/u_eval/"+filename,'r', encoding='UTF-8') as f:
            one_result = json.load(f)
        with open("../test-table-understanding/test-table-understanding/data/understanding_1000_batches/"+str(dvid)+"_tableinfo.json",'r', encoding='UTF-8') as f:
            tableinfos = json.load(f)
        if len(tableinfos) == 0:
            continue
        if ',' in one_result["refer_value"]:
            continue

        temp = one_result["refer_last_row"]
        one_result["refer_last_row"] = one_result["refer_last_column"]
        one_result["refer_last_column"] = temp

        if one_result["refer_first_column"] != one_result["refer_last_column"] and one_result["refer_first_row"] != one_result["refer_last_row"]:
            continue
    
        if one_result["refer_first_column"] == one_result["refer_last_column"] and one_result["refer_first_row"] == one_result["refer_last_row"]:
            continue



        
        is_found = False
        found_tableinfo = tableinfos[0]
        for tableinfo in tableinfos:
            if tableinfo["left_top_x"] <= one_result["refer_first_row"] and tableinfo["left_top_y"] <= one_result["refer_first_column"] and tableinfo["right_bottom_x"] >= one_result["refer_last_row"] and tableinfo["right_bottom_y"] >= one_result["refer_last_column"]:
                found_tableinfo = tableinfo
                is_found = True
                break
        if not is_found:
            continue
            
        # print(filename)
        if len(found_tableinfo["RowsScore"]) == 0:
            continue 
        # print(type(found_tableinfo["RowsScore"][0]).__name__)
        if type(found_tableinfo["RowsScore"][0]).__name__ == "dict":
            continue

        if found_tableinfo["RowsTagsScore"] >= threshold:
            pred_column_major = 1
        else:
            pred_column_major = 0
            # if found_tableinfo["SheetName"] not in ['Org structure']:
            if found_tableinfo["ColumnsTagsScore"] < 0.5:
                print(filename)
                print(found_tableinfo["RowsTagsScore"])
                print(found_tableinfo["ColumnsTagsScore"])
                print(found_tableinfo["SheetName"])
            pred_not_num +=1
                
        data_tag_num = 0
        header_tag_num = 0
        column_tag = "Data" 

        
        if one_result["refer_first_row"] == one_result["refer_last_row"]:
            for one_column in found_tableinfo["RowsScore"]:
                # print(one_column[one_result["refer_first_row"]-tableinfo["left_top_x"]]["Tag"])
                if one_column[one_result["refer_first_row"]-tableinfo["left_top_x"]]["Tag"] == "Header":
                    header_tag_num += 1
                if one_column[one_result["refer_first_row"]-tableinfo["left_top_x"]]["Tag"] == "Data":
                    data_tag_num += 1


            gt_orientation = 'row'

        else:
            for cell in found_tableinfo["RowsScore"][one_result["refer_first_column"]-tableinfo["left_top_y"]]:
                if cell["Tag"] == "Header":
                    header_tag_num += 1
                if cell["Tag"] == "Data":
                    data_tag_num += 1
            gt_orientation = 'column'
        # if gt_orientation == 'row':
        #     print(gt_orientation)
        #     print(header_tag_num, data_tag_num )
        #     print(filename)
        if data_tag_num < header_tag_num:
            column_tag = "Header"



        if one_result["refer_first_column"] == one_result["refer_last_column"]: # 如果gt是一个列
            gt_column_major = 1
                
        if one_result["refer_first_column"] != one_result["refer_last_column"]: #如果gt是一个行
            row_number += 1
            if column_tag == "Data":
                gt_column_major = 0
            else:
                gt_column_major = 1
                

        all_number += 1
        
        if gt_column_major == 0:
            not_column_major += 1
       
            # if found_tableinfo["SheetName"] not in ['Rectangular Cavity'] and found_tableinfo["FileName"] not in ["../../data/UnzipData/000/032f6d3df2794a7dc4c2a4e922c95e3a_d3d3LmNzLnR1dC5maQkxMzAuMjMwLjEzNy4xOTU=.xls.xlsx"]:
            # print("*****************")
            # print(filename)
            # print(found_tableinfo["RowsTagsScore"])
            # print(found_tableinfo["SheetName"])
            # print(gt_orientation)
        if pred_column_major == gt_column_major:
            if pred_column_major == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred_column_major == 1:
                fp += 1
            else:
                fn += 1
    print("  c1  c0 label")
    print("p1  "+str(tp)+'  '+str(fp))
    print("p0  "+str(fn)+'  '+str(tn))
    print("pred")
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    print("column_major:", threshold)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', 2*precision*recall/(precision+recall))

    print('not_column_major', not_column_major)
    print('row_number', row_number)
    print('pred_not_num', pred_not_num)
    # print('unknown', unknown_number)
    # print('all', all_number)

def check_orientation():
    file_list = os.listdir("../PredictDV/evaluates1/u_eval/")
    file_list.sort()
    # with open("../PredictDV/ListDV/continous_batch_0_1.json",'r', encoding='UTF-8') as f:
    #     listdvinfdelete_o_relax_strip_list = json.load(f)
    orientation_dic = {}

    cm_tp = 0 
    cm_fp = 0
    cm_fn = 0

    rm_tp = 0
    rm_fp = 0
    rm_fn = 0

    b_tp = 0
    b_fp = 0
    b_fn = 0

    cm_cm = 0
    cm_rm = 0
    cm_b = 0
    rm_cm = 0
    rm_rm = 0
    rm_b = 0
    b_cm = 0
    b_rm = 0
    b_b = 0

    unknown_number = 0
    all_number = 0
    for filename in file_list:
        dvid = int(filename.split('.')[0])
        with open("../PredictDV/evaluates1/u_eval/"+filename,'r', encoding='UTF-8') as f:
            one_result = json.load(f)
        with open("../test-table-understanding/test-table-understanding/data/understanding_1000_batches/"+str(dvid)+"_tableinfo.json",'r', encoding='UTF-8') as f:
            tableinfos = json.load(f)
        if len(tableinfos) == 0:
            continue
        if ',' in one_result["refer_value"]:
            continue



        temp = one_result["refer_last_row"]
        one_result["refer_last_row"] = one_result["refer_last_column"]
        one_result["refer_last_column"] = temp

        is_found = False
        found_tableinfo = tableinfos[0]
        for tableinfo in tableinfos:
            if tableinfo["left_top_x"] <= one_result["refer_first_row"] and tableinfo["left_top_y"] <= one_result["refer_first_column"] and tableinfo["right_bottom_x"] >= one_result["refer_last_row"] and tableinfo["right_bottom_y"] >= one_result["refer_last_column"]:
                found_tableinfo = tableinfo
                is_found = True
                break
        if not is_found:
            continue


            
        if found_tableinfo["RowsTagsScore"] >= 0.7 and found_tableinfo["ColumnsTagsScore"] >= 0.8:
            found_tableinfo["Orientation"] = 2
        elif found_tableinfo["RowsTagsScore"] < 0.7 and found_tableinfo["ColumnsTagsScore"] >= 0.8:
            found_tableinfo["Orientation"] = 1
        elif found_tableinfo["RowsTagsScore"] >= 0.7 and found_tableinfo["ColumnsTagsScore"] < 0.8:
            found_tableinfo["Orientation"] = 0
        else:
            
            if found_tableinfo["SheetName"] not in ['Org structure', 'U10m using hooks'] and found_tableinfo["FileName"] not in ["../../data/UnzipData/000/04b771f65311aa8a4fa78f69ebcc6f97_d3d3LnNlYWZpc2gub3JnCTQ2LjI1NS4xMTUuMjUw.xlsx", "../../data/UnzipData/000/032c0d31dc375797ff8ebfb521cff321_d3d3LmxlZS5rMTIubmMudXMJNTIuODYuMTE3LjE5NA==.xls.xlsx"]:
                print(filename)
                print(found_tableinfo["SheetName"])
                print(found_tableinfo["RowsTagsScore"])
                print(found_tableinfo["ColumnsTagsScore"] )
                unknown_number += 1
            found_tableinfo["Orientation"] = 2

        # if found_tableinfo["RowsTagsScore"] - found_tableinfo["ColumnsTagsScore"] > 0.1:
        #     found_tableinfo["Orientation"] = 0
        # elif found_tableinfo["ColumnsTagsScore"] - found_tableinfo["RowsTagsScore"] > 0.4:
        #     found_tableinfo["Orientation"] = 1
        # else:
        #     found_tableinfo["Orientation"] = 2
        all_number += 1
        if one_result["refer_first_column"] == one_result["refer_last_column"] and one_result["refer_first_row"] != one_result["refer_last_row"] and found_tableinfo["Orientation"] == 0: # gt只有一列，并且ts判断方向为0
            cm_tp += 1
            cm_cm += 1
        elif one_result["refer_first_column"] == one_result["refer_last_column"] and one_result["refer_first_row"] != one_result["refer_last_row"] and found_tableinfo["Orientation"] != 0:
            cm_fn += 1
            # print("ColumnTagScores: " + str(found_tableinfo["ColumnsTagsScore"]) + ", RowsTagScores: " + str(found_tableinfo["RowsTagsScore"]))
            # if found_tableinfo["RowsTagsScore"] < found_tableinfo["ColumnsTagsScore"]:
                # print(filename)
            # if found_tableinfo["RowsTagsScore"] < 0.7:
            #     print(filename)
            if found_tableinfo["Orientation"] == 1: #  gt只有一列，并且ts判断方向为1
                rm_fp += 1
                cm_rm += 1
                # print(filename)
            if found_tableinfo["Orientation"] == 2:
                b_fp += 1
                cm_b += 1
        else: #gt不止一列
            if one_result["refer_first_row"] == one_result["refer_last_row"] and one_result["refer_first_column"] != one_result["refer_last_column"] and found_tableinfo["Orientation"] == 1: # gt只有一行，并且ts判断为1
                rm_tp += 1
                rm_rm += 1
                
            elif one_result["refer_first_row"] == one_result["refer_last_row"] and one_result["refer_first_column"] != one_result["refer_last_column"] and found_tableinfo["Orientation"] != 1:
                rm_fn += 1
                if found_tableinfo["Orientation"] == 0:
                    cm_fp += 1
                    rm_cm += 1
                if found_tableinfo["Orientation"] == 2:
                    b_fp += 1
                    rm_b += 1
            else: #
                if found_tableinfo["Orientation"] == 2:
                    b_tp += 1
                    b_b += 1
                if found_tableinfo["Orientation"] == 0:
                    cm_fp += 1
                    b_fn += 1
                    b_cm += 1
                if found_tableinfo["Orientation"] == 1:
                    rm_fp += 1
                    b_fn += 1
                    b_rm += 1

        if found_tableinfo["Orientation"] not in orientation_dic:
            orientation_dic[found_tableinfo["Orientation"]] = 0
        orientation_dic[found_tableinfo["Orientation"]] += 1
    print("    cm  rm  b label")
    print("cm  "+str(cm_cm)+'  '+str(rm_cm)+"  "+str(b_cm))
    print("rm  "+str(cm_rm)+'  '+str(rm_rm)+"  "+str(b_rm))
    print("b  "+str(cm_b)+'  '+str(rm_b)+"  "+str(b_b))
    print("pred")
    
    # print('cm_tp', cm_tp)
    # print('cm_fp', cm_fp)
    # print('cm_fn', cm_fn)
    # print('rm_tp', rm_tp)
    # print('rm_fp', rm_fp)
    # print('rm_fn', rm_fn)
    # print('b_tp', b_tp)
    # print('b_fp', b_fp)
    # print('b_fn', b_fn)
        
    micrdelete_o_relax_strip_precision = (cm_tp + rm_tp + b_tp) / (cm_tp + cm_fp + rm_tp + rm_fp + b_tp + b_fp)
    micrdelete_o_relax_strip_recall = (cm_tp + rm_tp + b_tp) / (cm_tp + cm_fn + rm_tp + rm_fn  + b_tp + b_fn)
    micrdelete_o_relax_strip_f1 = 2*micrdelete_o_relax_strip_precision*micrdelete_o_relax_strip_recall / (micrdelete_o_relax_strip_precision + micrdelete_o_relax_strip_recall)

    print('micrdelete_o_relax_strip_precision', micrdelete_o_relax_strip_precision)
    print('micrdelete_o_relax_strip_recall', micrdelete_o_relax_strip_recall)
    print('micrdelete_o_relax_strip_f1', micrdelete_o_relax_strip_f1)

    cm_precision = cm_tp/ (cm_tp + cm_fp)
    
    b_precision = b_tp/ (b_tp + b_fp)
    cm_recall = cm_tp/ (cm_tp + cm_fn)
    try:
        rm_precision = rm_tp/ (rm_tp + rm_fp)
    except:
        pass
    rm_recall = rm_tp/ (rm_tp + rm_fn)
    b_recall = b_tp/ (b_tp + b_fn)


    print('cm_precision', cm_precision)
    print('cm_recall', cm_recall)
    print('cm_f1', 2*cm_precision*cm_recall/(cm_precision + cm_recall))

    try:
        print('rm_precision', rm_precision)
    except:
        pass
    print('rm_recall', rm_recall)
    try:
        print('rm_f1', 2*rm_precision*rm_recall/(rm_precision + rm_recall))
    except:
        pass
    print('b_precision', b_precision)
    print('b_recall', b_recall)
    try:
        print('b_f1', 2*b_precision*b_recall/(b_precision + b_recall))
    except:
        print("b_f1", 0)

    print('unknown', unknown_number)
    print('all', all_number)
        # if one_result["refer_first_column"] == one_result["refer_last_column"]:
            
        #     print(found_tableinfo["Orientation"])
        #     print("ColumnTagScores: " + str(found_tableinfo["ColumnsTagsScore"]) + ", RowsTagScores: " + str(found_tableinfo["RowsTagsScore"]))
        


        # print(orientation_dic)

            
        # if one_result['is_range_superset'] and found_tableinfo["Orientation"] == 2:
        #     print(dvid)
def delete_o_relax_strip_stat_evalutaion():
    file_list = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval/")
    file_list.sort()
    all_result = 0
    full_hit_number = 0
    cover_hit_number = 0
    content_hit_number = 0


    candidate_number = {}
    multiple_candidate_number = {}
    cover_candidate_number = {}

    success_min_redundancy = {}
    success_mean_redundancy = []
    success_number_list = {}

    content_hit_number_dic = {}
    hard_code_number = 0
    suc_div_cn = []
    fail_number = 0


    filenamelist = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval/")
    filenamelist1 = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval")
    c = 0
    for filename in file_list:
        with open("../PredictDV/evaluates1/delete_o_relax_strip_eval/"+filename,'r', encoding='UTF-8') as f:
            one_result = json.load(f)
        if filename not in filenamelist or filename not in filenamelist1:
            continue
      
        all_result += 1
        if "," in one_result["refer_value"]:
            hard_code_number += 1
            continue

        if(one_result["is_range_extract_match"]):
            full_hit_number += 1
        
        if(one_result["is_range_superset"]):
            cover_hit_number += 1
            if one_result["candidates_number"] not in cover_candidate_number:
                cover_candidate_number[one_result["candidates_number"]] = 0
            cover_candidate_number[one_result["candidates_number"]] += 1
            # print(filename)

            
        if(one_result["is_content_extract_match"]):
            content_hit_number += 1
            if(one_result["candidates_number"] not in content_hit_number_dic):
                content_hit_number_dic[one_result["candidates_number"]] = 0
            content_hit_number_dic[one_result["candidates_number"]] += 1


        if one_result["candidates_number"] not in candidate_number:
            candidate_number[one_result["candidates_number"]] = 0
        candidate_number[one_result["candidates_number"]] += 1

        if one_result["multiple_cells_candidates_number"] not in multiple_candidate_number:
            multiple_candidate_number[one_result["multiple_cells_candidates_number"]] = 0
        multiple_candidate_number[one_result["multiple_cells_candidates_number"]] += 1

 
    print(c)
    print(all_result)
    print(hard_code_number)
    print(all_result - hard_code_number)
    print(full_hit_number/(all_result-hard_code_number))
    print(cover_hit_number/(all_result-hard_code_number))
    print(content_hit_number/(all_result-hard_code_number))

    print(full_hit_number)


    plt.scatter(candidate_number.keys(), candidate_number.values())
    x = list(candidate_number.keys())
    y = list(candidate_number.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_candidate_number.png")
    plt.cla()
    #print(candidate_number)


    plt.scatter(multiple_candidate_number.keys(), multiple_candidate_number.values())
    x = list(multiple_candidate_number.keys())
    y = list(multiple_candidate_number.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_multiple_candidate_number.png")
    plt.cla()

    plt.scatter(content_hit_number_dic.keys(), content_hit_number_dic.values())
    x = list(content_hit_number_dic.keys())
    y = list(content_hit_number_dic.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_content_hit_number_dic.png")
    plt.cla()
    #print(multiple_candidate_number)

    plt.scatter(cover_candidate_number.keys(), cover_candidate_number.values())
    x = list(cover_candidate_number.keys())
    y = list(cover_candidate_number.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    plt.savefig("../PredictDV/evaluates1/delete_o_relax_strip_eval_cover_candidate_number.png")
    plt.cla()
    print(cover_candidate_number)

def stat_evalutaion():
    file_list = os.listdir("../PredictDV/evaluation_baseline/")
    file_list.sort()
    all_result = 0
    full_hit_number = 0
    cover_hit_number = 0
    content_hit_number = 0


    candidate_number = {}
    multiple_candidate_number = {}
    success_min_redundancy = {}
    success_mean_redundancy = []
    success_number_list = {}
    hard_code_number = 0
    suc_div_cn = []
    fail_number = 0

    filenamelist = os.listdir("../PredictDV/delete_o_relax_strip_eval/")
    filenamelist.sort()
    for filename in file_list:
        with open("../PredictDV/delete_o_relax_strip_eval/"+filename,'r', encoding='UTF-8') as f:
            one_result = json.load(f)
        if filename not in filenamelist:
            continue

        all_result += 1
        if "," in one_result["refer_value"]:
            hard_code_number += 1
            continue
        # if one_result["full_check_sucess"] == True:
        #     full_success_number += 1
        # else:
        #     # print(filename)
        if(one_result["full_hit"]):
            full_hit_number += 1
        # else:
        #     continue
        if(one_result["cover_hit"]):
            cover_hit_number += 1
        # else:
            # if(one_result["content_hit"]):
            #     print(filename)
        if one_result["multiple_cells_candidates_number"] >= 3:
            print(filename)
        # else:
        #     continue
            # continue
            
        if(one_result["content_hit"]):
            content_hit_number += 1
        # else:
        #     continue

        if one_result["candidates_number"] not in candidate_number:
            candidate_number[one_result["candidates_number"]] = 0
        candidate_number[one_result["candidates_number"]] += 1

        if one_result["multiple_cells_candidates_number"] not in multiple_candidate_number:
            multiple_candidate_number[one_result["multiple_cells_candidates_number"]] = 0
        multiple_candidate_number[one_result["multiple_cells_candidates_number"]] += 1





        # min_coverage = 10000
        # mean_coverage = 0
        # mean_number = 0
        

        # is_success = False
        # for index, i in enumerate(one_result["is_suc_list"]):
        #     if i == 1:
        #         is_success = True
        #         # print("##########")
        #         # print(one_result["candidate_len"][index])
        #         # print(one_result["refer_len"])
        #         if one_result["candidate_len"][index] - one_result["refer_len"] < min_coverage:
        #             min_coverage = one_result["candidate_len"][index] - one_result["refer_len"]
                
        #         mean_coverage += one_result["candidate_len"][index] - one_result["refer_len"]
        #         mean_number += 1
    
        # if(is_success):
        #     success_mean_redundancy.append(mean_coverage/mean_number)
        #     if(min_coverage not in success_min_redundancy):
        #         success_min_redundancy[min_coverage] = 0
        #     success_min_redundancy[min_coverage] += 1
        # else:
        #     fail_number += 1
            # print("fail: "+ filename)
        # if one_result["full_check_sucess"] == False and min_coverage == 0:
        #     print(filename)
        
    
        # if(one_result["success_number"] not in success_number_list):
        #     success_number_list[one_result["success_number"]] = 0
        # success_number_list[one_result["success_number"]] += 1

        # suc_div_cn.append(one_result["success_number"]/one_result["candidate_number"])
 
    print(all_result)
    print(hard_code_number)
    print(all_result - hard_code_number)
    print(full_hit_number/(all_result-hard_code_number))
    print(cover_hit_number/(all_result-hard_code_number))
    print(content_hit_number/(all_result-hard_code_number))

    print(full_hit_number)
    # pprint.pprint(boundary_gt)
    # plt.plot(suc_div_cn)
    # plt.cla()

    # plt.scatter(success_min_redundancy.keys(), success_min_redundancy.values())
    # # for a,b in zip(success_min_redundancy.keys(), success_min_redundancy.values()):
    # #     plt.text(a,b,b, ha='center', va='bottom', fontsize=20)
    # x = list(success_min_redundancy.keys())
    # y = list(success_min_redundancy.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,40)
    # plt.savefig("baseline_plot/success_min_redundancy.png")
    # plt.cla()

    # plt.plot(success_mean_redundancy)
    # plt.ylim(0,100)
    # plt.savefig("baseline_plot/success_mean_redundancy.png")
    # plt.cla()
    
    # plt.scatter(success_number_list.keys(), success_number_list.values())
    # for a,b in zip(success_number_list.keys(), success_number_list.values()):
    #     plt.text(a,b,b, ha='center', va='bottom', fontsize=20)
    
    # plt.savefig("baseline_plot/success_number_list.png")
    # plt.cla()

    # plt.scatter(candidate_number.keys(), candidate_number.values())
    # x = list(candidate_number.keys())
    # y = list(candidate_number.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlim(0,60)
    # plt.savefig("baseline_plot/delete_o_relax_strip_eval_candidate_number.png")
    # plt.cla()
    # print(candidate_number)


    # plt.scatter(multiple_candidate_number.keys(), multiple_candidate_number.values())
    # x = list(multiple_candidate_number.keys())
    # y = list(multiple_candidate_number.values())
    # for i in range(len(x)):
    #     plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # # plt.xlim(0,60)
    # plt.savefig("baseline_plot/delete_o_relax_strip_eval_multiple_candidate_number.png")
    # plt.cla()
    # print(multiple_candidate_number)

def single_column_row():
    file_list = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval/")
    filenamelist = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval/")
    filenamelist1 = os.listdir("../PredictDV/evaluates1/delete_o_relax_strip_eval")
    all_result_number = 0
    range_list_number = 0

    cover_and_intable = 0
    cover_not_intable = 0
    not_cover_and_intable = 0
    not_cover_and_not_intable = 0

    cover = 0
    not_cover = 0
    intable = 0
    not_intable = 0

    has_head = 0
    not_has_head = 0
    has_tail = 0
    not_has_tail = 0
    cover_and_hashead = 0
    cover_not_hashead = 0
    cover_and_hastail = 0
    cover_not_hastail = 0

    not_cover_and_hashead = 0
    not_cover_not_hashead = 0
    not_cover_and_hastail = 0
    not_cover_not_hastail = 0

    full_hit_and_hashead = 0
    full_hit_and_hastail = 0
    full_hit_not_hashead = 0
    full_hit_not_hastail = 0

    cover_and_not_hashead_hastail_and_not_fullhit = 0
    print(len(file_list))
    with open("../PredictDV/ListDV/continous_batch_0_0.json",'r', encoding='UTF-8') as f:
        dvinfos = json.load(f)
    for filename in file_list:
        with open("../PredictDV/evaluates1/delete_o_relax_strip_eval/"+filename,'r', encoding='UTF-8') as f:
            one_result = json.load(f)
        if filename not in filenamelist or filename not in filenamelist1:
            continue
        is_in_dvinfos = False
        dvinfo = dvinfos[0]
        for i in dvinfos:
            if str(i["ID"])+'.json' == filename:
                is_in_dvinfos = True
                dvinfo = i
                break
        if is_in_dvinfos != True:
            continue
        
        all_result_number += 1
        if "," in one_result["refer_value"]:
            # hard_code_number += 1
            continue
        range_list_number += 1

        if one_result["is_range_extract_match"] == True:
            if dvinfo["GTHasHead"] == True:
                full_hit_and_hashead += 1
               # print(filename)
            else:
                full_hit_not_hashead += 1

            if dvinfo["GTHasTail"] == True:
                full_hit_and_hastail += 1
            else:
                full_hit_not_hastail += 1

        if dvinfo["GTHasHead"] == False and dvinfo["GTHasTail"] == False and one_result["is_range_superset"] == False:
            cover_and_not_hashead_hastail_and_not_fullhit += 1
            print(filename)
        if one_result["is_range_superset"] == True:
            cover += 1
            if dvinfo["GTLocationType"] == 0:
                cover_and_intable += 1
                intable += 1
            else:
                # print(filename)
                cover_not_intable += 1
                not_intable += 1

            if dvinfo["GTHasHead"] == 0:
                cover_and_hashead += 1
                has_head += 1
            else:
                # print(filename)
                cover_not_hashead += 1
                not_has_head += 1
            
            if dvinfo["GTHasTail"] == 0:
                cover_and_hastail += 1
                has_tail += 1
            else:
                # print(filename)
                cover_not_hastail += 1
                not_has_tail += 1
        else:
            not_cover += 1
            if dvinfo["GTLocationType"] == 0:
                not_cover_and_intable += 1
                intable += 1
              #  print(filename)
            else:
                not_cover_and_not_intable += 1
                not_intable += 1

            if dvinfo["GTHasHead"] == 0:
                not_cover_and_hashead += 1
                has_head += 1
            else:
                # print(filename)
                not_cover_not_hashead += 1
                not_has_head += 1
            
            if dvinfo["GTHasTail"] == 0:
                not_cover_and_hastail += 1
                has_tail += 1
            else:
                # print(filename)
                not_cover_not_hastail += 1
                not_has_tail += 1

        
    print("all_result_number", all_result_number)
    print("range_list_number", range_list_number)
    print("cover", cover)
    print("not_cover", not_cover)
    print("intable", intable)
    print("not_intable", not_intable)
    print("cover_and_intable", cover_and_intable)
    print("cover_not_intable", cover_not_intable)
    print("not_cover_and_intable", not_cover_and_intable)
    print("not_cover_and_not_intable", not_cover_and_not_intable)

    print("has_head", has_head)
    print("not_has_head", not_has_head)
    print("has_tail", has_tail)
    print("not_has_tail", not_has_tail)
    print("cover_and_hashead", cover_and_hashead)
    print("cover_and_hastail", cover_and_hastail)
    print("cover_not_hashead", cover_not_hashead)
    print("cover_not_hastail", cover_not_hastail)
    print("not_cover_and_hashead", not_cover_and_hashead)
    print("not_cover_and_hastail", not_cover_and_hastail)
    print("not_cover_not_hashead", not_cover_not_hashead)
    print("not_cover_not_hastail", not_cover_not_hastail)

    print("full_hit_and_hashead", full_hit_and_hashead)
    print("full_hit_not_hashead", full_hit_not_hashead)
    print("full_hit_and_hastail", full_hit_and_hastail)
    print("full_hit_not_hastail", full_hit_not_hastail)

    print("cover_and_not_hashead_hastail_and_not_fullhit", cover_and_not_hashead_hastail_and_not_fullhit)
    
def get_label(range_relax_num, exact_match=True):
    filelist = os.listdir("../PredictDV/evaluates1/delete_o_strip_relax_"+str(range_relax_num)+"_eval/")
    filelist.sort()
    result = []
    for filename in filelist:
        dvid = int(filename.split('.')[0])

        with open("../PredictDV/evaluates1/delete_o_strip_relax_"+str(range_relax_num)+"_eval/"+filename,'r', encoding='UTF-8') as f:
            one_result = json.load(f)
 
        
        if exact_match:
            for index, label in enumerate(one_result["is_range_extract_match_list"]):
                print(dvid, index)
                result.append({"dvid": dvid, "cand_index": index, "label": label})
            
        else:
            for index, label in enumerate(one_result["is_range_superset_list"]):
                print(dvid, index)
                result.append({"dvid": dvid, "cand_index": index, "label": label})
            
    if exact_match:
        b = json.dumps(result)
        f2 = open('label_exact_match_relax_'+str(range_relax_num)+'_10000.json', 'w')
        f2.write(b)
        f2.close()
    else:
        b = json.dumps(result)
        f2 = open('label_range_superset_10000.json', 'w')
        f2.write(b)
        f2.close()
    

def get_both_label(range_relax_num, global_relax_num):
    if range_relax_num == 0:
        with open("label_exact_match_relax_0_10000.json",'r', encoding='UTF-8') as f:
            range_label = json.load(f)
    if range_relax_num == 1:
        with open("label_exact_match_relax_1_10000.json",'r', encoding='UTF-8') as f:
            range_label = json.load(f)
    if range_relax_num == 2:
        with open("label_exact_match_relax_2_10000.json",'r', encoding='UTF-8') as f:
            range_label = json.load(f)

    new_label = []
    last_id = 0
    last_cand_index = 0
    for label_item in range_label:
        if label_item["dvid"] != last_id and last_id != 0:
            with open("../PredictDV/global_candidate_1/"+str(last_id) + '.json','r', encoding='UTF-8') as f:
                global_cand = json.load(f)
            if range_relax_num == 0:
                with open("../PredictDV/evaluates1/delete_o_strip_relax_0_eval/"+str(last_id) + '.json','r', encoding='UTF-8') as f:
                    one_result = json.load(f)
            if range_relax_num == 1:
                with open("../PredictDV/evaluates1/delete_o_strip_relax_1_eval/"+str(last_id) + '.json','r', encoding='UTF-8') as f:
                    one_result = json.load(f)
            if range_relax_num == 2:
                with open("../PredictDV/evaluates1/delete_o_strip_relax_2_eval/"+str(last_id) + '.json','r', encoding='UTF-8') as f:
                    one_result = json.load(f)
            if "," not in one_result["refer_value"]:
                for global_can in global_cand:
                    last_cand_index += 1
                    new_label.append({"dvid": last_id, "cand_index": last_cand_index, "label": 0})
            else:
                for global_can in global_cand:
                    last_cand_index += 1
                    gt_list = [i.strip() for i in one_result["refer_value"][1:-1].split(',')]
                    # print("########################################")
                    # print(set(global_can["List"]))
                    # print(set(gt_list))
                    if set(global_can["List"]) == set(gt_list):
                        new_label.append({"dvid": last_id, "cand_index": last_cand_index, "label": 1})
                        print("global ground truth", gt_list)
                    else:
                        inner_set = set(global_can["List"]) & set(gt_list)
                        if(len(set(gt_list))-len(inner_set) <= global_relax_num and len(set(global_can["List"]))-len(inner_set) <= global_relax_num): # pred has more xxx candidate than gtlist
                            new_label.append({"dvid": last_id, "cand_index": last_cand_index, "label": 1})
                        else:
                            new_label.append({"dvid": last_id, "cand_index": last_cand_index, "label": 0})
        new_label.append(label_item)
        last_cand_index = label_item["cand_index"]
        last_id = label_item["dvid"]

    b = json.dumps(new_label)
    f2 = open('both_label_range_'+str(range_relax_num) + "_global_" +str(global_relax_num)+'_10000.json', 'w')
    f2.write(b)
    f2.close()

def count_both_label():
    with open("new_label.json",'r', encoding='UTF-8') as f:
        range_label = json.load(f)
    with open("c#_both_features_1000.json",'r', encoding='UTF-8') as f:
            features = json.load(f)
        
    new_label = []
    last_id = 0
    last_cand_index = 0

    range_positive_label_num = 0
    range_negative_label_num = 0
    global_positive_label_num = 0
    global_negative_label_num = 0
    for label_item in range_label:
        for feature_item in features:
            if label_item["dvid"] == feature_item["dvid"] and label_item["cand_index"] == feature_item["cand_index"]:
                if feature_item["type"] == 1:
                    if label_item["label"] == 0:
                        range_negative_label_num += 1
                    else:
                        range_positive_label_num += 1
                else:
                    if label_item["label"] == 0:
                        global_negative_label_num += 1
                    else:
                        global_positive_label_num += 1

    print('range_negative_label_num:', range_negative_label_num)
    print('range_positive_label_num:', range_positive_label_num)
    print('global_negative_label_num:', global_negative_label_num)
    print('global_positive_label_num:', global_positive_label_num)
    
    
    # if is_o:
    #     b = json.dumps(range_suc)
    #     f2 = open('o_range_suc.json', 'w')
    #     f2.write(b)
    #     f2.close()

    #     b = json.dumps(hard_code_suc)
    #     f2 = open('o_hard_code_suc.json', 'w')
    #     f2.write(b)
    #     f2.close()
    # else:
    #     b = json.dumps(range_suc)
    #     f2 = open('o_global_range_suc.json', 'w')
    #     f2.write(b)
    #     f2.close()

    #     b = json.dumps(hard_code_suc)
    #     f2 = open('o_global_hard_code_suc.json', 'w')
    #     f2.write(b)
    #     f2.close()
def get_range_not_fit():
    with open("o_range_suc.json", 'r', encoding='UTF-8') as f:
        o_range_suc = json.load(f)
    with open("o_global_range_suc.json", 'r', encoding='UTF-8') as f:
        o_global_range_suc = json.load(f)
    print(set(o_range_suc)-set(o_global_range_suc))

def get_hard_code_dict():
    file_path = "../PredictDV/ListDV/continous_batch_0_1.json"
    column_row_dic = {}
    single_cell_list = []
        
    with open(file_path, 'r', encoding='UTF-8') as f:
        listdvfindelete_o_relax_strip_list = json.load(f)
    all_num = 0

    random.shuffle(listdvfindelete_o_relax_strip_list)

    hard_code_dvinfo = []
    for i in listdvfindelete_o_relax_strip_list:
        if i["GTOrientationType"] == 0:
            hard_code_dvinfo.append(i)
            continue

    refer_id_dict = {}
    id_number_dict = {}
    id_filename_dict = {}
    id_header_list = {}

    id_ = 0
    for i in listdvfindelete_o_relax_strip_list:
    
        haed_list = []
        for head in i["header"][0:-1]:
            haed_list.append(head["Value"])
        if i["GTOrientationType"] == 0:
            is_in_dict = False
            set_k = set()
            
            value = i["Value"]
            refer_list = [k.strip() for k in value[1:-1].split(',')]
            for k in refer_list:
                set_k.add(k)
            found_id = 0
            for j in refer_id_dict: # []
                if refer_id_dict[j] == set_k:
                    is_in_dict = True
                    found_id = j
                    break
            if is_in_dict:
                id_number_dict[found_id] += 1
                id_filename_dict[found_id].append(i["FileName"])
                
                id_header_list[found_id].append(haed_list)
            else:
                id_ += 1
                refer_id_dict[id_] = set_k
                id_number_dict[id_] = 1
                id_filename_dict[id_] = [i["FileName"]]
                id_header_list[id_] = [haed_list]
                if id_ == 221:
                    print(haed_list)
        # else:
        # if i["GTOrientationType"] != 0:
        #     is_in_dict = False
        #     set_k = set()
           
        #     refer_list = [k["Value"] for k in i["refers"]["List"]]
        #     for k in refer_list:
        #         set_k.add(k)
        #     found_id = 0
        #     for j in refer_id_dict: # []
        #         if refer_id_dict[j] == set_k:
        #             is_in_dict = True
        #             found_id = j
        #             break
        #     if is_in_dict:
        #         id_number_dict[found_id] += 1
        #     else:
        #         id_ += 1
        #         refer_id_dict[id_] = set_k
        #         id_number_dict[id_] = 1
        #         id_filename_dict[id_] = [i["FileName"]]
    # print(refer_id_dict)
    temp_refer_id_dict = {}
    for key in refer_id_dict:
        temp_refer_id_dict[key] = list(refer_id_dict[key])

    # one_number = 0
    # all_number = 0
    # for i in id_number_dict:
    #     if id_number_dict[i] == 1:
    #         one_number += 1
    #     all_number += id_number_dict[i]
    # print(str(one_number) + '/' + str(all_number))


    # b = json.dumps(temp_refer_id_dict)
    # f2 = open('refer_id_dict.json', 'w')
    # f2.write(b)
    # f2.close()

    # b = json.dumps(id_number_dict)
    # f2 = open('id_number_dict.json', 'w')
    # f2.write(b)
    # f2.close()

    # b = json.dumps(id_filename_dict)
    # f2 = open('id_filename_dict.json', 'w')
    # f2.write(b)
    # f2.close()

    b = json.dumps(id_header_list)
    f2 = open('hard_code_id_header_value_dict.json', 'w')
    f2.write(b)
    f2.close()
    # pprint.pprint(refer_id_dict)
    # pprint.pprint(id_number_dict)

def get_hard_code_sheet_name_dict():
    file_path = "../PredictDV/ListDV/continous_batch_0_1.json"

        
    with open(file_path, 'r', encoding='UTF-8') as f:
        listdvfindelete_o_relax_strip_list = json.load(f)

    with open("hard_code_refer_id_dict.json", 'r', encoding='UTF-8') as f:
        refer_id_dict = json.load(f)
    random.shuffle(listdvfindelete_o_relax_strip_list)

    hard_code_dvinfo = []
    for i in listdvfindelete_o_relax_strip_list:
        if i["GTOrientationType"] == 0:
            hard_code_dvinfo.append(i)
            continue

    id_header_list = {}

    id_ = 0
    for i in listdvfindelete_o_relax_strip_list:
        if i["GTOrientationType"] == 0:
            is_in_dict = False
            set_k = set()
            
            value = i["Value"]
            refer_list = [k.strip() for k in value[1:-1].split(',')]
            for k in refer_list:
                set_k.add(k)
            found_id = 0
            for j in refer_id_dict: # []
                # print("################")
                # print(set_k)
                # print(refer_id_dict[j])
                if set(refer_id_dict[j]) == set_k:
                    is_in_dict = True
                    found_id = j
                    break
            # print(set_k)
            # print(is_in_dict)
            if is_in_dict:
                if found_id in id_header_list:
                    id_header_list[found_id].append(i["SheetName"])
                else:
                    id_header_list[found_id] = [i["SheetName"]]


    b = json.dumps(id_header_list)
    f2 = open('hard_code_id_sheetname_value_dict.json', 'w')
    f2.write(b)
    f2.close()

def get_hard_code_haeder_dict():
    file_path = "../PredictDV/ListDV/continous_batch_0_1.json"

        
    with open(file_path, 'r', encoding='UTF-8') as f:
        listdvfindelete_o_relax_strip_list = json.load(f)

    with open("hard_code_refer_id_dict.json", 'r', encoding='UTF-8') as f:
        refer_id_dict = json.load(f)
    random.shuffle(listdvfindelete_o_relax_strip_list)

    hard_code_dvinfo = []
    for i in listdvfindelete_o_relax_strip_list:
        if i["GTOrientationType"] == 0:
            hard_code_dvinfo.append(i)
            continue

    id_header_list = {}

    id_ = 0
    for i in listdvfindelete_o_relax_strip_list:
    
        haed_list = []
        for head in i["header"][0:-1]:
            haed_list.append(head["Value"])
        if i["GTOrientationType"] == 0:
            is_in_dict = False
            set_k = set()
            
            value = i["Value"]
            refer_list = [k.strip() for k in value[1:-1].split(',')]
            for k in refer_list:
                set_k.add(k)
            found_id = 0
            for j in refer_id_dict: # []
                # print("################")
                # print(set_k)
                # print(refer_id_dict[j])
                if set(refer_id_dict[j]) == set_k:
                    is_in_dict = True
                    found_id = j
                    break
            # print(set_k)
            # print(is_in_dict)
            if is_in_dict:
                if found_id in id_header_list:
                    id_header_list[found_id].append(haed_list)
                else:
                    id_header_list[found_id] = [haed_list]
                    if id_ == 221:
                        print(haed_list)


    b = json.dumps(id_header_list)
    f2 = open('hard_code_id_header_value_dict.json', 'w')
    f2.write(b)
    f2.close()
    # pprint.pprint(refer_id_dict)
    # pprint.pprint(id_number_dict)


def get_global_candidate():
    file_path = "../PredictDV/ListDV/continous_batch_0_1.json"
        
    with open(file_path, 'r', encoding='UTF-8') as f:
        listdvfindelete_o_relax_strip_list = json.load(f)
    with open("hard_code_refer_id_dict.json", 'r', encoding='utf-8') as f:
        hard_code_dict = json.load(f)
    with open("hard_code_id_number_dict.json", 'r', encoding='utf-8') as f:
        hard_code_id_number_dict = json.load(f)
    with open("hard_code_id_filename_dict.json", 'r', encoding='utf-8') as f:
        hard_code_id_filename_dict = json.load(f)
    with open("hard_code_id_header_value_dict.json", 'r', encoding='utf-8') as f:
        hard_code_id_haeder_value_dict = json.load(f)
    with open("hard_code_id_sheetname_value_dict.json", 'r', encoding='utf-8') as f:
        hard_code_id_sheetname_value_dict = json.load(f)

    need_find_file_path = "../PredictDV/delete_row_single_o_baseline/"
    need_find_files = os.listdir(need_find_file_path)
    need_find_files.sort()
    id_list = [int(i.split('_')[0]) for i in need_find_files]

    for i in listdvfindelete_o_relax_strip_list:
        if i["ID"] not in id_list:
            continue
        candidate_list = []


        content_list = [k["Value"] for k in i["content"]]

        result_id_list = []
        for refer_id in hard_code_dict: 
            all_in = True
            for k in content_list:
                if str(k) not in hard_code_dict[refer_id]:
                    all_in = False
                    break
            if all_in and len(list(set(hard_code_id_filename_dict[refer_id]) - set([i["FileName"]]))) >= 1:
                result_id_list.append(refer_id)

        for k in result_id_list:
            one_cand = {}
            one_cand["popularity"] = hard_code_id_number_dict[k]
            one_cand["List"] = hard_code_dict[k]
            one_cand["Filename"] = hard_code_id_filename_dict[k]
            one_cand["header_list"] = hard_code_id_haeder_value_dict[k]
            one_cand["sheet_list"] = hard_code_id_sheetname_value_dict[k]
            candidate_list.append(one_cand)

        b = json.dumps(candidate_list)
        f2 = open('../PredictDV/global_candidate_1/'+str(i["ID"]) +'.json', 'w')
        f2.write(b)
        f2.close()
       

        
   # print(hard_code_dvinfo)
def hard_code_baseline():
    file_path = "../PredictDV/ListDV/continous_batch_0_1.json"
        
    with open(file_path, 'r', encoding='UTF-8') as f:
        listdvfindelete_o_relax_strip_list = json.load(f)
    with open("hard_code_refer_id_dict.json", 'r', encoding='utf-8') as f:
        hard_code_dict = json.load(f)
    with open("hard_code_id_number_dict.json", 'r', encoding='utf-8') as f:
        hard_code_id_number_dict = json.load(f)
    with open("hard_code_id_filename_dict.json", 'r', encoding='utf-8') as f:
        hard_code_id_filename_dict = json.load(f)
    # with open("hard_code_id_dvid_dict.json", 'r', encoding='utf-8') as f:
    #     hard_code_id_dvid_dict = json.load(f)
    with open("range_refer_id_dict.json", 'r', encoding='utf-8') as f:
        range_dict = json.load(f)
    with open("range_id_number_dict.json", 'r', encoding='utf-8') as f:
        range_id_number_dict = json.load(f)

    hit = 0
    all_num = 0
    found = 0
    hard_code_dvinfo = []

    hard_code_number = 0
    range_number = 0
    for i in listdvfindelete_o_relax_strip_list:
        if i["GTOrientationType"] == 0:
            hard_code_dvinfo.append(i)
            continue
    for i in hard_code_dvinfo:
        content_list = [k["Value"] for k in i["content"]]

        result_id_list = []
        range_result_id = []
        for refer_id in hard_code_dict: 
            all_in = True
            
            
            for k in content_list:
                if str(k) not in hard_code_dict[refer_id]:
                    all_in = False
                    break
            if all_in and len(list(set(hard_code_id_filename_dict[refer_id]) - set([i["FileName"]]))) >= 1:
            
                result_id_list.append(refer_id)

        for refer_id in range_dict: 
            all_in = True
            
            
            for k in content_list:
                if str(k) not in range_dict[refer_id]:
                    all_in = False
                    break
            if all_in:
                range_result_id.append(refer_id)

        max_type = 0
        hard_code_max_num = 0
        hard_code_max_id = 0
        max_num = 0
        max_id = 0
        gt_refer_list = [k.strip() for k in i["Value"][1:-1].split(',')]

        is_found = False
        for k in result_id_list:
            if set(hard_code_dict[k]) == set(gt_refer_list):
                is_found = True
                # print("#####################")
                # print(i["FileName"])
                # print(set(hard_code_id_filename_dict[k]))
            if hard_code_id_number_dict[k] > hard_code_max_num:
                hard_code_max_num = hard_code_id_number_dict[k]
                hard_code_max_id = k
        # for k in range_result_id:
        #     if set(range_dict[k]) == set(gt_refer_list):
        #         is_found = True
        #     if range_id_number_dict[k] > max_num:
        #         max_num = range_id_number_dict[k]
        #         max_id = k

        if max_num < hard_code_max_num:
            max_num = hard_code_max_num
            max_id = hard_code_max_id
            max_type = 1

        if max_type == 0:
            range_number += 1
        else:
            hard_code_number += 1
        if is_found:
            found += 1

        if max_id != 0:
            if max_type == 0:
                if set(gt_refer_list) == set(range_dict[max_id]):
                    hit += 1
            else:
                if set(gt_refer_list) == set(hard_code_dict[max_id]):
                    hit += 1
            
        all_num += 1
    print(str(hit) + '/' + str(all_num))
    print(found)
    print(range_number)
    print(hard_code_number)

    
    # one_number = 0
    # all_number = 0
    # for i in range_id_number_dict:
    #     if range_id_number_dict[i] == 1:
    #         one_number += 1
    #     all_number += range_id_number_dict[i]
    # print(str(one_number) + '/' + str(all_number))

def sorted_hard_code_dict():
    with open("hard_code_refer_id_dict.json", 'r', encoding='utf-8') as f:
        hard_code_dict = json.load(f)
    with open("hard_code_id_number_dict.json", 'r', encoding='utf-8') as f:
        id_number_dict = json.load(f)
    a = sorted(id_number_dict.items(), key=lambda x: x[1], reverse=True)
    for i in a:
        print(hard_code_dict[i[0]], i[1])
    # print(a)

def get_custom_dict():
    result_dic = {}
    for filename in ["../share/dvinfoWithRef.json", "../share/dvinfoWithRef1.json", "../share/dvinfoWithRef2.json", "../share/dvinfoWithRef3.json"]:
        with open(filename, 'r', encoding='utf-8') as f:
            dvinfos = json.load(f)
        
        for dvinfo in dvinfos:
            if dvinfo["Type"] == 7:
                if dvinfo["Value"] + "     " + dvinfo["SheetName"] not in result_dic:
                    result_dic[dvinfo["Value"] + "     " + dvinfo["SheetName"]] = 0
                result_dic[dvinfo["Value"] + "     " + dvinfo["SheetName"]] += 1
    b = json.dumps(result_dic)
    f2 = open('custom_number_dic_sheet.json', 'w')
    f2.write(b)
    f2.close()

def count_cand():
    with open("c#_both_features_1000_3.json", 'r') as f:
        features = json.load(f)
    with open("../AnalyzeDV/continous_batch_0.json",'r', encoding='UTF-8') as f:
        dvinfos = json.load(f)

    type_1_dict = {}
    type_2_dict = {}
    for feature in features:
        for dvinfo in dvinfos:
            if dvinfo["ID"] == feature["dvid"]:
                if ',' not in dvinfo["Value"]:
                    type_1_dict[feature["dvid"]] = feature["cand_index"]
                else:
                    type_2_dict[feature["dvid"]] = feature["cand_index"]
    b = json.dumps(type_1_dict)
    f2 = open('range_cand_number.json', 'w')
    f2.write(b)
    f2.close()

    b = json.dumps(type_2_dict)
    f2 = open('global_cand_number.json', 'w')
    f2.write(b)
    f2.close()

def plot_cand_number():
    with open("range_cand_number.json", 'r') as f:
        range_cand_number = json.load(f)
    with open("global_cand_number.json",'r', encoding='UTF-8') as f:
        global_cand_number = json.load(f)

    range_number_number_dict = {}
    global_number_number_dict = {}

    for i in range_cand_number:
        if range_cand_number[i] not in range_number_number_dict:
            range_number_number_dict[range_cand_number[i]] = 0
        range_number_number_dict[range_cand_number[i]] += 1

    for i in global_cand_number:
        if global_cand_number[i] not in global_number_number_dict:
            global_number_number_dict[global_cand_number[i]] = 0
        global_number_number_dict[global_cand_number[i]] += 1

    plt.scatter(range_number_number_dict.keys(), range_number_number_dict.values())
    x = list(range_number_number_dict.keys())
    y = list(range_number_number_dict.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    plt.savefig("range_cand_number.png")
    plt.cla()

    plt.scatter(global_number_number_dict.keys(), global_number_number_dict.values())
    x = list(global_number_number_dict.keys())
    y = list(global_number_number_dict.values())
    for i in range(len(x)):
        plt.annotate("("+str(x[i]) + "," + str(y[i])+")", xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    plt.savefig("global_cand_number.png")
    plt.cla()

def get_headre_fail():
    with open("more_with_header_rank_global_fail.json", 'r') as f:
        with_header_range_global_fail = json.load(f)
    with open("more_rank_global_fail.json",'r', encoding='UTF-8') as f:
        range_global_fail = json.load(f)

    search_fail = []
    search_suc = []
    with_header_search_fail = []
    with_header_search_suc = []

    with open("more_header_pred_result.json", 'r') as f:
        pred_result = json.load(f)
    with open("more_with_header_pred_result.json", 'r') as f:
        with_header_pred_result = json.load(f)

    for i in pred_result:
        if 1 not in pred_result[i]["test"]:
            search_fail.append(int(i))
        else:
            search_suc.append(int(i))

    for i in with_header_pred_result:
        if 1 not in with_header_pred_result[i]["test"]:
            with_header_search_fail.append(int(i))
        else:
            with_header_search_suc.append(int(i))

    # print(set(range_global_fail))
    # print(set(with_header_search_suc))
    print(set(with_header_range_global_fail) - set(range_global_fail))
    print(len(set(search_suc) & set(range_global_fail)))
    print(len(set(with_header_search_fail) & set(with_header_range_global_fail)))
    print(len(set(search_suc)))
    print(len(set(search_fail)))
    # print(set(with_header_search_fail) & set(range_global_fail))

def merge_features():
    new_features = []
    add_dvid = {}
    with open("../test-table-understanding/test-table-understanding/bin/debug/c#_both_features_1000_3_01.json",'r', encoding='UTF-8') as f:
        features_3 = json.load(f)
    with open("c#_both_features_1000.json",'r', encoding='UTF-8') as f:
        features_1 = json.load(f)
    for i in features_3:
        if i["dvid"] in add_dvid:
            if i["cand_index"] in add_dvid[i["dvid"]]:
                continue
        if i["dvid"] not in add_dvid:
            add_dvid[i["dvid"]] = []
        add_dvid[i["dvid"]].append(i["cand_index"])
        new_features.append(i)

    for i in features_1:
        if i["dvid"] in add_dvid:
            if i["cand_index"] in add_dvid[i["dvid"]]:
                continue
        if i["dvid"] not in add_dvid:
            add_dvid[i["dvid"]] = []
        add_dvid[i["dvid"]].append(i["cand_index"])
        new_features.append(i)

    b = json.dumps(new_features)
    f2 = open("c#_both_features_1000_4.json", 'w')
    f2.write(b)
    f2.close()
    
def count_row():
    filelist = os.listdir("../PredictDV/delete_row_single_o_baseline")
    feature_number = len(filelist)
    print('feature_number', feature_number)
    with open("both_label_10000.json",'r', encoding='UTF-8') as f:
        labels = json.load(f)
    label = [one_label["dvid"] for one_label in labels]
    filelist1 = os.listdir("../PredictDV/evaluates1/delete_o_strip_relax_eval/")
    print('feature_number', len(filelist1))
    print("labels", len(set(label)))

def get_top_k_placeholder():
    word_freq = {}
    with open("hard_code_refer_id_dict.json", 'r') as f:
        hard_code_refer_id_dict = json.load(f)

    for id in hard_code_refer_id_dict.keys():
        for word in hard_code_refer_id_dict[id]:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
    # print(word_freq)
    
    word_freq_seq=list(word_freq.items())    
    word_freq_seq.sort(key=lambda x:x[1],reverse=False)
    pprint.pprint(word_freq_seq)

def get_max_baseline_recall(precision):
    evaluate_range = 1
    evaluate_global = 1
    root_path = "with_header_sname"
    with open("../test-table-understanding/test-table-understanding/bin/debug/cand_len_dict.json", 'r') as f:
        cand_len_dict = json.load(f)

    with open('both_label_range_'+str(evaluate_range).replace("-","n") + "_global_" +str(evaluate_global)+'_10000.json', 'r', encoding='UTF-8') as f:
        labels = json.load(f)
    test_range_label_dict_1 = {}
    test_range_label_dict_2 = {}
    test_range_label_dict_3 = {}
    test_range_label_dict_4 = {}

    test_global_label_dict_1 = {}
    test_global_label_dict_2 = {}
    test_global_label_dict_3 = {}
    test_global_label_dict_4 = {}

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

    for index, i in enumerate(labels):
        # print(index, len(labels))
        # if i["dvid"] == 31531 or i["dvid"] == '31531':
        #     print(i, i["cand_index"])
        if i["dvid"] in range_feature_dict_1 or i["dvid"] in global_feature_dict_1:
            if i["dvid"] in range_feature_dict_1:
                if i['cand_index'] in range_feature_dict_1[i['dvid']]:
                    if i["dvid"] not in test_range_label_dict_1:
                        test_range_label_dict_1[i["dvid"]] = {}
                    test_range_label_dict_1[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_1:
                if i['cand_index'] in global_feature_dict_1[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_1:
                        test_global_label_dict_1[i["dvid"]] = {}
                    test_global_label_dict_1[i["dvid"]][i["cand_index"]] = i["label"]
        elif i["dvid"] in range_feature_dict_2 or i["dvid"] in global_feature_dict_2:
            if i["dvid"] in range_feature_dict_2:
                if i['cand_index'] in range_feature_dict_2[i['dvid']]:
                    if i["dvid"] == 31531 or i["dvid"] == '31531':
                        print(i, i["cand_index"])
                    if i["dvid"] not in test_range_label_dict_2:
                        test_range_label_dict_2[i["dvid"]] = {}
                    test_range_label_dict_2[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_2:
                if i['cand_index'] in global_feature_dict_2[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_2:
                        test_global_label_dict_2[i["dvid"]] = {}
                    test_global_label_dict_2[i["dvid"]][i["cand_index"]] = i["label"]
        elif i["dvid"] in range_feature_dict_3 or i["dvid"] in global_feature_dict_3:
            if i["dvid"] in range_feature_dict_3:
                if i['cand_index'] in range_feature_dict_3[i['dvid']]:
                    if i["dvid"] not in test_range_label_dict_3:
                        test_range_label_dict_3[i["dvid"]] = {}
                    test_range_label_dict_3[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_3:
                if i['cand_index'] in global_feature_dict_3[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_3:
                        test_global_label_dict_3[i["dvid"]] = {}
                    test_global_label_dict_3[i["dvid"]][i["cand_index"]] = i["label"]
        elif i["dvid"] in range_feature_dict_4 or i["dvid"] in global_feature_dict_4:
            if i["dvid"] in range_feature_dict_4:
                if i['cand_index'] in range_feature_dict_4[i['dvid']]:
                    if i["dvid"] not in test_range_label_dict_4:
                        test_range_label_dict_4[i["dvid"]] = {}
                    test_range_label_dict_4[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_4:
                if i['cand_index'] in global_feature_dict_4[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_4:
                        test_global_label_dict_4[i["dvid"]] = {}
                    test_global_label_dict_4[i["dvid"]][i["cand_index"]] = i["label"]

    max_suc_dvid = []
    min_suc_dvid = []
    # print(type(list(cand_len_dict.keys())[0]))
    label_list = [test_range_label_dict_1,test_range_label_dict_2,test_range_label_dict_3,test_range_label_dict_4,test_global_label_dict_1,test_global_label_dict_2,test_global_label_dict_3,test_global_label_dict_4]
    for index in range(4):
        test_range_label_dict = label_list[index]
        test_global_label_dict = label_list[index+4]
        for dvid in test_range_label_dict:
            gt_index = []
            for cand_index in test_range_label_dict[int(dvid)]:
                if test_range_label_dict[int(dvid)][cand_index] == 1:
                    gt_index.append(int(cand_index))

            if dvid in test_global_label_dict:
                for cand_index in test_global_label_dict[int(dvid)]:
                    if test_global_label_dict[int(dvid)][cand_index] == 1:
                        gt_index.append(int(cand_index))
            
            max_number = 0
            max_index = 0
            min_number = 10000000
            min_index = 0
            max_dict = {}
            for cand_index in cand_len_dict[str(dvid)]:
                max_dict[cand_index] = cand_len_dict[str(dvid)][cand_index]

            sorted_dict = sorted(max_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
            print(sorted_dict)
            max_index_list = []
            min_index_list = []
            for ind in range(0,min(len(sorted_dict),precision)):
                max_index_list.append(int(sorted_dict[ind][0]))
            ind = len(sorted_dict)-1
            while(ind>= max(len(sorted_dict)-precision, 0)):
                min_index_list.append(int(sorted_dict[ind][0]))
                ind -= 1
         
            if len(set(max_index_list) & set(gt_index)) > 0:
                max_suc_dvid.append(int(dvid))
            if len(set(min_index_list) & set(gt_index)) > 0:
                min_suc_dvid.append(int(dvid))
                # if cand_len_dict[str(dvid)][cand_index] > max_number:
                #     max_index = cand_index
                #     max_number = cand_len_dict[str(dvid)][cand_index] 
                # if cand_len_dict[str(dvid)][cand_index] < min_number:
                #     min_indx = cand_index
                #     min_number = cand_len_dict[str(dvid)][cand_index]
            
            print("###")
            print('gt_index', gt_index)
            print("max_index_list", max_index_list)
            print("min_index_list", min_index_list)
    
            max_index = int(max_index)
            min_index = int(min_index)

            # if max_index in gt_index:
            #     max_suc_dvid.append(int(dvid))
            # if min_index in gt_index:
            #     min_suc_dvid.append(int(dvid))
            # break
    np.save('max_suc_dvid_'+str(precision), max_suc_dvid)
    np.save('min_suc_dvid_'+str(precision), min_suc_dvid)

def look_res(root_path="with_header_sname", evaluate_range=1, evaluate_global=1, precision=3, classifier=1,model=1):
    classifier=3
    if classifier in [3,4,5]:
        res_path =  root_path + "/" + "range_"+str(evaluate_range).replace("-","n")+"_global_"+str(evaluate_global)
        # with open(res_path+'/style_merge_range_suc_'+str(precision) +
        #             '_'+str(classifier)+str(model)+'.json') as f:
        #     suc_dvid = json.load(f)
        suc_dvid=np.load('min_suc_dvid_2.npy', allow_pickle=True)
    else:
        res_path = root_path + "/" + "range_"+str(evaluate_range).replace("-","n").replace('-','n')+"_global_"+str(evaluate_global)
        with open(res_path+'/style_merge_range_suc_'+str(precision) +
                    '_'+str(classifier)+str(model)+'.json') as f:
            suc_dvid1 = json.load(f)
        with open(res_path+'/style_merge_global_suc_'+str(precision) +
                    '_'+str(classifier)+str(model)+'.json') as f:
            suc_dvid2 = json.load(f)
        suc_dvid = suc_dvid1+suc_dvid2
    need_all =False
    if need_all:
        all_dvid = np.load('target_content_all_dvid_6.npy', allow_pickle=True)
    with open("../AnalyzeDV/continous_batch_0.json", 'r', encoding='UTF-8') as f:
        dvinfos = json.load(f)
    
    # suc_dvid = list(set(suc_dvid1) | set(suc_dvid2) | set(suc_dvid3) | set(suc_dvid4))
    # print(suc_dvid)
    print('len(suc_dvid', len(suc_dvid))
    # print("len inter suc dvid", len(list(set(suc_dvid1) & set(suc_dvid2))))
    # print("len inter suc dvid", len(list(set(suc_dvid1) & set(suc_dvid3))))
    # print("len inter suc dvid", len(list(set(suc_dvid1) & set(suc_dvid4))))
    # print("len inter suc dvid", len(list(set(suc_dvid2) & set(suc_dvid3))))
    # print("len inter suc dvid", len(list(set(suc_dvid2) & set(suc_dvid4))))
    # print("len inter suc dvid", len(list(set(suc_dvid3) & set(suc_dvid4))))

    # print("len all suc dvid", len(list(set(suc_dvid1) | set(suc_dvid2))))
    # print("len all suc dvid", len(list(set(suc_dvid1) | set(suc_dvid3))))
    # print("len all suc dvid", len(list(set(suc_dvid1) | set(suc_dvid4))))
    # print("len all suc dvid", len(list(set(suc_dvid2) | set(suc_dvid3))))
    # print("len all suc dvid", len(list(set(suc_dvid2) | set(suc_dvid4))))
    # print("len all suc dvid", len(list(set(suc_dvid3) | set(suc_dvid4))))


    
    # print('len all suc dvid', len(list(set(suc_dvid1) | set(suc_dvid2) | set(suc_dvid3) | set(suc_dvid4))))
    global_list = []
    range_list = []
    rank_range_suc = []
    rank_range_fail = []
    rank_global_suc = []
    rank_global_fail = []
    all_ = 0

    test_feature_dict_1 = np.load("with_header_sname" + "/" + "test_feature_dict_1.npy", allow_pickle=True).item()
    test_feature_dict_2 = np.load("with_header_sname" + "/" + "test_feature_dict_2.npy", allow_pickle=True).item()
    test_feature_dict_3 = np.load("with_header_sname" + "/" + "test_feature_dict_3.npy", allow_pickle=True).item()
    test_feature_dict_4 = np.load("with_header_sname" + "/" + "test_feature_dict_4.npy", allow_pickle=True).item()
    if not need_all:
        all_dvid = list(set(test_feature_dict_1) | set(test_feature_dict_2) | set(test_feature_dict_3) | set(test_feature_dict_4))
    for dvid in all_dvid:
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
                    if dvid in suc_dvid:
                        rank_global_suc.append(dvid)
                    else:
                        rank_global_fail.append(dvid)
                else:
                    range_list.append(dvid)
                    if dvid in suc_dvid:
                        rank_range_suc.append(dvid)
                    else:
                        rank_range_fail.append(dvid)

    # print(rank_range_fail)
    classifier=3
    with open(res_path + '/style_pair_rank_pred_result_' + str(precision) + '_' + str(classifier)+'.json', 'r') as f:
        rst = json.load(f)

    g_rank_fail = []
    r_rank_fail = []
    for i in rank_range_fail:
        if str(i) in rst:
            if 1 in rst[str(i)]['test']:
                r_rank_fail.append(i)
    for i in rank_global_fail:
        if str(i) in rst:
            if 1 in rst[str(i)]['test']:
                g_rank_fail.append(i)
    # print(rank_fail)
    print(len(g_rank_fail))
    print(len(r_rank_fail))
    print('len_range_suc', len(rank_range_suc))
    print('len_range_fail', len(rank_range_fail))
    print('len_global_suc', len(rank_global_suc))
    print('len_global_fail', len(rank_global_fail))
    # print(rank_range_fail)
    print('len_global', len(global_list))
    print('len_range', len(range_list))
    

    print("acc_1:", (len(rank_range_suc) + len(rank_global_suc))/(len(global_list) + len(range_list)))
    print("rank_range_acc_1:", len(rank_range_suc) / len(range_list))
    print("rank_global_acc_1:", len(rank_global_suc) / len(global_list))
    print("merge_range_acc_1:", len(rank_range_suc) / len(range_list))
    print("merge_global_acc_1:", len(rank_global_suc) / len(global_list))

def get_range_pop(precision):
    evaluate_range = 1
    evaluate_global = 1
    root_path = "with_header_sname"
    
    with open("../PredictDV/ListDV/continous_batch_0_0.json",'r', encoding='UTF-8') as f:
        dvinfos = json.load(f)
    with open('both_label_range_'+str(evaluate_range).replace("-","n") + "_global_" +str(evaluate_global)+'_10000.json', 'r', encoding='UTF-8') as f:
        labels = json.load(f)
    with open('range_gt_id_dict.json', 'r', encoding='UTF-8') as f:
        id_dict = json.load(f)
    with open('id_index_dict.json', 'r', encoding='UTF-8') as f:
        id_index_dict = json.load(f)
    with open('range_gt_id_number_dict.json', 'r', encoding='UTF-8') as f:
        id_number = json.load(f)
    test_range_label_dict_1 = {}
    test_range_label_dict_2 = {}
    test_range_label_dict_3 = {}
    test_range_label_dict_4 = {}

    test_global_label_dict_1 = {}
    test_global_label_dict_2 = {}
    test_global_label_dict_3 = {}
    test_global_label_dict_4 = {}

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

    for index, i in enumerate(labels):
        # print(index, len(labels))
        # if i["dvid"] == 31531 or i["dvid"] == '31531':
        #     print(i, i["cand_index"])
        if i["dvid"] in range_feature_dict_1 or i["dvid"] in global_feature_dict_1:
            if i["dvid"] in range_feature_dict_1:
                if i['cand_index'] in range_feature_dict_1[i['dvid']]:
                    if i["dvid"] not in test_range_label_dict_1:
                        test_range_label_dict_1[i["dvid"]] = {}
                    test_range_label_dict_1[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_1:
                if i['cand_index'] in global_feature_dict_1[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_1:
                        test_global_label_dict_1[i["dvid"]] = {}
                    test_global_label_dict_1[i["dvid"]][i["cand_index"]] = i["label"]
        elif i["dvid"] in range_feature_dict_2 or i["dvid"] in global_feature_dict_2:
            if i["dvid"] in range_feature_dict_2:
                if i['cand_index'] in range_feature_dict_2[i['dvid']]:
                    if i["dvid"] == 31531 or i["dvid"] == '31531':
                        print(i, i["cand_index"])
                    if i["dvid"] not in test_range_label_dict_2:
                        test_range_label_dict_2[i["dvid"]] = {}
                    test_range_label_dict_2[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_2:
                if i['cand_index'] in global_feature_dict_2[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_2:
                        test_global_label_dict_2[i["dvid"]] = {}
                    test_global_label_dict_2[i["dvid"]][i["cand_index"]] = i["label"]
        elif i["dvid"] in range_feature_dict_3 or i["dvid"] in global_feature_dict_3:
            if i["dvid"] in range_feature_dict_3:
                if i['cand_index'] in range_feature_dict_3[i['dvid']]:
                    if i["dvid"] not in test_range_label_dict_3:
                        test_range_label_dict_3[i["dvid"]] = {}
                    test_range_label_dict_3[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_3:
                if i['cand_index'] in global_feature_dict_3[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_3:
                        test_global_label_dict_3[i["dvid"]] = {}
                    test_global_label_dict_3[i["dvid"]][i["cand_index"]] = i["label"]
        elif i["dvid"] in range_feature_dict_4 or i["dvid"] in global_feature_dict_4:
            if i["dvid"] in range_feature_dict_4:
                if i['cand_index'] in range_feature_dict_4[i['dvid']]:
                    if i["dvid"] not in test_range_label_dict_4:
                        test_range_label_dict_4[i["dvid"]] = {}
                    test_range_label_dict_4[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_4:
                if i['cand_index'] in global_feature_dict_4[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_4:
                        test_global_label_dict_4[i["dvid"]] = {}
                    test_global_label_dict_4[i["dvid"]][i["cand_index"]] = i["label"]

    suc_dvid=[]
    # print(type(list(cand_len_dict.keys())[0]))
    label_list = [test_range_label_dict_1,test_range_label_dict_2,test_range_label_dict_3,test_range_label_dict_4,test_global_label_dict_1,test_global_label_dict_2,test_global_label_dict_3,test_global_label_dict_4]
    for index in range(4):
        test_range_label_dict = label_list[index]
        test_global_label_dict = label_list[index+4]
        for dvid in test_range_label_dict:
            gt_index = []
            for dvinfo in dvinfos:
                if dvinfo["ID"] == dvid:
                    found_dvinfo=dvinfo

            sheet_name = found_dvinfo["SheetName"]
            file_name = found_dvinfo["FileName"]
            kw = file_name+'***********'+sheet_name
            with open("../PredictDV/delete_row_single_o_baseline/" + str(dvid) + "_search_result.json", 'r', encoding="utf-8") as f:
                cand_list = json.load(f)
            with open("../PredictDV/global_candidate_1/" + str(dvid) + ".json", 'r') as f:
                g_cand_list = json.load(f)
            
            for cand_index in test_range_label_dict[int(dvid)]:
                if test_range_label_dict[int(dvid)][cand_index] == 1:
                    gt_index.append(int(cand_index))

            if dvid in test_global_label_dict:
                for cand_index in test_global_label_dict[int(dvid)]:
                    if test_global_label_dict[int(dvid)][cand_index] == 1:
                        gt_index.append(int(cand_index))
            
            pop_suc = []
            max_number = 0
            max_cand_index = {}
            cand_index=0

            for one_cand in cand_list:
                cand_ltx = one_cand['left_top_x']
                cand_lty = one_cand['left_top_y']
                cand_rbx = one_cand['right_bottom_x']
                cand_rby = one_cand['right_bottom_y']
                pop = 0
                if kw in id_index_dict:
                    for id_ in id_index_dict[kw]:
                        if id_index_dict[kw][id_]['ltx'] == cand_ltx and id_index_dict[kw][id_]['lty'] == cand_lty and id_index_dict[kw][id_]['rbx'] == cand_rbx and id_index_dict[kw][id_]['rby'] == cand_rby:
                            pop = id_number[kw][id_]
                            print('found')
                max_cand_index[cand_index] = pop
                # if pop > max_number:
                #     max_number = pop
                #     max_cand_index = cand_index
                cand_index+=1
            for one_cand in g_cand_list:
                max_cand_index[cand_index] = one_cand["popularity"] 
                # if max_number < one_cand["popularity"]:
                #     max_number = one_cand["popularity"]
                #     max_cand_index = cand_index
                cand_index+=1

            sorted_dict = sorted(max_cand_index.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
            print(sorted_dict)
            max_index_list = []
            for ind in range(0,min(len(sorted_dict),precision)):
                max_index_list.append(sorted_dict[ind][0])

            print("###")
            print('gt_index', gt_index)
            print("max_index", max_cand_index)

            if len(set(max_index_list)&set(gt_index))>=1:
                suc_dvid.append(int(dvid))
    
    np.save('pop_suc_dvid_'+str(precision), suc_dvid)

def get_random_baseline(precision):
    evaluate_range = 1
    evaluate_global = 1
    root_path = "with_header_sname"
    
    with open("../PredictDV/ListDV/continous_batch_0_0.json",'r', encoding='UTF-8') as f:
        dvinfos = json.load(f)
    with open('both_label_range_'+str(evaluate_range).replace("-","n") + "_global_" +str(evaluate_global)+'_10000.json', 'r', encoding='UTF-8') as f:
        labels = json.load(f)
    with open('range_gt_id_dict.json', 'r', encoding='UTF-8') as f:
        id_dict = json.load(f)
    with open('id_index_dict.json', 'r', encoding='UTF-8') as f:
        id_index_dict = json.load(f)
    with open('range_gt_id_number_dict.json', 'r', encoding='UTF-8') as f:
        id_number = json.load(f)
    test_range_label_dict_1 = {}
    test_range_label_dict_2 = {}
    test_range_label_dict_3 = {}
    test_range_label_dict_4 = {}

    test_global_label_dict_1 = {}
    test_global_label_dict_2 = {}
    test_global_label_dict_3 = {}
    test_global_label_dict_4 = {}

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

    for index, i in enumerate(labels):
        # print(index, len(labels))
        # if i["dvid"] == 31531 or i["dvid"] == '31531':
        #     print(i, i["cand_index"])
        if i["dvid"] in range_feature_dict_1 or i["dvid"] in global_feature_dict_1:
            if i["dvid"] in range_feature_dict_1:
                if i['cand_index'] in range_feature_dict_1[i['dvid']]:
                    if i["dvid"] not in test_range_label_dict_1:
                        test_range_label_dict_1[i["dvid"]] = {}
                    test_range_label_dict_1[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_1:
                if i['cand_index'] in global_feature_dict_1[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_1:
                        test_global_label_dict_1[i["dvid"]] = {}
                    test_global_label_dict_1[i["dvid"]][i["cand_index"]] = i["label"]
        elif i["dvid"] in range_feature_dict_2 or i["dvid"] in global_feature_dict_2:
            if i["dvid"] in range_feature_dict_2:
                if i['cand_index'] in range_feature_dict_2[i['dvid']]:
                    if i["dvid"] == 31531 or i["dvid"] == '31531':
                        print(i, i["cand_index"])
                    if i["dvid"] not in test_range_label_dict_2:
                        test_range_label_dict_2[i["dvid"]] = {}
                    test_range_label_dict_2[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_2:
                if i['cand_index'] in global_feature_dict_2[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_2:
                        test_global_label_dict_2[i["dvid"]] = {}
                    test_global_label_dict_2[i["dvid"]][i["cand_index"]] = i["label"]
        elif i["dvid"] in range_feature_dict_3 or i["dvid"] in global_feature_dict_3:
            if i["dvid"] in range_feature_dict_3:
                if i['cand_index'] in range_feature_dict_3[i['dvid']]:
                    if i["dvid"] not in test_range_label_dict_3:
                        test_range_label_dict_3[i["dvid"]] = {}
                    test_range_label_dict_3[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_3:
                if i['cand_index'] in global_feature_dict_3[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_3:
                        test_global_label_dict_3[i["dvid"]] = {}
                    test_global_label_dict_3[i["dvid"]][i["cand_index"]] = i["label"]
        elif i["dvid"] in range_feature_dict_4 or i["dvid"] in global_feature_dict_4:
            if i["dvid"] in range_feature_dict_4:
                if i['cand_index'] in range_feature_dict_4[i['dvid']]:
                    if i["dvid"] not in test_range_label_dict_4:
                        test_range_label_dict_4[i["dvid"]] = {}
                    test_range_label_dict_4[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_4:
                if i['cand_index'] in global_feature_dict_4[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_4:
                        test_global_label_dict_4[i["dvid"]] = {}
                    test_global_label_dict_4[i["dvid"]][i["cand_index"]] = i["label"]

    suc_dvid=[]
    # print(type(list(cand_len_dict.keys())[0]))
    label_list = [test_range_label_dict_1,test_range_label_dict_2,test_range_label_dict_3,test_range_label_dict_4,test_global_label_dict_1,test_global_label_dict_2,test_global_label_dict_3,test_global_label_dict_4]
    for index in range(4):
        test_range_label_dict = label_list[index]
        test_global_label_dict = label_list[index+4]
        for dvid in test_range_label_dict:
            gt_index = []
            for cand_index in test_range_label_dict[int(dvid)]:
                if test_range_label_dict[int(dvid)][cand_index] == 1:
                    gt_index.append(int(cand_index))

            if dvid in test_global_label_dict:
                for cand_index in test_global_label_dict[int(dvid)]:
                    if test_global_label_dict[int(dvid)][cand_index] == 1:
                        gt_index.append(int(cand_index))
            
            with open("../PredictDV/delete_row_single_o_baseline/" + str(dvid) + "_search_result.json", 'r', encoding="utf-8") as f:
                cand_list = json.load(f)
            with open("../PredictDV/global_candidate_1/" + str(dvid) + ".json", 'r') as f:
                g_cand_list = json.load(f)
            
            res_cand = cand_list + g_cand_list
            chose_list = [i for i in range(0,len(res_cand))]
            rand_index = random.choices(chose_list, k=min(precision, len(res_cand)))
        

            print("###")
            print(len(res_cand))
            print('gt_index', gt_index)
            print("rand_index", rand_index)
    
            # rand_index = int(rand_index)
            if len(set(rand_index) & set(gt_index)) > 0:
                suc_dvid.append(int(dvid))
            # break
    
    np.save('rand_suc_dvid_'+str(precision), suc_dvid)


def get_range_gt_id():
    file_path = "../PredictDV/ListDV/continous_batch_0_1 - Copy.json"
    
    all_result = 0
    with open(file_path, 'r', encoding='UTF-8') as f:
        listdvfindelete_o_relax_strip_list = json.load(f)
    
    id_dict = {}
    id_number = {}
    id_value_dict = {}
    id_index_dict = {}
    for dvinfo in listdvfindelete_o_relax_strip_list:
        if ',' in dvinfo["Value"]:
            continue
        kw = dvinfo["FileName"]+'***********'+dvinfo["SheetName"]
        if kw not in id_dict:
            id_dict[kw] = {}
            id_number[kw] = {}
            id_index_dict[kw] = {}
        is_in = False
        in_id_ = 0
        max_id = 0
        for id_ in id_dict[kw]:
            if dvinfo["Value"] == id_dict[kw][id_]:
                is_in = True
                in_id_ = id_
                break
            if id_ > max_id:
                max_id = id_
        max_id += 1
        if is_in ==False:
            id_dict[kw][max_id] = dvinfo["Value"]
            id_number[kw][max_id] = 1
            # id_value_dic[kw][max_id] = []
            id_index_dict[kw][max_id] = {}
            id_index_dict[kw][max_id]['ltx'] = dvinfo['ltx']
            id_index_dict[kw][max_id]['lty'] = dvinfo['lty']
            id_index_dict[kw][max_id]['rbx'] = dvinfo['rbx']
            id_index_dict[kw][max_id]['rby'] = dvinfo['rbx']
         
        else:
            id_number[kw][in_id_] += 1
    
    b = json.dumps(id_dict)
    f2 = open("range_gt_id_dict.json", 'w')
    f2.write(b)
    f2.close()

    b = json.dumps(id_number)
    f2 = open("range_gt_id_number_dict.json", 'w')
    f2.write(b)
    f2.close()

    b = json.dumps(id_index_dict)
    f2 = open("id_index_dict.json", 'w')
    f2.write(b)
    f2.close()

def get_target_content_sensitive():
    evaluate_range = 1
    evaluate_global = 1
    classifier = 3
    model = 1
    precision= 1
    root_path = "with_header_sname"
    res_path =  root_path + "/" + "range_"+str(evaluate_range).replace("-","n")+"_global_"+str(evaluate_global)
    with open(res_path+'/xgbrank_rank_suc_'+str(precision) +
                '_'+str(classifier)+str(model)+'.json') as f:
        suc_dvid = json.load(f)
    with open("../PredictDV/ListDV/continous_batch_0_0.json",'r', encoding='UTF-8') as f:
        dvinfos = json.load(f)
    # with open(res_path+'/merge_range_suc_'+str(precision)+'_'+str(classifier)+str(model)+'.json', 'r') as f:
    #     suc_dvid1 = json.load(f)
    # with open(res_path+'/merge_global_suc_'+str(precision)+'_'+str(classifier)+str(model)+'.json', 'r') as f:
    #     suc_dvid2 = json.load(f)
    # suc_dvid = suc_dvid1+suc_dvid2
    # with open(res_path+'/merge_global_suc_'+str(precision)+'_'+str(classifier)+str(model)+'.json', 'r') as f:
    #     suc_dvid2 = json.load(f)
    with open('both_label_range_'+str(evaluate_range).replace("-","n") + "_global_" +str(evaluate_global)+'_10000.json', 'r', encoding='UTF-8') as f:
        labels = json.load(f)
    test_range_label_dict_1 = {}
    test_range_label_dict_2 = {}
    test_range_label_dict_3 = {}
    test_range_label_dict_4 = {}

    test_global_label_dict_1 = {}
    test_global_label_dict_2 = {}
    test_global_label_dict_3 = {}
    test_global_label_dict_4 = {}

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

    for index, i in enumerate(labels):
        # print(index, len(labels))
        # if i["dvid"] == 31531 or i["dvid"] == '31531':
        #     print(i, i["cand_index"])
        if i["dvid"] in range_feature_dict_1 or i["dvid"] in global_feature_dict_1:
            if i["dvid"] in range_feature_dict_1:
                if i['cand_index'] in range_feature_dict_1[i['dvid']]:
                    if i["dvid"] not in test_range_label_dict_1:
                        test_range_label_dict_1[i["dvid"]] = {}
                    test_range_label_dict_1[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_1:
                if i['cand_index'] in global_feature_dict_1[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_1:
                        test_global_label_dict_1[i["dvid"]] = {}
                    test_global_label_dict_1[i["dvid"]][i["cand_index"]] = i["label"]
        elif i["dvid"] in range_feature_dict_2 or i["dvid"] in global_feature_dict_2:
            if i["dvid"] in range_feature_dict_2:
                if i['cand_index'] in range_feature_dict_2[i['dvid']]:
                    if i["dvid"] == 31531 or i["dvid"] == '31531':
                        print(i, i["cand_index"])
                    if i["dvid"] not in test_range_label_dict_2:
                        test_range_label_dict_2[i["dvid"]] = {}
                    test_range_label_dict_2[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_2:
                if i['cand_index'] in global_feature_dict_2[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_2:
                        test_global_label_dict_2[i["dvid"]] = {}
                    test_global_label_dict_2[i["dvid"]][i["cand_index"]] = i["label"]
        elif i["dvid"] in range_feature_dict_3 or i["dvid"] in global_feature_dict_3:
            if i["dvid"] in range_feature_dict_3:
                if i['cand_index'] in range_feature_dict_3[i['dvid']]:
                    if i["dvid"] not in test_range_label_dict_3:
                        test_range_label_dict_3[i["dvid"]] = {}
                    test_range_label_dict_3[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_3:
                if i['cand_index'] in global_feature_dict_3[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_3:
                        test_global_label_dict_3[i["dvid"]] = {}
                    test_global_label_dict_3[i["dvid"]][i["cand_index"]] = i["label"]
        elif i["dvid"] in range_feature_dict_4 or i["dvid"] in global_feature_dict_4:
            if i["dvid"] in range_feature_dict_4:
                if i['cand_index'] in range_feature_dict_4[i['dvid']]:
                    if i["dvid"] not in test_range_label_dict_4:
                        test_range_label_dict_4[i["dvid"]] = {}
                    test_range_label_dict_4[i["dvid"]][i["cand_index"]] = i["label"]
            if i["dvid"] in global_feature_dict_4:
                if i['cand_index'] in global_feature_dict_4[i['dvid']]:
                    if i["dvid"] not in test_global_label_dict_4:
                        test_global_label_dict_4[i["dvid"]] = {}
                    test_global_label_dict_4[i["dvid"]][i["cand_index"]] = i["label"]

    suc_dvid_1 = []
    suc_dvid_2 = []
    suc_dvid_3 = []
    suc_dvid_4 = []
    suc_dvid_5 = []
    suc_dvid_6 = []
    all_dvid_1 = []
    all_dvid_2 = []
    all_dvid_3 = []
    all_dvid_4 = []
    all_dvid_5 = []
    all_dvid_6 = []
    # print(type(list(cand_len_dict.keys())[0]))
    label_list = [test_range_label_dict_1,test_range_label_dict_2,test_range_label_dict_3,test_range_label_dict_4,test_global_label_dict_1,test_global_label_dict_2,test_global_label_dict_3,test_global_label_dict_4]
    for index in range(4):
        test_range_label_dict = label_list[index]    
        test_global_label_dict = label_list[index+4]
        for dvid in list(set(test_range_label_dict) | set(test_global_label_dict)):
            print("########")
            print(dvid)
            for dvinfo in dvinfos:
                if dvinfo["ID"] == dvid:
                    found_dvinfo=dvinfo
            print(found_dvinfo["ID"])
            content_set = set()
            for content in found_dvinfo['content']:
                content_set.add(content['Value'])
            print(content_set)
            content_lenth = len(content_set)
            if dvid in suc_dvid:
                if content_lenth == 1:
                    suc_dvid_1.append(dvid)
                if content_lenth == 2:
                    suc_dvid_2.append(dvid)
                if content_lenth == 3:
                    suc_dvid_3.append(dvid)
                if content_lenth == 4:
                    suc_dvid_4.append(dvid)
                if content_lenth == 5:
                    suc_dvid_5.append(dvid)
                if content_lenth > 5:
                    suc_dvid_6.append(dvid)
            if content_lenth == 1:
                all_dvid_1.append(dvid)
            if content_lenth == 2:
                all_dvid_2.append(dvid)
            if content_lenth == 3:
                all_dvid_3.append(dvid)
            if content_lenth == 4:
                all_dvid_4.append(dvid)
            if content_lenth == 5:
                all_dvid_5.append(dvid)
            if content_lenth > 5:
                all_dvid_6.append(dvid)
            
    
    
    np.save('target_content_suc_dvid_1', suc_dvid_1)
    np.save('target_content_suc_dvid_2', suc_dvid_2)
    np.save('target_content_suc_dvid_3', suc_dvid_3)
    np.save('target_content_suc_dvid_4', suc_dvid_4)
    np.save('target_content_suc_dvid_5', suc_dvid_5)
    np.save('target_content_suc_dvid_6', suc_dvid_6)

    np.save('target_content_all_dvid_1', all_dvid_1)
    np.save('target_content_all_dvid_2', all_dvid_2)
    np.save('target_content_all_dvid_3', all_dvid_3)
    np.save('target_content_all_dvid_4', all_dvid_4)
    np.save('target_content_all_dvid_5', all_dvid_5)
    np.save('target_content_all_dvid_6', all_dvid_6)

def send_data():
    for id_ in range(26):
        if id_ < 10:
            cmd = 'pscp \\\\msrasa\TableLint\Datasets\ToYeye\WithValidations\\00'+str(id_)+'.zip azureuser@104.215.61.197:00'+str(id_)+'.zip'
        else:
            cmd = 'pscp \\\\msrasa\TableLint\Datasets\ToYeye\WithValidations\\0'+str(id_)+'.zip azureuser@104.215.61.197:0'+str(id_)+'.zip'
        os.system(cmd)
        os.system('testtest123?')

def look_features():
    with open("../PredictDV/ListDV/c#_complete_both_features_10000.json", 'r', encoding='UTF-8') as f:
        feature_later = json.load(f)
    with open("../test-table-understanding/test-table-understanding/bin/debug/c#_both_features_1000_4.json", 'r', encoding='UTF-8') as f:
        feature_before = json.load(f)
    later_dvid = set([dv['dvid'] for dv in feature_later])
    before_dvid = set([dv['dvid'] for dv in feature_before])
    # print(later_dvid - before_dvid)
    # print(before_dvid - later_dvid)
    for later_feature in feature_later:
        for before_feature in feature_before:
            if later_feature['dvid'] == before_feature['dvid'] and later_feature['cand_index'] == before_feature['cand_index']:
                if(later_feature['d_char'] != before_feature['d_char'] or later_feature['d_len'] != before_feature['d_len'] or later_feature['emptiness'] != before_feature['emptiness'] or later_feature['distinctness'] != before_feature['distinctness'] or later_feature['completeness'] != before_feature['completeness']):
                    print(later_feature['dvid'], later_feature['cand_index'])
    
def look_features():
    with open("../PredictDV/ListDV/c#_complete_both_features_10000.json", 'r', encoding='UTF-8') as f:
        feature_later = json.load(f)
    with open("../test-table-understanding/test-table-understanding/bin/debug/c#_both_features_1000_4.json", 'r', encoding='UTF-8') as f:
        feature_before = json.load(f)
    later_dvid = set([dv['dvid'] for dv in feature_later])
    before_dvid = set([dv['dvid'] for dv in feature_before])

    lcount = 0
    cand_idnex_dict = {}
    cand_index_dict1 = {}

    for later_feature in feature_later:
        if later_feature['dvid'] not in cand_idnex_dict:
            cand_idnex_dict[later_feature['dvid']] = []
        cand_idnex_dict[later_feature['dvid']].append(later_feature['cand_index'])
    bcount = 0
    for before_feature in feature_before:
        if later_feature['dvid'] not in cand_index_dict1:
            cand_index_dict1[later_feature['dvid']] = []
        cand_index_dict1[later_feature['dvid']].append(before_feature['cand_index'])    
    
    for dvid in cand_idnex_dict:
        if dvid in cand_index_dict1:
            if cand_index_dict1[dvid] != cand_idnex_dict[dvid]:
                print(dvid)
    # print(lcount)    print(bcount)
    # print(later_dvid - before_dvid)
    # print(before_dvid - later_dvid)

def look_eval():
    root_path = 'with_header_sname'
    evaluate_range = 1
    evaluate_global = 1
    precision = 1
    model = 1
    classifier = 3
    res_path1 = "range_feature_a_1_keys.json"
    res_path_1 = "range_feature_1_keys.json"
    res_path = root_path + "/" + "range_" + \
        str(evaluate_range).replace("-", "n").replace('-', 'n') + \
            "_global_"+str(evaluate_global)
    res_path_2 = res_path+'/xgbrank_rank_suc_2_'+str(precision) + '_'+str(classifier)+str(model)+'.json'
    res_path_3 = res_path+'/xgbrank_rank_suc_1_'+str(precision) + '_'+str(classifier)+str(model)+'.json'

    
    with open(res_path_2) as f:
        suc_dvid = json.load(f)
    with open(res_path_3) as f:
        suc_dvid1 = json.load(f)
    with open(res_path1) as f:
        keys1 = json.load(f)
    with open(res_path_1) as f:
        keys2 = json.load(f)
    # print(set(keys1)==set(keys2))
    # print(set(suc_dvid)-set(suc_dvid1))

    with open('global_feature_dict_b_4.json', 'r', encoding='utf-8') as f:
        rfd_b = json.load(f)
    with open('global_feature_dict_4.json', 'r') as f:
        rfd_a = json.load(f)

    for index, dvid in enumerate(list(rfd_b.keys())):
        if dvid in rfd_a:
            for cand_index in rfd_b[dvid]:
                if cand_index not in rfd_a[dvid]:
                    print(dvid, cand_index)
                else:
                    for ind in range(0,len(rfd_b[dvid][cand_index])):
                        if rfd_a[dvid][cand_index][ind] != rfd_b[dvid][cand_index][ind]:
                            print(dvid, cand_index, ind)
                            print(rfd_a[dvid][cand_index][ind], rfd_b[dvid][cand_index][ind])
        else:
            print(dvid)
    for index, dvid in enumerate(list(rfd_a.keys())):
        if dvid in rfd_b:
            for cand_index in rfd_a[dvid]:
                if cand_index not in rfd_b[dvid]:
                    print(dvid, cand_index)
                else:
                    for ind in range(0,len(rfd_b[dvid][cand_index])):
                        if rfd_a[dvid][cand_index][ind] != rfd_b[dvid][cand_index][ind]:
                            print(dvid, cand_index, ind)
                            print(rfd_a[dvid][cand_index][ind], rfd_b[dvid][cand_index][ind])
        else:
            print(dvid)
# def get_local_dv_count():


if __name__ == "__main__":
    # sorted_hard_code_dict()
    # get_sematic_distance()
    # re_stat_evalutaion()
    # re_stat_head_len()
    # u_rmh_stat_evalutaion()
    # delete_o_relax_strip_stat_evalutaion()
    # delete_o_relax_strip_rmh_stat_evalutaion()
    # delete_o_relax_strip_stat_evaluation()
    # analyze_gt()
    # check_orientation_2_column_major(0.1)
    # check_orientation_2_row_major(1)
    # check_header()
    # single_column_row()
    # check_sorted_values()
    # get_sematic_distance()
    # get_leftness()
    # get_label(0)
    # get_both_label(0,0)
    # train()
    # count_both_label()
    # get_range_not_fit()
    # get_hard_code_dict()
    # get_global_candidate()
    # hard_code_baseline()
    # get_custom_dict()
    # get_hard_code_haeder_dict()
    # get_hard_code_sheet_name_dict()
    # plot_cand_number()
    # get_headre_fail()
    # merge_features()
    # count_row()
    # get_top_k_placeholder()
    # get_max_baseline_recall(2)
    look_res()
    # get_range_gt_id()
    # get_target_content_sensitive()
    # get_range_pop(5)
    # get_random_baseline(5)
    # send_data()
    # look_eval()