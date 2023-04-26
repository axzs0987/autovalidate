import json
import os
import pprint
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import gc
import time
import sys
import copy
import random
from cnn_model import CNNnetTripletBertL2Norm
from multiprocessing import Process
from analyze_formula import get_feature_vector_with_bert_keyw, change_word_to_save_word
from sentence_transformers import SentenceTransformer
import shutil
from rerank_model import RerankModel, RerankLinearModel1
from finegrain_model import FinegrainedModel
import faiss
import pickle
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, rand_score
"""
1. triplet 重新采样
2. semi-sample
3. retraining

1. train cos model

"""
version = 0
if version == 0:
    root_path = '/datadrive-2/data/fortune500_test/'
elif version == 1:
    root_path = '/datadrive-2/data/top10domain_test/'
logs = {}
def cos(a,b):
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    return np.matmul(a,b) / (ma*mb)
    
def generate_positive_pair():
    with open("TrainingFormulas_mergerange_custom_new_res_1.json",'r') as f:
        formulas = json.load(f)
    with open("json_data/sheetname_2_file_devided_prob_filter.json", 'r') as f:
        sheetname_2_file_devided_prob_filter = json.load(f)
    with open("formula_token_2_template_id_training.json", 'r') as f:
        formula_token_2_template_id_training = json.load(f)
    # print(formulas.keys())

    pair_same_position_num = 0
    pair_notsame_position_num = 0

    used_files = []    

    auchor_num = 0
    res_str = ''
    res_str_need = ''
    need_look_positive_filesheet_pair = []
    exists_list = os.listdir('formulas_training')
    print('exists_list', len(exists_list))
    print('formulas', len(formulas))
    print('generate saved formulas.....')

    # new_formulas = {}
    # saved_formulas = {}
    # for index, filesheet in enumerate(formulas):
    #     print(index, len(formulas))
    #     new_formulas[filesheet.split('/')[-1]] = formulas[filesheet]
    # formulas = new_formulas

    # for index, jsonname in enumerate(exists_list):
    #     print(index, len(exists_list))
    #     filename, sheetname, fr, fc = jsonname.replace('.json','').split('---')
    #     filesheet = filename + '---' + sheetname
    #     found = False
    #     for r1c1 in formulas[filesheet]:
    #         for id_ in formulas[filesheet][r1c1]:
    #             formula = formulas[filesheet][r1c1][id_]
    #             if fr == str(formula['fr']) and fc == str(formula['fc']):
    #                 found = True
    #                 found_r1c1 = r1c1
    #                 found_id = id_
    #                 found_formula = formula

    #     if found:
    #         if filesheet not in saved_formulas:
    #             saved_formulas[filesheet] = {}
    #         if found_r1c1 not in saved_formulas[filesheet]:
    #             saved_formulas[filesheet][found_r1c1] = {}
    #         saved_formulas[filesheet][found_r1c1][found_id] = found_formula
    # for index, filesheet in enumerate(formulas):
    #     print(index, len(formulas))
    #     for r1c1 in formulas[filesheet]:
    #         for id_ in formulas[filesheet][r1c1]:
    #             formula = formulas[filesheet][r1c1][id_]
    #             formula_token = filesheet.split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
    #             if formula_token in exists_list:
    #                 if filesheet not in saved_formulas:
    #                     saved_formulas[filesheet] = {}
    #                 if r1c1 not in saved_formulas[filesheet]:
    #                     saved_formulas[filesheet][r1c1] = {}
    #                 saved_formulas[filesheet][r1c1][id_] = formula
                # else:
                    # print('not in....')
    # with open('saved_formulas.json','w') as f:
    #     json.dump(saved_formulas, f)
    # formulas = saved_formulas
    sheetname2num = {}
    for index,filesheet in enumerate(formulas):
        print(index, len(formulas))
        used_files.append(filesheet)
        filename_sheet_list = filesheet.split('---')
        filename = filename_sheet_list[0]
        sheetname = filename_sheet_list[1]
        printed = False
        
        found = False
        if sheetname not in sheetname_2_file_devided_prob_filter:
            continue
        for class_ in sheetname_2_file_devided_prob_filter[sheetname]:
            for one_filename in class_['filenames']:
                if filename in one_filename:
                    found = True
                    found_class_ = class_
        if not found:
            # print('not found')
            continue

        # print("found_class_['filenames']", found_class_['filenames'])
        # break
        if len(found_class_['filenames'])==1:
            continue
        one_file_num = 0
        # print(filename, sheetname)
        need_continue = False
        if sheetname not in sheetname2num:
            sheetname2num[sheetname] = 0
        
        for r1c1 in formulas[filesheet]:
            if need_continue:
                continue
            for id_ in formulas[filesheet][r1c1]:
                if need_continue:
                    continue
                formula = formulas[filesheet][r1c1][id_]
                # if filesheet.split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc']) + '.json' not in exists_list:
                #     continue
                formula_token = filesheet.split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
                if formula_token not in formula_token_2_template_id_training:
                    continue
                auchor_found = False
                for other_filename in class_['filenames']:
                    if need_continue:
                        continue
                    if other_filename in used_files:
                        continue
                    if filename in other_filename:
                        continue
                    if other_filename + '---' + sheetname not in formulas:
                        continue
                    
                    other_formulas = formulas[other_filename+ '---' + sheetname]
                    # pprint.pprint(list(other_formulas.keys()))
                    for other_r1c1 in other_formulas:
                        # print("#######")
                        if need_continue:
                            continue
                        if sheetname2num[sheetname] > 3000:
                            need_continue = True
                            continue
                        sheetname2num[sheetname] += 1
                        candidate_formulas = []
                        for other_id in other_formulas[other_r1c1]:
                            other_one_formula = other_formulas[other_r1c1][other_id]
                            other_formula_token = filesheet.split('/')[-1] + '---' + str(other_one_formula['fr']) + '---' + str(other_one_formula['fc'])
                            if other_formula_token not in formula_token_2_template_id_training:
                                continue
                            if formula_token_2_template_id_training[formula_token] != formula_token_2_template_id_training[other_formula_token]:
                                continue
                            # if filesheet.split('/')[-1] + '---' + str(other_one_formula['fr']) + '---' + str(other_one_formula['fc']) + '.json' not in exists_list:
                            #     continue
                            distance = (formula['fr'] - other_one_formula['fr'])**2 + (formula['fc'] - other_one_formula['fc'])**2
                            candidate_formulas.append([other_one_formula, distance])
  
                        candidate_formulas = sorted(candidate_formulas, key=lambda x: x[1])
                        if len(candidate_formulas) == 0:
                            continue
                    
                        best_formula = candidate_formulas[0][0]
                        if formula['fr'] == best_formula['fr'] and formula['fc'] == best_formula['fc']:
                            res_str += filesheet + '   ' + other_filename+ '---' + sheetname + '   ' + r1c1 +'   ' + other_r1c1 + '   ' + str(formula['fr']) + '   ' + str(formula['fc']) + '   ' + str(best_formula['fr']) + '   '  + str(best_formula['fc']) + '\n'
                            pair_same_position_num += 1
                        else:
                            res_str += filesheet + '   ' + other_filename+ '---' + sheetname + '   ' + r1c1 +'   ' + other_r1c1 + '   ' + str(formula['fr']) + '   ' + str(formula['fc']) + '   ' + str(best_formula['fr']) + '   '  + str(best_formula['fc']) + '\n'
                            res_str_need += filesheet + '   ' + other_filename+ '---' + sheetname + '   ' + r1c1 + str(formula['fr']) + '   ' + str(formula['fc']) + '   ' + str(best_formula['fr']) + '   '  + str(best_formula['fc']) + '\n'
                            pair_notsame_position_num += 1
                            # if filesheet + '   ' + other_filename not in need_look_positive_filesheet_pair:
                            #     need_look_positive_filesheet_pair.append(filesheet + '   ' + other_filename)
                        auchor_found = True
                        
                        break
                        
                if auchor_found:
                    auchor_num += 1
                    one_file_num += 1
                    break
                print('auchor_num', auchor_num)
                print('pair_notsame_position_num', pair_notsame_position_num)
                print('pair_same_position_num', pair_same_position_num)
            if one_file_num >= 5:
                break
        # if auchor_num >= 1000000:
            # break

    # with open('need_look_positive_filesheet_pair.json', 'w') as f:
    #     json.dump(need_look_positive_filesheet_pair, f)
    with open('fine_tune_positive.txt','w') as f:
        f.write(res_str)
    with open('need_look_positive.txt', 'w') as f:
        f.write(res_str_need)

def generate_negative_pair():
    with open("TrainingFormulas_mergerange_custom_new_res_1.json",'r') as f:
        formulas = json.load(f)
    # with open("saved_formulas.json",'r') as f:
    #     formulas = json.load(f)
    with open("json_data/sheetname_2_file_devided_prob_filter.json", 'r') as f:
        sheetname_2_file_devided_prob_filter = json.load(f)
    with open("formula_token_2_template_id_training.json", 'r') as f:
        formula_token_2_template_id = json.load(f)
    print(formulas.keys())

    pair_same_position_num = 0
    pair_notsame_position_num = 0

    used_files = []    

    auchor_num = 0
    res_str = ''
    res_str_need = ''
    need_look_positive_filesheet_pair = []
    exists_list = os.listdir('formulas_training')

    saved_formulas = {}
    print('generate saved formulas.....')
    print(exists_list[0])
    res = set()
    for index, filesheet in enumerate(formulas):
        print(index, len(formulas))
        for r1c1 in formulas[filesheet]:
            for id_ in formulas[filesheet][r1c1]:
                formula = formulas[filesheet][r1c1][id_]
                formula_token = filesheet.split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
                res.add(formula_token + '.json')
                # if formula_token in exists_list:
                #     if filesheet not in saved_formulas:
                #         saved_formulas[filesheet] = {}
                #     if r1c1 not in saved_formulas[filesheet]:
                #         saved_formulas[filesheet][r1c1] = {}
                #     saved_formulas[filesheet][r1c1][id_] = formula
    print(len(res))
    print(len(exists_list))
    saved_formulas = list(set(res) & set(exists_list))
    with open('saved_formulas.json','w') as f:
        json.dump(saved_formulas, f)
    with open('saved_formulas.json','r') as f:
        saved_formulas = json.load(f)
    print(len(saved_formulas))

    sheetname2num = {}
    for index,filesheet in enumerate(formulas):
        print(index, len(formulas))
        used_files.append(filesheet)
        filename_sheet_list = filesheet.split('---')
        filename = filename_sheet_list[0]
        sheetname = filename_sheet_list[1]
        printed = False
        
        found = False
        if sheetname not in sheetname_2_file_devided_prob_filter:
            continue
        for class_ in sheetname_2_file_devided_prob_filter[sheetname]:
            for one_filename in class_['filenames']:
                if filename in one_filename:
                    found = True
                    found_class_ = class_
        if not found:
            # print('not found')
            continue

        # print("found_class_['filenames']", found_class_['filenames'])
        # break
        if len(found_class_['filenames'])==1:
            continue
        
        # print(filename, sheetname)
        if sheetname not in sheetname2num:
            sheetname2num[sheetname] = 0

        if sheetname2num[sheetname] > 3000:
            continue
        sheetname2num[sheetname] += 1
    
        saved_template = []
        for r1c1 in formulas[filesheet]:
            for id_ in formulas[filesheet][r1c1]:
                formula = formulas[filesheet][r1c1][id_]
                
                formula_token = filesheet.split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
                if formula_token_2_template_id[formula_token] in saved_template:
                    continue
                # if formula_token + '.json' not in exists_list:
                #     continue
                auchor_found = False
                for other_filename in class_['filenames']:
                    if other_filename in used_files:
                        continue
                    if filename in other_filename:
                        continue
                    if other_filename + '---' + sheetname not in formulas:
                        continue
                    
                    other_formulas = formulas[other_filename+ '---' + sheetname]
                   
                    # pprint.pprint(list(other_formulas.keys()))
                    candidate_formulas = []
                    for other_r1c1 in other_formulas:
                        if r1c1 != other_r1c1:
                            
                            for other_id in other_formulas[other_r1c1]:
                                other_one_formula = other_formulas[other_r1c1][other_id]
                                other_formula_token = filesheet.split('/')[-1] + '---' + str(other_one_formula['fr']) + '---' + str(other_one_formula['fc'])
                                # if filesheet.split('/')[-1] + '---' + str(other_one_formula['fr']) + '---' + str(other_one_formula['fc']) + '.json' not in exists_list:
                                #     continue
                                if other_formula_token not in formula_token_2_template_id:
                                    print('other fomrula not in template.....')
                                    continue
                                if formula_token_2_template_id[formula_token] == formula_token_2_template_id[other_formula_token]:
                                    continue
                                distance = (formula['fr'] - other_one_formula['fr'])**2 + (formula['fc'] - other_one_formula['fc'])**2
                                candidate_formulas.append([other_one_formula, distance])
    
                    candidate_formulas = sorted(candidate_formulas, key=lambda x: x[1])[0:5]
                    for index, cand_tuple in enumerate(candidate_formulas):
                   
                        best_formula = candidate_formulas[index][0]
                    

                        saved_template.append(formula_token_2_template_id[formula_token])
                        if formula['fr'] == best_formula['fr'] and formula['fc'] == best_formula['fc']:
                            res_str += filesheet + '   ' + other_filename+ '---' + sheetname + '   ' + r1c1 + '   ' +other_r1c1 + '   ' + str(formula['fr']) + '   ' + str(formula['fc']) + '   ' + str(best_formula['fr']) + '   '  + str(best_formula['fc']) + '\n'
                            res_str_need += filesheet + '   ' + other_filename+ '---' + sheetname + '   ' + r1c1 + '   ' + other_r1c1 + '   ' + str(formula['fr']) + '   ' + str(formula['fc']) + '   ' + str(best_formula['fr']) + '   '  + str(best_formula['fc']) + '\n'
                            pair_same_position_num += 1
                        else:
                            res_str += filesheet + '   ' + other_filename+ '---' + sheetname + '   ' + r1c1 + '   ' + other_r1c1 +'   '+ str(formula['fr']) + '   ' + str(formula['fc']) + '   ' + str(best_formula['fr']) + '   '  + str(best_formula['fc']) + '\n'
                            pair_notsame_position_num += 1
                            # if filesheet + '   ' + other_filename not in need_look_positive_filesheet_pair:
                            #     need_look_positive_filesheet_pair.append(filesheet + '   ' + other_filename)
                            auchor_found = True
                            auchor_num += 1
                    if auchor_found:
                        break    
                   
                if auchor_found:
                    break
         
                print('auchor_num', auchor_num)
                print('pair_notsame_position_num', pair_notsame_position_num)
                print('pair_same_position_num', pair_same_position_num)
            # if auchor_found:
            #     break
        # if auchor_num >= 1000000:
        #     break
        
    with open('need_look_negative_more.txt', 'w') as f:
        f.write(res_str_need)
    with open('fine_tune_negative_more.txt', 'w') as f:
        f.write(res_str)

def generate_shift_training_triples():
    with open("TrainingFormulas_mergerange_custom_new_res_1.json",'r') as f:
        formulas = json.load(f)
    # with open("json_data/sheetname_2_file_devided_prob_filter.json", 'r') as f:
    #     sheetname_2_file_devided_prob_filter = json.load(f)
    # with open("formula_token_2_template_id_training.json", 'r') as f:
    #     formula_token_2_template_id_training = json.load(f)
    # print(formulas.keys())
    max_count = 100
    all_filesheet_triple_list = []
    for index, filesheet in enumerate(formulas):
        print(index, len(formulas))
        one_filesheet_triple_list = []
        count = 0
        if len(formulas[filesheet]) == 1:
            continue
        print('len formulas[filesheet]', len(formulas[filesheet]))
        for index1, r1c1 in enumerate(formulas[filesheet]):
            # if len(one_filesheet_triple_list) >= 5:
            #     break
            if len(one_filesheet_triple_list) >= max_count:
                continue
            auchor_fr = formulas[filesheet][r1c1]['0']['fr']
            auchor_fc = formulas[filesheet][r1c1]['0']['fc']
            auchor_token = filesheet + '---' + str(auchor_fr) + '---' + str(auchor_fc)
            negative_list = list(set(formulas[filesheet].keys()) - set([r1c1]))
            for trytime in range(0,len(negative_list)):
                if len(one_filesheet_triple_list) >= index1*max_count:
                    continue
                # if 
                positive_option_list = ['left', 'right', 'up', 'down']
                if auchor_fr <= 11:
                    positive_option_list.pop(2)
                if auchor_fc <= 11:
                    positive_option_list.pop(0)
                positive_option = random.choice(positive_option_list)
                if positive_option == 'left':
                    positive_token = filesheet + '---' + str(auchor_fr) + '---' + str(auchor_fc - 1)
                    negative_fr = random.choice(list(range(max(auchor_fr-5, 1), auchor_fr + 5)))
                    negative_token = filesheet + '---' + str(negative_fr) + '---' + str(auchor_fc - 10)
                elif positive_option == 'right':
                    positive_token = filesheet + '---' + str(auchor_fr) + '---' + str(auchor_fc + 1)
                    negative_fr = random.choice(list(range(max(auchor_fr-5, 1), auchor_fr + 5)))
                    negative_token = filesheet + '---' + str(negative_fr) + '---' + str(auchor_fc + 10)
                elif positive_option == 'up':
                    positive_token = filesheet + '---' + str(auchor_fr - 1) + '---' + str(auchor_fc)
                    negative_fc = random.choice(list(range(max(auchor_fc-2, 1), auchor_fc + 2)))
                    negative_token = filesheet + '---' + str(auchor_fr - 10) + '---' + str(negative_fc)
                elif positive_option == 'down':
                    positive_token = filesheet + '---' + str(auchor_fr + 1) + '---' + str(auchor_fc)
                    negative_fc = random.choice(list(range(max(auchor_fc-2, 1), auchor_fc + 2)))
                    negative_token = filesheet + '---' + str(auchor_fr + 10) + '---' + str(negative_fc)

                # negative_r1c1 = random.choice(negative_list)
                # negative_list.remove(negative_r1c1)
                # negative_fr = formulas[filesheet][negative_r1c1]['0']['fr']
                # negative_fc = formulas[filesheet][negative_r1c1]['0']['fc']
                # negative_token = filesheet + '---' + str(negative_fr) + '---' + str(negative_fc)
                one_filesheet_triple_list.append([auchor_token, positive_token, negative_token])
                count += 1
        one_filesheet_triple_list = random.sample(one_filesheet_triple_list, min(max_count, len(one_filesheet_triple_list)))
        all_filesheet_triple_list += one_filesheet_triple_list
        # if len(all_filesheet_triple_list) == 1000:
        #     break
    # print('one_filesheet_triple_list', one_filesheet_triple_list)
    print("all_filesheet_triple_list", len(all_filesheet_triple_list))
    with open("shift_finetune_triples.json", 'w') as f:
        json.dump(all_filesheet_triple_list, f)

def generate_content_shift_training_triples():
    with open("TrainingFormulas_mergerange_custom_new_res_1.json",'r') as f:
        formulas = json.load(f)
    positive_pairs = []
    with open('fine_tune_positive.txt', 'r') as f:
        txtstr = f.read()
        lines = txtstr.split('\n')
        for line in lines:
            # print(line)
            items = line.split("   ")
            if len(items) != 8:
                continue
            first_filesheet = items[0]
            second_filesheet = items[1]
            first_rc = items[4] + '---' + items[5]
            second_rc = items[6] + '---' + items[7]
            auchor_r1c1 = items[2]
            first_token = first_filesheet + '---' + first_rc
            second_token = second_filesheet + '---' + second_rc
            positive_pairs.append([first_token, second_token, auchor_r1c1])
    # print(len(positive_pairs))
    # return
    # with open("json_data/sheetname_2_file_devided_prob_filter.json", 'r') as f:
    #     sheetname_2_file_devided_prob_filter = json.load(f)
    # with open("formula_token_2_template_id_training.json", 'r') as f:
    #     formula_token_2_template_id_training = json.load(f)
    # print(formulas.keys())
    max_count = 100
    all_filesheet_triple_list = []
    filesheets = set()
    for index, positive_pair in enumerate(positive_pairs):
        print(index, len(positive_pairs))
        first_filesheet = positive_pair[0].split('---')[0] + '---' + positive_pair[0].split('---')[1]
        second_filesheet = positive_pair[1].split('---')[0] + '---' + positive_pair[1].split('---')[1]
        filesheet = second_filesheet
        one_filesheet_triple_list = []
        count = 0
        if filesheet not in formulas:
            continue
        if len(formulas[filesheet]) == 1:
            continue
        print('len formulas[filesheet]', len(formulas[first_filesheet]))
        filesheets.add(first_filesheet)
        filesheets.add(second_filesheet)
        auchor_token = positive_pair[0]
        positive_token = positive_pair[1]
        r1c1 = positive_pair[2]
        positive_filesheet = positive_token.split('---')[0] + '---' +  positive_token.split('---')[1]
        positive_fr = int(positive_token.split('---')[2])
        positive_fc = int(positive_token.split('---')[3])
        shift = random.choice(range(1,5))
        option_list = ['left', 'right', 'up', 'down']
        if positive_fr <= shift + 1:
            option_list.pop(2)
        if positive_fc <= shift + 1:
            option_list.pop(0)
        positive_option = random.choice(option_list)
        if positive_option == 'left':
            positive_token = filesheet + '---' + str(positive_fr) + '---' + str(positive_fc - 1)
            negative_fr = random.choice(list(range(max(positive_fr-5, 1), positive_fr + 5)))
            negative_token = filesheet + '---' + str(negative_fr) + '---' + str(positive_fc - shift)
        elif positive_option == 'right':
            positive_token = filesheet + '---' + str(positive_fr) + '---' + str(positive_fc + 1)
            negative_fr = random.choice(list(range(max(positive_fr-5, 1), positive_fr + 5)))
            negative_token = filesheet + '---' + str(negative_fr) + '---' + str(positive_fc + shift)
        elif positive_option == 'up':
            positive_token = filesheet + '---' + str(positive_fr - 1) + '---' + str(positive_fc)
            negative_fc = random.choice(list(range(max(positive_fc-2, 1), positive_fc + 2)))
            negative_token = filesheet + '---' + str(positive_fr - shift) + '---' + str(negative_fc)
        elif positive_option == 'down':
            positive_token = filesheet + '---' + str(positive_fr + 1) + '---' + str(positive_fc)
            negative_fc = random.choice(list(range(max(positive_fc-2, 1), positive_fc + 2)))
            negative_token = filesheet + '---' + str(positive_fr + shift) + '---' + str(negative_fc)
                 
        negative_list = list(set(formulas[filesheet].keys()) - set([r1c1]))
        negative_r1c1 = random.choice(negative_list)
        negative_list.remove(negative_r1c1)
        negative_fr = formulas[filesheet][negative_r1c1]['0']['fr']
        negative_fc = formulas[filesheet][negative_r1c1]['0']['fc']
        negative_token = filesheet + '---' + str(negative_fr) + '---' + str(negative_fc)
        all_filesheet_triple_list.append([auchor_token, positive_token, negative_token])
        count += 1

    # print('one_filesheet_triple_list', one_filesheet_triple_list)
    print("all_filesheet_triple_list", len(all_filesheet_triple_list))
    with open("specific_finetune_triples_1_5.json", 'w') as f:
        json.dump(all_filesheet_triple_list, f) # specific_finetune_triples
    print('len(filesheets', len(filesheets))

def batch_gen_neighbor_tripliet():
    process = [Process(target=generate_neighbor_triplet, args=(1,20)),
        Process(target=generate_neighbor_triplet, args=(2,20)), 
        Process(target=generate_neighbor_triplet, args=(3,20)),
        Process(target=generate_neighbor_triplet, args=(4,20)), 
        Process(target=generate_neighbor_triplet, args=(5,20)),
        Process(target=generate_neighbor_triplet, args=(6,20)), 
        Process(target=generate_neighbor_triplet, args=(7,20)),
        Process(target=generate_neighbor_triplet, args=(8,20)), 
        Process(target=generate_neighbor_triplet, args=(9,20)),
        Process(target=generate_neighbor_triplet, args=(10,20)), 
        Process(target=generate_neighbor_triplet, args=(11,20)),
        Process(target=generate_neighbor_triplet, args=(12,20)), 
        Process(target=generate_neighbor_triplet,args=(13,20)),
        Process(target=generate_neighbor_triplet, args=(14,20)), 
        Process(target=generate_neighbor_triplet, args=(15,20)),
        Process(target=generate_neighbor_triplet, args=(16,20)), 
        Process(target=generate_neighbor_triplet, args=(17,20)),
        Process(target=generate_neighbor_triplet, args=(18,20)), 
        Process(target=generate_neighbor_triplet, args=(19,20)), 
        Process(target=generate_neighbor_triplet, args=(20,20)), 
    ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束

    ftp_str = []
    for index in range(1,21):
        with open('finetune_neighbor_triplet_'+str(index)+'.json','r') as f:
            temp1 =json.load(f)
        ftp_str += temp1
    with open('finetune_neighbor_triplet.json','w') as f:
        json.dump(ftp_str, f)

def look_finetune_neighbor_triplet():
    with open('finetune_neighbor_triplet.json','r') as f:
        finetune_neighbor_triplet = json.load(f) 

    for item in finetune_neighbor_triplet:
        print(item)
        break
def batch_gen_tripliet():
    # process = [Process(target=generate_triplet, args=(1,20)),
    #     Process(target=generate_triplet, args=(2,20)), 
    #     Process(target=generate_triplet, args=(3,20)),
    #     Process(target=generate_triplet, args=(4,20)), 
    #     Process(target=generate_triplet, args=(5,20)),
    #     Process(target=generate_triplet, args=(6,20)), 
    #     Process(target=generate_triplet, args=(7,20)),
    #     Process(target=generate_triplet, args=(8,20)), 
    #     Process(target=generate_triplet, args=(9,20)),
    #     Process(target=generate_triplet, args=(10,20)), 
    #     Process(target=generate_triplet, args=(11,20)),
    #     Process(target=generate_triplet, args=(12,20)), 
    #     Process(target=generate_triplet,args=(13,20)),
    #     Process(target=generate_triplet, args=(14,20)), 
    #     Process(target=generate_triplet, args=(15,20)),
    #     Process(target=generate_triplet, args=(16,20)), 
    #     Process(target=generate_triplet, args=(17,20)),
    #     Process(target=generate_triplet, args=(18,20)), 
    #     Process(target=generate_triplet, args=(19,20)), 
    #     Process(target=generate_triplet, args=(20,20)), 
    # ]
    # [p.start() for p in process]  # 开启了两个进程
    # [p.join() for p in process]   # 等待两个进程依次结束

    ftp_str = []
    for index in range(1,21):
        with open('finetune_triplet_more_'+str(index)+'.json','r') as f:
            temp1 =json.load(f)
        ftp_str += temp1
    with open('finetune_triplet_more.json','w') as f:
        json.dump(ftp_str, f)


def generate_triplet(thread_id, batch_num):
    with open('saved_fine_tune_positive.txt', 'r') as f:
        pos_txt = f.read()
    with open('saved_fine_tune_negative_more.txt', 'r') as f:
        neg_txt = f.read()

    res = []
    pos_lines = pos_txt.split('\n')
    neg_lines = neg_txt.split('\n')
    batch_len = len(neg_lines)/batch_num
    for index, neg_line in enumerate(neg_lines):
        if thread_id != batch_num:
            if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1 ):
                continue
        print(index, len(neg_lines))
        neg_splits = neg_line.split('   ')
        if len(neg_splits) != 8:
            continue
        
        neg_token1 = neg_splits[0] + '---' + neg_splits[4] + '---' + neg_splits[5]
        neg_token2 = neg_splits[1] + '---' + neg_splits[6] + '---' + neg_splits[7]
        for pos_line in pos_lines:
            # print(pos_line)
            pos_splits = pos_line.split('   ')
            if len(pos_splits) != 8:
                continue
            
            pos_token1 = pos_splits[0]+ '---' + pos_splits[4] + '---' + pos_splits[5]
            pos_token2 = pos_splits[1]+ '---' + pos_splits[6] + '---' + pos_splits[7]
            # print('pos_token1', pos_token1)
            # print('pos_token2', pos_token2)
            # print('neg_token1', neg_token1)
            # print('neg_token2', neg_token2)
            if pos_token1 == neg_token1:
                auchor = pos_token1
                positive = pos_token2
                negative = neg_token2
                print('add')
                res.append((auchor, positive, negative))
            elif pos_token1 == neg_token2:
                auchor = pos_token1
                positive = pos_token2
                negative = neg_token1
                print('add')
                res.append((auchor, positive, negative))
            elif pos_token2 == neg_token1:
                auchor = pos_token2
                positive = pos_token1
                negative = neg_token2
                print('add')
                res.append((auchor, positive, negative))
            elif pos_token2 == neg_token2:
                auchor = pos_token2
                positive = pos_token1
                negative = neg_token1
                print('add')
                res.append((auchor, positive, negative))
    with open('finetune_triplet_more_'+str(thread_id)+'.json', 'w') as f:
        json.dump(res, f)

def generate_neighbor_triplet(thread_id, batch_num):
    with open('saved_fine_tune_positive.txt', 'r') as f:
        pos_txt = f.read()
    with open('negative_neighbors_pairs.txt', 'r') as f:
        neg_txt = f.read()

    res = []
    pos_lines = pos_txt.split('\n')
    neg_lines = neg_txt.split('\n')
    batch_len = len(neg_lines)/batch_num
    for index, neg_line in enumerate(neg_lines):
        if thread_id != batch_num:
            if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1 ):
                continue
        print(index, len(neg_lines))
        neg_splits = neg_line.split('   ')
        # print('len(neg_splits)', len(neg_splits))
        
        neg_token1 = neg_splits[0]
        neg_token2 = neg_splits[1]
        for pos_line in pos_lines:
            # print(pos_line)
            pos_splits = pos_line.split('   ')
            if len(pos_splits) != 8:
                continue
            
            pos_token1 = pos_splits[0]+ '---' + pos_splits[4] + '---' + pos_splits[5]
            pos_token2 = pos_splits[1]+ '---' + pos_splits[6] + '---' + pos_splits[7]
            # print('pos_token1', pos_token1)
            # print('pos_token2', pos_token2)
            # print('neg_token1', neg_token1)
            # print('neg_token2', neg_token2)
            if pos_token1 == neg_token1:
                auchor = pos_token1
                positive = pos_token2
                negative = neg_token2
                print('add')
                res.append((auchor, positive, negative))
            elif pos_token1 == neg_token2:
                auchor = pos_token1
                positive = pos_token2
                negative = neg_token1
                print('add')
                res.append((auchor, positive, negative))
            elif pos_token2 == neg_token1:
                auchor = pos_token2
                positive = pos_token1
                negative = neg_token2
                print('add')
                res.append((auchor, positive, negative))
            elif pos_token2 == neg_token2:
                auchor = pos_token2
                positive = pos_token1
                negative = neg_token1
                print('add')
                res.append((auchor, positive, negative))
    with open('finetune_neighbor_triplet_'+str(thread_id)+'.json', 'w') as f:
        json.dump(res, f)

def download():
    with open('need_look_positive_filesheet_pair.json', 'r') as f:
        download_pair = json.load(f)

    download_files = set()
    for pair in download_pair:
        filename1 = pair.split('   ')[0].split('---')[0]
        filename2 = pair.split('   ')[1].split('---')[0]
        download_files.add(filename1)
        download_files.add(filename2)
    download_files = list(download_files)
    download_files.sort()
    for filename in download_files:
        target_path = filename.replace('../../data/', '/datadrive/data/')
        os.system('cp '+target_path + ' ' +'training_download/')

def look_negative():
    # with open('need_look_positive_filesheet_pair.json', 'r') as f:
    #     download_pair = json.load(f)

    # download_files = set()
    # for pair in download_pair:
    #     filename1 = pair.split('   ')[0].split('---')[0]
    #     filename2 = pair.split('   ')[1].split('---')[0]
    #     download_files.add(filename1)
    #     download_files.add(filename2)
    # download_files = list(download_files)

    # filelist = os.listdir('formulas_training/')
    # with open('saved_formulas.json','r') as f:
        # saved_formulas = json.load(f)
    saved_formulas = os.listdir('other_training_formulas')
    with open('fine_tune_negative_more.txt', 'r') as f: 
        str_ = f.read()
    res = str_.split('\n')
    print(len(res))
# with open('res_str_need.txt', 'r') as f:
#     count = 0
#     str_ = f.read()
#     res = str_.split('\n')
#     print(len(res))
    count = 0
    new_res = ''
    for index,line in enumerate(res):
        print(index, len(res))
        count += 1
        # if count <= 90:
        #     continue
        split_list = line.split('   ')
        found = True
        if len(split_list) != 8:
            continue
        # print(split_list)
        filename1 = split_list[0].split('/')[-1] + '---'+ split_list[4] + '---' + split_list[5] + '.json'
        filename2 = split_list[1].split('/')[-1] + '---'+ split_list[6] + '---' + split_list[7] + '.json'
        # print('xxxxx')
        # print('filename', filename1)
        # print(saved_formulas[0])
        if filename1 in saved_formulas and filename2 in saved_formulas:
            print('in')
            new_res += line + '\n'
        # for index,item in enumerate(split_list):
        #     # print(item)
        #     if index == 0 or index == 1:
        #         if item.split('---')[0] not in download_files:
        #             found = False
        #     if found == False:
        #         break
        #     if found:
        #         print(item)
        # if found:
        #     print('count', count)
        #     print("####")
        #     break
        # break
    with open("saved_fine_tune_negative_more.txt", 'w') as f:
        f.write(new_res)

def para_gen_saved_pos():
    # process = [Process(target=look_positive, args=(1,20)),
    #     Process(target=look_positive, args=(2,20)), 
    #     Process(target=look_positive, args=(3,20)),
    #     Process(target=look_positive, args=(4,20)), 
    #     Process(target=look_positive, args=(5,20)),
    #     Process(target=look_positive, args=(6,20)), 
    #     Process(target=look_positive, args=(7,20)),
    #     Process(target=look_positive, args=(8,20)), 
    #     Process(target=look_positive, args=(9,20)),
    #     Process(target=look_positive, args=(10,20)), 
    #     Process(target=look_positive, args=(11,20)),
    #     Process(target=look_positive, args=(12,20)), 
    #     Process(target=look_positive,args=(13,20)),
    #     Process(target=look_positive, args=(14,20)), 
    #     Process(target=look_positive, args=(15,20)),
    #     Process(target=look_positive, args=(16,20)), 
    #     Process(target=look_positive, args=(17,20)),
    #     Process(target=look_positive, args=(18,20)), 
    #     Process(target=look_positive, args=(19,20)), 
    #     Process(target=look_positive, args=(20,20)), 
    # ]
    # [p.start() for p in process]  # 开启了两个进程
    # [p.join() for p in process]   # 等待两个进程依次结束

    new_str = ''
    for index in range(1,21):
        with open('saved_fine_tune_positive_'+str(index)+'.txt','r') as f:
            temp1 = f.read()
        new_str += temp1

    with open('saved_fine_tune_positive.txt','w') as f:
        f.write(new_str)


def look_positive(thread_id, batch_num):
    # with open('need_look_positive_filesheet_pair.json', 'r') as f:
    #     download_pair = json.load(f)

    # download_files = set()
    # for pair in download_pair:
    #     filename1 = pair.split('   ')[0].split('---')[0]
    #     filename2 = pair.split('   ')[1].split('---')[0]
    #     download_files.add(filename1)
    #     download_files.add(filename2)
    # download_files = list(download_files)

    # filelist = os.listdir('formulas_training/')
    new_res = ''
    # with open('filter_positive.txt', 'r') as f:
    # with open('saved_formulas.json', 'r') as f:
    #     saved_formulas = json.load(f)
    saved_formulas = os.listdir('other_training_formulas')
    with open('fine_tune_positive.txt', 'r') as f:
        str_ = f.read()
    res = str_.split('\n')
    print(len(res))
# with open('res_str_need.txt', 'r') as f:
#     count = 0
#     str_ = f.read()
#     res = str_.split('\n')
#     print(len(res))
    count = 0
    added = []
    new_res = ''
    batch_len = len(res)/batch_num
    for ind,line in enumerate(res):
        
        if thread_id != batch_num:
            if(ind <= batch_len * (thread_id - 1 ) or ind > batch_len * thread_id):
                continue
        else:
            if ind <= batch_len * (thread_id - 1 ):
                continue
        print(ind, len(res))
        # if count <= 51:
        #     continue
        split_list = line.split('   ')
        # print(len(split_list))
        if len(split_list) == 8:
            filename1 = split_list[0].split('/')[-1] + '---'+ split_list[4] + '---' + split_list[5] + '.json'
            filename2 = split_list[1].split('/')[-1] + '---'+ split_list[6] + '---' + split_list[7] + '.json'
            # print("###########")
            # print('filename1', filename1)
            # print('filename2', filename2)
            if filename1 in saved_formulas and filename2 in saved_formulas:
                print('in')
                new_res += line + '\n'
                count += 1
            # pprint.pprint(split_list)
        #     ffr, ffc, lfr, lfc = split_list[4],split_list[5],split_list[6],split_list[7]
        #     # print(ffr, ffc, lfr, lfc)
        #     try:
        #         _ = int(ffr), int(ffc), int(lfr), int(lfc)
        #     except:
        #         print('error')
        #         continue

        # else:
        #     continue
        # if ind == 0:
        #     last_filename = filename
        #     last_ffr, last_ffc = ffr, ffc
        #     continue
        # if filename == last_filename:
        #     can_add = True
        #     for item in added:
        #         if abs(int(item[0]) - int(ffr)) == 0 and abs(int(item[1]) - int(ffc)) == 0:
        #             can_add = False
        #             break
        #     if can_add:
        #         new_res += line + '\n'
        #         added.append((ffr, ffc))
        # else:
        #     print(filename, added)
        #     added = []
        # last_filename = filename
        # last_ffr, last_ffc = ffr, ffc
        # if ffr == lfr and ffc == lfc:
        #     continue
        # found = False
        # for index,item in enumerate(split_list):
        #     # print(item)
        #     if index == 0 or index == 1:
        #         if item.split('---')[0] not in download_files:
        #             found = False
        #     if found == False:
        #         break
        #     if found:
        #         print(item)
        # if found:
        #     print('count', count)
        #     print("####")
        #     break
    # with open('filter_positive.txt', 'w') as f:
    #     f.write(new_res)
        if count % 10000 == 0:
            with open('saved_fine_tune_positive_'+str(thread_id)+'.txt', 'w') as f:
                f.write(new_res)
    with open('saved_fine_tune_positive_'+str(thread_id)+'.txt', 'w') as f:
        f.write(new_res)
def look_one_feature():
    filelist = os.listdir('formulas_training')

    with open('need_look_positive_filesheet_pair.json', 'r') as f:
        download_pair = json.load(f)
    download_files = set()
    for pair in download_pair:
        filename1 = pair.split('   ')[0].split('---')[0]
        filename2 = pair.split('   ')[1].split('---')[0]
        download_files.add(filename1)
        download_files.add(filename2)
    download_files = list(download_files)
    for jsonfile in filelist:
        filename = jsonfile.split('---')[0]
        found = False
        for dfile in download_files:
            dfilename = dfile.split('/')[-1]
            if dfilename == filename:
                found_file = jsonfile
                found = True
                break
        if found:
            break

    filename1 = found_file
    print(filename1)
    with open('formulas_training/'+filename1,'r') as f:
        res = json.load(f)
    count = 0

    for index,item in enumerate(res['sheetfeature']):
        # pprint.pprint(item)
        # print(index)
        # print(type(item))
        # print(len(item))
        if type(item).__name__ == 'list':

            if item[0]['width'] == 0:
                count += 1
            else:
                print('non width0', index, item[0]['width'])
        else:
            if item['width'] == 0:
                count += 1
            else:
                print('non width0', index, item['width'])
    print(count)

    index = 0
    start_r = int(filename1.replace('.json','').split('---')[2]) - 50
    start_c = int(filename1.replace('.json','').split('---')[3]) - 5
    for row in range(1, 101):
        for col in range(1, 11):
            if index == 504:
                print(start_r + row, start_c + col)
            index += 1

def look_saved_pos_neg():
    with open("saved_fine_tune_negative.txt", 'r') as f:
        str_ = f.read()
    res = str_.split('\n')
    print(len(res))
    neg_files = set()
    # print(res[0:10])

    pos_same = 0
    neg_pair = 0
    for index, line in enumerate(res):
        splits = line.split('   ')
        if len(splits) != 8:
            continue
        filename1 = splits[0]
        filename2 = splits[1]
        neg_files.add(filename1)
        neg_files.add(filename2)
        neg_pair += 1

    pos_files = set()
    pos_pair = 0
    with open("saved_fine_tune_positive.txt", 'r') as f:
        str_ = f.read()
    res = str_.split('\n')
    for index, line in enumerate(res):
        splits = line.split('   ')
        # print(line)
        # print(len(splits))
        if len(splits) != 8:
            continue
        
        filename1 = splits[0]
        filename2 = splits[1]
        pos_files.add(filename1)
        pos_files.add(filename2)
        # print("####")
        # print('splits[4]', splits[4])
        # print('splits[6]', splits[6])
        # print('splits[5]', splits[5])
        # print('splits[7]', splits[7])
        if splits[4] == splits[6] and splits[5] == splits[7]:
            pos_same += 1
        pos_pair += 1
    print('neg_files', len(neg_files))
    print('pos_files', len(pos_files))
    print('all_files', len(neg_files | pos_files))
    print('neg_pair', neg_pair)
    print('pos_pair', pos_pair)
    print('pos_same', pos_same)
    res = str_.split('\n')
    # files.add()
    print(len(res))
    # print(res[0:10])
    # for index, line in enumerate(res[0:10]):
        # print(line)


def generate_training_data():
    res = []
    with open("saved_fine_tune_negative.txt", 'r') as f:
        str_ = f.read()
    neg = str_.split('\n')
    print(len(neg))
    # print(res[0:10])
    neg_res = []
    pos_res = []
    res = pos_res + neg_res
    id_ = 1
    for index, line in enumerate(neg):
        print('neg', index, len(neg))
        items = line.split('   ')
        if len(items) != 8:
            continue
        formula_token1 = items[0]+'---'+items[4] + '---' + items[5]
        formula_token2 = items[1]+'---'+items[6] + '---' + items[7]
        neg_res.append((formula_token1, formula_token2, -1))
    
    print("###############")
    with open("saved_fine_tune_positive.txt", 'r') as f:
        str_ = f.read()
    pos = str_.split('\n')
    print(len(pos))
    # print(res[0:10])
    for index, line in enumerate(pos):
        print('pos', index, len(pos))
        items = line.split('   ')
        if len(items) != 8:
            continue
        formula_token1 = items[0]+'---'+items[4] + '---' + items[5]
        formula_token2 = items[1]+'---'+items[6] + '---' + items[7]
        pos_res.append((formula_token1, formula_token2, 1))
        if len(pos_res) == len(neg_res):
            break

    train_pos = pos_res[0:int(len(pos_res)*3/4)]
    test_pos = list(set(pos_res) - set(train_pos))

    train_neg = neg_res[0:int(len(neg_res)*3/4)]
    test_neg = list(set(neg_res) - set(train_neg))

    train_res = train_pos + train_neg
    test_res = test_pos + test_neg
    print(len(train_res))
    with open('fine_tune_training_pair_1.json', 'w') as f:
        json.dump(train_res[0:35000], f)
    with open('fine_tune_training_pair_2.json', 'w') as f:
        json.dump(train_res[35000:], f)
    print(len(test_res))
    with open('fine_tune_testing_pair.json', 'w') as f:
        json.dump(test_res, f)

# def generate_input_feature():
    

class FineTune():
    def __init__(self, batch_size, epoch_nums, l2 = True, iscontinue=False, continue_epoch=-1):
        self.batch_size = batch_size
        self.iscontinue = iscontinue
        self.continue_epoch = continue_epoch
        self.l2 = l2
        # if not iscontinue:
        #     if not self.l2:
        #         self.model = torch.load('196model/cnn_new_dynamic_triplet_margin_1_3_12')
        # else:
        #     # self.model = torch.load('/datadrive-2/data/finetune_specific_l2_new/epoch_'+str(self.continue_epoch))
        #     self.model = torch.load('/datadrive-2/data/finetune_specific_l2_5_10/epoch_'+str(self.continue_epoch))
        # if l2:
        #     # self.model = torch.load('/datadrive-2/data/l2_model_0_3')
        #     self.model = torch.load("/datadrive-2/data/l2_model/4_1")
        self.model = FinegrainedModel()
        print(self.model)
        # if not iscontinue:
        #     self.model = CNNnetTripletBert1010()
        # else:
        #     self.model = torch.load('model2/epoch_'+str(self.continue_epoch-1))
        self.epoch_nums = epoch_nums
        # self.loss_func = torch.nn.CosineEmbeddingLoss()
        self.opt = torch.optim.Adam(self.model.parameters(),lr=0.001)
        self.train_blen = 111
        self.test_blen = 37
        self.train_loader = 0

        gc.collect()

    def save_data(self, train=True, need_save=True):
        print('generate data.......')
        with open('fine_tune_training_pair.json', 'r') as f:
            training_pairs = json.load(f)
        train_features = []
        train_label = []
        batch = 1
        for index,triple in enumerate(training_pairs):
            if len(train_features) == 1000 or len(train_label)==1000:
                print('saving training......')
                np.save('finetune_training_feature_dataset/'+str(batch)+'.npy', train_features)
                np.save('finetune_training_label_dataset/'+str(batch)+'.npy', train_label)
                batch += 1
                train_features = []
                train_label = []
            print(index, len(training_pairs))
            feature1 = np.load('input_feature_finetune/'+triple[0].split('/')[-1]+'.npy', allow_pickle=True)
            feature2 = np.load('input_feature_finetune/'+triple[1].split('/')[-1]+'.npy', allow_pickle=True)
            train_features.append((feature1, feature2))
            train_label.append(int(triple[2]))
        
        print('saving training......')
        np.save('finetune_training_feature_dataset/'+str(batch)+'.npy', train_features)
        np.save('finetune_training_label_dataset/'+str(batch)+'.npy', train_label)
        batch += 1
        train_features = []
        train_label = []

        test_features = []
        test_label = []
        batch = 1
        with open('fine_tune_testing_pair.json', 'r') as f:
            testing_pairs = json.load(f)
        for index,triple in enumerate(testing_pairs):
            if os.path.exists('finetune_testing_feature_dataset/'+str(batch)+'.npy'):
                continue
            if len(test_features) == 1000 or len(test_label) == 1000:
                print('saving training......')
                np.save('finetune_testing_feature_dataset/'+str(batch)+'.npy', test_features)
                np.save('finetune_testing_label_dataset/'+str(batch)+'.npy', test_label)
                batch += 1
                test_features = []
                test_label = []
            print(index, len(testing_pairs))
            feature1 = np.load('input_feature_finetune/'+triple[0].split('/')[-1]+'.npy', allow_pickle=True)
            feature2 = np.load('input_feature_finetune/'+triple[1].split('/')[-1]+'.npy', allow_pickle=True)
            test_features.append((feature1, feature2))
            test_label.append(int(triple[2]))
        
        print('saving training......')
        # np.save('finetune_testing_feature_dataset/'+str(batch)+'.npy', test_features)
        np.save('finetune_testing_label_dataset/'+str(batch)+'.npy', test_label)
        batch += 1
        test_features = []
        test_label = []

    def save_triplet_data(self):
        print('generate data.......')
        with open('saved_finetune_triplet_more.json', 'r') as f:
            triples = json.load(f)
        # train_features = []
        batch = 1
        training_triples = triples[0:5*int(len(triples)/6)]
        test_triples = triples[5*int(len(triples)/6):]
        # for index,triple in enumerate(training_triples):
        #     if len(train_features) == 500:
        #         print('saving training......')
        #         np.save('triplet_finetune_training_feature_dataset/'+str(batch)+'.npy', train_features)
        #         batch += 1
        #         train_features = []
    
        #     print(index, len(training_triples))
        #     feature1 = np.load('input_feature_finetune/'+triple[0].split('/')[-1]+'.npy', allow_pickle=True)
        #     feature2 = np.load('input_feature_finetune/'+triple[1].split('/')[-1]+'.npy', allow_pickle=True)
        #     feature3 = np.load('input_feature_finetune/'+triple[2].split('/')[-1]+'.npy', allow_pickle=True)
        #     train_features.append((feature1, feature2, feature3))
        
        # print('saving training......')
        # np.save('triplet_finetune_training_feature_dataset/'+str(batch)+'.npy', train_features)
        # train_features = []
        # train_label = []

        test_features = []
        # test_label = []
        batch = 1

        for index,triple in enumerate(test_triples):
            if len(test_features) == 500:
                print('saving training......')
                np.save('/datadrive-2/data/triplet_finetune_training_feature_dataset_more/'+str(batch)+'.npy', test_features)
                batch += 1
                test_features = []
    
            print(index, len(test_triples))
            feature1 = np.load('input_feature_finetune/'+triple[0].split('/')[-1]+'.npy', allow_pickle=True)
            feature2 = np.load('input_feature_finetune/'+triple[1].split('/')[-1]+'.npy', allow_pickle=True)
            feature3 = np.load('input_feature_finetune/'+triple[2].split('/')[-1]+'.npy', allow_pickle=True)
            test_features.append((feature1, feature2, feature3))
        
        print('saving training......')
        np.save('datadrive-2/data/triplet_finetune_testing_feature_dataset_more/'+str(batch)+'.npy', test_features)
        batch += 1
        test_features = []
        test_label = []


    def load_data(self, bid, train=True):
        if train:
            del self.train_loader
            train_feature = np.load('triplet_finetune_training_feature_dataset/' + str(bid) + '.npy', allow_pickle=True)
            # train_label = np.load('triplet_finetune_training_feature_dataset/' + str(bid) + '.npy', allow_pickle=True)
            train_feature = torch.DoubleTensor(train_feature)
            # train_label = torch.LongTensor(train_label)
            # print(train_feature.shape, train_label.shape)
            train_data = TensorDataset(train_feature)
            self.train_loader = DataLoader(train_data,batch_size=self.batch_size,shuffle=True)
            del train_feature
            # del train_label
            del train_data
            gc.collect()
        else:
            test_feature = np.load('triplet_finetune_testing_feature_dataset/' + str(bid) + '.npy', allow_pickle=True)
            # test_label = np.load('finetune_testing_label_dataset/' + str(bid) + '.npy', allow_pickle=True)
            test_feature = torch.DoubleTensor(test_feature)
            test_data = TensorDataset(test_feature)
            self.test_loader = DataLoader(test_data,batch_size=1)
            del test_feature
            del test_data
        
        
    def outer_training(self):
        # self.load_feature(time)
        # self.features =np.load("features.npy")
        if self.iscontinue:
            start_epoch = self.continue_epoch
        else:
            start_epoch = 1
        for epoch in range(start_epoch, start_epoch+self.epoch_nums):
            logs[epoch] = {}
            for bid in range(1, self.train_blen+1):
                # if epoch == 11 and bid == 53:
                #     continue
                logs[epoch][bid] = {}
                print('epoch', epoch, 'bid', bid)
                self.load_data(bid, train=True)
                for i,(x, y) in enumerate(self.train_loader):
                    # x = x[0]
                    # print('x.shape', x.shape)
                    first_feature = x[:,0,:].reshape(len(x),100,10,399)
                    second_feature = x[:,1,:].reshape(len(x),100,10,399)
                    
                    first_feature = Variable(first_feature) # torch.Size([batch_size, 1000, 10])
                    second_feature = Variable(second_feature) ## torch.Size([batch_size, 1000, 10])
                    # print('first feature', first_feature.shape)
                    # print('second_feature', second_feature.shape)
                    first_feature = self.model(first_feature.to(torch.float32)) # torch.Size([128,10])
                    second_feature = self.model(second_feature.to(torch.float32))

                    self.loss = self.loss_func(first_feature,second_feature,y)
                    print('    loss:', self.loss)
                    logs[epoch][bid][i] = self.loss
                    self.opt.zero_grad()  # 清空上一步残余更新参数值
                    self.loss.backward() # 误差反向传播，计算参数更新值
                    self.opt.step() # 将参数更新值施加到net的parmeters上
            print('saving fine_tune_models/epoch_'+str(epoch))
            torch.save(self.model,'fine_tune_models/epoch_'+str(epoch))
            np.save('losslog.npy', logs)

    def initial_test(self):

        all_score = {}
        test_files = os.listdir('/datadrive-2/data/triplet_finetune_testing_feature_dataset_more/')
        self.test_blen = len(test_files)
        distance1 = []
        distance2 = []
        cos1 = []
        cos2 = []
        model = CNNnetTripletBert1010()

        hit_dis = 0
        hit_cos = 0
        all_ = 0

        res_ = ''
        res_1 = ''
        filelist = os.listdir('/datadrive-2/data/triplet_finetune_testing_feature_dataset_more/')
        filelist.sort()
        for filename in filelist:
            bid = filename.split('.')[0]
            self.retraining_load_data(bid, train=False)
            
            print('bid', bid)
    
            for i,(x) in enumerate(self.test_loader):
                x = x[0]
                # print('x.shape', x.shape)
                auchor = x[:,0,:].reshape(len(x),10,10,399)
                positive = x[:,1,:].reshape(len(x),10,10,399)
                negative = x[:,2,:].reshape(len(x),10,10,399)
                # print("######")
                # print(auchor.shape)
                # print(positive.shape)
                # print(negative.shape)
                auchor = Variable(auchor) # torch.Size([batch_size, 1000, 10])
                positive = Variable(positive) ## torch.Size([batch_size, 1000, 10])
                negative = Variable(negative) ## torch.Size([batch_size, 1000, 10])
                # print('first feature', first_feature.shape)
                # print('second_feature', second_feature.shape)
                model.eval()
                auchor = np.array(model(auchor.to(torch.float32)).detach()) # torch.Size([128,10])
                positive = np.array(model(positive.to(torch.float32)).detach())
                negative = np.array(model(negative.to(torch.float32)).detach())

                for index, emb in enumerate(auchor):
                    distance1.append(euclidean(auchor[index], positive[index]))
                    distance2.append(euclidean(auchor[index], negative[index]))

                    cos1.append(cos(auchor[index], positive[index]))
                    cos2.append(cos(auchor[index], negative[index]))
                    res_1 += str(cos1[-1]) + '\t' + str(cos2[-1]) + '\n'
                    res_ += str(distance1[-1]) + '\t' + str(distance2[-1]) + '\n'
                    if cos1[-1] < cos2[-1]:
                        hit_cos += 1
                    if distance1[-1] < distance2[-1]:
                        hit_dis += 1
                    all_ += 1
        print('hit_cos', hit_cos)
        print('hit_dis', hit_dis)
        print("all", all_)
        with open("initial_cos_distance.txt", 'w') as f:
            f.write(res_1)
        with open("initial_distance.txt", 'w') as f:
            f.write(res_)
                

    def retraining_load_data(self, bid, train):
        if train:
            print('load data')
            del self.train_loader
            # train_feature = np.load('triplet_retraining1010_training_feature_dataset/' + str(bid) + '.npy', allow_pickle=True)
            train_feature = np.load('/datadrive-2/data/triplet_finetune_training_feature_dataset_more/' + str(bid) + '.npy', allow_pickle=True)
            # train_label = np.load('triplet_finetune_training_feature_dataset/' + str(bid) + '.npy', allow_pickle=True)
            train_feature_copy = train_feature.copy()
            train_feature = torch.DoubleTensor(train_feature)
            
            new_feature = []
            print('inference.....')
            auchor = train_feature[:,0,:]
            positive = train_feature[:,1,:]
            negative = train_feature[:,2,:]

            auchor =  torch.DoubleTensor(auchor.reshape(len(auchor),10,10,399))
            positive =  torch.DoubleTensor(positive.reshape(len(positive),10,10,399))
            negative =  torch.DoubleTensor(negative.reshape(len(negative),10,10,399))

            auchor = Variable(auchor) # torch.Size([batch_size, 1000, 10])
            positive = Variable(positive) ## torch.Size([batch_size, 1000, 10])
            negative = Variable(negative) ## torch.Size([batch_size, 1000, 10])
            self.model.eval()
            auchor = self.model(auchor.to(torch.float32)) # torch.Size([128,10])
            positive = self.model(positive.to(torch.float32))
            negative = self.model(negative.to(torch.float32))
            self.model.train()
            print('select semi-hard features.....')
            for index1, lossvalue in enumerate(auchor):
                if torch.dist(auchor[index1], negative[index1]) - torch.dist(auchor[index1], positive[index1]) >= 0 and torch.dist(auchor[index1], negative[index1]) - torch.dist(auchor[index1], positive[index1]) < self.margin:
                    new_feature.append(train_feature_copy[index1])

            print('generate train loader......')
            if len(new_feature) > 1:
                new_feature = np.array(new_feature)
                new_feature = torch.DoubleTensor(new_feature)
                train_data = TensorDataset(new_feature)
            else:
                train_data = TensorDataset(train_feature)
            self.train_loader = DataLoader(train_data,batch_size=self.batch_size,shuffle=True)
            del train_feature
            del new_feature
            # del train_label
            del train_data
            gc.collect()
        else:
            # test_feature = np.load('triplet_retraining1010_testing_feature_dataset/' + str(bid) + '.npy', allow_pickle=True)
            test_feature = np.load('/datadrive-2/data/triplet_finetune_testing_feature_dataset_more/' + str(bid) + '.npy', allow_pickle=True)
            print('len(test_feature)', len(test_feature))
            # test_label = np.load('finetune_testing_label_dataset/' + str(bid) + '.npy', allow_pickle=True)
            test_feature = torch.DoubleTensor(test_feature)
            test_data = TensorDataset(test_feature)
            self.test_loader = DataLoader(test_data,batch_size=10)
            del test_feature
            del test_data
    def dynamic_load_data(self, top_size, load=False):
        if not load:
            self.loss_func = torch.nn.TripletMarginLoss(margin=self.margin, p=2, reduction='none')
            res = []
            print('dynamic load data..........')
            start_time = time.time()
            data_list = os.listdir('triplet_finetune_training_feature_dataset/')
            # data_list.sort()
            for index, data_file in enumerate(data_list):
                bid = int(data_file.split('.')[0])
                # print(index, 'load.....')
                if index == 160:
                    break
                feature = np.load('triplet_finetune_training_feature_dataset/'+data_file, allow_pickle=True)
                # print('load end')
                auchor = feature[:,0,:]
                positive = feature[:,1,:]
                negative = feature[:,2,:]
                del feature
                # auchor = np.load('input_feature_finetune/'+triple[0].split('/')[-1]+'.npy', allow_pickle=True)
                # positive = np.load('input_feature_finetune/'+triple[1].split('/')[-1]+'.npy', allow_pickle=True)
                # negative = np.load('input_feature_finetune/'+triple[2].split('/')[-1]+'.npy', allow_pickle=True)

                # print('auchor', auchor.shape)
                auchor =  torch.DoubleTensor(auchor.reshape(len(auchor),100,10,399))
                positive =  torch.DoubleTensor(positive.reshape(len(positive),100,10,399))
                negative =  torch.DoubleTensor(negative.reshape(len(negative),100,10,399))

                auchor = Variable(auchor) # torch.Size([batch_size, 1000, 10])
                positive = Variable(positive) ## torch.Size([batch_size, 1000, 10])
                negative = Variable(negative) ## torch.Size([batch_size, 1000, 10])

                auchor = self.model(auchor.to(torch.float32)) # torch.Size([128,10])
                positive = self.model(positive.to(torch.float32))
                negative = self.model(negative.to(torch.float32))

                loss = self.loss_func(auchor,positive,negative)
                for index1, lossvalue in enumerate(loss):
                    # print('torch.dist(auchor, negative)', torch.dist(auchor[index1], negative[index1]))
                    # print('torch.dist(auchor, positive)', torch.dist(auchor[index1], positive[index1]), 'data_file', data_file, 'index', index)

                    if torch.dist(auchor[index1], negative[index1]) - torch.dist(auchor[index1], positive[index1]) > 0 and torch.dist(auchor[index1], negative[index1]) - torch.dist(auchor[index1], positive[index1]) < self.margin:
                        # print('torch.dist(auchor, negative)', torch.dist(auchor[index1], negative[index1]))
                        # print('torch.dist(auchor, positive)', torch.dist(auchor[index1], positive[index1]))
                        res.append((bid*500 +index1, float(lossvalue)))
                        if len(res) == top_size:
                            break

            
                del auchor
                del positive
                del negative
                gc.collect()
                if len(res) == top_size:
                    break
                            
    
            end_time = time.time()
    
            print('dynamic load data calculate loss finished! time:', end_time - start_time, '(s)')
            # res = sorted(res, key=lambda x: x[1], reverse=True)
            end_time1 = time.time()
            # print('dynamic load data sort finished! time:', end_time1 - end_time, '(s)')
            load_index = [i[0] for i in res[0:min(top_size, len(res)+1)]]
            np.save('load_index.npy', load_index)
        else:
            load_index = np.load('load_index.npy', allow_pickle=True)
            load_index = load_index[0:top_size]
        features = []
        indexlist = []
        load_index.sort()

        for index, indexnum in enumerate(load_index):
            # bid_file = index_str.split('+')[0]
            print(index, len(load_index))
            bid = int(indexnum / 500)
            bid_file = str(bid) + '.npy'
            batchid = indexnum % 500

            if index == 0:
                before_file = bid_file

            if bid_file != before_file:
                print('start load', bid_file)
                print('before load feature:%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
                feature = np.load('triplet_finetune_training_feature_dataset/'+bid_file, allow_pickle=True)
                print('after load feature:%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
                for item in indexlist:
                    features.append(copy.copy(feature[item]))
                indexlist = []
                print('before delete feature:%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
                print('before feature count:', sys.getrefcount(feature))
                del feature
                gc.collect()
                
                print('after delete feature:%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
                # print('before feature count:', sys.getrefcount(feature))

            
            indexlist.append(batchid)
            before_file = bid_file
        
        feature = np.load('triplet_finetune_training_feature_dataset/'+bid_file, allow_pickle=True)
        
        for item in indexlist:
            features.append(copy.copy(feature[item]))
        del feature
        gc.collect()
        indexlist = []
            
        end_time2 = time.time()
        print('dynamic load data generate feature finished! time:', end_time2-end_time1, '(s)')
        train_data = TensorDataset(torch.DoubleTensor(features))
        del features
        self.train_loader = DataLoader(train_data,batch_size=len(train_data),shuffle=True)
        del train_data
        end_time3 = time.time()
        print('dynamic load data generate trian loader finished! time:', end_time3-end_time2, '(s)')
       
    def finetune_shift(self, margin, dynamic=False):
        # self.load_feature(time)
        # self.features =np.load("features.npy")
        # shift_finetune_triples
        # with open("shift_finetune_triples.json", 'r') as f:
        #     shift_finetune_triples = json.load(f)
        # with open("specific_finetune_triples.json", 'r') as f:
        # with open("specific_finetune_triples_5_10.json", 'r') as f:
        # with open("specific_finetune_triples_1_5.json", 'r') as f:
        #     shift_finetune_triples = json.load(f)
        # with open("specific_finetune_triples_5_10.json", 'r') as f:
        #     shift_finetune_triples += json.load(f)
        # with open("specific_finetune_triples.json", 'r') as f:
        #     shift_finetune_triples += json.load(f)
        with open("dedup_training_triples.json", 'r') as f:
            shift_finetune_triples = json.load(f)
        saved_shift_finetune_triples = []
        # before_path = '/datadrive-2/data/fortune500_test/training_before_specific_c2/'
        # before_path = '/datadrive-2/data/fortune500_test/training_before_shift_c2/'
        # before_path = '/datadrive-2/data/fortune500_test/cross_training_before_specific_c2/'
        before_path = '/datadrive-2/data/fortune500_test/training_before_features/'
        for triple in shift_finetune_triples:
            auchor = triple[0].split("/")[-1]
            positive = triple[1].split("/")[-1]
            negative = triple[2].split("/")[-1]

            if not os.path.exists(before_path + auchor + '.npy') or not os.path.exists(before_path + positive + '.npy') or not os.path.exists(before_path + negative + '.npy'):
                continue
            saved_shift_finetune_triples.append(triple)
        shift_finetune_triples = saved_shift_finetune_triples
        self.margin=margin
        
        if self.iscontinue:
            start_epoch = self.continue_epoch
        else:
            start_epoch = 1
        self.loss_func = torch.nn.TripletMarginLoss(margin=self.margin, p=2)
        # training_triples_files = os.listdir('/datadrive-2/data/fortune500_test/demo_tile_features_specific')
        training_triples_files = os.listdir('/datadrive-2/data/fortune500_test/training_tile_features')
        # training_triples_files = os.listdir('/datadrive-2/data/fortune500_test/demo_tile_features_shift')
        for epoch in range(start_epoch, start_epoch+self.epoch_nums):
            logs[epoch] = {}
            print('epoch', epoch)
            # if dynamic:
            #     self.dynamic_load_data(top_size=500)
            #     self.loss_func = torch.nn.TripletMarginLoss(margin=self.margin, p=2)
            #     for i,(x) in enumerate(self.train_loader):
            #         x = x[0]
            #         # print('x.shape', x.shape)
            #         # print(x)
            #         auchor = x[:,0,:].reshape(len(x),100,10,399)
            #         positive = x[:,1,:].reshape(len(x),100,10,399)
            #         negative = x[:,2,:].reshape(len(x),100,10,399)
                    
            #         auchor = Variable(auchor) # torch.Size([batch_size, 1000, 10])
            #         positive = Variable(positive) ## torch.Size([batch_size, 1000, 10])
            #         negative = Variable(negative) ## torch.Size([batch_size, 1000, 10])
            #         auchor = self.model(auchor.to(torch.float32)) # torch.Size([128,10])
            #         positive = self.model(positive.to(torch.float32))
            #         negative = self.model(negative.to(torch.float32))

            #         self.loss = self.loss_func(auchor,positive,negative)
            #         print('epoch',epoch,'loss:', self.loss)
            #         logs[epoch][i] = self.loss
            #         self.opt.zero_grad()  # 清空上一步残余更新参数值
            #         self.loss.backward() # 误差反向传播，计算参数更新值
            #         self.opt.step() # 将参数更新值施加到net的parmeters上
            #     print('saving semi_triplet_fine_tune_models/epoch_'+str(epoch))
            #     torch.save(self.model,'semi_triplet_fine_tune_models/epoch_'+str(epoch))
            #     np.save('semi-losslog.npy', logs)
            # else:
            cursor = 0
            while cursor < len(shift_finetune_triples):
                print('cursor', cursor, len(shift_finetune_triples))
                training_triples_list = []
                print('self.batch_size', self.batch_size)
                while len(training_triples_list) < self.batch_size and cursor < len(shift_finetune_triples):
                    auchor = shift_finetune_triples[cursor][0].split("/")[-1]
                    positive = shift_finetune_triples[cursor][1].split("/")[-1]
                    negative = shift_finetune_triples[cursor][2].split("/")[-1]

                    auchor_feature = np.load(before_path + auchor + '.npy', allow_pickle=True)
                    positive_feature = np.load(before_path + positive + '.npy', allow_pickle=True)
                    negative_feature = np.load(before_path + negative + '.npy', allow_pickle=True)
                    training_triples_list.append([auchor_feature, positive_feature, negative_feature])
                    cursor += 1
                # print("training_triples_list", training_triples_list)
                training_triples_features = np.array(training_triples_list)
                training_triples_features = torch.DoubleTensor(training_triples_features)
                print('training_triples_list', training_triples_features.shape)
                train_data = TensorDataset(training_triples_features)
                self.train_loader = DataLoader(train_data,batch_size=self.batch_size,shuffle=True)
                for i,(x) in enumerate(self.train_loader):
                    x = x[0]
                    # print('x.shape', x.shape)
                    # print(x)
                    auchor = x[:,0,:].reshape(len(x),100,10,399)
                    positive = x[:,1,:].reshape(len(x),100,10,399)
                    negative = x[:,2,:].reshape(len(x),100,10,399)
                    
                    auchor = Variable(auchor) # torch.Size([batch_size, 1000, 10])
                    positive = Variable(positive) ## torch.Size([batch_size, 1000, 10])
                    negative = Variable(negative) ## torch.Size([batch_size, 1000, 10])
        
                    auchor = self.model(auchor.to(torch.float32)) # torch.Size([128,10])
                    positive = self.model(positive.to(torch.float32))
                    negative = self.model(negative.to(torch.float32))

                    self.loss = self.loss_func(auchor,positive,negative)
                    print('epoch',epoch,'loss:', self.loss)
                    logs[epoch][i] = self.loss
                    self.opt.zero_grad()  # 清空上一步残余更新参数值
                    self.loss.backward() # 误差反向传播，计算参数更新值
                    self.opt.step() # 将参数更新值施加到net的parmeters上
                if self.l2:
                    # print('saving /datadrive-2/data/finetune_specific_l2_new/epoch_'+str(epoch))
                    # torch.save(self.model, '/datadrive-2/data/finetune_specific_l2_new/epoch_'+str(epoch))
                    # np.save('/datadrive-2/data/finetune_specific_l2_new/losslog.npy', logs)
                    # print('saving /datadrive-2/data/finetune_specific_l2_5_10/epoch_'+str(epoch))
                    # torch.save(self.model, '/datadrive-2/data/finetune_specific_l2_5_10/epoch_'+str(epoch))
                    # np.save('/datadrive-2/data/finetune_specific_l2_5_10/losslog.npy', logs)
                    # print('saving /datadrive-2/data/cross_finetune_specific_l2/epoch_'+str(epoch))
                    # torch.save(self.model, '/datadrive-2/data/cross_finetune_specific_l2/epoch_'+str(epoch)) # finetune_specific_l2_1_5
                    # np.save('/datadrive-2/data/cross_finetune_specific_l2/losslog.npy', logs)
                    # print('saving /datadrive-2/data/training_ref_model/epoch_'+str(epoch))
                    # torch.save(self.model, '/datadrive-2/data/training_ref_model/epoch_'+str(epoch)) # finetune_specific_l2_1_5
                    # np.save('/datadrive-2/data/training_ref_model/losslog.npy', logs)
                    # print('saving /datadrive-2/data/finegrained_model/epoch_'+str(epoch))
                    # torch.save(self.model, '/datadrive-2/data/finegrained_model/epoch_'+str(epoch)) # finetune_specific_l2_1_5
                    # np.save('/datadrive-2/data/finegrained_model/losslog.npy', logs)
                    print('saving /datadrive-2/data/finegrained_model_16/epoch_'+str(epoch))
                    torch.save(self.model, '/datadrive-2/data/finegrained_model_16/epoch_'+str(epoch)) # finetune_specific_l2_1_5
                    np.save('/datadrive-2/data/finegrained_model_16/losslog.npy', logs)
                    # print('saving /datadrive-2/data/finetune_shift_new_l2/epoch_'+str(epoch))
                    # torch.save(self.model, '/datadrive-2/data/finetune_shift_new_l2/epoch_'+str(epoch))
                    # np.save('/datadrive-2/data/finetune_shift_new_l2/losslog.npy', logs)
                # else:
                #     print('saving /datadrive-2/data/fintune_shift_models/epoch_'+str(epoch))
                #     torch.save(self.model, '/datadrive-2/data/fintune_shift_models/epoch_'+str(epoch))
                #     np.save('/datadrive-2/data/fintune_shift_models/losslog.npy', logs)
            
    def finetune_content_shift(self, margin, dynamic=False):
        # self.load_feature(time)
        # self.features =np.load("features.npy")
        # with open("shift_content_finetune_triples.json", 'r') as f:
        with open("shift_content_finetune_triples.json", 'r') as f:
            shift_finetune_triples = json.load(f)
        saved_shift_finetune_triples = []
        # before_path = '/datadrive-2/data/fortune500_test/demo_before_features/'
        before_path = '/datadrive-2/data/fortune500_test/training_before_features_shift_content_mask2/'
        for triple in shift_finetune_triples:
            auchor = triple[0].split("/")[-1]
            positive = triple[1].split("/")[-1]
            negative = triple[2].split("/")[-1]

            if not os.path.exists(before_path + auchor + '.npy') or not os.path.exists(before_path + positive + '.npy') or not os.path.exists(before_path + negative + '.npy'):
                continue
            saved_shift_finetune_triples.append(triple)
        shift_finetune_triples = saved_shift_finetune_triples
        self.margin=margin
        
        if self.iscontinue:
            start_epoch = self.continue_epoch
        else:
            start_epoch = 1
        self.loss_func = torch.nn.TripletMarginLoss(margin=self.margin, p=2)
        for epoch in range(start_epoch, start_epoch+self.epoch_nums):
            logs[epoch] = {}
            print('epoch', epoch)
            if dynamic:
                self.dynamic_load_data(top_size=500)
                self.loss_func = torch.nn.TripletMarginLoss(margin=self.margin, p=2)
                for i,(x) in enumerate(self.train_loader):
                    x = x[0]
                    # print('x.shape', x.shape)
                    # print(x)
                    auchor = x[:,0,:].reshape(len(x),100,10,399)
                    positive = x[:,1,:].reshape(len(x),100,10,399)
                    negative = x[:,2,:].reshape(len(x),100,10,399)
                    
                    auchor = Variable(auchor) # torch.Size([batch_size, 1000, 10])
                    positive = Variable(positive) ## torch.Size([batch_size, 1000, 10])
                    negative = Variable(negative) ## torch.Size([batch_size, 1000, 10])
                    auchor = self.model(auchor.to(torch.float32)) # torch.Size([128,10])
                    positive = self.model(positive.to(torch.float32))
                    negative = self.model(negative.to(torch.float32))

                    self.loss = self.loss_func(auchor,positive,negative)
                    print('epoch',epoch,'loss:', self.loss)
                    logs[epoch][i] = self.loss
                    self.opt.zero_grad()  # 清空上一步残余更新参数值
                    self.loss.backward() # 误差反向传播，计算参数更新值
                    self.opt.step() # 将参数更新值施加到net的parmeters上
                print('saving semi_triplet_fine_tune_models/epoch_'+str(epoch))
                torch.save(self.model,'semi_triplet_fine_tune_models/epoch_'+str(epoch))
                np.save('semi-losslog.npy', logs)
            else:
                cursor = 0
                while cursor < len(shift_finetune_triples):
                    print('cursor', cursor, len(shift_finetune_triples))
                    training_triples_list = []
                    print('self.batch_size', self.batch_size)
                    while len(training_triples_list) < self.batch_size and cursor < len(shift_finetune_triples):
                        auchor = shift_finetune_triples[cursor][0].split("/")[-1]
                        positive = shift_finetune_triples[cursor][1].split("/")[-1]
                        negative = shift_finetune_triples[cursor][2].split("/")[-1]

                        auchor_feature = np.load(before_path + auchor + '.npy', allow_pickle=True)
                        positive_feature = np.load(before_path + positive + '.npy', allow_pickle=True)
                        negative_feature = np.load(before_path + negative + '.npy', allow_pickle=True)
                        training_triples_list.append([auchor_feature, positive_feature, negative_feature])
                        cursor += 1
                    # print("training_triples_list", training_triples_list)
                    training_triples_features = np.array(training_triples_list)
                    training_triples_features = torch.DoubleTensor(training_triples_features)
                    print('training_triples_list', training_triples_features.shape)
                    train_data = TensorDataset(training_triples_features)
                    self.train_loader = DataLoader(train_data,batch_size=self.batch_size,shuffle=True)
                    for i,(x) in enumerate(self.train_loader):
                        x = x[0]
                        # print('x.shape', x.shape)
                        # print(x)
                        auchor = x[:,0,:].reshape(len(x),100,10,399)
                        positive = x[:,1,:].reshape(len(x),100,10,399)
                        negative = x[:,2,:].reshape(len(x),100,10,399)
                        
                        auchor = Variable(auchor) # torch.Size([batch_size, 1000, 10])
                        positive = Variable(positive) ## torch.Size([batch_size, 1000, 10])
                        negative = Variable(negative) ## torch.Size([batch_size, 1000, 10])
            
                        auchor = self.model(auchor.to(torch.float32)) # torch.Size([128,10])
                        positive = self.model(positive.to(torch.float32))
                        negative = self.model(negative.to(torch.float32))

                        self.loss = self.loss_func(auchor,positive,negative)
                        print('epoch',epoch,'loss:', self.loss)
                        logs[epoch][i] = self.loss
                        self.opt.zero_grad()  # 清空上一步残余更新参数值
                        self.loss.backward() # 误差反向传播，计算参数更新值
                        self.opt.step() # 将参数更新值施加到net的parmeters上
                    if self.l2:
                        print('saving /datadrive-2/data/finetune_shift_content_models_l2/epoch_'+str(epoch))
                        torch.save(self.model, '/datadrive-2/data/fintune_shift_models_l2/epoch_'+str(epoch))
                        np.save('/datadrive-2/data/finetune_shift_content_models_l2/losslog.npy', logs)
                    else:
                        # print('saving /datadrive-2/data/c2_ml_sc2/epoch_'+str(epoch))
                        # torch.save(self.model, '/datadrive-2/data/c2_ml_sc2/epoch_'+str(epoch))
                        # np.save('/datadrive-2/data/c2_ml_sc2/losslog.npy', logs)
                        print('saving /datadrive-2/data/sheetlevel_4_1/epoch_'+str(epoch))
                        torch.save(self.model, '/datadrive-2/data/sheetlevel_4_1/epoch_'+str(epoch))
                        np.save('/datadrive-2/data/sheetlevel_4_1/losslog.npy', logs)
            
    def outer_triplet_training(self, margin, dynamic= False):
        # self.load_feature(time)
        # self.features =np.load("features.npy")
        self.margin=margin
        
        if self.iscontinue:
            start_epoch = self.continue_epoch
        else:
            start_epoch = 1
        for epoch in range(start_epoch, start_epoch+self.epoch_nums):
            logs[epoch] = {}
            print('epoch', epoch)
            if dynamic:
                self.dynamic_load_data(top_size=500)
                self.loss_func = torch.nn.TripletMarginLoss(margin=self.margin, p=2)
                for i,(x) in enumerate(self.train_loader):
                    x = x[0]
                    # print('x.shape', x.shape)
                    # print(x)
                    auchor = x[:,0,:].reshape(len(x),100,10,399)
                    positive = x[:,1,:].reshape(len(x),100,10,399)
                    negative = x[:,2,:].reshape(len(x),100,10,399)
                    
                    auchor = Variable(auchor) # torch.Size([batch_size, 1000, 10])
                    positive = Variable(positive) ## torch.Size([batch_size, 1000, 10])
                    negative = Variable(negative) ## torch.Size([batch_size, 1000, 10])
        
                    auchor = self.model(auchor.to(torch.float32)) # torch.Size([128,10])
                    positive = self.model(positive.to(torch.float32))
                    negative = self.model(negative.to(torch.float32))

                    self.loss = self.loss_func(auchor,positive,negative)
                    print('epoch',epoch,'loss:', self.loss)
                    logs[epoch][i] = self.loss
                    self.opt.zero_grad()  # 清空上一步残余更新参数值
                    self.loss.backward() # 误差反向传播，计算参数更新值
                    self.opt.step() # 将参数更新值施加到net的parmeters上
                print('saving semi_triplet_fine_tune_models/epoch_'+str(epoch))
                torch.save(self.model,'semi_triplet_fine_tune_models/epoch_'+str(epoch))
                np.save('semi-losslog.npy', logs)
            else:
                self.loss_func = torch.nn.TripletMarginLoss(margin=self.margin, p=2)
                feature_files = os.listdir('triplet_finetune_training_feature_dataset')
                for bid in range(1, len(feature_files)+1):
                
                    logs[epoch][bid] = {}
                    print('epoch', epoch, 'bid', bid)
                    self.load_data(bid, train=True)
                    for i,(x) in enumerate(self.train_loader):
                        x = x[0]
                        # print('x.shape', x.shape)
                        # print(x)
                        auchor = x[:,0,:].reshape(len(x),100,10,399)
                        positive = x[:,1,:].reshape(len(x),100,10,399)
                        negative = x[:,2,:].reshape(len(x),100,10,399)
                        
                        auchor = Variable(auchor) # torch.Size([batch_size, 1000, 10])
                        positive = Variable(positive) ## torch.Size([batch_size, 1000, 10])
                        negative = Variable(negative) ## torch.Size([batch_size, 1000, 10])
            
                        auchor = self.model(auchor.to(torch.float32)) # torch.Size([128,10])
                        positive = self.model(positive.to(torch.float32))
                        negative = self.model(negative.to(torch.float32))

                        self.loss = self.loss_func(auchor,positive,negative)
                        print('epoch',epoch,'loss:', self.loss)
                        logs[epoch][i] = self.loss
                        self.opt.zero_grad()  # 清空上一步残余更新参数值
                        self.loss.backward() # 误差反向传播，计算参数更新值
                        self.opt.step() # 将参数更新值施加到net的parmeters上
                    if self.l2:
                        print('saving l2norm_triplet_all_fine_tune_models/epoch_'+str(epoch))
                        torch.save(self.model,'l2norm_triplet_all_fine_tune_models/epoch_'+str(epoch))
                        np.save('l2norm_triplet-all-losslog.npy', logs)
                    else:
                        print('saving triplet_all_fine_tune_models_1/epoch_'+str(epoch))
                        torch.save(self.model,'triplet_all_fine_tune_models_1/epoch_'+str(epoch))
                        np.save('triplet-all-losslog.npy', logs)
            
    def retraining_model_2(self, margin):
        self.margin=margin
        if self.iscontinue:
            start_epoch = self.continue_epoch
        else:
            start_epoch = 1

        train_list = os.listdir('/datadrive-2/data/triplet_finetune_testing_feature_dataset_more/')
        for epoch in range(start_epoch, start_epoch+self.epoch_nums):
            logs[epoch] = {}
            print('epoch', epoch)
            for bid_file in train_list:
                bid = bid_file.split('.')[0]
                self.retraining_load_data(bid, train=True)
                self.loss_func = torch.nn.TripletMarginLoss(margin=self.margin, p=2)
                for i,(x) in enumerate(self.train_loader):
                    try:
                        x = x[0]
                        # print('x.shape', x.shape)
                        # print(x)
                        auchor = x[:,0,:].reshape(len(x),10,10,399)
                        positive = x[:,1,:].reshape(len(x),10,10,399)
                        negative = x[:,2,:].reshape(len(x),10,10,399)
                        
                        auchor = Variable(auchor) # torch.Size([batch_size, 1000, 10])
                        positive = Variable(positive) ## torch.Size([batch_size, 1000, 10])
                        negative = Variable(negative) ## torch.Size([batch_size, 1000, 10])
            
                        auchor = self.model(auchor.to(torch.float32)) # torch.Size([128,10])
                        positive = self.model(positive.to(torch.float32))
                        negative = self.model(negative.to(torch.float32))

                        self.loss = self.loss_func(auchor,positive,negative)
                        print('epoch',epoch,'loss:', self.loss)
                        logs[epoch][i] = self.loss
                        self.opt.zero_grad()  # 清空上一步残余更新参数值
                        self.loss.backward() # 误差反向传播，计算参数更新值
                        self.opt.step() # 将参数更新值施加到net的parmeters上
                    except:
                        continue
                print('saving model2/epoch_'+str(epoch))
                torch.save(self.model,'model2/epoch_'+str(epoch))
                np.save('log_retraining_model2.npy', logs)
           
    def retrain_test(self, model_path, save_path):
        model_filenames = os.listdir(model_path)
        # model_filenames = os.listdir('triplet_all_fine_tune_models')
        model_filenames.sort()
        all_score = {}
        test_files = os.listdir('/datadrive-2/data/triplet_finetune_testing_feature_dataset_more')
        self.test_blen = len(test_files)
        distance1 = {}
        distance2 = {}
        for testfile in test_files:
            bid = testfile.split('.')[0]
            self.retraining_load_data(bid, train=False)
            for filename in model_filenames:
                if filename not in distance1:
                    distance1[filename] = []
                    distance2[filename] = []
                epoch = int(filename.split('_')[1])
                if epoch not in all_score:
                    all_score[epoch] = {}
                print('epoch', epoch)
                model = torch.load(model_path + '/'+filename)
            
                print('epoch', epoch, 'bid', bid)
        
                for i,(x) in enumerate(self.test_loader):
                    x = x[0]
                    # print('x.shape', x.shape)
                    auchor = x[:,0,:].reshape(len(x),10,10,399)
                    positive = x[:,1,:].reshape(len(x),10,10,399)
                    negative = x[:,2,:].reshape(len(x),10,10,399)
                    print("######")
                    print(auchor.shape)
                    print(positive.shape)
                    print(negative.shape)
                    auchor = Variable(auchor) # torch.Size([batch_size, 1000, 10])
                    positive = Variable(positive) ## torch.Size([batch_size, 1000, 10])
                    negative = Variable(negative) ## torch.Size([batch_size, 1000, 10])
                    # print('first feature', first_feature.shape)
                    # print('second_feature', second_feature.shape)
                    model.eval()
                    auchor = np.array(model(auchor.to(torch.float32)).detach()) # torch.Size([128,10])
                    positive = np.array(model(positive.to(torch.float32)).detach())
                    negative = np.array(model(negative.to(torch.float32)).detach())

                    for index, emb in enumerate(auchor):
                        distance1[filename].append(np.linalg.norm(auchor[index]-positive[index]))
                        distance2[filename].append(np.linalg.norm(auchor[index]-negative[index]))

                    
                    # # cos = torch.cosine_similarity(first_feature,second_feature,dim=1)
                    # print(cos)
                    # print(y)
                    # if 'y_score' not in all_score[epoch]:
                    #     all_score[epoch]['y_score'] = []
                    # if 'y_label' not in all_score[epoch]:
                    #     all_score[epoch]['y_label'] = []
                    # all_score[epoch]['y_score'].append(float(cos.detach()[0]))
                    # all_score[epoch]['y_label'].append(int(y[0]))
        np.save(save_path + '/distance1.npy', distance1)
        np.save(save_path + '/distance2.npy', distance2)
    
    def resave_distance(self, save_path):
        distance1 = np.load(save_path + '/distance1.npy', allow_pickle=True).item()
        distance2 = np.load(save_path + '/distance2.npy', allow_pickle=True).item()

        look_res = {}
        res = ""
        for keyw in distance1:
            if keyw != 'epoch_55':
                continue
            print('#####')
            print(keyw)
            # print(distance1[key][0:10])
            # print(distance2[key][0:10])
            
            look_res[int(keyw.split('_')[1])] = []
            for index,item in enumerate(distance1[keyw]):
                res += str(distance1[keyw][index]) + ' ' + str(distance2[keyw][index]) + '\n'
        # look_res = sorted(look_res.items(), key=lambda x: x[0])
        # pprint.pprint(look_res)
        # for key in distance1:
        #     distance1[key] = np.array(distance1[key]).tolist()
        # for key in distance2:
        #     distance2[key] = np.array(distance2[key]).tolist()
        # with open(save_path + '/distance1.json', 'w') as f:
        #     json.dump(distance1, f)
        with open(save_path + '/distance.txt', 'w') as f:
            f.write(res)

    def test(self, model_path, save_path):
        model_filenames = os.listdir(model_path)
        # model_filenames = os.listdir('triplet_all_fine_tune_models')
        model_filenames.sort()
        all_score = {}
        test_files = os.listdir('triplet_finetune_testing_feature_dataset')
        self.test_blen = len(test_files)
        distance1 = {}
        distance2 = {}
        for bid in range(1, self.test_blen+1):
            self.load_data(bid, train=False)
            for filename in model_filenames:
                if filename not in distance1:
                    distance1[filename] = []
                    distance2[filename] = []
                epoch = int(filename.split('_')[1])
                if epoch not in all_score:
                    all_score[epoch] = {}
                print('epoch', epoch)
                model = torch.load(model_path + '/'+filename)
            
                print('epoch', epoch, 'bid', bid)
        
                for i,(x) in enumerate(self.test_loader):
                    x = x[0]
                    # print('x.shape', x.shape)
                    auchor = x[:,0,:].reshape(len(x),100,10,399)
                    positive = x[:,1,:].reshape(len(x),100,10,399)
                    negative = x[:,2,:].reshape(len(x),100,10,399)
                    
                    auchor = Variable(auchor) # torch.Size([batch_size, 1000, 10])
                    positive = Variable(positive) ## torch.Size([batch_size, 1000, 10])
                    negative = Variable(negative) ## torch.Size([batch_size, 1000, 10])
                    # print('first feature', first_feature.shape)
                    # print('second_feature', second_feature.shape)
                    auchor = np.array(model(auchor.to(torch.float32)).detach()) # torch.Size([128,10])
                    positive = np.array(model(positive.to(torch.float32)).detach())
                    negative = np.array(model(negative.to(torch.float32)).detach())

                    for index, emb in enumerate(auchor):
                        distance1[filename].append(np.linalg.norm(auchor[index]-positive[index]))
                        distance2[filename].append(np.linalg.norm(auchor[index]-negative[index]))

                    
                    # # cos = torch.cosine_similarity(first_feature,second_feature,dim=1)
                    # print(cos)
                    # print(y)
                    # if 'y_score' not in all_score[epoch]:
                    #     all_score[epoch]['y_score'] = []
                    # if 'y_label' not in all_score[epoch]:
                    #     all_score[epoch]['y_label'] = []
                    # all_score[epoch]['y_score'].append(float(cos.detach()[0]))
                    # all_score[epoch]['y_label'].append(int(y[0]))
        np.save(save_path + '/distance1.npy', distance1)
        np.save(save_path + '/distance2.npy', distance2)
        # np.save('fine_tune_test_result.npy', all_score)

def look_test_result():
    """
    {
        epoch:{
            'y_score': [],
            'y_label': [],
        }
    }    
    """
    res = np.load('fine_tune_test_result.npy', allow_pickle=True).item()
    epoach0 = list(res.keys())[0]
    new_res = {}
    print(len(res[epoach0]['y_score']))
    print(len(res[epoach0]['y_label']))
    new_res['origin'] ={}
    new_res['origin']['y_score'] = res[epoach0]['y_score']
    new_res['origin']['y_label'] = res[epoach0]['y_score']
    np.save('finetune_origin_test_result.npy', new_res)

def pr_distance_curve():
    distance1 = np.load('model2_distance/distance1.npy', allow_pickle=True).item()
    distance2 = np.load('model2_distance/distance2.npy', allow_pickle=True).item()
    # res = np.load('fine_tune_test_result.npy', allow_pickle=True).item()
    for epoach in distance1:
        # epoach_id = int(epoach.split('_')[-1])
        # if epoach_id% 10 != 0 or epoach_id == 5 or epoach_id == 10:
            # continue
        print('epoach', epoach)
        if int(epoach.split('_')[-1]) % 5 != 0:
            continue
        y_score = []
        y_label = []
        for score in distance1[epoach]:
            y_score.append(0-float(score))
            y_label.append(1)
        for score in distance2[epoach]:
            y_score.append(0-float(score))
            y_label.append(0)

        # print('yscore', y_score[0:10])
        # print('ylabel', y_label)
        precision,recall,threshold = precision_recall_curve(y_label,y_score) ###计算真正率和假正率
        # print('precision', precision[-20:])
        # print('recall', recall[-20:])
        f1_list = []
        for index, one_p in enumerate(threshold):
            f1_list.append(2*precision[index]*recall[index]/(precision[index] + recall[index]))
        # print("#####")
        # print('recall',recall)
        # print('precision',precision)
        # print('threshold', len(threshold))
        # print('f1_list', len(f1_list))
        # plt.clf()
        plt.plot(threshold, f1_list, lw=2,label=epoach)
        # plt.plot(recall, precision, lw=2,label=epoach)

    plt.xlabel('threshold')
    # plt.xlabel('recall')
    plt.ylabel('f1')
    # plt.ylabel('precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.savefig("model2t_pr.png")

def pr_curve():
    
    res = np.load('fine_tune_test_result.npy', allow_pickle=True).item()
    for epoach in res:
        print('epoach', epoach)
        if epoach == 10 or epoach == 0:
            continue
        y_score = res[epoach]['y_score']
        y_label = res[epoach]['y_label']
        
        for index, item in enumerate(y_score):
            y_score[index] = 0-float(item)
        for index, item in enumerate(y_label):
            if int(item) == -1:
                y_label[index] = 0
        # print(y_score)
        # print(y_label)
        precision,recall,threshold = precision_recall_curve(y_label,y_score) ###计算真正率和假正率
        f1_list = []
        for index,item in enumerate(threshold):
            f1_list.append(2*precision[index]*recall[index]/(precision[index]  + recall[index]))
        
        # print("#####")
        # print('recall',recall)
        # print('precision',precision)
        # print('threshold', threshold)
        # plt.clf()
        # plt.plot(recall, precision, lw=2,label=epoach)
        plt.plot(threshold, f1_list, lw=2,label=epoach)

    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    plt.xlabel('threshold')
    plt.ylabel('f1')
    plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.savefig("finetune_test_result_f1_1.png")

def look_neg_pair():
    with open('fine_tune_testing_pair.json', 'r') as f:
        testing_pairs = json.load(f)
    pprint.pprint(testing_pairs)

def look_triplet():
    with open('finetune_triplet_more.json', 'r') as f:
        res = json.load(f)
    print(len(res))
    # for item in res:
        # pr
    result = []
    for index,item in enumerate(res):
        # print('#####')
        print(index, len(res))
        # print(item)
        if not os.path.exists('input_feature_finetune/'+item[0].split('/')[-1]+'.npy') or not os.path.exists('input_feature_finetune/'+item[1].split('/')[-1]+'.npy') or not os.path.exists('input_feature_finetune/'+item[2].split('/')[-1]+'.npy') :
            continue
        result.append(item)
    print(len(result))
    with open('saved_finetune_triplet_more.json', 'w') as f:
        json.dump(result, f)
        # print(item[1])
        # print(item[2])
    #     temp1 = item[0][-1]
    #     temp2 = item[1].split
    #     first_token = 
def test_del_collect():
    print('before set a:%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    a = range(1000000000)
    print('after set a:%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    del a
    gc.collect()
    print('after delete a:%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

def look_saved_finetune_triplet():
    with open('finetune_triplet_more.json', 'r') as f:
        triples = json.load(f)
        test_triples = triples
    filenames = {}
    for triple in triples:
        for token in triple:
            filename = token.split('---')[1]
            if filename not in filenames:
                filenames[filename] = 0
            filenames[filename] += 1

    print(filenames)

def relative_eval():
    # with open('saved_finetune_triplet_sampled.json', 'r') as f:
    #     triples = json.load(f)
    #     test_triples = triples[110000:150000]

    # distance1 = np.load('triplet_finetune_test_distance/semi_triplet_distance1.npy', allow_pickle=True).item()
    # distance2 = np.load('triplet_finetune_test_distance/semi_triplet_all_distance2.npy', allow_pickle=True).item()
    # distance1 = np.load('triplet_all_distance_1/distance1.npy', allow_pickle=True).item()
    # distance2 = np.load('triplet_all_distance_1/distance2.npy', allow_pickle=True).item()
    # distance1 = np.load('l2n_triplet_all_distance/distance1.npy', allow_pickle=True).item()
    # distance2 = np.load('l2n_triplet_all_distance/distance2.npy', allow_pickle=True).item()
    # distance1 = np.load('triplet_finetune_test_distance/distance1.npy', allow_pickle=True).item()
    # distance2 = np.load('triplet_finetune_test_distance/distance2.npy', allow_pickle=True).item()
    distance1 = np.load('model2_distance/distance1.npy', allow_pickle=True).item()
    distance2 = np.load('model2_distance/distance2.npy', allow_pickle=True).item()

    res = []
    res1 = []
    for epoach in distance1:
        dis = []
        hit = 0
        count = 0
        # if int(epoach.split('_')[1]) != 1:
        #     continue
        # if int(epoach.split('_')[1]) % 10 == 0:
        filenames = {}
        print(len(distance1[epoach]))
        for index, p_distance in enumerate(distance1[epoach]):
            n_distance = distance2[epoach][index]
            disdis = n_distance - p_distance
            # print('n_distance', n_distance)
            # print('p_distance', p_distance)
            # print('disdis', disdis)
            # print('test_triples', test_triples[index])
            # for token in test_triples[index]:
            #     filename = token.split('---')[1]
            #     if filename not in filenames:
            #         filenames[filename] = 0
            #     filenames[filename] += 1
            dis.append(disdis)
            if n_distance > p_distance:
                hit += 1
            # else:
            #     # res1.append((n_distance, p_distance, disdis, test_triples[index]))
            #     print('#################')
            #     print('n_distance', n_distance)
            #     print('p_distance', p_distance)
                # print('disdis', disdis)
                # print(test_triples[index])
            count += 1
        # print('filenames', filenames)
            # print('disdis', disdis)
        # print('epoach', epoach, hit/count)
        res.append((int(epoach.split('_')[1]),hit/count))
        # break
    # res1 = random.sample(res, 10)
    # print(res1)
    # for item in res1:
    #     print("##########")
    #     print('n_distance', item[0])
    #     print('p_distance', item[1])
    #     print('disdis', item[2])
    #     print('index', item[3])
    res = sorted(res, key=lambda x:x[0])
    # pprint.pprint(res)
    # res = [i[1] for i in res]
    print(res)
    # plt.plot(res)
    # plt.savefig("model2_training.png")

def look_one_batch_size():
    print('before load feature:%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    feature = np.load('triplet_finetune_training_feature_dataset/1.npy', allow_pickle=True)
    print('after load feature:%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

def save_triplet_data_1010(thread_id, batch_num):
    print('generate data.......')
    with open('saved_finetune_triplet_sampled.json', 'r') as f:
        triples = json.load(f)
    batch = 1
    training_triples = triples[0:5*int(len(triples)/6)]
    train_features = []
    batch_len = len(training_triples)/batch_num

    train_index2triple = {}
    test_index2triple = {}
    temp_triple = []
    for index,triple in enumerate(training_triples):
        if thread_id != batch_num:
            if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1 ):
                continue
        
        if len(train_features) == 5000:
            print('saving training......')
            np.save('/datadrive-2/data/triplet_finetune_training_feature_dataset_more/'+str(thread_id)+'_'+str(batch)+'.npy', train_features)
            train_index2triple[str(index)+'_'+str(batch)] = temp_triple
            batch += 1
            train_features = []
            temp_triple = []

        print(index, len(training_triples))
        feature1 = np.load('input_feature_finetune_1010/'+triple[0].split('/')[-1]+'.npy', allow_pickle=True)
        feature2 = np.load('input_feature_finetune_1010/'+triple[1].split('/')[-1]+'.npy', allow_pickle=True)
        feature3 = np.load('input_feature_finetune_1010/'+triple[2].split('/')[-1]+'.npy', allow_pickle=True)
        train_features.append((feature1, feature2, feature3))
        temp_triple.append(triple)
    
    train_index2triple[str(index)+'_'+str(batch)] = temp_triple
    print('saving training......')
    # np.save('triplet_retraining1010_training_feature_dataset/'+str(batch)+'.npy', train_features)
    np.save('/datadrive-2/data/triplet_finetune_training_feature_dataset_more/'+str(thread_id)+'_'+str(batch)+'.npy', train_features)
    train_features = []
    temp_triple = []

    test_triples = triples[5*int(len(triples)/6):]
    test_features = []
    batch = 1

    batch_len = len(test_triples)/batch_num
    for index,triple in enumerate(test_triples):
        if thread_id != batch_num:
            if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1 ):
                continue
        if len(test_features) == 5000:
            print('saving training......')
            test_index2triple[str(thread_id)+'_'+str(batch)] = temp_triple 
            np.save('/datadrive-2/data/triplet_finetune_testing_feature_dataset_more/'+str(thread_id)+'_'+str(batch)+'.npy', test_features)
            batch += 1
            test_features = []
            temp_triple = []

        print(index, len(test_triples))
        feature1 = np.load('input_feature_finetune_1010/'+triple[0].split('/')[-1]+'.npy', allow_pickle=True)
        feature2 = np.load('input_feature_finetune_1010/'+triple[1].split('/')[-1]+'.npy', allow_pickle=True)
        feature3 = np.load('input_feature_finetune_1010/'+triple[2].split('/')[-1]+'.npy', allow_pickle=True)
        test_features.append((feature1, feature2, feature3))
    
    print('saving testing......')
    test_index2triple[str(thread_id)+'_'+str(batch)] = temp_triple 
    # np.save('triplet_retraining1010_testing_feature_dataset/'+str(batch)+'.npy', test_features)
    np.save('/datadrive-2/data/triplet_finetune_testing_feature_dataset_more/'+str(thread_id)+'_'+str(batch)+'.npy', test_features)
    batch += 1
    test_features = []
    temp_triple = []
    test_label = []
    with open('test_index2triple_'+str(thread_id)+'.json', 'w') as f:
        json.dump(test_index2triple, f)
    with open('train_index2triple_'+str(thread_id)+'.json', 'w') as f:
        json.dump(train_index2triple, f)

def save_neighbor_triplet_data_1010(thread_id, batch_num):
    print('generate data.......')
    with open('finetune_neighbor_triplet.json', 'r') as f:
        triples = json.load(f)
    batch = 1
    training_triples = triples[0:5*int(len(triples)/6)]
    train_features = []
    batch_len = len(training_triples)/batch_num

    train_index2triple = {}
    test_index2triple = {}
    temp_triple = []
    for index,triple in enumerate(training_triples):
        if thread_id != batch_num:
            if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1 ):
                continue
        
        if len(train_features) == 5000:
            print('saving training......')
            np.save('/datadrive-2/data/neighbor_triplet_finetune_training_feature_dataset_more/'+str(thread_id)+'_'+str(batch)+'.npy', train_features)
            train_index2triple[str(index)+'_'+str(batch)] = temp_triple
            batch += 1
            train_features = []
            temp_triple = []

        print(index, len(training_triples))
        if not os.path.exists('input_feature_neighbor_finetune/'+triple[0].split('/')[-1]+'.npy') or not os.path.exists('input_feature_neighbor_finetune/'+triple[1].split('/')[-1]+'.npy') or not os.path.exists('input_feature_neighbor_finetune/'+triple[2].split('/')[-1]+'.npy'):
            continue
        feature1 = np.load('input_feature_neighbor_finetune/'+triple[0].split('/')[-1]+'.npy', allow_pickle=True)
        feature2 = np.load('input_feature_neighbor_finetune/'+triple[1].split('/')[-1]+'.npy', allow_pickle=True)
        feature3 = np.load('input_feature_neighbor_finetune/'+triple[2].split('/')[-1]+'.npy', allow_pickle=True)
        train_features.append((feature1, feature2, feature3))
        temp_triple.append(triple)
    
    train_index2triple[str(index)+'_'+str(batch)] = temp_triple
    print('saving training......')
    # np.save('triplet_retraining1010_training_feature_dataset/'+str(batch)+'.npy', train_features)
    np.save('/datadrive-2/data/neighbor_triplet_finetune_training_feature_dataset_more/'+str(thread_id)+'_'+str(batch)+'.npy', train_features)
    train_features = []
    temp_triple = []

    test_triples = triples[5*int(len(triples)/6):]
    test_features = []
    batch = 1

    batch_len = len(test_triples)/batch_num
    for index,triple in enumerate(test_triples):
        if thread_id != batch_num:
            if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1 ):
                continue
        if len(test_features) == 5000:
            print('saving training......')
            test_index2triple[str(thread_id)+'_'+str(batch)] = temp_triple 
            np.save('/datadrive-2/data/neighbor_triplet_finetune_testing_feature_dataset_more/'+str(thread_id)+'_'+str(batch)+'.npy', test_features)
            batch += 1
            test_features = []
            temp_triple = []

        print(index, len(test_triples))
        if not os.path.exists('input_feature_neighbor_finetune/'+triple[0].split('/')[-1]+'.npy') or not os.path.exists('input_feature_neighbor_finetune/'+triple[1].split('/')[-1]+'.npy') or not os.path.exists('input_feature_neighbor_finetune/'+triple[2].split('/')[-1]+'.npy'):
            continue
        feature1 = np.load('input_feature_neighbor_finetune/'+triple[0].split('/')[-1]+'.npy', allow_pickle=True)
        feature2 = np.load('input_feature_neighbor_finetune/'+triple[1].split('/')[-1]+'.npy', allow_pickle=True)
        feature3 = np.load('input_feature_neighbor_finetune/'+triple[2].split('/')[-1]+'.npy', allow_pickle=True)
        test_features.append((feature1, feature2, feature3))
    
    print('saving testing......')
    test_index2triple[str(thread_id)+'_'+str(batch)] = temp_triple 
    # np.save('triplet_retraining1010_testing_feature_dataset/'+str(batch)+'.npy', test_features)
    np.save('/datadrive-2/data/neighbor_triplet_finetune_testing_feature_dataset_more/'+str(thread_id)+'_'+str(batch)+'.npy', test_features)
    batch += 1
    test_features = []
    temp_triple = []
    test_label = []
    with open('test_neighbor_index2triple_'+str(thread_id)+'.json', 'w') as f:
        json.dump(test_index2triple, f)
    with open('train_neighbor_index2triple_'+str(thread_id)+'.json', 'w') as f:
        json.dump(train_index2triple, f)

def para_save_feature():
    # process = [
    #     Process(target=save_triplet_data_1010, args=(1,20)),
    #     Process(target=save_triplet_data_1010, args=(2,20)), 
    #     Process(target=save_triplet_data_1010, args=(3,20)),
    #     Process(target=save_triplet_data_1010, args=(4,20)), 
    #     Process(target=save_triplet_data_1010, args=(5,20)),
    #     Process(target=save_triplet_data_1010, args=(6,20)), 
    #     Process(target=save_triplet_data_1010, args=(7,20)),
    #     Process(target=save_triplet_data_1010, args=(8,20)), 
    #     Process(target=save_triplet_data_1010, args=(9,20)),
    #     Process(target=save_triplet_data_1010, args=(10,20)), 
    #     # Process(target=save_triplet_data_1010, args=(11,20)),
    #     # Process(target=save_triplet_data_1010, args=(12,20)), 
    #     # Process(target=save_triplet_data_1010,args=(13,20)),
    #     # Process(target=save_triplet_data_1010, args=(14,20)), 
    #     # Process(target=save_triplet_data_1010, args=(15,20)),
    #     # Process(target=save_triplet_data_1010, args=(16,20)), 
    #     # Process(target=save_triplet_data_1010, args=(17,20)),
    #     # Process(target=save_triplet_data_1010, args=(18,20)), 
    #     # Process(target=save_triplet_data_1010, args=(19,20)), 
    #     Process(target=save_triplet_data_1010, args=(20,20)), 
    # ]
    # [p.start() for p in process]  # 开启了两个进程
    # [p.join() for p in process]   # 等待两个进程依次结束

    train_index = {}
    test_index = {}
    for thread_id in range(1,21):
        with open('train_index2triple_'+str(thread_id)+'.json', 'r') as f:
            train_index2triple = json.load(f)
        with open('test_index2triple_'+str(thread_id)+'.json', 'r') as f:
            test_index2triple = json.load(f)
        
        for key in train_index2triple:
            train_index[key] = train_index2triple[key]
        for key in test_index2triple:
            test_index[key] = test_index2triple[key]

    # with open('/datadrive-2/data/train_index2triple.json', 'w') as f:
    #     json.dump(train_index, f)
    # with open('/datadrive-2/data/test_index2triple.json', 'w') as f:
    #     json.dump(test_index, f)
    

def para_save_neighbor_feature():
    process = [
        Process(target=save_neighbor_triplet_data_1010, args=(1,20)),
        Process(target=save_neighbor_triplet_data_1010, args=(2,20)), 
        Process(target=save_neighbor_triplet_data_1010, args=(3,20)),
        Process(target=save_neighbor_triplet_data_1010, args=(4,20)), 
        Process(target=save_neighbor_triplet_data_1010, args=(5,20)),
        Process(target=save_neighbor_triplet_data_1010, args=(6,20)), 
        Process(target=save_neighbor_triplet_data_1010, args=(7,20)),
        Process(target=save_neighbor_triplet_data_1010, args=(8,20)), 
        Process(target=save_neighbor_triplet_data_1010, args=(9,20)),
        Process(target=save_neighbor_triplet_data_1010, args=(10,20)), 
        # Process(target=save_neighbor_triplet_data_1010, args=(11,20)),
        # Process(target=save_neighbor_triplet_data_1010, args=(12,20)), 
        # Process(target=save_neighbor_triplet_data_1010,args=(13,20)),
        # Process(target=save_neighbor_triplet_data_1010, args=(14,20)), 
        # Process(target=save_neighbor_triplet_data_1010, args=(15,20)),
        # Process(target=save_neighbor_triplet_data_1010, args=(16,20)), 
        # Process(target=save_neighbor_triplet_data_1010, args=(17,20)),
        # Process(target=save_neighbor_triplet_data_1010, args=(18,20)), 
        # Process(target=save_neighbor_triplet_data_1010, args=(19,20)), 
        # Process(target=save_neighbor_triplet_data_1010, args=(20,20)), 
    ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束

    # train_index = {}
    # test_index = {}
    # for thread_id in range(1,21):
    #     with open('train_neighbor_index2triple_'+str(thread_id)+'.json', 'r') as f:
    #         train_index2triple = json.load(f)
    #     with open('test_neighbor_index2triple_'+str(thread_id)+'.json', 'r') as f:
    #         test_index2triple = json.load(f)
        
    #     for key in train_index2triple:
    #         train_index[key] = train_index2triple[key]
    #     for key in test_index2triple:
    #         test_index[key] = test_index2triple[key]

    # with open('/datadrive-2/data/train_neighbor_index2triple.json', 'w') as f:
    #     json.dump(train_index, f)
    # with open('/datadrive-2/data/test_neighbor_index2triple.json', 'w') as f:
    #     json.dump(test_index, f)

def generate_negative_neighbors(max_boundary = 10, num=10):
    with open("/datadrive-2/data/auchors.json", 'r') as f:
        auchors = json.load(f)
    auchors = list(set(auchors))
    with open("/datadrive-2/data/auchors.json", 'w') as f:
        json.dump(auchors, f)

    negative_list = []
    str_1 = ''
    for item in auchors:
        auchor_x, auchor_y = int(item.split("---")[-2]), int(item.split("---")[-1])
        itemlist = item.split("---")
    
        resset = set()
    
        smallest_x = min(0, auchor_x - max_boundary)
        smallest_y = min(0, auchor_y - max_boundary)
        largest_x = auchor_x + max_boundary
        largest_y = auchor_y + max_boundary
        while len(resset) < num:
            x = random.randrange(smallest_x, largest_x+1) 
            y = random.randrange(smallest_y, largest_y+1) 
            if str(x)+"-"+str(y) in resset:
                continue
            resset.add(str(x)+"-"+str(y))
            negative_list.append(itemlist[0] + '---' + itemlist[1] + '---' + str(x) + '---' + str(y))
            str_1 += item+'   ' +itemlist[0] + '---' + itemlist[1] + '---' + str(x) + '---' + str(y) + '\n'

    with open("negative_neighbors.json", 'w') as f:
        json.dump(negative_list, f)
    with open("negative_neighbors_pairs.txt", 'w') as f:
        f.write(str_1)

def trans_negative_neighbors():
    with open("negative_neighbors.json", 'r') as f:
        negative_list = json.load(f)
    res = {}
    for item in negative_list:
        split_list = item.split("---")
        filesheet = split_list[0] + '---' + split_list[1]
        if filesheet not in res:
            res[filesheet] = []
        res[filesheet].append(item)

    with open("negative_neighbors_dict.json", 'w') as f:
        json.dump(res, f)

def generate_test_triple2index():
    res = []
    
    for thrid in range(1,21):
        for bid in range(1,10):
            if thrid == 20 and bid >1:
                continue
            res.append(str(thrid)+'_'+str(bid))

    res.sort()
    print(res)

    for index,item in enumerate(res):
        print(index, item)


def look_train_idnex():
    rootpath = '/datadrive-2/data/triplet_finetune_training_feature_dataset_more'
    filelist = os.listdir(rootpath)
    for item in filelist:
        res = np.load(rootpath+'/'+item, allow_pickle=True)
        print(item, len(res))

def look_distance():
    # with open('saved_finetune_triplet_sampled.json', 'r') as f:
    #     triples = json.load(f)
    with open('finetune_neighbor_triplet_sampled.json', 'r') as f:
        triples = json.load(f)
    print('len(triples', len(triples))
    new_res = []
    dis_num = {}
    sheetnames = set()
    sheetname2num = {}
    auchors = []
    for index,triple in enumerate(triples):
        # print(triple)
        print(index, len(triples))
        auchors.append(triple[0])
        auchor_sheet = triple[0].split('---')[1]
        positive_sheet = triple[1].split('---')[1]
        negative_sheet = triple[2].split('---')[1]
        sheetnames.add(auchor_sheet)
        sheetnames.add(positive_sheet)
        sheetnames.add(negative_sheet)
        if auchor_sheet not in sheetname2num:
            sheetname2num[auchor_sheet] = 0
        if positive_sheet not in sheetname2num:
            sheetname2num[positive_sheet] = 0
        if negative_sheet not in sheetname2num:
            sheetname2num[negative_sheet] = 0
        sheetname2num[auchor_sheet] += 1
        sheetname2num[positive_sheet] += 1
        sheetname2num[negative_sheet] += 1

        # print(index, len(triples))
        # auchor_x, auchor_y = int(triple[0].split('---')[-2]), int(triple[0].split('---')[-1]) 
        # negative_x, negative_y = int(triple[2].split('---')[-2]), int(triple[2].split('---')[-1]) 
        # dis = ((negative_x - auchor_x)**2 + (negative_y - auchor_y)**2)**0.5
        # if dis not in dis_num:
        #     dis_num[dis] = 0
        # dis_num[dis] += 1
        if sheetname2num[auchor_sheet] < 3000:
            new_res.append(triple)
        # break
    # print('sheetname2num', sheetname2num)
    # print(dis_num)
    # print(len(sheetnames))
    # with open("distance2num.json",'w') as f:
    #     json.dump(dis_num, f)
    # with open("sheetname2num.json",'w') as f:
    #     json.dump(sheetname2num, f)
    # with open("/datadrive-2/data/auchors.json",'w') as f:
        # json.dump(auchors, f)
    with open('finetune_neighbor_triplet_sampled.json', 'w') as f:
        json.dump(new_res, f)

def euclidean(x, y):
    return np.sqrt(np.sum((x - y)**2))

def generate_one_after_feature(formula_token, model, before_path, after_path):
    # print('generate_one_after_feature......')
    if os.path.exists(after_path + formula_token + '.npy'):
        return
    
    try:
        feature_nparray = np.load(before_path)
        feature_nparray = feature_nparray.reshape(1,100,10,399)
        model.eval()
        feature_nparray = torch.DoubleTensor(feature_nparray)
        feature_nparray = Variable(feature_nparray).to(torch.float32)
        feature_nparray = model(feature_nparray).detach().numpy()
        # print('saving:', after_path)
        np.save(after_path, feature_nparray)
    except Exception as e:
        print('e', e)
        return

def generate_one_before_feaure(formula_token, bert_dict, content_tem_dict, mask, temp_dict, source_root_path = root_path + 'demo_tile_features_shift/', saved_root_path=root_path + 'demo_before_features/'):
    try:
        with open(source_root_path + formula_token + '.json', 'r') as f:
            origin_feature = json.load(f)
    except Exception as e:
        print('before e', e)
        if 'Expecting' in str(e):
            return "invalid json features"
        return
    temp_time1 = time.time()
    feature_nparray, temp_dict = get_feature_vector_with_bert_keyw(origin_feature,  bert_dict, content_tem_dict, mask, temp_dict)
    feature_nparray = np.array(feature_nparray)
    temp_time2 = time.time()
    # print('before_feature time:', temp_time2-temp_time1)
    np.save(saved_root_path + formula_token + '.npy', feature_nparray)
    return temp_dict

def before2cross(formula_token, bert_dict, content_tem_dict, mask, source_root_path = root_path + 'demo_before_features/', saved_root_path=root_path + 'cross_before_features/'):
    origin_row = int(formula_token.split('---')[2])
    origin_col = int(formula_token.split('---')[3])
    
    start_row = origin_row - 50
    end_row = origin_row + 50 - 1
    start_col = origin_col - 5
    end_col = origin_col + 5 - 1

    try:
        feature_nparray = np.load(source_root_path +formula_token + '.npy' , allow_pickle=True)
    except Exception as e:
        print('before2cross e', e)
        if 'Expecting' in str(e):
            return "invalid json features"
        return
    zero_vector = np.zeros(399, dtype=np.float32)
    res = []
    index = 0
    for row in range(start_row, start_row + 100):
        for col in range(start_col, start_col + 10):
            if row == origin_row or col == origin_col:
                res.append(feature_nparray[index])
            else:
                res.append(zero_vector)
            index += 1
    np.save(saved_root_path + formula_token + '.npy', feature_nparray)

def generate_demo_features(filename, sheetname, workbook_json, origin_row, origin_col, save_path, is_look=True, cross=False):
    def extract_sheet_wh_info(sheetname, found_sheet):
        sheet_wh_info = {'height':{}, 'width':{}}
        for row_info in found_sheet['Rows']:
            row_id = row_info['Row']
            if 'Height' in row_info:
                sheet_wh_info['height'][row_id] = row_info['Height']
            for cell_info in row_info['Cells']:
                if 'Width' in cell_info:
                    sheet_wh_info['width'][cell_info['C']] = cell_info['Width']
                if 'Height' in cell_info:
                    sheet_wh_info['height'][row_id] = cell_info['Height']
        return sheet_wh_info

    one_invalid_cell =  {
        "background_color_r": 255,
        "background_color_g": 255,
        "background_color_b": 255,
        "font_color_r": 0,
        "font_color_g": 0,
        "font_color_b": 0,
        "font_size": 11,
        "font_strikethrough": False,
        "font_shadow": False,
        "font_ita": False,
        "font_bold": False,
        "height": 0.0,
        "width": 0.0,
        "content": None,
        "content_template": None
    },

    def get_template(cell_value, cell_type):
        result = ''
        if cell_type == 'S':
            for char in cell_value:
                result += 'S'

        elif cell_type == 'N':
            for char in cell_value:
                result += 'N'

        elif cell_type == 'B':
            for char in cell_value:
                result += 'B'

        elif cell_type == 'D':
            for char in cell_value:
                result += 'D'

        elif cell_type == 'T':
            for char in cell_value:
                result += 'T'
        return result

    def argb_to_rgb(value):
        value = value[-6:]
        lv = len(value)
        return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))


    temp_start_time = time.time()
    
    temp_end_time = time.time()
    sheet_jsons = workbook_json['Sheets']
    start_row = origin_row - 50
    end_row = origin_row + 50 - 1
    start_col = origin_col - 5
    end_col = origin_col + 5 - 1
    res = []

    res1 = {}
    temp_start_time = time.time()
    # print('start_row', start_row)
    # print('end_row', end_row)
    for sheet_json in sheet_jsons:
        if sheet_json['Name'] == sheetname:  
            sheet_wh_info = extract_sheet_wh_info(sheetname, sheet_json)
            # print('sheet_wh_info', sheet_wh_info)
            for row_json in sheet_json['Rows']:
                if row_json['Row'] >= start_row and row_json['Row'] < end_row:
                    for cell in row_json['Cells']:
                        
                        new_cell = copy.deepcopy(one_invalid_cell[0])
                        if 'Fill' in cell:
                            if 'ARGB' in cell['Fill']:
                                bg_argb = cell['Fill']['ARGB']
                                r, g, b = argb_to_rgb(bg_argb)
                                new_cell['background_color_r'] = r
                                new_cell['background_color_g'] = g
                                new_cell['background_color_b'] = b
                        if 'Width' in cell:
                            new_cell['width'] = cell['Width']
                  
                        if 'Height' in cell:
                            new_cell['height'] = cell['Height']
                        elif 'Height' in row_json:
                            new_cell['height'] = row_json['Height']
                        elif 'DefaultHeight' in row_json:
                            new_cell['height'] = row_json['DefaultHeight']
                    
                        if "Font" in cell:
                            if 'Size' in cell['Font']:
                                new_cell['font_size'] = cell['Font']['Size']
                            if 'Color' in cell['Font']:
                                f_argb = cell['Font']['Color']['ARGB']
                                r, g, b = argb_to_rgb(f_argb)
                                new_cell['font_color_r'] = r
                                new_cell['font_color_g'] = g
                                new_cell['font_color_b'] = b

                        if "V" in cell:
                            keyw = list(cell['V'].keys())[0]
                            content = str(cell['V'][keyw])
                            content_template = get_template(content, keyw)
                            new_cell['content'] = content
                            new_cell['content_template'] = content_template
                            
                        res1[str(cell['R']) + '---' + str(cell['C'])] = new_cell
                        # print(str(cell['R']) + '---' + str(cell['C']), new_cell)

    temp_end_time = time.time()
    temp_start_time = time.time()
    for row in range(start_row, start_row + 100):
        for col in range(start_col, start_col + 10):
            
            if str(row) + '---' + str(col) in res1:
                if is_look:
                    res1[str(row) + '---' + str(col)]['row'] = row
                    res1[str(row) + '---' + str(col)]['col'] = col
                # res.append(res1[str(row) + '---' + str(col)])
                if not cross:
                    add_cell = res1[str(row) + '---' + str(col)]
                    # res.append(res1[str(row) + '---' + str(col)])
                else:
                    if origin_row == row or origin_col == col:
                        add_cell = res1[str(row) + '---' + str(col)]
                        # res.append(res1[str(row) + '---' + str(col)])
                    else:
                        add_cell = one_invalid_cell
                        # res.append(one_invalid_cell)
            else:
                if is_look:
                    new_cell = copy.deepcopy(one_invalid_cell[0])
                    new_cell['row'] = row
                    new_cell['col'] = col
                    add_cell = new_cell
                    # res.append(new_cell)
                else:
                    add_cell = one_invalid_cell
                    # res.append(one_invalid_cell)
            if add_cell['height'] == 0:
                if row in sheet_wh_info['height']:
                    add_cell['height'] = sheet_wh_info['height'][row]
            if add_cell['width'] == 0:
                if col in sheet_wh_info['width']:
                    add_cell['width'] = sheet_wh_info['width'][col] 
            res.append(add_cell)

    temp_end_time = time.time()

    temp_end_time = time.time()
    formula_token = filename + '---' + sheetname + '---' + str(origin_row) + '---' + str(origin_col)
    with open(save_path + formula_token + ".json", 'w') as f:
        json.dump(res, f)   
    

def copyround_finetune(file_path, model_path, tile_path, cross_path, before_path, after_path, second_save_path, mask, reverse=False, validate = False, cross=False):
    global version
    model = torch.load(model_path)
    print(model)
    top_k = 1


    with open("fail_res_3159.json", 'r') as f:
        fail_res_3159 = json.load(f)

    filename2wbjson = {}
    # if version == 0:
    #     model1_filename = 'dedup_model1_res'
    # else:
    #     model1_filename = 'model1_res'
    model1_filename = 'company_model1_res'

    # if validate == True:
    #     with open("fortune500_val_formula.json", 'r') as f:
    #         formulas = json.load(f)
    # else:
    #     with open("fortune500_test_formula.json", 'r') as f:
    #         formulas = json.load(f)
    # with open("sample_efficiency.json", 'r') as f:
    #     formulas = json.load(f)
    with open(file_path, 'r') as f:
        formulas = json.load(f)
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)
    
    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)
    key_list = list(fail_res_3159.keys())
    key_list.sort(reverse=reverse)
    print('len(formulas', len(formulas))

    all_time = 0
    all_count = 0
    feature_extract_time = 0
    before_time = 0
    after_time = 0
    files = os.listdir('.')
    files = [item for item in files if 'reduced_formulas_' in item]
    temp_dict = {}
    for filename in files:
        rm_list = []
        company_name = filename.split('_')[2].replace(".json", '')
        
        # if company_name not in ['cisco','ibm','ti','pge']:
        if company_name not in ['cisco']:
            continue
        print('company_name', company_name)
        with open(filename, 'r') as f:
            formulas = json.load(f)
        with open("fortune500_company2workbook.json", 'r') as f:
            constrained_workbooks = json.load(f)
        for index, formula_token in enumerate(formulas):

            # try:

                if os.path.exists(second_save_path + formula_token  +'.npy'):
                    print('res exists')
                    continue
                start_time = time.time()
                filesheet = formula_token.split('---')[0] + '---' + formula_token.split('---')[1]
                sorted_first_block = {}
                sorted_second_block = {}
                origin_file = formula_token.split('---')[0]
                origin_sheet = formula_token.split('---')[1]
                origin_filesheet = origin_file + '---' + origin_sheet
                print(index, len(formulas))
                if not os.path.exists(root_path + model1_filename + '/' + formula_token + '.json'):
                    print('model1_not_exists')
                    continue
                with open(root_path + model1_filename +  '/' + formula_token + '.json', 'r') as f:
                    mode1_res = json.load(f)
                if mode1_res[1] == '':
                    rm_list.append(formula_token)
                    continue
                found_formula_token = mode1_res[1]
                found_file = mode1_res[1].split('---')[0]
                print('mode1_res', mode1_res)
                found_filesheet = mode1_res[1].split('---')[0] + '---' + mode1_res[1].split('---')[1]

                if origin_file not in filename2wbjson:
                    if version == 0:
                        if os.path.exists("../Demo/fix_fortune500/" + origin_file + '.json'):
                            with open("../Demo/fix_fortune500/" + origin_file + '.json', 'r') as f:
                                filename2wbjson[origin_file] = json.load(f)
                        else:
                            continue
                    elif version == 1:
                        if os.path.exists("../Demo/fix_workbook_json/" + origin_file + '.json'):
                            with open("../Demo/fix_workbook_json/" + origin_file + '.json', 'r') as f:
                                filename2wbjson[origin_file] = json.load(f)
                        else:
                            continue

                if found_file not in filename2wbjson:
                    if version == 0:
                        if os.path.exists("../Demo/fix_fortune500/" + found_file + '.json'):
                            with open("../Demo/fix_fortune500/" + found_file + '.json', 'r') as f:
                                filename2wbjson[found_file] = json.load(f)
                        else:
                            continue
                    elif version == 1:
                        if os.path.exists("../Demo/fix_workbook_json/" + found_file + '.json'):
                            with open("../Demo/fix_workbook_json/" + found_file + '.json', 'r') as f:
                                filename2wbjson[found_file] = json.load(f)
                        else:
                            continue
                origin_wbjson = filename2wbjson[origin_file]
                found_wbjson = filename2wbjson[found_file]
                if not os.path.exists(root_path + 'test_refcell_position/' + found_formula_token + '.json'):
                    # print('test_refcell_position', test_refcell_position)
                    continue
                with open(root_path + 'test_refcell_position/'+found_formula_token + '.json' , 'r') as f:
                    test_refcell_position = json.load(f)

                found_formula_row = int(found_formula_token.split('---')[2])
                found_formula_col = int(found_formula_token.split('---')[3])
                origin_formula_row = int(formula_token.split('---')[2])
                origin_formula_col = int(formula_token.split('---')[3])
                for refcell_item in test_refcell_position:
                    ref_row = refcell_item['R']
                    ref_col = refcell_item['C']
                    # ref_row, ref_col = ref_cell_rc.split('---') 
                    ref_row = int(ref_row)
                    ref_col = int(ref_col)
                    
                    delta_row = ref_row - found_formula_row
                    delta_col = ref_col - found_formula_col

                    copy_row = origin_formula_row + delta_row
                    copy_col = origin_formula_col + delta_col

                    start_row = copy_row - 10
                    end_row = copy_row + 10
                    start_col =  copy_col - 10
                    end_col = copy_col + 10

                    ######### second level
                    token_npy = found_filesheet + '---' + str(ref_row) + '---' + str(ref_col) + '.npy'
                    if not os.path.exists(tile_path  + token_npy.replace('.npy', '.json')):
                        # print("generate: 2913")
                        generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], found_wbjson, ref_row, ref_col, tile_path, is_look=True, cross=cross)
                    if not os.path.exists(before_path + token_npy):
                        # print("generate: 2916")
                        if cross:
                            res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                        else:
                            res = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path, temp_dict=temp_dict)
                        if res == "invalid json features":
                            generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], found_wbjson, ref_row, ref_col, tile_path, is_look=True, cross=cross)
                            if cross:
                                res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                            else:
                                temp_dict = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path, temp_dict = temp_dict)
                        else:
                            temp_dict = res
                    if not os.path.exists(after_path + token_npy):
                        # print("generate: 2919")
                        generate_one_after_feature(token_npy.replace('.npy', ''), model, before_path + token_npy, after_path + token_npy)
                        
                    # feature_time_start = time.time()
                    # generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], found_wbjson, ref_row, ref_col, tile_path, is_look=True, cross=cross)
                    # feature_time_end= time.time()
                    # print('feature time:', feature_time_end - feature_time_start)
                    # feature_extract_time += (feature_time_end - feature_time_start)
                    # before_time_start = time.time()
                    # res = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                    # before_time_end = time.time()
                    # before_time += (before_time_end - before_time_start)
                    # print('before_time:', before_time_end - before_time_start)
                    # after_time_start = time.time()
                    # generate_one_after_feature(token_npy.replace('.npy', ''), model, before_path + token_npy, after_path + token_npy)
                    # after_time_end = time.time()
                    # after_time += (after_time_end - after_time_start)
                    # print('after_time:', after_time_end - after_time_start)

                    feature = np.load(after_path + token_npy, allow_pickle=True)

                    sorted_second_block[str(ref_row) + '---' + str(ref_col)] = {}
                    for row in range(start_row, end_row):
                        for col in range(start_col, end_col):

                            token_npy = origin_filesheet + '---' + str(row) + '---' + str(col) + '.npy'
                            if not os.path.exists(tile_path + token_npy.replace('.npy', '.json')):
                                # print("generate: 2947")
                                generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], origin_wbjson, row, col, tile_path, is_look=True, cross=cross)
                            # print('before_path + token_npy', before_path + token_npy)
                            if not os.path.exists(before_path + token_npy):
                                # print("generate: 2950")
                                if cross:
                                    # print('before2cross')
                                    res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                                else:
                                    # print('before')
                                    res = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path, temp_dict = temp_dict)
                                if res == "invalid json features":
                                    generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], origin_wbjson, row, col, tile_path, is_look=True, cross=cross)
                                    if cross:
                                        res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                                    else:
                                        temp_dict = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path, temp_dict = temp_dict)
                                else:
                                    temp_dict = res
                            if not os.path.exists(after_path + token_npy):
                                # print("generate: 2953")
                                generate_one_after_feature(token_npy.replace('.npy', ''), model, before_path + token_npy, after_path + token_npy)
                            # feature_time_start = time.time()
                            # generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], origin_wbjson, row, col, tile_path, is_look=True, cross=cross)
                            # feature_time_end= time.time()
                            # feature_extract_time += (feature_time_end - feature_time_start)
                            # print('feature time:', feature_time_end - feature_time_start)
                            # before_time_start = time.time()
                            # res = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                            # before_time_end = time.time()
                            # before_time += (before_time_end - before_time_start)
                            # print('before_time:', before_time_end - before_time_start)
                            # after_time_start = time.time()
                            # generate_one_after_feature(token_npy.replace('.npy', ''), model, before_path + token_npy, after_path + token_npy)
                            # after_time_end = time.time()
                            # after_time += (after_time_end - after_time_start)
                            # print('after_time:', after_time_end - after_time_start)

                            other_feature = np.load(after_path + token_npy, allow_pickle=True)
                            distance = euclidean(feature, other_feature) 
                            sorted_second_block[str(ref_row) + '---' + str(ref_col)][str(row) + '---'  + str(col)] = distance
                            # sorted_second_block['time'] = end_time - start_time
                            # print('sorted_second_block',str(ref_row) + '---' + str(ref_col),sorted_second_block[str(ref_row) + '---' + str(ref_col)].keys())
                print('saving:',second_save_path + formula_token  +'.npy')
                np.save(second_save_path + formula_token  +'.npy', sorted_second_block)
                end_time = time.time()
                all_time += (end_time - start_time)
                all_count += 1
        rm_list = list(set(formulas) - set(rm_list))
        with open(filename, 'w') as f:
            json.dump(rm_list, f)
            # except Exception as e:
            #     print('error', e)
            #     continue
    # print('all_time: ', all_time)
    # print('avg time: ', all_time / all_count)
    # print('feature_extract_time time: ', feature_extract_time / all_count)
    # print('before_time: ', before_time / all_count)
    # print('after_time: ', after_time / all_count)
            
def test_finetune(thread_id, batch_num, model_path, tile_path, cross_path, before_path, after_path, first_save_path, second_save_path, mask, reverse=False, validate = False, cross=False):# '/datadrive-2/data/finetune_shift_content_models/epoch_200'
    global version
    model = torch.load(model_path)
    print(model)
    top_k = 1

    with open("fail_res_3159.json", 'r') as f:
        fail_res_3159 = json.load(f)

    filename2wbjson = {}
    if version == 0:
        model1_filename = 'dedup_model1_res'
    else:
        model1_filename = 'model1_res'

    # formulas = os.listdir(root_path + model1_filename)
    if validate == True:
        with open("fortune500_val_formula.json", 'r') as f:
            formulas = json.load(f)
    else:
        with open("fortune500_test_formula.json", 'r') as f:
            formulas = json.load(f)
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)
    
    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)
    key_list = list(fail_res_3159.keys())
    key_list.sort(reverse=reverse)
    # for index, key in enumerate(key_list):
    print('len(formulas', len(formulas))
    batch_len = int(len(formulas)/batch_num)
    for index, formula_token in enumerate(formulas):
        if thread_id != batch_num:
            if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1 ):
                continue
        try:
            # formula_token = formula_file.replace('.json', '')
            # if formula_token != '100206920085844727873527872071004985725-3288.flybuck-flyback-calculator.xlsx---Fly-BB_Calc---4---34':
            #     continue
            # formula_token = key.split("#####")[0]
            if os.path.exists(second_save_path + formula_token  +'.npy'):
                print('first exists')
                continue
            # ref_cell_rc = key.split("#####")[1]
            filesheet = formula_token.split('---')[0] + '---' + formula_token.split('---')[1]
            start_time = time.time()
            sorted_first_block = {}
            sorted_second_block = {}
            origin_file = formula_token.split('---')[0]
            origin_sheet = formula_token.split('---')[1]
            origin_filesheet = origin_file + '---' + origin_sheet
            print(thread_id, index, len(formulas))
            if not os.path.exists(root_path + model1_filename + '/' + formula_token + '.json'):
                continue
            with open(root_path + model1_filename +  '/' + formula_token + '.json', 'r') as f:
                mode1_res = json.load(f)

            found_formula_token = mode1_res[1]
            found_file = mode1_res[1].split('---')[0]
            found_filesheet = mode1_res[1].split('---')[0] + '---' + mode1_res[1].split('---')[1]

            if origin_file not in filename2wbjson:
                if version == 0:
                    if os.path.exists("../Demo/fix_fortune500/" + origin_file + '.json'):
                        with open("../Demo/fix_fortune500/" + origin_file + '.json', 'r') as f:
                            filename2wbjson[origin_file] = json.load(f)
                    else:
                        continue
                elif version == 1:
                    if os.path.exists("../Demo/fix_workbook_json/" + origin_file + '.json'):
                        with open("../Demo/fix_workbook_json/" + origin_file + '.json', 'r') as f:
                            filename2wbjson[origin_file] = json.load(f)
                    else:
                        continue

            if found_file not in filename2wbjson:
                if version == 0:
                    if os.path.exists("../Demo/fix_fortune500/" + found_file + '.json'):
                        with open("../Demo/fix_fortune500/" + found_file + '.json', 'r') as f:
                            filename2wbjson[found_file] = json.load(f)
                    else:
                        continue
                elif version == 1:
                    if os.path.exists("../Demo/fix_workbook_json/" + found_file + '.json'):
                        with open("../Demo/fix_workbook_json/" + found_file + '.json', 'r') as f:
                            filename2wbjson[found_file] = json.load(f)
                    else:
                        continue
            origin_wbjson = filename2wbjson[origin_file]
            found_wbjson = filename2wbjson[found_file]
            if not os.path.exists(root_path + 'test_refcell_position/' + found_formula_token + '.json'):
                continue
            with open(root_path + 'test_refcell_position/'+found_formula_token + '.json' , 'r') as f:
                test_refcell_position = json.load(f)

            with open(root_path + 'tile_rows/' + origin_filesheet + '.json', 'r') as f:
                tile_rows = json.load(f)
            with open(root_path + 'tile_cols/' + origin_filesheet + '.json', 'r') as f:
                tile_cols = json.load(f)

            for refcell_item in test_refcell_position:
                ref_row = refcell_item['R']
                ref_col = refcell_item['C']
                # ref_row, ref_col = ref_cell_rc.split('---') 
                ref_row = int(ref_row)
                ref_col = int(ref_col)
                
                ######## first level
                    ########## check 4 nearest tile
                refcell_tile_row = int(ref_row/100)*100 + 1
                refcell_tile_col = int(ref_col/10)*10 + 1

                is_left = False
                is_up = False
                if ref_row - refcell_tile_row < refcell_tile_row + 100 - ref_row: # up
                    is_up = True
                if ref_col - refcell_tile_col < refcell_tile_col + 10 - ref_col: # left
                    is_left = True

                closed_four_tiles = []
                closed_four_tiles.append((refcell_tile_row, refcell_tile_col)) # first
                if is_up:
                    if refcell_tile_row >= 101:
                        closed_four_tiles.append((refcell_tile_row - 100, refcell_tile_col))  # second add up
                        if is_left:
                            if refcell_tile_col >= 11:
                                closed_four_tiles.append((refcell_tile_row - 100, refcell_tile_col-10))  # third add left, up
                        else:
                            if refcell_tile_col + 10 in tile_cols:
                                closed_four_tiles.append((refcell_tile_row - 100, refcell_tile_col+10))  # third add right, up
                else:
                    if refcell_tile_col + 100 in tile_rows:
                        closed_four_tiles.append((refcell_tile_row + 100, refcell_tile_col))  #0 second add down
                    if is_left:
                        if refcell_tile_col >= 11:
                            closed_four_tiles.append((refcell_tile_row + 100, refcell_tile_col-10))  # third add left, down
                    else:
                        if refcell_tile_col + 10 in tile_cols: 
                            closed_four_tiles.append((refcell_tile_row + 100, refcell_tile_col+10))  # third add right, down

                if is_left:
                    if refcell_tile_col >= 11:
                        closed_four_tiles.append((refcell_tile_row, refcell_tile_col-10))  # forth add left
                else:
                    if refcell_tile_col + 10 in tile_cols: 
                        closed_four_tiles.append((refcell_tile_row, refcell_tile_col+10))  # forth add right

                    ##############find closed tile on original sheet of 4 nearest tile on found sheet
                sorted_first_block[str(ref_row) + '---' + str(ref_col)] = {}
                print('tile_rows', tile_rows)
                print('tile_cols', tile_cols)
                print('closed_four_tiles', closed_four_tiles)
                for one_found_tile in closed_four_tiles:
                    one_found_row = one_found_tile[0]
                    one_found_col = one_found_tile[1]
                    token_npy = found_filesheet + '---' + str(one_found_row) + '---' + str(one_found_col) + '.npy'
                    if not os.path.exists(tile_path + token_npy.replace('.npy', '.json')):
                        # print("generate: 2882")
                        generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], found_wbjson, one_found_row, one_found_col, tile_path, is_look=True, cross=cross)
                    if not os.path.exists(before_path + token_npy):
                        # print("generate: 2885")
                        if cross:
                            res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                        else:
                            res = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                        if res == "invalid json features":
                            generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], found_wbjson, one_found_row, one_found_col, tile_path, is_look=True, cross=cross)
                            if cross:
                                res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                            else:
                                generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                    if not os.path.exists(after_path + token_npy):
                        # print("generate: 2888")
                        generate_one_after_feature(token_npy.replace('.npy', ''), model, before_path + token_npy, after_path + token_npy)     
                    feature = np.load(after_path + token_npy, allow_pickle=True)
                    sorted_first_block[str(ref_row) + '---' + str(ref_col)][str(one_found_row) +'---' +str(one_found_col)] = {}
                    for row in tile_rows:
                        for col in tile_cols:
                            token_npy =  origin_filesheet + '---' + str(row) + '---' + str(col) + '.npy'
                            if not os.path.exists(tile_path  + token_npy.replace('.npy', '.json')):
                                # print("generate: 2896")
                                generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], origin_wbjson, row, col, tile_path, is_look=True, cross=cross)
                            if not os.path.exists(before_path + token_npy):
                                # print("generate: 2899")
                                if cross:
                                    res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                                else:
                                    res = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                                if res == "invalid json features":
                                    generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], origin_wbjson, row, col, tile_path, is_look=True, cross=cross)
                                    if cross:
                                        res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                                    else:
                                        generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                            if not os.path.exists(after_path + token_npy):
                                # print("generate: 2902")
                                generate_one_after_feature(token_npy.replace('.npy', ''), model, before_path + token_npy, after_path + token_npy)
                            other_feature = np.load(after_path + token_npy, allow_pickle=True)
                            distance = euclidean(feature, other_feature)
                            sorted_first_block[str(ref_row) + '---' + str(ref_col)][str(one_found_row) +'---' +str(one_found_col)][str(row) + '---'  + str(col)] = distance
                first_end_time = time.time()
                sorted_first_block['time'] = first_end_time - start_time
                np.save(first_save_path + formula_token  +'.npy', sorted_first_block)
                ######### second level
                token_npy = found_filesheet + '---' + str(ref_row) + '---' + str(ref_col) + '.npy'
                if not os.path.exists(tile_path  + token_npy.replace('.npy', '.json')):
                    # print("generate: 2913")
                    generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], found_wbjson, ref_row, ref_col, tile_path, is_look=True, cross=cross)
                if not os.path.exists(before_path + token_npy):
                    # print("generate: 2916")
                    if cross:
                        res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                    else:
                        res = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                    if res == "invalid json features":
                        generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], found_wbjson, ref_row, ref_col, tile_path, is_look=True, cross=cross)
                        if cross:
                            res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                        else:
                            generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                if not os.path.exists(after_path + token_npy):
                    # print("generate: 2919")
                    generate_one_after_feature(token_npy.replace('.npy', ''), model, before_path + token_npy, after_path + token_npy)
                    
                feature = np.load(after_path + token_npy, allow_pickle=True)

                if str(ref_row) + '---' + str(ref_col) not in sorted_first_block:
                    # print('not in first', str(ref_row) + '---' + str(ref_col))
                    continue
                best_row_col_list = []
                for first_row_col in sorted_first_block[str(ref_row) + '---' + str(ref_col)]:
                    # print('first_row_col', first_row_col)
                    distance_dict = sorted_first_block[str(ref_row) + '---' + str(ref_col)][first_row_col]
                    # print('distance_dict', distance_dict)
                    sorted_list = sorted(distance_dict.items(), key=lambda x: x[1])
                    # print('sorted_list', [list(i)[0] for i in sorted_list[0:top_k]])
                    best_row_col_list += [list(i)[0] for i in sorted_list[0:top_k]]
                best_row_col_list = list(set(best_row_col_list))
                print('best_row_col_list', best_row_col_list)
                sorted_second_block[str(ref_row) + '---' + str(ref_col)] = {}
                for best_row_col in best_row_col_list:
                    best_row = int(best_row_col.split('---')[0])
                    best_col = int(best_row_col.split('---')[1])
                    
                    for row in range(best_row, best_row + 100):
                        for col in range(best_col, best_col + 10):
                            # if(best_row_col == '1---11'):
                            #     print('    ', row, col)
                            token_npy = origin_filesheet + '---' + str(row) + '---' + str(col) + '.npy'
                            if not os.path.exists(tile_path + token_npy.replace('.npy', '.json')):
                                # print("generate: 2947")
                                generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], origin_wbjson, row, col, tile_path, is_look=True, cross=cross)
                            # print('before_path + token_npy', before_path + token_npy)
                            if not os.path.exists(before_path + token_npy):
                                # print("generate: 2950")
                                if cross:
                                    # print('before2cross')
                                    res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                                else:
                                    # print('before')
                                    res = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                                if res == "invalid json features":
                                    generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], origin_wbjson, row, col, tile_path, is_look=True, cross=cross)
                                    if cross:
                                        res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                                    else:
                                        generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                            if not os.path.exists(after_path + token_npy):
                                # print("generate: 2953")
                                generate_one_after_feature(token_npy.replace('.npy', ''), model, before_path + token_npy, after_path + token_npy)
                            other_feature = np.load(after_path + token_npy, allow_pickle=True)
                            distance = euclidean(feature, other_feature) 
                            sorted_second_block[str(ref_row) + '---' + str(ref_col)][str(row) + '---'  + str(col)] = distance
                    end_time = time.time()
                    # sorted_second_block['time'] = end_time - start_time
                    # print('sorted_second_block',str(ref_row) + '---' + str(ref_col),sorted_second_block[str(ref_row) + '---' + str(ref_col)].keys())
            print('saving:',second_save_path + formula_token  +'.npy')
            np.save(second_save_path + formula_token  +'.npy', sorted_second_block)
        except Exception as e:
            print('error', e)
            continue

def check_befor_features():
    mask1 = os.listdir("/datadrive-2/data/fortune500_test/demo_before_features_mask_fix")
    for filename in mask1:
        feature = np.load("/datadrive-2/data/fortune500_test/demo_before_features_mask_fix/" + filename, allow_pickle=True)
        print('feature', feature.shape)

def cos(a,b):
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    return np.matmul(a,b) / (ma*mb)

def generate_bert_sim():
    bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    with open('fortune500_test_formula_token.json', 'r') as f:
        test_formula_token = json.load(f)
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)
    for formula_ind,formula_token in enumerate(test_formula_token):
        print(formula_ind, len(test_formula_token))
        if os.path.exists("/datadrive-2/data/fortune500_test/second_content_similarity/" + formula_token + '.json'):
            print('exist...')
            continue
        with open(root_path + 'dedup_model1_res/' + formula_token + '.json', 'r') as f:
            model1_res = json.load(f)
        second_res = np.load(root_path + "second_res_specific_epoch23_c2/" + formula_token + '.npy', allow_pickle=True).item()
        found_formula_token = model1_res[1]

        if not os.path.exists(root_path + 'test_refcell_position/' + found_formula_token + '.json'):
            continue
        with open(root_path + 'test_refcell_position/'+found_formula_token + '.json' , 'r') as f:
            found_ref_cell = json.load(f)

        fname = formula_token.split('---')[0]
        sheetname = formula_token.split('---')[1]
        found_fname = found_formula_token.split('---')[0]
        found_sheetname = found_formula_token.split('---')[1]

        if os.path.exists('../Demo/fix_fortune500/'+fname + '.json'):
            with open('../Demo/fix_fortune500/'+fname + '.json', 'r') as f:
                origin_workbook_json = json.load(f)
        elif os.path.exists('../Demo/origin_fortune500_workbook_json/'+fname + '.json'):
            with open('../Demo/origin_fortune500_workbook_json/'+fname + '.json', 'r') as f:
                origin_workbook_json = json.load(f)
        else:
            continue

        if os.path.exists('../Demo/fix_fortune500/'+found_fname + '.json'):
            with open('../Demo/fix_fortune500/'+found_fname + '.json', 'r') as f:
                found_workbook_json = json.load(f)
        elif os.path.exists('../Demo/origin_fortune500_workbook_json/'+found_fname + '.json'):
            with open('../Demo/origin_fortune500_workbook_json/'+found_fname + '.json', 'r') as f:
                found_workbook_json = json.load(f)
        else:
            continue

        for sname in origin_workbook_json["Sheets"]:
            if sname['Name'] == sheetname:
                origin_found_sheet = sname
                break
        for sname in found_workbook_json["Sheets"]:
            if sname['Name'] == found_sheetname:
                found_found_sheet = sname
                break
        res = {}
        for index, found_ref in enumerate(found_ref_cell):
            found_ref_str = str(found_ref['R']) + '---' + str(found_ref['C'])
            res[found_ref_str] = {}
            second_score = second_res[found_ref_str]
            found_r = int(found_ref_str.split('---')[0])
            found_c = int(found_ref_str.split('---')[1])
            found_rows = found_found_sheet['Rows']
            found_content = ''
            for row in found_rows:
                if row['Row'] == found_r:
                    for cell in row['Cells']:
                        if cell['C'] == found_c:
                            if 'V' in cell:
                                found_content = str(cell['V'][list(cell['V'].keys())[0]])
                                break
            if os.path.exists('/datadrive-2/data/bert_dict/' + change_word_to_save_word(found_content) + '.npy'):
                found_bert_feature = np.load('/datadrive-2/data/bert_dict/' + change_word_to_save_word(found_content) + '.npy', allow_pickle=True)
            elif found_content in bert_dict:
                found_bert_feature = bert_dict[found_content]
            else:
                found_bert_feature = np.array(bert_model.encode(found_content).tolist())

            second_position2content = {}
            second_position2feature = {}
            for second_position in second_score:
                second_position_r = int(second_position.split('---')[0])
                second_position_c = int(second_position.split('---')[1])
                origin_rows = origin_found_sheet['Rows']
                second_content = ''
                for row in origin_rows:
                    if row['Row'] == second_position_r:
                        for cell in row['Cells']:
                            if cell['C'] == second_position_c:
                                if 'V' in cell:
                                    second_content = str(cell['V'][list(cell['V'].keys())[0]])
                                    if os.path.exists('/datadrive-2/data/bert_dict/' + change_word_to_save_word(second_content) + '.npy'):
                                        second_bert_feature = np.load('/datadrive-2/data/bert_dict/' + change_word_to_save_word(second_content) + '.npy', allow_pickle=True)
                                    elif second_content in bert_dict:
                                        second_bert_feature = bert_dict[second_content]
                                    else:
                                        second_bert_feature = np.array(bert_model.encode(second_content).tolist())
                                    break
                second_position2content[second_position] = second_content
                second_position2feature[second_position] = second_bert_feature
                res[found_ref_str][second_position] = cos(found_bert_feature, second_bert_feature)
        with open("/datadrive-2/data/fortune500_test/second_content_similarity/" + formula_token + '.json', 'w') as f:
            json.dump(res, f)

def generate_sample_files():
    with open("../AnalyzeDV/fortune500_sheetname_2_file_devided.json", 'r') as f:
        devided_json = json.load(f)   
    sampled_sheets = random.sample(list(devided_json.keys()), int(len(list(devided_json.keys()))*1/2))
    test_sheets = random.sample(sampled_sheets, int(len(sampled_sheets)*3/4))
    validate_sheets = list(set(sampled_sheets) - set(test_sheets))
    test_workbooks = {}
    validate_workbooks = {}
    print(len(validate_sheets))
    for sheetname in test_sheets:
        for cluster in devided_json[sheetname]:
            for filename in cluster['filenames']:
                if filename not in test_workbooks:
                    test_workbooks[filename] = []
                test_workbooks[filename].append(sheetname)
    for sheetname in validate_sheets:
        for cluster in devided_json[sheetname]:
            for filename in cluster['filenames']:
                if filename not in validate_workbooks:
                    validate_workbooks[filename] = []
                validate_workbooks[filename].append(sheetname)
    with open('fortune500_test_formula_token.json', 'r') as f:
        test_formula_token = json.load(f)
    print('test_formula_token', len(test_formula_token))
    test_res = []
    test_sheet_num = {}
    for formula_token in test_formula_token:
        for one_workbook in test_workbooks:
            for one_sheet in test_workbooks[one_workbook]:
                if one_workbook in formula_token and one_sheet in formula_token:
                    if one_workbook + '---' + one_sheet not in test_sheet_num:
                        test_sheet_num[one_workbook + '---' + one_sheet] = 0
                    if test_sheet_num[one_workbook + '---' + one_sheet] < 10:
                        test_res.append(formula_token)
                        test_sheet_num[one_workbook + '---' + one_sheet] += 1
    val_res = []
    val_sheet_num = {}
    for formula_token in test_formula_token:
        for one_workbook in validate_workbooks:
            for one_sheet in validate_workbooks[one_workbook]:
                if one_workbook in formula_token and one_sheet in formula_token:
                    if one_workbook + '---' + one_sheet not in val_sheet_num:
                        val_sheet_num[one_workbook + '---' + one_sheet] = 0
                    if val_sheet_num[one_workbook + '---' + one_sheet] < 10:
                        val_res.append(formula_token)
                        val_sheet_num[one_workbook + '---' + one_sheet] += 1
    print(len(test_res))
    print(len(val_res))
    with open('fortune500_test_formula.json', 'w') as f:
        json.dump(test_res, f)
    with open('fortune500_val_formula.json', 'w') as f:
        json.dump(val_res, f)

def check_round_size(eval_file, second_res_path):
    before_path=root_path + 'demo_before_features_mask2_fix/'
    filenames = set()
    filesheets = set()
    sheets = set()
    formula_tokens = set()
    # rerank_model = torch.load("/datadrive-2/data/reranking_linear_models/epoch_3")
    rerank_model = torch.load("/datadrive-2/data/best_model")
    filelist1 = os.listdir(root_path + 'dedup_model1_res')
    # filelist = list(set(filelist1) & set(filelist2))
    filelist = filelist1
    filelist.sort()
    with open('fortune500_formulatoken2r1c1.json','r') as f:
        top10domain_formulatoken2r1c1 = json.load(f)
    with open('r1c12template_fortune500_constant.json','r') as f:
        r1c12template_top10domain = json.load(f)
    with open("crosstable_formulas.json", 'r') as f:
        crosstable_formulas = json.load(f)
    with open(eval_file, 'r') as f:
        sampled_token = json.load(f)

    print('len filelist', len(filelist))

    dollar_num = 0
    dollar_list = []
    model1_suc = []
    shift_content_suc = []
    shift_content_mask_suc = []
    second_suc = []
    print('len filelist', len(filelist))

    multi_table = 0
    constrain_found = 0
    second_found_formu = 0
    first_found_formul = 0
    model1_fail_but_suc = 0
    template_fail = 0
    except_log_num=  0
    filenames = set()
    naive_suc = 0
    simple_suc = 0
    all_ = 0

    test_formula_token = []

    log_str = ''
    fail_fname = set()
    fail_sheetname = set()
    all_fname = set()
    all_sheetname = set()

    second_not_exists_num = 0

    sec_not_found = 0
    best1 = 0
    best1_allfound = 0
    nottrue_allfound = 0

    filename2num = {}

    topk = 10
    not_in_top_num = 0

    in_top_num = 0
    filetoken_list = []

    fail = 0
    found_range_path_num = 0
    row_shift_list = []
    col_shift_list = []
    for index, formula_token in enumerate(sampled_token):
        filename = formula_token.split('---')[0]
        if filename not in filename2num:
            filename2num[filename] = 0
        if filename2num[filename] > 10:
            continue
        filename2num[filename] += 1

        print('\033[1;35m index ' + str(index) + ' ' + str(len(sampled_token)) + ' ' + formula_token + '\033[0m')
        log_str +=  '#########################################index ' + str(index) + ' ' + str(len(sampled_token)) + ' ' + formula_token + '\n'
        formula_r1c1 = top10domain_formulatoken2r1c1[formula_token]
        ############## before compare first level
        if '!' in formula_r1c1:
            continue
        if 'RC[-1]+ROUND(10*R' in formula_r1c1:
            except_log_num += 1
            continue
        
        if formula_r1c1 in crosstable_formulas:
            multi_table += 1
            continue
        
        if formula_token not in sampled_token:
            continue
       
        with open(root_path + 'dedup_model1_naive_res/' + formula_token + '.json', 'r') as f:
            naive_res = json.load(f)
        # print('naive_res', naive_res)
        if os.path.exists(root_path + 'dedup_simple_res/' + formula_token + '.json'):
            with open(root_path + 'dedup_simple_res/' + formula_token + '.json', 'r') as f:
                simple_res = json.load(f)
            simple_r1c1 = top10domain_formulatoken2r1c1[simple_res[0]]
            # print('formula_r1c1', formula_r1c1)
            if simple_r1c1 == formula_r1c1:
                simple_suc += 1
        if len(naive_res) > 1:
            if naive_res[1] == True:
                naive_suc += 1
    
        with open(root_path+'dedup_model1_res/'+formula_token + '.json', 'r') as f:
            model1_res = json.load(f)

        if model1_res[4] == True:
            model1_suc.append(formula_token)

        found_formula_token = model1_res[1]
        # if '127534955648942888021337347766059868298-to20-ry2022-attachment-b.xlsx' in found_formula_token or '51062703087863555942928902850393994897-to20-model_gs_ry2022_grossloadry2021.xlsx' in found_formula_token:
            # continue
        if "$" in formula_r1c1:
            dollar_num += 1
            dollar_list.append(formula_r1c1)
        all_ += 1
        all_fname.add(formula_token.split('---')[0])
        all_sheetname.add(formula_token.split('---')[1])
        test_formula_token.append(formula_token)
        if formula_r1c1 not in r1c12template_top10domain:
            gt_template_id = -1
        else:
            gt_template_id = r1c12template_top10domain[formula_r1c1]
        if model1_res[3] not in r1c12template_top10domain:
            found_template_id = -1
        else:
            found_template_id = r1c12template_top10domain[model1_res[3]]
        
        
        # print('found_template_id', found_template_id)
        # print('gt_template_id', gt_template_id)
        if found_template_id != gt_template_id:
            template_fail += 1
            if tempid2num[gt_template_id] <= 1:
                single_tmp += 1
            # print('template_fail')
            continue

        found_range_path = root_path+'test_refcell_position/' + found_formula_token + '.json'
        gt_range_path = root_path+'test_refcell_position/' + formula_token + '.json'
        if not os.path.exists(gt_range_path):
            if model1_res[2] == model1_res[3]:
                # print('3188 model1_res', model1_res)
                # second_found_formu, first_found_formul, is_found = secc(second_found_formu, first_found_formul, formula)
                second_found_formu += 1
                in_top_num += 1
                first_found_formul += 1
                shift_content_suc.append(formula_token)
                constrain_found += 1
                is_found = True
                if model1_res[4] == False:
                    model1_fail_but_suc += 1
            print('not exists: gt_range_path')
            continue

        # print('found_range_path', found_range_path)
        if not os.path.exists(found_range_path):
            print('not exists: found_range_path')
            found_range_path_num += 1
            continue
        
        with open(gt_range_path, 'r') as f:
            gt_ref_cell = json.load(f)
        with open(found_range_path, 'r') as f:
            found_ref_cell = json.load(f)

        
        if len(gt_ref_cell) == 0:
            if found_template_id == gt_template_id:
                # print('3208 model1_res', model1_res)
                # second_found_formu, first_found_formul, is_found = secc(second_found_formu, first_found_formul, formula)
                second_found_formu += 1
                in_top_num += 1
                first_found_formul += 1
                constrain_found += 1
                is_found = True
                shift_content_suc.append(formula_token)
                if model1_res[4] == False:
                    model1_fail_but_suc += 1
            continue

        ########## second compare (first is in it)'second_res_specific_epoch3_c2/' second_res_specific_5_10
        if os.path.exists(second_res_path + formula_token + '.npy'): # second_res_specific_epoch23_c2
        # if os.path.exists(root_path + 'second_res_shift_content/' + formula_token + '.npy'):
            second_res_shift_content = np.load(second_res_path + formula_token + '.npy', allow_pickle=True).item()
        else:
            second_not_exists_num += 1
            print('second_res_shift_content not exits')
            continue

        merge_score ={}
        for item1 in second_res_shift_content:
            merge_score[item1]  = {}
            for item2 in second_res_shift_content[item1]:
                # merge_score[item1][item2] = (second_res_shift_content[item1][item2]*1 + (1-second_content_similarity[item1][item2])*0.)
                merge_score[item1][item2] = second_res_shift_content[item1][item2]*1
        ref_rc_list = list(set(second_res_shift_content.keys()))

        all_found = True
        all_first_found = True

        second_not_exists = False
        not_in_top = False
        in_top = False
        combination_list = [[(formula_token.split('---')[2] + '---' + formula_token.split('---')[3], 0)]]
        ref_r_template = [int(found_formula_token.split('---')[2])]
        ref_c_template = [int(found_formula_token.split('---')[3])]
        
        found_formula_r = int(found_formula_token.split('---')[2])
        found_formula_c = int(found_formula_token.split('---')[3])

        formula_r = int(formula_token.split('---')[2])
        formula_c = int(formula_token.split('---')[3])
        for ref_rc in ref_rc_list:
            ref_r = int(ref_rc.split('---')[0])
            ref_c = int(ref_rc.split('---')[1])

            ref_r_template.append(ref_r)
            ref_c_template.append(ref_c)

        
        for ref_ind, ref_rc in enumerate(ref_rc_list):
            ref_r = int(ref_rc.split('---')[0])
            ref_c = int(ref_rc.split('---')[1])

            for findex, item in enumerate(found_ref_cell):
                print(item)
                if item['R'] == ref_r and item['C'] == ref_c:
                    found_index = findex
                    r_dollar = item['R_dollar']
                    c_dollar = item['C_dollar']

            delta_row = ref_r - found_formula_r
            delta_col = ref_c - found_formula_c

            copy_row = formula_r + delta_row
            copy_col = formula_c + delta_col

            gt_rc = str(gt_ref_cell[found_index]['R']) + '---' + str(gt_ref_cell[found_index]['C'])
            gt_r = int(gt_rc.split('---')[0])
            gt_c = int(gt_rc.split('---')[1])

            row_shift_list.append(abs(gt_r - copy_row))
            col_shift_list.append(abs(gt_c - copy_col))

    print('row_shift_list', row_shift_list)
    print('col_shift_list', col_shift_list)

def eval_fintune(eval_file, after_path, second_res_path, save_path, log_path='log.txt'): # "fortune500_test_formula.json", second_res_path=root_path + second_res_specific_5_10
    # before_path=root_path + 'demo_before_features_mask2_fix/'
    before_path=root_path + 'demo_after_finegrain_16/'
    filenames = set()
    filesheets = set()
    sheets = set()
    formula_tokens = set()
    rerank_model = torch.load("/datadrive-2/data/reranking_finegrain_models/epoch_3")
    # rerank_model = torch.load("/datadrive-2/data/best_model")
    filelist1 = os.listdir(root_path + 'dedup_model1_res')
    # filelist = list(set(filelist1) & set(filelist2))
    filelist = filelist1
    filelist.sort()
    with open('fortune500_formulatoken2r1c1.json','r') as f:
        top10domain_formulatoken2r1c1 = json.load(f)
    with open('r1c12template_fortune500_constant.json','r') as f:
        r1c12template_top10domain = json.load(f)
    with open("crosstable_formulas.json", 'r') as f:
        crosstable_formulas = json.load(f)
    with open(eval_file, 'r') as f:
        sampled_token = json.load(f)
    with open("fortune500_workbook2company.json", 'r') as f:
        workbook2company = json.load(f)

    with open('reduced_formulas_ibm.json', 'r') as f:
        reduced_formulas = json.load(f)
    tempid2num = {}
    for formula_token in reduced_formulas:
        r1c1 = top10domain_formulatoken2r1c1[formula_token]
        if r1c1 not in r1c12template_top10domain:
            continue
        template_id = r1c12template_top10domain[r1c1]
        if template_id not in tempid2num:
            tempid2num[template_id] = 0
        tempid2num[template_id] += 1
    

    dollar_num = 0
    dollar_list = []
    model1_suc = []
    shift_content_suc = []
    shift_content_mask_suc = []
    second_suc = []
    print('len filelist', len(filelist))

    multi_table = 0
    constrain_found = 0
    second_found_formu = 0
    first_found_formul = 0
    model1_fail_but_suc = 0
    template_fail = 0
    except_log_num=  0
    filenames = set()
    naive_suc = 0
    simple_suc = 0
    all_ = 0

    test_formula_token = []

    log_str = ''
    fail_fname = set()
    fail_sheetname = set()
    all_fname = set()
    all_sheetname = set()

    second_not_exists_num = 0

    sec_not_found = 0
    best1 = 0
    best1_allfound = 0
    nottrue_allfound = 0

    filename2num = {}

    topk = 10
    not_in_top_num = 0

    in_top_num = 0
    filetoken_list = []

    fail = 0
    found_range_path_num = 0
    start_time = 0

    all_count = 0
    all_time = 0

    fail = 0
    len_not_same = 0

    single_tempate = 0
    for index, formula_token in enumerate(sampled_token):
        filename = formula_token.split('---')[0]
        if filename not in filename2num:
            filename2num[filename] = 0
        if filename2num[filename] > 10:
            continue
        if workbook2company[filename] not in ['ti']:
            continue
        # try:
        
        filename2num[filename] += 1
        # formula_token = filename.replace('.json', '')
        
        # if not os.path.exists(root_path+'model1_res/'+formula_token + '.json') or not os.path.exists(root_path + 'second_res_shift_epoch3_c2/' + formula_token + '.npy'):
        #     continue
        # if '127534955648942888021337347766059868298-to20-ry2022-attachment-b.xlsx' in formula_token or '51062703087863555942928902850393994897-to20-model_gs_ry2022_grossloadry2021.xlsx' in formula_token:
            # continue
        print('\033[1;35m index ' + str(index) + ' ' + str(len(sampled_token)) + ' ' + formula_token + '\033[0m')
        log_str +=  '#########################################index ' + str(index) + ' ' + str(len(sampled_token)) + ' ' + formula_token + '\n'
        formula_r1c1 = top10domain_formulatoken2r1c1[formula_token]
        print('workbook2company[filename]', workbook2company[filename])
        ############## before compare first level
        if '!' in formula_r1c1:
            continue
        if 'RC[-1]+ROUND(10*R' in formula_r1c1:
            except_log_num += 1
            continue
        
        if formula_r1c1 in crosstable_formulas:
            multi_table += 1
            print('multi_table')
            continue
        
        if formula_token not in sampled_token:
            print('not in sampled')
            continue
    
    
        # with open(root_path + 'dedup_model1_naive_res/' + formula_token + '.json', 'r') as f:
            # naive_res = json.load(f)
        # print('naive_res', naive_res)
        # if os.path.exists(root_path + 'dedup_simple_res/' + formula_token + '.json'):
            # with open(root_path + 'dedup_simple_res/' + formula_token + '.json', 'r') as f:
                # simple_res = json.load(f)
            # simple_r1c1 = top10domain_formulatoken2r1c1[simple_res[0]]
            # print('formula_r1c1', formula_r1c1)
            # if simple_r1c1 == formula_r1c1:
                # simple_suc += 1
        # if len(naive_res) > 1:
        #     if naive_res[1] == True:
        #         naive_suc += 1
        if not os.path.exists(root_path+'company_model1_res/'+formula_token + '.json'):
            continue
        # if not os.path.exists(root_path+'dedup_model1_res/'+formula_token + '.json'):
            # continue
        # with open(root_path+'dedup_model1_res/'+formula_token + '.json', 'r') as f:
        with open(root_path+'company_model1_res/'+formula_token + '.json', 'r') as f:
            model1_res = json.load(f)
        if model1_res[3] == '':
            continue
        if model1_res[4] == True:
            model1_suc.append(formula_token)

        found_formula_token = model1_res[1]
        # if '127534955648942888021337347766059868298-to20-ry2022-attachment-b.xlsx' in found_formula_token or '51062703087863555942928902850393994897-to20-model_gs_ry2022_grossloadry2021.xlsx' in found_formula_token:
            # continue
        if "$" in formula_r1c1:
            dollar_num += 1
            dollar_list.append(formula_r1c1)
        all_ += 1
        all_fname.add(formula_token.split('---')[0])
        all_sheetname.add(formula_token.split('---')[1])
        test_formula_token.append(formula_token)
        if formula_r1c1 not in r1c12template_top10domain:
            gt_template_id = -1
        else:
            gt_template_id = r1c12template_top10domain[formula_r1c1]
        if model1_res[3] not in r1c12template_top10domain:
            found_template_id = -1
        else:
            found_template_id = r1c12template_top10domain[model1_res[3]]
        
        if found_template_id != gt_template_id:
            print('found_template_id', found_template_id)
            print('gt_template_id', gt_template_id)
            template_fail += 1
            print(model1_res)
            print('template_fail')
            if gt_template_id in tempid2num:
                if tempid2num[gt_template_id] <= 1:
                    print("is_single")
                    single_tempate += 1
            continue

        found_range_path = root_path+'test_refcell_position/' + found_formula_token + '.json'
        gt_range_path = root_path+'test_refcell_position/' + formula_token + '.json'
        if not os.path.exists(gt_range_path):
            if model1_res[2] == model1_res[3]:
                # print('3188 model1_res', model1_res)
                # second_found_formu, first_found_formul, is_found = secc(second_found_formu, first_found_formul, formula)
                second_found_formu += 1
                in_top_num += 1
                first_found_formul += 1
                shift_content_suc.append(formula_token)
                constrain_found += 1
                is_found = True
                if model1_res[4] == False:
                    model1_fail_but_suc += 1
            print('not exists: gt_range_path')
            continue

        # print('found_range_path', found_range_path)
        if not os.path.exists(found_range_path):
            print('not exists: found_range_path')
            found_range_path_num += 1
            continue
        
        with open(gt_range_path, 'r') as f:
            gt_ref_cell = json.load(f)
        with open(found_range_path, 'r') as f:
            found_ref_cell = json.load(f)

        
        if len(gt_ref_cell) == 0:
            if found_template_id == gt_template_id:
                # print('3208 model1_res', model1_res)
                # second_found_formu, first_found_formul, is_found = secc(second_found_formu, first_found_formul, formula)
                second_found_formu += 1
                in_top_num += 1
                first_found_formul += 1
                constrain_found += 1
                is_found = True
                shift_content_suc.append(formula_token)
                if model1_res[4] == False:
                    model1_fail_but_suc += 1
            print('len gtref == 0')
            continue

        if len(gt_ref_cell) != len(found_ref_cell):
            print('model1_res', model1_res)
            len_not_same += 1
            continue
        ########## second compare (first is in it)'second_res_specific_epoch3_c2/' second_res_specific_5_10
        if os.path.exists(second_res_path + formula_token + '.npy'): # second_res_specific_epoch23_c2
        # if os.path.exists(root_path + 'second_res_shift_content/' + formula_token + '.npy'):
            second_res_shift_content = np.load(second_res_path + formula_token + '.npy', allow_pickle=True).item()
        else:
            second_not_exists_num += 1
            print('second_res_shift_content not exits')
            continue

        # if os.path.exists(root_path + "second_content_similarity/" + formula_token + '.json'):
        # # if os.path.exists(root_path + 'second_res_shift_content/' + formula_token + '.npy'):
        #     with open(root_path + "second_content_similarity/" + formula_token + '.json', 'r') as f:
        #         second_content_similarity = json.load(f)
        # else:
        #     print('second_content_similarity not exits')
        #     continue
        
        # if os.path.exists(root_path + 'second_res_shift_epoch3_c2/' + formula_token + '.npy'):# second_res_shift_content, second_res_pretrain_4_1, second_res_pretrain_epoch8, second_res_shift_epoch3_c2, second_res_specific_epoch3_c2
        #     second_res = np.load(root_path + 'second_res_shift_epoch3_c2/' + formula_token + '.npy', allow_pickle=True).item()
        # else:
        #     second_res.append(formula_token)
        
        all_count += 1
        start_time = time.time()
        merge_score ={}
        for item1 in second_res_shift_content:
            merge_score[item1]  = {}
            for item2 in second_res_shift_content[item1]:
                # merge_score[item1][item2] = (second_res_shift_content[item1][item2]*1 + (1-second_content_similarity[item1][item2])*0.)
                merge_score[item1][item2] = second_res_shift_content[item1][item2]*1
        ref_rc_list = list(set(second_res_shift_content.keys()))

        all_found = True
        all_first_found = True

        second_not_exists = False
        not_in_top = False
        in_top = False
        combination_list = [[(formula_token.split('---')[2] + '---' + formula_token.split('---')[3], 0)]]
        ref_r_template = [int(found_formula_token.split('---')[2])]
        ref_c_template = [int(found_formula_token.split('---')[3])]
        for ref_rc in ref_rc_list:
            ref_r = int(ref_rc.split('---')[0])
            ref_c = int(ref_rc.split('---')[1])
            ref_r_template.append(ref_r)
            ref_c_template.append(ref_c)

        # print('ref_r_template', ref_r_template)
        # print('ref_c_template', ref_c_template)
        print('ref_rc_list', ref_rc_list)
        for ref_ind, ref_rc in enumerate(ref_rc_list):
            ref_r = int(ref_rc.split('---')[0])
            ref_c = int(ref_rc.split('---')[1])
            print(formula_r1c1)
            print('found_ref_cell', found_ref_cell)
            print('gt_ref_cell', gt_ref_cell)
            for findex, item in enumerate(found_ref_cell):
                print(item)
                if item['R'] == ref_r and item['C'] == ref_c:
                    found_index = findex
                    r_dollar = item['R_dollar']
                    c_dollar = item['C_dollar']
                    
            print('ref_rc', ref_rc)
            print('r_dollar', r_dollar)
            print('c_dollar', c_dollar)
            gt_rc = str(gt_ref_cell[found_index]['R']) + '---' + str(gt_ref_cell[found_index]['C'])
            print('gt_rc', gt_rc)
            if 'best_distance' in second_res_shift_content[ref_rc]:
                second_res_shift_content[ref_rc].pop('best_distance')
            if 'best_row' in second_res_shift_content[ref_rc]:
                second_res_shift_content[ref_rc].pop('best_row')
            if 'best_row_col' in second_res_shift_content[ref_rc]:
                second_res_shift_content[ref_rc].pop('best_row_col')
            
            
            sorted_second_res = sorted(merge_score[ref_rc].items(), key=lambda x:x[1])
            best_tuple = sorted(merge_score[ref_rc].items(), key=lambda x:x[1])

            gt_r = int(gt_rc.split('---')[0])
            gt_c = int(gt_rc.split('---')[1])
            # print("best_tuple", best_tuple)
            # new_best_tuple = []
            # for item in best_tuple:
                
            #     cand_r = int(item[0].split('---')[0])
            #     cand_c = int(item[0].split('---')[1])
            #     if r_dollar:
            #         if cand_r != ref_r:
            #             continue

            #     if c_dollar:
            #         if cand_c != ref_c:
            #             continue
            #     new_best_tuple.append(item)
            # if len(new_best_tuple) == 0:
            #     all_found = False
            #     continue
            new_best_tuple=  best_tuple
            print("new_best_tuple", new_best_tuple[0:topk])
            best_tuple = new_best_tuple[0]
            print('found_formula_token', found_formula_token)
            log_str +=  'found_formula_token :' + str(found_formula_token) + '\n'
            target_before_feature = np.load(before_path + found_formula_token.split('---')[0] + '---' + found_formula_token.split('---')[1] + '---' + ref_rc + '.npy', allow_pickle=True)
            best_score = 0
            # cand_list = [item for item in new_best_tuple[0:topk]]
            cand_list = copy.deepcopy(new_best_tuple[0:topk])
            print('new_best_tuple[0:topk].keys()', cand_list)
            ## -------------------------------------- Rerank Model ------------------------------
            
            # cand_list.append((gt_rc, 0))
            # for cand in cand_list:
            # # for cand in new_best_tuple[0:topk]:
            #     # print('cand:', cand)
            #     # log_str +=  'cand:' + str(cand) + '\n'
            #     cand_rc, score = cand
            #     print('after path', before_path + formula_token.split('---')[0] + '---' + formula_token.split('---')[1] + '---' + cand_rc + '.npy')
            #     if not os.path.exists(before_path + formula_token.split('---')[0] + '---' + formula_token.split('---')[1] + '---' + cand_rc + '.npy'):
            #         continue
            #     cand_before_feature = np.load(before_path + formula_token.split('---')[0] + '---' + formula_token.split('---')[1] + '---' + cand_rc + '.npy', allow_pickle=True)
            #     delta_feature = cand_before_feature - target_before_feature
            #     rerank_feature = torch.from_numpy(np.array([cand_before_feature,target_before_feature,delta_feature]))
            #     # delta_feature = torch.from_numpy(delta_feature.reshape(1,100,10,16)).to(torch.float32)
            #     print('rerank_feature', rerank_feature.shape)
            #     cand_score = rerank_model(rerank_feature)
            #     if cand_score[0][1] > best_score:
            #         best_score = cand_score[0][1]
            #         best_tuple = cand

            #     print('cand_score:', cand_score)
            #     log_str +=  'cand_score:' + str(cand_score) + '\n'
            ## ------------------------------------- End -----------------------------------------------------------
            found_rc_shift_content = best_tuple[0]
            
            
            best_tuple_r = int(best_tuple[0].split('---')[0])
            best_tuple_c = int(best_tuple[0].split('---')[1])
            print("************")
            print('gt_rc', gt_rc)
            print('best_rc', best_tuple[0])
            
            ############ test one ref level ######################
            first_gt_c = int(gt_c/10)*10 + 1 if gt_c % 10 != 0 else int((gt_c-1)/10)*10 + 1
            first_gt_r = int(gt_r/100)*100 + 1 if gt_r % 100 != 0 else int((gt_c-1)/100)*100 + 1
            first_found_c = int(best_tuple_c/10)*10 + 1 if best_tuple_c % 10 != 0 else int((best_tuple_c-1)/10)*10 + 1
            first_found_r = int(best_tuple_r/100)*100 + 1 if gt_c % 100 != 0 else int((best_tuple_r-1)/100)*100 + 1
            if first_gt_c != first_found_c or first_gt_r != first_found_r:
                all_first_found = False

            cand_list= [item[0] for item in cand_list]
            if gt_rc not in cand_list:
                not_in_top = True
                print('cand_list', cand_list)
                log_str += str(cand_list) + '\n'
                print('not_in_top', not_in_top)
                log_str += 'not_in_top\n'
            else:
                in_top = True
                # if found_rc_shift_content != gt_rc:
                    
            filetoken_list.append(formula_token)
            if gt_rc in second_res_shift_content[ref_rc]:
                # log_str +=  'gt_second_res:' + str(second_res_shift_content[ref_rc][gt_rc]) + '\n'
                print('gt_second_res:' + str(second_res_shift_content[ref_rc][gt_rc]))
            else:
                second_not_exists = True
                # log_str += 'gt_second_res not exists.'
                # print("gt_second_res not exists.")
            if found_rc_shift_content == gt_rc:
                is_found = True
                shift_content_suc.append(formula_token)
            else:
                all_found = False
                if model1_res[4] == True:
                    filetoken_list.append(formula_token)
                    log_str +=  'formula_token: ' + formula_token + '\n'
                    log_str +=  'found_formula_token: ' + found_formula_token + '\n'
                    print('gt_r1c1', formula_r1c1)
                    log_str +=  'gt_r1c1 ' + formula_r1c1 + '\n'
                    print('found_r1c1', model1_res[3])
                    log_str +=  'found_r1c1 ' + model1_res[3] + '\n'
                    print('gt_ref_cell', gt_ref_cell)
                    log_str +=  'gt_ref_cell ' + str(gt_ref_cell) + '\n'
                    print('found_ref_cell', found_ref_cell)
                    log_str +=  'found_ref_cell ' + str(found_ref_cell) + '\n'
                    print('gt_rc', 'best_tuple[0]', gt_rc, best_tuple[0])
                    log_str +=  'gt_rc,  best_tuple[0]:' + str(gt_rc) +',' +str(best_tuple[0]) + '\n'
                    # first_res = np.load(root_path + 'first_res_specific_epoch23_c2/' + formula_token + '.npy', allow_pickle=True).item()
                    # print('first_res', first_res[ref_rc])
                    print('second_res', sorted_second_res[0:5])
                    log_str +=  'second_res:' + str(sorted_second_res[0:10]) + '\n'
                    print('second_res_shift_content', sorted(second_res_shift_content[ref_rc].items(), key=lambda x:x[1])[0:5])
                    log_str +=  'second_res_shift_content:' + str(sorted(second_res_shift_content[ref_rc].items(), key=lambda x:x[1])[0:10]) + '\n'
                    if gt_rc in second_res_shift_content[ref_rc]:
                        log_str +=  'gt_second_res:' + str(second_res_shift_content[ref_rc][gt_rc]) + '\n'
                        print('gt_second_res:' + str(second_res_shift_content[ref_rc][gt_rc]))
                    else:
                        second_not_exists = True
                        log_str += 'gt_second_res not exists.'
                        print("gt_second_res not exists.")
                # if model1_res[4] == True:
                
                    # print('content', sorted(second_content_similarity[ref_rc].items(), key=lambda x:x[1])[0:5])
                    # log_str +=  'content:' + str(sorted(second_content_similarity[ref_rc].items(), key=lambda x:x[1])[0:10]) + '\n'
                    # if gt_rc in second_content_similarity[ref_rc]:
                        # log_str +=  'gt_content:' + str(second_content_similarity[ref_rc][gt_rc]) + '\n'
                    # else:
                        # log_str += 'gt_content not exists.'
                #     fail_fname.add(formula_token.split('---')[0])
                #     fail_sheetname.add(formula_token.split('---')[1])
            ############ end test one ref level ######################

            ############ combination constrain ######################
        #     print('second_res', sorted(merge_score[ref_rc].items(), key=lambda x:x[1])[0:10])
        #     new_best_combination = []
        #     for before_combination in combination_list:
        #         for this_ref in sorted(merge_score[ref_rc].items(), key=lambda x:x[1])[0:10]:
        #             tmp_before_combination = copy.deepcopy(before_combination)
        #             tmp_before_combination.append(this_ref)
        #             new_best_combination.append(tmp_before_combination)
        #     combination_list = new_best_combination
        #     # print('add combination_list', combination_list)
        #     #### check
        #     new_best_combination = []
        #     for combination in combination_list:
        #         is_bad_combination = False
        #         for cid in range(0, len(combination)):
        #             combination_r = int(combination[cid][0].split('---')[0])
        #             combination_c = int(combination[cid][0].split('---')[1])
                    
        #             for other_cid in range(cid, len(combination)):
        #                 if cid == other_cid:
        #                     continue
        #                 other_combination_r = int(combination[other_cid][0].split('---')[0])
        #                 other_combination_c = int(combination[other_cid][0].split('---')[1])
        #                 if (ref_r_template[cid] == ref_r_template[other_cid] and combination_r != other_combination_r) or (ref_c_template[cid] == ref_c_template[other_cid] and combination_c != other_combination_c):
        #                     is_bad_combination = True
        #                     break
        #             if is_bad_combination:
        #                 break
        #         if not is_bad_combination:
        #             new_best_combination.append(combination)
        #     # print('new_best_combination', new_best_combination)
        #     combination_list = new_best_combination
        #     tuple_list = []
        #     for combination_index,combination in enumerate(combination_list):
        #         score_list = [item[1] for item in combination]
        #         score = np.array(score_list).mean()
        #         tuple_list.append([combination, score])

        #     tuple_list = sorted(tuple_list, key=lambda x:x[1])
        #     combination_list = [item[0] for item in tuple_list[0:10]]
        #     print('combination_list', combination_list)
            
        # best_score = 100
        # best_index = -1
        # for combination_index1,combination in enumerate(combination_list):
        #     score_list = [item[1] for item in combination]
        #     score = np.array(score_list).mean()
        #     if score < best_score:
        #         best_score = score
        #         best_index = combination_index1

        # # if len(combination_list) != len(ref_rc_list) + 1:
        # #     continue
        # if best_index == -1:
        #     best1 += 1
        #     if all_found:
        #         print('\033[1;34m index ' + str(index) + ' ' + str(len(sampled_token)) + ' ' + formula_token + '\033[0m')
        #         print('ref_r_template', ref_r_template)
        #         print('ref_c_template', ref_c_template)
        #         print('combination_list', combination_list)
        #         best1_allfound += 1
        #     continue
        # # print('best_score', best_score)
        # # print('best_index', best_index)
        # best_combination = combination_list[best_index]
        # print('best_combination', best_combination)
        # # print('ref_rc_list', len(ref_rc_list))
        # is_true = True
        # for ref_index, ref_rc in enumerate(ref_rc_list):
        #     print('ref_index,', ref_index)
        #     for index, item in enumerate(found_ref_cell):
        #         ref_r = int(ref_rc.split('---')[0])
        #         ref_c = int(ref_rc.split('---')[1])
        #         if item['R'] == ref_r and item['C'] == ref_c:
        #             found_index = index

        #     gt_rc = str(gt_ref_cell[found_index]['R']) + '---' + str(gt_ref_cell[found_index]['C'])
        #     if ref_index + 1 >= len(best_combination):
        #         is_true = False
        #         break
        #     com_rc = best_combination[ref_index + 1][0]
        #     # print("##############")
        #     # print('gt_rc', gt_rc)
        #     # print('com_rc', com_rc)
        #     if gt_rc != com_rc:
        #         is_true = False

        # if is_true:
        #     constrain_found += 1
        # else:
        #     if all_found:
        #         # print('\033[1;35m index ' + str(index) + ' ' + str(len(sampled_token)) + ' ' + formula_token + '\033[0m')
        #         # print('ref_r_template', ref_r_template)
        #         # print('ref_c_template', ref_c_template)
        #         print('\033[1;34m best_combination ' + str(best_combination)+ '\033[0m')
        #         nottrue_allfound = 0
        if not_in_top:
            not_in_top_num += 1
        if in_top:
            in_top_num += 1
        if second_not_exists:
            second_not_exists_num += 1
        if all_found:
            second_found_formu += 1
            first_found_formul += 1
        else:
            fail += 1
            if all_first_found:
                first_found_formul += 1
        end_time = time.time()
        all_time += (end_time - start_time)
        # if len(ref_rc_list) > 2:
        #     break
            # if found_rc == gt_rc:
            #     second_suc.append(formula_token)
            # if found_rc_shift_content != gt_rc and found_rc == gt_rc:
            #     print('gt_rc', gt_rc)
            #     print('found epoch 3', found_rc_shift_content)
            #     print('found_rc_shift_content', found_rc)
                
            #     print('model1_res', model1_res)
            #     filenames.add(model1_res[0].split('---')[0])
            # if found_rc_shift_content_mask == gt_rc:
                # shift_content_mask_suc.append(formula_token)
            

            # if found_rc_shift_content != gt_rc or found_rc_shift_content_mask != gt_rc:
            #     # print('second_res_shift_content', second_res_shift_content)
            #     # print('second_res_shift_content_mask', second_res_shift_content_mask)
            #     for keyword in second_res_shift_content[ref_rc]:
            #         # if second_res_shift_content[ref_rc][keyword] == second_res_shift_content_mask[ref_rc][keyword]:
            #         #     print('same')
            #         if second_res_shift_content[ref_rc][keyword] != second_res_shift_content_mask[ref_rc][keyword]:
            #             print('not same')
            #     break
        # break
        # except:
        #     fail += 1
        #     continue
    with open(log_path, 'w') as f:
        f.write(log_str)
    with open('filetoken_list.json', 'w') as f:
        json.dump(list(set(filetoken_list)), f)
    # with open('fortune500_test_formula_token.json', 'w') as f:
        # json.dump(test_formula_token, f)
    print('filenames', filenames)
    print('shift_content_suc', len(shift_content_suc))
    print('first_found_formul', first_found_formul)
    print('second_found_formu', second_found_formu)
    print('model1_fail_but_suc', model1_fail_but_suc)
    print('model1_suc', len(model1_suc))
    print('second_suc', len(second_suc))
    print('filenames', len(filenames))

    print('template_fail', template_fail)
    print('single_tempate',single_tempate)
    # print('fail_files', fail_files)
    # print('fail_sheets', fail_sheets)
    # print('fail_filesheets', fail_filesheets)
    # print('fail_tokens', fail_tokens)
    # print('mult-table', multi_table)

    print('simple_suc',simple_suc)
    print('naive_suc', naive_suc)
    print('all_', all_)
    print("dollar_num", dollar_num)

    print('not_in_top_num', not_in_top_num)
    print('in_top_num', in_top_num)

    # print('fail_fname', len(list(fail_fname)))
    # print('fail_sheetname', len(list(fail_sheetname)))
    # print('all_fname', len(list(all_fname)))
    # print('all_sheetname', len(list(all_sheetname)))
    print('second_not_exists_num', second_not_exists_num)
    print('constrain_found', constrain_found)
    print('best1', best1)
    print('best1_allfound', best1_allfound)
    print('nottrue_allfound', nottrue_allfound)
    print("fail", fail)
    print('found_range_path_num', found_range_path_num)
    # with open(save_path, 'w') as f:
    #     json.dump([second_found_formu, all_, second_found_formu/all_], f)
    # with open("dollar_list.json", 'w') as f:
    #     json.dump(dollar_list, f)

    print('all_time:', all_time)
    print('avg_time:', all_time / all_count)

    print('fail', fail)
    print('len_not_same', len_not_same)

    # print(tempid2num)
# def check_first_second_res():
#     need_rerun = []
#     filelist = os.listdir('/datadrive-2/data/fortune_test/second_res_specific_epoch23_c2')
#     for filename in filelist:
#         res = 
def upperbound():
    with open('fortune500_formulatoken2r1c1.json','r') as f:
        top10domain_formulatoken2r1c1 = json.load(f)
    with open('r1c12template_fortune500_constant.json','r') as f:
        r1c12template_top10domain = json.load(f)
    with open("fortune500_test_formula.json", 'r') as f:
        sampled_token = json.load(f)

    filesheet2formula_token = {}
    for formula_token in top10domain_formulatoken2r1c1:
        filesheet=  formula_token.split('---')[0] + '---' + formula_token.split('---')[1]
        if filesheet not in filesheet2formula_token:
            filesheet2formula_token[filesheet] = []
        filesheet2formula_token[filesheet].append(formula_token)
    all_found = 0
    all_temp_found = 0
    round5_found = 0
    round5_temp_found = 0
    all_ = 0
    for index, formula_token in enumerate(sampled_token):
        print(index, len(sampled_token))
        af = False
        atf = False
        rf = False
        rtf  = False
        with open(root_path + 'dedup_model1_res/' + formula_token + '.json', 'r') as f:
            model1_res = json.load(f)
        target_r1c1 = model1_res[2]
        if target_r1c1 in r1c12template_top10domain:
            target_template_id = r1c12template_top10domain[target_r1c1]
        else:
            target_template_id = -1
        origin_r = formula_token.split('---')[2]
        origin_c = formula_token.split('---')[3]
        origin_filesheet = formula_token.split('---')[0] + '---' + formula_token.split('---')[1]
        
        found_formula_token = model1_res[1]
        found_r = found_formula_token.split('---')[2]
        found_c = found_formula_token.split('---')[3]
        found_filesheet = found_formula_token.split('---')[0] + '---' + found_formula_token.split('---')[1]
        for other_formula_token in filesheet2formula_token[found_filesheet]:
            other_filesheet = other_formula_token.split('---')[0] + '---' + other_formula_token.split('---')[1]
            other_r = other_formula_token.split('---')[2]
            other_c = other_formula_token.split('---')[3]
            if other_filesheet != found_filesheet:
                continue
            other_r1c1 = top10domain_formulatoken2r1c1[other_formula_token]
            if other_r1c1 in r1c12template_top10domain:
                other_template_id = r1c12template_top10domain[other_r1c1]
            else:
                other_template_id = -1
            if other_template_id == target_template_id:
                atf = True
            if other_r1c1 == target_r1c1:
                af = True
            if ((int(other_c) - int(origin_c))**2 + (int(other_r) - int(origin_r))**2)**(1/2) < 10:
                if other_template_id == target_template_id:
                    rtf = True
                if other_r1c1 == target_r1c1:
                    rf = True
        if atf:
            all_temp_found += 1
        if af:
            all_found += 1
        if rtf:
            round5_temp_found += 1
        if rf:
            round5_found += 1
        all_ += 1

    print("all_temp_found", all_temp_found)
    print("all_found", all_found)
    print("round5_temp_found", round5_temp_found)
    print("round5_found", round5_found)
    print("all_", all_)
def cross_validate(model_id):
    # model_id = 4
    if not os.path.exists(root_path.replace('-2', '-3') + 'validate_cross/'+str(model_id)):
        os.mkdir(root_path.replace('-2', '-3') + 'validate_cross/'+str(model_id))
    if not os.path.exists(root_path.replace('-2', '-3') + 'validate_cross/'+str(model_id)+'/demo_after_specific/'):
        os.mkdir(root_path.replace('-2', '-3') + 'validate_cross/'+str(model_id)+'/demo_after_specific/')
    if not os.path.exists(root_path.replace('-2', '-3') + 'validate_cross/'+str(model_id)+'/first_res_specific/'):
        os.mkdir(root_path.replace('-2', '-3') + 'validate_cross/'+str(model_id)+'/first_res_specific/')
    if not os.path.exists(root_path.replace('-2', '-3') + 'validate_cross/'+str(model_id)+'/second_res_specific/'):
        os.mkdir(root_path.replace('-2', '-3') + 'validate_cross/'+str(model_id)+'/second_res_specific/')
    # demo_tile_features_fix
    test_finetune(1,1,model_path = '/datadrive-2/data/cross_finetune_specific_l2/epoch_'+ str(model_id), \
        tile_path = root_path + 'demo_tile_features_fix/',\
        before_path=root_path + 'cross_before_features/', \
        cross_path = root_path + 'demo_before_features_mask2_fix/', \
        after_path=root_path.replace('-2', '-3') + 'validate_cross/'+str(model_id)+'/demo_after_specific/', \
        first_save_path=root_path.replace('-2', '-3') + 'validate_cross/' + str(model_id) +'/first_res_specific/', \
        second_save_path = root_path.replace('-2', '-3')  + 'validate_cross/' + str(model_id) + '/second_res_specific/', \
        mask=2,\
        validate = True, \
        cross=True) #### please rename path  # demo_before_features_mask2_fix demo_after_specific 
    # 
def validate(model_id):
    # model_id = model_id
    if not os.path.exists(root_path.replace('-2', '-3') + 'validate/'+str(model_id)):
        os.mkdir(root_path.replace('-2', '-3') + 'validate/'+str(model_id))
    if not os.path.exists(root_path.replace('-2', '-3') + 'validate/'+str(model_id)+'/demo_after_specific/'):
        os.mkdir(root_path.replace('-2', '-3') + 'validate/'+str(model_id)+'/demo_after_specific/')
    if not os.path.exists(root_path.replace('-2', '-3') + 'validate/'+str(model_id)+'/first_res_specific/'):
        os.mkdir(root_path.replace('-2', '-3') + 'validate/'+str(model_id)+'/first_res_specific/')
    if not os.path.exists(root_path.replace('-2', '-3') + 'validate/'+str(model_id)+'/second_res_specific/'):
        os.mkdir(root_path.replace('-2', '-3') + 'validate/'+str(model_id)+'/second_res_specific/')
    # demo_tile_features_fix
    test_finetune(1,1,model_path = '/datadrive-2/data/finetune_specific_l2_1_5/epoch_'+ str(model_id), \
        tile_path = root_path + 'demo_tile_features_fix/',\
        before_path=root_path + 'demo_before_features_mask2_fix/', \
        cross_path = root_path + 'demo_before_features_mask2_fix/', \
        after_path=root_path.replace('-2', '-3') + 'validate/'+str(model_id)+'/demo_after_specific/', \
        first_save_path=root_path.replace('-2', '-3') + 'validate/' + str(model_id) +'/first_res_specific/', \
        second_save_path = root_path.replace('-2', '-3')  + 'validate/' + str(model_id) + '/second_res_specific/', \
        mask=2,\
        validate = True, \
        cross=False) #### please rename path  # demo_before_features_mask2_fix demo_after_specific 

def is_all_same():
    # distance_dict = np.load("distance_dict.npy", allow_pickle=True).item()
    # print('distance_dict', distance_dict["244228682981234450530384331754331956360-mls-mqc-qos-converter-v1.xlsx---MQCtoMLSQOS"])
    
    # with open(root_path + 'company_model1_similar_sheet/' + "244228682981234450530384331754331956360-mls-mqc-qos-converter-v1.xlsx---MQCtoMLSQOS" + '.json', 'r') as f:
    #     similar_sheets = json.load(f)
    # print('similar_sheets', similar_sheets)

    # with open(root_path + 'company_model1_res/' + "244228682981234450530384331754331956360-mls-mqc-qos-converter-v1.xlsx---MQCtoMLSQOS---15---2" + '.json', 'r') as f:
    #     company_model1_res = json.load(f)
    # print('company_model1_res', company_model1_res)
    # res1 = np.load(root_path + 'demo_before_features_mask2_fix/' + "127534955648942888021337347766059868298-to20-ry2022-attachment-b.xlsx---20-RevenueCredits---14---5.npy", allow_pickle=True)
    # res2 = np.load(root_path + 'demo_before_features_mask2_fix/' + "127534955648942888021337347766059868298-to20-ry2022-attachment-b.xlsx---20-RevenueCredits---24---4.npy", allow_pickle=True)
    # res3 = np.load(root_path + 'demo_before_features_mask2_fix/' + "51062703087863555942928902850393994897-to20-model_gs_ry2022_grossloadry2021.xlsx---20-RevenueCredits---14---5.npy", allow_pickle=True)
    # res1 = np.load(root_path + 'demo_before_features_mask2_fix/' + "51062703087863555942928902850393994897-to20-model_gs_ry2022_grossloadry2021.xlsx---12-DepRates---10---8.npy", allow_pickle=True)
    # res2 = np.load(root_path + 'demo_before_features_mask2_fix/' + "127534955648942888021337347766059868298-to20-ry2022-attachment-b.xlsx---12-DepRates---10---8.npy", allow_pickle=True)
    # res3 = np.load(root_path + 'demo_before_features_mask2_fix/' + "127534955648942888021337347766059868298-to20-ry2022-attachment-b.xlsx---12-DepRates---48---3.npy", allow_pickle=True)
    
    # res1 = np.load(root_path + 'demo_before_features_mask2_fix/' + "302639065149441824925691920801208145121-llc-curves.xlsx---Ln=10---171---2.npy", allow_pickle=True)
    # res2 = np.load(root_path + 'demo_before_features_mask2_fix/' + "302639065149441824925691920801208145121-llc-curves.xlsx---Ln=6---171---2.npy", allow_pickle=True)
    # res1 = np.load(root_path + 'sheet_after_features/' + "110059629631842288643883464290182071710-table-of-contents.xlsx---2017 Sales---1---1.npy", allow_pickle=True)
    # res2 = np.load(root_path + 'sheet_after_features/' + "328621492309825452137276453653810262580-punch-clock-sample.xlsx---MyTimeSheet---1---1.npy", allow_pickle=True)
    # print(euclidean(res1, res2))
    # print((res1 == res2).all())

    # res1 = np.load(root_path + 'sheet_after_features/' + "244228682981234450530384331754331956360-mls-mqc-qos-converter-v1.xlsx---MQCtoMLSQOS---1---1.npy", allow_pickle=True)
    # res2 = np.load(root_path + 'sheet_after_features/' + "216377366699890469054590655152413942641-ucs-boq-25aug2014_-_sf.xlsx---ABL-UCS_BOQ---1---1.npy", allow_pickle=True)
    # print(euclidean(res1, res2))
    # print((res1 == res2).all())
    # print(euclidean(res1, res2))
    # print((res1 == res3).all())

    # with open("../Demo/fix_fortune500/127534955648942888021337347766059868298-to20-ry2022-attachment-b.xlsx.json", 'r') as f:
    #     workbook_json1 = json.load(f)
    # with open("../Demo/fix_fortune500/51062703087863555942928902850393994897-to20-model_gs_ry2022_grossloadry2021.xlsx.json", 'r') as f:
    #     workbook_json2 = json.load(f)
    # generate_demo_features("127534955648942888021337347766059868298-to20-ry2022-attachment-b.xlsx", "12-DepRates", workbook_json1, 10, 8, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # generate_demo_features("51062703087863555942928902850393994897-to20-model_gs_ry2022_grossloadry2021.xlsx", "12-DepRates", workbook_json2, 10, 8, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # with open("../Demo/fix_fortune500/162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx.json", 'r') as f:
    #     workbook_json1 = json.load(f)
    # with open("../Demo/fix_fortune500/162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx.json", 'r') as f:
    #     workbook_json2 = json.load(f)
    # generate_demo_features("162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx", "beagleLT(DDR3)", workbook_json1, 11, 2, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # generate_demo_features("162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx", "EVM_1_3 (DDR3)", workbook_json2, 11, 2, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # with open(root_path + 'demo_tile_features_fix/' + "127534955648942888021337347766059868298-to20-ry2022-attachment-b.xlsx---12-DepRates---10---9.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "51062703087863555942928902850393994897-to20-model_gs_ry2022_grossloadry2021.xlsx---12-DepRates---10---9.json", 'r') as f:
    #     res2 = json.load(f)

    # with open("../Demo/fix_fortune500/162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx.json", 'r') as f:
    #     workbook_json1 = json.load(f)
    # with open("../Demo/fix_fortune500/162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx.json", 'r') as f:
    #     workbook_json2 = json.load(f)
    # generate_demo_features("162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx", "beagleLT(DDR3)", workbook_json1, 11, 2, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # generate_demo_features("162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx", "EVM_1_3 (DDR3)", workbook_json2, 11, 2, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # with open(root_path + 'demo_tile_features_fix/' + "162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx---beagleLT(DDR3)---11---2.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx---EVM_1_3 (DDR3)---11---2.json", 'r') as f:
    #     res2 = json.load(f)

    # with open("../Demo/fix_fortune500/163350182546811267831688234176624693724-to20-ry2022-attachment-d.xlsx.json", 'r') as f:
    #     workbook_json1 = json.load(f)
    # with open("../Demo/fix_fortune500/38977567708509069650478659519615552179-to-formula-rate-model-2021.xlsx.json", 'r') as f:
    #     workbook_json2 = json.load(f)
    # generate_demo_features("163350182546811267831688234176624693724-to20-ry2022-attachment-d.xlsx", "13-WorkCap", workbook_json1, 12, 1, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # generate_demo_features("38977567708509069650478659519615552179-to-formula-rate-model-2021.xlsx", "13-WorkCap", workbook_json2, 12, 1, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # with open(root_path + 'demo_tile_features_fix/' + "163350182546811267831688234176624693724-to20-ry2022-attachment-d.xlsx---13-WorkCap---12---1.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "38977567708509069650478659519615552179-to-formula-rate-model-2021.xlsx---13-WorkCap---12---1.json", 'r') as f:
    #     res2 = json.load(f)  

    # with open("../Demo/fix_fortune500/108138953513719859234385013398650948303-obf-lighting-workbook-032022.xlsx.json", 'r') as f:
    #     workbook_json1 = json.load(f)
    # with open("../Demo/fix_fortune500/318833306171241075689682530186143716076-obf-tier1a-workbook-032022.xlsx.json", 'r') as f:
    #     workbook_json2 = json.load(f)
    # generate_demo_features("108138953513719859234385013398650948303-obf-lighting-workbook-032022.xlsx", "1. START", workbook_json1, 20, 1, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # generate_demo_features("318833306171241075689682530186143716076-obf-tier1a-workbook-032022.xlsx", "1. START", workbook_json2, 23, 1, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # with open(root_path + 'demo_tile_features_fix/' + "108138953513719859234385013398650948303-obf-lighting-workbook-032022.xlsx---1. START---20---1.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "318833306171241075689682530186143716076-obf-tier1a-workbook-032022.xlsx---1. START---23---1.json", 'r') as f:
    #     res2 = json.load(f)

    # with open("../Demo/fix_fortune500/130080049238776506264743525541948806006-dpr%20form-farc-0066462-fa-0429212-cpn%2374-119583-01.xlsx.json", 'r') as f:
    #     workbook_json1 = json.load(f)
    # with open("../Demo/fix_fortune500/330956518078954468761521805141433307945-device_problem_report_rev4.xlsx.json", 'r') as f:
    #     workbook_json2 = json.load(f)
    # generate_demo_features("130080049238776506264743525541948806006-dpr%20form-farc-0066462-fa-0429212-cpn%2374-119583-01.xlsx", "FPGA-Intel", workbook_json1, 29, 18, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # generate_demo_features("330956518078954468761521805141433307945-device_problem_report_rev4.xlsx", "FPGA-Intel", workbook_json2, 29, 18, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # with open(root_path + 'demo_tile_features_fix/' + "130080049238776506264743525541948806006-dpr%20form-farc-0066462-fa-0429212-cpn%2374-119583-01.xlsx---FPGA-Intel---29---18.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "330956518078954468761521805141433307945-device_problem_report_rev4.xlsx---FPGA-Intel---29---18.json", 'r') as f:
    #     res2 = json.load(f)
    
    # with open(root_path + 'demo_tile_features_fix/' + "11900081385525998988782552947219454956-tps2352x_5f00_excel_5f00_tool_5f00_web_5f00_revbexp2.xlsx---Error_Calculation---8---4.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "173427596559219994540832088453684885564-tps2352x_5f00_excel_5f00_tool_5f00_web_5f00_revb_5f00_20191202.xlsx---Error_Calculation---8---4.json", 'r') as f:
    #     res2 = json.load(f)

    # with open(root_path + 'demo_sheet_features/' + "110059629631842288643883464290182071710-table-of-contents.xlsx---2017 Sales---1---1.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_sheet_features/' + "328621492309825452137276453653810262580-punch-clock-sample.xlsx---MyTimeSheet---1---1.json", 'r') as f:
    #     res2 = json.load(f)
    # with open(root_path + 'demo_tile_features_fix/' + "11900081385525998988782552947219454956-tps2352x_5f00_excel_5f00_tool_5f00_web_5f00_revbexp2.xlsx---SOA_Checks---5---8.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "150763181938560220412217118135597138020-tps23525_5f00_excel_5f00_tool.xlsx---SOA_Checks---5---8.json", 'r') as f:
    #     res2 = json.load(f)

    # with open(root_path + 'demo_tile_features_fix/' + "148772266108869536476680766951864409419-pg%26e-monthly-srac-20180309.xlsx---SRAC---52---11.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "227705918389663749800296411556819732979-20170209srac.xlsx---SRAC---52---11.json", 'r') as f:
    #     res2 = json.load(f)

    # with open(root_path + 'demo_tile_features_fix/' + "100206920085844727873527872071004985725-3288.flybuck-flyback-calculator.xlsx---Fly-Buck_Calc---4---11.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "100206920085844727873527872071004985725-3288.flybuck-flyback-calculator.xlsx---Flyback_DCM_Calc---4---11.json", 'r') as f:
    #     res2 = json.load(f)

    # with open(root_path + 'demo_tile_features_fix/' + "127534955648942888021337347766059868298-to20-ry2022-attachment-b.xlsx---15-NUC---8---1.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "51062703087863555942928902850393994897-to20-model_gs_ry2022_grossloadry2021.xlsx---15-NUC---8---1.json", 'r') as f:
    #     res2 = json.load(f)

    # with open(root_path + 'demo_tile_features_fix/' + "130080049238776506264743525541948806006-dpr%20form-farc-0066462-fa-0429212-cpn%2374-119583-01.xlsx---FPGA-Intel---29---18.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "330956518078954468761521805141433307945-device_problem_report_rev4.xlsx---FPGA-Intel---29---18.json", 'r') as f:
    #     res2 = json.load(f)

    # with open(root_path + 'demo_tile_features_fix/' + "139240084001401108359407090258780823407-bq40z50-family-thermistor-coef-calculator.xlsx---What If---4---13.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "280844045619136162074491477750463726924-bq76952-family-thermistor-coef-calculator.xlsx---What If---4---13.json", 'r') as f:
    #     res2 = json.load(f)

    # with open(root_path + 'demo_tile_features_fix/' + "162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx---DDR3---13---3.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx---EVM_1_3 (DDR3)---13---3.json", 'r') as f:
    #     res2 = json.load(f)

    # with open("../Demo/fix_fortune500/195680212009286979128475846136495926436-lm_2800_2_2900_5183-psr-flyback-converter-quickstart-tool-_2d002d00_-24vin-18v-dual-0.3aout.xlsx.json", 'r') as f:
    #     workbook_json1 = json.load(f)
    # with open("../Demo/fix_fortune500/337139341872094914954708004034747505888-lm_2800_2_2900_518x-psr-flyback-converter-quickstart-tool.xlsx.json", 'r') as f:
    #     workbook_json2 = json.load(f)
    # generate_demo_features("195680212009286979128475846136495926436-lm_2800_2_2900_5183-psr-flyback-converter-quickstart-tool-_2d002d00_-24vin-18v-dual-0.3aout.xlsx", "Calculations - Dual", workbook_json1, 110, 14, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # generate_demo_features("337139341872094914954708004034747505888-lm_2800_2_2900_518x-psr-flyback-converter-quickstart-tool.xlsx", "Calculations - Dual", workbook_json2, 110, 14, save_path=root_path + 'demo_tile_features_fix/', is_look=True, cross=False)
    # with open(root_path + 'demo_tile_features_fix/' + "195680212009286979128475846136495926436-lm_2800_2_2900_5183-psr-flyback-converter-quickstart-tool-_2d002d00_-24vin-18v-dual-0.3aout.xlsx---Calculations - Dual---110---14.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "337139341872094914954708004034747505888-lm_2800_2_2900_518x-psr-flyback-converter-quickstart-tool.xlsx---Calculations - Dual---110---14.json", 'r') as f:
    #     res2 = json.load(f)

    # with open(root_path + 'demo_tile_features_fix/' + "259790987351705571734860328226745516825-50a_2d00_lm5069_5f00_design_5f00_calculator_5f00_rev_5f00_c.xlsx---Start_up---10---9.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "256214073906650542186400731664449664759-lm5066-design_5f00_calculator_5f00_rev_5f00_b-_2d00_-filled.xlsx---Start_up---10---9.json", 'r') as f:
    #     res2 = json.load(f)
    # with open(root_path + 'demo_tile_features_fix/' + "266528405654445681529278389558779510118-lm5069_5f00_back_5f00_to_5f00_back.xlsx---dv_dt_recommendations---58---24.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "280789179676994360357294026746642648383-lm5066i_5f00_design_5f00_calculator_5f00_rev_5f00_c.xlsx---dv_dt_recommendations---58---24.json", 'r') as f:
    #     res2 = json.load(f)

    # with open(root_path + 'demo_tile_features_fix/' + "282801639974372749482363029255337975320-aot-openshift-capacity-planning-analysis_v1.3.xlsx---S1 - Schematics---5---6.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "282801639974372749482363029255337975320-aot-openshift-capacity-planning-analysis_v1.3.xlsx---S3 - CP4I (ACE+MQ+APIC)---5---6.json", 'r') as f:
    #     res2 = json.load(f)

    # with open(root_path + 'demo_tile_features_fix/' + "289777234214497513427023181872086299118-tps65lux_5f00_sequencer_5f00_12-gck-with-2-gsp_2c00_-ggp1_2c00_-ggp2_2c00_-gcp.xlsx---Customer Input---13---5.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "289777234214497513427023181872086299118-tps65lux_5f00_sequencer_5f00_12-gck-with-2-gsp_2c00_-ggp1_2c00_-ggp2_2c00_-gcp.xlsx---Template---13---5.json", 'r') as f:
    #     res2 = json.load(f)

    # with open(root_path + 'demo_tile_features_fix/' + "305083579727401949921996547691592868534-calculator_5f00_rev1p03.xlsx---Parts---13---61.json", 'r') as f:
    #     res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    # with open(root_path + 'demo_tile_features_fix/' + "71830531429966026667023738346702335760-lm5160_2d00_ver1.xlsx---Parts---13---61.json", 'r') as f:
    #     res2 = json.load(f)

    root_path = '/datadrive-2/data/fuste_test/'    
    with open(root_path + 'sheets_json_feaures/' + "753fc9f3-0214-43b5-a9cb-2133e4719ac0.xlsx---Metadata - Indicators.json", 'r') as f:
        res1 = json.load(f) # fixed_workbook_json origin_fortune500_workbook_json
    with open(root_path + 'sheets_json_feaures/' + "ef77d010-d1cf-4da0-b240-384b5d794ab5.xlsx---Sheet1.json", 'r') as f:
        res2 = json.load(f)
    # all_same = True
    for index in range(0, len(res1)):
        # print(res1[index])
        for keyw in res1[index]:
            if keyw == 'row' or keyw == 'col':
                continue
            if type(res2[index]).__name__ == 'list':
                res2[index] = res2[index][0]
            if type(res1[index]).__name__ == 'list':
                res1[index] = res1[index][0]
            # print('keyw', keyw)
        # print(res1[index].keys())
        for keyw in res1[index]:
            
            # print('type(res2[index]).__name__', type(res2[index]).__name__)
            all_same = False
            if res1[index][keyw] != res2[index][keyw] and keyw != 'row' and keyw != 'col' and keyw != 'width' :
                
                print('#########')
                print(index)
                print(keyw)
                print(res1[index])
                print(res2[index])
                print(res1[index][keyw])
                print(res2[index][keyw])
    # print('all_same', all_same)
    # with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
    #     bert_dict = json.load(f)
    # with open("json_data/content_temp_dict_1.json", 'r') as f:
    #     content_tem_dict = json.load(f)
    # generate_one_before_feaure("11900081385525998988782552947219454956-tps2352x_5f00_excel_5f00_tool_5f00_web_5f00_revbexp2.xlsx---Error_Calculation---8---4", bert_dict, content_tem_dict, mask=2, source_root_path = root_path + 'demo_tile_features_fix/', saved_root_path=root_path + 'demo_before_features_mask2_fix/')
    # generate_one_before_feaure("173427596559219994540832088453684885564-tps2352x_5f00_excel_5f00_tool_5f00_web_5f00_revb_5f00_20191202.xlsx---Error_Calculation---8---4", bert_dict, content_tem_dict, mask=2, source_root_path = root_path + 'demo_tile_features_fix/', saved_root_path=root_path + 'demo_before_features_mask2_fix/')
    # res1 = np.load(root_path + 'demo_before_features_mask2_fix/' + "11900081385525998988782552947219454956-tps2352x_5f00_excel_5f00_tool_5f00_web_5f00_revbexp2.xlsx---Error_Calculation---8---4.npy", allow_pickle=True)
    # res2 = np.load(root_path + 'demo_before_features_mask2_fix/' + "173427596559219994540832088453684885564-tps2352x_5f00_excel_5f00_tool_5f00_web_5f00_revb_5f00_20191202.xlsx---Error_Calculation---8---4.npy", allow_pickle=True)
    # index = 0
    # start_row = 8-50
    # start_col = 4-5
    # print(res1.shape)
    # for row in range(0,100):
    #     for col in range(0,10):
    #         for featureid in range(0,399):
    #             if res1[index][featureid] != res2[index][featureid] :
    #                 print(start_row + row, start_col + col, featureid, res1[index][featureid], res2[index][featureid])
    #         index += 1
    # print((res1 == res2).all())
    # print((res1 == res3).all())

def look_accuracy():
    for model_id in range(1,31):
        with open("/datadrive-3/data/fortune500_test/validate_cross/" + str(model_id) + '/accuracy.json', 'r') as f:
            res = json.load(f)
            print(res)
def check_ctime():
    filelist = os.listdir(root_path + 'demo_tile_features_fix/')
    need_rerun = []
    for filename in filelist:
        t = os.path.getmtime(root_path + 'demo_tile_features_fix/' + filename)
        timeStruce = time.localtime(t)
        # print('timeStruce', timeStruce.tm_year)
        times = time.strftime("%Y-%m-%d", timeStruce)
        print('timeStruce',times)
        if times == '2023-01-24':
            need_rerun.append(filename)
    #     print(filename, times)
    with open("need_rerun.json", 'w') as f:
        json.dump(need_rerun, f)

def rerun_demo_features(thread_id, batch_num):
    with open('fortune500_val_formula.json', 'r') as f:
        fortune500_val_formula = json.load(f)
    with open('fortune500_test_formula.json', 'r') as f:
        fortune500_test_formula = json.load(f)
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)
    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)

    with open("need_rerun.json", 'r') as f:
        need_rerun = json.load(f)
        need_rerun = [item.split('---')[0] for item in need_rerun]
    print(need_rerun)
    print(len(set(need_rerun)))
    # formula_tokens = os.listdir(root_path + 'demo_tile_features_fix/')
    formula_tokens = fortune500_test_formula + fortune500_val_formula
    filename2token = {}
    for formula_token in formula_tokens:
        formula_token = formula_token.replace('.json', '')
        filename = formula_token.split('---')[0]
        if filename not in need_rerun:
            continue
        if filename not in filename2token:
            filename2token[filename] = []
        filename2token[filename].append(formula_token)

    batch_len = int(len(filename2token)/batch_num)
    print(filename2token.keys())
    for index, filename in enumerate(filename2token):
        if thread_id != batch_num:
            if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1 ):
                continue
        print(index, len(filename2token))
        with open("../Demo/fix_fortune500/" + filename + '.json', 'r') as f:
            workbook_json = json.load(f)
        for formula_token in filename2token[filename]:
            sheetname = formula_token.split('---')[1]
            row = int(formula_token.split('---')[2])
            col = int(formula_token.split('---')[3])
            # generate_demo_features(filename, sheetname, workbook_json, row, col, root_path + 'demo_tile_features_fix/', is_look=True)
            generate_one_before_feaure(formula_token, bert_dict, content_tem_dict, mask=2, source_root_path = root_path + 'demo_tile_features_fix/', saved_root_path=root_path + 'demo_before_features_fix/')
            res = before2cross(formula_token, bert_dict, content_tem_dict, mask=2, source_root_path = root_path + 'demo_before_features_fix/', saved_root_path=root_path + 'cross_before_features/')

def generate_all_demo_features():
    filelist = os.listdir("/datadrive/projects/Demo/fix_fortune500/")
    with open("sensity_workbooks.json", 'r') as f:
        filelist1 = list(json.load(f).keys())
    filelist1 = [item + '.json' for item in filelist1]
    filelist = filelist1 + filelist
    for index, filename in enumerate(filelist):
        try:
            print(index, len(filelist))
            filename = filename.replace(".json", '')
            with open("/datadrive/projects/Demo/fix_fortune500/"+filename + '.json', 'r') as f:
                workbook_json = json.load(f)
            sheetnames = [item['Name'] for item in workbook_json['Sheets']]
            for sheetname in sheetnames:
                if os.path.exists(root_path + 'demo_sheet_features/' + filename + '---' + sheetname + "---1---1.json"):
                    continue
                generate_demo_features(filename = filename, sheetname = sheetname, workbook_json = workbook_json, origin_row = 1, origin_col = 1, save_path = root_path + 'demo_sheet_features/', is_look=True, cross=False)
        except:
            continue
def generate_all_before():
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)
    
    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)

    filelist = os.listdir(root_path + 'demo_sheet_features/')
    for index, filename in enumerate(filelist):
        print(index, len(filelist))
        formula_token = filename.replace('.json', '')
        if os.path.exists(root_path + 'sheet_before_features/' + formula_token + '.npy'):
            continue
        generate_one_before_feaure(formula_token, bert_dict = bert_dict, content_tem_dict = content_tem_dict, mask=2, source_root_path = root_path + 'demo_sheet_features/', saved_root_path=root_path + 'sheet_before_features/')

def generate_all_after():
    # model = torch.load("/datadrive-2/data/finegrained_model_16/epoch_31")
    model = torch.load("196model/cnn_new_dynamic_triplet_margin_1_3_12")
    root_path = '/datadrive-2/data/deco_test/'
    # root_path = '/datadrive-2/data/fuste_test/'
    # filelist = os.listdir(root_path + 'sheet_before_features/')
    filelist = os.listdir(root_path + 'sheets_before_features/')
    for index, filename in enumerate(filelist):
        print(index, len(filelist))
        formula_token = filename.replace('.npy', '')
        if os.path.exists(root_path + 'sheet_after_features/' + formula_token + '.npy'):
            continue
        generate_one_after_feature(formula_token, model, before_path=root_path + 'sheets_before_features/' + formula_token + '.npy', after_path=root_path + 'sheet_after_features/' + formula_token + '.npy')

def similarity_test_sheets():
    with open("fortune500_company2workbook.json", 'r') as f:
        company2workbook = json.load(f)
    with open('../AnalyzeDV/fortune500_filename2sheetname.json', 'r') as f:
        filename2sheetname = json.load(f)
    test = []
    test_res = {}
    company_filesheets = {}
    for company in company2workbook:
        sheets = []
        company_workbooks = company2workbook[company]
        if company not in ['ibm','cisco','pge','microsoft','ti']:
            continue
        # print('company_workbooks', company_workbooks)
        for workbook in company_workbooks:
            if workbook not in filename2sheetname:
                continue
            for sheet in filename2sheetname[workbook]:
                sheets.append(workbook + '---' + sheet)
        company_filesheets[company] = sheets
        test_res[company] = random.sample(sheets, min(5000, len(sheets)))
    with open('fortune500_similar_testsheets.json', 'w') as f:
        json.dump(test_res, f)
    with open('fortune500_company2filesheets.json', 'w') as f:
        json.dump(company_filesheets, f)

def similarity_test_distance():
    # with open("fortune500_similar_testsheets.json", 'r') as f:
    #     similar_testsheets = json.load(f)
    # with open("fortune500_company2filesheets.json", 'r') as f:
    #     company_filesheets = json.load(f)
    
    # for company_name in similar_testsheets:
    #     test_sheets = similar_testsheets[company_name]
    #     filesheets_list = company_filesheets[company_name]
    #     filesheet_features = []
    #     filesheet_ids = []
    #     res = {}
    #     for filesheet in filesheets_list:
    #         if not os.path.exists(root_path + 'sheet_after_features/' + filesheet + '---1---1.npy'):
    #             continue
    #         second_feature = np.load(root_path + 'sheet_after_features/' + filesheet + '---1---1.npy', allow_pickle=True)
    #         filesheet_features.append(second_feature)
    #         filesheet_ids.append(filesheet)
    #     filesheet_features = np.array(filesheet_features).reshape(-1, 384)
    #     filesheet_ids1 =  np.array(list(range(len(filesheet_ids))))
    #     faiss_index = faiss.IndexFlatL2(len(filesheet_features[0]))
    #     faiss_index2 = faiss.IndexIDMap(faiss_index)
    #     faiss_index2.add_with_ids(filesheet_features, filesheet_ids1)
    #     for index, filesheet in enumerate(test_sheets):
    #         print(index, len(test_sheets))
    #         if filesheet not in filesheet_ids:
    #             continue
    #         feature = filesheet_features[filesheet_ids.index(filesheet)].reshape((1,384))
    #         search_list = np.array(feature)
    #         D, I = faiss_index.search(np.array(search_list), len(filesheet_features)) # sanity check
    #         top_k = []
    #         for ind,i in enumerate(I[0]):
    #             top_k.append((filesheet_ids[i], float(D[0][ind])))
    #             # print((filesheet_ids[i], float(D[0][ind])))
    #         res[filesheet] = top_k
    #     np.save("fortune500_simtest_"+company_name+"_distance_dict.npy", res)
    root_path = '/datadrive-2/data/deco_test/'
    # root_path = '/datadrive-2/data/fuste_test/'
    filesheet_features = []
    filesheet_ids = []
    res = {}
    similar_testsheets = os.listdir("/datadrive-2/data/deco_test/sheet_after_features/")
    # similar_testsheets = os.listdir("/datadrive-2/data/fuste_test/sheet_after_features/")
    for filesheet in similar_testsheets:
        filesheet = filesheet.replace('.npy', '')
        if not os.path.exists(root_path + 'sheet_after_features/' + filesheet + '.npy'):
            continue
        second_feature = np.load(root_path + 'sheet_after_features/' + filesheet + '.npy', allow_pickle=True)
        # print('second_feature', second_feature.shape)
        filesheet_features.append(second_feature)
        filesheet_ids.append(filesheet)
    # filesheet_features = np.array(filesheet_features).reshape(-1, 384)
    filesheet_features = np.array(filesheet_features).reshape(-1, 896)
    filesheet_ids1 =  np.array(list(range(len(filesheet_ids))))
    faiss_index = faiss.IndexFlatL2(len(filesheet_features[0]))
    faiss_index2 = faiss.IndexIDMap(faiss_index)
    faiss_index2.add_with_ids(filesheet_features, filesheet_ids1)
    # for index, filesheet in enumerate(test_sheets):
    #     print(index, len(test_sheets))
    for index, filesheet in enumerate(similar_testsheets):
        filesheet = filesheet.replace('.npy', '')
        print(index, len(similar_testsheets))
        if filesheet not in filesheet_ids:
            continue
        feature = filesheet_features[filesheet_ids.index(filesheet)].reshape((1,896))
        search_list = np.array(feature)
        D, I = faiss_index.search(np.array(search_list), len(filesheet_features)) # sanity check
        top_k = []
        for ind,i in enumerate(I[0]):
            top_k.append((filesheet_ids[i], float(D[0][ind])))
            # print((filesheet_ids[i], float(D[0][ind])))
        res[filesheet] = top_k

    np.save("deco_simtest_distance_dict.npy", res)
    # np.save("fuste_simtest_distance_dict.npy", res)

def clustering(dataset, threshold):
    root_path = '/datadrive-2/data/' + dataset + '_test/'
    res = np.load(dataset + "_simtest_distance_dict.npy", allow_pickle=True).item()
    first_key = list(res.keys())[0]
    # print('res', first_key)
    # print('res', res[first_key][0:400])

    if dataset == 'fuste':
        filesheets = os.listdir('/datadrive/projects/Mondrian-master/res/fuste/csv')
    if dataset == 'deco':
        filesheets = os.listdir('/datadrive/projects/Mondrian-master/res/deco/csv')
    filesheets = [item[0:-4] for item in filesheets]
    ground_truth_path = '../Mondrian-master/' + dataset + '_groundtruth.json'
    with open(ground_truth_path, 'r') as f:
        gt_json = json.load(f)
    new_filesheets = list(gt_json['filesheet2clusterid'].keys())
    # print('filesheets', filesheets)
    # new_filesheets = []
    # for filesheet in filesheets:
    #     if '.xls_' in filesheet:
    #         splited_list = filesheet.split('.xls_')
    #         sheetname = splited_list[-1].replace('.xlsx', '')
    #         filename = splited_list[0]+'.xls'
    #     else:
    #         splited_list = filesheet.split('.xlsx_')
    #         sheetname = splited_list[-1]
    #         filename = splited_list[0]+'.xlsx'
    #     new_filesheets.append(filename + '---' + sheetname)

    new_res = {}
    # print('len res', len(res))
    # print('new_filesheets', len(new_filesheets))
    print(len(set(res.keys()) & set(new_filesheets)))
    for filesheet in res:
        if filesheet not in new_filesheets:
            continue
        new_item = []
        for item in res[filesheet]:
            if item[0] not in new_filesheets:
                continue
            new_item.append(item)
        new_res[filesheet] = new_item

    res = new_res
    np.save(dataset + '_res_distance.npy', new_res)
    cluster = {}
    clustered = {}
    max_id = 0

    added = []
    nodes = new_filesheets
    pairs = []
    for index, filesheet in enumerate(res):
        for item in res[filesheet]:
            if item[1] <= threshold:
                if filesheet + '---' + item[0] in added:
                    continue
                added.append(item[0] + '---' + filesheet)
                added.append(filesheet + '---' + item[0])
                pairs.append([filesheet, item[0]])
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(pairs)
    predicted_templates = nx.connected_components(G)
    predicted_templates = {"template_" + str(idx): list(t) for idx, t in enumerate(predicted_templates)}
    # print('predicted_templates', predicted_templates)
    # # print('lenres', len(res))
    # for index, filesheet in enumerate(res):
    #     # print(index, len(res))
    #     if filesheet in clustered:
    #         continue
    #     similar_sheets = []
    #     for item in res[filesheet]:
    #         if item[1] <= threshold:
    #             similar_sheets.append(item[0])

    #     is_clustered = False
    #     for similar_sheet in similar_sheets:
    #         if filesheet in clustered:
    #             is_clustered = True
    #             clustered_id = clustered[filesheet]
    #             break

    #     if not is_clustered:
    #         max_id += 1
    #         clustered_id = max_id
    #         cluster[clustered_id] = [filesheet]
    #         clustered[filesheet] = clustered_id
            
    #     for simialr_sheet in similar_sheets:
    #         if simialr_sheet not in cluster[clustered_id]:
    #             cluster[clustered_id].append(simialr_sheet)
    #             clustered[similar_sheet] = clustered_id

    # new_clusters = {}
    # index = 0
    # while index < len(list(cluster.keys())):
    #     one_cls = list(cluster.keys())[index]
    #     index1 = 0
    #     print('index', index)
    #     while index1 < len(list(cluster.keys())):
    #         other_one_cls = list(cluster.keys())[index1]
    #         # print('    index1', index1)
    #         if one_cls == other_one_cls:
    #             index1 += 1
    #             continue
    #         if len(set(cluster[one_cls]) & set(cluster[other_one_cls])) != 0:
    #             cluster[one_cls] = list(set(cluster[one_cls]) | set(cluster[other_one_cls]))
    #             del cluster[other_one_cls]
    #             print('    del', other_one_cls)
    #         else:
    #             index1 += 1
    #     index += 1

    # max_id = 1
    # new_clustered = {}
    # for key in cluster:
    #     new_clusters[max_id] = cluster[key]
    #     for item in cluster[key]:
    #         new_clustered[item] = max_id
    #     max_id += 1
    max_id = 1
    new_clustered = {}
    for key in predicted_templates:
        for item in predicted_templates[key]:
            new_clustered[item] = max_id
        max_id += 1
    # # print('cluster', cluster)
    # # print('clustered', clustered)    
    # with open(root_path + "clusters/" + str(threshold) + '.json','w') as f:
    #     json.dump({'clusters': new_clusters, 'filesheet2clusterid': new_clustered}, f)
    with open(root_path + "clusters/" + str(threshold) + '.json','w') as f:
        json.dump({'clusters': predicted_templates, 'filesheet2clusterid': new_clustered}, f)


def pair_distance():
    sheet_list = os.listdir(root_path + 'sheet_after_features/')
    filesheet2feature = {}
    res = {}

    with open('sensity_workbooks.json', 'r') as f:
        sensity_workbooks = json.load(f)
    sensity_list = []
    for wb in sensity_workbooks:
        for sheet in sensity_workbooks[wb]:
            sensity_list.append(wb + '---' + sheet + '---1---1')
    # for formula_token_npy in sheet_list:
    #     print('formula_token_npy', formula_token_npy)
    #     filesheet = formula_token_npy.split('---')[0] + '---' + formula_token_npy.split('---')[1]
    #     feature = np.load(root_path + 'sheet_after_features/' + formula_token_npy, allow_pickle=True)
    #     filesheet2feature[filesheet] = feature
    # np.save("filesheet2feature.npy", filesheet2feature)
    filesheet2feature = np.load("filesheet2feature.npy", allow_pickle=True).item()
    filesheet_ids = list(filesheet2feature.keys())
    print('filesheet_ids',filesheet_ids[0:1])
    filesheet_features = list(filesheet2feature.values())
    print('filesheet_features',filesheet_features[0:1])
    sheet_list = sensity_list + sheet_list

    filesheet_features = np.array(filesheet_features).reshape((69746, 896))
    filesheet_ids1 =  np.array(list(range(len(filesheet_ids))))
    print(filesheet_ids1)
    faiss_index = faiss.IndexFlatL2(len(filesheet_features[0]))
    faiss_index2 = faiss.IndexIDMap(faiss_index)
    
    faiss_index2.add_with_ids(filesheet_features, filesheet_ids1)
    print('filesheet_features', filesheet_features.shape)
    print('filesheet_ids1', filesheet_ids1.shape)
    for index, formula_token_npy in enumerate(sheet_list):
        print(index, len(sheet_list))
        filesheet = formula_token_npy.split('---')[0] + '---' + formula_token_npy.split('---')[1]
        feature = filesheet_features[filesheet_ids.index(filesheet)].reshape((1,896))
        search_list = np.array(feature)
        D, I = faiss_index.search(np.array(search_list), len(filesheet_features)) # sanity check
        top_k = []
        for ind,i in enumerate(I[0]):
            top_k.append((filesheet_ids[i], float(D[0][ind])))
            # print((filesheet_ids[i], float(D[0][ind])))
        res[filesheet] = top_k
    np.save("distance_dict.npy", res)
def sample_efficiency():
    with open("fortune500_test_formula.json", 'r') as f:
        res = json.load(f)
    res = random.sample(res, 50)
    with open("sample_efficiency.json", 'w') as f:
        json.dump(res, f)
    
def look_distance_dict():
    with open('sensity_workbooks.json', 'r') as f:
        sensity_workbooks = json.load(f)
    sensity_list = []
    for wb in sensity_workbooks:
        for sheet in sensity_workbooks[wb]:
            sensity_list.append(wb + '---' + sheet)
    res = np.load("distance_dict.npy", allow_pickle=True).item()
    print('sensity_list[0]', sensity_list[15])
    print(res[sensity_list[15]])
    print('248141262801860332898172356095461823691-table-of-contents.xlsx---2018 Sales' in sensity_list)
    with open('fortune500_similar_testsheets.json', 'r') as f:
        sensity_workbooks = json.load(f)
    sensity_list = []
    for company in sensity_workbooks:
        for filesheet in sensity_workbooks[company]:
            sensity_list.append(filesheet)
    res = np.load("fortune500_simtest_"+company+"_distance_dict.npy", allow_pickle=True).item()
    # print('sensity_list[0]', sensity_list[0])
    # print(res.keys())
    for filesheet in sensity_list:
        if filesheet in res:
            print('filesheet', filesheet)
            print(res[filesheet])
            break
    # print('248141262801860332898172356095461823691-table-of-contents.xlsx---2018 Sales' in sensity_list)
    # res = np.load('fuste_res_distance.npy', allow_pickle=True).item()
    # print(res)

def is_in_same_cluster(filename1, filename2, threshold):
    with open('../AnalyzeDV/fortune500_filename2sheetname.json', 'r') as f:
        fortune500_filename2sheetname = json.load(f)
    all_sheet = 0
    sheet2num = {}
    y_pred = []
    for filename in fortune500_filename2sheetname:
        for sheet in fortune500_filename2sheetname[filename]:
            if sheet not in sheet2num:
                sheet2num[sheet] = 0
            sheet2num[sheet] += 1
            all_sheet += 1
    if filename1 not in fortune500_filename2sheetname or filename2 not in fortune500_filename2sheetname:
        return False
    if len(fortune500_filename2sheetname[filename1]) != len(fortune500_filename2sheetname[filename2]):
        return False
    
    for sheet in fortune500_filename2sheetname[filename1]:
        if sheet not in fortune500_filename2sheetname[filename2]:
            return False
    # print(thrd, filename1, filename2)
    prob = 1
    for sheet in fortune500_filename2sheetname[filename1]:
        prob *= (sheet2num[sheet] / all_sheet)
    # print('prob', prob)
    # print(prob < threshold)
    if prob < threshold:
        return True
    else:
        return False

def generate_sensitivity_ground_truth():

    # print(res.keys())
    
    distance_dict = np.load('distance_dict.npy', allow_pickle=True).item()
    with open("sensity_workbooks.json", 'r') as f:
        workbooks = list(json.load(f).keys())
    #     auchor_list = random.sample(workbooks, k=int(len(workbooks)/3))
    # with open("sensity_auchor_workbooks.json", 'w') as f:
    #     json.dump(auchor_list, f)
    with open("sensity_auchor_workbooks.json", 'r') as f:
        auchor_list = json.load(f)
    # auchor_list = list(set(auchor_list) - set(workbooks))
    auchor_list = [item for item in auchor_list if item in workbooks]
    # with open("sensity_auchor_workbooks.json", 'w') as f:
    #     json.dump(auchor_list, f)

    workbook2filesheet = {}

    for filesheet in distance_dict:
        workbook_name = filesheet.split('---')[0]
        sheetname = filesheet.split('---')[1]
        if workbook_name not in workbook2filesheet:
            workbook2filesheet[workbook_name] = []
        workbook2filesheet[workbook_name].append(filesheet)

    test_workbook_list = list(set(workbook2filesheet.keys()) - set(auchor_list))
    with open("sensity_test_workbooks.json", 'w') as f:
        json.dump(test_workbook_list, f)
    with open("sensity_test_workbooks.json", 'r') as f:
        test_workbook_list = json.load(f)
    def check_workbooks(workbook1, workbook2, threshold):
        filesheets1 = workbook2filesheet[workbook1]
        filesheets2 = workbook2filesheet[workbook2]
        if len(filesheets1) != len(filesheets2):
            return False

        filesheet_list_1 = []
        score_list_1 =  []
        filesheet_list_2 = []
        score_list_2 =  []
        
        paired_filesheets = []
        for filesheet in filesheets1:
            similar_filesheet_pairs = distance_dict[filesheet]
            found = False
            for similar_filesheet_pair in similar_filesheet_pairs:
                similar_filesheet, score = similar_filesheet_pair
                if similar_filesheet not in filesheets2 or similar_filesheet in paired_filesheets:
                    continue
                if score < threshold:
                    found = True
                    paired_filesheets.append(similar_filesheet)
                    break
            if found == False:
                return False
        return True

    def baseline_pr(threshold):
        y_pred = []
        for index, test_workbook in enumerate(test_workbook_list):
            print(threshold, index, len(test_workbook_list))
            is_sensity = False
            for auchor_wokrbook in auchor_list:
                is_sensity = is_in_same_cluster(test_workbook, auchor_wokrbook, threshold)
                if is_sensity:
                    break
                # print('is_sensity', is_sensity)

            if is_sensity:
                y_pred.append(True)
            else:
                y_pred.append(False)
        return y_pred

    def precision_and_recall(threshold):
        y_pred = []
        y_test = []
        
        with open('../AnalyzeDV/fortune500_filename2sheetname.json', 'r') as f:
            fortune500_filename2sheetname = json.load(f)
        for test_workbook in test_workbook_list:
            is_sensity = False
            found_auchor = ''
            for auchor_wokrbook in auchor_list:
                if check_workbooks(test_workbook, auchor_wokrbook, threshold):
                    is_sensity = True
                    found_auchor = auchor_wokrbook
                    break
                    
            if is_sensity:
                y_pred.append(True)
            else:
                y_pred.append(False)
            if test_workbook in workbooks:
                y_test.append(True)
                
            else:
                y_test.append(False)
                if is_sensity and 'add-images' not in found_auchor: #
                    print("##############", threshold)
                    print('test_workbook', test_workbook)
                    print(fortune500_filename2sheetname[test_workbook])
                    print(test_workbook in workbooks)
                    print('auchor_wokrbook', found_auchor)
                    print(fortune500_filename2sheetname[found_auchor])
                    print(found_auchor in workbooks)
        # print(len(workbooks))
        return y_pred, y_test

    thrd_list =list(range(100))
    thrd_list += [0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    thrd_list += [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.0001, 0.00001,0.000001,0.0000001, 0.00000001, 0.000000001]
    
    # thrd_list = [18]
    for thrd in thrd_list:
        res = precision_and_recall(thrd)
        with open('/datadrive-2/data/fortune500_test/sensitivity_res/' + str(thrd) + '.json', 'w') as f:
            json.dump(res, f)

    # for thrd in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 1]:
    for thrd in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.5, 0.8, 1]:
        base_res = baseline_pr(thrd)
        print('thrd', thrd)
        with open('/datadrive-2/data/fortune500_test/sensitivity_semi_baseline/'+str(thrd) + '.json', 'w') as f:
            json.dump(base_res, f)
        
def sensitivity_evaluation():
    # ours_recall = []
    # ours_precision = []
    thrd_list =list(range(1, 100))
    thrd_list += [0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    thrd_list += [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.0001, 0.00001,0.000001,0.0000001, 0.00000001, 0.000000001]
    # thrd_list =list(range(100))
    # thrd_list += [0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.0001, 0.00001,0.000001,0.0000001, 0.00000001, 0.000000001]
    # thrd_list = [0.000000001]
    ours_recall_precision = {}
    for thrd in thrd_list:
        with open('/datadrive-2/data/fortune500_test/sensitivity_res/' + str(thrd) + '.json', 'r') as f:
            res = json.load(f)
        y_pred, y_test = res
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for index, item in enumerate(y_pred):
            # print('y_pred[index] == y_test[index]', y_pred[index], y_test[index], y_pred[index] == y_test[index])
            if y_pred[index] == y_test[index]:
                if y_test[index] == True:
                    tp += 1
                else:
                    tn += 1
            else:
                if y_test[index] == True:
                    fn += 1
                else:
                    fp += 1
        print('tp + fn', tp + fn)
        print('tp + tn', tp + tn)
        print('all', tp + fp + fn + tn)
        print('tp, tn , fp, fn', tp, tn, fp, fn)
        if tp + fp == 0:
            precision = 0
        else:
            precision = (tp / (tp + fp))

        if tp + fn == 0:
            recall = 0
        else:
            recall = (tp / (tp + fn))

        # ours_recall.append(recall)
        if recall not in ours_recall_precision:
            ours_recall_precision[recall] = precision
        else:
            if precision > ours_recall_precision[recall]:
                ours_recall_precision[recall] = precision
        # ours_precision.append(precision)
        print('ours:', thrd, precision, recall)

    baseline_recall = []
    baseline_precision = []
    for thrd in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.5, 0.8, 1]:
        # print('thrd', thrd)
        with open('/datadrive-2/data/fortune500_test/sensitivity_semi_baseline/'+str(thrd) + '.json', 'r') as f:
            res = json.load(f)
        y_pred = res
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for index, item in enumerate(y_pred):
            # print('y_pred[index] == y_test[index]', y_pred[index], y_test[index], y_pred[index] == y_test[index])
            if y_pred[index] == y_test[index]:
                if y_test[index] == True:
                    tp += 1
                else:
                    tn += 1
            else:
                if y_test[index] == True:
                    fn += 1
                else:
                    fp += 1
        # print('tp + fn', tp + fn)
        # print('tp + tn', tp + tn)
        # print('all', tp + fp + fn + tn)
        # print('tp, tn , fp, fn', tp, tn, fp, fn)
        if tp + fp == 0:
            precision = 0
        else:
            precision = (tp / (tp + fp))

        if tp + fn == 0:
            recall = 0
        else:
            recall = (tp / (tp + fn))

        baseline_recall.append(recall)
        baseline_precision.append(precision)
        # print('baseline:', thrd, precision, recall)

    plt.xlabel('recall')
    # plt.xlabel('recall')
    plt.ylabel('precision')

    ours_recall_precision = sorted(ours_recall_precision.items(), key=lambda x: x[0])
    recalls = [item[0] for item in ours_recall_precision]
    precisions = [item[1] for item in ours_recall_precision]

    plt.plot(recalls, precisions, label='ours')
    # plt.plot(baseline_recall, baseline_precision, label='baseline')
    # plt.ylabel('precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.savefig("sensitivity_res.png")
    return

def workbook2domain():
    map_dict = {
        'campusship': 'ups',
        'microsoftonline': 'microsoft',
        'dhe': 'ibm',
        'dailyrates': 'ups',
        'git': 'ibm',
        'e-logistics': 'ups',
        'thailand': 'intel',
        'ep': 'microsoft',
        'ext': 'ti',
        'csg': 'ups',
        'ec': 'ups',
        'my': 'ups',
        'elogistics': 'ups',
        'chinaquest': 'ups',
        'seller': 'microsoft',
        'transform': 'microsoft',
        'prize': 'ups',
        'lahgrqm5xee': 'ibm',
        'cursurimicrosoft': 'microsoft',
        'wwx': 'ups',
        'ads': 'microsoft',
        'w3': 'ibm',
        'shmicrosoft': 'microsoft',
        'mp': 'microsoft',
        'cloudapps': 'cisco',
        'techcommunity': 'microsoft',
        'office': 'fedex',
        'secure': 'fedex',
        'saptools': 'ups',
    }

    workbook2domain = {}
    domain2workbook = {}
    workbook2company = {}
    company2workbook = {}
    with open("/datadrive/data_fortune500/crawled_index_fortune500.txt", 'r') as f:
        txt_file = f.read()
        lines = txt_file.split('\n')
        for line in lines:
            try:
                workbook_name, url = line.split('\t')
                if 'https://' in url:
                    domain = 'https://' + url.replace('https://', '').split('/')[0]
                elif 'http://' in url:
                    domain = 'http://' + url.replace('http://', '').split('/')[0]
                print(workbook_name, domain)
                company = domain.split('.')[1]
                if company == 'com':
                    company = domain.replace('https://', '').replace('http://', '').split('.')[0]
                if company in map_dict:
                    company = map_dict[company]
                workbook2domain[workbook_name] = domain
                workbook2company[workbook_name] = company
                if domain not in domain2workbook:
                    domain2workbook[domain] = []
                domain2workbook[domain].append(workbook_name)
                if company not in company2workbook:
                    company2workbook[company] = []
                company2workbook[company].append(workbook_name)
            except:
                continue
    with open("fortune500_workbook2domain.json", 'w') as f:
        json.dump(workbook2domain, f)
    with open("fortune500_domain2workbook.json", 'w') as f:
        json.dump(domain2workbook, f)
    with open("fortune500_workbook2company.json", 'w') as f:
        json.dump(workbook2company, f)
    with open("fortune500_company2workbook.json", 'w') as f:
        json.dump(company2workbook, f)

def look_workbook2domain():
    with open("sensity_workbooks.json", 'r') as f:
        sensity_workbooks = json.load(f)
    with open('fortune500_formulatoken2r1c1.json','r') as f:
        top10domain_formulatoken2r1c1 = json.load(f)
    with open("fortune500_workbook2domain.json", 'r') as f:
        workbook2domain = json.load(f)
    with open("fortune500_domain2workbook.json", 'r') as f:
        domain2workbook = json.load(f)
    with open("fortune500_workbook2company.json", 'r') as f:
        workbook2company = json.load(f)
    with open("fortune500_company2workbook.json", 'r') as f:
        company2workbook = json.load(f)

    # print(domain2workbook.keys())
    companys = list(company2workbook.keys())
    # print(companys)
    print(len(top10domain_formulatoken2r1c1))
    company2formulatoken = {}
    company2sensityworkbook = {}
    # for company in company2workbook:
    #     print(company, len(company2workbook[company]))
    for formula_token in top10domain_formulatoken2r1c1:
        workbook = formula_token.split('---')[0]
        if workbook not in workbook2company:
            continue
        company = workbook2company[workbook]
        if company not in company2formulatoken:
            company2formulatoken[company] = []
        company2formulatoken[company].append(formula_token)
    for company in company2formulatoken:
        print(company, len(company2formulatoken[company]))
        with open("formulas_"+company + '.json', 'w') as f:
            json.dump(company2formulatoken[company], f)
    for workbook in sensity_workbooks:
        company = workbook2company[workbook]
        if company not in company2sensityworkbook:
            company2sensityworkbook[company] = []
        company2sensityworkbook[company].append(workbook)

    print("############")
    for company in company2sensityworkbook:
        print(company, len(company2sensityworkbook[company]))

def reduce_companies():
    with open(root_path + 'dedup_workbooks.json','r') as f:
        dedup_workbooks = json.load(f)
    with open("sensity_workbooks.json", 'r') as f:
        sensity_workbooks = json.load(f)
    with open("fortune500_company2workbook.json", 'r') as f:
        company2workbook = json.load(f)
    with open("Formulas_fortune500_with_id.json",'r') as f:
        formulas = json.load(f)

    all_res = []
    for company in company2workbook:
        workbook2num = {}
        res = []
        if os.path.exists("formulas_"+company + '.json'):
            with open("formulas_"+company + '.json', 'r') as f:
                formula_tokens = json.load(f)
            for formula_token in formula_tokens:
                workbook_name = formula_token.split('---')[0]
                if workbook_name not in dedup_workbooks:
                    continue
                if workbook_name not in workbook2num:
                    workbook2num[workbook_name] = 0
                workbook2num[workbook_name] += 1
                if workbook2num[workbook_name] < 10:
                    res.append(formula_token)
                    all_res.append(formula_token)
            with open("reduced_formulas_"+company + '.json', 'w') as f:
                json.dump(res, f)
            print(company, len(res))
    with open("reduced_formulas.json", 'w') as f:
        json.dump(all_res, f)

def find_best_formula(company_name):
    
    model = torch.load('/datadrive-2/data/finegrained_model_16/epoch_31')
    files = os.listdir('.')

    rm_list = []
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)
    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)
    temp_dict = {}
    with open('fortune500_formulatoken2r1c1.json','r') as f:
        top10domain_formulatoken2r1c1 = json.load(f)
    with open("Formulas_fortune500_with_id.json",'r') as f:
        formulas = json.load(f)
    with open("fortune500_workbook2company.json", 'r') as f:
        workbook2company = json.load(f)
    # files = [item for item in files if 'reduced_formulas_' in item]
    # files = ['reduced_formulas_ibm.json', 'reduced_formulas_ti.json','reduced_formulas_cisco.json',]
    files = ['reduced_formulas_' + company_name + '.json']
    for filename in files:
        # company_name = filename.split('_')[2].replace(".json", '')
        
        # if company_name not in ['cisco','ibm','ti','pge']:
        # if company_name not in ['cisco']:
        #     continue
        print('company_name', company_name)
        with open("fortune500_company2workbook.json", 'r') as f:
            constrained_workbooks = json.load(f)
        print(len(constrained_workbooks[company_name]))
        new_constrained_workbooks = []
        wbjson = {}

        for workbook_name in constrained_workbooks[company_name]:
            if os.path.exists("../Demo/fix_fortune500/" + workbook_name + '.json'):
                # with open("../Demo/fix_fortune500/" + workbook_name + '.json', 'r') as f:
                    # wbjson[workbook_name] = json.load(f)
                new_constrained_workbooks.append(workbook_name)
        constrained_workbooks[company_name] = new_constrained_workbooks

        with open(filename, 'r') as f:
            reduced_formulas = json.load(f)
        with open("fortune500_company2workbook.json", 'w') as f:
            json.dump(constrained_workbooks, f)
        for index, formula_token in enumerate(reduced_formulas):
            # if formula_token != '274875045929873670489611466378788773284-ibm_india_rohs_information_q4-2020_annexure_a_ibm.xlsx---Annexure A---2---10':
            #     continue
            # if formula_token == '112098117031976528283222326480875385993-9120axi-pwr-chn-17-6.xlsx---2.4GHz (XOR)---131---14':
            #     continue
            if index == 140:
                continue
            print(company_name, index, len(reduced_formulas))
            print(formula_token)
            filesheet = formula_token.split('---')[0] + '---' + formula_token.split('---')[1]
            if os.path.exists(root_path + "company_model1_res/" + formula_token + '.json'):
                continue
            if not os.path.exists(root_path + 'company_model1_similar_sheet/' + filesheet + '.json'):
                continue
            with open(root_path + 'company_model1_similar_sheet/' + filesheet + '.json', 'r') as f:
                similar_sheets = json.load(f)
            
            # print('formula_token', formula_token)
            filename = formula_token.split('---')[0]
            sheetname = formula_token.split('---')[1]
            origin_row = int(formula_token.split('---')[2])
            origin_col = int(formula_token.split('---')[3])
            with open("../Demo/fix_fortune500/" + filename + '.json', 'r') as f:
                    wbjson = json.load(f)
            if not os.path.exists(root_path + 'company_formula_jsons/' + formula_token + '.json'):
                generate_demo_features(filename=filename, sheetname=sheetname, workbook_json=wbjson, origin_row=origin_row, origin_col=origin_col, save_path=root_path + 'company_formula_jsons/', is_look=True, cross=False)
            if not os.path.exists(root_path + 'company_before_formulas/' + formula_token + '.npy'):
                res = generate_one_before_feaure(formula_token=formula_token, bert_dict=bert_dict, content_tem_dict=content_tem_dict, mask=2, source_root_path = root_path + 'company_formula_jsons/', saved_root_path=root_path + 'company_before_formulas/')
            if not os.path.exists(root_path + 'company_after_formulas/' + formula_token + '.npy'):
                generate_one_after_feature(formula_token=formula_token, model=model, before_path=root_path + 'company_before_formulas/' + formula_token + '.npy', after_path=root_path + 'company_after_formulas/' + formula_token + '.npy')
            origin_after_feature = np.load(root_path + 'company_after_formulas/' + formula_token + '.npy', allow_pickle=True)
            origin_r1c1 = top10domain_formulatoken2r1c1[formula_token]
            # res = {}
            best_formula_token = ''
            best_r1c1 = ''
            best_score = np.inf
            for sim_pair in similar_sheets:
                similar_sheet = sim_pair[0]
                for formula in formulas:
                    # print(formula)
                    other_formula_token = formula['filesheet'] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
                    if formula_token == other_formula_token:
                        continue
                    other_filename = other_formula_token.split('---')[0]
                    if other_filename not in workbook2company:
                        continue
                    if workbook2company[other_filename] != company_name:
                        continue
                    other_sheetname = other_formula_token.split('---')[1]
                    
                    # print("similar_sheet", similar_sheet)
                    if other_filename + '---' + other_sheetname != similar_sheet:
                        continue
                    # print("other_filename + '---' + other_sheetname", other_filename + '---' + other_sheetname)
                    with open("../Demo/fix_fortune500/" + other_filename + '.json', 'r') as f:
                        wbjson = json.load(f)
                    # print('similar_sheet', similar_sheet, 'other_filename', other_filename)
                    other_origin_row = formula['fr']
                    other_origin_col = formula['fc']
                    if not os.path.exists(root_path + 'company_formula_jsons/' + other_formula_token + '.json'):
                        generate_demo_features(filename=other_filename, sheetname=other_sheetname, workbook_json=wbjson, origin_row=other_origin_row, origin_col=other_origin_col, save_path=root_path + 'company_formula_jsons/', is_look=True, cross=False)
                    if not os.path.exists(root_path + 'company_before_formulas/' + other_formula_token + '.npy'):
                        temp_dict = generate_one_before_feaure(formula_token=other_formula_token, bert_dict=bert_dict, content_tem_dict=content_tem_dict, mask=2, source_root_path = root_path + 'company_formula_jsons/', saved_root_path=root_path + 'company_before_formulas/', temp_dict = temp_dict)
                    if not os.path.exists(root_path + 'company_after_formulas/' + other_formula_token + '.npy'):
                        generate_one_after_feature(formula_token=other_formula_token, model=model, before_path=root_path + 'company_before_formulas/' + other_formula_token + '.npy', after_path=root_path + 'company_after_formulas/' + other_formula_token + '.npy')
                    other_after_feature = np.load(root_path + 'company_after_formulas/' + other_formula_token + '.npy', allow_pickle=True)
                    # print('dis', euclidean(origin_after_feature, other_after_feature))
                    if best_score > euclidean(origin_after_feature, other_after_feature):
                        best_score = euclidean(origin_after_feature, other_after_feature)
                        best_formula_token = other_formula_token
                        best_r1c1 = top10domain_formulatoken2r1c1[best_formula_token]
                # print('best_formula_token', best_formula_token)
            with open(root_path + "company_model1_res/" + formula_token + '.json', 'w') as f:
                json.dump([formula_token, best_formula_token, origin_r1c1, best_r1c1, origin_r1c1 == best_r1c1], f)
                # print('filesheet', filesheet)
                # print('similar_sheet', similar_sheet)
            print([formula_token, best_formula_token, origin_r1c1, best_r1c1, origin_r1c1 == best_r1c1])
            # if not origin_r1c1 == best_r1c1:
            # break
        # break
# def temp():
#     filelist = os.listdir(root_path + 'company_model1_res/')
#     for filename in filelist:
#         with open(root_path + 'company_model1_res/' + filename, 'r') as f:
#             res = json.load(f)
#         print(res)

def similar_sheet_baseline():
    files = ['reduced_formulas_ibm.json', 'reduced_formulas_ti.json','reduced_formulas_cisco.json','reduced_formulas_pge.json']
    res = []
    with open('fortune500_company2workbook.json', 'r') as f:
        company2workbook = json.load(f)
    with open('fortune500_formulatoken2r1c1.json','r') as f:
        formulatoken2r1c1 = json.load(f)
    save_path = root_path + 'company_simple_res/'
    for filename in files:
        with open(filename, 'r') as f:
            reduced_formulas = json.load(f)
        company_name = filename.replace('.json','').split('_')[-1]
        workbooks = company2workbook[company_name]
        for index, formula_token in enumerate(reduced_formulas):
            print(company_name, index, len(reduced_formulas))
            final_res = False
            filename = formula_token.split('---')[0]
            sheetname = formula_token.split('---')[1]
            fr = formula_token.split('---')[2]
            fc = formula_token.split('---')[3]
            r1c1 = formulatoken2r1c1[formula_token]
            for other_filename in workbooks:
                if other_filename == filename:
                    continue
                issame = is_in_same_cluster(filename, other_filename, 0.2)
                if issame == False:
                    continue
                other_formula_token = other_filename + '---' + sheetname + '---' + fr + '---' + fc
                if other_formula_token in formulatoken2r1c1:
                    found_r1c1 = formulatoken2r1c1[other_formula_token]
                if r1c1 ==found_r1c1:
                    final_res == True
                    break
            with open(save_path + formula_token + '.json', 'w') as f:
                json.dump([final_res], f)
                
def test_eff():
    
    # filelist = os.listdir("/datadrive-2/data/fortune500_test/demo_before_features_mask2_fix/")
    # test_eff_formulas = random.sample(filelist, 10)
    # test_eff_formulas = [item.replace(".json", '') for item in test_eff_formulas]
    # with open("test_eff_formulas.json", 'w') as f:
        # json.dump(test_eff_formulas, f)
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)
    
    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)

    with open("test_eff_formulas.json", 'r') as f:
        test_eff_formulas = json.load(f)
    temp_bert_dict = {}
    for formula_token in test_eff_formulas:
        formula_token = formula_token.replace('.npy', '')
        start_time = time.time()
        before_res = generate_one_before_feaure(formula_token, bert_dict, content_tem_dict, mask=2, temp_dict = temp_bert_dict, source_root_path = root_path + 'demo_tile_features_fix/', saved_root_path=root_path + 'demo_before_features_mask2_fix/')
        if type(before_res).__name__ != 'str':
            temp_bert_dict = before_res
        end_time = time.time()
        print(end_time - start_time)
        # break
    
def temp():
    files = ['reduced_formulas_ibm.json', 'reduced_formulas_ti.json','reduced_formulas_cisco.json','reduced_formulas_pge.json']
    res = []
    for filename in files:
        with open(filename, 'r') as f:
            red_res = json.load(f)
        res += red_res
    with open('reduced_formulas.json', 'w') as f:
        json.dump(res, f)  

def generate_cluster_res():
    datasets = ['deco', 'fuste']
    thresh_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
    thresh_list += list(np.arange(1,100,5))
    # thresh_list = [100]
    for dataset in datasets:
        for threshold in thresh_list:
            clustering(dataset, threshold)
            # break
        # break

def eval_similar_sheets():
    # datasets = ['deco']
    datasets = ['fuste']
    thresh_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
    thresh_list += list(np.arange(1,100,5))
    thresh_list = list(set(thresh_list))
    thresh_list.sort()
    for dataset in datasets:
        with open("../Mondrian-master/" + dataset + '_filename_index.json', 'r') as f:
            fileindex = json.load(f)
        
        for threshold in thresh_list:
            ground_truth_path = '../Mondrian-master/' + dataset + '_groundtruth.json'
            with open(ground_truth_path, 'r') as f:
                gt_json = json.load(f)
            predict_path = '/datadrive-2/data/' + dataset + '_test/clusters/' + str(threshold) + '.json'
            with open(predict_path, 'r') as f:
                pred_json = json.load(f)
            new_pred_json = {}
            for keyw in pred_json['filesheet2clusterid']:
                if keyw not in list(gt_json['filesheet2clusterid'].keys()):
                    continue
                new_pred_json[keyw] = pred_json['filesheet2clusterid'][keyw]


            filesheets_list1 = list(gt_json['filesheet2clusterid'].keys())
            # print(len(filesheets_list1))
            filesheets_list2 = list(new_pred_json.keys())
            # print(len(filesheets_list2))

            keylist = gt_json['filesheet2clusterid'].keys()
            y_true = []
            y_pred = []
            for keyw in keylist:
                y_true.append(gt_json['filesheet2clusterid'][keyw])
                y_pred.append(new_pred_json[keyw])

            nmi = normalized_mutual_info_score(y_pred, y_true)
            ami = adjusted_mutual_info_score(y_pred, y_true)
            homo = homogeneity_score(y_true, y_pred)
            completeness = completeness_score(y_true, y_pred)
            randscore = rand_score(y_true, y_pred)
            ars = adjusted_rand_score(y_true, y_pred)
            vm = v_measure_score(y_true, y_pred)
            print('xxxxxxxxxxxx' + dataset  + '   ' + str(threshold))
            # print("Template threshold:", threshold)
            print("Normalized mutual info:", nmi)
            print("Adjusted mutual info:", ami)
            print('Rand score', randscore)
            print('Adjusted random score', ars)
            print("Homogeneity score:", homo)
            print("Completeness score:", completeness)
            print("V measure score:", vm)


            # break
        # break

def check_loss():
    dataset = 'deco'
    # filelist = os.listdir("/datadrive-2/data/"+dataset+"_test/sheet_after_features")
    filelist = os.listdir("/datadrive-2/data/"+dataset+"_test/sheets_before_features")
    # filelist = os.listdir("/datadrive-2/data/"+dataset+"_test/sheets_json_feaures")
    # filesheets = os.listdir('/datadrive/projects/Mondrian-master/res/'+dataset+'/csv')
    with open("../Mondrian-master/" + dataset + '_filename_index.json', 'r') as f:
        filesheets = json.load(f)
    filesheets = [item[0:-4] for item in filesheets]
    filename_list = []
    new_filesheets = []
    for filesheet in filesheets:
        if '.xls_' in filesheet:
            splited_list = filesheet.split('.xls_')
            sheetname = splited_list[-1].replace('.xlsx', '')
            filename = splited_list[0]+'.xls'
        else:
            splited_list = filesheet.split('.xlsx_')
            sheetname = splited_list[-1]
            filename = splited_list[0]+'.xlsx'
        new_filesheets.append(filename + '---' + sheetname)
        if filename not in filename_list:
            filename_list.append(filename)

    filelist = [item.replace(".npy", '') for item in filelist]
    # print('filelist',filelist)
    print(len(set(filelist) & set(new_filesheets)))
    print(len(new_filesheets))
    print(set(new_filesheets) - (set(filelist) & set(new_filesheets)))

    flist = os.listdir('../Demo/fix_'+dataset + '/')
    flist = [item.replace('.json','') for item in flist]
    print(len(set(filename_list) & set(flist)))
   
    print(len(filename_list))
    with open('../Demo/' + dataset + '_filename_list.json', 'w') as f:
        json.dump(filename_list, f)
def look_res_test_distance():
    res = np.load('fuste_res_distance.npy', allow_pickle=True).item()
    pprint.pprint(res['ef77d010-d1cf-4da0-b240-384b5d794ab5.xlsx---Sheet1'][0:100])
if __name__ == '__main__':
    # rerun_demo_features(1,1)
    # check_ctime()
    # generate_positive_pair()
    # generate_negative_pair()
    # download()
    # look_negative()
    # generate_training_data()
    # look_positive()
    # para_gen_saved_pos()
    # look_one_feature()
    # look_saved_pos_neg()
    # batch_gen_neighbor_tripliet()
    # generate_neighbor_triplet(1,20)
    # finetune = FineTune(batch_size=100, epoch_nums=200, iscontinue=False, l2=True, continue_epoch=42)
    # finetune.finetune_shift(margin = 1)
    # finetune.finetune_content_shift(margin = 20)
    # finetune.save_data(train=True, need_save=True)
    # finetune.outer_triplet_training(margin = 20, dynamic=False)
    # test_del_collect()
    # finetune.save_triplet_data()
    # finetune.save_triplet_data_1010()
    # finetune.retraining_model_2(margin = 20)
    # finetune.outer_training()
    # finetune.test(model_path = 'triplet_all_fine_tune_models_1', save_path = 'triplet_all_distance_1') # l2n_triplet_all_distance
    # finetune.retrain_test(model_path = 'model2', save_path = 'model2_distance') # l2n_triplet_all_distance
    # finetune.resave_distance(save_path = 'model2_distance')
    # finetune.initial_test()
    # generate_test_triple2index()
    # look_train_idnex()
    # relative_eval()
    # look_test_result()
    # pr_curve()
    # generate_triplet()
    # batch_gen_tripliet()
    # look_triplet()
    # pr_distance_curve() # triplet_all_fine_tune_models_1 
    # 
    # relative_eval()
    # look_saved_finetune_triplet()
    # look_training_data()
    # look_fine_tune_test_result()
    # look_one_batch_size()
    # para_save_feature()
    # look_distance()
    # generate_negative_neighbors()
    # trans_negative_neighbors()
    # para_save_neighbor_feature()
    # look_finetune_neighbor_triplet()
    # generate_shift_training_triples()
    # generate_content_shift_training_triples()
    # test_finetune(model_path = '/datadrive-2/data/finetune_shift_content_models/epoch_200', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_fix/', after_path=root_path + 'demo_after_shift_content/', first_save_path=root_path + 'first_res_shift_content/', second_save_path = root_path + 'second_res_shift_content/', mask=0) #### please rename path
    # test_finetune(model_path = '/datadrive-2/data/finetune_shift_content_models/epoch_200', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask_fix/', after_path=root_path + 'demo_after_shift_content_mask/', first_save_path=root_path + 'first_res_shift_content_mask/', second_save_path = root_path + 'second_res_shift_content_mask/', mask=1) #### please rename path
    # test_finetune(model_path = '/datadrive-2/data/finetune_shift_content_models/epoch_200', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_shift_content_mask2/', first_save_path=root_path + 'first_res_shift_content_mask2/', second_save_path = root_path + 'second_res_shift_content_mask2/', mask=2, reverse = True) #### please rename path
    # test_finetune(model_path = '196model/cnn_new_dynamic_triplet_margin_1_3_12', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_fix/', after_path=root_path + 'demo_after_features_model1_mask0/', first_save_path=root_path + 'first_res_model1_mask0/', second_save_path = root_path + 'second_res_model1_mask0/', mask=0)
    # test_finetune(model_path = '196model/cnn_new_dynamic_triplet_margin_1_3_12', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask_fix/', after_path=root_path + 'demo_after_features_model1_mask1/', first_save_path=root_path + 'first_res_model1_mask1/', second_save_path = root_path + 'second_res_model1_mask1/', mask=1)
    # test_finetune(model_path = '196model/cnn_new_dynamic_triplet_margin_1_3_12', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_features_model1_mask2/', first_save_path=root_path + 'first_res_model1_mask2/', second_save_path = root_path + 'second_res_model1_mask2/', mask=2, reverse=True)
    # test_finetune(model_path = '/datadrive-2/data/fintune_shift_models_l2/epoch_30', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_30/', first_save_path=root_path + 'first_res_pretrain_30/', second_save_path = root_path + 'second_res_pretrain_30/', mask=2) #### please rename path
    # test_finetune(model_path = '/datadrive-2/data/l2_model/4_1', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_4_1/', first_save_path=root_path + 'first_res_pretrain_4_1/', second_save_path = root_path + 'second_res_pretrain_4_1/', mask=2) #### please rename path
    # test_finetune(model_path = '/datadrive-2/data/finetune_specific_l2/epoch_3', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_epoch_3/', first_save_path=root_path + 'first_res_fintune_epoch_3/', second_save_path = root_path + 'second_res_finetune_epoch_3/', mask=2) #### please rename path
    # test_finetune(model_path = '/datadrive-2/data/fintune_shift_models_l2/epoch_8', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_epoch8/', first_save_path=root_path + 'first_res_pretrain_epoch8/', second_save_path = root_path + 'second_res_pretrain_epoch8/', mask=2) #### please rename path
    # test_finetune(model_path = '/datadrive-2/data/finetune_shift_new_l2/epoch_3', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_shift_epoch3_c2/', first_save_path=root_path + 'first_res_shift_epoch3_c2/', second_save_path = root_path + 'second_res_shift_epoch3_c2/', mask=2) #### please rename path
    # test_finetune(1,1,model_path = '/datadrive-2/data/finetune_specific_l2_new/epoch_23', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_specific_epoch23_c2/', first_save_path=root_path + 'first_res_specific_epoch23_c2/', second_save_path = root_path + 'second_res_specific_epoch23_c2/', mask=2, cross_path='') #### please rename path
    # test_finetune(1,1,model_path = '/datadrive-2/data/cross_finetune_specific_l2/epoch_3', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'cross_before_features/', after_path=root_path + 'after_cross_3/', first_save_path=root_path + 'first_res_cross_3/', second_save_path = root_path + 'second_res_cross_3/', mask=2, cross=True, cross_path = root_path+'demo_before_features_mask2_fix/') #### please rename path cross_before_features

    # eval_fintune(eval_file="fortune500_test_formula.json", second_res_path=root_path + 'second_res_specific_epoch23_c2/', save_path='epoch23_accuracy.json')
    # eval_fintune(eval_file="fortune500_test_formula.json", second_res_path=root_path + 'second_res_specific_epoch23_c2/', save_path='epoch23_accuracy.json')
    # eval_fintune(eval_file="fortune500_test_formula.json", second_res_path=root_path + 'second_res_specific_5_10/', save_path='5_10_accuracy.json', log_path = '5_10_log.txt')
    # eval_fintune(eval_file="fortune500_test_formula.json", second_res_path=root_path + 'second_res_ref/', save_path='ref_accuracy.json', log_path = 'ref_log.txt')
    # eval_fintune(eval_file="fortune500_test_formula.json", second_res_path=root_path + 'second_res_specific_1_5/', save_path='1_5_accuracy.json', log_path = '1_5_log.txt')
    # eval_fintune(eval_file="fortune500_test_formula.json", second_res_path=root_path + 'second_res_cross_3/', save_path='cross_3_accuracy.json', log_path = 'cross_3_log.txt')
    # eval_fintune(eval_file="fortune500_test_formula.json", second_res_path=root_path + 'second_res_specific_5_10_mask1/', save_path='5_10_mask1_accuracy.json', log_path = '5_10_mask1_log_dollar.txt', after_path = '')
    # eval_fintune(eval_file='reduced_formulas.json', second_res_path=root_path + 'company_second_res/', save_path='company_second_res_accuracy.json', log_path = 'company_second_res_log.txt', after_path = '')
    # generate_all_demo_features()
    # generate_all_before()
    # generate_all_after()

    # pair_distance()
    # look_distance_dict()
    # sample_efficiency()
    # generate_sensitivity_ground_truth()
    # sensitivity_evaluation()


    # eval_fintune(eval_file="fortune500_test_formula.json", second_res_path=root_path + 'second_res_finegrain/', save_path='fingrain_accuracy.json', log_path = 'fingrain_log.txt')
    # eval_fintune(eval_file="fortune500_test_formula.json", second_res_path=root_path + 'second_res_finegrain_16/', save_path='fingrain_16_accuracy.json', log_path = 'fingrain_16_log.txt')
    # eval_fintune(eval_file="fortune500_test_formula.json", after_path = root_path + 'demo_after_finegrain_16/', second_res_path=root_path + 'second_res_finegrain_16_copyround/', save_path='fingrain_16_copyround_accuracy.json', log_path = 'fingrain_16_copyround_log.txt')
    # eval_fintune(eval_file="fortune500_test_formula.json", after_path = root_path + 'demo_after_finegrain_16/', second_res_path=root_path + 'second_res_finegrain_16_copyround/', save_path='uvd_accuracy.json', log_path = 'uvd_log.txt')
    # eval_fintune(eval_file="sample_efficiency.json", after_path = root_path + 'demo_after_finegrain_16/', second_res_path=root_path + 'second_res_finegrain_16_copyround/', save_path='sample_efficiency_accuracy.json', log_path = 'sample_efficiency_log.txt')
    # check_round_size(eval_file="fortune500_test_formula.json", second_res_path=root_path + 'second_res_finegrain_16/')
    # validate
    # for model_id in range(42,60):
    #     validate(model_id)

    # cross validate
    # for model_id in range(11,31):
        # cross_validate(model_id)


    # validate_root_path = '/datadrive-3/data/fortune500_test/validate/'
    # validate_root_path = '/datadrive-3/data/fortune500_test/validate_cross/'
    # for model_id in list(os.listdir(validate_root_path)):
    #     if os.path.exists(validate_root_path + str(model_id)+'/accuracy.json'):
    #         continue
    #     eval_fintune(eval_file="fortune500_val_formula.json", second_res_path=validate_root_path + str(model_id)+'/second_res_specific/', save_path=validate_root_path + str(model_id)+'/accuracy.json', log_path = validate_root_path + str(model_id)+'/log.txt')

    # look_accuracy()
    # generate_sample_files()
    # generate_bert_sim()
    # check_befor_features()
    # /datadrive-2/data/l2_model/4_1


    # upperbound()


    # is_all_same()


    ## 93634798945550820643207473511448801554-20190509ab1613.xlsx---AB 1613---10---8重跑一下mask
    # test_finetune(5,5,model_path = '/datadrive-2/data/finetune_specific_l2_5_10/epoch_42', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_specific_5_10/', first_save_path=root_path + 'first_res_specific_5_10/', second_save_path = root_path + 'second_res_specific_5_10/', mask=2, cross_path = '') #### please rename path
    # test_finetune(5,5,model_path = '/datadrive-2/data/finetune_specific_l2_5_10/epoch_42', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask1_fix/', after_path=root_path + 'demo_after_specific_5_10_mask1/', first_save_path=root_path + 'first_res_specific_5_10_mask1/', second_save_path = root_path + 'second_res_specific_5_10_mask1/', mask=1, cross_path = '') #### please rename path
    # test_finetune(1,1,model_path = '/datadrive-2/data/training_ref_model/epoch_20', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_ref/', first_save_path=root_path + 'first_res_ref/', second_save_path = root_path + 'second_res_ref/', mask=2, cross_path = '') #### please rename path
    # test_finetune(1,1,model_path = '/datadrive-2/data/finegrained_model/epoch_38', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_finegrain/', first_save_path=root_path + 'first_res_finegrain/', second_save_path = root_path + 'second_res_finegrain/', mask=2, cross_path = '') #### please rename path
    # test_finetune(1,1,model_path = '/datadrive-2/data/finegrained_model_16/epoch_31', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_finegrain_16/', first_save_path=root_path + 'first_res_finegrain_16/', second_save_path = root_path + 'second_res_finegrain_16/', mask=2, cross_path = '') #### please rename path
    # copyround_finetune(model_path = '/datadrive-2/data/finegrained_model_16/epoch_31', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_finegrain_16/', second_save_path = root_path + 'second_res_finegrain_16_copyround/', mask=2, cross_path = '') #### please rename path
    # copyround_finetune(model_path = '/datadrive-2/data/finegrained_model_16/epoch_31', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_finegrain_16/', second_save_path = root_path + 'second_res_finegrain_16_copyround/', mask=2, cross_path = '') #### please rename path
    # copyround_finetune(file_path = 'reduced_formulas.json', model_path = '/datadrive-2/data/finegrained_model_16/epoch_31', tile_path = root_path + 'demo_tile_features_fix/',before_path=root_path + 'demo_before_features_mask2_fix/', after_path=root_path + 'demo_after_finegrain_16/', second_save_path = root_path + 'company_second_res/', mask=2, cross_path = '') #### please rename path
    # workbook2domain()
    # look_workbook2domain()
    # reduce_companies()
    # find_best_formula('ibm')
    # find_best_formula("cisco")
    # find_best_formula("ti")
    # find_best_formula("pge")
    # temp()
    # similar_sheet_baseline()
    # test_eff()

    # similarity_test_sheets()
    # similarity_test_distance()
    # look_res_test_distance()
    # generate_cluster_res()
    # check_loss()
    eval_similar_sheets()
    