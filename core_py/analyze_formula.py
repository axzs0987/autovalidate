import json
import pprint
import os
import torch
import numpy as np
from multiprocessing import Process
from sentence_transformers import SentenceTransformer
from torch.autograd import Variable
import shutil
import time
import faiss
import copy
import matplotlib.pyplot as plt
import random
import xlrd

# root_path = '/datadrive-2/data/top10domain_test/'
# root_path = '/datadrive-2/data/middle10domain_test/'
# root_path = '/datadrive-2/data/fortune500_test/'

root_path = '../data_drive/data_two/'


def look_savedsheets():
    filenames = os.listdir("filename2bertfeature/")
    print(len(filenames))


def look_fromula():
    with open("TrainingFormulas_mergerange_custom_new_res_1.json", 'r') as f:
        formulas = json.load(f)

    print(formulas[
              '../../data/000/01be5462b92dd3f2540f868351fb2310_YXRsYW50aWNxdWFsaXR5Lm9yZwk1NC4xNjMuMjI1LjIzNQ==.xls.xlsx---Terminations by LOS'][
              'RC[-1]/R12C4'])


def look_sheetfeatures():
    with open('custom_bert_notmask_dvid2sheetfeature.json', 'r') as f:
        dvid2sheetfeature = json.load(f)
    with open("Formulas_20000sheets_groupr1c1_custom.json", 'r') as f:
        formulas_20000sheets = json.load(f)
    pprint.pprint(len(dvid2sheetfeature.keys()))
    first_dvid = list(dvid2sheetfeature.keys())[0]
    print(first_dvid)
    print(dvid2sheetfeature[first_dvid])
    print(len(dvid2sheetfeature[first_dvid][0]))
    print(len(dvid2sheetfeature[first_dvid]))
    # with open("../AnalyzeDV/data/types/custom/custom_list.json",'r') as f:
    #     custom_list = json.load(f)
    # # print(custom_list[0])

    # res = {}
    # # model = torch.load('cnn_new_dynamic_triplet_margin_1_3_12')
    # for filename in formulas_20000sheets:
    #     if filename not in res:
    #         for item in custom_list:
    #             if item['FileName'] + '---' + item['SheetName'] == filename:
    #                 kw = str(item['ID']) +'---' + str(item['batch_id'])
    #                 if kw in dvid2sheetfeature:
    #                     np.save('filename2bertfeature/'+filename.split('/')[-1], dvid2sheetfeature[kw])
    #                     # print(res[filename])
    #                     # return
    #         # break


def save_files():
    with open("Formulas_20000sheets_groupr1c1.json", 'r') as f:
        formulas_20000sheets = json.load(f)

    with open("20000sampled_files.json", 'w') as f:
        json.dump(list(formulas_20000sheets.keys()), f)


def look_Formulas_20000sheets_r1c1():
    with open("Formulas_20000sheets_groupr1c1.json", 'r') as f:
        formulas_20000sheets = json.load(f)

    # pprint.pprint(list(formulas_20000sheets.keys()))
    filename = list(formulas_20000sheets.keys())[1]
    for item in formulas_20000sheets[filename]:
        if len(formulas_20000sheets[filename][item]) > 1:
            pprint.pprint(formulas_20000sheets[filename][item])


def devide_range():
    # with open("Formulas_20000sheets_groupr1c1.json",'r') as f:
    #     formulas_20000sheets = json.load(f)
    with open("Formulas_20000sheets_groupr1c1.json", 'r') as f:
        formulas_20000sheets = json.load(f)

    result = {}
    for filesheet in formulas_20000sheets:
        result[filesheet] = {}
        for r1c1 in formulas_20000sheets[filesheet]:
            id_ = 0
            res = {}
            item = {'fr': 0, 'fc': 0, 'lr': 0, 'lc': 0, 'r1c1': r1c1, 'formulas': []}

            for formula in formulas_20000sheets[filesheet][r1c1]:
                is_add = False
                if item['fr'] == 0:
                    item['fr'] = formula['row']
                    item['lr'] = formula['row']
                    item['fc'] = formula['column']
                    item['lc'] = formula['column']
                    item['formulas'].append(formula)
                elif formula['row'] == item['fr'] - 1 and formula['column'] >= item['fc'] and formula['column'] <= item[
                    'lc']:
                    item['fr'] = formula['row']
                    item['formulas'].append(formula)
                elif formula['row'] == item['lr'] + 1 and formula['column'] >= item['fc'] and formula['column'] <= item[
                    'lc']:
                    item['lr'] = formula['row']
                    item['formulas'].append(formula)
                elif formula['column'] == item['fc'] - 1 and formula['row'] >= item['fr'] and formula['row'] <= item[
                    'lr']:
                    item['fc'] = formula['column']
                    item['formulas'].append(formula)
                elif formula['column'] == item['lc'] + 1 and formula['row'] >= item['fr'] and formula['row'] <= item[
                    'lr']:
                    item['lc'] = formula['column']
                    item['formulas'].append(formula)
                else:
                    res[id_] = item
                    is_add = True
                    item = {'fr': 0, 'fc': 0, 'lr': 0, 'lc': 0, 'r1c1': r1c1, 'formulas': []}
                    item['fr'] = formula['row']
                    item['lr'] = formula['row']
                    item['fc'] = formula['column']
                    item['lc'] = formula['column']
                    item['formulas'].append(formula)
                    id_ += 1

            # if len(formulas_20000sheets[filesheet][r1c1]) == 1:
            if item['fr'] != 0:
                res[id_] = item
            result[filesheet][r1c1] = res
    with open("Formulas_20000sheets_mergerange.json", 'w') as f:
        json.dump(result, f)


def devide_range_recheck():
    # with open("Formulas_20000sheets_groupr1c1_custom.json",'r') as f:
    #     formulas_20000sheets = json.load(f)
    with open("origin_top10domain_formulas_groupr1c1.json", 'r') as f:
        formulas_20000sheets = json.load(f)

    result = {}
    count = 0
    for filesheet in formulas_20000sheets:
        # if filesheet.replace('/UnzipData','') != '../../data/000/0176061ac2ea686b8887dc8fed2805f0_d3cyLmp1c3RhbnN3ZXIuY29tCTEwNC4xNi4xMjEuMTEy.xls.xlsx---Questionnaire_Input':
        # continue
        result[filesheet] = {}
        count += 1
        print(count, len(formulas_20000sheets))
        for r1c1 in formulas_20000sheets[filesheet]:
            id_ = 0
            res = {}
            item = {'fr': 0, 'fc': 0, 'lr': 0, 'lc': 0, 'r1c1': r1c1, 'formulas': []}
            # print('r1c1', r1c1)
            for formula in formulas_20000sheets[filesheet][r1c1]:
                is_add = False
                # print("#######")
                # print('fr', item['fr'])
                # print('fc', item['fc'])
                # print('lr', item['lr'])
                # print('lc', item['lc'])
                if item['fr'] == 0:
                    item['fr'] = formula['row']
                    item['lr'] = formula['row']
                    item['fc'] = formula['column']
                    item['lc'] = formula['column']
                    item['formulas'].append(formula)
                elif formula['row'] == item['fr'] - 1 and formula['column'] >= item['fc'] and formula['column'] <= item[
                    'lc']:
                    # print('r-1')
                    item['fr'] = formula['row']
                    item['formulas'].append(formula)
                elif formula['row'] == item['lr'] + 1 and formula['column'] >= item['fc'] and formula['column'] <= item[
                    'lc']:
                    # print('r+1')
                    item['lr'] = formula['row']
                    item['formulas'].append(formula)
                elif formula['column'] == item['fc'] - 1 and formula['row'] >= item['fr'] and formula['row'] <= item[
                    'lr']:
                    # print('c-1')
                    item['fc'] = formula['column']
                    item['formulas'].append(formula)
                elif formula['column'] == item['lc'] + 1 and formula['row'] >= item['fr'] and formula['row'] <= item[
                    'lr']:
                    # print('c+1')
                    item['lc'] = formula['column']
                    item['formulas'].append(formula)
                else:
                    # print('else')
                    res[id_] = item
                    # if not os.path.exists("formulas/"+filesheet.split('/')[5]):
                    #     os.mkdir("formulas/"+filesheet.split('/')[5])
                    # with open("formulas/"+filesheet.split('/')[5]+"/"+str(item['fr'])+'_'+str(item['fc'])+".json",'w') as f:
                    #     json.dump(item, f)
                    is_add = True
                    item = {'fr': 0, 'fc': 0, 'lr': 0, 'lc': 0, 'r1c1': r1c1, 'formulas': []}
                    item['fr'] = formula['row']
                    item['lr'] = formula['row']
                    item['fc'] = formula['column']
                    item['lc'] = formula['column']
                    item['formulas'].append(formula)
                    id_ += 1

            # if len(formulas_20000sheets[filesheet][r1c1]) == 1:
            if item['fr'] != 0:
                res[id_] = item
                # if not os.path.exists("formulas/"+filesheet.split('/')[5]):
                #     os.mkdir("formulas/"+filesheet.split('/')[5])
                # with open("formulas/"+filesheet.split('/')[5]+"/"+str(item['fr'])+'_'+str(item['fc'])+".json",'w') as f:
                #     json.dump(item, f)
            result[filesheet][r1c1] = res

    res_num = 0
    for filesheet in result:
        for r1c1 in result[filesheet]:
            res_num += len(result[filesheet][r1c1])

    print(res_num)

    # with open("Formulas_20000sheets_mergerange_custom.json",'w') as f:
    #     json.dump(result, f)
    with open("origin_top10domain_mergerange.json", 'w') as f:
        json.dump(result, f)


def devide_training_range_recheck():
    # with open("TrainingFormulas_groupby_r1c1.json",'r') as f:
    #     formulas_20000sheets = json.load(f)
    # with open("origin_top10domain_groupby_r1c1.json",'r') as f:
    # with open("origin_middle10domain_groupby_r1c1.json",'r') as f:
    #     formulas_20000sheets = json.load(f)
    with open("../data_set/formula_data_set/origin_fortune500_groupby_r1c1.json", 'r') as f:
        formulas_20000sheets = json.load(f)

    result = {}
    count = 0
    for filesheet in formulas_20000sheets:
        # if filesheet.replace('/UnzipData','') != '../../data/000/0176061ac2ea686b8887dc8fed2805f0_d3cyLmp1c3RhbnN3ZXIuY29tCTEwNC4xNi4xMjEuMTEy.xls.xlsx---Questionnaire_Input':
        # continue
        result[filesheet] = {}
        count += 1
        print(count, len(formulas_20000sheets))
        for r1c1 in formulas_20000sheets[filesheet]:
            id_ = 0
            res = {}
            item = {'fr': 0, 'fc': 0, 'lr': 0, 'lc': 0, 'r1c1': r1c1, 'formulas': []}
            # print('r1c1', r1c1)
            for formula in formulas_20000sheets[filesheet][r1c1]:
                is_add = False
                # print("#######")
                # print('fr', item['fr'])
                # print('fc', item['fc'])
                # print('lr', item['lr'])
                # print('lc', item['lc'])
                if item['fr'] == 0:
                    item['fr'] = formula['row']
                    item['lr'] = formula['row']
                    item['fc'] = formula['column']
                    item['lc'] = formula['column']
                    item['formulas'].append(formula)
                elif formula['row'] == item['fr'] - 1 and formula['column'] >= item['fc'] and formula['column'] <= item[
                    'lc']:
                    # print('r-1')
                    item['fr'] = formula['row']
                    item['formulas'].append(formula)
                elif formula['row'] == item['lr'] + 1 and formula['column'] >= item['fc'] and formula['column'] <= item[
                    'lc']:
                    # print('r+1')
                    item['lr'] = formula['row']
                    item['formulas'].append(formula)
                elif formula['column'] == item['fc'] - 1 and formula['row'] >= item['fr'] and formula['row'] <= item[
                    'lr']:
                    # print('c-1')
                    item['fc'] = formula['column']
                    item['formulas'].append(formula)
                elif formula['column'] == item['lc'] + 1 and formula['row'] >= item['fr'] and formula['row'] <= item[
                    'lr']:
                    # print('c+1')
                    item['lc'] = formula['column']
                    item['formulas'].append(formula)
                else:
                    # print('else')
                    res[id_] = item
                    # if not os.path.exists("training_formulas/"+filesheet.split('/')[-1]):
                    #     os.mkdir("training_formulas/"+filesheet.split('/')[-1])
                    # with open("training_formulas/"+filesheet.split('/')[-1]+"/"+str(item['fr'])+'_'+str(item['fc'])+".json",'w') as f:
                    #     json.dump(item, f)
                    is_add = True
                    item = {'fr': 0, 'fc': 0, 'lr': 0, 'lc': 0, 'r1c1': r1c1, 'formulas': []}
                    item['fr'] = formula['row']
                    item['lr'] = formula['row']
                    item['fc'] = formula['column']
                    item['lc'] = formula['column']
                    item['formulas'].append(formula)
                    id_ += 1

            # if len(formulas_20000sheets[filesheet][r1c1]) == 1:
            if item['fr'] != 0:
                res[id_] = item
                # if not os.path.exists("training_formulas/"+filesheet.split('/')[-1]):
                #     os.mkdir("training_formulas/"+filesheet.split('/')[-1])
                # with open("training_formulas/"+filesheet.split('/')[-1]+"/"+str(item['fr'])+'_'+str(item['fc'])+".json",'w') as f:
                #     json.dump(item, f)
            result[filesheet][r1c1] = res
    # with open("TrainingFormulas_mergerange_custom.json",'w') as f:
    #     json.dump(result, f)
    # with open("origin_top10domain_mergerange.json",'w') as f:
    #     json.dump(result, f)
    # with open("origin_middle10domain_mergerange.json",'w') as f:
    #     json.dump(result, f)
    with open("../data_set/formula_data_set/origin_fortune500_mergerange.json", 'w') as f:
        json.dump(result, f)


def resave_mergerange():
    with open("Formulas_77772sheets_mergerange_custom.json", 'r') as f:
        formulas_20000sheets = json.load(f)

    all_cell = 0
    has_range = 0
    has_range_cell = 0
    one_cell = 0

    print_num = 0
    lenlist = []

    res = []

    filesname = set()
    no_formula = 0
    print(len(set(formulas_20000sheets.keys())))
    for filesheet_name in formulas_20000sheets:
        if len(formulas_20000sheets[filesheet_name].keys()) == 0:
            no_formula += 1
        for r1c1 in formulas_20000sheets[filesheet_name]:
            for id_ in formulas_20000sheets[filesheet_name][r1c1]:
                formula = {}
                filesname.add(filesheet_name)
                formula['filesheet'] = filesheet_name
                formula['r1c1'] = r1c1
                formula['fr'] = formulas_20000sheets[filesheet_name][r1c1][id_]['fr']
                formula['fc'] = formulas_20000sheets[filesheet_name][r1c1][id_]['fc']
                formula['lr'] = formulas_20000sheets[filesheet_name][r1c1][id_]['lr']
                formula['lc'] = formulas_20000sheets[filesheet_name][r1c1][id_]['lc']
                # if len(formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']) == 1:
                # one_cell += 1
                # if len(formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']) > 1:
                # has_range += 1
                # lenlist.append(len(formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']))
                # for item in formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']:
                # all_cell += 1
                # if len(formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']) > 1:
                # has_range_cell += 1
                res.append(formula)
            # if print_num < 20:

            #     print('#######')
            #     print(filesheet_name)
            #     print(r1c1)
            #     pprint.pprint(formulas_20000sheets[filesheet_name][r1c1])
            #     print_num += 1

    # pprint.pprint(res[file_])
    # print('has_range_cell', has_range_cell)
    # print('has_range', has_range)
    # print('all_cell', all_cell)
    # print('one_cell', one_cell)
    # print('len(filesname)', len(filesname))

    # print(len(res))
    # print('no_formula', no_formula)
    with open("Formula_77772.json", 'w') as f:
        json.dump(res, f)


def resave_training_mergerange():
    # with open("TrainingFormulas_mergerange_custom_new_res_1.json",'r') as f:
    #     formulas_20000sheets = json.load(f)
    # with open("origin_top10domain_mergerange_new_res_1.json",'r') as f:
    # with open("origin_middle10domain_mergerange_new_res_1.json",'r') as f:
    with open("../data_set/formula_data_set/origin_fortune500_mergerange_new_res_1.json", 'r') as f:
        formulas_20000sheets = json.load(f)

    all_cell = 0
    has_range = 0
    has_range_cell = 0
    one_cell = 0

    print_num = 0
    lenlist = []

    res = []

    filesname = set()
    no_formula = 0
    count = 0

    formula_id = 1
    for filesheet_name in formulas_20000sheets:
        count += 1
        print(count, len(set(formulas_20000sheets.keys())))
        if len(formulas_20000sheets[filesheet_name].keys()) == 0:
            no_formula += 1
        for r1c1 in formulas_20000sheets[filesheet_name]:
            for id_ in formulas_20000sheets[filesheet_name][r1c1]:
                formula = {}
                filesname.add(filesheet_name)
                formula['id'] = formula_id
                formula['filesheet'] = filesheet_name
                formula['r1c1'] = r1c1
                formula['fr'] = formulas_20000sheets[filesheet_name][r1c1][id_]['fr']
                formula['fc'] = formulas_20000sheets[filesheet_name][r1c1][id_]['fc']
                formula['lr'] = formulas_20000sheets[filesheet_name][r1c1][id_]['lr']
                formula['lc'] = formulas_20000sheets[filesheet_name][r1c1][id_]['lc']
                formula_id += 1
                # if len(formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']) == 1:
                # one_cell += 1
                # if len(formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']) > 1:
                # has_range += 1
                # lenlist.append(len(formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']))
                # for item in formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']:
                # all_cell += 1
                # if len(formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']) > 1:
                # has_range_cell += 1
                res.append(formula)
            # if print_num < 20:

            #     print('#######')
            #     print(filesheet_name)
            #     print(r1c1)
            #     pprint.pprint(formulas_20000sheets[filesheet_name][r1c1])
            #     print_num += 1

    # pprint.pprint(res[file_])
    # print('has_range_cell', has_range_cell)
    # print('has_range', has_range)
    # print('all_cell', all_cell)
    # print('one_cell', one_cell)
    # print('len(filesname)', len(filesname))

    # print(len(res))
    # print('no_formula', no_formula)
    # with open("Formulas_training_with_id.json",'w') as f:
    #     json.dump(res, f)
    # with open("Formulas_top10domain_with_id.json",'w') as f:
    # with open("Formulas_middle10domain_with_id.json",'w') as f:
    #     json.dump(res, f)
    with open("../data_set/formula_data_set/Formulas_fortune500_with_id.json", 'w') as f:
        json.dump(res, f)


def generate_file_sheet():
    # with open("Formulas_top10domain_with_id.json", 'r') as f:
    # with open("Formulas_middle10domain_with_id.json", 'r') as f:
    with open("Formulas_fortune500_with_id.json", 'r') as f:
        formulas = json.load(f)

    res = {}
    for index, formula in enumerate(formulas):
        print(index, len(formulas))
        filename = formula['filesheet'].split("---")[0]
        sheetname = formula['filesheet'].split("---")[1]
        if filename not in res:
            res[filename] = []
        res[filename].append(sheetname)

    # with open("origin_top10domain_filesheets.json", 'w') as f:
    # with open("origin_middle10domain_filesheets.json", 'w') as f:
    with open("origin_fortune500_filesheets.json", 'w') as f:
        json.dump(res, f)


def look_mergerange():
    with open("Formulas_20000sheets_mergerange_custom.json", 'r') as f:
        formulas_20000sheets = json.load(f)

    all_cell = 0
    has_range = 0
    has_range_cell = 0
    one_cell = 0

    print_num = 0
    lenlist = []
    for filesheet_name in formulas_20000sheets:
        for r1c1 in formulas_20000sheets[filesheet_name]:
            for id_ in formulas_20000sheets[filesheet_name][r1c1]:

                if len(formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']) == 1:
                    one_cell += 1
                if len(formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']) > 1:
                    has_range += 1
                    lenlist.append(len(formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']))
                for item in formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']:
                    all_cell += 1
                    if len(formulas_20000sheets[filesheet_name][r1c1][id_]['formulas']) > 1:
                        has_range_cell += 1
            if print_num < 20:
                print('#######')
                print(filesheet_name)
                print(r1c1)
                pprint.pprint(formulas_20000sheets[filesheet_name][r1c1])
                print_num += 1

    # pprint.pprint(res[file_])
    print('has_range_cell', has_range_cell)
    print('has_range', has_range)
    print('all_cell', all_cell)
    print('one_cell', one_cell)

    all_ = 0
    with open("Formulas_20000sheets_groupr1c1_custom.json", 'r') as f:
        formulas_20000sheets = json.load(f)
    for filesheet in formulas_20000sheets:
        for r1c1 in formulas_20000sheets[filesheet]:
            all_ += len(formulas_20000sheets[filesheet][r1c1])
        # break
    print('all_', all_)
    # print(lenlist)


def best_sketch_recall():
    with open("Formulas_20000sheets_mergerange_custom.json", 'r') as f:
        formulas_20000sheets = json.load(f)
    with open("r1c12template.json", 'r') as f:
        r1c12template = json.load(f)

    fail = 0
    can_find_number = 0
    for filesheet_name in formulas_20000sheets:
        workbookname, sheetname = filesheet_name.split('---')
        for r1c1 in formulas_20000sheets[filesheet_name]:
            found = False
            if r1c1 not in r1c12template:
                fail += 1
                continue
            for cand_filesheet_name in formulas_20000sheets:
                cand_workbookname, cand_sheetname = cand_filesheet_name.split('---')
                if cand_workbookname == workbookname:
                    continue
                for cand_r1c1 in formulas_20000sheets[cand_filesheet_name]:
                    if cand_r1c1 not in r1c12template:
                        continue
                    if r1c12template[cand_r1c1] == r1c12template[r1c1]:
                        found = True
                        break
                if found:
                    break
            if found:
                can_find_number += len(formulas_20000sheets[filesheet_name][r1c1])

    print('can_find_number', can_find_number)
    print('fail', fail)


def para_sketch_best_recall():
    # process = [
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(1,20)),
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(2,20)), 
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(3,20)),
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(4,20)), 
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(5,20)),
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(6,20)), 
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(7,20)),
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(8,20)), 
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(9,20)),
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(10,20)), 
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(11,20)),
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(12,20)), 
    #     Process(target=best_sketch_recall_with_sheetsimilarity,args=(13,20)),
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(14,20)), 
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(15,20)),
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(16,20)), 
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(17,20)),
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(18,20)), 
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(19,20)), 
    #     Process(target=best_sketch_recall_with_sheetsimilarity, args=(20,20)), 
    # ]
    # [p.start() for p in process]  # 开启了两个进程
    # [p.join() for p in process]   # 等待两个进程依次结束

    need_sketch_look_recall = []
    best_sketch_recall_formula_ids = []
    for thread_id in range(1, 21):
        print('threa_id', thread_id)
        with open('need_sketch_look_recall_' + str(thread_id) + '.json', 'r') as f:
            temp1 = json.load(f)
        need_sketch_look_recall += temp1

        with open('best_sketch_recall_formula_ids_' + str(thread_id) + '.json', 'r') as f:
            temp2 = json.load(f)
        best_sketch_recall_formula_ids += temp2

    with open('need_sketch_look_recall.json', 'w') as f:
        json.dump(need_sketch_look_recall, f)
    with open('best_sketch_recall_formula_ids.json', 'w') as f:
        json.dump(best_sketch_recall_formula_ids, f)


def best_sketch_recall_with_sheetsimilarity(thread_id, batch_num):
    with open('Formulas_middle10domain_with_id.json', 'r') as f:
        formulas = json.load(f)
    # with open('most_simular_sheet_1900.json', 'r') as f:
    # with open(root_path + 'top10domain_most_simular_sheet.json', 'r') as f:
    # similar_sheets = json.load(f)
    # with open('formula_token_2_template_id_custom.json', 'r') as f:
    # formula_token_2_template_id = json.load(f)
    with open('r1c12template_middle10domain.json', 'r') as f:
        r1c12template_top10domain = json.load(f)

    similar_sheets_list = os.listdir(root_path + 'model1_similar_sheet')
    similar_sheets_list = [i.replace('---1---1.json', '') for i in similar_sheets_list]

    test_list = os.listdir(root_path + 'model1_middle10domain_formula2afterfeature_test')
    test_list = [i.replace('.npy', '') for i in test_list]

    # count = 0
    best_recall_formula_ids = []
    found_formulas = 0
    need_look_recall = []

    best_found_formulas = 0
    batch_len = len(formulas) / batch_num

    count = 0
    for index, formula in enumerate(formulas):
        if thread_id != batch_num:
            if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1):
                continue
        print(index, len(formulas))
        formula_token = formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
        if formula_token not in test_list:
            continue
        target_r1c1 = formula['r1c1']
        filesheet = formula['filesheet'].split('/')[-1]
        found = False
        best_found = False
        if filesheet not in similar_sheets_list:
            continue

        count += 1
        with open(root_path + 'model1_similar_sheet/' + filesheet + '---1---1.json', 'r') as f:
            sheet_feature = json.load(f)
        for similar_sheet_pair in sheet_feature:
            similar_sheet = similar_sheet_pair[0].replace('---1---1', '')
            if similar_sheet == filesheet:
                continue

            for other_formula in formulas:
                # print("other_formula['filesheet'].split('/')[-1]", other_formula['filesheet'].split('/')[-1])
                # print('similar_sheet', similar_sheet)
                if other_formula['filesheet'].split('/')[-1] == similar_sheet:
                    other_formula_token = similar_sheet + '---' + str(other_formula['fr']) + '---' + str(
                        other_formula['fc'])
                    res_r1c1 = other_formula['r1c1']
                    print('other_formula_token', other_formula_token)
                    if target_r1c1 not in r1c12template_top10domain:
                        target_tid = -1
                    else:
                        target_tid = r1c12template_top10domain[target_r1c1]
                    if res_r1c1 not in r1c12template_top10domain:
                        res_tid = -1
                    else:
                        res_tid = r1c12template_top10domain[res_r1c1]
                    if target_tid == res_tid:
                        best_found = True
                        if other_formula['fr'] <= formula['fr'] + 5 and other_formula['fr'] >= formula['fr'] - 5 and \
                                other_formula['fc'] <= formula['fc'] + 5 and other_formula['fc'] >= formula['fc'] - 5:
                            found = True
                            # print('found')
                            # print('formula', formula)
                            # print('otherformula', other_formula)
                            best_recall_formula_ids.append(formula['id'])
                            break
                        else:
                            need_look_recall.append([formula, other_formula])

            if found:
                break
        if found:
            found_formulas += 1
        if best_found:
            best_found_formulas += 1
        print('best found', best_found)
    print('sketch +-5 recall', found_formulas / count)
    print('sketch best recall', best_found_formulas / count)
    # print('found_formulas', found_formulas)
    # print('best_found_formulas', best_found_formulas)
    # with open("need_sketch_look_recall_"+str(thread_id)+".json", 'w') as f:
    #     json.dump(need_look_recall, f)
    # with open("best_sketch_recall_formula_ids_"+str(thread_id)+".json", 'w') as f:
    #     json.dump(best_recall_formula_ids, f)


def best_hasfunc_recall_with_sheetsimilarity():
    with open('Formula_hasfunc_with_id.json', 'r') as f:
        formulas = json.load(f)
    with open('most_simular_sheet_1900.json', 'r') as f:
        similar_sheets = json.load(f)
    with open('formula_token_2_template_id_custom.json', 'r') as f:
        formula_token_2_template_id = json.load(f)

    # count = 0
    best_recall_formula_ids = []
    found_formulas = 0
    need_look_recall = []

    best_found_formulas = 0
    for index, formula in enumerate(formulas):
        print(index, len(formulas))
        formula_token = formula['filesheet'].split('/')[5] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
        filesheet = formula['filesheet'].split('/')[5]
        found = False
        best_found = False
        for similar_sheet_pair in similar_sheets[filesheet]:
            similar_sheet = similar_sheet_pair[0]
            if similar_sheet == filesheet:
                continue

            for other_formula in formulas:
                if other_formula['filesheet'].split('/')[-1] == similar_sheet:
                    other_formula_token = similar_sheet + '---' + str(other_formula['fr']) + '---' + str(
                        other_formula['fc'])
                    if formula['r1c1'] == other_formula['r1c1']:
                        best_found = True
                        if other_formula['fr'] <= formula['fr'] + 5 and other_formula['fr'] >= formula['fr'] - 5 and \
                                other_formula['fc'] <= formula['fc'] + 5 and other_formula['fc'] >= formula['fc'] - 5:
                            found = True
                            print('found')
                            print('formula', formula)
                            print('otherformula', other_formula)
                            best_recall_formula_ids.append(formula['id'])
                            break
                        else:
                            need_look_recall.append([formula, other_formula])

            if found:
                break
        if found:
            found_formulas += 1
        if best_found:
            best_found_formulas += 1
    print(' +-5 recall', found_formulas / len(formulas))
    print('best recall', best_found_formulas / len(formulas))
    with open("need_hasfunc_look_recall.json", 'w') as f:
        json.dump(need_look_recall, f)
    with open("best_hasfunc_recall_formula_ids.json", 'w') as f:
        json.dump(best_recall_formula_ids, f)


def generate_faild_case():
    with open("best_recall_formula_ids.json", 'r') as f:
        best_recall_formula_ids = json.load(f)
    test_fail = []
    for thread_id in range(1, 21):
        with open("only_not_found_res_" + str(thread_id) + ".json", 'r') as f:
            temp = json.load(f)
            test_fail += temp
    failed_case = list(set(test_fail) & set(best_recall_formula_ids))
    print(len(failed_case))
    with open('failed_case.json', 'w') as f:
        json.dump(failed_case, f)


def look_failed_case(thread_id, count_id):
    with open('failed_case.json', 'r') as f:
        failed_case = json.load(f)
    with open('formula_token_2_template_id.json', 'r') as f:
        formula_token_2_template_id = json.load(f)
    with open('formula_token_2_r1c1.json', 'r') as f:
        formula_token_2_r1c1 = json.load(f)
    with open('Formula_77772_with_id.json', 'r') as f:
        formulas = json.load(f)

    res = np.load("deal00_most_similar_formula_1900_" + str(thread_id) + ".npy", allow_pickle=True).item()
    found_res = []
    not_found_res = []
    count = 0
    print('start', thread_id)

    all_fail = 0
    same_template = 0
    for index, formula in enumerate(formulas):
        if formula['id'] not in failed_case:
            continue
        all_fail += 1
        formula_token = formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
        r1c1 = formula['r1c1']
        if formula_token not in res:
            continue
        res_list = sorted(res[formula_token].items(), key=lambda x: x[1], reverse=True)[0]
        top1_formula_token = res_list[0]

        if formula_token not in formula_token_2_template_id:
            template_id_1 = -1
        else:
            template_id_1 = formula_token_2_template_id[formula_token]

        if top1_formula_token not in formula_token_2_template_id:
            template_id_2 = -1
        else:
            template_id_2 = formula_token_2_template_id[top1_formula_token]
        if template_id_1 == template_id_2:
            same_template += 1
        # print(template_id_1, template_id_2)

        if res_list[1] < 0.7 or res_list[1] > 0.8:
            continue
        count += 1
        if count == count_id:
            print("#############")
            print(index)
            print('formula_token', formula_token, r1c1)
            # if os.path.exists('deal00formula2afterbertfeature/'+formula_token + '.npy'):
            #     feature = np.load('deal00formula2afterbertfeature/'+formula_token + '.npy', allow_pickle=True)
            # else:
            #     feature = np.load('formula2afterbertfeature/'+formula_token + '.npy', allow_pickle=True)
            # before_feature = feature
            res_list = sorted(res[formula_token].items(), key=lambda x: x[1], reverse=True)[0:20]
            # print(feature)
            for item in res_list:
                print(item, formula_token_2_r1c1[item[0]])
                # print()
                # if os.path.exists('deal00formula2afterbertfeature/'+item[0] + '.npy'):
                #     other_feature = np.load('deal00formula2afterbertfeature/'+item[0] + '.npy', allow_pickle=True)
                # else:
                #     other_feature = np.load('formula2afterbertfeature/'+item[0] + '.npy', allow_pickle=True)

                # print((feature==other_feature).all())
                # print(other_feature)
                # before_feature = other_feature

            break
    # return count
    return all_fail, same_template, count


def count_model_bad():
    with open('failed_case.json', 'r') as f:
        failed_case = json.load(f)  # r1c1可能不同，template相同
    with open('formula_token_2_template_id.json', 'r') as f:
        formula_token_2_template_id = json.load(f)
    with open('formula_token_2_r1c1.json', 'r') as f:
        formula_token_2_r1c1 = json.load(f)
    with open('Formula_77772_with_id.json', 'r') as f:
        formulas = json.load(f)

    count = 0
    all_fail = 0
    same_template = 0

    found = 0
    found_same_position = 0
    found_not_same_position = 0
    not_found = 0

    r1c1_fail = 0
    template_fail = 0
    for thread_id in range(1, 21):
        print(thread_id, 20)
        res = np.load("deal00_most_similar_formula_1900_" + str(thread_id) + ".npy", allow_pickle=True).item()
        for index, formula in enumerate(formulas):
            if formula['id'] not in failed_case:
                continue
            all_fail += 1
            formula_token = formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' + str(
                formula['fc'])
            r1c1 = formula['r1c1']
            if formula_token not in res:
                continue
            r1c1_fail += 1
            res_list = sorted(res[formula_token].items(), key=lambda x: x[1], reverse=True)[0]
            top1_formula_token = res_list[0]

            if formula_token not in formula_token_2_template_id:
                template_id_1 = -1
            else:
                template_id_1 = formula_token_2_template_id[formula_token]

            if top1_formula_token not in formula_token_2_template_id:
                template_id_2 = -1
            else:
                template_id_2 = formula_token_2_template_id[top1_formula_token]
            if template_id_1 == template_id_2:
                continue
            template_fail += 1
            res_list = sorted(res[formula_token].items(), key=lambda x: x[1], reverse=True)[0:20]
            not_found_flag = True
            for item in res_list:

                cand_fr = item[0].split('---')[2]
                cand_fc = item[0].split('---')[3]

                fr = formula_token.split('---')[2]
                fc = formula_token.split('---')[3]
                if r1c1 == formula_token_2_r1c1[item[0]]:
                    found += 1
                    if fr == cand_fr and fc == cand_fc:
                        found_same_position += 1
                    else:
                        found_not_same_position += 1
                    not_found_flag = False
                    break
            if not_found_flag:
                not_found += 1

    print('found', found)
    print('not_found', not_found)
    print('found_same_position', found_same_position)
    print('found_not_same_position', found_not_same_position)
    print('all_fail', all_fail)
    print('template_fail', template_fail)
    print('r1c1_fail', r1c1_fail)
    return all_fail, same_template, count


def best_recall():
    with open("Formulas_20000sheets_mergerange_custom.json", 'r') as f:
        formulas_20000sheets = json.load(f)

    all_cell = 0
    has_range = 0
    has_range_cell = 0
    one_cell = 0

    print_num = 0
    lenlist = []

    can_find_number = 0
    for filesheet_name in formulas_20000sheets:
        workbookname, sheetname = filesheet_name.split('---')
        for r1c1 in formulas_20000sheets[filesheet_name]:
            found = False
            for cand_filesheet_name in formulas_20000sheets:
                cand_workbookname, cand_sheetname = cand_filesheet_name.split('---')
                if cand_workbookname == workbookname:
                    continue
                for cand_r1c1 in formulas_20000sheets[cand_filesheet_name]:
                    if cand_r1c1 == r1c1:
                        found = True
                        break
                if found:
                    break
            if found:
                can_find_number += len(formulas_20000sheets[filesheet_name][r1c1])

    print('can_find_number', can_find_number)


def batch_anaylze_range():
    batch_origin_data = []
    # with open("../AnalyzeDV/Formulas_20000sheets_recheck.json",'r') as f:

    # batch_origin_data.append(formulas_20000sheets)
    for index in range(1, 58):
        with open("../AnalyzeDV/origin_top10domain/origin_top10domain_formulas_" + str(index) + ".json", 'r') as f:
            formulas_20000sheets = json.load(f)
        batch_origin_data.append(formulas_20000sheets)
    new_res = {}
    for formulas_20000sheets in batch_origin_data:
        for filesheet_name in list(formulas_20000sheets.keys()):
            new_res[filesheet_name] = {}
            for formula in formulas_20000sheets[filesheet_name]:
                if formula['formulaR1C1'] not in new_res[filesheet_name]:
                    new_res[filesheet_name][formula['formulaR1C1']] = []
                formu = {}
                formu['column'] = formula['column']
                formu['row'] = formula['row']
                formu['formulaR1C1'] = formula['formulaR1C1']
                new_res[filesheet_name][formula['formulaR1C1']].append(formu)

    # with open("Formulas_20000sheets_groupr1c1_check.json",'w') as f:
    with open("origin_top10domain_formulas_groupr1c1.json", 'w') as f:
        json.dump(new_res, f)


def anaylze_range():
    with open("../AnalyzeDV/Formulas_20000sheets_custom.json", 'r') as f:
        formulas_20000sheets = json.load(f)

    new_res = {}
    for filesheet_name in list(formulas_20000sheets.keys()):
        new_res[filesheet_name] = {}
        for formula in formulas_20000sheets[filesheet_name]:
            if formula['formulaR1C1'] not in new_res[filesheet_name]:
                new_res[filesheet_name][formula['formulaR1C1']] = []
            formu = {}
            formu['column'] = formula['column']
            formu['row'] = formula['row']
            formu['formulaA1'] = formula['formulaA1']
            formu['formulaR1C1'] = formula['formulaR1C1']
            new_res[filesheet_name][formula['formulaR1C1']].append(formu)

    with open("Formulas_20000sheets_groupr1c1_custom.json", 'w') as f:
        json.dump(new_res, f)


def anaylze_training_range():
    new_res = {}
    # for save_id in range(1,7):
    # for save_id in range(58, 98):
    # print(save_id, 98)
    # with open("../AnalyzeDV/TrainingFormulas_"+str(save_id)+".json",'r') as f:
    #     training_formulas = json.load(f)
    # with open("../AnalyzeDV/origin_top10domain/origin_top10domain_formulas_"+str(save_id)+".json",'r') as f:
    #     training_formulas = json.load(f)
    # with open("../AnalyzeDV/origin_middle10domain/origin_middle10domain_formulas_"+str(save_id)+".json",'r') as f:
    #     training_formulas = json.load(f)
    with open("../data_set/formula_data_set/origin_data_formulas_" + str(58) + ".json", 'r') as f:
        training_formulas = json.load(f)

    for filesheet_name in list(training_formulas.keys()):
        new_res[filesheet_name] = {}
        for formula in training_formulas[filesheet_name]:
            if formula['formulaR1C1'] not in new_res[filesheet_name]:
                new_res[filesheet_name][formula['formulaR1C1']] = []
            formu = {}
            formu['column'] = formula['column']
            formu['row'] = formula['row']
            new_res[filesheet_name][formula['formulaR1C1']].append(formu)
    print(len(new_res))
    # if save_id % 10 == 0:
    #     with open("origin_fortune500_groupby_r1c1.json", 'w') as f:
    #         json.dump(new_res, f)

    with open("../data_set/formula_data_set/origin_fortune500_groupby_r1c1.json", 'w') as f:
        json.dump(new_res, f)


def save_all_r1c1():
    with open("Formulas_77772sheets_mergerange_custom.json", 'r') as f:
        formulas_20000sheets = json.load(f)
    res = []
    for filename in formulas_20000sheets:
        res += list(formulas_20000sheets[filename].keys())
    with open("Formulas_77772r1c1_custom.json", 'w') as f:
        json.dump(list(set(res)), f)


def save_training_r1c1():
    # with open("origin_middle10domain_mergerange_new_res_1.json",'r') as f:
    #     formulas_20000sheets = json.load(f)
    with open("origin_fortune500_mergerange_new_res_1.json", 'r') as f:
        formulas_20000sheets = json.load(f)
    res = []
    for filename in formulas_20000sheets:
        res += list(formulas_20000sheets[filename].keys())
    # with open("middle10domain_formulas_list.json",'w') as f:
    #     json.dump(list(set(res)), f)
    with open("fortune500_formulas_list.json", 'w') as f:
        json.dump(list(set(res)), f)


def count_all_r1c1():
    with open("Formulas_20000r1c1_custom.json", 'r') as f:
        res = json.load(f)
    print(len(res))


def look_sketch_all_sheetname_2_num():
    with open("Formulas_20000sheets_mergerange_custom.json", 'r') as f:
        formulas_20000sheets = json.load(f)

    with open("../AnalyzeDV/boundary_sheetname_2_file_devided_1.json", 'r') as f:
        boundary_sheetname_2_num = json.load(f)

    with open("../AnalyzeDV/custom_sheetname_2_file_devided_1.json", 'r') as f:
        custom_sheetname_2_num = json.load(f)

    with open("r1c12template.json", 'r') as f:
        r1c12template = json.load(f)
    # print(list(all_sheetname_2_num.keys()))
    # print(custom_sheetname_2_num['Input'])
    files = set()
    for item in formulas_20000sheets.keys():
        files.add(item.split('---')[0])
    # files = set(formulas_20000sheets.keys())
    count = 0

    has_num = 0
    f_has_num = 0
    printed_file = []
    for filename in formulas_20000sheets:

        # if count >=10:
        # break
        workbookname, sheetname = filename.split('---')
        if sheetname not in custom_sheetname_2_num:
            # print('xxxxxxxx')
            # if filename == '../../data/UnzipData/002/1346696a6af6d6c2fca67068b13bd935_d3d3LmFlci5nb3YuYXUJMTUyLjkxLjUzLjE5Mw==.xls.xlsx---Input':
            #     print('workbookname', workbookname)
            #     print('filenamelist', filenamelist)
            # print('cand_files+workbookname',set(boundary_sheetname_2_num[sheetname][0]) & files)
            cand_files = set(boundary_sheetname_2_num[sheetname][0]['filenames']) & files - set([workbookname])
            # print('workbookname', set([workbookname]))
            # print('cand_files', cand_files)
        else:
            # print(filename)
            # print('xxssssssxxxxxx')
            # print('cand_files+workbookname',set(custom_sheetname_2_num[sheetname][0]) & files)
            for filenamelist in custom_sheetname_2_num[sheetname]:
                if workbookname in filenamelist:
                    # if filename == '../../data/UnzipData/002/1346696a6af6d6c2fca67068b13bd935_d3d3LmFlci5nb3YuYXUJMTUyLjkxLjUzLjE5Mw==.xls.xlsx---Input':
                    # print('workbookname', workbookname)
                    # print('filenamelist', filenamelist)
                    cand_files = set(filenamelist) & files - set([workbookname])
                    break
            # print('workbookname', set([workbookname]))
            # print('cand_files', cand_files)

        if len(cand_files) > 0:
            for r1c1 in formulas_20000sheets[filename]:
                has_num += len(formulas_20000sheets[filename][r1c1])
                found = False
                if r1c1 not in r1c12template:
                    continue
                for wbname in cand_files:
                    for fsname in formulas_20000sheets:
                        if wbname in fsname:
                            for cand_r1c1 in formulas_20000sheets[fsname]:
                                if cand_r1c1 not in r1c12template:
                                    continue
                                if r1c12template[cand_r1c1] == r1c12template[r1c1]:
                                    found = True
                                    break
                            if found:
                                break
                    if found:
                        break
                if found:
                    f_has_num += len(formulas_20000sheets[filename][r1c1])
                else:
                    if len(cand_files) > 0 and filename not in printed_file:
                        print('xxxxxxxxxx')
                        print('filename', filename)
                        # print('cand file', list(cand_files))
                        print('r1c1', r1c1)
                        print('Formula', formulas_20000sheets[filename][r1c1])
                        printed_file.append(filename)
                        count += 1
        # print(has_num)
    print(has_num)
    print(f_has_num)


def look_all_sheetname_2_num():
    with open("Formulas_20000sheets_mergerange_custom.json", 'r') as f:
        formulas_20000sheets = json.load(f)

    with open("../AnalyzeDV/boundary_sheetname_2_file_devided_1.json", 'r') as f:
        boundary_sheetname_2_num = json.load(f)

    with open("../AnalyzeDV/custom_sheetname_2_file_devided_1.json", 'r') as f:
        custom_sheetname_2_num = json.load(f)
    # print(list(all_sheetname_2_num.keys()))

    files = set()
    for item in formulas_20000sheets.keys():
        files.add(item.split('---')[0])
    # files = set(formulas_20000sheets.keys())
    count = 0

    has_num = 0
    f_has_num = 0

    inboundarynum = 0
    for filename in formulas_20000sheets:

        # if count >=10:
        # break
        workbookname, sheetname = filename.split('---')
        if sheetname not in custom_sheetname_2_num:
            inboundarynum += 1
            print('xxxxxxxx')
            print('cand_files+workbookname', set(boundary_sheetname_2_num[sheetname][0]) & files)
            cand_files = set(boundary_sheetname_2_num[sheetname][0]['filenames']) & files - set([workbookname])
            print('workbookname', set([workbookname]))
            print('cand_files', cand_files)
        else:

            print('xxssssssxxxxxx')
            print('cand_files+workbookname', set(custom_sheetname_2_num[sheetname][0]) & files)
            # cand_files = set(custom_sheetname_2_num[sheetname][0]) & files - set([workbookname])
            for filenamelist in custom_sheetname_2_num[sheetname]:
                if workbookname in filenamelist:
                    print('workbookname', workbookname)
                    print('filenamelist', filenamelist)
                    cand_files = set(filenamelist) & files - set([workbookname])
                    break
            print('workbookname', set([workbookname]))
            print('cand_files', cand_files)

        if len(cand_files) > 0:
            for r1c1 in formulas_20000sheets[filename]:
                has_num += len(formulas_20000sheets[filename][r1c1])
                found = False
                for wbname in cand_files:
                    for fsname in formulas_20000sheets:
                        if wbname in fsname:
                            if r1c1 in formulas_20000sheets[fsname]:
                                found = True
                                break
                    if found:
                        break
                if found:
                    f_has_num += len(formulas_20000sheets[filename][r1c1])

        print(has_num)
    print(has_num)
    print(f_has_num)

    print(inboundarynum)


def download_sampled_file():
    with open('../AnalyzeDV/sampled_file.json', 'r') as f:
        sampled_file = json.load(f)
    for filename in sampled_file:
        filename = filename.replace("/UnzipData", "").split('---')[0]
        print(filename)
        splits = filename.split('/')
        if not os.path.exists('/datadrive/data/sampled_sheets/' + splits[3]):
            os.mkdir('/datadrive/data/sampled_sheets/' + splits[3])
        os.system('cp ' + filename + ' /datadrive/data/sampled_sheets/' + splits[3] + '/' + splits[4])


def template_analyze():
    with open("Formula_77772_with_id.json", 'r') as f:
        formulas = json.load(f)

    # with open('formula_r1c1_template_custom.json','r') as f:
    #     formula_r1c1_template = json.load(f)

    with open('r1c12template_custom.json', 'r') as f:
        r1c12template = json.load(f)

    template2num = {}

    formula_token_2_template_id = {}
    formula_token_2_r1c1 = {}
    # print(r1c12template.keys())
    for formula in formulas:
        formula_token = formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
        formula_token_2_r1c1[formula_token] = formula['r1c1']
        if formula['r1c1'] not in r1c12template:
            formula_token_2_template_id[formula_token] = -1
            continue
        if r1c12template[formula['r1c1']] not in template2num:
            template2num[r1c12template[formula['r1c1']]] = 0
        template2num[r1c12template[formula['r1c1']]] += 1
        formula_token_2_template_id[formula_token] = r1c12template[formula['r1c1']]
    template2num = sorted(template2num.items(), key=lambda x: x[1], reverse=True)
    with open('template2num.json', 'w') as f:
        json.dump(template2num, f)
    # pprint.pprint(template2num)
    # res = []
    # for template_pair in template2num:
    #     for template in formula_r1c1_template:
    #         if template_pair[0] == template['id']:
    #             template['popularity'] = template_pair[1]
    #             res.append(template)
    # pprint.pprint(formula_r1c1_template[0])

    # with open('formula_r1c1_template_custom_sorted.json','w') as f:
    #     json.dump(res, f)
    # with open('formula_token_2_template_id_custom.json','w') as f:
    #     json.dump(formula_token_2_template_id, f)
    # with open('formula_token_2_r1c1_custom.json','w') as f:
    #     json.dump(formula_token_2_r1c1, f)


def check_formula():
    # with open("Formulas_20000sheets_mergerange_custom.json", 'r') as f:
    #     res = json.load(f)
    with open("origin_top10domain_mergerange.json", 'r') as f:
        res = json.load(f)

    multi_res = {}
    index = 0
    for filesheet in res:
        index += 1
        multi_res[filesheet] = {}
        print(index, len(res))
        for r1c1 in res[filesheet]:
            multi_res[filesheet][r1c1] = {}
            # print('r1c1:', r1c1)
            new_res = {}
            for batch_id in res[filesheet][r1c1]:
                found = False
                for item in new_res:
                    if res[filesheet][r1c1][batch_id]['fc'] <= new_res[item]['lc'] and res[filesheet][r1c1][batch_id][
                        'fc'] >= new_res[item]['fc'] and res[filesheet][r1c1][batch_id]['fr'] <= new_res[item]['lr'] and \
                            res[filesheet][r1c1][batch_id]['fr'] >= new_res[item]['fr'] and \
                            res[filesheet][r1c1][batch_id]['lc'] <= new_res[item]['lc'] and \
                            res[filesheet][r1c1][batch_id]['lc'] >= new_res[item]['fc'] and \
                            res[filesheet][r1c1][batch_id]['lr'] <= new_res[item]['lr'] and \
                            res[filesheet][r1c1][batch_id]['lr'] >= new_res[item]['fr']:
                        found = True
                if found:
                    continue
                # if res[filesheet][r1c1][batch_id]['fc'] != res[filesheet][r1c1][batch_id]['lc']:
                #     new_res[batch_id] = res[filesheet][r1c1][batch_id]
                #     multi_res[filesheet][r1c1][batch_id] = {}
                #     multi_res[filesheet][r1c1][batch_id]['fc'] = new_res[batch_id]['fc']
                #     multi_res[filesheet][r1c1][batch_id]['fr'] = new_res[batch_id]['fr']
                #     multi_res[filesheet][r1c1][batch_id]['lc'] = new_res[batch_id]['lc']
                #     multi_res[filesheet][r1c1][batch_id]['lr'] = new_res[batch_id]['lr']
                #     continue
                for another_batch_id in res[filesheet][r1c1]:
                    if batch_id == another_batch_id:
                        continue
                    if res[filesheet][r1c1][batch_id]['fc'] == res[filesheet][r1c1][another_batch_id]['fc'] and \
                            res[filesheet][r1c1][batch_id]['lc'] == res[filesheet][r1c1][another_batch_id]['lc'] and \
                            res[filesheet][r1c1][another_batch_id]['fr'] == res[filesheet][r1c1][batch_id]['lr'] + 1:
                        res[filesheet][r1c1][batch_id]['lr'] = res[filesheet][r1c1][another_batch_id]['lr']
                        # print("    ##########")
                        # print('    fc', res[filesheet][r1c1][batch_id]['fc'])
                        # print('    fr',res[filesheet][r1c1][batch_id]['fr'])
                        # print('    lc',res[filesheet][r1c1][batch_id]['lc'])
                        # print('    lr',res[filesheet][r1c1][batch_id]['lr'])
                    new_res[batch_id] = res[filesheet][r1c1][batch_id]
                if batch_id not in new_res:
                    new_res[batch_id] = res[filesheet][r1c1][batch_id]
                multi_res[filesheet][r1c1][batch_id] = {}
                multi_res[filesheet][r1c1][batch_id]['fc'] = new_res[batch_id]['fc']
                multi_res[filesheet][r1c1][batch_id]['fr'] = new_res[batch_id]['fr']
                multi_res[filesheet][r1c1][batch_id]['lc'] = new_res[batch_id]['lc']
                multi_res[filesheet][r1c1][batch_id]['lr'] = new_res[batch_id]['lr']
            # print('new_res', new_res)
            # pprint.pprint(res[filesheet][r1c1][batch_id])
    # with open("Formulas_20000sheets_mergerange_custom_new_res_1.json", 'w') as f:
    #     json.dump(multi_res, f)
    with open("orgin_top10domain_mergerange_new_res_1.json", 'w') as f:
        json.dump(multi_res, f)


def check_training_formula():
    # with open("TrainingFormulas_mergerange_custom.json", 'r') as f:
    #     res = json.load(f)
    # with open("origin_top10domain_mergerange.json", 'r') as f:
    # with open("origin_middle10domain_mergerange.json", 'r') as f:
    with open("../data_set/formula_data_set/origin_fortune500_mergerange.json", 'r') as f:
        res = json.load(f)

    multi_res = {}
    index = 0
    for filesheet in res:
        index += 1
        multi_res[filesheet] = {}
        print(index, len(res))
        for r1c1 in res[filesheet]:
            multi_res[filesheet][r1c1] = {}
            # print('r1c1:', r1c1)
            new_res = {}
            for batch_id in res[filesheet][r1c1]:
                found = False
                for item in new_res:
                    if res[filesheet][r1c1][batch_id]['fc'] <= new_res[item]['lc'] and res[filesheet][r1c1][batch_id][
                        'fc'] >= new_res[item]['fc'] and res[filesheet][r1c1][batch_id]['fr'] <= new_res[item]['lr'] and \
                            res[filesheet][r1c1][batch_id]['fr'] >= new_res[item]['fr'] and \
                            res[filesheet][r1c1][batch_id]['lc'] <= new_res[item]['lc'] and \
                            res[filesheet][r1c1][batch_id]['lc'] >= new_res[item]['fc'] and \
                            res[filesheet][r1c1][batch_id]['lr'] <= new_res[item]['lr'] and \
                            res[filesheet][r1c1][batch_id]['lr'] >= new_res[item]['fr']:
                        found = True
                if found:
                    continue
                # if res[filesheet][r1c1][batch_id]['fc'] != res[filesheet][r1c1][batch_id]['lc']:
                #     new_res[batch_id] = res[filesheet][r1c1][batch_id]
                #     multi_res[filesheet][r1c1][batch_id] = {}
                #     multi_res[filesheet][r1c1][batch_id]['fc'] = new_res[batch_id]['fc']
                #     multi_res[filesheet][r1c1][batch_id]['fr'] = new_res[batch_id]['fr']
                #     multi_res[filesheet][r1c1][batch_id]['lc'] = new_res[batch_id]['lc']
                #     multi_res[filesheet][r1c1][batch_id]['lr'] = new_res[batch_id]['lr']
                #     continue
                for another_batch_id in res[filesheet][r1c1]:
                    if batch_id == another_batch_id:
                        continue
                    if res[filesheet][r1c1][batch_id]['fc'] == res[filesheet][r1c1][another_batch_id]['fc'] and \
                            res[filesheet][r1c1][batch_id]['lc'] == res[filesheet][r1c1][another_batch_id]['lc'] and \
                            res[filesheet][r1c1][another_batch_id]['fr'] == res[filesheet][r1c1][batch_id]['lr'] + 1:
                        res[filesheet][r1c1][batch_id]['lr'] = res[filesheet][r1c1][another_batch_id]['lr']
                        # print("    ##########")
                        # print('    fc', res[filesheet][r1c1][batch_id]['fc'])
                        # print('    fr',res[filesheet][r1c1][batch_id]['fr'])
                        # print('    lc',res[filesheet][r1c1][batch_id]['lc'])
                        # print('    lr',res[filesheet][r1c1][batch_id]['lr'])
                    new_res[batch_id] = res[filesheet][r1c1][batch_id]
                if batch_id not in new_res:
                    new_res[batch_id] = res[filesheet][r1c1][batch_id]
                multi_res[filesheet][r1c1][batch_id] = {}
                multi_res[filesheet][r1c1][batch_id]['fc'] = new_res[batch_id]['fc']
                multi_res[filesheet][r1c1][batch_id]['fr'] = new_res[batch_id]['fr']
                multi_res[filesheet][r1c1][batch_id]['lc'] = new_res[batch_id]['lc']
                multi_res[filesheet][r1c1][batch_id]['lr'] = new_res[batch_id]['lr']
            # print('new_res', new_res)
            # pprint.pprint(res[filesheet][r1c1][batch_id])
    # with open("TrainingFormulas_mergerange_custom_new_res_1.json", 'w') as f:
    #     json.dump(multi_res, f)
    # with open("origin_top10domain_mergerange_new_res_1.json", 'w') as f:
    # with open("origin_middle10domain_mergerange_new_res_1.json", 'w') as f:
    with open("../data_set/formula_data_set/origin_fortune500_mergerange_new_res_1.json", 'w') as f:
        json.dump(multi_res, f)


def count_formula():
    # with open("TrainingFormulas_mergerange_custom_new_res_1.json", 'r') as f:
    #     res = json.load(f)
    # with open("origin_top10domain_mergerange_new_res_1.json", 'r') as f:
    # with open("origin_middle10domain_mergerange_new_res_1.json", 'r') as f:
    with open("../data_set/formula_data_set/origin_fortune500_mergerange_new_res_1.json", 'r') as f:
        res = json.load(f)

    count = 0

    c = 0
    for filesheet in res:
        filesheet_num = 0
        for r1c1 in res[filesheet]:
            for batch_id in res[filesheet][r1c1]:
                count += 1
                filesheet_num += 1
        if filesheet_num > 100:
            # print(filesheet.split('/')[5], filesheet_num)
            c += 1
    # print(c)
    print(count)


def mv_small_count_formula_features():
    with open("Formulas_20000sheets_mergerange_custom_new_res_1.json", 'r') as f:
        res = json.load(f)

    for filesheet in res:
        for r1c1 in res[filesheet]:
            for batch_id in res[filesheet][r1c1]:
                print('cp ../AnalyzeDV/FormulaFeatures/' + filesheet.split('---')[0].split('/')[5] + '---' +
                      filesheet.split('---')[1] + '---' + str(res[filesheet][r1c1][batch_id]['fr']) + "---" + str(
                    res[filesheet][r1c1][batch_id]['fc']) + '.json ../AnalyzeDV/FormulaFeatures77772/')
                os.system('cp ../AnalyzeDV/FormulaFeatures/' + filesheet.split('---')[0].split('/')[5] + '---' +
                          filesheet.split('---')[1] + '---' + str(res[filesheet][r1c1][batch_id]['fr']) + "---" + str(
                    res[filesheet][r1c1][batch_id]['fc']) + '.json ../AnalyzeDV/FormulaFeatures77772/')


def formula_token_2_domain():
    import base64
    with open("Formula_77772.json", 'r') as f:
        formulas = json.load(f)
    res = {}
    for formula in formulas:
        formula_token = formula['filesheet'] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
        domain_token = formula['filesheet'].split('/')[-1].split('.')[0].split('_')[-1]
        domain = base64.b64decode(domain_token)
        # print(domain)
        res[formula_token] = str(domain)
    with open("formula_token_2_domain_1900.json", 'w') as f:
        json.dump(res, f)


def all_filesheet_2_domain():
    import base64
    folds_list = os.listdir('/datadrive/data/')
    folds_list = [item for item in folds_list if len(item) == 3]

    res = {}
    domain2num = {}
    for foldid in folds_list:
        filelist = os.listdir('/datadrive/data/' + foldid + '/')
        for index, filename in enumerate(filelist):
            try:
                print(foldid, index, len(filelist))
                domain_token = filename.split('.')[0].split('_')[-1]
                # print('domain_token', domain_token )
                domain = base64.b64decode(domain_token)
                res[foldid + '/' + filename] = str(domain)
                print('str(domain)', str(domain))
                if str(domain) not in domain2num:
                    domain2num[str(domain)] = 0
                domain2num[str(domain)] += 1
            except:
                continue

    with open("filename2domain.json", 'w') as f:
        json.dump(res, f)
    with open("domain2num.json", 'w') as f:
        json.dump(domain2num, f)

    with open("domain2num.json", 'r') as f:
        domain2num = json.load(f)

    domain2num = sorted(domain2num.items(), key=lambda x: x[1], reverse=True)
    new_res = {}
    for tuple in domain2num:
        new_res[tuple[0]] = tuple[1]

    with open("domain2num.json", 'w') as f:
        json.dump(new_res, f)


def select_top10_domain():
    with open("domain2num.json", 'r') as f:
        domain2num = json.load(f)
    with open("filename2domain.json", 'r') as f:
        filename2domain = json.load(f)
    res = []
    res1 = []
    for index, domain in enumerate(domain2num):
        if index == 10:
            break
        print(domain, domain2num[domain])
        for filename in filename2domain:
            if filename2domain[filename] == domain:
                res.append(filename)
            else:
                res1.append(filename)

    # print(len(res))
    # print(len(res1))
    # with open('top10_domain_filenames.json', 'w') as f:
    #     json.dump(res, f)
    # with open('training_filenames.json', 'w') as f:
    #     json.dump(res1, f)


def select_middle10_domain():
    with open("domain2num.json", 'r') as f:
        domain2num = json.load(f)
    with open("filename2domain.json", 'r') as f:
        filename2domain = json.load(f)
    res = []

    res1 = [domain for domain in domain2num if domain2num[domain] > 10]
    middle_index = int(len(res1) / 2)

    up_index = middle_index
    down_index = middle_index

    now_index = up_index
    is_up = True
    while len(res) < 10:

        domain = list(domain2num.keys())[now_index]
        print(domain)
        print('now_index', now_index)
        if '.com' in domain:
            res.append((domain, domain2num[domain]))
            if is_up:
                up_index += 1
                now_index = down_index
            else:
                down_index -= 1
                now_index = up_index
            is_up = not is_up

        else:
            if is_up:
                up_index += 1
                now_index = up_index
            else:
                down_index -= 1
                now_index = down_index

    pprint.pprint(res)
    res1 = []
    for index, domain in enumerate(res):
        # print(domain, domain2num[domain])
        for filename in filename2domain:
            if filename2domain[filename] == domain[0]:
                res1.append(filename)

    # for index,domain in enumerate(domain2num):
    #     if index == 10:
    #         break
    #     print(domain, domain2num[domain])
    #     for filename in filename2domain:
    #         if filename2domain[filename] == domain:
    #             res.append(filename)
    #         else:
    #             res1.append(filename)

    # print(len(res))
    # print(len(res1))
    with open('middle10_domain_filenames.json', 'w') as f:
        json.dump(res1, f)
    # with open('training_filenames.json', 'w') as f:
    #     json.dump(res1, f)


def all_filename_2_domain():
    import base64
    dataset_path = '/datadrive/data/'
    batches = [item for item in list(os.listdir(dataset_path)) if len(item) == 3]
    res = {}
    for batch in batches:
        subpath = dataset_path + batch
        filenames = os.listdir(subpath)
        for filename in filenames:
            domain_token = filename.split('/')[-1].split('.')[0].split('_')[-1]
            # print(domain_token)

            try:
                domain = base64.b64decode(domain_token)
                res[filename] = str(domain)
            except:
                res[filename] = 'unknown'
    with open("all_filenames_2_domain.json", 'w') as f:
        json.dump(res, f)


def generate_mondrain_sheet_features_by_workbook_json(is_look=True):
    def argb_to_rgb(value):
        value = value[-6:]
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

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

    one_invalid_cell = {
        "background_color_r": 0,
        "background_color_g": 0,
        "background_color_b": 0,
        "font_color_r": 0,
        "font_color_g": 0,
        "font_color_b": 0,
        "font_size": 0.0,
        "font_strikethrough": False,
        "font_shadow": False,
        "font_ita": False,
        "font_bold": False,
        "height": 0.0,
        "width": 0.0,
        "content": '',
        "content_template": '',
    },

    workbooks = os.listdir('/datadrive/data_deco/completed/')
    # workbooks = os.listdir('/datadrive/data_fuste/')
    for index, filename in enumerate(workbooks):
        print(index, len(workbooks))
        try:
            wb = xlrd.open_workbook(filename='/datadrive/data_deco/completed/' + filename)
            # wb = xlrd.open_workbook(filename='/datadrive/data_fuste/' + filename)
        except:
            continue
        sheets = wb.sheets()
        if not os.path.exists('../Demo/fix_deco/' + filename + '.json'):
            print('not exists: fix_deco')
            continue
        with open('../Demo/fix_deco/' + filename + '.json', 'r') as f:
            workbook_info = json.load(f)

        for sheetname in sheets:
            sheetname = sheetname.name
            filesheet = filename + '---' + sheetname

            filename = filesheet.split('---')[0].split('/')[-1]
            sheetname = filesheet.split('---')[1]

            if os.path.exists(
                    '/datadrive-2/data/deco_test/sheets_json_feaures/' + filename + '---' + sheetname + '.json'):
                # if os.path.exists('/datadrive-2/data/fuste_test/sheets_json_feaures/' + filename + '---' + sheetname+'.json'):
                print('exists: sheets_json_feaures')
                continue

            start_row = 1
            end_row = start_row + 100
            start_col = 1
            end_col = start_col + 100

            res1 = {}
            res = []
            for sheet_dict in workbook_info['Sheets']:
                if sheet_dict['Name'] == sheetname:
                    sheet_info = sheet_dict

            for row_json in sheet_info['Rows']:
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
                            # print('v in ')
                            keyw = list(cell['V'].keys())[0]
                            content = str(cell['V'][keyw])
                            content_template = get_template(content, keyw)
                            new_cell['content'] = content
                            new_cell['content_template'] = content_template
                        # res.append(new_cell)
                        # print('new_cell', new_cell)
                        res1[str(cell['R']) + '---' + str(cell['C'])] = new_cell

            for row in range(start_row, start_row + 100):
                for col in range(start_col, start_col + 10):

                    if str(row) + '---' + str(col) in res1:
                        if is_look:
                            res1[str(row) + '---' + str(col)]['row'] = row
                            res1[str(row) + '---' + str(col)]['col'] = col

                        res.append(res1[str(row) + '---' + str(col)])

                    else:
                        if is_look:
                            new_cell = copy.deepcopy(one_invalid_cell[0])
                            new_cell['row'] = row
                            new_cell['col'] = col
                            res.append(new_cell)
                        else:
                            res.append(one_invalid_cell)
            print('start saving ....')
            with open('/datadrive-2/data/deco_test/sheets_json_feaures/' + filename + '---' + sheetname + '.json',
                      'w') as f:
                # with open('/datadrive-2/data/fuste_test/sheets_json_feaures/' + filename + '---' + sheetname +'.json', 'w') as f:
                json.dump(res, f)


def generate_sheet_features_by_workbook_json(thread_id, batch_num):
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

    invalid_cell_feature = {
        "background_color_r": 0,
        "background_color_g": 0,
        "background_color_b": 0,
        "font_color_r": 0,
        "font_color_g": 0,
        "font_color_b": 0,
        "font_size": 0.0,
        "font_strikethrough": False,
        "font_shadow": False,
        "font_ita": False,
        "font_bold": False,
        "height": 0.0,
        "width": 0.0,
        "content": '',
        "content_template": '',
    },

    # with open("Formulas_training_with_id.json",'r') as f:
    #     formulas = json.load(f)

    # with open("Formulas_middle10domain_with_id.json",'r') as f:
    #     formulas = json.load(f)
    with open("Formulas_fortune500_with_id.json", 'r') as f:
        formulas = json.load(f)

    batch_len = len(formulas) / batch_num
    for index, formula in enumerate(formulas):
        if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
            continue
        filesheet = formula['filesheet']
        filename = filesheet.split('---')[0].split('/')[-1]
        sheetname = filesheet.split('---')[1]

        fr = 1
        fc = 1
        # if not os.path.exists('other_training_formulas/' + filename + '---' + sheetname + '---' + str(formula['fr']) + '---' + str(formula['fc']) + '.json'):
        #     continue
        print('filename', filename)
        if not os.path.exists(root_path + 'origin_sheet_features/' + filename + '---' + sheetname + '.json'):
            print('not exists: middle10domain_origin_sheet_features')
            continue
        if not os.path.exists('../Demo/origin_fortune500_workbook_json/' + filename + '.json'):
            print('not exists: origin_fortune500_workbook_json')
            continue
        # if os.path.exists('fixed_formulas_training/' + filename + '---' + sheetname  + '---' + str(fr) + '---' + str(fc)+'.json'):
        #     continue
        if os.path.exists(root_path + 'sheets_json_feaures/' + filename + '---' + sheetname + '.json'):
            print('exists: sheets_json_feaures')
            continue
        print(index, len(formulas))
        print(formula)
        print('start load file......')
        # with open('../Demo/fixed_workbook_json/'+filename+'.json', 'r') as f:
        #     workbook_info = json.load(f)
        with open('../Demo/origin_fortune500_workbook_json/' + filename + '.json', 'r') as f:
            workbook_info = json.load(f)
        # with open('other_training_formulas/' +  filename + '---' + sheetname + '---' + str(fr) + '---' + str(fc) + '.json', 'r') as f:
        #     origin_feature = json.load(f)
        with open(root_path + 'origin_sheet_features/' + filename + '---' + sheetname + '.json', 'r') as f:
            origin_feature = json.load(f)

        if fr - 50 >= 1:
            start_row = fr - 50
        else:
            start_row = 1

        if fc - 5 >= 1:
            start_column = fc - 5
        else:
            start_column = 1

        feature_list = origin_feature['sheetfeature']

        table_feature = {}
        count = 0
        for row in range(start_row, start_row + 100):
            table_feature[row] = {}
            for col in range(start_column, start_column + 10):
                table_feature[row][col] = feature_list[count]
                count += 1

        for sheet_dict in workbook_info['Sheets']:
            if sheet_dict['Name'] == sheetname:
                sheet_info = sheet_dict

        new_start_row = 1
        new_start_column = 1
        new_feature_list = []
        valid_count = 0
        print('start found row......')
        for row in range(new_start_row, new_start_row + 100):
            found_row = False
            for row_dict in sheet_info['Rows']:
                if row_dict['Row'] == row:
                    found_row = True
                    row_info = row_dict

            for col in range(new_start_column, new_start_column + 10):
                if row <= 0 or col <= 0:
                    new_feature_list.append(invalid_cell_feature)
                else:
                    valid_count += 1
                    if not found_row:
                        cell_value = ''
                        cell_type = 'S'
                    else:
                        # print('found_row')
                        found_col = False
                        for cell_dict in row_info['Cells']:
                            # print(col, cell_dict['C'])
                            if cell_dict['C'] == col:
                                found_col = True
                                cell_info = cell_dict
                        if not found_col:
                            cell_value = ''
                            cell_type = 'S'
                        else:
                            # print('found col')
                            # print('cell_info', cell_info)
                            if 'V' in cell_info:
                                # print('V in cell_info')
                                # print(row, col)
                                cell_type = list(cell_info['V'].keys())[0]
                                cell_value = str(cell_info['V'][cell_type])
                            else:
                                cell_value = ''
                                cell_type = 'S'
                    feature_template = get_template(cell_value, cell_type)
                    table_feature[row][col]['content'] = cell_value
                    table_feature[row][col]['content_template'] = feature_template
                    new_feature_list.append(table_feature[row][col])
                    # print(table_feature[row][col])
        origin_feature['sheetfeature'] = new_feature_list
        print('start saving ....')
        # with open('fixed_formulas_training/' + filename + '---' + sheetname  + '---' + str(fr) + '---' + str(fc)+'.json', 'w') as f:
        #     json.dump(origin_feature, f)
        with open(root_path + 'sheets_json_feaures/' + filename + '---' + sheetname + '.json', 'w') as f:
            json.dump(origin_feature, f)


def generate_formula_features_by_workbook_json(thread_id, batch_num):
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

    invalid_cell_feature = {
        "background_color_r": 0,
        "background_color_g": 0,
        "background_color_b": 0,
        "font_color_r": 0,
        "font_color_g": 0,
        "font_color_b": 0,
        "font_size": 0.0,
        "font_strikethrough": False,
        "font_shadow": False,
        "font_ita": False,
        "font_bold": False,
        "height": 0.0,
        "width": 0.0,
        "content": '',
        "content_template": '',
    },

    # with open("Formulas_training_with_id.json",'r') as f:
    #     formulas = json.load(f)

    # with open("Formulas_middle10domain_with_id.json",'r') as f:
    with open("Formulas_fortune500_with_id.json", 'r') as f:
        formulas = json.load(f)

    batch_len = len(formulas) / batch_num
    for index, formula in enumerate(formulas):
        if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
            continue
        filesheet = formula['filesheet']
        filename = filesheet.split('---')[0].split('/')[-1]
        sheetname = filesheet.split('---')[1]

        fr = formula['fr']
        fc = formula['fc']
        # if not os.path.exists('other_training_formulas/' + filename + '---' + sheetname + '---' + str(formula['fr']) + '---' + str(formula['fc']) + '.json'):
        #     continue
        # filename = filename.replace('')
        print('../Demo/fix_fortune500/' + filename + '---' + sheetname + '---' + str(formula['fr']) + '---' + str(
            formula['fc']) + '.json')
        if not os.path.exists(
                '../Demo/fix_fortune500/' + filename + '---' + sheetname + '---' + str(formula['fr']) + '---' + str(
                    formula['fc']) + '.json'):
            continue
        print("XXXXXX")
        if not os.path.exists('../Demo/fix_fortune500/' + filename + '.json'):
            continue
        # if os.path.exists('fixed_formulas_training/' + filename + '---' + sheetname  + '---' + str(fr) + '---' + str(fc)+'.json'):
        #     continue
        # if os.path.exists(root_path + 'formulas_json_feaures/' + filename + '---' + sheetname  + '---' + str(fr) + '---' + str(fc)+'.json'):
        # continue
        print(index, len(formulas))
        print(formula)
        print('start load file......')
        start_time = time.time()
        # with open('../Demo/fixed_workbook_json/'+filename+'.json', 'r') as f:
        #     workbook_info = json.load(f)
        with open('../Demo/fix_fortune500/' + filename + '.json', 'r') as f:
            workbook_info = json.load(f)
        # with open('other_training_formulas/' +  filename + '---' + sheetname + '---' + str(fr) + '---' + str(fc) + '.json', 'r') as f:
        #     origin_feature = json.load(f)
        with open(root_path + 'origin_fortune500_formula_features/' + filename + '---' + sheetname + '---' + str(
                fr) + '---' + str(fc) + '.json', 'r') as f:
            origin_feature = json.load(f)

        if fr - 50 >= 1:
            start_row = fr - 50
        else:
            start_row = 1

        if fc - 5 >= 1:
            start_column = fc - 5
        else:
            start_column = 1

        feature_list = origin_feature['sheetfeature']

        table_feature = {}
        count = 0
        for row in range(start_row, start_row + 100):
            table_feature[row] = {}
            for col in range(start_column, start_column + 10):
                table_feature[row][col] = feature_list[count]
                count += 1

        for sheet_dict in workbook_info['Sheets']:
            if sheet_dict['Name'] == sheetname:
                sheet_info = sheet_dict

        new_start_row = formula['fr'] - 50
        new_start_column = formula['fc'] - 5
        new_feature_list = []
        valid_count = 0
        print('start found row......')
        for row in range(new_start_row, new_start_row + 100):
            found_row = False
            for row_dict in sheet_info['Rows']:
                if row_dict['Row'] == row:
                    found_row = True
                    row_info = row_dict

            for col in range(new_start_column, new_start_column + 10):
                if row <= 0 or col <= 0:
                    new_feature_list.append(invalid_cell_feature)
                else:
                    valid_count += 1
                    if not found_row:
                        cell_value = ''
                        cell_type = 'S'
                    else:
                        # print('found_row')
                        found_col = False
                        for cell_dict in row_info['Cells']:
                            # print(col, cell_dict['C'])
                            if cell_dict['C'] == col:
                                found_col = True
                                cell_info = cell_dict
                        if not found_col:
                            cell_value = ''
                            cell_type = 'S'
                        else:
                            # print('found col')
                            # print('cell_info', cell_info)
                            if 'V' in cell_info:
                                # print('V in cell_info')
                                # print(row, col)
                                cell_type = list(cell_info['V'].keys())[0]
                                cell_value = str(cell_info['V'][cell_type])
                            else:
                                cell_value = ''
                                cell_type = 'S'
                    feature_template = get_template(cell_value, cell_type)
                    table_feature[row][col]['content'] = cell_value
                    table_feature[row][col]['content_template'] = feature_template
                    new_feature_list.append(table_feature[row][col])
                    # print(table_feature[row][col])
        origin_feature['sheetfeature'] = new_feature_list
        print('start saving ....')
        # with open('fixed_formulas_training/' + filename + '---' + sheetname  + '---' + str(fr) + '---' + str(fc)+'.json', 'w') as f:
        #     json.dump(origin_feature, f)
        with open(root_path + 'formulas_json_feaures/' + filename + '---' + sheetname + '---' + str(fr) + '---' + str(
                fc) + '.json', 'w') as f:
            json.dump(origin_feature, f)
        end_time = time.time()
        print(end_time - start_time)
        break


def generate_ref_features_by_workbook_json(thread_id, batch_num):
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

    invalid_cell_feature = {
        "background_color_r": 0,
        "background_color_g": 0,
        "background_color_b": 0,
        "font_color_r": 0,
        "font_color_g": 0,
        "font_color_b": 0,
        "font_size": 0.0,
        "font_strikethrough": False,
        "font_shadow": False,
        "font_ita": False,
        "font_bold": False,
        "height": 0.0,
        "width": 0.0,
        "content": '',
        "content_template": '',
    },

    # with open("Formulas_training_with_id.json",'r') as f:
    #     formulas = json.load(f)

    # with open("Formulas_middle10domain_with_id.json",'r') as f:
    formula_token_files = os.listdir(root_path + "refcell_features")

    batch_len = len(formula_token_files) / batch_num
    for index, formula_token_file in enumerate(formula_token_files):
        if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
            continue

        filename = formula_token_file.split('---')[0].split('/')[-1]
        sheetname = formula_token_file.split('---')[1]

        fr = int(formula_token_file.split('---')[2])
        fc = int(formula_token_file.split('---')[3].replace('.json', ''))
        # if not os.path.exists('other_training_formulas/' + filename + '---' + sheetname + '---' + str(formula['fr']) + '---' + str(formula['fc']) + '.json'):
        #     continue
        # filename = filename.replace('')
        print('formula_token_file', formula_token_file)
        if not os.path.exists(root_path + 'refcell_features/' + formula_token_file):
            continue
        print("XXXXXX")
        print('filename', filename)
        if not os.path.exists('../Demo/origin_fortune500_workbook_json/' + filename + '.json'):
            continue
        # if os.path.exists('fixed_formulas_training/' + filename + '---' + sheetname  + '---' + str(fr) + '---' + str(fc)+'.json'):
        #     continue
        if os.path.exists(root_path + 'refcell_json_features/' + formula_token_file):
            continue
        print(index, len(formula_token_files))
        print('start load file......')
        # with open('../Demo/fixed_workbook_json/'+filename+'.json', 'r') as f:
        #     workbook_info = json.load(f)
        with open('../Demo/origin_fortune500_workbook_json/' + filename + '.json', 'r') as f:
            workbook_info = json.load(f)
        # with open('other_training_formulas/' +  filename + '---' + sheetname + '---' + str(fr) + '---' + str(fc) + '.json', 'r') as f:
        #     origin_feature = json.load(f)
        with open(root_path + 'refcell_features/' + formula_token_file, 'r') as f:
            origin_feature = json.load(f)

        if fr - 50 >= 1:
            start_row = fr - 50
        else:
            start_row = 1

        if fc - 5 >= 1:
            start_column = fc - 5
        else:
            start_column = 1

        feature_list = origin_feature['sheetfeature']

        table_feature = {}
        count = 0
        for row in range(start_row, start_row + 100):
            table_feature[row] = {}
            for col in range(start_column, start_column + 10):
                table_feature[row][col] = feature_list[count]
                count += 1

        for sheet_dict in workbook_info['Sheets']:
            if sheet_dict['Name'] == sheetname:
                sheet_info = sheet_dict

        new_start_row = fr - 50
        new_start_column = fc - 5
        new_feature_list = []
        valid_count = 0
        print('start found row......')
        for row in range(new_start_row, new_start_row + 100):
            found_row = False
            for row_dict in sheet_info['Rows']:
                if row_dict['Row'] == row:
                    found_row = True
                    row_info = row_dict

            for col in range(new_start_column, new_start_column + 10):
                if row <= 0 or col <= 0:
                    new_feature_list.append(invalid_cell_feature)
                else:
                    valid_count += 1
                    if not found_row:
                        cell_value = ''
                        cell_type = 'S'
                    else:
                        # print('found_row')
                        found_col = False
                        for cell_dict in row_info['Cells']:
                            # print(col, cell_dict['C'])
                            if cell_dict['C'] == col:
                                found_col = True
                                cell_info = cell_dict
                        if not found_col:
                            cell_value = ''
                            cell_type = 'S'
                        else:
                            # print('found col')
                            # print('cell_info', cell_info)
                            if 'V' in cell_info:
                                # print('V in cell_info')
                                # print(row, col)
                                cell_type = list(cell_info['V'].keys())[0]
                                cell_value = str(cell_info['V'][cell_type])
                            else:
                                cell_value = ''
                                cell_type = 'S'
                    feature_template = get_template(cell_value, cell_type)
                    table_feature[row][col]['content'] = cell_value
                    table_feature[row][col]['content_template'] = feature_template
                    new_feature_list.append(table_feature[row][col])
                    # print(table_feature[row][col])
        origin_feature['sheetfeature'] = new_feature_list
        print('start saving ....')
        # with open('fixed_formulas_training/' + filename + '---' + sheetname  + '---' + str(fr) + '---' + str(fc)+'.json', 'w') as f:
        #     json.dump(origin_feature, f)
        with open(root_path + 'refcell_json_features/' + formula_token_file, 'w') as f:
            json.dump(origin_feature, f)


def generate_tile_features_by_workbook_json(thread_id, batch_num):
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

    invalid_cell_feature = {
        "background_color_r": 0,
        "background_color_g": 0,
        "background_color_b": 0,
        "font_color_r": 0,
        "font_color_g": 0,
        "font_color_b": 0,
        "font_size": 0.0,
        "font_strikethrough": False,
        "font_shadow": False,
        "font_ita": False,
        "font_bold": False,
        "height": 0.0,
        "width": 0.0,
        "content": '',
        "content_template": '',
    },

    # with open("Formulas_training_with_id.json",'r') as f:
    #     formulas = json.load(f)

    # with open("Formulas_middle10domain_with_id.json",'r') as f:
    formula_token_files = os.listdir(root_path + "tile_features")

    batch_len = len(formula_token_files) / batch_num
    for index, formula_token_file in enumerate(formula_token_files):
        if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
            continue

        filename = formula_token_file.split('---')[0].split('/')[-1]
        sheetname = formula_token_file.split('---')[1]

        fr = int(formula_token_file.split('---')[2])
        fc = int(formula_token_file.split('---')[3].replace('.json', ''))
        # if not os.path.exists('other_training_formulas/' + filename + '---' + sheetname + '---' + str(formula['fr']) + '---' + str(formula['fc']) + '.json'):
        #     continue
        # filename = filename.replace('')
        print('formula_token_file', formula_token_file)
        if not os.path.exists(root_path + 'tile_features/' + formula_token_file):
            continue
        print("XXXXXX")
        if not os.path.exists('../Demo/origin_fortune500_workbook_json/' + filename + '.json'):
            continue
        # if os.path.exists('fixed_formulas_training/' + filename + '---' + sheetname  + '---' + str(fr) + '---' + str(fc)+'.json'):
        #     continue
        if os.path.exists(root_path + 'tile_json_feaures/' + formula_token_file):
            continue
        print(index, len(formula_token_files))
        print('start load file......')
        # with open('../Demo/fixed_workbook_json/'+filename+'.json', 'r') as f:
        #     workbook_info = json.load(f)
        with open('../Demo/origin_fortune500_workbook_json/' + filename + '.json', 'r') as f:
            workbook_info = json.load(f)
        # with open('other_training_formulas/' +  filename + '---' + sheetname + '---' + str(fr) + '---' + str(fc) + '.json', 'r') as f:
        #     origin_feature = json.load(f)
        with open(root_path + 'tile_features/' + formula_token_file, 'r') as f:
            origin_feature = json.load(f)

        if fr - 50 >= 1:
            start_row = fr - 50
        else:
            start_row = 1

        if fc - 5 >= 1:
            start_column = fc - 5
        else:
            start_column = 1

        feature_list = origin_feature['sheetfeature']

        table_feature = {}
        count = 0
        for row in range(start_row, start_row + 100):
            table_feature[row] = {}
            for col in range(start_column, start_column + 10):
                table_feature[row][col] = feature_list[count]
                count += 1

        for sheet_dict in workbook_info['Sheets']:
            if sheet_dict['Name'] == sheetname:
                sheet_info = sheet_dict

        new_start_row = fr - 50
        new_start_column = fc - 5
        new_feature_list = []
        valid_count = 0
        print('start found row......')
        for row in range(new_start_row, new_start_row + 100):
            found_row = False
            for row_dict in sheet_info['Rows']:
                if row_dict['Row'] == row:
                    found_row = True
                    row_info = row_dict

            for col in range(new_start_column, new_start_column + 10):
                if row <= 0 or col <= 0:
                    new_feature_list.append(invalid_cell_feature)
                else:
                    valid_count += 1
                    if not found_row:
                        cell_value = ''
                        cell_type = 'S'
                    else:
                        # print('found_row')
                        found_col = False
                        for cell_dict in row_info['Cells']:
                            # print(col, cell_dict['C'])
                            if cell_dict['C'] == col:
                                found_col = True
                                cell_info = cell_dict
                        if not found_col:
                            cell_value = ''
                            cell_type = 'S'
                        else:
                            # print('found col')
                            # print('cell_info', cell_info)
                            if 'V' in cell_info:
                                # print('V in cell_info')
                                # print(row, col)
                                cell_type = list(cell_info['V'].keys())[0]
                                cell_value = str(cell_info['V'][cell_type])
                            else:
                                cell_value = ''
                                cell_type = 'S'
                    feature_template = get_template(cell_value, cell_type)
                    table_feature[row][col]['content'] = cell_value
                    table_feature[row][col]['content_template'] = feature_template
                    new_feature_list.append(table_feature[row][col])
                    # print(table_feature[row][col])
        origin_feature['sheetfeature'] = new_feature_list
        print('start saving ....')
        # with open('fixed_formulas_training/' + filename + '---' + sheetname  + '---' + str(fr) + '---' + str(fc)+'.json', 'w') as f:
        #     json.dump(origin_feature, f)
        with open(root_path + 'tile_json_feaures/' + formula_token_file, 'w') as f:
            json.dump(origin_feature, f)


def generate_neighbors_features_by_workbook_json(thread_id, batch_num):
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

    invalid_cell_feature = {
        "background_color_r": 0,
        "background_color_g": 0,
        "background_color_b": 0,
        "font_color_r": 0,
        "font_color_g": 0,
        "font_color_b": 0,
        "font_size": 0.0,
        "font_strikethrough": False,
        "font_shadow": False,
        "font_ita": False,
        "font_bold": False,
        "height": 0.0,
        "width": 0.0,
        "content": '',
        "content_template": '',
    },

    filelist = os.listdir('/datadrive-2/data/neighbors/')
    filelist.sort()
    batch_len = len(filelist) / batch_num
    for index, filename in enumerate(filelist):
        if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
            continue
        token = filename.replace('.json', '')
        filesheet = token.split('---')[0]
        filename = token.split('---')[0].split('/')[-1]
        sheetname = token.split('---')[1]

        fr = int(token.split('---')[2])
        fc = int(token.split('---')[3])

        if os.path.exists('fixed_neighbors_training/' + filename + '---' + sheetname + '---' + str(fr) + '---' + str(
                fc) + '.json'):
            continue
        print(index, len(filelist))
        print('start load file......')
        if not os.path.exists('../Demo/fixed_workbook_json/' + filename + '.json'):
            continue
        with open('../Demo/fixed_workbook_json/' + filename + '.json', 'r') as f:
            workbook_info = json.load(f)
        with open('/datadrive-2/data/neighbors/' + filename + '---' + sheetname + '---' + str(fr) + '---' + str(
                fc) + '.json', 'r') as f:
            origin_feature = json.load(f)

        if fr - 50 >= 1:
            start_row = fr - 50
        else:
            start_row = 1

        if fc - 5 >= 1:
            start_column = fc - 5
        else:
            start_column = 1

        feature_list = origin_feature['sheetfeature']

        table_feature = {}
        count = 0
        for row in range(start_row, start_row + 100):
            table_feature[row] = {}
            for col in range(start_column, start_column + 10):
                table_feature[row][col] = feature_list[count]
                count += 1

        for sheet_dict in workbook_info['Sheets']:
            if sheet_dict['Name'] == sheetname:
                sheet_info = sheet_dict

        new_start_row = fr - 50
        new_start_column = fc - 5
        new_feature_list = []
        valid_count = 0
        print('start found row......')
        for row in range(new_start_row, new_start_row + 100):
            found_row = False
            for row_dict in sheet_info['Rows']:
                if row_dict['Row'] == row:
                    found_row = True
                    row_info = row_dict

            for col in range(new_start_column, new_start_column + 10):
                if row <= 0 or col <= 0:
                    new_feature_list.append(invalid_cell_feature)
                else:
                    valid_count += 1
                    if not found_row:
                        cell_value = ''
                        cell_type = 'S'
                    else:
                        # print('found_row')
                        found_col = False
                        for cell_dict in row_info['Cells']:
                            # print(col, cell_dict['C'])
                            if cell_dict['C'] == col:
                                found_col = True
                                cell_info = cell_dict
                        if not found_col:
                            cell_value = ''
                            cell_type = 'S'
                        else:
                            # print('found col')
                            # print('cell_info', cell_info)
                            if 'V' in cell_info:
                                # print('V in cell_info')
                                # print(row, col)
                                cell_type = list(cell_info['V'].keys())[0]
                                cell_value = str(cell_info['V'][cell_type])
                            else:
                                cell_value = ''
                                cell_type = 'S'
                    feature_template = get_template(cell_value, cell_type)
                    table_feature[row][col]['content'] = cell_value
                    table_feature[row][col]['content_template'] = feature_template
                    new_feature_list.append(table_feature[row][col])
                    # print(table_feature[row][col])
        origin_feature['sheetfeature'] = new_feature_list
        print('start saving ....')
        with open('fixed_neighbors_training/' + filename + '---' + sheetname + '---' + str(fr) + '---' + str(
                fc) + '.json', 'w') as f:
            json.dump(origin_feature, f)
        # break


def para_run_sheet_feature():
    process = [Process(target=generate_sheet_features_by_workbook_json, args=(1, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(2, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(3, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(4, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(5, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(6, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(7, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(8, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(9, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(10, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(11, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(12, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(13, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(14, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(15, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(16, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(17, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(18, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(19, 20)),
               Process(target=generate_sheet_features_by_workbook_json, args=(20, 20)),
               ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束


def para_run():
    process = [Process(target=generate_formula_features_by_workbook_json, args=(1, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(2, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(3, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(4, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(5, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(6, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(7, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(8, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(9, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(10, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(11, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(12, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(13, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(14, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(15, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(16, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(17, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(18, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(19, 20)),
               Process(target=generate_formula_features_by_workbook_json, args=(20, 20)),
               ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束


def para_tile_run():
    process = [Process(target=generate_tile_features_by_workbook_json, args=(1, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(2, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(3, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(4, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(5, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(6, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(7, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(8, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(9, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(10, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(11, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(12, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(13, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(14, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(15, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(16, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(17, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(18, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(19, 20)),
               Process(target=generate_tile_features_by_workbook_json, args=(20, 20)),
               ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束


def para_refcell_run():
    process = [Process(target=generate_ref_features_by_workbook_json, args=(1, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(2, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(3, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(4, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(5, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(6, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(7, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(8, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(9, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(10, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(11, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(12, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(13, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(14, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(15, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(16, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(17, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(18, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(19, 20)),
               Process(target=generate_ref_features_by_workbook_json, args=(20, 20)),
               ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束


def para_neighbors_run():
    process = [Process(target=generate_neighbors_features_by_workbook_json, args=(1, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(2, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(3, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(4, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(5, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(6, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(7, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(8, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(9, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(10, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(11, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(12, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(13, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(14, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(15, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(16, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(17, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(18, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(19, 20)),
               Process(target=generate_neighbors_features_by_workbook_json, args=(20, 20)),
               ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束


def look_remove_formulas():
    with open("Formulas_77772sheets_mergerange_custom.json", 'r') as f:
        formulas = json.load(f)
    with open('../AnalyzeDV/sampled_file.json', 'r') as f:
        sampled_file = json.load(f)
    for filesheet in formulas:
        filename = filesheet.split('---')[0]
        sheetname = filesheet.split('---')[1]
        filename = filename.split('/')[-1]
        found = False
        for item in sampled_file:
            if filename in item:
                found = True
                break
        if not found:
            continue
        # if not os.path.exists("../AnalyzeDV/remove_formulas_200/"+filename):
        #     continue

        for r1c1 in formulas[filesheet]:
            # if 'SUM(' not in r1c1 and 'AVERAGE(' not in r1c1:
            #     continue
            # if len(r1c1) > 20:
            # continue
            if 'AVERAGE(' not in r1c1:
                continue
            for id_ in formulas[filesheet][r1c1]:
                print("#########")
                print('filename', filename)
                print('sheetname', sheetname)
                print('r1c1', r1c1)
                print('fr', formulas[filesheet][r1c1][id_]['fr'])
                print('fc', formulas[filesheet][r1c1][id_]['fc'])
            break


def delete_no_func_formulas(thread_id, batch_num):
    with open('Formula_77772_with_id.json', 'r') as f:
        formulas = json.load(f)
    with open('formula_token_2_template_id_custom.json', 'r') as f:
        formula_token_2_template_id = json.load(f)
    need_delete_id = [113, 71, 51, 134, 19, 459, 10, 84]
    result = []
    batch_len = len(formulas) / batch_num
    for index, formula in enumerate(formulas):
        if thread_id != batch_num:
            if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1):
                continue
        print(index, len(formulas))
        filesheet = formula['filesheet'].split('/')[5]
        need_remove = False
        formula_token = filesheet + '---' + str(formula['fr']) + '---' + str(formula['fc'])
        template_id = formula_token_2_template_id[formula_token]
        if template_id in need_delete_id:
            continue
        result.append(formula)
    print('len(result', len(result))
    with open("Formula_hasfunc_with_id.json", 'w') as f:
        json.dump(result, f)


def no_saved_filesheet():
    root_path = "/datadrive/projects/AnalyzeDV/origin_top10domain/"
    filelist = os.listdir(root_path)

    all_res = []
    for index, filename in enumerate(filelist):
        if 'origin_middle10domain_formulas' not in filename:
            continue
        print(index, len(filelist))
        batch_id = int(filename.split('.')[0].split('_')[-1])

        path = root_path + filename
        if batch_id > 57:
            print('rm ' + str(batch_id))
            os.system('rm ' + path)
            continue
        print('load...')
        with open(path, 'r') as f:
            jsonfile = json.load(f)
        for item in jsonfile:
            all_res.append(item)

    with open(root_path + 'saved_filesheet.json', 'w') as f:
        json.dump(all_res, f)


def generate_training_files():
    with open("domain2num.json", 'r') as f:
        domain2num = json.load(f)
    with open("filename2domain.json", 'r') as f:
        filename2domain = json.load(f)
    res = []
    res1 = []

    select_domain_num = int(len(domain2num) / 10)
    domian_list = []
    for index, domain in enumerate(domain2num):
        if index == 10:
            continue
        if len(domian_list) == select_domain_num and domain not in domian_list:
            continue
        if domain not in domian_list:
            domian_list.append(domain)

        for filename in filename2domain:
            if filename2domain[filename] == domain:
                res.append(filename)

    # print(len(res))
    # print(len(res1))
    with open('top10%_domain_filenames.json', 'w') as f:
        json.dump(res, f)
    # with open('training_filenames.json', 'w') as f:
    #     json.dump(res1, f)


def generate_one_before_features(filename, sheetname, row, col, source_root_path, saved_root_path, bert_dict=None,
                                 content_tem_dict=None):
    if bert_dict is None:
        bert_dict = {}
    if content_tem_dict is None:
        content_tem_dict = {}
    bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    formula_token = filename + '---' + sheetname + '---' + str(row) + '---' + str(col)
    if not os.path.exists(saved_root_path + formula_token + '.npy'):
        with open(source_root_path + formula_token + '.json', 'r') as f:
            origin_feature = json.load(f)
            feature_nparray = np.array(
                get_feature_vector_with_bert_keyw(origin_feature['sheetfeature'], content_tem_dict=content_tem_dict,
                                                  bert_dict=bert_dict))
            np.save(saved_root_path + formula_token + '.npy', feature_nparray)


def generate_refcell_before_features(bert_dict, content_tem_dict, thread_id, batch_num):
    bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    formula_tokens = os.listdir(root_path + 'refcell_json_features/')
    batch_len = len(formula_tokens) / batch_num
    for index, formula_token_path in enumerate(formula_tokens):
        formula_token = formula_token_path.replace(".json", '')
        if thread_id != batch_num:
            if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1):
                continuedemo
        print(index, len(formula_tokens))
        saved_root_path = root_path + 'before_features/'
        source_root_path = root_path + 'refcell_json_features/'
        if not os.path.exists(saved_root_path + formula_token + '.npy'):
            try:
                with open(source_root_path + formula_token_path, 'r') as f:
                    origin_feature = json.load(f)
                    feature_nparray = np.array(get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']),
                                               bert_dict=bert_dict, content_tem_dict=content_tem_dict)
                    np.save(saved_root_path + formula_token + '.npy', feature_nparray)
            except:
                continue


def generate_tile_before_features(bert_dict, content_tem_dict, thread_id, batch_num):
    bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    formula_tokens = os.listdir(root_path + 'tile_json_feaures/')
    batch_len = len(formula_tokens) / batch_num
    for index, formula_token_path in enumerate(formula_tokens):
        formula_token = formula_token_path.replace(".json", '')
        # if thread_id != batch_num:
        #     if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
        #         continue
        # else:
        #     if index <= batch_len * (thread_id - 1 ):
        #         continue
        print(index, len(formula_tokens))
        saved_root_path = root_path + 'tile_before_features/'
        source_root_path = root_path + 'tile_json_feaures/'
        if not os.path.exists(saved_root_path + formula_token + '.npy'):
            try:
                with open(source_root_path + formula_token_path, 'r') as f:
                    origin_feature = json.load(f)
                    feature_nparray = np.array(get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
                    np.save(saved_root_path + formula_token + '.npy', feature_nparray)
            except:
                continue


def generate_similarsheet_before_features(bert_dict, content_tem_dict, thread_id, batch_num):
    bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    formula_tokens = os.listdir(root_path + 'sheets_json_feaures/')
    batch_len = len(formula_tokens) / batch_num
    for index, formula_token_path in enumerate(formula_tokens):
        formula_token = formula_token_path.replace(".json", '')
        if thread_id != batch_num:
            if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1):
                continue
        # print(index, len(formula_tokens))
        saved_root_path = root_path + 'similarsheet_before_features/'
        source_root_path = root_path + 'sheets_json_feaures/'
        if not os.path.exists(saved_root_path + formula_token + '.npy'):
            print('not exists')
            print('formula_token', formula_token)
            with open(source_root_path + formula_token_path, 'r') as f:
                try:
                    origin_feature = json.load(f)

                    feature_nparray = np.array(get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
                    np.save(saved_root_path + formula_token + '.npy', feature_nparray)
                except:
                    print('load fail')
                    continue


def para_gen_similarsheet_before_feature():
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)

    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)

    process = [Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 1, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 2, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 3, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 4, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 5, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 6, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 7, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 8, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 9, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 10, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 11, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 12, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 13, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 14, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 15, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 16, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 17, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 18, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 9, 20)),
               Process(target=generate_similarsheet_before_features, args=(bert_dict, content_tem_dict, 20, 20)),
               ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束


def para_gen_before_feature():
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)

    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)

    process = [Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 1, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 2, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 3, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 4, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 5, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 6, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 7, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 8, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 9, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 10, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 11, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 12, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 13, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 14, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 15, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 16, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 17, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 18, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 9, 20)),
               Process(target=generate_before_features, args=(bert_dict, content_tem_dict, 20, 20)),
               ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束


def para_gen_tile_before_feature():
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)

    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)

    # process = [Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 1,20)),
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 2,20)), 
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 3,20)),
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 4,20)), 
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 5,20)),
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 6,20)), 
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 7,20)),
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 8,20)), 
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 9,20)),
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 10,20)), 
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 11,20)),
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 12,20)), 
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 13,20)),
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 14,20)), 
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 15,20)),
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 16,20)), 
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 17,20)),
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 18,20)), 
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 9,20)), 
    #            Process(target=generate_tile_before_features, args=(bert_dict, content_tem_dict, 20,20)), 
    #         ]
    # [p.start() for p in process]  # 开启了两个进程
    # [p.join() for p in process]   # 等待两个进程依次结束
    generate_tile_before_features(bert_dict, content_tem_dict, 1, 1)


def para_gen_refcell_before_feature():
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)

    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)

    process = [Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 1, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 2, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 3, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 4, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 5, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 6, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 7, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 8, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 9, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 10, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 11, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 12, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 13, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 14, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 15, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 16, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 17, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 18, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 9, 20)),
               Process(target=generate_refcell_before_features, args=(bert_dict, content_tem_dict, 20, 20)),
               ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束


def save_sheet_bert_feature_for_77721(thread_id, batch_num, save_path=root_path + 'sheetfeature2afterfeature/',
                                      model_path="/datadrive-2/data/l2_model/4_1",
                                      load_path=root_path + 'similarsheet_before_features/'):  # load_path = 'deal00formula2beforebertfeature'):model_path = '196model/cnn_new_dynamic_triplet_margin_1_3_12'
    # e2e = End2End()
    # if model_path:
    model = torch.load(model_path)
    # with open('need_rerun_list.json','r') as f:
    #     need_rerun_list = json.load(f)
    # print('thread_id', thread_id, batch_num)
    filenames = os.listdir(load_path)
    # with open('res197.json', 'r') as f:
    #     res197 = json.load(f)

    saved_filenames1 = [i.replace('.npy', '.json') for i in os.listdir(save_path)]
    # saved_filenames = os.listdir('deal00formula2afterbertfeature')
    saved_filenames1 = os.listdir(save_path)
    filenames = list(set(filenames) - set(saved_filenames1))
    # filenames = res197
    print('filenames', len(filenames))
    filenames.sort()
    print('saved_filenames1', len(saved_filenames1))

    filename_list = []
    feature_list = []
    batch_len = len(filenames) / batch_num
    for index, filename in enumerate(filenames):
        # print(index, batch_len * (thread_id - 1 ), batch_len * (thread_id ))
        if thread_id != batch_num:
            if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
                continue
        else:
            if (index <= batch_len * (thread_id - 1)):
                continue

        print(index, len(filenames))

        # if filename.replace('.json', '.npy') in saved_filenames1:
        #     continue
        print("save_path + filename")
        if os.path.exists(save_path + filename.replace('.json', '') + '.npy'):
            print('exists.......')
            continue
        # print(filename)
        # print('load origin feature....')
        # if not os.path.exists(load_path + '/' + filename.replace('.json','.npy')):
        #     if not os.path.exists('formulas_deal_00/'+filename):
        #         continue
        #     with open("formulas_deal_00/" + filename, 'r') as f:
        #         origin_feature = json.load(f)
        #     # print('origin_feature', origin_feature)
        #     print(origin_feature.keys())

        #     # print(self.get_feature_vector_with_bert(origin_feature['sheetfeature']))
        #     print('transform to np array....')
        #     feature_nparray = np.array(e2e.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
        #     print(feature_nparray.shape)
        #     print('before model predict....')
        #     res =  torch.DoubleTensor(feature_nparray)
        #     feature = res.reshape(1,100,10,399)
        #     np.save(load_path + '/' + filename.replace('.json',''), feature)
        print("filename", filename)
        feature_nparray = np.load(load_path + filename.replace('.json', '.npy'))
        feature_nparray = feature_nparray.reshape(1, 100, 10, 399)
        model.eval()

        feature_nparray = torch.DoubleTensor(feature_nparray)
        feature_nparray = Variable(feature_nparray).to(torch.float32)
        feature_nparray = model(feature_nparray).detach().numpy()
        print('after model predict....')
        # print('feature_list', feature_list)
        np.save(save_path + filename.replace('.json', '') + '.npy', feature_nparray)


def save_bert_feature_for_77721(thread_id, batch_num, save_path=root_path + 'after_feature/',
                                model_path='/datadrive-2/data/l2_model/4_1',
                                load_path=root_path + 'before_features/'):  # load_path = 'deal00formula2beforebertfeature'):
    # def save_bert_feature_for_77721(thread_id, batch_num, save_path=root_path + 'tile_after_features/', model_path = '196model/cnn_new_dynamic_triplet_margin_1_3_12', load_path = root_path+'tile_before_features/' ):#load_path = 'deal00formula2beforebertfeature'):
    # def save_bert_feature_for_77721(thread_id, batch_num, save_path=root_path + 'refcell_after_features/', model_path = '196model/cnn_new_dynamic_triplet_margin_1_3_12', load_path = root_path+'refcell_before_features/' ):#load_path = 'deal00formula2beforebertfeature'):
    # def save_bert_feature_for_77721(thread_id, batch_num, save_path=root_path + 'demo_after_features/', model_path = '196model/cnn_new_dynamic_triplet_margin_1_3_12', load_path = root_path+'demo_before_features/' ):#load_path = 'deal00formula2beforebertfeature'):
    # model1_middle10domain_formula2afterfeature
    # e2e = End2End()

    # if model_path:
    model = torch.load(model_path)
    # with open('need_rerun_list.json','r') as f:
    #     need_rerun_list = json.load(f)
    # print('thread_id', thread_id, batch_num)
    filenames = os.listdir(load_path)
    # with open('res197.json', 'r') as f:
    #     res197 = json.load(f)

    # saved_filenames1 = [i.replace('.npy','.json') for i in os.listdir(save_path)]
    # saved_filenames = os.listdir('deal00formula2afterbertfeature')
    saved_filenames1 = os.listdir(save_path)
    print('before filenames', len(filenames))
    filenames = list(set(filenames) - set(saved_filenames1))
    # filenames = res197
    print('after filenames', len(filenames))
    filenames.sort()
    print('saved_filenames1', len(saved_filenames1))

    filename_list = []
    feature_list = []
    batch_len = len(filenames) / batch_num
    # print(os.path.exists('/datadrive-2/data/fortune500_test/demo_tile_features/234679781916000373063272744719263521823-8054.ddr3-phy-calc-v11-for-1600.xlsx---PHY CALC---101---1.json'))
    # print(os.path.exists('/datadrive-2/data/fortune500_test/demo_tile_features/234679781916000373063272744719263521823-8054.ddr3-phy-calc-v11-for-1600.xlsx---PHY CALC---201---1.json'))
    # print(os.path.exists(load_path + '234679781916000373063272744719263521823-8054.ddr3-phy-calc-v11-for-1600.xlsx---PHY CALC---101---1' + '.npy'))
    # print(os.path.exists(load_path + '234679781916000373063272744719263521823-8054.ddr3-phy-calc-v11-for-1600.xlsx---PHY CALC---201---1' + '.npy'))
    # print(os.path.exists(save_path + '234679781916000373063272744719263521823-8054.ddr3-phy-calc-v11-for-1600.xlsx---PHY CALC---101---1' + '.npy'))
    # print(os.path.exists(save_path + '234679781916000373063272744719263521823-8054.ddr3-phy-calc-v11-for-1600.xlsx---PHY CALC---201---1' + '.npy'))

    for index, filename in enumerate(filenames):
        print(index, batch_len * (thread_id - 1), batch_len * (thread_id))
        if thread_id != batch_num:
            if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
                continue
        else:
            if (index <= batch_len * (thread_id - 1)):
                continue

        print(index, len(filenames))

        # if filename.replace('.json', '.npy') in saved_filenames1:
        #     continue
        if os.path.exists(save_path + filename):
            print('exists.......')
            continue
        # if filename != '234679781916000373063272744719263521823-8054.ddr3-phy-calc-v11-for-1600.xlsx---PHY CALC---201---1.npy':
        # continue
        # print(filename)
        # print('load origin feature....')
        # if not os.path.exists(load_path + '/' + filename.replace('.json','.npy')):
        #     if not os.path.exists('formulas_deal_00/'+filename):
        #         continue
        #     with open("formulas_deal_00/" + filename, 'r') as f:
        #         origin_feature = json.load(f)
        #     # print('origin_feature', origin_feature)
        #     print(origin_feature.keys())

        #     # print(self.get_feature_vector_with_bert(origin_feature['sheetfeature']))
        #     print('transform to np array....')
        #     feature_nparray = np.array(e2e.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
        #     print(feature_nparray.shape)
        #     print('before model predict....')
        #     res =  torch.DoubleTensor(feature_nparray)
        #     feature = res.reshape(1,100,10,399)
        #     np.save(load_path + '/' + filename.replace('.json',''), feature)
        start_time = time.time()
        print("filename", filename)

        feature_nparray = np.load(load_path + filename.replace('.json', '.npy'))
        temp_time1 = time.time()
        print('load time:', temp_time1 - start_time)
        feature_nparray = feature_nparray.reshape(1, 100, 10, 399)
        temp_time2 = time.time()
        print('reshape time:', temp_time2 - temp_time1)
        model.eval()

        feature_nparray = torch.DoubleTensor(feature_nparray)
        feature_nparray = Variable(feature_nparray).to(torch.float32)
        feature_nparray = model(feature_nparray).detach().numpy()
        print('after model predict....')
        # print('feature_list', feature_list)
        end_time = time.time()
        print(end_time - start_time)

        np.save(save_path + filename, feature_nparray)
        with open(root_path + 'after_time/' + filename.replace(".npy", '.json'), 'w') as f:
            json.dump({'time': end_time - start_time}, f)
        # if index > 10:
        break


def para_sheet_features_save_after():
    process = [
        Process(target=save_sheet_bert_feature_for_77721, args=(1, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(2, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(3, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(4, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(5, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(6, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(7, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(8, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(9, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(10, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(11, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(12, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(13, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(14, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(15, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(16, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(17, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(18, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(19, 20)),
        Process(target=save_sheet_bert_feature_for_77721, args=(20, 20)),
    ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束


def para_save_after():
    process = [Process(target=save_bert_feature_for_77721, args=(1, 20)),
               Process(target=save_bert_feature_for_77721, args=(2, 20)),
               Process(target=save_bert_feature_for_77721, args=(3, 20)),
               Process(target=save_bert_feature_for_77721, args=(4, 20)),
               Process(target=save_bert_feature_for_77721, args=(5, 20)),
               Process(target=save_bert_feature_for_77721, args=(6, 20)),
               Process(target=save_bert_feature_for_77721, args=(7, 20)),
               Process(target=save_bert_feature_for_77721, args=(8, 20)),
               Process(target=save_bert_feature_for_77721, args=(9, 20)),
               Process(target=save_bert_feature_for_77721, args=(10, 20)),
               Process(target=save_bert_feature_for_77721, args=(11, 20)),
               Process(target=save_bert_feature_for_77721, args=(12, 20)),
               Process(target=save_bert_feature_for_77721, args=(13, 20)),
               Process(target=save_bert_feature_for_77721, args=(14, 20)),
               Process(target=save_bert_feature_for_77721, args=(15, 20)),
               Process(target=save_bert_feature_for_77721, args=(16, 20)),
               Process(target=save_bert_feature_for_77721, args=(17, 20)),
               Process(target=save_bert_feature_for_77721, args=(18, 20)),
               Process(target=save_bert_feature_for_77721, args=(19, 20)),
               Process(target=save_bert_feature_for_77721, args=(20, 20)),
               ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束


def look_sheetname2num():
    with open("Formulas_top10domain_with_id.json", 'r') as f:
        res = json.load(f)

    top10domain_sheetname2num = {}
    top10domain_r1c12num = {}
    for formula in res:
        sheetname = formula['filesheet'].split("---")[1]
        r1c1 = formula['r1c1']
        if sheetname not in top10domain_sheetname2num:
            top10domain_sheetname2num[sheetname] = 0
        if r1c1 not in top10domain_r1c12num:
            top10domain_r1c12num[r1c1] = 0
        top10domain_sheetname2num[sheetname] += 1
        top10domain_r1c12num[r1c1] += 1

    top10domain_sheetname2num = sorted(top10domain_sheetname2num.items(), key=lambda x: x[1], reverse=True)
    top10domain_r1c12num = sorted(top10domain_r1c12num.items(), key=lambda x: x[1], reverse=True)
    new_top10domain_sheetname2num = {}
    new_top10domain_r1c12num = {}

    for tuple in top10domain_sheetname2num:
        new_top10domain_sheetname2num[tuple[0]] = tuple[1]
    for tuple in top10domain_r1c12num:
        new_top10domain_r1c12num[tuple[0]] = tuple[1]
    with open('/datadrive-2/data/top10domain_test/top10domain_sheetname2num.json', 'w') as f:
        json.dump(new_top10domain_sheetname2num, f)
    with open('/datadrive-2/data/top10domain_test/top10domain_r1c12num.json', 'w') as f:
        json.dump(new_top10domain_r1c12num, f)


def generate_test_formulas():
    with open("Formulas_middle10domain_with_id.json", 'r') as f:
        #     formulas = json.load(f)
        # with open("Formulas_fortune500_with_id.json", 'r') as f:
        formulas = json.load(f)

    new_formulas = []
    sheetname2num = {}
    for formula in formulas:
        sheetname = formula['filesheet'].split("---")[1]
        r1c1 = formula['r1c1']
        if sheetname not in sheetname2num:
            sheetname2num[sheetname] = 0

        if sheetname2num[sheetname] > 100:
            continue

        new_formulas.append(formula)
        sheetname2num[sheetname] += 1

    with open("Formulas_test_middle10domain_with_id.json", 'w') as f:
        #     json.dump(new_formulas,f)
        # with open("Formulas_test_fortune500_with_id.json", 'w') as f:
        json.dump(new_formulas, f)

    # with open("Formulas_test_fortune500_with_id.json", 'r') as f:
    with open("Formulas_test_middle10domain_with_id.json", 'r') as f:
        new_formulas = json.load(f)
    print(len(new_formulas))

    for index, formula in enumerate(new_formulas):
        print(index, len(new_formulas))
        formula_token = formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
        # if os.path.exists(root_path + 'model1_middle10domain_formula2afterfeature_test/' + formula_token +'.npy'):
        #     continue
        if os.path.exists(root_path + 'after_feature_test/' + formula_token + '.npy'):
            continue

        if os.path.exists(root_path + 'after_feature/' + formula_token + '.npy.npy'):
            shutil.move(root_path + 'after_feature/' + formula_token + '.npy.npy',
                        root_path + 'after_feature/' + formula_token + '.npy')
        if not os.path.exists(root_path + 'after_feature/' + formula_token + '.npy'):
            print(formula_token)
            print('not exists')
            continue

            # break
        # shutil.move(root_path + 'model1_top10domain_formula2afterfeature/' + formula_token + '.npy.npy', root_path + 'model1_top10domain_formula2afterfeature/' + formula_token +'.npy')
        shutil.copy(root_path + 'after_feature/' + formula_token + '.npy',
                    root_path + 'after_feature_test/' + formula_token + '.npy')


def generate_demo_features(filename, sheetname, workbook_json, origin_row, origin_col, save_path, is_look=False,
                           cross=False):
    """
    根据传入的参数从Workbook JSON文件中提取特定工作表和单元格的信息，并将结果保存为JSON文件。这些提取的信息包括背景颜色、宽度、高度、字体大小、字体颜色和单元格内容等。
    :param filename:
    :param sheetname:
    :param workbook_json:
    :param origin_row:
    :param origin_col:
    :param save_path:
    :param is_look:
    :param cross:
    :return:
    """
    one_invalid_cell = {
        "background_color_r": 0,
        "background_color_g": 0,
        "background_color_b": 0,
        "font_color_r": 0,
        "font_color_g": 0,
        "font_color_b": 0,
        "font_size": 0.0,
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
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    temp_start_time = time.time()

    temp_end_time = time.time()
    # print('load workbook json:', temp_end_time-temp_start_time)
    sheet_jsons = workbook_json['Sheets']
    start_row = origin_row - 50
    end_row = origin_row + 50 - 1
    start_col = origin_col - 5
    end_col = origin_col + 5 - 1
    res = []

    res1 = {}
    temp_start_time = time.time()
    for sheet_json in sheet_jsons:
        if sheet_json['Name'] == sheetname:
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
                            # print('v in ')
                            keyw = list(cell['V'].keys())[0]
                            content = str(cell['V'][keyw])
                            content_template = get_template(content, keyw)
                            new_cell['content'] = content
                            new_cell['content_template'] = content_template
                        # res.append(new_cell)
                        # print('new_cell', new_cell)
                        res1[str(cell['R']) + '---' + str(cell['C'])] = new_cell

    temp_end_time = time.time()
    # print('first iter:', temp_end_time-temp_start_time)

    temp_start_time = time.time()
    for row in range(start_row, start_row + 100):
        for col in range(start_col, start_col + 10):

            if str(row) + '---' + str(col) in res1:
                if is_look:
                    res1[str(row) + '---' + str(col)]['row'] = row
                    res1[str(row) + '---' + str(col)]['col'] = col
                if not cross:
                    res.append(res1[str(row) + '---' + str(col)])
                else:
                    if origin_row == row or origin_col == col:
                        res.append(res1[str(row) + '---' + str(col)])
                    else:
                        res.append(one_invalid_cell)
            else:
                if is_look:
                    # print('one_invalid_cell', one_invalid_cell)
                    # print(row, col)
                    new_cell = copy.deepcopy(one_invalid_cell[0])
                    new_cell['row'] = row
                    new_cell['col'] = col
                    res.append(new_cell)
                else:
                    res.append(one_invalid_cell)

    temp_end_time = time.time()
    # print('second iter:', temp_end_time-temp_start_time)

    temp_end_time = time.time()
    with open(save_path + filename + '---' + sheetname + '---' + str(origin_row) + '---' + str(origin_col) + ".json",
              'w') as f:
        # print('save:'+ save_path + filename + '---' + sheetname + '---' + str(origin_row) + '---' + str(origin_col) + ".json")
        json.dump(res, f)


def para_tile_demo_first(save_path=root_path + 'demo_tile_features/'):
    refcell_list = os.listdir(root_path + 'tile_rows')
    filename2sheetname = {}
    for filesheet_json in refcell_list:
        # if filesheet_json != '25343295814095882086846077333982515046-ddr3-phy-calc-v11-for-1600.xlsx---PHY CALC.json':
        #     continue
        filesheet = filesheet_json.replace('.json', '')
        filename = filesheet.split('---')[0]
        sheetname = filesheet.split('---')[1]
        if filename not in filename2sheetname:
            filename2sheetname[filename] = []
        filename2sheetname[filename].append(sheetname)
    filenames = list(filename2sheetname.keys())

    def batch_generate_tile_demo_features(thread_id, batch_num):

        batch_len = int(len(filenames) / batch_num)
        count = 0

        all_num = 56144857

        for index, filename in enumerate(filenames):
            # if index != batch_num:
            #     if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
            #         continue
            # else:
            #     if index <= batch_len * (thread_id - 1 ):
            #         continue
            print(index, len(filenames))
            # if filename != '25343295814095882086846077333982515046-ddr3-phy-calc-v11-for-1600.xlsx':
            #     continue
            with open('../Demo/origin_fortune500_workbook_json/' + filename + '.json', 'r') as f:
                workbook_json = json.load(f)
            for sheetname in filename2sheetname[filename]:
                # if sheetname != 'PHY CALC':
                # continue
                filesheet = filename + '---' + sheetname
                filesheet_json = filesheet + '.json'
                print('filesheet_json', filesheet_json)
                with open(root_path + 'tile_rows/' + filesheet_json, 'r') as f:
                    tile_row = json.load(f)
                with open(root_path + 'tile_cols/' + filesheet_json, 'r') as f:
                    tile_col = json.load(f)
                # start_row, last_row, start_col, last_col = tile_range[0][0], tile_range[0][1], tile_range[1][0], tile_range[1][1]
                # with open(root_path + 'tile_row/' + filesheet_json, 'r') as f:
                #     tile_row = json.load(f)
                # with open(root_path + 'tile_col/' + filesheet_json, 'r') as f:
                #     tile_col = json.load(f)
                for row in tile_row:
                    for col in tile_col:
                        # count += 1
                        # if os.path.exists(save_path + filename + '---' + sheetname + '---' + str(row) + '---' + str(col) + ".json"):
                        # print('exist savepath.')
                        # continue
                        # print('save_path:', save_path + filename + '---' + sheetname + '---' + str(row) + '---' + str(col) + ".json")
                        # print("not exists")
                        start_time = time.time()
                        # print('start')
                        generate_demo_features(filename, sheetname, workbook_json, row, col, save_path)
                        end_time = time.time()
        print('count', count)

    batch_generate_tile_demo_features(1, 1)


def find_second_best():
    refcell_list = os.listdir(root_path + 'first_tile_res_new/')
    filename2sheetname = {}
    for filesheet_json in refcell_list:
        filesheet = filesheet_json.replace('.json', '')
        filename = filesheet.split('---')[0]
        sheetname = filesheet.split('---')[1]
        if filename not in filename2sheetname:
            filename2sheetname[filename] = []
        filename2sheetname[filename].append(sheetname)
    formula_filelists = os.listdir(root_path + 'test_refcell_position')

    # print(len(formula_filelists))

    def one_process(thread_id, batch_num):
        batch_len = int(len(formula_filelists) / batch_num)
        for index, formula_token_json in enumerate(formula_filelists):
            # if formula_token_json != "85896140957661697634686825738462695896-pg%26e-monthly-srac-b-20180110.xlsx---Option B---33---11.json":
            # continue
            if index != batch_num:
                if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
                    continue
            else:
                if index <= batch_len * (thread_id - 1):
                    continue
            formula_token = formula_token_json.replace('.json', '')
            # if index <= 15:
            # continue
            print(index, len(formula_filelists), formula_token)
            res = {}
            origin_file = formula_token.split('---')[0]
            origin_sheet = formula_token.split('---')[1]
            origin_filesheet = origin_file + '---' + origin_sheet

            if os.path.exists(root_path + 'second_tile_res_new/' + formula_token + '.npy'):
                continue

            if not os.path.exists(root_path + 'model1_res/' + formula_token_json):
                print('not exists:model1_res')
                continue
            with open(root_path + 'model1_res/' + formula_token_json, 'r') as f:
                mode1_res = json.load(f)
            found_filesheet = mode1_res[1].split('---')[0] + '---' + mode1_res[1].split('---')[1]
            found_formula_token = mode1_res[1]
            # print('mode1_res', mode1_res)
            # print('found_formula_token', found_formula_token)

            if not os.path.exists(root_path + 'test_refcell_position/' + found_formula_token + '.json'):
                print('not exists:test_refcell_position')
                continue
            with open(root_path + 'test_refcell_position/' + found_formula_token + '.json', 'r') as f:
                test_refcell_position = json.load(f)
            # print('found_formula_token', found_formula_token)
            # print('test_refcell_position', test_refcell_position)

            # print('found_sheet', found_filesheet)

            # print('test_refcell_position', test_refcell_position)
            for position in test_refcell_position:
                ref_row = position['R']
                ref_col = position['C']
                # print(ref_row, ref_col)
                if not os.path.exists(
                        root_path + 'demo_after_features/' + found_filesheet + '---' + str(ref_row) + '---' + str(
                            ref_col) + '.npy'):
                    print('not exists:demo_after_features')
                    continue
                if not os.path.exists('/datadrive-2/data/fortune500_test/first_tile_res_new/' + formula_token + '.npy'):
                    print('not exists:first_tile_res_new')
                    continue
                feature = np.load(
                    root_path + 'demo_after_features/' + found_filesheet + '---' + str(ref_row) + '---' + str(
                        ref_col) + '.npy', allow_pickle=True)
                first_tile_res_new = np.load(
                    '/datadrive-2/data/fortune500_test/first_tile_res_new/' + formula_token + '.npy',
                    allow_pickle=True).item()
                # print('res', res)
                if str(ref_row) + "---" + str(ref_col) not in first_tile_res_new:
                    print('not in first', str(ref_row) + "---" + str(ref_col))
                    continue
                if 'best_row_col' not in first_tile_res_new[str(ref_row) + "---" + str(ref_col)]:
                    pritn('no best_row_col')
                    continue
                best_row_col = first_tile_res_new[str(ref_row) + "---" + str(ref_col)]['best_row_col']
                # print('ref row col', str(ref_row)+"---"+str(ref_col))
                best_row = int(best_row_col.split('---')[0])
                best_col = int(best_row_col.split('---')[1])
                res[str(ref_row) + '---' + str(ref_col)] = first_tile_res_new[str(ref_row) + "---" + str(ref_col)]
                # print("str(ref_row) + '---' + str(ref_col)", str(ref_row) + '---' + str(ref_col))
                for row in range(best_row, best_row + 100):
                    for col in range(best_col, best_col + 10):
                        if not os.path.exists(
                                root_path + 'demo_after_features/' + origin_filesheet + '---' + str(row) + '---' + str(
                                    col) + '.npy'):
                            # print("not exists:in demo_after_features")
                            continue
                        other_feature = np.load(
                            root_path + 'demo_after_features/' + origin_filesheet + '---' + str(row) + '---' + str(
                                col) + '.npy', allow_pickle=True)
                        # print('other_feature', other_feature)
                        distance = euclidean(feature, other_feature)
                        # print('distance', distance)
                        res[str(ref_row) + '---' + str(ref_col)][str(row) + '---' + str(col)] = distance
                        if distance < res[str(ref_row) + '---' + str(ref_col)]['best_distance']:
                            res[str(ref_row) + '---' + str(ref_col)]['best_distance'] = distance
                            res[str(ref_row) + '---' + str(ref_col)]['best_row_col'] = str(row) + '---' + str(col)
                            # print(row, col, distance)
                # print('res', res)
                # print('res', res.keys())
                np.save(root_path + 'second_tile_res_new/' + formula_token + '.npy', res)
            # break

    process = [
        Process(target=one_process, args=(1, 20)),
        Process(target=one_process, args=(2, 20)),
        Process(target=one_process, args=(3, 20)),
        Process(target=one_process, args=(4, 20)),
        Process(target=one_process, args=(5, 20)),
        Process(target=one_process, args=(6, 20)),
        Process(target=one_process, args=(7, 20)),
        Process(target=one_process, args=(8, 20)),
        Process(target=one_process, args=(9, 20)),
        Process(target=one_process, args=(10, 20)),
        Process(target=one_process, args=(11, 20)),
        Process(target=one_process, args=(12, 20)),
        Process(target=one_process, args=(13, 20)),
        Process(target=one_process, args=(14, 20)),
        Process(target=one_process, args=(15, 20)),
        Process(target=one_process, args=(16, 20)),
        Process(target=one_process, args=(17, 20)),
        Process(target=one_process, args=(18, 20)),
        Process(target=one_process, args=(19, 20)),
        Process(target=one_process, args=(20, 20)),
    ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束


def naive_one_by_one(save_path=root_path + 'demo_tile_features/'):
    refcell_list = os.listdir(root_path + 'tile_rows/')
    filename2sheetname = {}
    for index, formula_token_file in enumerate(refcell_list):
        formula_token = formula_token_file.replace(".json", '')

        origin_filename = formula_token.split("---")[0]
        origin_sheetname = formula_token.split("---")[1]

        if origin_filename not in filename2sheetname:
            filename2sheetname[origin_filename] = []
        filename2sheetname[origin_filename].append(origin_sheetname)

    def batch_generate_tile_demo_features(thread_id, batch_num):

        batch_len = int(len(filename2sheetname) / batch_num)
        count = 0

        all_num = 56144857

        count = 0
        print('filename2sheetname', len(filename2sheetname))

        for index, filename in enumerate(filename2sheetname):
            if index != batch_num:
                if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
                    continue
            else:
                if index <= batch_len * (thread_id - 1):
                    continue
            print(index, len(filename2sheetname))
            with open('../Demo/origin_fortune500_workbook_json/' + filename + '.json', 'r') as f:
                workbook_json = json.load(f)
            for sheetname in filename2sheetname[filename]:
                filesheet = filename + '---' + sheetname
                start_time = time.time()
                with open(root_path + 'tile_rows/' + filename + '---' + sheetname + '.json', 'r') as f:
                    tile_rows = json.load(f)
                with open(root_path + 'tile_cols/' + filename + '---' + sheetname + '.json', 'r') as f:
                    tile_cols = json.load(f)
                max_tile_row = tile_rows[-1]
                max_tile_col = tile_cols[-1]

                extract_one_demo_feature_timelist = {}
                for row in range(1, max_tile_row + 100):
                    for col in range(1, max_tile_col + 10):
                        count += 1
                        if os.path.exists(save_path + filename + '---' + sheetname + '---' + str(row) + '---' + str(
                                col) + ".json"):
                            continue
                        tmp_stime = time.time()
                        generate_demo_features(filename, sheetname, workbook_json, row, col, save_path)
                        tmp_etime = time.time()
                        extract_one_demo_feature_timelist[str(row) + "---" + str(col)] = tmp_etime - tmp_stime
                end_time = time.time()

                time_result = {
                    'end2end': end_time - start_time,
                    'cell_level': extract_one_demo_feature_timelist
                }
                with open(root_path + "time_from_wbjson_to_demofeatures/" + filename + '---' + sheetname + '.json',
                          'w') as f:
                    json.dump(time_result, f)
        print('count', count)

        # batch_generate_tile_demo_features(1,1)

    process = [
        Process(target=batch_generate_tile_demo_features, args=(1, 20)),
        Process(target=batch_generate_tile_demo_features, args=(2, 20)),
        Process(target=batch_generate_tile_demo_features, args=(3, 20)),
        Process(target=batch_generate_tile_demo_features, args=(4, 20)),
        Process(target=batch_generate_tile_demo_features, args=(5, 20)),
        Process(target=batch_generate_tile_demo_features, args=(6, 20)),
        Process(target=batch_generate_tile_demo_features, args=(7, 20)),
        Process(target=batch_generate_tile_demo_features, args=(8, 20)),
        Process(target=batch_generate_tile_demo_features, args=(9, 20)),
        Process(target=batch_generate_tile_demo_features, args=(10, 20)),
        Process(target=batch_generate_tile_demo_features, args=(11, 20)),
        Process(target=batch_generate_tile_demo_features, args=(12, 20)),
        Process(target=batch_generate_tile_demo_features, args=(13, 20)),
        Process(target=batch_generate_tile_demo_features, args=(14, 20)),
        Process(target=batch_generate_tile_demo_features, args=(15, 20)),
        Process(target=batch_generate_tile_demo_features, args=(16, 20)),
        Process(target=batch_generate_tile_demo_features, args=(17, 20)),
        Process(target=batch_generate_tile_demo_features, args=(18, 20)),
        Process(target=batch_generate_tile_demo_features, args=(19, 20)),
        Process(target=batch_generate_tile_demo_features, args=(20, 20)),
    ]

    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束


# class GenerateTool:
#     def __init__(self):
#         self.root_path = '../data_drive/data_two/'
#         self.demo_path = 'demo_tile_features_look/'
#
#     def generate_view_json(self, filename, sheetname, row, col, workbook_feature_path, save_path, cross=False):
def generate_view_json(filename, sheetname, row, col, workbook_feature_path,
                       save_path=root_path + 'demo_tile_features_look/', cross=False):
    """
    :param filename:文件名
    :param sheetname:sheet名
    :param row:行号
    :param col:列号
    :param workbook_feature_path:保存的路径
    :param save_path:保存的路径
    :param cross:
    :return:
    """
    with open(workbook_feature_path + filename + '.json', 'r', encoding='utf-8') as f:
        workbook_json = json.load(f)
    generate_demo_features(filename, sheetname, workbook_json, row, col, save_path, is_look=True, cross=cross)


def para_tile_demo_second(save_path=root_path + 'demo_tile_features/'):
    refcell_list = os.listdir(root_path + 'first_tile_res_new')
    filename2sheetname = {}
    for index, formula_token_file in enumerate(refcell_list):
        formula_token = formula_token_file.replace(".npy", '')

        origin_filename = formula_token.split("---")[0]
        origin_sheetname = formula_token.split("---")[1]
        origin_row = formula_token.split("---")[2]
        origin_col = formula_token.split("---")[3]

        if origin_filename not in filename2sheetname:
            filename2sheetname[origin_filename] = {}
        if origin_sheetname not in filename2sheetname[origin_filename]:
            filename2sheetname[origin_filename][origin_sheetname] = []
        filename2sheetname[origin_filename][origin_sheetname].append(str(origin_row) + '---' + str(origin_col))

    def batch_generate_tile_demo_features(thread_id, batch_num):

        batch_len = int(len(filename2sheetname) / batch_num)
        count = 0

        all_num = 56144857

        count = 0
        print('filename2sheetname', len(filename2sheetname))

        for index, filename in enumerate(filename2sheetname):
            if index != batch_num:
                if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
                    continue
            else:
                if index <= batch_len * (thread_id - 1):
                    continue
            print(index, len(filename2sheetname))
            with open('../Demo/origin_fortune500_workbook_json/' + filename + '.json', 'r') as f:
                workbook_json = json.load(f)
            for sheetname in filename2sheetname[filename]:
                filesheet = filename + '---' + sheetname
                for row_col in filename2sheetname[filename][sheetname]:
                    res = np.load(
                        '/datadrive-2/data/fortune500_test/first_tile_res_new/' + filesheet + '---' + row_col + '.npy',
                        allow_pickle=True).item()
                    for key in res:
                        if 'best_row_col' not in res[key]:
                            continue
                        best_row_col = res[key]['best_row_col']
                        best_row = int(best_row_col.split('---')[0])
                        best_col = int(best_row_col.split('---')[1])

                        for row in range(best_row, best_row + 100):
                            for col in range(best_col, best_col + 10):
                                count += 1
                                if os.path.exists(
                                        save_path + filename + '---' + sheetname + '---' + str(row) + '---' + str(
                                            col) + ".json"):
                                    continue
                                generate_demo_features(filename, sheetname, workbook_json, row, col, save_path)
        print('count', count)

    batch_generate_tile_demo_features(1, 1)
    # process = [
    #     Process(target=batch_generate_tile_demo_features, args=(1,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(2,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(3,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(4,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(5,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(6,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(7,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(8,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(9,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(10,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(11,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(12,20)), 
    #     Process(target=batch_generate_tile_demo_features,args=(13,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(14,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(15,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(16,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(17,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(18,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(19,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(20,20)), 
    # ]
    # [p.start() for p in process]  # 开启了两个进程
    # [p.join() for p in process]   # 等待两个进程依次结束


def generate_one_after_feature(filename, sheetname, row, col, source_root_path, save_path, model_path):
    formula_token = filename + '---' + sheetname + '---' + str(row) + '---' + str(col)
    model = torch.load(model_path)
    if os.path.exists(save_path + formula_token + '.npy'):
        return
    try:
        feature_nparray = np.load(source_root_path + formula_token + '.npy', allow_pickle=True)
        feature_nparray = feature_nparray.reshape(1, 100, 10, 399)
        model.eval()
        feature_nparray = torch.DoubleTensor(feature_nparray)
        feature_nparray = Variable(feature_nparray).to(torch.float32)
        feature_nparray = model(feature_nparray).detach().numpy()
        np.save(save_path + formula_token + '.npy', feature_nparray)
    except Exception as e:
        print('error: generate_one_after_feature:')
        print(e)
        return


def para_tile_demo_second_block(top_k=1, save_path=root_path + 'demo_tile_features/',
                                save_path2=root_path + 'demo_after_features/'):
    model_path = '196model/cnn_new_dynamic_triplet_margin_1_3_12'
    model = torch.load(model_path)
    bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)

    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)

    # refcell_list = os.listdir(root_path+'sorted_first_block')
    with open("need_run_list.json", 'r') as f:
        need_run_list = json.load(f)
    filename2sheetname = {}
    for index, formula_token_file in enumerate(need_run_list):
        formula_token = formula_token_file.replace(".npy", '')

        origin_filename = formula_token.split("---")[0]
        origin_sheetname = formula_token.split("---")[1]
        origin_row = formula_token.split("---")[2]
        origin_col = formula_token.split("---")[3]

        if origin_filename not in filename2sheetname:
            filename2sheetname[origin_filename] = {}
        if origin_sheetname not in filename2sheetname[origin_filename]:
            filename2sheetname[origin_filename][origin_sheetname] = []
        filename2sheetname[origin_filename][origin_sheetname].append(str(origin_row) + '---' + str(origin_col))

    def batch_generate_tile_demo_features(bert_dict, content_tem_dict, model, thread_id, batch_num):

        batch_len = int(len(filename2sheetname) / batch_num)
        count = 0

        all_num = 56144857

        count = 0
        print('filename2sheetname', len(filename2sheetname))

        for index, filename in enumerate(filename2sheetname):
            # if index != batch_num:
            #     if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
            #         continue
            # else:
            #     if index <= batch_len * (thread_id - 1 ):
            #         continue
            # print(index, len(filename2sheetname))
            load_wbj_stime = time.time()
            if filename != '331430564339131039422768272513617590289-sluc487b.xlsx':
                continue
            with open('../Demo/origin_fortune500_workbook_json/' + filename + '.json', 'r') as f:
                workbook_json = json.load(f)
            load_wbj_etime = time.time()
            for index1, sheetname in enumerate(filename2sheetname[filename]):
                print("    ", index1, len(filename2sheetname[filename]))
                filesheet = filename + '---' + sheetname
                if sheetname != 'SCHEMATIC AND BoM':
                    continue
                for row_col in filename2sheetname[filename][sheetname]:
                    start_time = time.time()
                    res = np.load(
                        '/datadrive-2/data/fortune500_test/sorted_first_block/' + filesheet + '---' + row_col + '.npy',
                        allow_pickle=True).item()
                    best_row_col_list = []
                    if filename + '---' + sheetname + '---' + row_col != '331430564339131039422768272513617590289-sluc487b.xlsx---SCHEMATIC AND BoM---42---5':
                        continue
                    print('res', res)
                    for key in res:
                        if 'time' == key:
                            continue
                        for first_row_col in res[key]:
                            distance_dict = res[key][first_row_col]
                            sorted_list = sorted(distance_dict.items(), key=lambda x: x[1])
                            best_row_col_list += [list(i)[0] for i in sorted_list[0:top_k]]
                        print('best_row_col_list', best_row_col_list)
                        for row_col_tuple in best_row_col_list:
                            best_row = int(row_col_tuple.split('---')[0])
                            best_col = int(row_col_tuple.split('---')[1])
                            print('row_col_tuple', row_col_tuple)
                            for row in range(best_row, best_row + 100):
                                for col in range(best_col, best_col + 10):
                                    # print(row, col)
                                    count += 1
                                    if os.path.exists(
                                            root_path + 'demo_after_features/' + filename + '---' + sheetname + '---' + str(
                                                row) + '---' + str(col) + ".npy"):
                                        continue
                                    print("generate features")
                                    generate_demo_features(filename, sheetname, workbook_json, row, col, save_path)
                                    generate_one_after_feature(
                                        filename + '---' + sheetname + '---' + str(row) + '---' + str(col), bert_dict,
                                        content_tem_dict, model)
                    end_time = time.time()
                    # with open(root_path + 'second_level_time/' + filename + '---' + sheetname + '---' + row_col + ".json", 'w') as f:
                    # json.dump({'second_level_time': end_time - start_time, 'load_wbj_time': load_wbj_etime - load_wbj_stime}, f)
            #     break
            # break
        print('count', count)

    batch_generate_tile_demo_features(bert_dict, content_tem_dict, model, 1, 1)
    # process = [
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 1,5)),
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 2,5)),
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 3,5)), 
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 4,5)),
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 5,5)), 
    #     # Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 5,5)),
    # ]
    # process = [
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 1,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 2,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 3,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 4,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 5,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 6,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 7,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 8,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 9,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 10,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 11,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 12,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 13,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 14,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 15,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 16,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 17,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 18,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 19,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(bert_dict, content_tem_dict, model, 20,20)), 
    # ]
    # [p.start() for p in process]  # 开启了两个进程
    # [p.join() for p in process]   # 等待两个进程依次结束


def sort_second_block(top_k=1):
    refcell_list = os.listdir(root_path + 'sorted_first_block')
    # with open("need_run_list.json", 'r') as f:
    # need_run_list = json.load(f)
    filename2sheetname = {}
    for index, formula_token_file in enumerate(refcell_list):
        formula_token = formula_token_file.replace(".npy", '')
        # if formula_token == '173586831586798430403847316065547903587-wcp_edg_workbook.xlsx---5. FileSystem - Dir. Structure---7---2':
        #     print("first exists")
        origin_filename = formula_token.split("---")[0]
        origin_sheetname = formula_token.split("---")[1]
        origin_row = formula_token.split("---")[2]
        origin_col = formula_token.split("---")[3]

        if origin_filename not in filename2sheetname:
            filename2sheetname[origin_filename] = {}
        if origin_sheetname not in filename2sheetname[origin_filename]:
            filename2sheetname[origin_filename][origin_sheetname] = []
        filename2sheetname[origin_filename][origin_sheetname].append(str(origin_row) + '---' + str(origin_col))

    def batch_generate_tile_demo_features(thread_id, batch_num):
        batch_len = int(len(filename2sheetname) / batch_num)
        count = 0

        all_num = 56144857

        count = 0
        print('filename2sheetname', len(filename2sheetname))

        for index, filename in enumerate(filename2sheetname):
            # if index != batch_num:
            #     if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
            #         continue
            # else:
            #     if index <= batch_len * (thread_id - 1 ):
            #         continue
            # print(index, len(filename2sheetname))
            # if filename != '331430564339131039422768272513617590289-sluc487b.xlsx':
            # continue
            for sheetname in filename2sheetname[filename]:
                origin_filesheet = filename + '---' + sheetname
                # if sheetname != 'SCHEMATIC AND BoM':
                #     continue
                for row_col in filename2sheetname[filename][sheetname]:
                    start_time = time.time()
                    formula_token = filename + '---' + sheetname + '---' + row_col
                    # print('formula_token', formula_token)
                    if formula_token != '289605582270926027305192306518209639446-acceptedtable.xlsx---Personalize Table of UNs I Ship---18---4':
                        continue
                    need_continue = False

                    tmp_continue = True
                    # if os.path.exists(root_path + 'sorted_second_block/' + formula_token + ".npy"):
                    #     exists_res = np.load(root_path + 'sorted_second_block/' + formula_token + ".npy", allow_pickle=True).item()
                    #     # print('exists_res.keys()', exists_res.keys())
                    #     for key in exists_res:
                    #         if key == 'time':
                    #             continue
                    #         if len(exists_res[key]) == 0:
                    #             tmp_continue = False
                    #             break

                    #     if len(exists_res) == 0:
                    #         tmp_continue = False
                    #     if tmp_continue:
                    #         continue
                    print('formula_token', formula_token)
                    res = {}
                    if not os.path.exists(root_path + 'model1_res/' + formula_token + '.json'):
                        print('not exists:model1_res')
                        continue
                    with open(root_path + 'model1_res/' + formula_token + '.json', 'r') as f:
                        mode1_res = json.load(f)
                    found_filesheet = mode1_res[1].split('---')[0] + '---' + mode1_res[1].split('---')[1]
                    found_formula_token = mode1_res[1]

                    if not os.path.exists(root_path + 'test_refcell_position/' + found_formula_token + '.json'):
                        print('not exists:test_refcell_position')
                        continue
                    with open(root_path + 'test_refcell_position/' + found_formula_token + '.json', 'r') as f:
                        test_refcell_position = json.load(f)

                    print('len(test_refcell_position)', len(test_refcell_position))
                    print(test_refcell_position)
                    for position in test_refcell_position:
                        ref_row = position['R']
                        ref_col = position['C']
                        print(ref_row, ref_col)
                        if not os.path.exists(root_path + 'demo_after_features/' + found_filesheet + '---' + str(
                                ref_row) + '---' + str(ref_col) + '.npy'):
                            print('not exists:demo_after_features ref_row, ref_col')
                            continue
                        if not os.path.exists(
                                '/datadrive-2/data/fortune500_test/sorted_first_block/' + formula_token + '.npy'):
                            print('not exists:first_tile_res_new')
                            continue
                        feature = np.load(
                            root_path + 'demo_after_features/' + found_filesheet + '---' + str(ref_row) + '---' + str(
                                ref_col) + '.npy', allow_pickle=True)
                        sorted_first_block = np.load(
                            '/datadrive-2/data/fortune500_test/sorted_first_block/' + formula_token + '.npy',
                            allow_pickle=True).item()

                        if str(ref_row) + '---' + str(ref_col) not in sorted_first_block:
                            print('not in first', str(ref_row) + '---' + str(ref_col))
                            continue
                        best_row_col_list = []
                        print("sorted_first_block[str(ref_row) + '---' + str(ref_col)]",
                              sorted_first_block[str(ref_row) + '---' + str(ref_col)])
                        for first_row_col in sorted_first_block[str(ref_row) + '---' + str(ref_col)]:
                            # print('first_row_col', first_row_col)
                            distance_dict = sorted_first_block[str(ref_row) + '---' + str(ref_col)][first_row_col]
                            # print('distance_dict', distance_dict)
                            sorted_list = sorted(distance_dict.items(), key=lambda x: x[1])
                            # print('sorted_list', [list(i)[0] for i in sorted_list[0:top_k]])
                            best_row_col_list += [list(i)[0] for i in sorted_list[0:top_k]]
                        print('best_row_col_list', best_row_col_list)
                        best_row_col_list = list(set(best_row_col_list))
                        res[str(ref_row) + '---' + str(ref_col)] = {}
                        for best_row_col in best_row_col_list:
                            print('best_row_col', best_row_col)
                            best_row = int(best_row_col.split('---')[0])
                            best_col = int(best_row_col.split('---')[1])

                            for row in range(best_row, best_row + 100):
                                for col in range(best_col, best_col + 10):
                                    # if(best_row_col == '1---11'):
                                    #     print('    ', row, col)
                                    if not os.path.exists(
                                            root_path + 'demo_after_features/' + origin_filesheet + '---' + str(
                                                row) + '---' + str(col) + '.npy'):
                                        print("not exists:in demo_after_features")
                                        continue
                                    other_feature = np.load(
                                        root_path + 'demo_after_features/' + origin_filesheet + '---' + str(
                                            row) + '---' + str(col) + '.npy', allow_pickle=True)
                                    # print('other_feature', other_feature)
                                    distance = euclidean(feature, other_feature)
                                    # print('distance', distance)
                                    res[str(ref_row) + '---' + str(ref_col)][str(row) + '---' + str(col)] = distance
                                    res[str(ref_row) + '---' + str(ref_col)]['best_distance'] = np.inf
                                    res[str(ref_row) + '---' + str(ref_col)]['best_row'] = -1
                                    if distance < res[str(ref_row) + '---' + str(ref_col)]['best_distance']:
                                        res[str(ref_row) + '---' + str(ref_col)]['best_distance'] = distance
                                        res[str(ref_row) + '---' + str(ref_col)]['best_row_col'] = str(
                                            row) + '---' + str(col)
                    end_time = time.time()
                    res['time'] = end_time - start_time
                    # print('res', res)
                    print(root_path + 'sorted_second_block/' + formula_token + '.npy')
                    np.save(root_path + 'sorted_second_block/' + formula_token + '.npy', res)
            #     break
            # break
        # print('count', count)

    batch_generate_tile_demo_features(1, 1)
    # process = [
    #     Process(target=batch_generate_tile_demo_features, args=(1,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(2,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(3,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(4,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(5,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(6,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(7,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(8,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(9,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(10,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(11,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(12,20)), 
    #     Process(target=batch_generate_tile_demo_features,args=(13,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(14,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(15,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(16,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(17,20)),
    #     Process(target=batch_generate_tile_demo_features, args=(18,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(19,20)), 
    #     Process(target=batch_generate_tile_demo_features, args=(20,20)), 
    # ]
    # [p.start() for p in process]  # 开启了两个进程
    # [p.join() for p in process]   # 等待两个进程依次结束


def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def sort_first_block():
    refcell_list = os.listdir(root_path + 'test_refcell_position')
    tile_rows = os.listdir(root_path + 'tile_rows')
    filename2sheetname = {}
    for filesheet_json in refcell_list:
        filesheet = filesheet_json.replace('.json', '')
        filename = filesheet.split('---')[0]
        sheetname = filesheet.split('---')[1]
        if filename not in filename2sheetname:
            filename2sheetname[filename] = []
        filename2sheetname[filename].append(sheetname)
    filenames = list(filename2sheetname.keys())

    # formula_filelists = os.listdir(root_path + 'test_refcell_position')
    formula_filelists = os.listdir(root_path + 'after_feature_test')
    # print(len(filelist)) 
    # need_continue = True
    for index, formula_token_npy in enumerate(formula_filelists):
        # if formula_token_npy != "234679781916000373063272744719263521823-8054.ddr3-phy-calc-v11-for-1600.xlsx---PHY CALC---123---4.npy":
        # continue
        if formula_token_npy != "289605582270926027305192306518209639446-acceptedtable.xlsx---Personalize Table of UNs I Ship---18---4.npy":
            continue

        print(index, len(formula_filelists))
        # if formula_token_npy == "112642096631551186185766412613832910547-ads1262_5f00_input_5f00_short_5f00_sinc3.xlsx---Gain32---2---10.npy":
        #     need_continue = False
        #     continue
        # if need_continue:
        #     continue
        start_time = time.time()
        formula_token = formula_token_npy.replace('.npy', '')
        res = {}
        origin_file = formula_token.split('---')[0]
        origin_sheet = formula_token.split('---')[1]
        origin_filesheet = origin_file + '---' + origin_sheet

        print(index, len(formula_filelists))
        if not os.path.exists(root_path + 'model1_res/' + formula_token + '.json'):
            # print("not exists")
            # print(root_path + 'model1_res/' + formula_token + '.json')
            continue
        print('formula_token', formula_token)
        with open(root_path + 'model1_res/' + formula_token + '.json', 'r') as f:
            mode1_res = json.load(f)

        found_formula_token = mode1_res[1]
        found_filesheet = mode1_res[1].split('---')[0] + '---' + mode1_res[1].split('---')[1]
        # print('found_sheet', found_filesheet)

        print("found_formula_token", found_formula_token)
        if not os.path.exists(root_path + 'test_refcell_position/' + found_formula_token + '.json'):
            continue
        with open(root_path + 'test_refcell_position/' + found_formula_token + '.json', 'r') as f:
            test_refcell_position = json.load(f)

        with open(root_path + 'tile_rows/' + origin_filesheet + '.json', 'r') as f:
            tile_rows = json.load(f)
        with open(root_path + 'tile_cols/' + origin_filesheet + '.json', 'r') as f:
            tile_cols = json.load(f)

        # print('tile_rows', tile_rows)
        # print('tile_cols', tile_cols)
        # print('test_refcell_position', test_refcell_position)
        # print('found_formula_token', found_formula_token)
        for position in test_refcell_position:
            ref_row = position['R']
            ref_col = position['C']

            refcell_tile_row = int(ref_row / 100) * 100 + 1
            refcell_tile_col = int(ref_col / 10) * 10 + 1

            is_left = False
            is_up = False
            if ref_row - refcell_tile_row < refcell_tile_row + 100 - ref_row:  # up
                is_up = True
            if ref_col - refcell_tile_col < refcell_tile_col + 10 - ref_col:  # left
                is_left = True

            closed_four_tiles = []
            closed_four_tiles.append((refcell_tile_row, refcell_tile_col))  # first
            if is_up:
                if refcell_tile_row >= 101:
                    closed_four_tiles.append((refcell_tile_row - 100, refcell_tile_col))  # second add up
                    if is_left:
                        if refcell_tile_col >= 11:
                            closed_four_tiles.append(
                                (refcell_tile_row - 100, refcell_tile_col - 10))  # third add left, up
                    else:
                        if refcell_tile_col + 10 in tile_cols:
                            closed_four_tiles.append(
                                (refcell_tile_row - 100, refcell_tile_col + 10))  # third add right, up
            else:
                if refcell_tile_col + 100 in tile_rows:
                    closed_four_tiles.append((refcell_tile_row + 100, refcell_tile_col))  # 0 second add down
                if is_left:
                    if refcell_tile_col >= 11:
                        closed_four_tiles.append(
                            (refcell_tile_row + 100, refcell_tile_col - 10))  # third add left, down
                else:
                    if refcell_tile_col + 10 in tile_cols:
                        closed_four_tiles.append(
                            (refcell_tile_row + 100, refcell_tile_col + 10))  # third add right, down

            if is_left:
                if refcell_tile_col >= 11:
                    closed_four_tiles.append((refcell_tile_row, refcell_tile_col - 10))  # forth add left
            else:
                if refcell_tile_col + 10 in tile_cols:
                    closed_four_tiles.append((refcell_tile_row, refcell_tile_col + 10))  # forth add right
            # print("str(ref_row) + '---' + str(ref_col)", str(ref_row) + '---' + str(ref_col))
            # print("closed_four_tiles", closed_four_tiles)
            res[str(ref_row) + '---' + str(ref_col)] = {}
            for one_found_tile in closed_four_tiles:
                one_found_row = one_found_tile[0]
                one_found_col = one_found_tile[1]
                # print('one_found_row', one_found_row)
                # print('one_found_col', one_found_col)
                if not os.path.exists(
                        root_path + 'demo_after_features/' + found_filesheet + '---' + str(one_found_row) + '---' + str(
                            one_found_col) + '.npy'):
                    print("not exists")
                    continue
                feature = np.load(
                    root_path + 'demo_after_features/' + found_filesheet + '---' + str(one_found_row) + '---' + str(
                        one_found_col) + '.npy', allow_pickle=True)

                # print('feature', feature)

                # res[str(ref_row) + '---' + str(ref_col)]['best_distance'] = np.inf
                # res[str(ref_row) + '---' + str(ref_col)]['best_row_col'] = ''
                res[str(ref_row) + '---' + str(ref_col)][str(one_found_row) + '---' + str(one_found_col)] = {}
                for row in tile_rows:
                    for col in tile_cols:
                        if not os.path.exists(
                                root_path + 'demo_after_features/' + origin_filesheet + '---' + str(row) + '---' + str(
                                    col) + '.npy'):
                            print('after not exists:',
                                  root_path + 'demo_after_features/' + origin_filesheet + '---' + str(
                                      row) + '---' + str(col) + '.npy')
                            continue
                        other_feature = np.load(
                            root_path + 'demo_after_features/' + origin_filesheet + '---' + str(row) + '---' + str(
                                col) + '.npy', allow_pickle=True)
                        # print('other_feature', other_feature)
                        distance = euclidean(feature, other_feature)
                        # print('distance', distance)
                        res[str(ref_row) + '---' + str(ref_col)][str(one_found_row) + '---' + str(one_found_col)][
                            str(row) + '---' + str(col)] = distance
                        # if distance  < res[str(ref_row) + '---' + str(ref_col)]['best_distance']:
                        #     res[str(ref_row) + '---' + str(ref_col)]['best_distance'] = distance
                        #     res[str(ref_row) + '---' + str(ref_col)]['best_row_col'] = str(row) + '---' + str(col)     
        end_time = time.time()
        res['time'] = end_time - start_time
        # print("res", res)
        np.save(root_path + 'sorted_first_block/' + formula_token + '.npy', res)


def find_first_top1_center(save_path=root_path + 'demo_tile_features/'):
    refcell_list = os.listdir(root_path + 'test_refcell_position')
    tile_rows = os.listdir(root_path + 'tile_rows')
    filename2sheetname = {}
    for filesheet_json in refcell_list:
        filesheet = filesheet_json.replace('.json', '')
        filename = filesheet.split('---')[0]
        sheetname = filesheet.split('---')[1]
        if filename not in filename2sheetname:
            filename2sheetname[filename] = []
        filename2sheetname[filename].append(sheetname)
    filenames = list(filename2sheetname.keys())

    # formula_filelists = os.listdir(root_path + 'test_refcell_position')
    formula_filelists = os.listdir(root_path + 'after_feature_test')
    # print(len(filelist)) 
    for index, formula_token_npy in enumerate(formula_filelists):
        # if formula_token_npy != "234679781916000373063272744719263521823-8054.ddr3-phy-calc-v11-for-1600.xlsx---PHY CALC---123---4.npy":
        # continue
        # print(index, len(formula_filelists))
        formula_token = formula_token_npy.replace('.npy', '')
        res = {}
        origin_file = formula_token.split('---')[0]
        origin_sheet = formula_token.split('---')[1]
        origin_filesheet = origin_file + '---' + origin_sheet

        print(index, len(formula_filelists))
        if not os.path.exists(root_path + 'model1_res/' + formula_token + '.json'):
            # print("not exists")
            # print(root_path + 'model1_res/' + formula_token + '.json')
            continue
        with open(root_path + 'model1_res/' + formula_token + '.json', 'r') as f:
            mode1_res = json.load(f)

        found_formula_token = mode1_res[1]
        found_filesheet = mode1_res[1].split('---')[0] + '---' + mode1_res[1].split('---')[1]
        print('found_sheet', found_filesheet)

        if not os.path.exists(root_path + 'test_refcell_position/' + found_formula_token + '.json'):
            continue
        with open(root_path + 'test_refcell_position/' + found_formula_token + '.json', 'r') as f:
            test_refcell_position = json.load(f)

        with open(root_path + 'tile_rows/' + origin_filesheet + '.json', 'r') as f:
            tile_rows = json.load(f)
        with open(root_path + 'tile_cols/' + origin_filesheet + '.json', 'r') as f:
            tile_cols = json.load(f)

        print('tile_rows', tile_rows)
        print('tile_cols', tile_cols)
        print('test_refcell_position', test_refcell_position)
        print('found_formula_token', found_formula_token)
        for position in test_refcell_position:
            ref_row = position['R']
            ref_col = position['C']
            if not os.path.exists(
                    root_path + 'demo_after_features/' + found_filesheet + '---' + str(ref_row) + '---' + str(
                        ref_col) + '.npy'):
                print("not exists")
                continue
            feature = np.load(root_path + 'demo_after_features/' + found_filesheet + '---' + str(ref_row) + '---' + str(
                ref_col) + '.npy', allow_pickle=True)
            print('ref_row', ref_row)
            print('ref_col', ref_col)
            # print('feature', feature)
            res[str(ref_row) + '---' + str(ref_col)] = {}
            res[str(ref_row) + '---' + str(ref_col)]['best_distance'] = np.inf
            res[str(ref_row) + '---' + str(ref_col)]['best_row_col'] = ''

            for row in tile_rows:
                for col in tile_cols:
                    if not os.path.exists(
                            root_path + 'demo_after_features/' + origin_filesheet + '---' + str(row) + '---' + str(
                                col) + '.npy'):
                        print('after not exists:',
                              root_path + 'demo_after_features/' + origin_filesheet + '---' + str(row) + '---' + str(
                                  col) + '.npy')
                        continue
                    other_feature = np.load(
                        root_path + 'demo_after_features/' + origin_filesheet + '---' + str(row) + '---' + str(
                            col) + '.npy', allow_pickle=True)
                    # print('other_feature', other_feature)
                    distance = euclidean(feature, other_feature)
                    print('distance', distance)
                    res[str(ref_row) + '---' + str(ref_col)][str(row) + '---' + str(col)] = distance
                    if distance < res[str(ref_row) + '---' + str(ref_col)]['best_distance']:
                        res[str(ref_row) + '---' + str(ref_col)]['best_distance'] = distance
                        res[str(ref_row) + '---' + str(ref_col)]['best_row_col'] = str(row) + '---' + str(col)
        np.save(root_path + 'first_tile_res_new/' + formula_token + '.npy', res)
        # for 


def count_demo_recell(save_path=root_path + 'demo_tile_features/'):
    model_path = '196model/cnn_new_dynamic_triplet_margin_1_3_12'
    model = torch.load(model_path)
    refcell_list = os.listdir(root_path + 'test_refcell_position')
    ref_cell_fail_res = []
    filename2sheetname = {}
    for filesheet_json in refcell_list:
        filesheet = filesheet_json.replace('.json', '')
        filename = filesheet.split('---')[0]
        sheetname = filesheet.split('---')[1]
        row = filesheet.split('---')[2]
        col = filesheet.split('---')[3]
        row_col = row + '---' + col
        if filename not in filename2sheetname:
            filename2sheetname[filename] = {}
        if sheetname not in filename2sheetname[filename]:
            filename2sheetname[filename][sheetname] = []
        filename2sheetname[filename][sheetname].append(row_col)
    filenames = list(filename2sheetname.keys())

    filelist = os.listdir(root_path + 'test_refcell_position')
    print(len(filelist))
    bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)

    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)
    for index, filename in enumerate(filenames):
        print(index, len(filenames))
        with open('../Demo/origin_fortune500_workbook_json/' + filename + '.json', 'r') as f:
            workbook_json = json.load(f)
        for index1, sheetname in enumerate(filename2sheetname[filename]):
            print('    ', index1, len(filename2sheetname[filename]))

            for index2, row_col in enumerate(filename2sheetname[filename][sheetname]):

                print('        ', index2, len(filename2sheetname[filename][sheetname]))
                with open(
                        root_path + 'test_refcell_position/' + filename + '---' + sheetname + '---' + row_col + '.json',
                        'r') as f:
                    positions = json.load(f)

                for position in positions:
                    if 'R' not in position or 'C' not in position:
                        ref_cell_fail_res.append(filename + '---' + sheetname + '---' + row_col)
                        break
                    row = position['R']
                    col = position['C']
                    if os.path.exists(root_path + 'demo_after_features/' + filename + '---' + sheetname + '---' + str(
                            row) + '---' + str(col) + ".npy"):
                        continue
                    print('generate features', filename + '---' + sheetname + '---' + str(row) + '---' + str(col))
                    generate_demo_features(filename, sheetname, workbook_json, row, col, save_path)
                    generate_one_after_feature(filename + '---' + sheetname + '---' + str(row) + '---' + str(col),
                                               bert_dict, content_tem_dict, model)
    with open('ref_cell_parse_fail_res.json', 'w') as f:
        json.dump(ref_cell_fail_res, f)


def look_features():
    filelist = os.listdir("/datadrive-2/data/fortune500_test/demo_tile_features/")
    pirnt_set = set()
    token_set = set()

    for filename_json in filelist:
        formula_token = filename_json.replace('.json', '')
        filename = filename_json.split('---')[0]
        sheetname = filename_json.split('---')[1]
        filesheet = filename + '---' + sheetname
        pirnt_set.add(filesheet)
        token_set.add(formula_token)

    for item in token_set:
        # print(item)
        if '173586831586798430403847316065547903587-wcp_edg_workbook.xlsx---Start Here' in item:
            print(item)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def get_feature_vector_with_bert_keyw(feature, bert_dict, content_tem_dict, mask, temp_bert_dict, bert_dict_path,
                                      bert_dict_file):
    bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    result = []
    for index, item1 in enumerate(feature):
        one_cel_feature = []
        if type(feature[index]).__name__ == 'list':
            feature[index] = feature[index][0]

        for index1, item in enumerate(feature[index]):
            start_time = time.time()
            if index1 == 0:  # background color r
                one_cel_feature.append(feature[index][item])
            if index1 == 1:  # background color b
                one_cel_feature.append(feature[index][item])
            if index1 == 2:  # background color g
                one_cel_feature.append(feature[index][item])
            if index1 == 3:  # font color r
                one_cel_feature.append(feature[index][item])
            if index1 == 4:  # font color b
                one_cel_feature.append(feature[index][item])
            if index1 == 5:  # font color g
                one_cel_feature.append(feature[index][item])
            if index1 == 6:  # font size
                one_cel_feature.append(feature[index][item])
            if index1 == 7:  # font_strikethrough
                if (feature[index][item] == True):
                    one_cel_feature.append(1)
                else:
                    one_cel_feature.append(0)
            if index1 == 8:  # font_shadow
                if (feature[index][item] == True):
                    one_cel_feature.append(1)
                else:
                    one_cel_feature.append(0)
            if index1 == 9:  # font_ita
                if (feature[index][item] == True):
                    one_cel_feature.append(1)
                else:
                    one_cel_feature.append(0)
            if index1 == 10:  # font_bold
                if (feature[index][item] == True):
                    one_cel_feature.append(1)
                else:
                    one_cel_feature.append(0)
            if index1 == 11:  # height
                one_cel_feature.append(feature[index][item])
            if index1 == 12:  # width
                one_cel_feature.append(feature[index][item])
            if index1 == 13:  # content\\
                if mask == 0:
                    feature[index][item] = 'content'
                if is_number(str(feature[index][item])):
                    cell_type = 1
                    if mask == 1:
                        feature[index][item] = 0
                elif str(feature[index][item]) == '':
                    cell_type = 0
                else:
                    cell_type = 2
                if str(feature[index][item]) in bert_dict:
                    bert_feature = bert_dict[str(feature[index][item])]
                elif str(feature[index][item]) in temp_bert_dict:
                    bert_feature = temp_bert_dict[str(feature[index][item])]
                elif os.path.exists(bert_dict_path + change_word_to_save_word(str(feature[index][item])) + '.npy'):
                    bert_feature = np.load(
                        bert_dict_path + change_word_to_save_word(str(feature[index][item])) + '.npy',
                        allow_pickle=True)
                    temp_bert_dict[str(feature[index][item])] = bert_feature
                else:
                    bert_feature = bert_model.encode(str(feature[index][item])).tolist()
                    try:
                        np.save(bert_dict_path + change_word_to_save_word(str(feature[index][item])) + '.npy',
                                bert_feature)
                    except:
                        bert_dict[str(feature[index][item])] = bert_feature
                        with open(bert_dict_file, 'w') as f:
                            json.dump(bert_dict, f)
                for i in bert_feature:
                    one_cel_feature.append(i)
            if index1 == 14:  # content_template
                if mask == 1:
                    if 'N' in str(feature[index][item]):
                        feature[index][item] = 'N'
                if not str(feature[index][item]) in content_tem_dict:
                    one_cel_feature.append(len(content_tem_dict) + 1)
                else:
                    one_cel_feature.append(content_tem_dict[str(feature[index][item])])
                one_cel_feature.append(cell_type)
            end_time = time.time()
        result.append(one_cel_feature)
    return result, temp_bert_dict


def para_demo_before():
    # bert_dict = {}
    # content_tem_dict = {}
    # root_path = '/datadrive-2/data/fuste_test/'
    root_path = '/datadrive-2/data/deco_test/'
    offline_time1 = time.time()
    # model_path = '196model/cnn_new_dynamic_triplet_margin_1_3_12'
    # model = torch.load(model_path)
    try:
        with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
            bert_dict = json.load(f)
    except:
        bert_dict = {}

    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)
    offline_time2 = time.time()

    def generate_demo_before_features(bert_dict, content_tem_dict, thread_id, batch_num):
        bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # saved_root_path = root_path + 'training_before_features_shift_content_mask2/'
        # saved_root_path = root_path + 'cross_training_before_specific_c2/'
        # saved_root_path = root_path + 'training_before_features/'
        saved_root_path = root_path + 'sheets_before_features/'
        # saved_root_path = root_path + 'training_before_shift_c2/'
        # saved_root_path = root_path + 'demo_after_features/'
        # source_root_path = root_path + 'training_tile_features/' # demo_tile_features_specific
        source_root_path = root_path + 'sheets_json_feaures/'  # demo_tile_features_specific
        # source_root_path = root_path + 'cross_demo_tile_features_specific/' # demo_tile_features_specific
        # source_root_path = root_path + 'demo_tile_features_shift/'
        # source_root_path = root_path + 'demo_before_features/'
        formula_tokens = os.listdir(source_root_path)
        exists_files = os.listdir(saved_root_path)
        exists_files = [i.replace('.npy', '.json') for i in exists_files]
        formula_tokens = list(set(formula_tokens) - set(exists_files))
        formula_tokens.sort()
        batch_len = len(formula_tokens) / batch_num
        print('formula_tokens', len(formula_tokens))
        temp_bert_dict = {}
        for index, formula_token_path in enumerate(formula_tokens):
            formula_token = formula_token_path.replace(".json", '').replace('.npy', '')
            start_time = time.time()

            # if formula_token != 'andrea_ring__37__IFERCJan.xlsx---February 2000':
            # continue
            if thread_id != batch_num:
                if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
                    continue
            else:
                if index <= batch_len * (thread_id - 1):
                    continue
            print(index, len(formula_tokens))
            if os.path.exists(saved_root_path + formula_token + '.npy'):
                print('exists.....')
                continue
            # if index == 50:
            # break
            print('extracting....')
            try:
                if '.json' in formula_token_path:
                    with open(source_root_path + formula_token_path, 'r') as f:
                        # print('formula_token_path', formula_token_path)
                        # except:
                        #     continue
                        origin_feature = json.load(f)
                    temp_time1 = time.time()
                    feature_nparray, temp_bert_dict = get_feature_vector_with_bert_keyw(origin_feature,
                                                                                        bert_dict=bert_dict,
                                                                                        content_tem_dict=content_tem_dict,
                                                                                        mask=2,
                                                                                        temp_bert_dict=temp_bert_dict)
                    feature_nparray = np.array(feature_nparray)
                    np.save(saved_root_path + formula_token + '.npy', feature_nparray)
                    # except:
                    #     continue
                    end_time = time.time()
                    print('load time:', temp_time1 - start_time)
                    before_time = end_time - start_time

            #     else:
            #         feature_nparray = np.load(source_root_path + formula_token_path, allow_pickle=True)
            #         before_time = -1

            #     feature_nparray = feature_nparray.reshape(1,100,10,399)
            #     model.eval()

            #     feature_nparray = torch.DoubleTensor(feature_nparray)
            #     feature_nparray = Variable(feature_nparray).to(torch.float32)
            #     feature_nparray = model(feature_nparray).detach().numpy()
            #     print('after model predict....')
            #         # print('feature_list', feature_list)
            #     end_time = time.time()
            #     print(end_time - temp_time1)

            #     np.save(saved_root_path + formula_token + '.npy', feature_nparray)
            #     after_time = end_time - temp_time1
            #     with open(root_path + 'after_time/' + formula_token + '.json', 'w') as f:
            #         json.dump({'before_time': before_time, 'after_time': after_time}, f)
            except Exception as e:
                print('e', e)
                continue
            # break
            # print(end_time - start_time)
            # break

    # process = [
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,1,20)),
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,2,20)), 
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,3,20)),
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,4,20)), 
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,5,20)),
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,6,20)), 
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,7,20)),
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,8,20)), 
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,9,20)),
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,10,20)), 
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,11,20)),
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,12,20)), 
    #     Process(target=generate_demo_before_features,args=(bert_dict, content_tem_dict, 13,20)),
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,14,20)), 
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,15,20)),
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,16,20)), 
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,17,20)),
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,18,20)), 
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,19,20)), 
    #     Process(target=generate_demo_before_features, args=(bert_dict, content_tem_dict,20,20)), 
    # ]
    # [p.start() for p in process]  # 开启了两个进程
    # [p.join() for p in process]   # 等待两个进程依次结束
    generate_demo_before_features(bert_dict, content_tem_dict, 1, 1)

    print("offline_time", offline_time2 - offline_time1)


def test_time():
    filelist = os.listdir("/datadrive-2/data/fortune500_test/demo_tile_features")
    num = len(filelist)

    start_time = time.time()
    filelist = os.listdir("/datadrive-2/data/fortune500_test/demo_tile_features")
    new_num = len(filelist)

    end_time = time.time()
    print('new_num', new_num)
    print('num', num)
    print(new_num - num)
    print(end_time - start_time)
    print((new_num - num) / (end_time - start_time))

    # filelist = os.listdir(root_path + 'first_tile_res/')
    # count = 0
    # for filename in filelist:
    #     if '.json' in filename:
    #         count += 1
    #         print('count', count)
    #         shutil.move(root_path + 'first_tile_res/' + filename, root_path + 'demo_tile_features/' + filename)


def check_best():
    need_run_list = []

    def secc(second_found_formu, first_found_formul, formula):
        second_found_formu += 1
        first_found_formul += 1
        is_found = True
        found_list.append(formula['id'])
        return second_found_formu, first_found_formul, is_found

    with open("Formulas_fortune500_with_id.json", 'r') as f:
        formulas = json.load(f)
    with open('small_data_test_formulatoken2r1c1.json', 'r') as f:
        top10domain_formulatoken2r1c1 = json.load(f)
    with open('r1c12template_fortune500_constant.json', 'r') as f:
        r1c12template_top10domain = json.load(f)
    with open("crosstable_formulas.json", 'r') as f:
        crosstable_formulas = json.load(f)
    with open('fortune500_test_formula_token.json', 'r') as f:
        fortune500_test_formula_token = json.load(f)

    second_found_formu = 0
    first_found_formul = 0

    first_out_formu = 0
    first_mul_formu = 0
    out_and_multitable_formu = 0

    second_found_ref = 0
    first_found_ref = 0
    first_out_ref = 0
    first_mul_ref = 0

    not_in_best_ref = 0
    kw_not_in_best_ref = 0

    out_and_multitable_ref = 0
    multi_table = 0

    except_log_num = 0

    all_ref = 0

    template_fail = 0

    all_ = 0
    valid_all = 0

    count = 0
    found_list = []
    second_not_found = []

    formula_list = []

    first_cha = 0
    first_fancha = 0
    multi_fail = 0
    first_multi_fail = 0
    second_multi_fail = 0
    not_in_best_formu = 0
    kw_not_in_best_formu = 0

    tmp_stat = 0
    first_fail = 0

    fail_res = {}

    model1_suc_but_fail = 0
    model1_fail_but_suc = 0
    model1_fail_but_suc1 = 0
    model1_fail_but_suc2 = 0

    model1_suc = 0

    tmp_all = 0

    gt_not_in_second = 0
    first_out_mul_formu = 0
    for index, formula in enumerate(formulas):

        is_found = False
        suc = False

        formula_token = formula['filesheet'] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
        if formula_token not in fortune500_test_formula_token:
            continue
        if not os.path.exists(root_path + 'model1_res/' + formula_token + '.json'):
            continue
        if not os.path.exists(root_path + 'after_feature_test/' + formula_token + '.npy'):
            continue
        print('\033[1;35m index ' + str(index) + ' ' + str(len(formulas)) + ' ' + formula_token + '\033[0m')
        # if formula_token != '234679781916000373063272744719263521823-8054.ddr3-phy-calc-v11-for-1600.xlsx---PHY CALC---123---4':
        # continue
        all_ += 1
        formula_list.append((formula['filesheet'].split('---')[1], formula['r1c1']))
        if 'RC[-1]+ROUND(10*R' in formula['r1c1']:
            except_log_num += 1
            continue
        if formula['r1c1'] in crosstable_formulas:
            multi_table += 1
            # print(formula['r1c1'])
            # continue
        with open(root_path + 'dedup_model1_res/' + formula_token + '.json', 'r') as f:
            model1_res = json.load(f)
        if model1_res[4] == True:
            model1_suc += 1
        valid_all += 1
        found_formula_token = model1_res[1]
        print('found_formula_token', found_formula_token)
        if formula['r1c1'] not in r1c12template_top10domain:
            gt_template_id = -1
        else:
            gt_template_id = r1c12template_top10domain[formula['r1c1']]
        if model1_res[3] not in r1c12template_top10domain:
            found_template_id = -1
        else:
            found_template_id = r1c12template_top10domain[model1_res[3]]

        if found_template_id != gt_template_id:
            template_fail += 1
            # print('template fail')
            continue

        found_range_path = root_path + 'test_refcell_position/' + found_formula_token + '.json'
        gt_range_path = root_path + 'test_refcell_position/' + formula_token + '.json'
        if not os.path.exists(gt_range_path):
            if model1_res[2] == model1_res[3]:
                second_found_formu, first_found_formul, is_found = secc(second_found_formu, first_found_formul, formula)
                if model1_res[4] == False:
                    model1_fail_but_suc += 1
            continue

        if not os.path.exists(found_range_path):
            # print('not exists: test_refcell_position')
            continue

        # print('gt_exists', os.path.exists(gt_cell_path))
        # print('found_exists', os.path.exists(ref_cell_path))

        print('gt formula_token', model1_res[2])
        print('found formula_token', model1_res[3])
        with open(gt_range_path, 'r') as f:
            gt_ref_cell = json.load(f)

        with open(found_range_path, 'r') as f:
            found_ref_cell = json.load(f)
        if len(gt_ref_cell) != len(found_ref_cell):
            # print('gt != found')
            continue
        if len(found_ref_cell) == 0:
            if found_template_id == gt_template_id:
                second_found_formu, first_found_formul, is_found = secc(second_found_formu, first_found_formul, formula)
                if model1_res[4] == False and found_template_id != 13:
                    model1_fail_but_suc += 1
                    model1_fail_but_suc1 += 1
                    print('found_template_id', found_template_id)
                    print('found_ref_cell', found_ref_cell)
                    print(model1_res)
                    # break
            continue
        # if not os.path.exists(root_path + 'sorted_second_block/' + formula_token + '.npy'):
        # if not os.path.exists(root_path + 'second_tile_res_new/' + formula_token + '.npy'):
        # print('second_tile_res_new not exists')
        # second_not_found += 1
        # continue
        # break
        second_best = {}
        if os.path.exists(root_path + 'sorted_second_block/' + formula_token + '.npy'):
            second_best = np.load(root_path + 'sorted_second_block/' + formula_token + '.npy', allow_pickle=True).item()
        else:
            print("not exists sorted_second_block")
            second_not_found.append(formula_token)
        first_best = np.load(root_path + 'sorted_first_block/' + formula_token + '.npy', allow_pickle=True).item()
        # first_best1 = np.load(root_path + 'first_tile_res_new/' + formula_token + '.npy', allow_pickle=True).item()

        second_is_all_same = True
        first_is_all_same = True
        first_is_all_same1 = True
        first_is_out = False
        first_is_mul = False
        first_is_out_and_mul = False

        is_not_in_best_ref = False
        is_kw_not_in_best_ref = False
        f_is_not_in_best_ref = False
        is_gt_not_in_second = False

        is_exception = False
        # print('second_best', second_best)

        found_fail = False

        for index, rc in enumerate(found_ref_cell):
            print("****")
            print('rc', rc)
            gt_rc = str(gt_ref_cell[index]['R']) + '---' + str(gt_ref_cell[index]['C'])
            if str(rc['R']) + '---' + str(rc['C']) not in second_best:
                found_rc = '*1---*1'
                kw_not_in_best_ref += 1
                is_kw_not_in_best_ref = True
            else:
                if len(second_best[str(rc['R']) + '---' + str(rc['C'])]) == 0:
                    found_rc = '#1---#1'
                    not_in_best_ref += 1
                    is_not_in_best_ref = True
                else:
                    if 'best_distance' in second_best[str(rc['R']) + '---' + str(rc['C'])]:
                        second_best[str(rc['R']) + '---' + str(rc['C'])].pop('best_distance')
                    if 'best_row' in second_best[str(rc['R']) + '---' + str(rc['C'])]:
                        second_best[str(rc['R']) + '---' + str(rc['C'])].pop('best_row')
                    if 'best_row_col' in second_best[str(rc['R']) + '---' + str(rc['C'])]:
                        second_best[str(rc['R']) + '---' + str(rc['C'])].pop('best_row_col')
                    # print('second_best', second_best[str(rc['R']) + '---' + str(rc['C'])])
                    best_tuple = sorted(second_best[str(rc['R']) + '---' + str(rc['C'])].items(), key=lambda x: x[1])[0]
                    found_rc = best_tuple[0]
                    if gt_rc not in second_best[str(rc['R']) + '---' + str(rc['C'])]:
                        is_gt_not_in_second = True
                # print(second_best[str(rc['R']) + '---' + str(rc['C'])], second_best[str(rc['R']) + '---' + str(rc['C'])])

            first_cand = []
            first_found = []
            # print("first_best[str(rc['R']) + '---' + str(rc['C'])]", first_best[str(rc['R']) + '---' + str(rc['C'])])
            for one_of_four in first_best[str(rc['R']) + '---' + str(rc['C'])]:
                first_found.append(
                    sorted(first_best[str(rc['R']) + '---' + str(rc['C'])][one_of_four].items(), key=lambda x: x[1])[0][
                        0])
                first_cand += list(first_best[str(rc['R']) + '---' + str(rc['C'])][one_of_four].keys())
            first_cand = list(set(first_cand))
            print('first_found', first_found)
            print('first_cand', first_cand)

            # found_rc = str(found_ref_cell[index]['R']) + '---' + str(found_ref_cell[index]['C'])

            # print('gt_rc', gt_rc)
            if gt_ref_cell[index]['R'] % 100 == 0:
                first_row = int(gt_ref_cell[index]['R'] - 100) + 1
            else:
                first_row = int(gt_ref_cell[index]['R'] / 100) * 100 + 1

            if gt_ref_cell[index]['C'] % 10 == 0:
                first_col = int(gt_ref_cell[index]['C'] - 10) + 1
            else:
                first_col = int(gt_ref_cell[index]['C'] / 10) * 10 + 1
            gt_first_rc = str(first_row) + "---" + str(first_col)
            print('gt_first_rc', gt_first_rc)
            # print('first is in', gt_first_rc in first_found)
            print('gt_rc', gt_rc)
            print('found_rc', found_rc)
            gt_r = gt_ref_cell[index]['R']
            gt_c = gt_ref_cell[index]['C']
            if found_rc != gt_rc:
                second_is_all_same = False
                if gt_first_rc in first_found and not formula['r1c1'] in crosstable_formulas and model1_res[
                    4] == True and found_rc != '*1---*1' and found_rc != '#1---#1' and not is_gt_not_in_second:
                    fail_res[formula_token + '#####' + str(rc['R']) + '---' + str(rc['C'])] = {}
                    fail_res[formula_token + '#####' + str(rc['R']) + '---' + str(rc['C'])]['model_res'] = model1_res
                    fail_res[formula_token + '#####' + str(rc['R']) + '---' + str(rc['C'])]['found_rc'] = found_rc
                    fail_res[formula_token + '#####' + str(rc['R']) + '---' + str(rc['C'])]['found_score'] = float(
                        second_best[str(rc['R']) + '---' + str(rc['C'])][found_rc])
                    fail_res[formula_token + '#####' + str(rc['R']) + '---' + str(rc['C'])]['gt_rc'] = gt_rc
                    fail_res[formula_token + '#####' + str(rc['R']) + '---' + str(rc['C'])]['gt_score'] = float(
                        second_best[str(rc['R']) + '---' + str(rc['C'])][gt_rc])
                    sorted_tuples = sorted(second_best[str(rc['R']) + '---' + str(rc['C'])].items(), key=lambda x: x[1])
                    for index1, item_tuple in enumerate(sorted_tuples):
                        if item_tuple[0] == gt_rc:
                            fail_res[formula_token + '#####' + str(rc['R']) + '---' + str(rc['C'])][
                                'gt_rank'] = index1 + 1


            else:
                second_found_ref += 1

            if gt_first_rc not in first_found:
                first_is_all_same = False

                # if not (found_first_r <= gt_r and found_first_r + 100 > gt_r and found_first_c <= gt_c and found_first_c + 10 > gt_c):

                if '!' in formula['r1c1']:
                    #     first_mul_ref += 1
                    first_is_mul = True
                if gt_first_rc not in first_cand:
                    first_is_out = True
                    first_out_ref += 1

            else:
                first_found_ref += 1
            all_ref += 1

        # if not first_is_all_same:
        # break
        if is_kw_not_in_best_ref:
            kw_not_in_best_formu += 1
            need_run_list.append(formula_token)
            print("formula_token is_kw_not_in_best_ref", formula_token)
            print("second_best", second_best)
            # break
        if is_not_in_best_ref and not first_is_out:
            not_in_best_formu += 1
            print("formula_token", formula_token)
            print('second_best', second_best)
            # break
        if is_gt_not_in_second and first_is_all_same:
            gt_not_in_second += 1
            need_run_list.append(formula_token)
        if second_is_all_same:
            if model1_res[4] == False:
                model1_fail_but_suc += 1
                model1_fail_but_suc2 += 1
            if found_template_id == gt_template_id:
                if not first_is_all_same:
                    tmp_stat += 1
                    is_exception = True
                second_found_formu, first_found_formul, is_found = secc(second_found_formu, first_found_formul, formula)
            # else:
            #     if first_is_all_same:
            #         first_found_formul += 1
            else:
                if model1_res[4] == True:
                    model1_suc_but_fail += 1
        else:
            if model1_res[4] == True:
                model1_suc_but_fail += 1
            if formula['r1c1'] in crosstable_formulas:
                multi_fail += 1
            if first_is_all_same:
                first_found_formul += 1
                if formula['r1c1'] in crosstable_formulas:
                    second_multi_fail += 1
            else:
                if formula['r1c1'] in crosstable_formulas:
                    first_multi_fail += 1

        if not first_is_all_same:
            first_fail += 1
            if first_is_out:
                first_out_formu += 1
                if first_is_mul:
                    out_and_multitable_formu += 1

        tmp_all += 1
        if first_is_out and not first_is_mul and '20*R' not in model1_res[2] and 'C[8485]' not in model1_res[2]:
            count += 1
            if count <= 20:
                continue
            print('model1_res', model1_res)
            break
        # if model1_res[4] == False:
        #     count += 1
        #     if count <= 400:
        #         continue
        #     print(model1_res)

        #     break
        # if not first_is_all_same and not formula['r1c1'] in crosstable_formulas:
        # if model1_res[4] == True and first_is_all_same and not second_is_all_same and not is_kw_not_in_best_ref and not is_gt_not_in_second:
        # if gt_first_rc in first_found and not formula['r1c1'] in crosstable_formulas and model1_res[4] == True:

        # if is_gt_not_in_second and first_is_all_same:
        #     print("is_gt_not_in_second")
        #     print(formula_token)
        #     break
        # if count <= 2:
        #     count +=1
        # print(model1_res)
        # #     continue
        # count +=1
        # print('first res', first_best)
        # print("######## fail")
        # print('second_best', second_best)
        # print('formula_token', formula_token)
        # print(model1_res[2], model1_res[3])
        # print('found_rc', found_rc)
        # print('gt_rc', gt_rc)
        # print("first_is_out", first_is_out)
        # print(gt_rc not in second_best[str(rc['R']) + '---' + str(rc['C'])])
        # print(second_best[str(rc['R']) + '---' + str(rc['C'])][gt_rc])
        # print(second_best[str(rc['R']) + '---' + str(rc['C'])][found_rc])
        # tmp_stat += 1

        # break
        # if second_is_all_same and formula['r1c1'] in crosstable_formulas:
        #     print(formula)
        #     break
    # with open("fortune500_formula_list.json", 'w') as f:
    #     json.dump(formula_list, f)
    # with open("fail_res_3159.json", 'w') as f:
    #     json.dump(fail_res, f)
    # with open("need_run_list.json", 'w') as f:
    #     json.dump(need_run_list, f)
    print("first_found_formul", first_found_formul)
    print("first_out_formu", first_out_formu)
    print("first_mul_formu", first_mul_formu)
    print("first_out_mul_formu", out_and_multitable_formu)
    print("second_found_formu", second_found_formu)
    print("all_formu", all_)
    print("##########")
    print("first_found_ref", first_found_ref)
    print("first_out_ref", first_out_ref)
    print("first_mul_ref", first_mul_ref)
    print("first_out_mul_ref", out_and_multitable_ref)
    print("second_found_ref", second_found_ref)
    print("kw_not_in_best_ref", kw_not_in_best_ref, all_ref - second_found_ref)
    print("not_in_best_ref", not_in_best_ref, all_ref - second_found_ref - kw_not_in_best_ref)

    print("all_ref", all_ref)
    print('#############')
    print('template fail:', template_fail)
    print('except_log_num', except_log_num)
    print('multi table', multi_table)
    print('valid_all', valid_all)

    print("second_not_found", len(second_not_found))
    print("multi_fail", multi_fail)
    print("#1---#1 fail", not_in_best_formu)
    print("*1---*1 fail", kw_not_in_best_formu)
    print("first_fail", first_fail)

    print("first_multi_fail", first_multi_fail)
    print("second_multi_fail", second_multi_fail)

    print('model1_fail_but_suc', model1_fail_but_suc)
    print('model1_fail_but_suc1', model1_fail_but_suc1)
    print('model1_fail_but_suc2', model1_fail_but_suc2)
    print('model1_suc_but_fail', model1_suc_but_fail)
    print('model1_suc', model1_suc)
    print('tmp_all', tmp_all)
    print("gt_not_in_second", gt_not_in_second)
    print('tmp_stat', tmp_stat)
    # with open("second_not_found.json", 'w') as f:
    #     json.dump(second_not_found, f)


def temp():
    formula_token = '112937346618466950862670149644017255835-drv2624-and-drv2625-configuration-tool-and-design-equations_2d00_awa.xlsx---Auto-Calibration---12---3'
    first_res = np.load(root_path + 'first_tile_res_new/' + formula_token + '.npy', allow_pickle=True).item()
    second_res = np.load(root_path + 'second_tile_res_new/' + formula_token + '.npy', allow_pickle=True).item()
    pprint.pprint(first_res)
    # print('second_res', second_res)
    for itemkey in second_res:
        #     print("###################################")
        print('origin key', itemkey)
        best_distance = second_res[itemkey].pop('best_distance')
        best_row_col = second_res[itemkey].pop('best_row_col')

        print('best_distance', best_distance)
        print('best_row_col', best_row_col)
        # break
    #     pprint.pprint(sorted(second_res[itemkey].items(), key=lambda x:x[1]))


def remove_imdata():
    after_filelist = os.listdir(root_path + 'demo_after_features')
    for filename in after_filelist:
        rm_path = root_path + 'demo_before_features/' + filename
        if os.path.exists(rm_path):
            print('remove', rm_path)
            os.remove(rm_path)

    before_filelist = os.listdir(root_path + 'demo_before_features')
    for filename in list(set(after_filelist) | set(before_filelist)):
        rm_path = 'demo_tile_features/' + filename.replace(".npy", ".json")
        if os.path.exists(rm_path):
            print('remove', rm_path)
            os.remove(rm_path)


def look_befor_after_time():
    filelist = os.listdir("/datadrive-2/data/fortune500_test/after_time_test")
    all_before_time = 0
    all_after_time = 0
    for filename in filelist:
        with open("/datadrive-2/data/fortune500_test/after_time_test/" + filename, 'r') as f:
            timeres = json.load(f)

        before_time = timeres['before_time']
        after_time = timeres['after_time']

        all_before_time += before_time
        all_after_time += after_time

    print('before_time', all_before_time / len(filelist))
    print('after_time', all_after_time / len(filelist))

    all_compare_time1 = 0
    filelist = os.listdir(root_path + 'sorted_first_block/')
    for index, filename in enumerate(filelist):
        if index == 49:
            break
        timeres = np.load(root_path + 'sorted_first_block/' + filename, allow_pickle=True).item()
        compare_time = timeres['time']
        all_compare_time1 += compare_time
    print('compare_time1', all_compare_time1 / 50)

    all_compare_time2 = 0
    filelist = os.listdir(root_path + 'sorted_second_block/')
    for index, filename in enumerate(filelist):
        if index == 49:
            break
        timeres = np.load(root_path + 'sorted_second_block/' + filename, allow_pickle=True).item()
        compare_time = timeres['time']
        all_compare_time2 += compare_time
    print('compare_time2', all_compare_time2 / 50)

    avg_tilenum = 0
    filelist = os.listdir(root_path + 'tile_range')
    for filename in filelist:
        with open(root_path + 'tile_rows/' + filename, 'r') as f:
            rows = json.load(f)
        with open(root_path + 'tile_cols/' + filename, 'r') as f:
            cols = json.load(f)
        tile_num = len(rows) * len(cols)
        avg_tilenum += tile_num
    print('avg_tilenum', avg_tilenum / len(filelist))
    print('all timenum', avg_tilenum)


def look_template():
    with open('r1c12template_fortune500_constant.json', 'r') as f:
        r1c12template_fortune500_constant = json.load(f)
    with open('r1c12template_fortune500.json', 'r') as f:
        r1c12template_fortune500 = json.load(f)
    with open("Formulas_fortune500_with_id.json", 'r') as f:
        formulas = json.load(f)

    count = 0
    for index, formula in enumerate(formulas):
        is_found = False
        suc = False

        formula_token = formula['filesheet'] + '---' + str(formula['fr']) + '---' + str(formula['fc'])

        if not os.path.exists(root_path + 'model1_res/' + formula_token + '.json'):
            continue
        if not os.path.exists(root_path + 'after_feature_test/' + formula_token + '.npy'):
            continue
        print('\033[1;35m index ' + str(index) + ' ' + str(len(formulas)) + ' ' + formula_token + '\033[0m')

        if 'RC[-1]+ROUND(10*R' in formula['r1c1']:
            except_log_num += 1
            continue
        with open(root_path + 'model1_res/' + formula_token + '.json', 'r') as f:
            model1_res = json.load(f)

        if formula['r1c1'] not in r1c12template_fortune500:
            gt_template_id = -1
        else:
            gt_template_id = r1c12template_fortune500[formula['r1c1']]
        if model1_res[3] not in r1c12template_fortune500:
            found_template_id = -1
        else:
            found_template_id = r1c12template_fortune500[model1_res[3]]

        if formula['r1c1'] not in r1c12template_fortune500_constant:
            gt_template_id_constant = -1
        else:
            gt_template_id_constant = r1c12template_fortune500_constant[formula['r1c1']]
        if model1_res[3] not in r1c12template_fortune500_constant:
            found_template_id_constant = -1
        else:
            found_template_id_constant = r1c12template_fortune500_constant[model1_res[3]]

        if gt_template_id == found_template_id and found_template_id_constant != gt_template_id_constant:
            count += 1
            if count <= 13:
                continue
            print(model1_res)
            break


def analyze_fail_res_3159():
    with open('fail_res_3159.json', 'r') as f:
        fail_res_3159 = json.load(f)

    gt_rank_dict = {}
    score_del_list = []
    score_del_ratio_list = []
    rank_list = []
    min_del = 0.00000001

    sheetname_dict = {}

    score_del_dict = {'0-2': 0, '2-10': 0, '10-20': 0, '20+': 0}
    delta_ratio_dict = {'0-0.5': 0, '0.5-1': 0, '1-20': 0, '20+': 0}
    distance_dict = {'0-5': 0, '5-10': 0, '10-20': 0, '20+': 0}
    rank_dict = {'2-5': 0, '5-10': 0, '10-20': 0, '20+': 0}
    distance_list = []

    score_0 = 0
    for item in fail_res_3159:
        sheetname = item.split('#####')[0].split('---')[1]
        if sheetname not in sheetname_dict:
            sheetname_dict[sheetname] = 0
        sheetname_dict[sheetname] += 1
        if fail_res_3159[item]['gt_rank'] not in gt_rank_dict:
            gt_rank_dict[fail_res_3159[item]['gt_rank']] = 0
        gt_rank_dict[fail_res_3159[item]['gt_rank']] += 1
        rank_list.append(fail_res_3159[item]['gt_rank'])
        score_del = fail_res_3159[item]['gt_score'] - fail_res_3159[item]['found_score']
        score_del_list.append(score_del)
        if score_del < 2:
            score_del_dict['0-2'] += 1
        elif score_del >= 2 and score_del < 10:
            score_del_dict['2-10'] += 1
        elif score_del >= 10 and score_del < 20:
            score_del_dict['10-20'] += 1
        else:
            score_del_dict['20+'] += 1
        gt_r = int(fail_res_3159[item]['gt_rc'].split('---')[0])
        gt_c = int(fail_res_3159[item]['gt_rc'].split('---')[1])

        found_r = int(fail_res_3159[item]['found_rc'].split('---')[0])
        found_c = int(fail_res_3159[item]['found_rc'].split('---')[1])

        distance = ((gt_r - found_r) ** 2 + (gt_c - found_c) ** 2) ** 0.5
        distance_list.append(distance)
        if fail_res_3159[item]['gt_score'] == fail_res_3159[item]['found_score'] and fail_res_3159[item][
            'gt_score'] == 0:
            score_0 += 1
        if distance < 5:
            distance_dict['0-5'] += 1
        elif distance >= 5 and distance < 10:
            distance_dict['5-10'] += 1
        elif distance >= 10 and distance < 20:
            distance_dict['10-20'] += 1
        else:
            distance_dict['20+'] += 1

        if fail_res_3159[item]['gt_rank'] < 5:
            rank_dict['2-5'] += 1
        elif fail_res_3159[item]['gt_rank'] >= 5 and fail_res_3159[item]['gt_rank'] < 10:
            rank_dict['5-10'] += 1
        elif fail_res_3159[item]['gt_rank'] >= 10 and fail_res_3159[item]['gt_rank'] < 20:
            rank_dict['10-20'] += 1
        else:
            rank_dict['20+'] += 1

        delta_ration = (fail_res_3159[item]['gt_score'] - fail_res_3159[item]['found_score']) / (
                fail_res_3159[item]['found_score'] + min_del)
        score_del_ratio_list.append(delta_ration)
        if delta_ration < 0.5:
            delta_ratio_dict['0-0.5'] += 1
        elif delta_ration >= 0.5 and delta_ration < 1:
            delta_ratio_dict['0.5-1'] += 1
        elif delta_ration >= 1 and delta_ration < 20:
            delta_ratio_dict['1-20'] += 1
        else:
            delta_ratio_dict['20+'] += 1
        # if fail_res_3159[item]['gt_score']-fail_res_3159[item]['found_score'] > 5:
        # if sheetname == 'LEA Y_FFT':
        #     print('item', item)
        #     print('fail_res_3159[item]', fail_res_3159[item])
        #     break
    gt_rank_dict = sorted(gt_rank_dict.items(), key=lambda x: x[0])
    # print('gt_rank_dict', gt_rank_dict)
    print('score_del_list', np.array(score_del_list).mean())
    print('rank_list', np.array(rank_list).mean())
    # sheetname_dict = sorted(sheetname_dict.items(), key=lambda x:x[1])
    # print('sheetname_dict', sheetname_dict)
    print('distance_list', np.array(distance_list).mean())
    print('score_del_ratio_list', np.array(score_del_ratio_list).mean())
    print('score_del_dict', score_del_dict)
    print('distance_dict', distance_dict)
    print('rank_dict', rank_dict)

    print('delta_ratio_dict', delta_ratio_dict)
    print('all', len(fail_res_3159))
    print('score_0', score_0)


def look_copy_distance():
    with open("fail_res_3159.json", 'r') as f:
        fail_res = json.load(f)
    with open("position2distance_dict.json", 'r') as f:
        position2distance_dict = json.load(f)
    shift_dis_list = []
    content_dis_list = []
    selected_dis_list = []
    shift_better = 0
    content_better = 0

    selected_better = 0
    content1_better = 0
    same1 = 0
    same = 0
    for item_key in fail_res:
        model_res = fail_res[item_key]['model_res']
        found_rc = fail_res[item_key]['found_rc']
        gt_rc = fail_res[item_key]['gt_rc']

        selected_key = random.choice(list(fail_res.keys()))
        while selected_key.split("---")[1] == item_key.split('---')[1] or selected_key.split("---")[0] == \
                item_key.split('---')[0]:
            selected_key = random.choice(list(fail_res.keys()))

        origin_filesheet = model_res[0].split("---")[0] + '---' + model_res[0].split("---")[1]
        found_filesheet = model_res[1].split("---")[0] + '---' + model_res[1].split("---")[1]

        target_rc = item_key.split("#####")[1]
        target_ref_cell_token = found_filesheet + '---' + target_rc
        found_ref_cell_token = origin_filesheet + '---' + found_rc
        gt_ref_cell_token = origin_filesheet + '---' + gt_rc
        selected_cell_token = fail_res[selected_key]['model_res'][0].split("---")[0] + '---' + \
                              fail_res[selected_key]['model_res'][0].split("---")[1] + '---' + fail_res[selected_key][
                                  'found_rc']

        target_feature = np.load(root_path + 'demo_after_features/' + target_ref_cell_token + '.npy', allow_pickle=True)
        found_feature = np.load(root_path + 'demo_after_features/' + found_ref_cell_token + '.npy', allow_pickle=True)
        gt_feature = np.load(root_path + 'demo_after_features/' + gt_ref_cell_token + '.npy', allow_pickle=True)
        selected_feature = np.load(root_path + 'demo_after_features/' + selected_cell_token + '.npy', allow_pickle=True)

        target2gt_distance = euclidean(target_feature, gt_feature)  # content difference
        found2gt_distance = euclidean(found_feature, gt_feature)  # shift difference
        selected2target_distance = euclidean(target_feature, selected_feature)

        print("##########")
        print('target2gt_distance', target2gt_distance)
        print('found2gt_distance', found2gt_distance)
        print("selected2target_distance", selected2target_distance)
        shift_dis_list.append(found2gt_distance)
        content_dis_list.append(target2gt_distance)
        selected_dis_list.append(selected2target_distance)
        if selected2target_distance < target2gt_distance:
            selected_better += 1
        elif selected2target_distance > target2gt_distance:
            content1_better += 1
        else:
            same1 += 1

        if found2gt_distance < target2gt_distance:
            shift_better += 1
        elif found2gt_distance > target2gt_distance:
            content_better += 1
        else:
            same += 1

    large_list = []
    small_list = []
    large_better = 0
    small_better = 0
    same2 = 0
    for filesheet in position2distance_dict:
        for rc in position2distance_dict[filesheet]:
            if len(list(position2distance_dict[filesheet][rc])) < 2:
                continue
            position_dis0 = list(position2distance_dict[filesheet][rc])[0]
            position_dis1 = list(position2distance_dict[filesheet][rc])[1]
            if position_dis1 > position_dis0:
                large_position_dis = position_dis1
                small_position_dis = position_dis0
            else:
                large_position_dis = position_dis0
                small_position_dis = position_dis1

            large_list.append(position2distance_dict[filesheet][rc][large_position_dis])
            small_list.append(position2distance_dict[filesheet][rc][small_position_dis])
            if position2distance_dict[filesheet][rc][large_position_dis] > position2distance_dict[filesheet][rc][
                small_position_dis]:
                large_better += 1
            elif position2distance_dict[filesheet][rc][large_position_dis] < position2distance_dict[filesheet][rc][
                small_position_dis]:
                small_better += 1
            else:
                same2 += 1

            large_list.append(position2distance_dict[filesheet][rc][large_position_dis])
            small_list.append(position2distance_dict[filesheet][rc][small_position_dis])

    print('shift_better', shift_better)
    print('content_better', content_better)
    print('same', same)

    print('selected_better', selected_better)
    print('content1_better', content1_better)
    print('same1', same1)
    print('shift_dis_list', np.array(shift_dis_list).mean())
    print('content_dis_list', np.array(content_dis_list).mean())
    print('selected_dis_list', np.array(selected_dis_list).mean())

    print('large_list', np.array(large_list).mean())
    print('small_list', np.array(small_list).mean())
    print('large_better', large_better)
    print('small_better', small_better)
    print('same2', same2)


def check_shift():
    # invalid_feature = np.load('/datadrive-2/data/fortune500_test/demo_after_features/188252747975611141807669656201781967552-15368134-vcs%20-%20dial%20plan%20example%20configuration%20to%20avoid%20interworking.xlsx---Number plan---13501---9111.npy', allow_pickle=True)
    demo_after_list = os.listdir(root_path + 'demo_after_features')
    filesheet2rc = {}
    # not_invalid_token = []
    # invalid_token = []
    # for index, filename in enumerate(demo_after_list):
    #     print("check invalid:", index, len(demo_after_list))
    #     try:
    #         first_feature = np.load(root_path + 'demo_after_features/' +filename, allow_pickle=True)
    #     except:
    #         invalid_token.append(filename)
    #         continue
    #     if euclidean(first_feature, invalid_feature):
    #         not_invalid_token.append(filename)
    #     else:
    #         invalid_token.append(filename)

    # with open('not_invalid_token.json', 'w') as f:
    #     json.dump(not_invalid_token, f)
    # with open('invalid_token.json', 'w') as f:
    #     json.dump(invalid_token, f)
    with open('not_invalid_token.json', 'r') as f:
        not_invalid_token = json.load(f)
    print("not_invalid_token", len(not_invalid_token))
    for filename in not_invalid_token:
        formula_token = filename.replace('.npy', '')
        split_list = formula_token.split("---")
        filesheet = split_list[0] + '---' + split_list[1]
        rc = split_list[2] + '---' + split_list[3]
        if filesheet not in filesheet2rc:
            filesheet2rc[filesheet] = []
        filesheet2rc[filesheet].append(rc)

    # max_check = 1000
    count = 0
    position2distance = []
    position2distance_dict = {}
    # content_dict = {}
    for index, filesheet in enumerate(filesheet2rc):
        position2distance_dict[filesheet] = {}
        print(index, len(filesheet2rc))
        # if count == max_check:
        #     break
        one_filesheet_num = 0
        for rc in filesheet2rc[filesheet]:

            # if count == max_check:
            #     break
            try:
                first_feature = np.load(root_path + 'demo_after_features/' + filesheet + '---' + rc + ".npy",
                                        allow_pickle=True)
            except:
                continue
            # if euclidean(first_feature, invalid_feature) == 0:
            #     continue
            rc_count = 0
            for other_rc in filesheet2rc[filesheet]:
                if rc_count == 2:
                    break
                # print('count', count)
                # if count == max_check:
                #     break
                if rc == other_rc:
                    continue

                r = int(rc.split('---')[0])
                c = int(rc.split('---')[1])
                other_r = int(other_rc.split('---')[0])
                other_c = int(other_rc.split('---')[1])
                position_distance = ((other_r - r) ** 2 + (other_c - c) ** 2) ** 0.5

                second_feature = np.load(root_path + 'demo_after_features/' + filesheet + '---' + other_rc + ".npy",
                                         allow_pickle=True)
                # if euclidean(second_feature, invalid_feature) == 0:
                #     continue
                print('#############')
                print('second_feature', root_path + 'demo_after_features/' + filesheet + '---' + other_rc + ".npy")
                print('first_feature', root_path + 'demo_after_features/' + filesheet + '---' + rc + ".npy")
                feature_distance = euclidean(first_feature, second_feature)
                if feature_distance != 0:
                    if rc not in position2distance_dict[filesheet]:
                        position2distance_dict[filesheet][rc] = {}
                    if str(position_distance) not in position2distance_dict[filesheet][rc]:
                        position2distance_dict[filesheet][rc][str(position_distance)] = []
                    print('feature_distance', feature_distance)
                    position2distance_dict[filesheet][rc][str(position_distance)].append(float(feature_distance))
                    position2distance.append([position_distance, feature_distance])
                    count += 1
                    rc_count += 1
                    # break
            break

    # position2distance = np.array(position2distance)
    # plt.scatter(position2distance[:, 0], position2distance[:, 1])
    # plt.ylim(0,0.5)
    # plt.xlim(0,100)
    # plt.savefig('position2distance.png')
    with open('position2distance_dict.json', 'w') as f:
        json.dump(position2distance_dict, f)


def finetuning_shift_features():
    # with open("shift_finetune_triples.json", 'r') as f:
    # with open("specific_finetune_triples.json", 'r') as f:
    # with open("specific_finetune_triples_5_10.json", 'r') as f:
    # with open("shift_finetune_triples.json", 'r') as f:
    # with open("specific_finetune_triples_1_5.json", 'r') as f:
    #     specific_finetune_triples_1_5 = json.load(f)
    # with open("specific_finetune_triples_5_10.json", 'r') as f:
    #     specific_finetune_triples_5_10 = json.load(f)
    # with open("specific_finetune_triples.json", 'r') as f:
    #     specific_finetune_triples = json.load(f)
    with open("dedup_training_triples.json", 'r') as f:
        specific_finetune_triples = json.load(f)

    # shift_content_finetune_triples = specific_finetune_triples_1_5 + specific_finetune_triples_5_10 + specific_finetune_triples
    def one_process(thread_id, batch_num):
        finetune_filenames = set()
        batch_len = int(len(specific_finetune_triples) / batch_num)
        for index, triple in enumerate(specific_finetune_triples):
            # if index % 1000 == 0:
            #     if len(os.listdir(root_path + 'demo_tile_features_specific/')) >= 100000:
            #         return
            if thread_id != batch_num:
                if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
                    continue
            else:
                if index <= batch_len * (thread_id - 1):
                    continue
            print(index, len(specific_finetune_triples))
            for token in triple:
                filename, sheetname, row, col = token.split('---')
                # finetune_filenames.add(filename)
                filename = filename.split("/")[-1]
                row = int(row)
                col = int(col)
                if os.path.exists(root_path + 'training_tile_features/' + filename + '---' + sheetname + '---' + str(
                        row) + '---' + str(col) + '.json'):
                    continue
                generate_one_demo_features(filename, sheetname, row, col,
                                           save_path=root_path + 'training_tile_features/', cross=True)
            #     break
            # break

    # one_process(1,1)
    process = [
        Process(target=one_process, args=(1, 20)),
        Process(target=one_process, args=(2, 20)),
        Process(target=one_process, args=(3, 20)),
        Process(target=one_process, args=(4, 20)),
        Process(target=one_process, args=(5, 20)),
        Process(target=one_process, args=(6, 20)),
        Process(target=one_process, args=(7, 20)),
        Process(target=one_process, args=(8, 20)),
        Process(target=one_process, args=(9, 20)),
        Process(target=one_process, args=(10, 20)),
        Process(target=one_process, args=(11, 20)),
        Process(target=one_process, args=(12, 20)),
        Process(target=one_process, args=(13, 20)),
        Process(target=one_process, args=(14, 20)),
        Process(target=one_process, args=(15, 20)),
        Process(target=one_process, args=(16, 20)),
        Process(target=one_process, args=(17, 20)),
        Process(target=one_process, args=(18, 20)),
        Process(target=one_process, args=(19, 20)),
        Process(target=one_process, args=(20, 20)),
    ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束
    # with open('shift_finetune_filenames.json', 'w') as f:
    #     json.dump(list(finetune_filenames), f)

    # with open('shift_finetune_filenames.json', 'r') as f:
    #     shift_finetune_filenames = json.load(f)

    # for item in shift_finetune_filenames:
    #     print('#####')
    #     print('/datadrive/projects/Demo/fixed_workbook_json/'+item.split('/')[-1] + '.json')
    #     print('fix_workbook_json', os.path.exists('/datadrive/projects/Demo/fixed_workbook_json/'+item.split('/')[-1] + '.json'))
    #     break


def generate_one_before_feature(workbook_name, sheetname, row, col,
                                source_root_path=root_path + 'demo_tile_features_shift/',
                                saved_root_path=root_path + 'demo_before_features/',
                                bert_dict_file='data_drive/data_one/bert_dict/bert_dict.json',
                                bert_dict_path="data_drive/data_two/bert_dict",
                                content_template_dict_file="json_data/content_temp_dict_1.json", mask=2):
    formula_token = workbook_name + '---' + sheetname + '---' + str(row) + '---' + str(col)

    bert_dict = {}
    try:
        with open(content_template_dict_file, 'r') as f:
            content_tem_dict = json.load(f)
    except Exception as e:
        print(e)
        content_tem_dict = {}
    try:
        with open(source_root_path + formula_token + '.json', 'r') as f:
            origin_feature = json.load(f)
    except Exception as e:
        print('error:generate_one_before_feature')
        print(e)
        return
    temp_bert_dict = {}
    feature_nparray, temp_bert_dict = get_feature_vector_with_bert_keyw(origin_feature, bert_dict, content_tem_dict,
                                                                        mask, bert_dict_path=bert_dict_path,
                                                                        bert_dict_file=bert_dict_file,
                                                                        temp_bert_dict=temp_bert_dict)
    feature_nparray = np.array(feature_nparray)
    np.save(saved_root_path + formula_token + '.npy', feature_nparray)


def generate_test_before_feature_for_finetune(mask):
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)
    with open("json_data/content_temp_dict_1.json", 'r') as f:
        content_tem_dict = json.load(f)

    with open("fail_res_3159.json", 'r') as f:
        fail_res_3159 = json.load(f)

    need_save_token = []
    for key in fail_res_3159:
        formula_token = key.split("#####")[0]
        ref_cell_position = key.split("#####")[1]
        filesheet = formula_token.split('---')[0] + '---' + formula_token.split('---')[1]
        need_save_token.append(filesheet + '---' + ref_cell_position)

    tile_list = os.listdir(root_path + 'tile_rows/')
    for filesheet_json in tile_list:
        with open(root_path + 'tile_rows/' + filesheet_json, 'r') as f:
            tile_rows = json.load(f)
        with open(root_path + 'tile_cols/' + filesheet_json, 'r') as f:
            tile_cols = json.load(f)
        filesheet = filesheet_json.replace('.json', '')
        for row in tile_rows:
            for col in tile_cols:
                tile_token = filesheet + '---' + str(row) + '---' + str(col)
                # print('tile_token', tile_token)
                need_save_token.append(tile_token)

    file2sheet2rc = {}
    for token in need_save_token:
        filename, sheetname, row, col = token.split('---')
        if filename not in file2sheet2rc:
            file2sheet2rc[filename] = {}
        if sheetname not in file2sheet2rc[filename]:
            file2sheet2rc[filename][sheetname] = []
        file2sheet2rc[filename][sheetname].append([row, col])

    demo_tile_feature_path = root_path + 'demo_tile_features_fix/'
    if not mask:
        demo_before_feature_path = root_path + 'demo_before_features_fix/'
    if mask:
        demo_before_feature_path = root_path + 'demo_before_features_mask_fix/'
    for index, filename in enumerate(file2sheet2rc):
        print(index, len(file2sheet2rc))
        if os.path.exists('../Demo/fix_fortune500/' + filename + '.json'):
            with open('../Demo/fix_fortune500/' + filename + '.json', 'r') as f:
                workbook_json = json.load(f)
        elif os.path.exists('../Demo/origin_fortune500_workbook_json/' + filename + '.json'):
            with open('../Demo/origin_fortune500_workbook_json/' + filename + '.json', 'r') as f:
                workbook_json = json.load(f)
        else:
            continue
        for sheetname in file2sheet2rc[filename]:
            for row_col in file2sheet2rc[filename][sheetname]:
                row = int(row_col[0])
                col = int(row_col[1])
                formula_token = filename + '---' + sheetname + '---' + row_col[0] + '---' + row_col[1]
                if not os.path.exists(demo_tile_feature_path + formula_token + '.json'):
                    generate_demo_features(filename, sheetname, workbook_json, row, col,
                                           save_path=demo_tile_feature_path, is_look=True)
                if not os.path.exists(demo_before_feature_path + formula_token + '.npy'):
                    generate_one_before_feaure(formula_token, bert_dict, content_tem_dict, mask,
                                               source_root_path=demo_tile_feature_path,
                                               saved_root_path=demo_before_feature_path)


def change_word_to_save_word(word):
    save_word = word.replace('/', '*#*line*#*')
    return save_word


def change_save_word_to_word(save_word):
    word = save_word.replace('*#*line*#*', '/')
    return word


def generate_bert_dict():
    with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
        bert_dict = json.load(f)
    new_bert_dict = {}
    for key in bert_dict:
        print("saving ", key)
        save_key = change_word_to_save_word(key)
        try:
            np.save("/datadrive-2/data/bert_dict/" + save_key + '.npy', bert_dict[key])
        except:
            new_bert_dict[key] = bert_dict[key]
    with open('/datadrive/data/bert_dict/bert_dict.json', 'w') as f:
        json.dump(new_bert_dict, f)


def check_shift_feature():
    saved_root_path = root_path + 'training_before_shift_c2/'
    source_root_path = root_path + 'demo_tile_features_shift/'
    filenames = os.listdir(source_root_path)
    filenames.sort()
    count = 0
    for item in filenames:
        need_num = 0
        if os.path.exists(saved_root_path + item.replace('.json', '.npy')):
            count += 1
            first_filejson = item
            first_filenpy = first_filejson.replace('.json', '.npy')

            with open(source_root_path + first_filejson, 'r') as f:
                filejson = json.load(f)
            for index, item in enumerate(filejson):
                if item['content'] is not None:
                    if is_number(item['content']) == False:
                        print(item['content'])
                        continue
                    print("item", item['content'])
                    need_num += 1
                    if need_num == 1:
                        found_index1 = index
                    if need_num == 2:
                        found_index2 = index
                    # print(is_number(item['content'] ))
                if need_num == 2:
                    break
            if need_num == 2:
                before_feature = np.load(saved_root_path + first_filenpy, allow_pickle=True)
                one_feature1 = before_feature[found_index1]
                one_feature2 = before_feature[found_index2]
                # print(one_feature1[13:-1])
                # print(one_feature2[13:-1])
                # print(len(one_feature1[13:-1]))
                # print(len(one_feature2[13:-1]))
                print(one_feature1[13:-1] == one_feature2[13:-1])
                break
    # print('filejson', first_filejson)
    # for index, item in enumerate(filejson):
    #     try:
    #         print(filejson[0]['content'])
    #         print(is_number(filejson[0]['content']))
    #     except:
    #         continue
    # print(filejson['sheetfeatures'])
    # print(before_feature.shape)


def generate_similar_sheet_eval_data():
    with open("../Mondrian-master/filtered_list.json", 'r') as f:
        filtered_list = json.load(f)
    with open('../AnalyzeDV/fortune500_sheetname_2_file_devided.json', 'r') as f:
        sheetname_2_file = json.load(f)
    # print(list(sheetname_2_file.keys())[0])
    # print(sheetname_2_file[list(sheetname_2_file.keys())[0]])
    # positive_list = []
    # negative_list = []
    cluster = []
    ne = 0
    for index, filesheet in enumerate(filtered_list):
        print(index, len(filtered_list))
        sheetname = filesheet.split('---')[1].replace('.csv', '')
        if sheetname not in sheetname_2_file:
            ne += 1
            print('not exists in devided', filesheet, sheetname)
            continue

        for clst in sheetname_2_file[sheetname]:
            if filesheet in clst['filenames']:
                new_clst = []
                for filename in clst['filenames']:
                    new_clst.append(filename + '---' + sheetname)
                new_clst.sort()
                is_in = False

                for other_clst in cluster:
                    if len(other_clst) != len(new_clst):
                        continue
                    other_clst.sort()
                    is_same = True
                    for ind in range(0, len(new_clst)):
                        if other_clst != new_clst:
                            is_same = False
                            break
                    if not is_same:
                        continue
                    is_in = True
                    break
                if is_in:
                    continue
                else:
                    cluster.append(new_clst)
    print('cluster', cluster)
    print('ne', ne)


def look_loss_val():
    # first_losslog = np.load("/datadrive-2/data/finetune_specific_l2_new/losslog.npy", allow_pickle=True).item()
    # second_losslog = np.load("/datadrive-2/data/finetune_specific_l2_5_10/losslog.npy", allow_pickle=True).item()
    # thirld_losslog = np.load("/datadrive-2/data/finetune_specific_l2_1_5/losslog.npy", allow_pickle=True).item()
    first_losslog = np.load("/datadrive-2/data/cross_finetune_specific_l2/losslog.npy", allow_pickle=True).item()
    # print('first_losslog', first_losslog)
    # print('second_losslog', second_losslog)
    # print('thirld_losslog', thirld_losslog)

    loss_list = []
    first_y = [float(first_losslog[item][0].detach()) for item in first_losslog]
    first_x = list(first_losslog.keys())
    # second_y =[float(second_losslog[item][0].detach()) for item in second_losslog]
    # second_x = list(second_losslog.keys())
    # third_y =[float(thirld_losslog[item][0].detach()) for item in thirld_losslog]
    # third_x = list(thirld_losslog.keys())

    epoch_x = range(1, 11)
    for epoch_id in range(1, 11):
        loss_list.append(float(first_losslog[epoch_id][0].detach()))
    # epoch_x = range(1,64)
    # for epoch_id in range(1,23):
    #     loss_list.append(float(first_losslog[epoch_id][0].detach()))
    # for epoch_id in range(23,42):
    #     loss_list.append(float(second_losslog[epoch_id][0].detach()))
    # for epoch_id in range(42,64):
    #     loss_list.append(float(thirld_losslog[epoch_id][0].detach()))
    # print(len(loss_list))
    validate_y = []

    validate_x = []

    for epoch_id in epoch_x:
        if os.path.exists('/datadrive-3/data/fortune500_test/validate_cross/' + str(epoch_id) + '/accuracy.json'):
            with open('/datadrive-3/data/fortune500_test/validate_cross/' + str(epoch_id) + '/accuracy.json', 'r') as f:
                accuracy = json.load(f)[2]
            validate_y.append(accuracy)
            validate_x.append(epoch_id)
    print('loss_list', loss_list)
    print('validate_y', validate_y)

    plt.plot(epoch_x, loss_list)
    plt.plot(validate_x, validate_y)
    plt.savefig("loss_cross_validate.png")

    with open("fortune500_val_formula.json", 'r') as f:
        fortune500_val_formula = json.load(f)
    filelist = set([item.split('---')[0] for item in fortune500_val_formula])
    sheetlist = set([item.split('---')[1] for item in fortune500_val_formula])

    print('filelist', len(filelist))
    print('sheetlist', len(sheetlist))
    # loss_list.append(first_losslog[epoch_id][0].detach())


def look_accuracy():
    # filelist = os.listdir("/datadrive-3/data/fortune500_test/validate_cross/")
    filelist = os.listdir("/datadrive-3/data/fortune500_test/validate/")
    filelist.sort()
    for filename in filelist:

        if os.path.exists("/datadrive-3/data/fortune500_test/validate/" + filename + '/accuracy.json'):
            with open("/datadrive-3/data/fortune500_test/validate/" + filename + '/accuracy.json', 'r') as f:
                accuracy = json.load(f)
                print(filename)
                print(accuracy)


def extract_sheet_wh_info(sheetname, wb_json):
    wb_json = wb_json['Sheets']
    sheet_wh_info = {'height': {}, "width": {}}
    for sheet in wb_json:
        if sheet['Name'] == sheetname:
            found_sheet = sheet

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


def batch_deal_invalid():
    filelist = os.listdir(root_path + 'demo_tile_features_fix/')
    filelist.sort()
    filename2sheetname = {}
    for fname in filelist:
        filename = fname.split('---')[0]
        sheetname = fname.split('---')[1]
        formula_token = fname.replace('.json', '')
        if filename not in filename2sheetname:
            filename2sheetname[filename] = {}
        if sheetname not in filename2sheetname[filename]:
            filename2sheetname[filename][sheetname] = []
        filename2sheetname[filename][sheetname].append(formula_token)

    for index, filename in enumerate(filename2sheetname):
        print(index, len(filename2sheetname))
        with open('../Demo/fix_fortune500/' + filename + '.json', 'r') as f:
            wb_json = json.load(f)
        for index1, sheetname in enumerate(filename2sheetname[filename]):
            print('    ', index1, len(filename2sheetname[filename]))
            sheet_wh_info = extract_sheet_wh_info(sheetname, wb_json)
            for formula_token in filename2sheetname[filename][sheetname]:
                deal_invalid_cell(formula_token, sheet_wh_info)


def deal_invalid_cell(formula_token, sheet_wh_info):
    def check_invalid_cell(feature):
        if (feature['background_color_r'] == 0 and \
            feature['background_color_g'] == 0 and \
            feature['background_color_b'] == 0 and \
            feature['font_color_r'] == 0 and \
            feature['font_color_g'] == 0 and \
            feature['font_color_b'] == 0 and \
            feature['font_size'] == 0 and \
            feature['font_strikethrough'] == False and \
            feature['font_shadow'] == False and \
            feature['font_ita'] == False and \
            feature['font_bold'] == False and \
            feature['height'] == 0 and \
            feature['width'] == 0 and \
            feature['content'] is None and \
            feature['content_template'] is None) or \
                (feature['background_color_r'] == 0 and \
                 feature['background_color_g'] == 0 and \
                 feature['background_color_b'] == 0 and \
                 feature['font_color_r'] == 255 and \
                 feature['font_color_g'] == 255 and \
                 feature['font_color_b'] == 255 and \
                 feature['font_size'] == 11 and \
                 feature['font_strikethrough'] == False and \
                 feature['font_shadow'] == False and \
                 feature['font_ita'] == False and \
                 feature['font_bold'] == False and \
                 feature['height'] == 0 and \
                 feature['width'] == 0 and \
                 feature['content'] is None and \
                 feature['content_template'] is None):
            return True
        return False

    filename = formula_token.split('---')[0]
    sheetname = formula_token.split('---')[1]
    fr = int(formula_token.split('---')[2])
    fc = int(formula_token.split('---')[3])
    new_start_row = fr - 50
    new_start_col = fc - 5
    try:
        with open(root_path + 'demo_tile_features_fix/' + formula_token + '.json', 'r') as f:
            json_features = json.load(f)
    except:
        os.remove(root_path + 'demo_tile_features_fix/' + formula_token + '.json')
        return

    index = 0
    for row in range(new_start_row, new_start_row + 100):
        for col in range(new_start_col, new_start_col + 10):
            # print('row, col', row, col)
            if len(json_features[index]) == 1:
                json_features[index] = json_features[index][0]
            if check_invalid_cell(json_features[index]):
                json_features[index]['font_color_r'] = 255
                json_features[index]['font_color_g'] = 255
                json_features[index]['font_color_b'] = 255
                json_features[index]['font_size'] = 11
                # print(sheet_wh_info['height'], sheet_wh_info['height'] )
                if row in sheet_wh_info['height']:
                    # print(sheet_wh_info['height'], sheet_wh_info['height'] )
                    json_features[index]['height'] = sheet_wh_info['height'][row]
                    # print('exists row')
                if col in sheet_wh_info['width']:
                    json_features[index]['width'] = sheet_wh_info['width'][col]
                    # print('exists col')
            index += 1

    # with open(root_path + 'demo_tile_features_fix/' + formula_token + '.json', 'w') as f:
    #     json.dump(json_features, f)


# def all_fortune500():

if __name__ == '__main__':
    # batch_deal_invalid()
    # look_loss_val()
    # 1step
    # anaylze_training_range()
    # 2step
    devide_training_range_recheck()
    check_training_formula()
    # count_formula()
    # 3step
    resave_training_mergerange()
    # 4step
    # generate_file_sheet()
    # 5step
    # generate_formula_features_by_workbook_json(1,1)
    # 6step
    # para_gen_before_feature()
    # with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
    #     bert_dict = json.load(f)

    # with open("json_data/content_temp_dict_1.json", 'r') as f:
    #     content_tem_dict = json.load(f)
    # generate_before_features(bert_dict,content_tem_dict, 1,1)
    # 8step
    # para_save_after()
    # save_bert_feature_for_77721(1,1)
    # 9step
    # look_sheetname2num()
    # 10step
    # generate_test_formulas()
    # generate_test_formulas()

    # similar sheet
    # para_run_sheet_feature()
    # para_gen_similarsheet_before_feature()

    # with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
    #     bert_dict = json.load(f)

    # with open("json_data/content_temp_dict_1.json", 'r') as f:
    #     content_tem_dict = json.load(f)
    # generate_similarsheet_before_features(bert_dict,content_tem_dict, 1,1)
    # para_sheet_features_save_after()
    # para_neighbors_run()
    # delete_no_func_formulas(1,1)
    # para_run()

    # look_fromula()
    # save_training_r1c1()
    # generate_sheet_features_by_workbook_json(1,1)
    # 1step
    # batch_anaylze_range()
    # 2step
    # devide_range_recheck()
    # check_formula()
    # count_formula()
    # 3 step
    # devide_range()
    # look_mergerange()
    # best_recall()
    # best_sketch_recall()
    # best_sketch_recall_with_sheetsimilarity()
    # generate_faild_case()
    # look_all_sheetname_2_num()
    # look_sheetfeatures()
    # look_savedsheets()
    # devide_range_recheck()
    # resave_mergerange()
    # look_sketch_all_sheetname_2_num()
    # save_files()
    # save_all_r1c1()
    # download_sampled_file()
    # anaylze_range()
    # template_analyze()
    # formula_token_2_domain()
    # all_filename_2_domain()
    # count_all_r1c1()
    # check_formula()
    # count_formula()
    # mv_small_count_formula_features()
    # look_failed_case(1, 2)
    # count_model_bad()
    # look_remove_formulas()
    # best_hasfunc_recall_with_sheetsimilarity()
    # best_sketch_recall_with_sheetsimilarity(1,1)
    # select_middle10_domain()
    # anaylze_training_range()
    # para_sketch_best_recall()
    # all_filesheet_2_domain()
    # sort_domain2num()
    # select_top10_domain()
    # no_saved_filesheet()
    # generate_training_files()

    # para_tile_run()
    # para_gen_tile_before_feature()

    # para_refcell_run()
    # para_gen_refcell_before_feature()

    # para_tile_demo()
    # find_best_tile()
    # para_tile_demo_first()
    # para_refcell_demo()
    # count_demo_recell()
    # move_demo_tile_features()

    # find_second_best()
    # check_best()
    # analyze_fail_res_3159()
    # remove_imdata()
    # temp()
    # look_features()
    # generate_demo_before_features(bert_dict,content_tem_dict,1,1)
    # para_demo_before() 
    # generate_similar_sheet_eval_data()
    # check_shift_feature()
    # para_save_after()
    # save_bert_feature_for_77721(1,1)
    # para_tile_demo_second()
    # generate_one_demo_features("163350182546811267831688234176624693724-to20-ry2022-attachment-d.xlsx", "5-CostofCap-4",26,4)
    # generate_one_demo_features("51062703087863555942928902850393994897-to20-model_gs_ry2022_grossloadry2021.xlsx", "20-RevenueCredits",14, 5)
    # generate_one_demo_features("127534955648942888021337347766059868298-to20-ry2022-attachment-b.xlsx","12-DepRates",10,8)
    # generate_one_demo_features("51062703087863555942928902850393994897-to20-model_gs_ry2022_grossloadry2021.xlsx","12-DepRates",10,8)
    # generate_one_demo_features("162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx","EVM_1_3 (DDR3)",11,2)
    # generate_one_demo_features("162126785355680924912775347171635038280-8053.ratioseed_5f00_am335x_5f00_boards.xlsx","beagleLT(DDR3)",11,2)

    # look_accuracy()
    # para_tile_demo_second_block()
    # test_time()
    # test_time()

    # sort_first_block()
    # naive_one_by_one()
    # sort_second_block()
    # look_template()
    # look_befor_after_time()
    # look_copy_distance()

    # check_shift()
    # finetuning_shift_features()
    # para_demo_before() 
    # generate_test_before_feature_for_finetune(mask=True)
    # generate_bert_dict()
    # generate_test_before_feature_for_finetune(mask=False)

    # generate_mondrain_sheet_features_by_workbook_json()
