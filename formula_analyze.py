import torch
from torch.autograd import Variable
from core_py.analyze_formula import generate_view_json, generate_one_before_feature, generate_one_after_feature, \
    generate_demo_features, get_feature_vector_with_bert_keyw, euclidean
import json
import os
import numpy as np
import faiss
import shutil
from generateJson import GenerateJsonMsg
from utils.excel_tool import GetExcelMsg

class FormulaCleasing:
    def __init__(self, origin_data_formulas_path, save_group_path, save_merge_path, save_merge_new_path, save_res_path):
        self.origin_data_formulas_path = origin_data_formulas_path # ../data_set/formula_data_set/origin_data_formulas_+ str(58) + ".json"
        self.save_group_path = save_group_path # ../data_set/formula_data_set/origin_fortune500_groupby_r1c1.json
        self.save_merge_path = save_merge_path # ../data_set/formula_data_set/origin_fortune500_mergerange.json
        self.save_merge_new_path = save_merge_new_path # ../data_set/formula_data_set/origin_fortune500_mergerange_new_res_1.json
        self.save_res_path = save_res_path # ../data_set/formula_data_set/Formulas_fortune500_with_id.json

    def find_formula(self):
        self.anaylze_training_range()
        self.devide_training_range_recheck()
        self.check_training_formula()
        self.resave_training_mergerange()
        print('end of cleansing')


    def anaylze_training_range(self):
        new_res = {}
        with open(self.origin_data_formulas_path, 'r', encoding='utf-8') as f:
            training_formulas = json.load(f)

        for file_sheet_name in list(training_formulas.keys()):
            new_res[file_sheet_name] = {}
            for formula in training_formulas[file_sheet_name]:
                if formula['formulaR1C1'] not in new_res[file_sheet_name]:
                    new_res[file_sheet_name][formula['formulaR1C1']] = []
                formu = {}
                formu['column'] = formula['column']
                formu['row'] = formula['row']
                new_res[file_sheet_name][formula['formulaR1C1']].append(formu)
        # print(len(new_res))
        # print(new_res)
        with open(self.save_group_path, 'w') as f:
            json.dump(new_res, f)

    def devide_training_range_recheck(self):
        with open(self.save_group_path, 'r') as f:
            formulas_20000sheets = json.load(f)

        result = {}
        count = 0
        for file_sheet in formulas_20000sheets:
            result[file_sheet] = {}
            count += 1
            # print(count, len(formulas_20000sheets))
            for r1c1 in formulas_20000sheets[file_sheet]:
                id_ = 0
                res = {}
                item = {'fr': 0, 'fc': 0, 'lr': 0, 'lc': 0, 'r1c1': r1c1, 'formulas': []}
                for formula in formulas_20000sheets[file_sheet][r1c1]:
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

                if item['fr'] != 0:
                    res[id_] = item
                result[file_sheet][r1c1] = res
        with open(self.save_merge_path, 'w') as f:
            json.dump(result, f)

    def check_training_formula(self):
        with open(self.save_merge_path, 'r') as f:
            res = json.load(f)

        multi_res = {}
        index = 0
        for filesheet in res:
            index += 1
            multi_res[filesheet] = {}
            # print(index, len(res))
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
                    for another_batch_id in res[filesheet][r1c1]:
                        if batch_id == another_batch_id:
                            continue
                        if res[filesheet][r1c1][batch_id]['fc'] == res[filesheet][r1c1][another_batch_id]['fc'] and \
                                res[filesheet][r1c1][batch_id]['lc'] == res[filesheet][r1c1][another_batch_id]['lc'] and \
                                res[filesheet][r1c1][another_batch_id]['fr'] == res[filesheet][r1c1][batch_id]['lr'] + 1:
                            res[filesheet][r1c1][batch_id]['lr'] = res[filesheet][r1c1][another_batch_id]['lr']
                        new_res[batch_id] = res[filesheet][r1c1][batch_id]
                    if batch_id not in new_res:
                        new_res[batch_id] = res[filesheet][r1c1][batch_id]
                    multi_res[filesheet][r1c1][batch_id] = {}
                    multi_res[filesheet][r1c1][batch_id]['fc'] = new_res[batch_id]['fc']
                    multi_res[filesheet][r1c1][batch_id]['fr'] = new_res[batch_id]['fr']
                    multi_res[filesheet][r1c1][batch_id]['lc'] = new_res[batch_id]['lc']
                    multi_res[filesheet][r1c1][batch_id]['lr'] = new_res[batch_id]['lr']
        with open(self.save_merge_new_path, 'w') as f:
            json.dump(multi_res, f)

    def resave_training_mergerange(self):

        with open(self.save_merge_path, 'r') as f:
            formulas_20000sheets = json.load(f)

        res = []

        filesname = set()
        no_formula = 0
        count = 0

        formula_id = 1
        for file_sheet_name in formulas_20000sheets:
            count += 1
            # print(count, len(set(formulas_20000sheets.keys())))
            if len(formulas_20000sheets[file_sheet_name].keys()) == 0:
                no_formula += 1
            for r1c1 in formulas_20000sheets[file_sheet_name]:
                for id_ in formulas_20000sheets[file_sheet_name][r1c1]:
                    formula = {}
                    filesname.add(file_sheet_name)
                    formula['id'] = formula_id
                    formula['filesheet'] = file_sheet_name
                    formula['r1c1'] = r1c1
                    formula['fr'] = formulas_20000sheets[file_sheet_name][r1c1][id_]['fr']
                    formula['fc'] = formulas_20000sheets[file_sheet_name][r1c1][id_]['fc']
                    formula['lr'] = formulas_20000sheets[file_sheet_name][r1c1][id_]['lr']
                    formula['lc'] = formulas_20000sheets[file_sheet_name][r1c1][id_]['lc']
                    formula_id += 1
                    res.append(formula)
        with open(self.save_res_path, 'w') as f:
            json.dump(res, f)