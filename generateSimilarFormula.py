import json
import os
import re
import shutil
import sys

import numpy as np

from core_py.formulae2e import euclidean


class SimilarFormulaGenerator:
    """
    用来找到相似的公式
    """

    def __init__(self, root_path, load_path, save_path, workbooks_path,
                 constrain_workbooks_path, data_set_path, data_set_name,
                 ana_path, file_name, sheet_name, row, col):
        # workbooks ../analyze-dv-1/data_set_name/reduced_formulas_small.json
        # constrain_workbooks ../analyze-dv-1/data_set_name/small_data_set_workbook.json
        # root_path = '../data_drive/data_two/data_set_name/'
        self.root_path = root_path
        self.load_path = load_path
        self.save_path = save_path
        self.workbooks_path = workbooks_path
        self.constrain_workbooks_path = constrain_workbooks_path
        self.data_set_path = data_set_path
        self.data_set_name = data_set_name
        self.ana_path = ana_path
        self.file_name = file_name
        self.sheet_name = sheet_name
        self.row = row
        self.col = col
        self.file_sheet_name_51_6 = file_name + '---' + sheet_name + '---51---6.npy.json'
        self.target_sheet_name = file_name + '---' + sheet_name + f'---{row}---{col}.npy'

    def find_closed_formula(self, res_formula_sheet):
        # similar_sheets_path = self.root_path + self.data_set_name +  '_similar_res/',
        # save_filepath = self.root_path + self.data_set_name + '_formulas_dis/',
        testpath = self.root_path + 'after_feature_test/'
        filepath = self.root_path + 'after_feature/'
        target_feature = np.load(testpath + self.target_sheet_name, allow_pickle=True)
        if len(target_feature) == 1:
            target_feature = target_feature[0]
        # 定义单调栈存储结果
        res_stack = []
        topK = 1
        for formula in res_formula_sheet:
            file_sheet = formula['filesheet']
            row = formula['fr']
            col = formula['fc']
            file_sheet_name = file_sheet + '---' + str(row) + '---' + str(col) + '.npy'
            other_formula_feature = np.load(filepath + file_sheet_name, allow_pickle=True)
            if len(other_formula_feature) == 1:
                other_formula_feature = other_formula_feature[0]
            formula_dis = euclidean(target_feature, other_formula_feature)
            formula['dis'] = formula_dis
            # 进行单调栈的入栈操作
            while res_stack and formula_dis <= res_stack[-1]['dis']:
                res_stack.pop()
            res_stack.append(formula)
            if len(res_stack) > topK:
                res_stack = res_stack[:topK]
        for single_recommand_formula in res_stack:
            print('------------------------')
            print(f"推荐结果如下 excel表名和对应的sheet名称为:{single_recommand_formula['filesheet']} ")
            print(f"在此文件的第{single_recommand_formula['fr']}行 第{single_recommand_formula['fc']}列")
            print(f"推荐的公式结果为:{single_recommand_formula['r1c1']}")
            print('------------------------')
            # pattern = r"\.xlsx.*$"  # 匹配以 .xlsx 开头的内容直到行尾的部分
            # result = re.sub(pattern, ".xlsx", dict_item)
            # print(f'本系统推荐的excel文件名为: {result}')
            # index = index + 1
            # if index > top_k:
            #     break

    def find_only_closed_formula(self, thread_id, batch_num):
        """
        目的就是获取相似表下所有有公式的，并且生成npy文件，返回一个距离最近的
        :param thread_id:
        :param batch_num:
        :return:
        """
        similar_sheets = self.root_path + self.data_set_name +  '_similar_res/',
        save_filepath = self.root_path + self.data_set_name + '_formulas_dis/',
        testpath = self.root_path + 'after_feature_test/',
        filepath = self.root_path + 'after_feature/'
        # with open('Formula_77772_with_id.json','r') as f:
        #     formulas = json.load(f)
        # with open('Formula_hasfunc_with_id.json','r') as f:
        #     formulas = json.load(f)
        # with open('Formulas_test_top10domain_with_id.json','r') as f:
        #     formulas = json.load(f)

        # 从生成的formulas读取公式,在这里面读取相似表的
        with open('./data_set/' + self.data_set_name + '/save_formulas_jsons/Formulas_with_id.json', 'r') as f:
            formulas = json.load(f)

        ne_count = 0
        filesheet2token = {}
        for formula in formulas:
            if formula['filesheet'].split('/')[-1] not in filesheet2token:
                filesheet2token[formula['filesheet'].split('/')[-1]] = []
            # print("xxxx", formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' +  str(formula['fc']))
            filesheet2token[formula['filesheet'].split('/')[-1]].append(
                formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc']))
        count = 0
        print(len(formulas))
        resset = set()

        batch_len = len(formulas) / batch_num
        for index, formula in enumerate(formulas):
            # 如果启动多线程需要取消注释
            # if index != batch_num:
            #     if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
            #         continue
            # else:
            #     if index <= batch_len * (thread_id - 1 ):
            #         continue
            formula_token = formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' + str(
                formula['fc'])
            file_name = formula['filesheet']
            # formula_token = formula_token.replace(".xls.xlsx", '.xlsx')
            # 需要在testpath下存在
            if not os.path.exists(testpath[0] + formula_token + '.npy'):
                ne_count += 1
                continue
            # 在save_path下存在
            if os.path.exists(save_filepath[0] + formula_token + '.npy'):
                continue

            # if os.path.exists(save_filepath +'/' + formula_token + '.npy'):
            # continue
            # if formula_token != '012dcc10c41410112265d9556ec89bb1_d3d3LmFlci5nb3YuYXUJMTUyLjkxLjUzLjE5Mw==.xls.xlsx---Analysis---10---7':
            # continue
            print('count', index)
            filesheet = formula['filesheet'].split('/')[-1]

            res = {}

            if not os.path.exists(similar_sheets + filesheet + '.json'):
                continue
            with open(similar_sheets + filesheet + '.json', 'r') as f:
                similar_sheet_feature = json.load(f)
            target_feature = np.load(filepath + '/' + formula_token + '.npy', allow_pickle=True)
            if len(target_feature) == 1:
                target_feature = target_feature[0]
            for similar_sheet_pair in similar_sheet_feature:
                similar_sheet = similar_sheet_pair[0].replace('---51---6.npy', '')
                if similar_sheet == filesheet:
                    continue
                if similar_sheet not in filesheet2token:
                    continue
                for other_formula_token in filesheet2token[similar_sheet]:
                    other_formula_token = other_formula_token.replace(".xls.xlsx", '.xlsx')
                    try:
                        other_formula_feature = np.load(filepath + '/' + other_formula_token + '.npy',
                                                        allow_pickle=True)
                    except:
                        continue

                    print("Xxxxxxxxx")
                    if len(other_formula_feature) == 1:
                        other_formula_feature = other_formula_feature[0]
                    formula_dis = euclidean(target_feature, other_formula_feature)
                    res[other_formula_token] = formula_dis
            np.save(save_filepath + '/' + formula_token + '.npy', res)
        print('ne_count', ne_count)

    def para_only_run_eval(self):
        """
        运行find_similar
        :return:
        """
        file_path = self.root_path + self.data_set_name + '_similar_res'
        load_path = self.root_path + self.data_set_name + '_similar_res'
        save_path = self.root_path + self.data_set_name + '_model1_res'
        # workbook加载的是formulas的集合
        with open(self.workbooks_path, 'r') as f:
            workbooks = json.load(f)
        with open(self.constrain_workbooks_path, 'r') as f:
            constrain_workbooks = json.load(f)
        self.sort_only_most_similar_formula(1, 1, 1, file_path, load_path, save_path, constrain_workbooks, workbooks)

    def sort_only_most_similar_formula(self, topk, thread_id, batch_num, filepath, load_path, save_path,
                                       constrain_workbooks,
                                       workbooks):  # l2_most_simular_sheet_1900
        """
        :param topk: 返回topk结果
        :param thread_id: 线程Id
        :param batch_num: batch_number
        :param filepath:
        :param load_path:
        :param save_path: 结果需要保存的路径
        :param constrain_workbooks: 对应的json
        :param workbooks: 去重后的
        :return: 返回找到的topK个公式
        """
        with open(self.data_set_path + self.data_set_name + '_formulatoken2r1c1.json', 'r') as f:
            top10domain_formulatoken2r1c1 = json.load(f)
        # with open(self.data_set_path + 'dedup_workbooks.json', 'r') as f:
        with open(self.ana_path + 'dedup_workbooks.json', 'r') as f:
            dedup_workbooks = json.load(f)
        with open(self.data_set_path + 'save_formulas_jsons/' + 'Formulas_with_id.json', 'r') as f:
            formulas = json.load(f)

        num = 0
        batch_len = len(formulas) / batch_num
        count = 0
        ne_cout = 0
        ne_cout1 = 0
        ne_cout2 = 0
        for index, formula in enumerate(formulas):
            if index != batch_num:
                if (index <= batch_len * (thread_id - 1) or index > batch_len * thread_id):
                    continue
            else:
                if index <= batch_len * (thread_id - 1):
                    continue
            formula_token = formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' + str(
                formula['fc'])
            if formula_token not in workbooks:
                continue

            if not os.path.exists(self.root_path + 'after_feature_test/' + formula_token + '.npy'):
                if not os.path.exists(self.root_path + 'after_feature/' + formula_token + '.npy'):
                    ne_cout2 += 1
                    continue
                shutil.copy(self.root_path + 'after_feature/' + formula_token + '.npy',
                            self.root_path + 'after_feature_test/' + formula_token + '.npy')
                if not os.path.exists(self.root_path + 'after_feature_test/' + formula_token + '.npy'):
                    ne_cout2 += 1
                    continue
            if not os.path.exists(self.load_path + '/' + formula_token + '.npy'):
                ne_cout1 += 1
                continue
            count += 1
            # print(count)
            if os.path.exists(self.save_path + '/' + formula_token + '.json'):
                ne_cout += 1
                continue
            print(self.root_path + 'after_feature_test/' + formula_token + '.npy')
            formula_item = np.load(self.load_path + '/' + formula_token + '.npy', allow_pickle=True).item()
            res_list = sorted(formula_item.items(), key=lambda x: x[1], reverse=False)
            new_res = {}
            for tuple in res_list:
                new_res[tuple[0]] = tuple[1]

            res_list = new_res
            found_ = False
            found_formula_token = ''
            found_r1c1 = ''
            print('len res_list', len(res_list))
            if len(res_list) == 0:
                continue
            for other_formula_token in res_list:
                other_file = other_formula_token.split('---')[0]
                other_filename = other_formula_token.split('---')[0]

                if other_filename not in dedup_workbooks and other_filename not in constrain_workbooks:
                    continue

                # print('itemfile', itemfile)
                # print('other_file', other_file)
                # if other_file == itemfile:
                # print('other_formula_token', other_formula_token)
                # print('item', item)
                # print("other_formula['filesheet']", other_formula['filesheet'])
                # if other_formula_token != item:
                #     continue
                # print('candidate')
                print("formula['r1c1']", formula['r1c1'])
                print("other_formula['r1c1']", found_r1c1)
                found_formula_token = other_formula_token
                if found_formula_token in top10domain_formulatoken2r1c1:
                    found_r1c1 = top10domain_formulatoken2r1c1[found_formula_token]
                else:
                    found_r1c1 = ''
                if formula['r1c1'] == found_r1c1:
                    found_ = True

                    break

                    # print("formula['r1c1']", formula['r1c1'])
                    # print("other_formula['r1c1']", other_formula['r1c1'])
            # print('found_', found_)
            if found_:
                print("\033[1;32m found \033[0m")
                with open(save_path + '/' + formula_token + '.json', 'w') as f:
                    json.dump([formula_token, found_formula_token, formula['r1c1'], found_r1c1, True], f)
                # found_res.append(formula['id'])
            else:
                print("\033[1;33m not found \033[0m")
                with open(save_path + '/' + formula_token + '.json', 'w') as f:
                    json.dump([formula_token, found_formula_token, formula['r1c1'], found_r1c1, False], f)
            # break
            # not_found_res.append(formula['id'])
        print('count', count)
        print('exists', ne_cout)
        print('no load', ne_cout1)
        print('no after feature', ne_cout2)

