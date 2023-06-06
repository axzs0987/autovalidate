import torch
from torch.autograd import Variable
from core_py.analyze_formula import generate_view_json, generate_one_before_feature, generate_one_after_feature, \
    generate_demo_features, get_feature_vector_with_bert_keyw, euclidean
import json
import os
import numpy as np
import faiss
class SimilarSheetGenerator:
    """
    给定一个目标表，和一些候选表格，对候选表格通过与目标的相似程度进行排序。
    我们需要先通过前面的方法获得所有的目标表格和候选表格的特征
    """

    def __init__(self, file_lists_path, constrained_file_path, root_path, save_name, file_name, sheet_name):
        self.file_lists_path = file_lists_path
        self.constrained_file_path = constrained_file_path
        self.root_path = root_path
        self.save_path = self.root_path + save_name
        self.file_name = file_name
        self.sheet_name = sheet_name

    def para_most_similar_sheet(self, data_set_name, file_sheet_name, batch, is_save=False):
        constrained_path = self.constrained_file_path
        with open(constrained_path, 'r') as f:
            constrained_workbooks = json.load(f)[data_set_name]
        if batch == False:
            need_find_file_sheet_name = self.file_name + "---" + self.sheet_name + "---" + "51---6.npy"
        else:
            need_find_file_sheet_name = file_sheet_name
        self.find_middle10domain_closed_sheet(is_save, 1, 1, constrained_workbooks, save_path=self.save_path, need_find_file_sheet_name=need_find_file_sheet_name)

    def find_middle10domain_closed_sheet(self, need_save, thread_id, batch_num, constrain_workbooks,
                                         save_path, need_find_file_sheet_name):
        """
        :param need_find_file_sheet_name:
        :param need_save: 是否需要保存
        :param thread_id: 线程id
        :param batch_num: batch_number
        :param constrain_workbooks: 数据集的路径
        :param save_path: 需要保存的路径
        :return: 返回相似的表
        """
        filesheet = need_find_file_sheet_name
        if os.path.exists(save_path + '/' + filesheet + '.json'):
            print(f'{filesheet}相似表的结果已存在')
            return
        all_sheet_features = {}
        feature_files = os.listdir(self.root_path + 'sheet_after_features')
        feature_files.sort()
        for filename in feature_files:
            all_sheet_features[filename] = \
            np.load(self.root_path + 'sheet_after_features/' + filename, allow_pickle=True)[0]
        res = {}
        count = 0
        batch_len = len(all_sheet_features) / batch_num
        num = set()

        # for index, filesheet in enumerate(all_sheet_features):
        num.add(filesheet)
        if filesheet in res:
            return
        count += 1
        all_features = []
        id2filesheet = {}
        id_ = 0
        ids = []
        for one_filesheet in all_sheet_features:
            if one_filesheet != filesheet:
                id2filesheet[id_] = one_filesheet
                ids.append(id_)
                id_ += 1
                all_features.append(all_sheet_features[one_filesheet])
            else:
                target_feature = all_sheet_features[one_filesheet]

        all_features = np.array(all_features)
        ids = np.array(ids)

        res = {}
        for i_ind, other_feature in enumerate(all_features):
            distance = euclidean(target_feature, other_feature)
            res[id2filesheet[ids[i_ind]]] = distance
        res = sorted(res.items(), key=lambda x: x[1])
        # 传入的必须是二维矩阵，即比较的是整张表
        all_features = all_features.astype(np.float32)
        faiss_index = faiss.IndexFlatL2(len(all_features[0]))
        faiss_index2 = faiss.IndexIDMap(faiss_index)
        faiss_index2.add_with_ids(all_features, np.array(range(0, len(all_features))).astype(np.int64))

        search_list = np.array([target_feature])

        D, I = faiss_index.search(np.array(search_list).astype(np.float32), 4)  # sanity check

        top_k = []
        for i_index, i in enumerate(I[0]):
            other_feature = all_features[i_index]
            if float(D[0][i_index]) < 50:
                print('------------------------')
                print(need_find_file_sheet_name + ' 寻找到了比较相似的特征')
            top_k.append((id2filesheet[ids[i]], float(D[0][i_index])))
        print(f'------------------------end search of {filesheet}')
        with open(save_path + '/' + filesheet + '.json', 'w') as f:
            json.dump(top_k, f)
