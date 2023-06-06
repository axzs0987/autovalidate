import torch
from torch.autograd import Variable
from core_py.analyze_formula import generate_view_json, generate_one_before_feature, generate_one_after_feature, \
    generate_demo_features, get_feature_vector_with_bert_keyw, euclidean
import json
import os
import numpy as np
class ViewFeaturesGenerator:
    """
        用来生成特征的类，特征分成两种，粗粒度和细粒度
        还可以优化的空间: 不同的生成粒度存放到不同的路径下, 以便于接下来路径都按照指定的路径生成
    """

    def __init__(self, workbook_feature_path, view_json_path, view_nparray_path, bert_dict_file,
                 content_template_dict_file, bert_dict_path, save_path, sheet_json_path, sheet_nparray_path,
                 row, col, fine_save_feature_path, fine_save_feature_test_path):
        self.workbook_feature_path = workbook_feature_path
        self.view_json_path = view_json_path
        self.view_nparray_path = view_nparray_path
        self.bert_dict_file = bert_dict_file
        self.content_template_dict_file = content_template_dict_file
        self.bert_dict_path = bert_dict_path
        self.save_path = save_path
        self.sheet_json_path = sheet_json_path
        self.sheet_nparray_path = sheet_nparray_path
        self.row = row
        self.col = col
        self.save_feature_path = fine_save_feature_path
        self.save_feature_test_path = fine_save_feature_test_path


    def generate_view_features(self, workbook_name, sheet_name, is_test, is_sheet=False, row=None, col=None):
        """
        :param workbook_name: xlsx表名称
        :param sheet_name: xlsx sheet名称
        :param is_sheet: 是否需要提取粗粒度的特征
        :param row: 指定某行
        :param col: 指定某列
        :return: npy数组,生成的特征
        """
        if not is_sheet and row is not None and col is not None:
            # 提取细粒度的特征
            model_path = './finegrained_model'
            self.generate_view_json(workbook_name, sheet_name, row, col,
                                    workbook_feature_path=self.workbook_feature_path, save_path=self.view_json_path)
            self.generate_one_before_feature(workbook_name, sheet_name, row, col,
                                             source_root_path=self.view_json_path,
                                             saved_root_path=self.view_nparray_path,
                                             bert_dict_file=self.bert_dict_file,
                                             content_template_dict_file=self.content_template_dict_file,
                                             bert_dict_path=self.bert_dict_path)
            save_feature = ""
            if is_test:
                save_feature = self.save_feature_test_path
            else:
                save_feature = self.save_feature_path
            self.generate_one_after_feature(workbook_name, sheet_name, row, col,
                                            source_root_path=self.view_nparray_path,
                                            save_path=save_feature, model_path=model_path)
        else:
            ### row=51, col=6是指视窗的中心位置，对应到左上角就是1,1, 提取粗粒度的特征
            model_path = './coarsegrained_model'
            self.generate_view_json(workbook_name, sheet_name, 51, 6, workbook_feature_path=self.workbook_feature_path,
                                    save_path=self.sheet_json_path)
            self.generate_one_before_feature(workbook_name, sheet_name, 51, 6,
                                             source_root_path=self.sheet_json_path,
                                             saved_root_path=self.sheet_nparray_path,
                                             bert_dict_file=self.bert_dict_file,
                                             content_template_dict_file=self.content_template_dict_file,
                                             bert_dict_path=self.bert_dict_path)
            self.generate_one_after_feature(workbook_name, sheet_name, 51, 6,
                                            source_root_path=self.sheet_nparray_path,
                                            save_path=self.save_path, model_path=model_path)

    def generate_view_json(self, file_name, sheet_name, row, col, workbook_feature_path,
                           save_path, cross=False):
        """
        :param sheet_name: sheet名
        :param file_name: excel文件名
        :param row:行号
        :param col:列号
        :param workbook_feature_path:保存的路径
        :param save_path:保存的路径
        :param cross:
        :return: 返回view_json
        """
        with open(workbook_feature_path + file_name + '.json', 'r', encoding='utf-8') as f:
            workbook_json = json.load(f)
        generate_demo_features(file_name, sheet_name, workbook_json, row, col, save_path, is_look=True, cross=cross)

    def generate_one_before_feature(self, workbook_name, sheet_name, row, col, source_root_path, saved_root_path,
                                    bert_dict_file, content_template_dict_file, bert_dict_path, mask=2):
        """
        :param workbook_name: xlsx表名
        :param sheet_name: xlsx sheet名称
        :param row: 行号
        :param col: 列号
        :param source_root_path: 原来的特征路径
        :param saved_root_path:  需要保存的路径
        :param bert_dict_file: 保存字符串到bert feature的字典，来避免重复的计算
        :param content_template_dict_file: 保存内容的content template id
        :param bert_dict_path: 将字符串到bert的映射存入磁盘，因为直接全部内存太大了
        :param mask: 掩码特征
        :return:  将原始nparray使用模型做表示
        """
        formula_token = workbook_name + '---' + sheet_name + '---' + str(row) + '---' + str(col)

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

    def generate_one_after_feature(self, file_name, sheet_name, row, col, source_root_path, save_path, model_path):
        """
        :param sheet_name: sheet名
        :param file_name: excel文件名
        :param row:行号
        :param col:列号
        :param source_root_path: 原来的特征
        :param save_path: 最后要保存的特征
        :param model_path: 模型的路径
        :return: 返回最后的numpy数组特征
        """
        formula_token = f"{file_name}---{sheet_name}---{row}---{col}"
        model = torch.load(model_path)
        if os.path.exists(save_path + formula_token + '.npy'):
            print('特征已创建')
            return
        try:
            feature_nparray = np.load(source_root_path + formula_token + '.npy', allow_pickle=True)
            feature_nparray = feature_nparray.reshape(1, 100, 10, 399)
            model.eval()
            feature_nparray = torch.DoubleTensor(feature_nparray)
            feature_nparray = Variable(feature_nparray).to(torch.float32)
            feature_nparray = model(feature_nparray).detach().numpy()
            # formula_token = formula_token.encode("utf-8")
            np.save(save_path + formula_token + '.npy', feature_nparray)
            print('创建成功')
        except Exception as e:
            print('error: generate_one_after_feature:')
            print(e)
            return