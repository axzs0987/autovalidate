import json
import os
import re
from generateFeature import ViewFeaturesGenerator
from generateJson import GenerateJsonMsg
from generateSimilarSheet import SimilarSheetGenerator
from utils.excel_tool import GetExcelMsg
from formula_analyze import FormulaCleasing


"""
离线模块,主要是处理整个数据集的方法
"""
data_set_name = "ibm_data_set"
excel_name = "105896392748108092905439016459507990018-technote%e4%b8%80%e8%a6%a7_201905.xlsx"
sheet_name = "2017年"
row = 5
col = 4
def generate_feature(data_set_name):
    """
    生成粗粒度的特征
    :param data_set_name: 数据集名称
    :return: 每个file-sheet对应的npy数组文件
    """
    data_set_msg = GenerateJsonMsg(data_set_name)
    generator = ViewFeaturesGenerator(workbook_feature_path=data_set_msg.workbook_feature_path,
                                      view_json_path=data_set_msg.view_json_path,
                                      view_nparray_path=data_set_msg.view_nparray_path,
                                      bert_dict_file=data_set_msg.bert_dict_file_path,
                                      content_template_dict_file=data_set_msg.content_template_dict_file,
                                      bert_dict_path=data_set_msg.bert_dict_path, save_path=data_set_msg.save_feature_path,
                                      sheet_json_path=data_set_msg.sheet_json_path,
                                      sheet_nparray_path=data_set_msg.sheet_nparray_path,
                                      row=51, col=6, fine_save_feature_path=data_set_msg.fine_save_feature_path,
                                      fine_save_feature_test_path=data_set_msg.fine_save_feature_path_test
                                      )

    excel_tool = GetExcelMsg(data_set_msg.source_json_path, data_set_msg.save_file_names_json_path)
    file_names = excel_tool.load_file_names()
    print(file_names)
    for file in file_names:
        print('------------------------')
        print("文件名为 " + file)
        sheets = excel_tool.read_sheet_names(file)
        for sheet in sheets:
            # 生成粗粒度的特征
            print("开始创建文件名为 " + file + "sheet为 " + sheet + "的特征")
            generator.generate_view_features(workbook_name=file,
                                             sheet_name=sheet, row=generator.row, col=generator.col,
                                             is_sheet=True, is_test=False)


def find_similar_by_all_file(data_set_name):
    """
     寻找相似的表,file_sheet
    :param data_set_name:
    :return: 生成每个file_sheet对应的npy文件,key表示file_sheet,value表示相似度(越低越相似)
    """
    data_set_msg = GenerateJsonMsg(data_set_name, file_name="", sheet_name="",
                                   row=51, col=6)
    for root, dirs, files in os.walk(data_set_msg.save_feature_path):
        for file in files:
            similar = SimilarSheetGenerator(file_lists_path=data_set_msg.file_lists_path,
                                            constrained_file_path=data_set_msg.constrained_file_path,
                                            root_path=data_set_msg.similar_root_path,
                                            save_name=data_set_msg.save_similar_name,
                                            file_name="", sheet_name="")
            similar.para_most_similar_sheet(data_set_name, file, True)


def formula_cleasing(data_set_name):
    """
    对C#生成的有关公式的文件进行数据清洗
    :param data_set_name: 数据集名称
    :return: 返回公式文件formulas_with_id.json
    """
    data_set_msg = GenerateJsonMsg(data_set_name)
    formula_clean = FormulaCleasing(origin_data_formulas_path=data_set_msg.origin_data_formulas_path,
                                    save_group_path=data_set_msg.save_group_path, save_merge_path=data_set_msg.save_merge_path,
                                    save_merge_new_path=data_set_msg.save_merge_new_path, save_res_path=data_set_msg.save_res_path)
    formula_clean.find_formula()


def generate_feature_fine_all(data_set_name):
    """
    生成细粒度的所有特征
    :param data_set_name: 数据集名称
    :return: 返回细粒度的所有特征
    """
    data_set_msg = GenerateJsonMsg(data_set_name)
    all_formula_json = data_set_msg.save_formulas_path + 'Formulas_with_id.json'
    generator = ViewFeaturesGenerator(workbook_feature_path=data_set_msg.workbook_feature_path,
                                      view_json_path=data_set_msg.view_json_path,
                                      view_nparray_path=data_set_msg.view_nparray_path,
                                      bert_dict_file=data_set_msg.bert_dict_file_path,
                                      content_template_dict_file=data_set_msg.content_template_dict_file,
                                      bert_dict_path=data_set_msg.bert_dict_path, save_path=data_set_msg.save_feature_path,
                                      sheet_json_path=data_set_msg.sheet_json_path,
                                      sheet_nparray_path=data_set_msg.sheet_nparray_path,
                                      row=row, col=col, fine_save_feature_path=data_set_msg.fine_save_feature_path,
                                      fine_save_feature_test_path=data_set_msg.fine_save_feature_path_test)
    # Test

    with open(all_formula_json, 'r', encoding='utf-8') as f:
        formulas = json.load(f)
    for formula in formulas:
        generator.row = formula['fr']
        generator.col = formula['fc']
        file_sheet = formula['filesheet']
        pattern = r"\.xlsx.*$"  # 匹配以 .xlsx 开头的内容直到行尾的部分
        file_name = re.sub(pattern, ".xlsx", file_sheet)
        pattern = r"\.xlsx---(.*)"  # 匹配以 .xlsx 开头的部分后的任意字符
        match = re.search(pattern, file_sheet)
        sheet_name = match.group(1)  # 提取匹配的内容
        print('生成表名为: ' + excel_name + ' sheet名为 ' + str(sheet_name) + ' row = ' + str(formula['fr']) + 'col = ' + str(formula['fc']) + ' 的特征')
        generator.generate_view_features(workbook_name=file_name,
                                         sheet_name=sheet_name,
                                         row=formula['fr'],
                                         col=formula['fc'], is_test=False)

if __name__ == '__main__':
    print("------------------------")
    generate_feature(data_set_name)
    find_similar_by_all_file(data_set_name)
    formula_cleasing(data_set_name)
    generate_feature_fine_all(data_set_name)