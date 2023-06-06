from generateFeature import ViewFeaturesGenerator
from generateJson import GenerateJsonMsg
from generateSimilarFormula import SimilarFormulaGenerator
from generateSimilarSheet import SimilarSheetGenerator

"""
在线模块, 主要是单表操作对应的各种function
"""
data_set_name = "ibm_data_set"
excel_name = "105896392748108092905439016459507990018-technote%e4%b8%80%e8%a6%a7_201905.xlsx"
sheet_name = "2017年"
row = 5
col = 4
def generate_feature_fine_by_single(data_set_name, file_name, sheet_name, row, col):
    """
    :param data_set_name: 数据集名称
    :param file_name: 文件名称
    :param sheet_name: 对应的sheet名称
    :param row: 行
    :param col: 列
    :return: 生成对应的表的细粒度的特征, 指定某张表的某个位置
    """
    data_set_msg = GenerateJsonMsg(data_set_name, file_name=file_name, sheet_name=sheet_name,
                                   row=row, col=col)

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
    generator.generate_view_features(workbook_name=file_name,
                                     sheet_name=sheet_name,
                                     row=row,
                                     col=col, is_test=True)

def find_similar_by_single_file(data_set_name, file_name, sheet_name):
    """
    :param data_set_name: 数据集名称
    :param file_name: 文件名
    :param sheet_name: sheet名称
    :return: 返回相似的表的内容
    """
    # 寻找相似的表 指定sheet和excel_name
    data_set_msg = GenerateJsonMsg(data_set_name, file_name=file_name, sheet_name=sheet_name,
                                   row=51, col=6)
    similar = SimilarSheetGenerator(file_lists_path=data_set_msg.file_lists_path, constrained_file_path=data_set_msg.constrained_file_path,
                                    root_path=data_set_msg.similar_root_path, save_name=data_set_msg.save_similar_name,
                                    file_name=file_name, sheet_name=sheet_name)
    similar.para_most_similar_sheet(data_set_name, "", False)


def find_similar_formular_by_single(data_set_name, excel_name, sheet_name, row, col):
    data_set_msg = GenerateJsonMsg(data_set_name, file_name=excel_name, sheet_name=sheet_name, row=row, col=col)
    res_formula = data_set_msg.get_reduce_formula(50)
    if res_formula == {}:
        print("this sheet can't find the similar sheet, so there is no similar formula result")
        return
    similarFormular = SimilarFormulaGenerator(root_path=data_set_msg.formula_search_root_path, load_path=data_set_msg.formula_load_path,
                                              save_path=data_set_msg.formula_save_path,
                                              workbooks_path=data_set_msg.formula_workbooks_path,
                                              constrain_workbooks_path=data_set_msg.formula_constrain_workbooks_path,
                                              data_set_path=data_set_msg.formula_data_set_path, data_set_name=data_set_name,
                                              ana_path=data_set_msg.save_file_names_path, row=row, col=col,
                                              sheet_name=sheet_name ,file_name=excel_name)

    similarFormular.find_closed_formula(res_formula)

# def gen_candidate_formula_sort():
#     data_set_msg = GenerateJsonMsg(data_set_name, file_name=excel_name, sheet_name=sheet_name, row=row, col=col)
#     similarFormular = SimilarFormulaGenerator(root_path=data_set_msg.formula_search_root_path, load_path=data_set_msg.formula_load_path,
#                                               save_path=data_set_msg.formula_save_path,
#                                               workbooks_path=data_set_msg.formula_workbooks_path,
#                                               constrain_workbooks_path=data_set_msg.formula_constrain_workbooks_path,
#                                               data_set_path=data_set_msg.formula_data_set_path, data_set_name=data_set_name,
#                                               ana_path=data_set_msg.save_file_names_path)
#     similarFormular.find_only_closed_formula(1, 1)

if __name__ == '__main__':
    # Test Case
    print('------------------------')
    # generate_feature_fine_by_single(data_set_name, excel_name, sheet_name, row, col)
    find_similar_formular_by_single(data_set_name, excel_name, sheet_name, row, col)
