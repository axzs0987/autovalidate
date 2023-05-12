from core_py.analyze_formula import generate_view_json, generate_one_before_feature, generate_one_after_feature
from core_py.finegrain_model import FinegrainedModel
import json

def generate_view_features(workbook_name, sheetname, workbook_feature_path, save_path, is_sheet=False, row=None, col=None):
   
    
    with open(workbook_feature_path + workbook_name + '.json', 'r') as f:
        workbook_feature = json.load(f)
    if not is_sheet and row is not None and col is not None:
        model_path = 'finegrained_model'
        generate_view_json(workbook_name, sheetname, row, col, workbook_feature_path, save_path='tmp_data/view_json/')
        generate_one_before_feature(workbook_name, sheetname, row, col, source_root_path='tmp_data/view_json/', saved_root_path='tmp_data/view_nparray/', bert_dict_file='/datadrive/data/bert_dict/bert_dict.json', content_template_dict_file="json_data/content_temp_dict_1.json", bert_dict_path ="/datadrive-2/data/bert_dict/")
        generate_one_after_feature(workbook_name, sheetname, row, col, source_root_path='tmp_data/view_nparray/', save_path=save_path, model_path=model_path)
    else:
        ### row=51, col=6是指视窗的中心位置，对应到左上角就是1,1
        model_path = 'coarsegrained_model'
        generate_view_json(workbook_name, sheetname, 51, 6, workbook_feature_path, save_path='tmp_data/sheet_json/')
        generate_one_before_feature(workbook_name, sheetname, 51, 6, source_root_path='tmp_data/sheet_json/', saved_root_path='tmp_data/sheet_nparray/', bert_dict_file='/datadrive/data/bert_dict/bert_dict.json', content_template_dict_file="json_data/content_temp_dict_1.json", bert_dict_path ="/datadrive-2/data/bert_dict/")
        generate_one_after_feature(workbook_name, sheetname, 51, 6, source_root_path='tmp_data/sheet_nparray/', save_path=save_path, model_path=model_path)

    
if __name__ == '__main__':
    # generate view feature
    generate_view_features(workbook_name = "Excel - Data Validation Examples - Reduced.xlsx", sheetname = "HR Budget", row=7, col=2, workbook_feature_path="tmp_data/workbook_features/", save_path = "", )
    generate_view_features(workbook_name = "Excel - Data Validation Examples - Reduced.xlsx", sheetname = "HR Budget", workbook_feature_path="tmp_data/workbook_features/", save_path = "", is_sheet = True)