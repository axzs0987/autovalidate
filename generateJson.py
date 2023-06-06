import glob
import json
import os
import re
import shutil


class GenerateJsonMsg:
    """
    生成数据集对应的路径及其json文件
    """

    def __init__(self, data_set_name, file_name="", sheet_name="", row=-1, col=-1):
        """
        :param data_set_name: 数据集名称
        """
        self.data_set_name = data_set_name
        self.data_set_path = "./data_set/" + data_set_name + "/"
        self.save_file_names_path = "./analyze-dv-1/" + data_set_name + "/"
        self.xlsx_path = self.data_set_path + "source_xlsx/"

        # 用户需要找到特征的表名和sheet_name
        self.file_name = file_name
        self.sheet_name = sheet_name
        self.row = row
        self.col = col
        self.file_sheet_name = self.file_name + '---' + self.sheet_name + '---' + str(row) + '---' + str(col)
        self.file_sheet_name_51_6 = self.file_name + '---' + self.sheet_name + '---' + str(51) + '---' + str(6)

        # 测试用的path
        # self.xlsx_path = self.data_set_path + "source_xlsx/_1/"

        self.source_json_path = self.data_set_path + "source_json/"
        # self.source_json_path = self.data_set_path + "source_json/_1/"

        self.save_formulas_path = self.data_set_path + "save_formulas_jsons/"
        self.zip_file_path = self.data_set_path + "zip_files/"
        # self.save_file_names_json_path = self.save_file_names_path + self.data_set_name + "_file_names"
        # 测试用
        self.save_file_names_json_path = self.save_file_names_path + "file_names"

        # data_driver_path
        self.data_driver_path = './data_drive/data_two/' + data_set_name + '/'


        # data_feature_path
        self.data_feature_path = "./data_feature/" + data_set_name + "/"
        self.workbook_feature_path = self.data_set_path + "source_json/"
        self.view_json_path = self.data_feature_path + "view_json/"
        self.view_nparray_path = self.data_feature_path + "view_nparray/"
        self.save_feature_path = self.data_driver_path + "sheet_after_features/"
        self.sheet_json_path = self.data_feature_path + "sheet_json/"
        self.sheet_nparray_path = self.data_feature_path + "sheet_nparray/"

        # 定义dict的path
        self.bert_dict_file_path = "data_drive/data_one/bert_dict/bert_dict.json"
        self.content_template_dict_file = "json_data/content_temp_dict_1.json"
        self.bert_dict_path = "data_drive/data_two/bert_dict"

        # 定义寻找相似表时用到的路径
        self.file_lists_path = "./analyze-dv-1"
        self.constrained_file_path = "./analyze-dv-1/" + data_set_name + "/" + data_set_name + "_workbook.json"
        self.similar_root_path = "./data_drive/data_two/" + data_set_name + "/"
        self.save_similar_path = self.data_driver_path + "similar_res/"
        self.save_similar_name= "similar_res/"
        # 保存和这张表相似的表的公式


        # 寻找相似公式的路径
        self.formula_cleaning_path = "./data_set/" + data_set_name + "/" + "formula_json/"
        self.formula_cleaning_json_name = data_set_name + "formulatokenr1c1.json"

        # 清洗公式的路径
        self.origin_data_formulas_path = "./data_set/" + data_set_name + "/save_formulas_jsons/origin_data_formulas.json"
        self.save_group_path = "./data_set/" + data_set_name + "/save_formulas_jsons/origin_groupby_r1c1.json"
        self.save_merge_path = "./data_set/" + data_set_name + "/save_formulas_jsons/origin_mergerange.json"
        self.save_merge_new_path = "./data_set/" + data_set_name + "/save_formulas_jsons/origin_mergernew_res_1.json"
        self.save_res_path = "./data_set/" + data_set_name + "/save_formulas_jsons/Formulas_with_id.json"

        # 寻找公式的路径
        self.formula_search_root_path = self.data_driver_path
        self.formula_load_path = self.data_driver_path + 'model1_formulas_dis/'
        self.formula_save_path = self.data_driver_path + data_set_name + '_model1_res/'
        # 存放的是指定单元格名称的xlsx的json名称
        self.formula_workbooks_path = './analyze-dv-1/' + data_set_name + '/reduced_formulas.json'
        # 存放的是excel的名称
        self.formula_constrain_workbooks_path = './analyze-dv-1/' + data_set_name + '/' + data_set_name + '_workbook.json'
        # 存放数据集的路径
        self.formula_data_set_path = './data_set/' + data_set_name + '/'

        # 保存细粒度特征的路径
        self.fine_save_feature_path = self.data_driver_path + 'after_feature/'
        self.fine_save_feature_path_test = self.data_driver_path + 'after_feature_test/'

    def create_folder(self, folder_path):
        """
        :param folder_path: 文件夹路径
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"数据集{self.data_set_name}文件夹 '{folder_path}' 创建成功！")
        else:
            print(f"数据集{self.data_set_name}文件夹 '{folder_path}' 已经存在。")

    def gen_data_dir(self):
        """
        :return: 生成数据集对应的文件夹
        """
        self.create_folder(self.xlsx_path)
        self.create_folder(self.source_json_path)
        self.create_folder(self.save_formulas_path)
        self.create_folder(self.zip_file_path)
        self.create_folder(self.save_file_names_path)

        self.create_folder(self.data_feature_path)
        self.create_folder(self.view_json_path)
        self.create_folder(self.view_nparray_path)
        self.create_folder(self.save_feature_path)
        self.create_folder(self.sheet_json_path)
        self.create_folder(self.sheet_nparray_path)

        self.create_folder(self.save_similar_path)

    def gen_files_json(self):
        """
        :return: 获取xlsx文件夹路径下的内容,生成对应的json,将所有文件生成json
        """
        file_names = []

        for root, dirs, files in os.walk(self.xlsx_path):
            for file in files:
                file_names.append(file)
        file_dict = {self.data_set_name: file_names}
        if not os.path.exists(self.save_file_names_json_path):
            with open(self.save_file_names_json_path, 'w') as f:
                json.dump(file_names, f)
            print(f'{self.save_file_names_json_path}创建成功')
        if not os.path.exists(self.constrained_file_path):
            with open(self.constrained_file_path, 'w') as f:
                json.dump(file_dict, f)
            print(f'{self.constrained_file_path}创建成功')
        return file_names

    def get_reduce_formula(self, threshold):
        """
        生成需要生成公式的表
        从Formulas_with_id.json读取,
        filesheet来自data_drive/data_two/data_set_name/similar_res/
        105419042552415731769252671704639115290-part_2_p8_5.2_and_5.2.1_patch_compatibility_matrix.xlsx-
        --Fix Pack
        Information---51---6.npy.json
        :return:
        """
        # TODO
        read_similar_json = self.save_similar_path + self.file_sheet_name_51_6 + '.npy.json'
        all_formula_json = self.save_formulas_path + 'Formulas_with_id.json'
        with open(read_similar_json, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        with open(all_formula_json, 'r', encoding='utf-8') as f:
            formulas = json.load(f)

        res = []
        pattern = r"\.---51---6.npy"
        if datas[0][1] > threshold:
            print('empty set of similar_json')
            return False
        for data in datas:
            data[0] = data[0].replace("---51---6.npy", "")
            res.append(data[0])
        # print(res)
        formula_exists = False
        res_file_sheet = []
        # ['105419042552415731769252671704639115290-part_2_p8_5.2_and_5.2.1_patch_compatibility_matrix.xlsx---Useful Links']
        for formula in formulas:
            # print(formula)
            # print(formula["filesheet"])
            if formula["filesheet"] in res:
                res_file_sheet.append(formula)
                formula_exists = True
        if formula_exists == False:
            print('相似表上不存在公式')
        return res_file_sheet
                # single_file_sheet[]
        # print(replaced_strings)



    def gen_split_files_and_json(self, max_files_per_json, max_files_per_folder, folder_path):
        """
        :param max_files_per_json: 每个json最多包含的文件个数,为了避免爆内存
        :return: 生成若干个文件
        """
        file_names = []
        json_files = []
        folders = []

        for root, dirs, files in os.walk(self.xlsx_path):
            for file in files:
                file_names.append(file)

        # 排序后再进行比较
        file_names = sorted(file_names)

        # 计算需要创建的文件夹数量
        num_folders = len(file_names) // max_files_per_folder
        if len(file_names) % max_files_per_folder != 0:
            num_folders += 1

        # 创建编号文件夹
        for i in range(num_folders):
            folder_name = f'{self.xlsx_path}_{i + 1}/'
            self.create_folder(folder_name)
            folders.append(folder_name)


        # 分割文件名称为每个json文件最多包含的文件数
        chunks = [file_names[i:i + max_files_per_json] for i in range(0, len(file_names), max_files_per_json)]
        i = 0
        # 创建每个json文件并存储文件名称
        for chunk in chunks:
            json_file = chunk
            json_files.append(json_file)
            for single_file in chunk:
                folder_index = i // max_files_per_folder
                destination_folder = folders[folder_index]
                shutil.copy(os.path.join(folder_path, single_file), os.path.join(destination_folder, single_file))
                print(f"文件 '{single_file}' 复制到文件夹 '{destination_folder}' 成功！")
                i = i + 1
        # 保存为json文件
        for i, json_file in enumerate(json_files):
            output_file = f'{self.save_file_names_json_path}_{i + 1}.json'
            with open(output_file, 'w') as f:
                json.dump(json_file, f)
                print(f"JSON文件 '{output_file}' 创建成功！")


    def gen_r1c1_json(self, json_name):
        with open(self.save_formulas_path + json_name, 'r', encoding='utf-8') as file:
            formula_datas = json.load(file)
        res_dict = {}
        for excel_name in formula_datas.keys():
            excel_formulas = formula_datas[excel_name]
            if len(excel_formulas) > 0:
                for formula in excel_formulas:
                    row = formula['row']
                    col = formula['column']
                    r1c1 = formula['formulaR1C1']
                    file_name = excel_name + '---' + str(row) + '---' + str(col)
                    res_dict[file_name] = r1c1
        with open(self.formula_cleaning_path + self.formula_cleaning_json_name, 'w') as f:
            json.dump(res_dict, f)

    def gen_similar_table(self):
        sheet_names = []
        with open(self.save_similar_path + self.file_sheet_name + '.npy.json', 'r') as f:
            sheet_names = json.load(f)
        res_sheets = []
        for sheet_name in sheet_names:
            print(sheet_name[0])
            res_sheets.append(sheet_name[0])
        pattern = r"\.xlsx.*"
        replaced_strings = [re.sub(pattern, ".xlsx", string) for string in res_sheets]
        print(replaced_strings)


if __name__ == '__main__':
    data_set_name = "ibm_data_set"
    # x---Daeja Fix Information---51---6.npy
    excel_name = "105896392748108092905439016459507990018-technote%e4%b8%80%e8%a6%a7_201905.xlsx"
    sheet_name = "2017年"
    row = 5
    col = 4
    testGenJson = GenerateJsonMsg(data_set_name, excel_name, sheet_name, row, col)

    # 生成需要的文件夹
    # testGenJson.gen_data_dir()

    # 文件分块
    # testGenJson.gen_split_files_and_json(20, 20, testGenJson.xlsx_path)

    # 读取数据
    # testGenJson.gen_similar_table()
    # 清洗json数据
    # testGenJson.gen_r1c1_json("origin_data_formulas_1.json")
    # testGenJson.gen_files_json()
    # 生成reduce_fomula的json
    testGenJson.get_reduce_formula(50)
    # testGenJson.gen_data_dir()
    # testGenJson.gen_files_json()
    # testGenJson.gen_split_files_to_json()
    # file_sheet_name
    # testGenJson.gen_distribute_files_to_folders(testGenJson.xlsx_path, 20)
