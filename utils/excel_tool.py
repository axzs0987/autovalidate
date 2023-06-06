import json

class GetExcelMsg:
    """
    获取excel详细信息
    """
    def __init__(self, xlsx_json_path, json_path):
        self.xlsx_json_path = xlsx_json_path
        self.json_path = json_path


    def load_file_names(self):
        with open(self.json_path + ".json", 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    def read_sheet_names(self, file_name):

        # 读取JSON文件
        with open(self.xlsx_json_path + file_name + '.json', 'r',  encoding='utf-8') as file:
            data = json.load(file)

        # 获取所有Sheet的名称
        names = [sheet['Name'] for sheet in data['Sheets']]
        return names
