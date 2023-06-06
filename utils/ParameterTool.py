import json

class ParameterTool:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.parameters = self.read_parameters_from_json()

    def read_parameters_from_json(self):
        with open(self.json_file_path, 'r') as f:
            parameters = json.load(f)
        return parameters

    def write_parameters_to_json(self):
        with open(self.json_file_path, 'w') as f:
            json.dump(self.parameters, f, indent=4)

    def get_parameter(self, parameter_name):
        return self.parameters.get(parameter_name)

    def set_parameter(self, parameter_name, value):
        self.parameters[parameter_name] = value

    def execute_function(self, function):
        workbook_name = self.get_parameter('workbook_name')
        sheet_name = self.get_parameter('sheet_name')
        row = self.get_parameter('row')
        col = self.get_parameter('col')
        workbook_feature_path = self.get_parameter('workbook_feature_path')
        save_path = self.get_parameter('save_path')
        function(workbook_name, sheet_name, row, col, workbook_feature_path, save_path)
        self.write_parameters_to_json()