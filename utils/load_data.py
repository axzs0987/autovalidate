import os
import shutil
import json


if __name__ == '__main__':
    """
    读取服务器上混合数据集下的json,并且分类到指定公司的方法
    """
    # 读取JSON文件
    with open('/datadrive/projects/analyze-dv-1/fortune500_company2workbook.json') as json_file:
        data = json.load(json_file)

    # 创建根文件夹
    root_folder = '/datadrive/data_fortune500'  # 设置根文件夹的名称
    os.makedirs(root_folder, exist_ok=True)
    # 遍历字典的键值对
    for company, files in data.items():
        if company != 'ibm':
            continue
        # 创建子文件夹
        company_folder = os.path.join(root_folder, company)
        os.makedirs(company_folder, exist_ok=True)

        # 移动或复制XLSX文件到子文件夹
        for file in files:
            source_path = os.path.join('/datadrive/data_fortune500/crawled_xlsx_fortune500', file)  # 假设XLSX文件与Python脚本在同一目录下
            destination_path = os.path.join(company_folder, file)
            shutil.copy(source_path, destination_path)  # 如果要复制而不是移动文件，可以使用shutil.copy()函数
        print("移动文件")
        print("文件移动完成")