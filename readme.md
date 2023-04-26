# 环境配置

python = 3.6.13
dotnet = 5.0.302

```
pip install -r requirements.txt
```

# 生成粗/细粒度的特征

## 第一步：从xlsx文件抽取信息workbook level的信息，存为json
```
cd Demo
Dotnet run
```

具体的参数需要在“Demo/Program.cs”里改，其中函数包括：
（1）extractExcel(filename, entry, zip_file, save_path); 抽取单个xlsx文件的特征。filename:workbook的名字，zip_file:将workbook的上层路径进行压缩的压缩文件，entry:压缩文件中目标文件的entry，save_path:保存json的路径。
（2）extract_all_workbook(wokrbook_root_path, save_path, zip_file_path, entry_root_path); 批量抽取wokrbook_root_path下所有xlsx文件的特征。


## 第二步：生成特征

函数 generate_view_features(workbook_name, sheetname, workbook_feature_path, save_path, is_sheet=False, row=None, col=None);
workbook_name: workbook的名字
sheetname: workbook中的目标表名
workbook_feature_path: 第一步中的save_path
save_path: 保存结果的路径
is_sheet: True表示粗粒度；False表示细粒度
row: 目标视窗的行号（粗粒度不需要这个参数）
row: 目标视窗的列号（粗粒度不需要这个参数）

```
cd ..
python example.py
```

### 1. 从workbook level的信息中抽取目标视窗【100*10 cells】的信息，存为json
generate_view_json(workbook_name, sheetname, row, col, workbook_feature_path, save_path='tmp_data/view_json/')

### 2. 从目标视窗的json信息转位float的nparray 【主要是对内容使用sentencebert来表示】
generate_one_before_feature(workbook_name, sheetname, row, col, source_root_path='tmp_data/view_json/', saved_root_path='tmp_data/view_nparray/', bert_dict_file='/datadrive/data/bert_dict/bert_dict.json', content_template_dict_file="json_data/content_temp_dict_1.json", bert_dict_path ="/datadrive-2/data/bert_dict/")

bert_dict_file是保存字符串到bert feature的字典，来避免重复的计算。
bert_dict_path将字符串到bert的映射存入磁盘，因为直接全部内存太大了。
content_template_dict保存内容的content template id。content template：比如“0.55 -> NNNN”， “hello -> SSSSS“

### 3. 将原始nparray使用我们的模型做表示
generate_one_after_feature(workbook_name, sheetname, row, col, source_root_path='tmp_data/view_json/', saved_root_path='tmp_data/view_nparray/')