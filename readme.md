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

# 寻找相似的公式

## 第一步：寻找相似表
这一步的目标是：给定一个目标表，和一些候选表格，对候选表格通过与目标的相似程度进行排序。
我们需要先通过前面的方法获得所有的目标表格和候选表格的特征。然后使用下面的函数：

core_py/formulae2e.py -- find_middle10domain_closed_sheet()，

203行开始遍历，其中filesheet是目标表格的名字，格式是filename---sheetname，all_sheet_features是候选表格的特征，结果为top_k，其格式是排序好的tuple，每个tuple(候选表名字，相似度分数)

## 第二步：在相似表上找到所有的公式
这一步的目标是：给定一个表，使用c#抽取出上面所有的公式

core_csharp/Analyzer.cs: analyze_formula_fortune500()：给定一个workbook集合（../analyze-dv-1/fortune500_filenames.json），抽取所有的公式。
core_py/analyze_formula.py 对公式进一步的处理：
依次运行：anaylze_training_range()，devide_training_range_recheck，check_training_formula， resave_training_mergerange。最后得到“Formulas_fortune500_with_id.json”, 里面存了所有wokrbook里的所有公式，每一个formula对象包括id, filesheet, r1c1, fr, fc, lr, lc字段。filesheet是公式所在的表格，r1c1是公式的内容，fr,fc,lr,lc分别对应公式区域的左上角行列和右下角行列。


## 第三步：寻找相似的公式
这一步的目标是：给一个目标公式视窗，对相似表上的候选公式视窗进行相似度排序。

core_py/formulae2e.py -- find_only_closed_formula()：该函数生成所有候选公式的相似度排序
1336行开始遍历，其中formula是一个公式的对象，formula['filesheet']是表名：filename---sheetname，formula['fr']是公式开始的行，formula['fc']是公式开始的列，formula_token：filename---sheetname---fr---fc
filepath是所有公式的特征存储的位置，特征生成的方法如前所述。


core_py/formulae2e.py -- sort_only_most_similar_formula()：该函数生成结果（选择最像的公式）

## 第四步：寻找参考单元格 --- 解析公式中的参考单元格

core_csharp/test_formula.cs --- get_all_refcell()：该函数根据公式的语法树。
core_py/ref_shift.py --- generate_ref_position(): 该函数抽取公式的参考单元格。结果为一个列表，列表中的每一个单元格对象，包含'R','C'属性，表示该单元格的位置，有些还会包含‘R_dollar’，‘C_dollar’属性，表示该单元格应该视为绝对位置的参考对象，而不是相对位置的参考对象。

## 第五步：寻找参考单元格 --- 寻找相似的单元格
这一步的目标是：在相似表S1上给定一个参考单元格c1，在目标表格St上找到与之最像的单元格ct，候选范围为St上与c1相同位置的单元格c1‘周围的单元格。

core_py/cnn_fine_tune.py --- copyround_finetune()
2990行开始遍历，formula_token为目标公式，model1_filename是sort_only_most_similar_formula()中存储的结果，即在相似表上找到的最相似的公式，作为参考公式found_formula_token。origin_wbjson/found_wbjson是“生成粗/细粒度的特征---第一步”的结果。test_refcell_position是“第四步：寻找参考单元格 --- 解析公式中的参考单元格”的结果。
该函数首先找到参考公式，然后对参考公式中的每个参考单元格，抽取特征，并抽取其在目标表St上的候选单元格的特征，最后进行排序。
