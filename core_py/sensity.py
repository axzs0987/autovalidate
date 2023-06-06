import zipfile
import shutil
import os
import xlrd2 as xl
import json
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
# zfile = zipfile.ZipFile('test-excel-label.zip', 'r')
# print(zfile.infolist())
# for item in zfile.infolist():
#     print(item.filename)

# shutil.copy('azure-spend-has-sensitivity-label.xlsx', 'azure-spend-has-sensitivity-label.zip')
# zfile = zipfile.ZipFile('azure-spend-has-sensitivity-label.zip', 'r')
# print(zfile.infolist())
# for item in zfile.infolist():
#     print(item.filename)



def check_fortune500():
    res = {}
    all_ = 0
    has_label = 0
    # filelist = os.listdir("/datadrive/data_fortune500/crawled_xlsx_fortune500/")
    filelist = os.listdir("/datadrive/projects/Demo/fix_fortune500/")
    filelist = [item.replace('.json','') for item in filelist]
    set_num = {}
    set_id = {}
    id_ = 1
    need_break = False
    for index,filename in enumerate(filelist):
        # print(index, len(filelist))
        shutil.copy("/datadrive/data_fortune500/crawled_xlsx_fortune500/" + filename, "/datadrive/data_fortune500/fortune500_zip/"+filename.replace('.xlsx','.zip'))
        if filename != "255727299294126114432788607487043242758-table-filter.xlsx" and filename != '270994555332380340493190056172932166700-table-filter.xlsx':
            continue
        print(filename)
        try:
            zfile = zipfile.ZipFile("/datadrive/data_fortune500/fortune500_zip/"+filename.replace('.xlsx','.zip'), 'r')
        except:
            print("bad: /datadrive/data_fortune500/fortune500_zip/"+filename.replace('.xlsx','.zip'))
            continue
        for item in zfile.infolist():
            if(item.filename == "docMetadata/LabelInfo.xml"):
                
                data = str(zfile.read(item.filename))
                print(data)
                if ' enabled="1"' in data and ' removed="0"':
                    # try:
                    wb = xl.open_workbook("/datadrive/data_fortune500/crawled_xlsx_fortune500/" + filename)
                    # except:
                    #     continue
                    pointSheets = wb.sheet_names()
                    res[filename] = pointSheets
                    found = False
                    found_id = -1
                    for one_id in set_id:
                        one_sheets = set_id[one_id]
                        if set(one_sheets) == set(pointSheets):
                            found=True
                            found_id = one_id
                            set_num[found_id] += 1
                    if found == False:
                        set_id[id_] = pointSheets
                        if id_ not in set_num:
                            set_num[id_] = 0
                        set_num[id_] += 1
                        id_ += 1
                    
                    has_label += 1
                    break
                # else:
                #     # need_break = True
                #     print(filename, data)
        all_ += 1
        # if need_break:
        #     break
    # print('all', all_)
    # print('has_label', has_label)
    # with open("sensity_workbooks.json", 'w') as f:
    #     json.dump(res, f)
    # with open("set_num.json", 'w') as f:
    #     json.dump(set_num, f)
    # with open("set_id.json", 'w') as f:
    #     json.dump(set_id, f)

check_fortune500()