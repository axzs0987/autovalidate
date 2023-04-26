import json
import os
import shutil
import zipfile
from xml.dom.minidom import parse

xml_dvinfo = {}
def get_key(filename, sheetname):
    return filename + '------'+sheetname 
def get_need_files():
    need_dvid=[]
    with open('../AnalyzeDV/data/types/custom/all_templates.json', 'r', encoding='utf-8') as f:
        templates = json.load(f)
    for template in templates:
        for node in template['node_list']:
            # print(node['Token'])
            if node['Term'] == 'CellToken':
                for dvid in template['dvid_list']:
                    need_dvid.append(dvid)
                break
    # print(need_dvid)
    with open('../AnalyzeDV/data/types/custom/dedup_shifted_custom_info.json', 'r', encoding='utf-8') as f:
        dvinfos = json.load(f)
    res_dict = {}
    new_res_dict = {}
    files = []
    sheets = []
    for dvinfo in dvinfos:
        if str(dvinfo['ID'])+'---'+str(dvinfo['batch_id']) not in need_dvid:
            continue
        if dvinfo["FileName"] not in files:
            files.append(dvinfo["FileName"])
        if dvinfo["FileName"]+'---'+dvinfo['SheetName']:
            sheets.append(dvinfo["FileName"]+'---'+dvinfo['SheetName'])
    return files, sheets

def un_zip(file_name):
    """unzip zip file"""
    print(file_name[:-4])
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(file_name[:-4]):
        return
    else:
        os.mkdir(file_name[:-4])
    for names in zip_file.namelist():
        zip_file.extract(names,file_name[:-4])
    zip_file.close()

def change_format(files):
    new_files = []
    for filename in files:
        # print(filename[25:-5]+'.zip')
        source_path = "xml_data"
        new_file = "xml_data/" + filename[25:-5]+'.zip'
        cmd = "copy "+filename + " " + source_path
        # print(cmd)
        # # os.system(cmd)
        # try:
        #     shutil.copy(filename, source_path)
        # except:
        #     continue
        # break
        # new_files.append("xml_data/" + filename[25:])
        # print("xml_data/" + filename[25:])
        # try:
        #     os.rename("xml_data/" + filename[25:], "xml_data/" + filename[25:-5]+'.zip')
        # except:
        #     continue
        try:
            un_zip("xml_data/" + filename[25:-5]+'.zip')
        except:
            continue
        # break

def get_sheet_index(folder_name, sheetname):
    # folder_name = filename.split('/')[-1][:-5]
    workbook_path = 'xml_data/'+ folder_name+'/xl/workbook.xml'
    domTree = parse(workbook_path)
    rootNode = domTree.documentElement
    # print(rootNode.nodeName)
    sheets = rootNode.getElementsByTagName("sheet")
    for index,sheet in enumerate(sheets):
        # print(sheet.getAttribute('name'))
        # print('sheetname', sheetname)
        if sheetname == sheet.getAttribute('name'):
            return index+1

def get_data_validations(filename, sheetname):
    def get_dvinfo(datavalidation):
        # Value, InputTitle, InputMessage, ErrorTitle, ErrorMessage, ErrorStyle, RangeAddress, Height, Width, FileName, SheetName, content, header, refers, 
        RangeAddress = datavalidation.getAttribute('sqref')
        value = datavalidation.getElementsByTagName("formula1")[0].childNodes[0].data
        # print(RangeAddress)
        # print(value)
        return RangeAddress, value
    folder_name = filename.split('/')[-1][:-5]
    sheet_index = get_sheet_index(folder_name, sheetname)
    sheet_path = 'xml_data/'+ folder_name+'/xl/worksheets/sheet'+str(sheet_index)+'.xml'
    domTree = parse(sheet_path)
    rootNode = domTree.documentElement

    data_validations = rootNode.getElementsByTagName("dataValidation")
    for datavalidation in data_validations:
        if datavalidation.getAttribute("type") == 'custom':
            # print("#############")
            range_, value_ = get_dvinfo(datavalidation)
            
            # print(datavalidation.getAttribute("type"))
            key = get_key(filename, sheetname)
            if key not in xml_dvinfo:
                xml_dvinfo[key] = []
            xml_dvinfo[key].append({'range': range_, 'value': value_})


def batch_extract(file_sheets):

    for index,file_sheet in enumerate(file_sheets):
        try:
            print(index, len(file_sheets))
            filename, sheetname = file_sheet.split('---')
            get_data_validations(filename, sheetname)
            if index % 100 == 0:
                with open("xml_dvinfo.json",'w') as f:
                    json.dump(xml_dvinfo, f)
        except:
            continue


def dedup_parse_xml():
    with open("xml_dvinfo.json",'r') as f:
        xml_dvinfo = json.load(f)
    res = {}
    for key in xml_dvinfo:
        new_set = []
        new_list = []
        for item in xml_dvinfo[key]:
            range_list = item['range'].split(' ')
            new_range = ''
            for range_ in range_list:
                if ':' not in range_:
                    range_ = range_+':'+range_
                new_range += range_
                new_range += ' '
            item['range'] = new_range[:-1]
            # if ':' not in item['range'] and ' ' not in item['range']:
            #     item['range'] = item['range']+':'+item['range']
            if item['range'] not in new_set:
                new_set.append(item['range'])
                new_list.append(item)
        res[key] = new_list
    with open("dedup_xml_dvinfo.json",'w') as f:
        json.dump(res, f)


if __name__ == '__main__':
    files, sheets = get_need_files()
    # change_format(files)


    # sheet_index = get_sheet_index('0a2b18848e13accc4528abe3a076987c_d2lraS5uY2kubmloLmdvdgkxMjkuNDMuMjU0LjIyNA==', "Test Cases")
    # get_data_validations('0a2b18848e13accc4528abe3a076987c_d2lraS5uY2kubmloLmdvdgkxMjkuNDMuMjU0LjIyNA==',sheet_index)
    # batch_extract(sheets)
    dedup_parse_xml()