import json
import random
import pprint
import numpy as np
import os

def same_sheet_boundary_global_tamplate_example():
    with open('../AnalyzeDV/data/types/boundary/boundary_template.json', 'r', encoding='utf-8') as f:
        all_temp = json.load(f)
    result = []
    id_list = [temp['id'] for temp in all_temp]
    # sample_keys = random.sample(id_list,10)
    all_dvid = []
    for template in all_temp:
        # if template['id'] not in sample_keys:
        #     continue
        dvid_list = {}
        # file_index = {}
        for index1, dvid in enumerate(template['dvid_list']):
            print('dvid', dvid)
            all_dvid.append(template['dvid_list'][index1])
            for index,fname in enumerate(template['file_sheet_name_list']):
                if template['file_sheet_name_list'][index1].split('-------------')[0]==fname.split('-------------')[0]:
                    continue
                
                if dvid not in dvid_list:
                    dvid_list[dvid] = 0

                dvid_list[dvid] += 1
                print('#################')
                print(template['file_sheet_name_list'][index1])
                print(fname.split('-------------')[0])
                # file_index[fname1].append(index)
        for dvid in dvid_list:
            if dvid_list[dvid] >= 1:
                result.append(dvid)
    random.shuffle(result)
    # for i in result:
    #     print(i)
    print(len(all_dvid))
    print(len(result))
    print(len(result)/len(all_dvid))

def same_sheet_tamplate_example():
    with open('../AnalyzeDV/data/types/custom/change_xml_all_templates.json', 'r', encoding='utf-8') as f:
        all_temp = json.load(f)
    result = []
    id_list = [temp['id'] for temp in all_temp]
    # sample_keys = random.sample(id_list,10)
    all_dvid = []
    for template in all_temp:
        # if template['id'] not in sample_keys:
        #     continue
        if template['id']==3:
            continue
        dvid_list = {}
        # file_index = {}
        for index1, dvid in enumerate(template['dvid_list']):
            print('dvid', dvid)
            all_dvid.append(template['dvid_list'][index1])
            for index,fname in enumerate(template['file_sheet_name_list']):
                if template['file_sheet_name_list'][index1].split('-------------')[0]==fname.split('-------------')[0]:
                    continue
                
                if dvid not in dvid_list:
                    dvid_list[dvid] = 0

                dvid_list[dvid] += 1
                print('#################')
                print(template['file_sheet_name_list'][index1])
                print(fname.split('-------------')[0])
                # file_index[fname1].append(index)
        for dvid in dvid_list:
            if dvid_list[dvid] >= 1:
                result.append(dvid)
    random.shuffle(result)
    # for i in result:
    #     print(i)
    print(len(all_dvid))
    print(len(result))
    print(len(result)/len(all_dvid))

def all_templates_dif_files():
    with open('../AnalyzeDV/data/types/custom/all_templates.json', 'r', encoding='utf-8') as f:
        all_temp = json.load(f)
    result = []
    for temp in all_temp:
        temp_file = []
        for filename in temp["file_sheet_name_list"]:
            temp_file.append(filename.split('-------------')[0])
        temp["file_sheet_name_list"] = list(set(temp_file))
        result.append(temp)
    with open('../AnalyzeDV/data/types/custom/distinct_file.json', 'w') as f:
        json.dump(result,f)

def all_boundary_templates_dif_files():
    with open('../AnalyzeDV/data/types/boundary/boundary_template.json', 'r', encoding='utf-8') as f:
        all_temp = json.load(f)
    result = []
    for temp in all_temp:
        temp_file = []
        for filename in temp["file_sheet_name_list"]:
            temp_file.append(filename.split('-------------')[0])
        temp["file_sheet_name_list"] = list(set(temp_file))
        result.append(temp)
    with open('../AnalyzeDV/data/types/boundary/distinct_file.json', 'w') as f:
        json.dump(result,f)

def count_execute():
    with open('../AnalyzeDV/data/types/custom/execute_suc_dvinfos.json', 'r', encoding='utf-8') as f:
        suc_dvinfos = json.load(f)
    with open('../AnalyzeDV/data/types/custom/execute_fail_dvinfos.json', 'r', encoding='utf-8') as f:
        fail_dvinfos = json.load(f)
    with open('../AnalyzeDV/data/types/custom/execute_error_dvinfos.json', 'r', encoding='utf-8') as f:
        error_dvinfos = json.load(f)

    print(len(suc_dvinfos))
    print(len(fail_dvinfos))
    print(len(error_dvinfos))


def get_boundary_template():
    result = []
    with open('../AnalyzeDV/data/types/boundary/boundary_list.json', 'r', encoding='utf-8') as f:
        boundary_list = json.load(f)
    for boundary_dvinfo in boundary_list:
        is_found = False
        new_temp = {'Type': boundary_dvinfo["Type"], 'Operator': boundary_dvinfo["Operator"], 'MinValue': boundary_dvinfo['MinValue'],'MaxValue': boundary_dvinfo['MaxValue']}
        max_id=0
        for template in result:
            max_id+=1
            if template['template']['Type'] == new_temp['Type'] and template['template']['Operator'] == new_temp['Operator'] and template['template']['MinValue'] == new_temp['MinValue'] and template['template']['MaxValue'] == new_temp['MaxValue']:
                is_found=True
                found_id = template['id']
                
                break
        if is_found==True:
            for index, template in enumerate(result):
                if template['id'] == found_id:
                    result[index]['number']+=1
                    result[index]['dvid_list'].append(boundary_dvinfo['ID'])
                    result[index]['file_sheet_name_list'].append(boundary_dvinfo['FileName']+"-------------"+boundary_dvinfo['SheetName'])
        else:
            template = {}
            template['id']=max_id
            template['number'] = 1
            template['dvid_list'] = [boundary_dvinfo['ID']]
            template['template'] = new_temp
            template['file_sheet_name_list'] = [boundary_dvinfo['FileName']+"-------------"+boundary_dvinfo['SheetName']]
            result.append(template)
    with open('../AnalyzeDV/data/types/boundary/boundary_template.json','w') as f:
        json.dump(result, f)        

    
def rand_sample():

    dvid = 0
    f_dvid=0
    with open('../AnalyzeDV/data/types/custom/execute_fail_dvinfos.json', 'r', encoding='utf-8') as f:
        execute_fail_dvinfos = json.load(f)
    with open('../AnalyzeDV/data/types/custom/distinct_file.json', 'r') as f:
        all_temp = json.load(f)

    fail_dvid = [str(i['ID'])+'---'+str(i['batch_id']) for i in execute_fail_dvinfos]
    for i in all_temp:
        for j in i['dvid_list']:
            dvid+=1
    for i in all_temp:
        if len(i['file_sheet_name_list'])>1:
            for j in i['dvid_list']:
                f_dvid+=1

    print(len(all_temp))
    filter_all_temp = [i for i in all_temp if len(i['file_sheet_name_list'])>1]
    result = []
    print(fail_dvid)
    for temp in filter_all_temp:
        print("$$$$$$$$$$$$$")
        # new_temp_dvidlist = [i.split('---')[0] for i in temp['dvid_list']]
        # new_temp_batchlist = [i.split('---')[1] for i in temp['dvid_list']]
        print(set(temp['dvid_list']))
        print(len(temp['dvid_list']))
        temp['dvid_list'] = list(set(temp['dvid_list']) - (set(temp['dvid_list'])&set(fail_dvid)))
        print(len(temp['dvid_list']))
        print(temp['formulas_list'][0])
        if len(temp['dvid_list'])>0:
            result.append(temp)
    with open('../AnalyzeDV/data/types/custom/filter_fail_distinct_file.json', 'w') as f:
        json.dump(result,f)
    print(len(result))
    random.shuffle(result)
    print([i['id'] for i in result[0:20]])

def rand_boundary_sample():

    dvid = 0
    f_dvid=0
    with open('../AnalyzeDV/data/types/boundary/distinct_file.json', 'r') as f:
        all_temp = json.load(f)

    for i in all_temp:
        for j in i['dvid_list']:
            dvid+=1
    for i in all_temp:
        if len(i['file_sheet_name_list'])>1:
            for j in i['dvid_list']:
                f_dvid+=1

    print(len(all_temp))
    filter_all_temp = [i for i in all_temp if len(i['file_sheet_name_list'])>1]
    result = []
    for temp in filter_all_temp:
        # print("$$$$$$$$$$$$$")
        # new_temp_dvidlist = [i.split('---')[0] for i in temp['dvid_list']]
        # new_temp_batchlist = [i.split('---')[1] for i in temp['dvid_list']]
        # print(set(temp['dvid_list']))
        # print(len(temp['dvid_list']))
        # print(len(temp['dvid_list']))
        # print(temp['formulas_list'][0])
        if len(temp['dvid_list'])>0:
            result.append(temp)
    with open('../AnalyzeDV/data/types/boundary/filter_fail_distinct_file.json', 'w') as f:
        json.dump(result,f)
    print(len(result))
    random.shuffle(result)
    print([i['id'] for i in result[0:20]])


def same_sheet_local_tamplate_example():
    with open('../AnalyzeDV/data/types/custom/all_templates.json', 'r', encoding='utf-8') as f:
        all_temp = json.load(f)
    result = []
    id_list = [temp['id'] for temp in all_temp]
    # sample_keys = random.sample(id_list,10)
    all_dvid = []
    for template in all_temp:
        # if template['id'] not in sample_keys:
        #     continue
        dvid_list = {}
        # file_index = {}
        for index1, dvid in enumerate(template['dvid_list']):
            print('dvid', dvid)
            all_dvid.append(template['dvid_list'][index1])
            for index,fname in enumerate(template['file_sheet_name_list']):
                if dvid not in dvid_list:
                    dvid_list[dvid] = 0
                if template['file_sheet_name_list'][index1]==fname:
                    dvid_list[dvid] += 1
                
                

                
                # print('#################')
                # print(template['file_sheet_name_list'][index1])
                # print(fname.split('-------------')[0])
                # file_index[fname1].append(index)
        for dvid in dvid_list:
            if dvid_list[dvid] >= 2:
                result.append(dvid)
    random.shuffle(result)
    # for i in result:
    #     print(i)
    print(len(all_dvid))
    print(len(result))
    print(len(result)/len(all_dvid))

def same_sheet_boundary_local_tamplate_example():
    with open('../AnalyzeDV/data/types/boundary/boundary_template.json', 'r', encoding='utf-8') as f:
        all_temp = json.load(f)
    result = []
    id_list = [temp['id'] for temp in all_temp]
    # sample_keys = random.sample(id_list,10)
    all_dvid = []
    for template in all_temp:
        # if template['id'] not in sample_keys:
        #     continue
        dvid_list = {}
        # file_index = {}
        for index1, dvid in enumerate(template['dvid_list']):
            print('dvid', dvid)
            all_dvid.append(template['dvid_list'][index1])
            for index,fname in enumerate(template['file_sheet_name_list']):
                if dvid not in dvid_list:
                    dvid_list[dvid] = 0
                if template['file_sheet_name_list'][index1]==fname:
                    dvid_list[dvid] += 1
                
                

                
                # print('#################')
                # print(template['file_sheet_name_list'][index1])
                # print(fname.split('-------------')[0])
                # file_index[fname1].append(index)
        for dvid in dvid_list:
            if dvid_list[dvid] >= 2:
                result.append(dvid)
    random.shuffle(result)
    # for i in result:
    #     print(i)
    print(len(all_dvid))
    print(len(result))
    print(len(result)/len(all_dvid))

def look_tamplate():
    with open('../AnalyzeDV/data/types/custom/distinct_file.json', 'r', encoding='utf-8') as f:
        templates = json.load(f)
    file_number_list = []
    cover_temp = 0
    for template in templates:
        file_number_list.append(len(template['file_sheet_name_list']))
        if(len(template['file_sheet_name_list'])>1):
            cover_temp += 1

    print(cover_temp)
    print(np.array(file_number_list).mean())
    print(len(templates))

def look_one_boundary(dvid):
    with open('../AnalyzeDV/data/types/boundary/boundary_list.json', 'r', encoding='utf-8') as f:
        dvinfos = json.load(f)
    for dvinfo in dvinfos:
        if dvinfo['ID'] == dvid:
            pprint.pprint(dvinfo)
            break

def look_same_sheet_formula():
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
        if "@@@@@@@@@"+dvinfo["FileName"]+'---'+dvinfo['SheetName']+'---'+dvinfo['Value'] not in res_dict:
            res_dict["@@@@@@@@@"+dvinfo["FileName"]+'---'+dvinfo['SheetName']+'---'+dvinfo['Value']] = []
        res_dict["@@@@@@@@@"+dvinfo["FileName"]+'---'+dvinfo['SheetName']+'---'+dvinfo['Value']].append(dvinfo)

    
    print(len(files))
    print(len(sheets))
    # for i in res_dict:
    #     if len(res_dict[i]) > 1:
    #         new_res_dict[i] = res_dict[i]

    # with open('../AnalyzeDV/data/types/custom/look_same_formula.json', 'w') as f:
    #     json.dump(new_res_dict, f)

def get_same_files():
    # with open('../AnalyzeDV/data/types/custom/all_template_with_simularity.json', 'r', encoding='utf-8') as f:
    #     templates = json.load(f)

    # dedup_templates = []
    # add_id = []
    # for template in templates:
    #     if template['id'] in add_id:
    #         continue
    #     add_id.append(template['id'])
    #     dedup_templates.append(template)

    with open('../AnalyzeDV/data/types/custom/all_template_with_simularity_dedup.json', 'r', encoding='utf-8') as f:
        dedup_templates = json.load(f)
    print(len(dedup_templates))
    same_style_dict = {}
    same_style_dvid_dict = {}
    max_id = 0
    dvid_max_id=0
    for template in dedup_templates:
        # if template['id']!=303:
        #     continue
        # print('found')
        for index in range(0, len(template['value_similarity_list'])):
            style_simlarity_5 = template['value_similarity_list'][index]['score']
            style_simlarity_1 = template['fill_color_similarity_list'][index]['score']
            style_simlarity_2 = template['font_color_similarity_list'][index]['score']
            style_simlarity_3 = template['height_similarity_list'][index]['score']
            style_simlarity_4 = template['width_similarity_list'][index]['score']
            style_simlarity_5 = template['type_similarity_list'][index]['score']

            # style_similarity = 0.45*style_simlarity_1+0.45*style_simlarity_2+0.05*style_simlarity_3+0.05*style_simlarity_4
            style_similarity = 0.2*style_simlarity_1+0.2*style_simlarity_2+0.2*style_simlarity_3+0.2*style_simlarity_4 + 0.2*style_simlarity_5
            index1 = template['fill_color_similarity_list'][index]['index1']
            index2 = template['fill_color_similarity_list'][index]['index2']
            dvid_index1_list = []
            dvid_index2_list = []
            file_sheet_name1 = template['file_sheet_name_list'][index1]
            file_sheet_name2 = template['file_sheet_name_list'][index2]
            print('filesheet1', file_sheet_name1)
            print('filesheet2', file_sheet_name2)
            for index_1,dvid_1 in enumerate(template['dvid_list']):
                if(template['file_sheet_name_list'][index_1]==file_sheet_name1):
                    dvid_index1_list.append(dvid_1)
                if(template['file_sheet_name_list'][index_1]==file_sheet_name2):
                    dvid_index2_list.append(dvid_1)
            
            found_id = max_id
            dvid_found_id = dvid_max_id
            found=False
            dvid_found = False

            
            # print(index1, index2)
            for id_ in same_style_dict:
                if file_sheet_name1 in same_style_dict[id_]:
                    found_id=id_
                    found=True
                    break
            for id_ in same_style_dvid_dict:
                if template['dvid_list'][index1] in same_style_dvid_dict[id_]:
                    # if(same_style_dvid_dict[id_]=='271733---0'):
                        # print('found_list:', same_style_dvid_dict[id_])
                        # print('foundt',emplate['dvid_list'][index1])
                    dvid_found_id=id_
                    dvid_found=True
                    break
            # print('found',found)
            # print(found_id)
            if not found:
                print('add new 1:',file_sheet_name1)
                # print(template['dvid_list'][index1])
                same_style_dict[found_id] = []
                same_style_dict[found_id].append(file_sheet_name1)
                max_id +=1
                
            if not dvid_found:
                print('add new 1:', dvid_index1_list)
                same_style_dvid_dict[dvid_found_id] = []
                for dvid_1 in dvid_index1_list:
                    same_style_dvid_dict[dvid_found_id].append(dvid_1)
                dvid_max_id+=1
            

            if style_similarity>0.8: # not sim
                if file_sheet_name1.split('-------------')[0] == file_sheet_name2.split('-------------')[0]:
                    continue
                # print('simu')
                same_style_dict[found_id].append(file_sheet_name2)
                for dvid_1 in dvid_index2_list:
                    same_style_dvid_dict[dvid_found_id].append(dvid_1)
                # same_style_dvid_dict[dvid_found_id].append(template['dvid_list'][index2])
            else:
                # print('not simu')
                if index == len(template['value_similarity_list'])-1:
                    # print('add index2')
                    found_id2 = max_id
                    found2=False
                    found_id3 = dvid_max_id
                    found3=False
                    
                    for id_ in same_style_dict:
                        if file_sheet_name2 in same_style_dict[id_]:
                            found_id2=id_
                            found2=True
                            break
                    
                    for id_ in same_style_dvid_dict:
                        if template['dvid_list'][index2] in same_style_dvid_dict[id_]:
                            found_id3=id_
                            found3=True
                            break
                    if found2 == False:
                        print('add new 2:',file_sheet_name2)
                        same_style_dict[found_id2] = []
                        same_style_dict[found_id2].append(file_sheet_name2)
                        max_id+=1
                    if found3==False:
                        same_style_dvid_dict[found_id3] = []
                        for dvid_1 in dvid_index2_list:
                            same_style_dvid_dict[dvid_found_id].append(dvid_1)
                        # same_style_dvid_dict[found_id3].append(template['dvid_list'][index2])
                        dvid_max_id +=1
        # break

    pprint.pprint(same_style_dict)
    # with open('../AnalyzeDV/data/types/custom/all_template_with_simularity_dedup.json', 'w', encoding='utf-8') as f:
    #     json.dump(dedup_templates, f)
    with open('../AnalyzeDV/data/types/custom/same_files_weighted_height_content_0.8.json', 'w', encoding='utf-8') as f:
        json.dump(same_style_dict, f)
    with open('../AnalyzeDV/data/types/custom/same_files_weighted_height_content_dvid_0.8.json', 'w', encoding='utf-8') as f:
        json.dump(same_style_dvid_dict, f)
    print(same_style_dvid_dict)

def count_has_global_dvs():
    # with open('../AnalyzeDV/data/types/custom/same_files_weighted_height_0.8.json', 'r', encoding='utf-8') as f:
    with open('../AnalyzeDV/data/types/custom/same_files_weighted_height_content_0.8.json', 'r', encoding='utf-8') as f:
    # with open('../AnalyzeDV/data/types/custom/same_files_weighted_height_width.json', 'r', encoding='utf-8') as f:
        same_files = json.load(f)
    with open('../AnalyzeDV/data/types/custom/all_templates.json', 'r', encoding='utf-8') as f:
        templates = json.load(f)
    with open('sheet_name_0.8.json', 'r') as f:
        sheet_name_95 = json.load(f)
    distinct_same_files_sheets = {}
    distinct_same_files = {}
    sheet_name = {}
    for id_ in same_files:
        distinct_same_files_sheets[id_] = list(set(same_files[id_]))
        distinct_same_files[id_] = set()
        sheet_name[id_] = set()
        for one_file in same_files[id_]:
            distinct_same_files[id_].add(one_file.split('-------------')[0])
            sheet_name[id_].add(one_file.split('-------------')[1])
        distinct_same_files[id_] = list(distinct_same_files[id_])
        sheet_name[id_] = list(sheet_name[id_])
        
    
    file_sheet_counts = {}
    for id_ in distinct_same_files_sheets:
        for one_file in distinct_same_files_sheets[id_]:
            if one_file not in file_sheet_counts:
                file_sheet_counts[one_file] = 0
            if(len(distinct_same_files_sheets[id_])>file_sheet_counts[one_file]):
                file_sheet_counts[one_file] = len(distinct_same_files_sheets[id_])

    file_counts = {}
    for id_ in distinct_same_files:
        for one_file in distinct_same_files[id_]:
            if one_file not in file_counts:
                file_counts[one_file] = 0
            if(len(distinct_same_files[id_])>file_counts[one_file]):
                file_counts[one_file] = len(distinct_same_files[id_])

    found=0
    all_=0
    dif_file = 0
    add_id = []
    res_sheet_name = {}
    for id_ in sheet_name:
        if len(sheet_name[id_]) > 1 and len(distinct_same_files[id_])>1:
            dif_file += 1
            print(id_, sheet_name[id_])
            res_sheet_name[id_] = sheet_name[id_]
    # print(add_id)
    print(dif_file)
    print(len(sheet_name))
    # print(res_sheet_name)
    # print()
    for id_ in sheet_name_95:
        print(id_, sheet_name_95[id_])
    with open('sheet_name_0.8.json', 'w') as f:
        json.dump(res_sheet_name, f)
    # print(set(res_sheet_name.keys())-set(sheet_name_95.keys()))
    for template in templates:
        if template['id']==3:
            continue
        for index,dvid in enumerate(template['dvid_list']):
            all_ +=1
            filename = template['file_sheet_name_list'][index]
            if filename.split('-------------')[0] in file_counts:
                if file_counts[filename.split('-------------')[0]] > 1:
                    found+=1
        
    print('found', found)
    print('all_', all_)

def random_base_with_same_file(global_dvid):

    with open('../AnalyzeDV/data/types/custom/same_files_weighted_height_content_dvid_0.8.json', 'r', encoding='utf-8') as f:
    # with open('../AnalyzeDV/data/types/custom/same_files_weighted_height_width.json', 'r', encoding='utf-8') as f:
        same_dvids = json.load(f)
    with open('../AnalyzeDV/data/types/custom/all_templates.json', 'r', encoding='utf-8') as f:
        templates = json.load(f)

    with open('C:/Users/v-sibeichen/Desktop/template_splited_by_style.json', 'r', encoding='utf-8') as f:
        template_splited_by_style = json.load(f)
    # print(same_dvids)
    all_dvid_list = {}
    # for id_ in same_dvids:
    #     for dvid in same_dvids[id_]:
    #         print(dvid)

    #         found_index = 0
    #         found_id = 0
    #         found=False
            
    #         for template in templates:
    #             for index,tem_dvid in enumerate(template['dvid_list']):
    #                 if dvid == tem_dvid:
    #                     found_index=index
    #                     found_id = template['id']
    #                     found=True
    #                     found_template = template
    #                     break
    #             if found:
    #                 break
    #         found_key = ""
    #         found=False
    #         for diction in template_splited_by_style:
    #             if list(diction.keys())[0].split('---')[0] == str(found_id):
    #                 for key in diction.keys():
    #                     for split_index in diction[key]:
    #                         if split_index == found_index:
    #                             found_key = key
    #                             found=True
    #                             break
    #                     if found:
    #                         break
    #                 if found:
    #                     for dvid_index in diction[found_key]:
    #                         if(found_template['dvid_list'][dvid_index] not in same_dvids[id_]):
    #                             same_dvids[id_].append(found_template['dvid_list'][dvid_index])
    # with open('../AnalyzeDV/data/types/custom/same_files_weighted_height_dvid_0.8_1.json', 'w', encoding='utf-8') as f:
    # # with open('../AnalyzeDV/data/types/custom/same_files_weighted_height_width.json', 'r', encoding='utf-8') as f:
    #     json.dump(same_dvids, f)
    files = {}

    for id_ in same_dvids:
        files[id_] = set()
        for dvid in same_dvids[id_]:
            if dvid not in all_dvid_list:
                all_dvid_list[dvid] = set()
            all_dvid_list[dvid].add(id_)
            for template in templates:
                for index,tem_dvid in enumerate(template['dvid_list']):
                    if dvid == tem_dvid:
                        found_index=index
                        found_id = template['id']
                        found=True
                        found_template = template
                        break
            files[id_].add(found_template['file_sheet_name_list'][found_index].split('-------------')[0])
        
    
    for dvid in all_dvid_list:
        all_dvid_list[dvid] = list(all_dvid_list[dvid])
    # pprint.pprint(all_dvid_list)
    found=0
    all_ = 0
    print('all_dvid_list', len(all_dvid_list))
    has_morethan_one_id_ = 0


    for dvid in all_dvid_list:
        
        if(len(all_dvid_list[dvid])>1):
            has_morethan_one_id_ +=1
        max_id = 0
        max_len= 0
        for one_id in all_dvid_list[dvid]:
            if len(same_dvids[one_id])>max_len:
                max_len = len(same_dvids[one_id])
                max_id=one_id
        id_ = max_id
        
        if(len(same_dvids[id_])==1):
            continue
        if dvid not in global_dvid:
            continue
        if len(files[id_])==1:
            continue
        found_dvid = random.choice(same_dvids[id_])
        
        all_+=1
            
        while(found_dvid == dvid):
            # print(same_dvids[id_])
            found_dvid = random.choice(same_dvids[id_])
        
        for template in templates:
            for index,template_dvid in enumerate(template['dvid_list']):
                if template_dvid == dvid:
                    gt_formula = template['formulas_list'][index]
                if template_dvid == found_dvid:
                    found_formula = template['formulas_list'][index]
        # print("#########")
        # print(gt_formula)
        # print(found_formula)
        if gt_formula == found_formula:
            found+=1
        
    print('found:', found)
    print('all_', all_)
    print('has_morethan_one_id_', has_morethan_one_id_)
    return all_dvid_list

def count_all_dvids():
    res_set = set()
    res_set1 = set()
    with open('../AnalyzeDV/data/types/custom/same_files_weighted_height_content_dvid_0.8.json', 'r', encoding='utf-8') as f:
        same_dvids = json.load(f)
    with open('../AnalyzeDV/data/types/custom/same_files_weighted_height_content_dvid_0.8.json', 'r', encoding='utf-8') as f:
        same_dvids1 = json.load(f)
    for id_ in same_dvids:
        for dvid in same_dvids[id_]:
            res_set.add(dvid)
    for id_ in same_dvids1:
        for dvid in same_dvids1[id_]:
            res_set1.add(dvid)
    print(len(res_set))
    print(len(res_set1))
    print(res_set-res_set1)

    with open('../AnalyzeDV/data/types/custom/all_template_with_simularity_dedup.json','r', encoding='utf-8') as f:
        templates = json.load(f)

    # res_set_1 = set()
    # for tempalte in templates:
    #     for sim in templates['value_similarity_list']:
    #         filename1 = templates['file_sheet_name_list'][sim[index1]]
    #         filename2 = templates['file_sheet_name_list'][sim[index1]]
    #         if filename1.split('-------------')[0]==filename2.split('-------------')[0]:
    #                 continue
    # res_1_set = set()
    
    result = []
    id_list = [temp['id'] for temp in templates]
    # sample_keys = random.sample(id_list,10)
    all_dvid = []
    dvids = []
    for template in templates:
        # if template['id'] not in sample_keys:
        #     continue
        if template['id']==3 or template['id']==23 or template['id']==176:
            continue
        dvid_list = {}
        # file_index = {}
        for index1, dvid in enumerate(template['dvid_list']):
            # print('dvid', dvid)
        
            all_dvid.append(template['dvid_list'][index1])
            for index,fname in enumerate(template['file_sheet_name_list']):
                if template['file_sheet_name_list'][index1].split('-------------')[0]==fname.split('-------------')[0]:
                    continue
                
                if dvid not in dvid_list:
                    dvid_list[dvid] = 0

                dvid_list[dvid] += 1
                # print('#################')
                # print(template['file_sheet_name_list'][index1])
                # print(fname.split('-------------')[0])
                # file_index[fname1].append(index)
        for dvid in dvid_list:
            if dvid_list[dvid] >= 1:
                result.append(dvid)
    print(len(set(result)))
    print(len(set(all_dvid)))
    # print(set(result)-res_set)
    # print(set(all_dvid_list.keys())-set(result))
    return result

def look_sheet():
    with open("../AnalyzeDV/sheetname_2_num.json", 'r', encoding='utf-8') as f:
        sheetname_2_num = json.load(f)
    
    sheetname_2_num=list(sheetname_2_num.items()) 
    sheetname_2_num.sort(key=lambda x:x[1],reverse=False)
    pprint.pprint(sheetname_2_num)

    # for key_pari in sheetname_2_num:

def batch_save_cnn_features():
    with open("../AnalyzeDV/data/types/custom/CNN_training_origin_dict.json", 'r', encoding='utf-8') as f:
        cnn_feature = json.load(f)
    print(list(cnn_feature.keys())[0].keys())
    print(cnn_feature[list(cnn_feature.keys())[0]][0])

def look_sheet_devide():
    with open("sheetname_2_file_devided_prob_filter.json", 'r') as f:
        sheetname_2_file_devided = json.load(f)
    for i in sheetname_2_file_devided['Edges']:
        print(len(i['filenames']))
    # print(len(sheetname_2_file_devided['Edges']))
def create_features():
    with open("../AnalyzeDV/CNN_training_origin_dict_filter.json", 'r', encoding='utf-8') as f:
        cnn_feature_0 = json.load(f)
    with open("../AnalyzeDV/CNN_training_origin_dict_filter_1.json", 'r', encoding='utf-8') as f:
        cnn_feature_1 = json.load(f)
    with open("../AnalyzeDV/CNN_training_origin_dict_filter_2.json", 'r', encoding='utf-8') as f:
        cnn_feature_2 = json.load(f)
    result = {}
    # print(list(cnn_feature_0.keys())[0])
    print(cnn_feature_0[list(cnn_feature_0.keys())[0]][0]['sheetfeature'][0].keys())
    # all_count = len(cnn_feature_0)+ len(cnn_feature_1)+ len(cnn_feature_2)
    # count = 1
    # for cnn_feature in [cnn_feature_0, cnn_feature_1, cnn_feature_2]:
    #     for sheetname in cnn_feature.keys():
    #         print(count, all_count)
    #         count+=1
    #         result[sheetname] = {}
    #         for item in cnn_feature[sheetname]:
    #             filename = item['filename']
    #             feature = []
    #             for one_cell_feature in item['sheetfeature']:
    #                 new_feature = []
    #                 for key in one_cell_feature:
    #                     new_feature.append(one_cell_feature[key])
    #                 feature.append(new_feature)
    #             result[sheetname][filename] = feature
    # print(list(cnn_feature.keys())[0])
    # print(cnn_feature[list(cnn_feature.keys())[0]][0])
    # with open("cnn_features_dict.json", 'w') as f:
    #     json.dump(result, f)
    # print(cnn_feature[list(cnn_feature.keys())[0]])



def get_similarity_feature(feature1, feature2):
    # 'background_color', 'font_color', 'font_size', 'font_strikethrough', 'font_shadow', 'font_ita', 'font_bold', 'height', 'width', 'content', 'content_template'
    result = []
    for index, item1 in enumerate(feature1):
        for index1, item in enumerate(feature1[index]):
            if index1==0: #bankgroud color
                if(item==feature2[index][index1]):
                    result.append(1)
                else:
                    result.append(0)
            if index1==1:#font color
                if(item==feature2[index][index1]):
                    result.append(1)
                else:
                    result.append(0)
            if index1==2:#font size
                if(item==feature2[index][index1]):
                    result.append(1)
                else:
                    result.append(0)
            if index1==3:#font_strikethrough
                if(item==feature2[index][index1]):
                    result.append(1)
                else:
                    result.append(0)
            if index1==4:#font_shadow
                if(item==feature2[index][index1]):
                    result.append(1)
                else:
                    result.append(0)
            if index1==5:#font_ita
                if(item==feature2[index][index1]):
                    result.append(1)
                else:
                    result.append(0)
            if index1==6:#font_bold
                if(item==feature2[index][index1]):
                    result.append(1)
                else:
                    result.append(0)
            if index1==6:#height
                result.append(1-(item-feature2[index][index1]))
            if index1==7:#width
                result.append(1-(item-feature2[index][index1]))
            # print(result)
            # break
        # break



def count_cnn_training():

    with open("../AnalyzeDV/training_sheet.json", 'r', encoding='utf-8') as f:
        training_sheet = json.load(f)
    print("sheet_num", len(training_sheet))
    with open("../AnalyzeDV/sheetname_2_file.json", 'r', encoding='utf-8') as f:
        sheetname_2_file = json.load(f)
    count=0
    for sheetname in training_sheet:
        for file_ in sheetname_2_file[sheetname]:
            count+=1
    print(count)

def look_feature():
    # with open("../AnalyzeDV/custom_all_training_origin_dict_filter_2.json",'r',encoding='utf-8') as f:
    with open("../AnalyzeDV/data/types/custom/change_xml_cutom_list.json" ,'r') as f:
        dvinfos = json.load(f) 
    for dvinfo in dvinfos:
        if(dvinfo["ID"]==10560 or dvinfo["ID"] == 157405):
            print("######################")
            print(dvinfo['ID'])
            print(dvinfo['Value'])
            print(dvinfo['RangeAddress'])
            print(dvinfo['SheetName'])
            print(dvinfo['FileName'])
            # print(dvinfo['ltx'])
            # print(dvinfo['lty'])
            # print(dvinfo['rbx'])
            # print(dvinfo['rby'])

def look_check_temp():
    with open("exact_sheet.json",'r',encoding='utf-8') as f:
        exact_sheet = json.load(f)   
    with open("closed_sheet.json",'r',encoding='utf-8') as f:
        closed_sheet = json.load(f)    
    with open("not_found_sheet.json",'r',encoding='utf-8') as f:
        not_found_sheet = json.load(f)
    print(len(closed_sheet))
    print(len(exact_sheet))
    print(len(set(exact_sheet)&set(not_found_sheet)))    
    print(len(set(closed_sheet)&set(not_found_sheet)))
    print(len(set(exact_sheet)|set(closed_sheet)|set(not_found_sheet)))


def look_devide():
    with open("../AnalyzeDV/custom_sheetname_2_file_devided.json" ,'r') as f:
        custom_sheetname_2_file_devided = json.load(f) 
    with open("../AnalyzeDV/data/types/custom/custom_sheet_2_num.json" ,'r') as f:
        custom_sheet_2_num = json.load(f) 
    with open("../AnalyzeDV/data/types/custom/custom_sheet_2_file.json" ,'r') as f:
        custom_sheet_2_file = json.load(f) 
    for sheetname in custom_sheet_2_file:
        custom_sheet_2_file[sheetname] = list(set(custom_sheet_2_file[sheetname]))
        custom_sheet_2_num[sheetname] = len(custom_sheet_2_file[sheetname])
    with open("../AnalyzeDV/data/types/custom/custom_sheet_2_file.json" ,'w') as f:
        json.dump(custom_sheet_2_file, f) 
    with open("../AnalyzeDV/data/types/custom/custom_sheet_2_num.json" ,'w') as f:
        json.dump(custom_sheet_2_num, f) 

def look_custom_features():
    with open("../AnalyzeDV/data/types/custom/custom_sheet_2_file.json" ,'r') as f:
        custom_sheet_2_file = json.load(f)
    found = 0
    not_found = 0
    for sheet in custom_sheet_2_file:
        for file_ in custom_sheet_2_file[sheet]:
            print("../../data/sheet_features/FeaturesDictionary/"+file_.split('/')[-1]+"---"+sheet+".json")
            if(os.path.exists("../../data/sheet_features/FeaturesDictionary/"+file_.split('/')[-1]+"---"+sheet+".json")):
                found+=1
            else:
                not_found+=1
    print(found)
    print(not_found)


def get_custom_sheet_file():
    with open("../AnalyzeDV/data/types/custom/change_xml_cutom_list.json" ,'r') as f:
        dvinfos = json.load(f) 
    result = {}
    sheet_num = {}
    for dvinfo in dvinfos:
        if dvinfo['SheetName'] not in result:
            result[dvinfo["SheetName"]] = set()
            sheet_num[dvinfo["SheetName"]] = 0
        result[dvinfo["SheetName"]].add(dvinfo["FileName"])
        # sheet_num[dvinfo["SheetName"]] += 1
    for sheetname in result:
        result[sheetname] = list(result[sheetname])
        sheet_num[sheetname] = len(result[sheetname])
    with open("../AnalyzeDV/data/types/custom/custom_sheet_2_file.json" ,'w') as f:
        json.dump(result, f) 
    with open("../AnalyzeDV/data/types/custom/custom_sheet_2_num.json" ,'w') as f:
        json.dump(sheet_num, f) 

def same_template_sheetname_coverage():
    with open("../AnalyzeDV/data/types/custom/change_xml_all_templates.json", 'r', encoding='utf-8') as f:
        all_templates = json.load(f)
    
    all_ = 0
    dvid_list = []
    found_same_sheet = 0
    for template in all_templates:
        for index, dvid in enumerate(template['dvid_list']):
            filename, sheetname = template['file_sheet_name_list'][index].split("-------------")
            for index1,file_sheet in enumerate(template['file_sheet_name_list']):
                one_filename, one_sheetname = file_sheet.split("-------------")
                if(index1==index):
                    continue
                if one_sheetname == sheetname and one_filename!= filename:
                    found_same_sheet += 1 
                    dvid_list.append((dvid, template['dvid_list'][index1]))
                    break
            all_ += 1
    print(found_same_sheet)
    print(all_)
    with open("can_find_same_sheet_dvid_list.json",'w') as f:
        json.dump(dvid_list, f)

def look_not_found():
    with open("can_find_same_sheet_dvid_list.json",'r',encoding='utf-8') as f:
        can_find_same_sheet_dvid_list = json.load(f)
    with open("not_found_dvid_list.json",'r',encoding='utf-8') as f:
        not_found_dvid_list = json.load(f)

    check_list = []
    for i in can_find_same_sheet_dvid_list:
        for j in not_found_dvid_list:
            if i[0] == j:
                print(i)
                check_list.append(i)
    with open("custom_positive_pair.json",'r') as f:
        positive_pair = json.load(f)



    print(positive_pair['FY19MN_TRAFFIC_&_SAFETY_OPS'])
    with open("../AnalyzeDV/custom_sheetname_2_file_devided.json", 'r') as f:
        sheetname_2_file_devided = json.load(f)


    print(sheetname_2_file_devided['FY19MN_TRAFFIC_&_SAFETY_OPS'])
    # pprint.pprint(set(not_found_dvid_list)&set(can_find_same_sheet_dvid_list))

def look_postive_origin_dict():
    with open("../AnalyzeDV/positive_training_origin_dict.json",'r', encoding='utf-8') as f:
        res = json.load(f)
    with open("100000_positive_need_feature.json", 'r') as f:
        res1 = json.load(f)
    with open("400000_negative_need_feature.json", 'r') as f:
        res2 = json.load(f)
    count = 0

    set_1 = set()
    for sheetname in res1:
        for filename in res1[sheetname]:
            set_1.add(filename+'---'+sheetname)

    set_2 = set()
    for sheetname in res2:
        for filename in res2[sheetname]:
            set_2.add(filename+'---'+sheetname)
    print(len(set_1))
    print(len(set_2))
    print(len(res.keys()))
    for i in res.keys():
        count += len(res[i])
    print(count)
    # print(res['']][0])
    # print(list(res.keys()))

def mv_files():
    files = os.listdir('../../data/sheet_features/')
    for file_ in files:
        # target_file = file_
        print(file_)
        if("'" == file_[0] and "'"==file_[-1]):
            target_file = file_[1:-1]
            cmd = 'mv ../../data/sheet_features/'+file_ + " ../../data/sheet_features/" + target_file
            os.system(cmd)

def look_pair_in_devided():
    with open("100000_positive_need_feature.json",'r', encoding='utf-8') as f:
        positive_pair = json.load(f)
    with open("../AnalyzeDV/sheetname_2_file_devided.json", 'r') as f:
        sheetname_2_file_devided = json.load(f)
    for sheet_name in positive_pair:
        print(sheet_name in sheetname_2_file_devided)

def look_extracted_features():
    extracted_res = []
    positive_res = []
    negative_res = []
    # with open("100000_positive_pair.json",'r', encoding='utf-8') as f:
    #     positive_pair = json.load(f)
    with open("all_positive_need_extract.json", 'r') as f:
        positive_need_feature = json.load(f)
    # with open("400000_negative_pair.json", 'r') as f:
    #     negative_pair = json.load(f)
    # # with open("400000_negative_need_feature.json", 'r') as f:
    #     negative_need_feature = json.load(f)

    positive_extracted = 0
    negative_extracted = 0

    positive_has_one = 0
    negative_has_one = 0

    positive_in_need = 0

    all_should_need = 0
    all_need = 0
    all_pair = 0

    filelist = os.listdir('../AnalyzeDV/FeaturesDictionary')
    filelist1 = os.listdir('../PredictDV/FeaturesDictionary')

    for i in filelist:
        extracted_res.append(i)
        positive_res.append(i)
    for i in filelist1:
        extracted_res.append(i)
        negative_res.append(i)
    print(extracted_res)
    print(len(extracted_res))

    
    with open("extracted_res.json", 'w') as f:
        json.dump(extracted_res, f)
    with open("positive_res.json", 'w') as f:
        json.dump(positive_res, f)
    with open("negative_res.json", 'w') as f:
        json.dump(negative_res, f)
    # for sheet_name in positive_pair:
        
    #     for pairs in positive_pair[sheet_name]:
    #         all_pair += 1
    #         print(pairs)
    #         if pairs[0] in positive_need_feature[sheet_name] and pairs[1] in positive_need_feature[sheet_name]:
    #             positive_in_need += 1
    #         else:
    #             all_should_need += 1

    # for sheet_name in positive_need_feature:
    #     for filename in positive_need_feature[sheet_name]:
    #         all_need += 1
    # print('positive_in_need', positive_in_need)
    # print('all_should_need', all_should_need)
    # print('all_need', all_need)
    # print('all_pair', all_pair)
    # for sheet_name in positive_pair:
    #     for pairs in positive_pair[sheet_name]:
    #         # print("../AnalyzeDV/FeaturesDictionary/"+pairs[0].split('/')[-1]+"---"+sheet_name)
    #         # break
    #         if((os.path.exists("../AnalyzeDV/FeaturesDictionary/"+pairs[0].split('/')[-1]+"---"+sheet_name+".json") or  os.path.exists("../PredictDV/FeaturesDictionary/"+pairs[0].split('/')[-1]+"---"+sheet_name+".json"))and (os.path.exists("../AnalyzeDV/FeaturesDictionary/"+pairs[1].split('/')[-1]+"---"+sheet_name+".json") or os.path.exists("../PredictDV/FeaturesDictionary/"+pairs[1].split('/')[-1]+"---"+sheet_name+".json"))):
    #             positive_extracted += 1
    #         elif ((os.path.exists("../AnalyzeDV/FeaturesDictionary/"+pairs[0].split('/')[-1]+"---"+sheet_name+".json") or  os.path.exists("../PredictDV/FeaturesDictionary/"+pairs[0].split('/')[-1]+"---"+sheet_name+".json")) or (os.path.exists("../AnalyzeDV/FeaturesDictionary/"+pairs[1].split('/')[-1]+"---"+sheet_name+".json") or os.path.exists("../PredictDV/FeaturesDictionary/"+pairs[1].split('/')[-1]+"---"+sheet_name+".json"))):
    #             positive_has_one += 1
    # for pairs in negative_pair:
    #     filename1 = pairs['file'][0]
    #     filename2 = pairs['file'][1]
    #     sheetname1 = pairs['sheet'][0]
    #     sheetname2 = pairs['sheet'][1]
    #     if((os.path.exists("../PredictDV/FeaturesDictionary/"+filename1.split('/')[-1]+"---"+sheetname1+".json") or os.path.exists("../AnalyzeDV/FeaturesDictionary/"+filename1.split('/')[-1]+"---"+sheetname1+".json")) and (os.path.exists("../PredictDV/FeaturesDictionary/"+filename2.split('/')[-1]+"---"+sheetname2+".json") or os.path.exists("../AnalyzeDV/FeaturesDictionary/"+filename2.split('/')[-1]+"---"+sheetname2+".json"))):
    #         negative_extracted += 1
    #     elif((os.path.exists("../PredictDV/FeaturesDictionary/"+filename1.split('/')[-1]+"---"+sheetname1+".json") or os.path.exists("../AnalyzeDV/FeaturesDictionary/"+filename1.split('/')[-1]+"---"+sheetname1+".json")) or (os.path.exists("../PredictDV/FeaturesDictionary/"+filename2.split('/')[-1]+"---"+sheetname2+".json") or os.path.exists("../AnalyzeDV/FeaturesDictionary/"+filename2.split('/')[-1]+"---"+sheetname2+".json"))):
    #         negative_has_one += 1

    # print(positive_extracted)
    # print(negative_extracted)

    # print(positive_has_one)
    # print(negative_has_one)

def look_extracted_postive():
    with open("all_positive_need_extract.json",'r') as f:
        all_positive_need_extract = json.load(f)
    with open("extracted_res.json",'r') as f:
        extracted_res = json.load(f)
    
    positive_meta_sheet = {}
    count = 0
    for filename in all_positive_need_extract:
        for sheetname in all_positive_need_extract[filename]:
            # print(filename, sheetname)
            # print(extracted_res[0])
            # print(filename.split('/')[-1]+"---"+sheetname+".json")
            # print(extracted_res[0])
            if filename.split('/')[-1]+"---"+sheetname+".json" in extracted_res:
                count += 1
                if filename not in positive_meta_sheet:
                    positive_meta_sheet[filename] = []
                positive_meta_sheet[filename].append(sheetname)
            # break
        # break
    with open("positive_meta_sheet.json",'w') as f:
        json.dump(positive_meta_sheet, f)
    print(count)

def count_number_sheet():
    with open("../AnalyzeDV/all_sheetname_2_num.json",'r') as f:
        all_sheetname_2_num = json.load(f)
    with open("../AnalyzeDV/sheetname_2_file_devided_1.json",'r') as f:
        sheetname_2_file_devided_1 = json.load(f)

    all_ = 0
    for sheetname in all_sheetname_2_num:
        all_ += all_sheetname_2_num[sheetname]

    res = {}

    res1 = []
    for sheetname in sheetname_2_file_devided_1:
        # print(sheetname)
        res[sheetname] = []
        for sametemp in sheetname_2_file_devided_1[sheetname]:
            # print(sametemp['sheetnames'])
            sametemp["prob"] = 1
            for sheetname1 in sametemp['sheetnames']:
                sametemp["prob"] *= all_sheetname_2_num[sheetname1]/all_
            if sametemp["prob"] < 0.00001:
                res[sheetname].append(sametemp)
            res1.append((sheetname, sametemp["prob"]))
    # pprint.pprint(res1)
    res1 = sorted(res1, key=lambda d: d[1], reverse=False)
    pprint.pprint(res1)
    with open("sheetname_2_file_devided_prob_filter.json", 'w') as f:
        json.dump(res, f)

def look_custom_list():
    with open("../AnalyzeDV/data/types/custom/change_xml_cutom_list.json", 'r') as f:
        custom_list = json.load(f)
    print(custom_list[0])
    print(len(custom_list))

def get_all_positive_filesheet():
    with open("sheetname_2_file_devided_prob_filter.json", 'r') as f:
        sheetname_2_file_devided_prob_filter = json.load(f)

    need_extract = {}

    for sheetname in sheetname_2_file_devided_prob_filter:
        for sametemp in sheetname_2_file_devided_prob_filter[sheetname]:
            # print(sametemp)
            if(len(sametemp['filenames'])>1):
                for filename in sametemp['filenames']:
                    if filename not in need_extract:
                        need_extract[filename] = set()
                    need_extract[filename].add(sheetname)
    for filename in need_extract:
        need_extract[filename] = list(need_extract[filename])

    with open("all_positive_need_extract.json", 'w') as f:
        json.dump(need_extract, f)
def look_need_extract():
    with open("all_positive_need_extract.json", 'r') as f:
        need_extract = json.load(f)
    count = 0
    for filename in need_extract:
        for sheetname in need_extract[filename]:
            # print(sheetname)
            count += 1
    print(count)

def look_positive_pair():
    with open("100000_positive_pair.json",'r') as f:
        positive_pair = json.load(f)
    with open("100000_positive_pair1.json",'r') as f:
        positive_pair1 = json.load(f)

    res = set()
    for sheetname in positive_pair:
        for filename in positive_pair[sheetname]:
            res.add(sheetname+"---"+filename[0]+","+sheetname+"---"+filename[1])
    for sheetname in positive_pair1:
        for filename in positive_pair1[sheetname]:
            res.add(sheetname+"---"+filename[0]+","+sheetname+"---"+filename[1])
    print(len(res))

def look_neg_feature():
    res = set()
    with open("100000_negative_pair.json",'r') as f:
        negative_feature = json.load(f)
    for pair in negative_feature:
        res.add(pair[0][0]+"---"+pair[0][1]+","+pair[1][0]+"---"+pair[1][1])
    print(len(res))

def look_clusterd_list():
    with open("../AnalyzeDV/clusterd_list_list_0.json", 'r') as f:
        clustered_list0 = json.load(f)
    with open("../AnalyzeDV/clusterd_list_list_1.json", 'r') as f:
        clustered_list1 = json.load(f)
    with open("../AnalyzeDV/clusterd_list_list_2.json", 'r') as f:
        clustered_list2 = json.load(f)
    with open("../AnalyzeDV/clusterd_list_list_3.json", 'r') as f:
        clustered_list3 = json.load(f)
    with open("../AnalyzeDV/clusterd_list_list_4.json", 'r') as f:
        clustered_list4 = json.load(f)
    with open("../AnalyzeDV/clusterd_list_list_5.json", 'r') as f:
        clustered_list5 = json.load(f)
    with open("../AnalyzeDV/clusterd_list_list_6.json", 'r') as f:
        clustered_list6 = json.load(f)
 
    sampled_list_list = []
    # print(clustered_list.keys())
    cluster_num = 0
    all_dv_num = 0 
    for clustered_list in [clustered_list0,clustered_list1,clustered_list2,clustered_list3, clustered_list4, clustered_list5, clustered_list6]:
        for item in clustered_list:
            if len(clustered_list[item]) > 1:
                cluster_num += 1
                all_dv_num += len(clustered_list[item])
                print(item, len(clustered_list[item]))
            
                for dvinfo in clustered_list[item]:
                    sampled_list_list.append(dvinfo)
    print(cluster_num)
    print(all_dv_num)
    with open("../AnalyzeDV/sampled_list_list.json", 'w') as f:
        json.dump(sampled_list_list, f)


def look_neg_feature():
    res = set()
    with open("100000_negative_pair.json",'r') as f:
        negative_feature = json.load(f)
    for pair in negative_feature:
        res.add(pair[0][0]+"---"+pair[0][1]+","+pair[1][0]+"---"+pair[1][1])
    print(len(res))

def look_boundary_list():
    with open("../AnalyzeDV/clusterd_boundary_list.json", 'r') as f:
        clusterd_boundary_list = json.load(f)
 
    sampled_list_list = []
    # print(clustered_list.keys())
    cluster_num = 0
    all_dv_num = 0 
    clusterd_boundary_list = list(clusterd_boundary_list.items())
    clusterd_boundary_list.sort(key=lambda x:len(x[1]),reverse=False)
    
    for item in clusterd_boundary_list:
        if len(item[1]) > 1:
            
            
            print(item[0], len(item[1]))
            # for dvinfo in item[1]:
            #     sampled_list_list.append(dvinfo)
        # if len(clusterd_boundary_list[item]) > 1:
        #     cluster_num += 1
        #     all_dv_num += len(clusterd_boundary_list[item])
        #     print(item, len(clusterd_boundary_list[item]))
            if len(item[1]) > 500 and len(item[1])<1500:
                all_dv_num += len(item[1])
                cluster_num += 1
                for dvinfo in item[1]:
                    sampled_list_list.append(dvinfo)
                
    print(cluster_num)
    print(all_dv_num)
    with open("../AnalyzeDV/sampled_boundary_list.json", 'w') as f:
        json.dump(sampled_list_list, f)

def look_dvinfos():
    with open("sampled_boundary_list.json" , 'r') as f:
        sampled_list_list = json.load(f)
    for item in sampled_list_list:
        if ':' not in item['RangeAddress']:
            print(item['RangeAddress'])
        # break

if __name__ == '__main__':


    # rand_sample()
    # get_boundary_template()
    # count_execute()
    # same_sheet_tamplate_example()
    # same_sheet_local_tamplate_example()
    # same_sheet_boundary_global_tamplate_example()
    # same_sheet_boundary_local_tamplate_example()
    # all_boundary_templates_dif_files()
    # rand_boundary_sample()
    # look_one_boundary(2830337)
    # look_tamplate()
    # look_same_sheet_formula()
    # get_same_files()
    # count_has_global_dvs()
    # global_dvid = count_all_dvids()
    # all_dvid_list = random_base_with_same_file(global_dvid)
    # look_sheet()
    # batch_save_cnn_features()
    # create_features()
    # get_similarity_feature()
    # get_positive_pair()
    # get_positive_feature()
    # get_negative_pair()
    # get_negative_feature()
    # count_cnn_training()
    # look_feature()
    # get_custom_sheet_file()
    # look_check_temp()
    # look_devide()
    # same_template_sheetname_coverage()
    # look_not_found()
    # look_postive_origin_dict()
    # look_extracted_features()
    # look_pair_in_devided()
    # mv_files()
    # count_number_sheet()
    # get_all_positive_filesheet()
    # look_need_extract()
    # look_extracted_postive()
    # look_custom_list()
    # look_custom_features()
    # look_positive_pair()
    # look_neg_feature()
    # look_sheet_devide()
    # look_clusterd_list()
    # look_boundary_list()
     look_dvinfos()