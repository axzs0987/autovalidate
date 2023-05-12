import os
import json
from cnn_model import CNNnet, CNNnet_Cosine
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pprint
import gc
import faiss
import unicodedata

content_dict = {}
content_tem_dict={}
if os.path.exists('content_dict.json'):
    with open("content_dict.json",'r') as f:
        content_dict = json.load(f)
if os.path.exists('content_tem_dict.json'):
    with open("content_tem_dict.json",'r') as f:
        content_tem_dict = json.load(f)

def create_custom_features():
    with open("../AnalyzeDV/custom_all_training_origin_dict_filter_2.json", 'r') as f:
        cnn_feature = json.load(f)
    result = {}
    sheet_2_id = {}
    id_=1
    id_2_feature = {}
    # print(list(cnn_feature_0.keys())[0])
    # print(cnn_feature_0[list(cnn_feature_0.keys())[0]][0]['sheetfeature'][0].keys())
    all_count = len(cnn_feature)
    print(all_count)
    count = 1

    # for key in custom_all_training_origin_dict_filter_2.keys():
    #     print(count, all_count)
    #     count+=1
    #     result[sheetname] = {}
    #     filename, sheetname = key.split('----')
    # for dvinfo in 
    for key in cnn_feature:
        # filename, sheetname = key.split("----")
        feature = []
        # feature = get_feature_vector(item[''])
        for one_cell_feature in cnn_feature[key]:
            new_feature = []
            for key1 in one_cell_feature:
                new_feature.append(one_cell_feature[key1])
            feature.append(new_feature)
        
        result[key] = feature
        id_2_feature[id_] = get_feature_vector(feature)
        if key not in sheet_2_id:
            sheet_2_id[key] = {}
        sheet_2_id[key] = id_
        id_+=1

    # print(list(cnn_feature.keys())[0])
    # print(cnn_feature[list(cnn_feature.keys())[0]][0])
    with open("custom_cnn_features_origin_dict.json", 'w') as f:
        json.dump(result, f)
    with open("custom_sheet_2_id.json", 'w') as f:
        json.dump(sheet_2_id, f)
    with open("custom_id_2_feature.json", 'w') as f:
        json.dump(id_2_feature, f)
    with open("custom_content_dict.json", 'w') as f:
        json.dump(content_dict, f)
    with open("custom_content_tem_dict.json", 'w') as f:
        json.dump(content_tem_dict, f)
    print(cnn_feature[list(cnn_feature.keys())[0]]) 

def create_features(is_bert=0):
    # with open("../AnalyzeDV/CNN_training_origin_dict_filter.json", 'r', encoding='utf-8') as f:
    #     cnn_feature_0 = json.load(f)
    # with open("../AnalyzeDV/CNN_training_origin_dict_filter_1.json", 'r', encoding='utf-8') as f:
    #     cnn_feature_1 = json.load(f)
    with open("../AnalyzeDV/CNN_training_origin_dict_filter_2.json", 'r', encoding='utf-8') as f:
        cnn_feature_2 = json.load(f)
    with open("../AnalyzeDV/CNN_training_origin_dict_filter_3.json", 'r', encoding='utf-8') as f:
        cnn_feature_3 = json.load(f)
    # with open("../AnalyzeDV/positive_training_origin_dict.json", 'r', encoding='utf-8') as f:
    #     cnn_feature_2 = json.load(f)
    result = {}
    sheet_2_id = {}
    id_=1
    id_2_feature = {}
    # print(list(cnn_feature_0.keys())[0])
    # print(cnn_feature_0[list(cnn_feature_0.keys())[0]][0]['sheetfeature'][0].keys())
    # all_count = len(cnn_feature_0)+ len(cnn_feature_1)+ len(cnn_feature_2)
    all_count = len(cnn_feature_2)
    count = 1
    for cnn_feature in [cnn_feature_2, cnn_feature_3]:
        for sheetname in cnn_feature.keys():
            print(count, all_count)
            count+=1
            result[sheetname] = {}
            for item in cnn_feature[sheetname]:
                # print(item)
                filename = item['filename']
                feature = []
                # feature = get_feature_vector(item[''])
                for one_cell_feature in item['sheetfeature']:
                    new_feature = []
                    for key in one_cell_feature:
                        new_feature.append(one_cell_feature[key])
                    feature.append(new_feature)
                result[sheetname][filename] = feature
                id_2_feature[id_] = get_feature_vector(feature)
                if sheetname not in sheet_2_id:
                    sheet_2_id[sheetname] = {}
                sheet_2_id[sheetname][filename] = id_
                id_+=1

    # print(list(cnn_feature.keys())[0])
    # print(cnn_feature[list(cnn_feature.keys())[0]][0])
    with open("new_cnn_features_origin_dict.json", 'w') as f:
        json.dump(result, f)
    with open("new_sheet_2_id.json", 'w') as f:
        json.dump(sheet_2_id, f)
    with open("new_id_2_feature.json", 'w') as f:
        json.dump(id_2_feature, f)
    # with open("json_data/content_dict.json", 'w') as f:
    #     json.dump(content_dict, f)
    # with open("json_data/content_tem_dict.json", 'w') as f:
    #     json.dump(content_tem_dict, f)
    # with open("content_dict.json", 'w') as f:
    #     json.dump(content_dict, f)
    # with open("content_tem_dict.json", 'w') as f:
    #     json.dump(content_tem_dict, f)
    # print(cnn_feature[list(cnn_feature.keys())[0]])

def get_feature_vector(feature):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass
    
        try:
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
    
        return False
    result = []
    for index, item1 in enumerate(feature):
        print("######")
        print(feature[index])
        one_cel_feature = []
        for index1, item in enumerate(feature[index]):
            if index == 0:
                one_cel_feature.append()
            if index1==1:#font color
                one_cel_feature.append(item)
                    
            if index1==2:#font size
                one_cel_feature.append(item)

            if index1==3:#font_strikethrough
                if(item==True):
                    one_cel_feature.append(1)
                else:
                    one_cel_feature.append(0)
            if index1==4:#font_shadow
                if(item==True):
                    one_cel_feature.append(1)
                else:
                    one_cel_feature.append(0)
            if index1==5:#font_ita
                if(item==True):
                    one_cel_feature.append(1)
                else:
                    one_cel_feature.append(0)
            if index1==6:#font_bold
                if(item==True):
                    one_cel_feature.append(1)
                else:
                    one_cel_feature.append(0)
            if index1==7:#height
                one_cel_feature.append(item)
            if index1==8:#width
                one_cel_feature.append(item)
            if index1==9:#content
                if not str(item) in content_dict:
                    content_dict[str(item)] = len(content_dict)+1
                one_cel_feature.append(content_dict[str(item)])
                if str(item) == '':
                    one_type = 3
                elif is_number(str(item)):
                    one_type = 2
                else:
                    one_type = 1
            if index1==10:#content_template
                if not str(item) in content_tem_dict:
                    content_tem_dict[str(item)] = len(content_tem_dict)+1
                one_cel_feature.append(content_tem_dict[str(item)])
            one_cel_feature.append(one_type)
        result.append(one_cel_feature)
        print(one_cel_feature)
    return result


def middle_negative_pair():
    with open("positive_meta_sheet.json",'r') as f:
        positive_meta_sheet = json.load(f)
    with open("sheetname_2_file_devided_prob_filter.json", 'r') as f:
        sheetname_2_file_devided_prob_filter = json.load(f)
    # with open("100000_positive_pair.json",'r') as f:
    #     positive_pair = json.load(f)

    negative_pair = []
    added = []
    count = 0
    while count < 100000:
        filename = random.choice(list(positive_meta_sheet.keys()))
        sheetname = random.choice(list(positive_meta_sheet[filename]))
        filename1 = random.choice(list(positive_meta_sheet.keys()))
        sheetname1 = random.choice(list(positive_meta_sheet[filename1]))
        if filename+'---'+sheetname+","+filename1+'---'+sheetname1 in added:
            continue
        if sheetname1 == sheetname:
            continue
        negative_pair.append(((filename, sheetname), (filename1, sheetname1)))
        count += 1
        print(count, 100000)
        added.append(filename+'---'+sheetname+","+filename1+'---'+sheetname1)
     
        # if sheetname in positive_pair:
        #     if filename in positive_pair:
        #         continue 
        # if sheetname in positive_pair1:
        #     if filename in positive_pair1:
        #         continue 
        # if sheetname in sheetname_2_file_devided_prob_filter:
        #     for sametemp in sheetname_2_file_devided_prob_filter[sheetname]:
        #         # print(filename)
        #         # print(sametemp['filenames'][0])
            
        #         if filename in sametemp['filenames']:
        #             filename1 = random.choice(sametemp['filenames'])
        #             while(filename1 ==filename):
        #                 filename1 = random.choice(sametemp['filenames'])
        #             if sheetname not in positive_pair1:
        #                 positive_pair1[sheetname] = []
        #             positive_pair1[sheetname].append((filename, filename1))
        #             count += 1
        #             print(count, 100000)
        #             break


    with open("100000_negative_pair.json",'w') as f:
        json.dump(negative_pair, f)


def middle_postive_pair():
    with open("positive_meta_sheet.json",'r') as f:
        positive_meta_sheet = json.load(f)
    with open("sheetname_2_file_devided_prob_filter.json", 'r') as f:
        sheetname_2_file_devided_prob_filter = json.load(f)
    with open("100000_positive_pair.json",'r') as f:
        positive_pair = json.load(f)

    positive_pair1 = {}
    count = 0
    while count < 100000:
        filename = random.choice(list(positive_meta_sheet.keys()))
        sheetname = random.choice(list(positive_meta_sheet[filename]))
        if sheetname in positive_pair:
            if filename in positive_pair:
                continue 
        if sheetname in positive_pair1:
            if filename in positive_pair1:
                continue 
        if sheetname in sheetname_2_file_devided_prob_filter:
            for sametemp in sheetname_2_file_devided_prob_filter[sheetname]:
                # print(filename)
                # print(sametemp['filenames'][0])
            
                if filename in sametemp['filenames']:
                    filename1 = random.choice(sametemp['filenames'])
                    while(filename1 ==filename):
                        filename1 = random.choice(sametemp['filenames'])
                    if sheetname not in positive_pair1:
                        positive_pair1[sheetname] = []
                    positive_pair1[sheetname].append((filename, filename1))
                    count += 1
                    print(count, 100000)
                    break


    with open("100000_positive_pair1.json",'w') as f:
        json.dump(positive_pair1, f)


def get_positive_pair():
    with open("../AnalyzeDV/list_sheetname_2_file_devided_1.json", 'r') as f:
        sheetname_2_file_devided = json.load(f)
    res = {}
    count=0
    for sheetname in sheetname_2_file_devided:
        count+=1
        
        res[sheetname] = []
        print(count, len(sheetname_2_file_devided))
        # print(sheetname_2_file_devided[sheetname])
        file_list_list = sheetname_2_file_devided[sheetname]
        # if len(file_list_list) != 1:
        #     print(count, len(sheetname_2_file_devided))
        #     print(len(file_list_list))
        for file_list in file_list_list:
            # if len(file_list_list) != 1:
            #     print(count, len(sheetname_2_file_devided))
            #     print(len(file_list_list))
            #     print(len(file_list['filenames']))
            for file1_ in file_list['filenames']:
                need_continue = True
                for file2_ in file_list['filenames']:
                    if(file1_ == file2_):
                        need_continue=False
                        continue
                    if need_continue:
                        continue
                    res[sheetname].append((file1_, file2_))
    with open("list_positive_pair.json", 'w') as f:
        json.dump(res, f)


def get_most_positive_pair():
    with open("../AnalyzeDV/sheetname_2_file_devided.json", 'r') as f:
        sheetname_2_file_devided = json.load(f)
    with open('positive_res.json', 'r') as f:
        extracted_res = json.load(f)
        
    res = {}
    count=0
    coun = 0
    need_feature = {}
    added = []
    print(sheetname_2_file_devided.keys())
    while coun <= 100000:
        file_sheet = random.choice(extracted_res)
        filename = file_sheet.split('---')[0]
        sheetname = file_sheet.split('---')[1][0:-5]
        print(file_sheet)
        file_list_list = sheetname_2_file_devided[sheetname]

        for file_list in file_list_list:
            found = False
            for file_ in file_list:
                if file_.split('/')[-1] == filename:
                    full_file = file_
                    found = True
                    break
            if found:
                anther_file_ = random.choice(file_list)
                count = 0
                while anther_file_.split('/')[-1] == filename and count < len(file_list):
                    anther_file_ = random.choice(file_list)
                    count += 1
                break

        if full_file+'--'+sheetname+','+anther_file_+'--'+sheetname in added:
            continue
        res[sheetname].append((full_file, anther_file_))
        if sheetname not in need_feature:
            need_feature[sheetname] = []
        if full_file not in need_feature[sheetname]:
            need_feature[sheetname].append(full_file)
        if anther_file_ not in need_feature[sheetname]:
            need_feature[sheetname].append(anther_file_)
        coun += 1
        added.append(file1_+'--'+sheetname+','+file2_+'--'+sheetname)
        
                
        # if coun >= 100000:
        #     break
    with open("100000_positive_pair_1.json", 'w') as f:
        json.dump(res, f)
    with open("100000_positive_need_feature_1.json", 'w') as f:
        json.dump(need_feature, f)
    print(coun)

def get_10000_negative_feature():
    with open("100000_negative_pair.json",'r') as f:
        negative_pair = json.load(f)
    index2filesheet = []
    index2filesheet1 = []
    negative_feature = []
    negative_feature1 = []
    count =0
    
    for pair in negative_pair:
        if count < 50000:
            filename1 = pair[0][0]
            sheetname1 = pair[0][1]
            filename2 = pair[1][0]
            sheetname2 = pair[1][1]
            # print("#############")
            # print(os.path.exists("../../data/sheet_features/"+filename1.split('/')[-1]+"---"+sheetname1+".json"), os.path.exists("../../data/sheet_features/"+filename2.split('/')[-1]+"---"+sheetname2+".json"))
            # print("../../data/sheet_features/"+filename1.split('/')[-1]+"---"+sheetname1+".json", "../../data/sheet_features/"+filename2.split('/')[-1]+"---"+sheetname2+".json")
            if not os.path.exists("../../data/sheet_features/"+filename1.split('/')[-1]+"---"+sheetname1+".json") or not os.path.exists("../../data/sheet_features/"+filename2.split('/')[-1]+"---"+sheetname2+".json"):
                continue
            with open("../../data/sheet_features/"+filename1.split('/')[-1]+"---"+sheetname1+".json", 'r') as f:
                origin_feature_1 = json.load(f)
            with open("../../data/sheet_features/"+filename2.split("/")[-1]+"---"+sheetname2+".json", 'r') as f:
                origin_feature_2 = json.load(f)
            feature1 = []
            feature2 = []
            # feature = get_feature_vector(item[''])
            for one_cell_feature in origin_feature_1['sheetfeature']:
                new_feature = []
                for key1 in one_cell_feature:
                    new_feature.append(one_cell_feature[key1])
                feature1.append(new_feature)
            feature1 = get_feature_vector(feature1)

            for one_cell_feature in origin_feature_2['sheetfeature']:
                new_feature = []
                for key1 in one_cell_feature:
                    new_feature.append(one_cell_feature[key1])
                feature2.append(new_feature)
            feature2 = get_feature_vector(feature2)
            # print(origin_feature_1)
            # feature2 = get_feature_vector(origin_feature_2['sheetfeature'])
            negative_feature.append((feature1, feature2))
            index2filesheet.append((filename1+"---"+sheetname1, filename2+"---"+sheetname2))
            count += 1
        else:
            filename1 = pair[0][0]
            sheetname1 = pair[0][1]
            filename2 = pair[1][0]
            sheetname2 = pair[1][1]
            # print("#############")
            # print(os.path.exists("../../data/sheet_features/"+filename1.split('/')[-1]+"---"+sheetname1+".json"), os.path.exists("../../data/sheet_features/"+filename2.split('/')[-1]+"---"+sheetname2+".json"))
            # print("../../data/sheet_features/"+filename1.split('/')[-1]+"---"+sheetname1+".json", "../../data/sheet_features/"+filename2.split('/')[-1]+"---"+sheetname2+".json")
            if not os.path.exists("../../data/sheet_features/"+filename1.split('/')[-1]+"---"+sheetname1+".json") or not os.path.exists("../../data/sheet_features/"+filename2.split('/')[-1]+"---"+sheetname2+".json"):
                continue
            with open("../../data/sheet_features/"+filename1.split('/')[-1]+"---"+sheetname1+".json", 'r') as f:
                origin_feature_1 = json.load(f)
            with open("../../data/sheet_features/"+filename2.split("/")[-1]+"---"+sheetname2+".json", 'r') as f:
                origin_feature_2 = json.load(f)
            feature1 = []
            feature2 = []
            # feature = get_feature_vector(item[''])
            for one_cell_feature in origin_feature_1['sheetfeature']:
                new_feature = []
                for key1 in one_cell_feature:
                    new_feature.append(one_cell_feature[key1])
                feature1.append(new_feature)
            feature1 = get_feature_vector(feature1)

            for one_cell_feature in origin_feature_2['sheetfeature']:
                new_feature = []
                for key1 in one_cell_feature:
                    new_feature.append(one_cell_feature[key1])
                feature2.append(new_feature)
            feature2 = get_feature_vector(feature2)
            negative_feature1.append((feature1, feature2))
            index2filesheet1.append((filename1+"---"+sheetname1, filename2+"---"+sheetname2))
            count += 1
        # break
    # print(count)
    # with open("100000_negative_feature.json", 'w') as f:
    #     json.dump(negative_feature, f)
    # with open("100000_negative_index.json", 'w') as f:
    #     json.dump(index2filesheet, f)
    # with open("100000_negative_feature1.json", 'w') as f:
    #     json.dump(negative_feature1, f)
    # with open("100000_negative_index1.json", 'w') as f:
    #     json.dump(index2filesheet1, f)

def look_positive_feature():
    res_set = set()
    with open('100000_positive_pair1.json','r') as f:
        positive_pair1 = json.load(f)
    with open("100000_positive_pair.json",'r') as f:
        positive_pair = json.load(f)

    for sheetname in positive_pair1:
        for pair in positive_pair1[sheetname]:
            res_set.add(sheetname+"---"+pair[0]+"---"+pair[1])
    for sheetname in positive_pair:
        for pair in positive_pair[sheetname]:
            res_set.add(sheetname+"---"+pair[0]+"---"+pair[1])
    print(len(res_set))
        # sheetname1 = list(positive_pair.keys())[0]
    # sheetname2 = list(positive_pair1.keys())[1]
    # print(sheetname1)
    # print(sheetname2)
    # print(positive_pair[sheetname1])
    # print(positive_pair1[sheetname2])

def get_10000_positive_feature():
    with open("100000_positive_pair.json",'r') as f:
        positive_pair = json.load(f)
    
    # with open("100000_positive_feature.json", 'w') as f:
    #     json.dump(positive_feature, f)
    # with open("100000_positive_index.json", 'w') as f:
    #     json.dump(index2filesheet, f)

    index2filesheet = []
    positive_feature = []
    count =0
    for sheetname in positive_pair:
        for pair in positive_pair[sheetname]:
            print(pair)
            if not os.path.exists("../../data/sheet_features/"+pair[0].split('/')[-1]+"---"+sheetname+".json") or not os.path.exists("../../data/sheet_features/"+pair[1].split('/')[-1]+"---"+sheetname+".json"):
                continue
    
            with open("../../data/sheet_features/"+pair[0].split('/')[-1]+"---"+sheetname+".json", 'r') as f:
                origin_feature_1 = json.load(f)
            with open("../../data/sheet_features/"+pair[1].split("/")[-1]+"---"+sheetname+".json", 'r') as f:
                origin_feature_2 = json.load(f)
            # print(origin_feature_1.keys())
            feature1 = []
            feature2 = []
            # feature = get_feature_vector(item[''])
            for one_cell_feature in origin_feature_1['sheetfeature']:
                new_feature = []
                for key1 in one_cell_feature:
                    new_feature.append(one_cell_feature[key1])
                feature1.append(new_feature)
            feature1 = get_feature_vector(feature1)

            for one_cell_feature in origin_feature_2['sheetfeature']:
                new_feature = []
                for key1 in one_cell_feature:
                    new_feature.append(one_cell_feature[key1])
                feature2.append(new_feature)
            feature2 = get_feature_vector(feature2)

            positive_feature.append((feature1, feature2))
            index2filesheet.append((pair[0]+"---"+sheetname, pair[1]+"---"+sheetname))
            count += 1
    print(count)
    with open("100000_positive_feature.json", 'w') as f:
        json.dump(positive_feature, f)
    with open("100000_positive_index.json", 'w') as f:
        json.dump(index2filesheet, f)

def get_positive_feature():
    with open("all_positive_pair.json", 'r') as f:
        positive_pair = json.load(f)
    with open("id_2_feature.json", 'r') as f:
        id_2_feature = json.load(f)
    with open("sheet_2_id.json", 'r') as f:
        sheet_2_id = json.load(f)
    res = []
    id_list = []
    positive_num=0

    count=1
    for sheetname in positive_pair:
        print(count, len(positive_pair))
        count+=1
        for pair in positive_pair[sheetname]:
            # print(cnn_features_dict[sheetname][pair[0]])
            # print(cnn_features_dict[sheetname][pair[1]])
            
            if sheetname not in sheet_2_id:
                continue
            if pair[0] not in sheet_2_id[sheetname] or pair[1] not in sheet_2_id[sheetname]:
                continue
            feature1 = id_2_feature[str(sheet_2_id[sheetname][pair[0]])]
            # feature1.append(sheet_2_id[sheetname][pair[0]])
            feature2 = id_2_feature[str(sheet_2_id[sheetname][pair[1]])]
            # feature2.append(sheet_2_id[sheetname][pair[1]])
            res.append((feature1, feature2))
            id_list.append((sheet_2_id[sheetname][pair[0]], sheet_2_id[sheetname][pair[1]]))
            # break
            positive_num+=1
        # break
    print(positive_num)
    with open("positive_feature.json",'w') as f:
        json.dump(res, f)
    with open("positive_id.json",'w') as f:
        json.dump(id_list, f)

def get_most_positive_feature():
    with open("positive_pair.json", 'r') as f:
        positive_pair = json.load(f)
    with open("id_2_feature.json", 'r') as f:
        id_2_feature = json.load(f)
    with open("sheet_2_id.json", 'r') as f:
        sheet_2_id = json.load(f)
    res = []
    id_list = []
    positive_num=0

    count=1
    for sheetname in positive_pair:
        print(count, len(positive_pair))
        count+=1
        for pair in positive_pair[sheetname]:
            # print(cnn_features_dict[sheetname][pair[0]])
            # print(cnn_features_dict[sheetname][pair[1]])
            
            if sheetname not in sheet_2_id:
                continue
            if pair[0] not in sheet_2_id[sheetname] or pair[1] not in sheet_2_id[sheetname]:
                continue
            feature1 = id_2_feature[str(sheet_2_id[sheetname][pair[0]])]
            # feature1.append(sheet_2_id[sheetname][pair[0]])
            feature2 = id_2_feature[str(sheet_2_id[sheetname][pair[1]])]
            # feature2.append(sheet_2_id[sheetname][pair[1]])
            res.append((feature1, feature2))
            id_list.append((sheet_2_id[sheetname][pair[0]], sheet_2_id[sheetname][pair[1]]))
            # break
            positive_num+=1
        # break
    print(positive_num)
    with open("positive_feature.json",'w') as f:
        json.dump(res, f)
    with open("positive_id.json",'w') as f:
        json.dump(id_list, f)

def get_negative_pair():
    with open("../AnalyzeDV/sheetname_2_file_devided.json", 'r') as f:
            sheetname_2_file_devided = json.load(f)

    res = []
    count=0
    need_feature = {}
    while count <= 400000:
        print(count, 400000)
        sheetname1 = random.choice(list(sheetname_2_file_devided.keys()))
        sheetname2 = random.choice(list(sheetname_2_file_devided.keys()))
        filename1 = random.choice(random.choice(list(sheetname_2_file_devided[sheetname1])))
        filename2 = random.choice(random.choice(list(sheetname_2_file_devided[sheetname2])))
        if(sheetname1==sheetname2 and filename1==filename2):
            continue
        if(sheetname1==sheetname2):
            index1=0
            index2=0
            need_continue=False
            is_in_1= False
            is_in_2 = False
            for index,filelist in enumerate(sheetname_2_file_devided[sheetname1]):
                if filename1 in filelist:
                    is_in_1=True
                if filename2 in filelist:
                    is_in_2=True
                if is_in_1 and is_in_2:
                    need_continue = True
                    break
            if need_continue:
                continue
        pair={}
        pair['sheet'] = (sheetname1, sheetname2)
        pair['file'] = (filename1, filename2)
        res.append(pair)
        if sheetname1 not in need_feature:
            need_feature[sheetname1] = []
        if sheetname2 not in need_feature:
            need_feature[sheetname2] = []
        if filename1 not in need_feature[sheetname1]:
            need_feature[sheetname1].append(filename1)
        if filename2 not in need_feature[sheetname2]:
            need_feature[sheetname2].append(filename2)

        count+=1
                 
    with open("400000_negative_pair.json", 'w', encoding='utf-8') as f:
        json.dump(res, f)
    with open("400000_negative_need_feature.json", 'w', encoding='utf-8') as f:
        json.dump(need_feature, f)
# def get_negative_feature():
def get_negative_feature():
    with open("negative_pair.json", 'r') as f:
        negative_pair = json.load(f)
    with open("id_2_feature.json", 'r') as f:
        id_2_feature = json.load(f)
    with open("sheet_2_id.json", 'r') as f:
        sheet_2_id = json.load(f)
    res = []
    id_list = []
    positive_num=0

    count=1
    feature_id = 0
    for pair in negative_pair:
        sheetname1 = pair['sheet'][0]
        sheetname2 = pair['sheet'][1]
        filename1 = pair['file'][0]
        filename2 = pair['file'][1]
        print(count, len(negative_pair))
        count+=1
    
        if sheetname1 not in sheet_2_id or sheetname2 not in sheet_2_id:
            continue
        if filename1 not in sheet_2_id[sheetname1] or filename2 not in sheet_2_id[sheetname2]:
            continue
        feature1 = id_2_feature[str(sheet_2_id[sheetname1][filename1])]
        # feature1.append(sheet_2_id[sheetname1][filename1])
        feature2 = id_2_feature[str(sheet_2_id[sheetname2][filename2])]
        # feature2.append(sheet_2_id[sheetname2][filename2])
        res.append((feature1, feature2))
        id_list.append((sheet_2_id[sheetname1][filename1], sheet_2_id[sheetname2][filename2]))
        # break
        positive_num+=1
        # break
    print(positive_num)
    with open("negative_feature.json",'w',encoding='utf-8') as f:
        json.dump(res, f)
    with open("negative_id.json",'w',encoding='utf-8') as f:
        json.dump(id_list, f)

# def found_sheet_file(batch, index):


def split_data(is_positive):
    print('start saving data...')
    if is_positive:
        filenamelist = ['100000_positive_feature.json', '100000_positive_feature1.json']
    else:
        filenamelist = ['100000_negative_feature.json', '100000_negative_feature1.json']

    batch = 1
    for filename in filenamelist:

        with open(filename,'r') as f:
            negative_feature = json.load(f)
        # with open("100000_negative_feature.json",'r') as f:
        #     negative_feature = json.load(f)
        with open(filename.replace('feature','index'),'r') as f:
            negative_id = json.load(f)
    # with open("100000_negative_index.json",'r') as f:
    #     negative_id = json.load(f)
    # with open("100000_positive_feature1.json",'r') as f:
    #     positive_feature1 = json.load(f)
    # with open("100000_positive_index1.json",'r') as f:
    #     positive_id1 = json.load(f)
    # with open("positive_id.json",'r') as f:
    #     positive_id = json.load(f)
    # with open("negative_id.json",'r') as f:
    #     negative_id = json.load(f)

        x_1 = []
        x_2 = []
        x_3 = []
        x_4 = []
        y_1 = []
        y_2 = []
        y_3 = []
        y_4 = []
        index1 = []
        index2 = []
        index3 = []
        index4 = []
        print("start extract.....")
        # randnum = random.randint(0,100)
        # random.seed(randnum)
        # random.shuffle(positive_feature)
        # random.seed(randnum)
        # random.shuffle(positive_id)
        # randnum = random.randint(0,100)
        # random.seed(randnum)
        # random.shuffle(negative_feature)
        # random.seed(randnum)
        # random.shuffle(negative_id)
        count = 0
        # positive_features = [positive_feature, positive_feature1]

        # positive_ids = [positive_id, positive_id1]
        # for ind,positive_feature in enumerate(positive_features):
        #     positive_id = positive_ids[ind]
        for index, item in enumerate(negative_feature):
            # if index<int(len(positive_feature)/4):
            print(index, len(negative_feature))
            x_1.append(item)
            y_1.append(0)
            index1.append("neg:"+str(negative_feature[index]))
            if index % 5000 == 0:
                if is_positive:
                    np.save("5000_positive_x_"+str(batch)+".npy", x_1)
                    np.save("5000_positive_y_"+str(batch)+".npy", y_1)
                    np.save("5000_positive_ind_"+str(batch)+".npy", index1)
                else:
                    np.save("5000_negative_x_"+str(batch)+".npy", x_1)
                    np.save("5000_negative_y_"+str(batch)+".npy", y_1)
                    np.save("5000_negative_ind_"+str(batch)+".npy", index1)
                gc.collect()
                del x_1
                del y_1
                del index1
                gc.collect()
                x_1=[]
                y_1=[]
                index1=[]
                batch += 1
        if is_positive:
            np.save("5000_positive_x_"+str(batch)+".npy", x_1)
            np.save("5000_positive_y_"+str(batch)+".npy", y_1)
            np.save("5000_positive_ind_"+str(batch)+".npy", index1)
        else:
            np.save("5000_negative_x_"+str(batch)+".npy", x_1)
            np.save("5000_negative_y_"+str(batch)+".npy", y_1)
            np.save("5000_negative_ind_"+str(batch)+".npy", index1)
        gc.collect()
        del x_1
        del y_1
        del index1
        del negative_feature
        del negative_id
        gc.collect()
        x_1=[]
        y_1=[]
        index1=[]
        batch += 1

            # if index>=int(len(positive_feature)/4) and index<int(len(positive_feature)/2):
            #     x_2.append(item)
            #     y_2.append(1)
            #     index2.append("pos:"+str(positive_id[index]))
            # if index>=int(len(positive_feature)/2) and index<int(3*len(positive_feature)/4):
            #     x_3.append(item)
            #     y_3.append(1)
            #     index3.append("pos:"+str(positive_id[index]))
            # if index>=int(3*len(positive_feature)/4):
            #     x_4.append(item)
            #     y_4.append(1)
            #     index4.append("pos:"+str(positive_id[index]))

    
        # if index>=int(len(negative_feature)/4) and index<int(len(negative_feature)/2):
        #     x_2.append(item)
        #     y_2.append(0)
        #     index2.append("neg:"+str(negative_id[index]))
        # if index>=int(len(negative_feature)/2) and index<int(3*len(negative_feature)/4):
        #     x_3.append(item)
        #     y_3.append(0)
        #     index3.append("neg:"+str(negative_id[index]))
        # if index>=int(3*len(negative_feature)/4):
        #     x_4.append(item)
        #     y_4.append(0)
        #     index4.append("neg:"+str(negative_id[index]))

    # with open("shuffle_positive_feature1", 'w') as f:
    #     json.dump(positive_feature, f)
    # with open("negative_feature", 'w') as f:
    #     json.dump(positive_feature, f)

    # with open("100000_cnn_x_1.json", 'w') as f:
    #     json.dump(x_1, f)
    # with open("100000_cnn_x_2.json", 'w') as f:
    #     json.dump(x_2, f)
    # with open("100000_cnn_x_3.json", 'w') as f:
    #     json.dump(x_3, f)
    # with open("100000_cnn_x_4.json", 'w') as f:
    #     json.dump(x_4, f)

    # with open("100000_cnn_y_1.json", 'w') as f:
    #     json.dump(y_1, f)
    # with open("100000_cnn_y_2.json", 'w') as f:
    #     json.dump(y_2, f)
    # with open("100000_cnn_y_3.json", 'w') as f:
    #     json.dump(y_3, f)
    # with open("100000_cnn_y_4.json", 'w') as f:
    #     json.dump(y_4, f)

    # with open("100000_cnn_index_1.json", 'w') as f:
    #     json.dump(index1, f)
    # with open("100000_cnn_index_2.json", 'w') as f:
    #     json.dump(index2, f)
    # with open("100000_cnn_index_3.json", 'w') as f:
    #     json.dump(index3, f)
    # with open("100000_cnn_index_4.json", 'w') as f:
    #     json.dump(index4, f)
    # return x_1,x_2,x_3,x_4,y_1,y_2,y_3,y_4


    # np.save("cnn_x_1.npy",x_1)
    # np.save("cnn_x_2.npy",x_2)
    # np.save("cnn_x_3.npy",x_3)
    # np.save("cnn_x_4.npy",x_4)

    # np.save("cnn_y_1.npy",y_1)
    # np.save("cnn_y_2.npy",y_2)
    # np.save("cnn_y_3.npy",y_3)
    # np.save("cnn_y_4.npy",y_4)

def training_testing_cosine(x_train, y_train, x_test, y_test,batch):
    print('start training testing '+ str(batch)+"......")
    batch_size = 128
    model = CNNnet_Cosine()
    loss_func = torch.nn.L1Loss()
    # loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(),lr=0.001)
    # print(x_train[0])
    train_data = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.FloatTensor(x_test),torch.Tensor(y_test))

    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_data,batch_size = len(test_data), shuffle=False)

    loss_count = []
    for epoch in range(10):
        for i,(x,y) in enumerate(train_loader):
            # print('len trainx', len(x))
            x1 = x[:,0,:].reshape(len(x),100,10,10).permute(0,3,1,2)
            x2 = x[:,1,:].reshape(len(x),100,10,10).permute(0,3,1,2)
            

            x1 = Variable(x1) # torch.Size([batch_size, 1000, 10])
            x2 = Variable(x2) ## torch.Size([batch_size, 1000, 10])
            # y = torch.LongTensor(y)
            batch_y = Variable(y) # torch.Size([batch_size])
            # 获取最后输出
            # print(batch_x)
            out = model(x1, x2) # torch.Size([128,10])
            # 获取损失
            # print(out)
            # print(batch_y)
            loss = loss_func(out,batch_y)
            # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上
            if i%20 == 0:
                loss_count.append(loss)
                print('{}:\t'.format(i), loss.item())
                torch.save(model,'cnn_model_'+str(batch))
            if i % 100 == 0:
                for test_x,test_y in test_loader:
                    print('len testx', len(test_x))
                    test_x1 = test_x[:,0,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
                    test_x2 = test_x[:,1,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
                    test_x1 = Variable(test_x1) # torch.Size([batch_size, 1000, 10])
                    test_x2 = Variable(test_x2) ## torch.Size([batch_size, 1000, 10])
                        # y = torch.LongTensor(y)
                    test_y = Variable(test_y) # torch.Size([batch_size])
                    out = model(test_x1,test_x2)
                    # print('test_out:\t',torch.max(out,1)[1])
                    # print('test_y:\t',test_y)
                    # print(out
                    bina_out = []
                    for i in out:
                        bina_out.append(0 if i<0.5 else 1)
                    # # print(test_y)
                    # # print('torch.max(out,0).numpy',torch.max(out,0).numpy())
                    accuracy = bina_out== test_y.numpy()
                    # accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
                    
                    print('accuracy:\t',accuracy.mean())
                    break

    for test_x,test_y in test_loader:
        print('len testx', len(test_x))
        test_x1 = test_x[:,0,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
        test_x2 = test_x[:,1,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
        test_x1 = Variable(test_x1) # torch.Size([batch_size, 1000, 10])
        test_x2 = Variable(test_x2) ## torch.Size([batch_size, 1000, 10])
            # y = torch.LongTensor(y)
        test_y = Variable(test_y) # torch.Size([batch_size])
        out = model(test_x1,test_x2)
        # print('test_out:\t',torch.max(out,1)[1])
        # print('test_y:\t',test_y)
        bina_out = []
        for i in out:
            bina_out.append(0 if i<0.5 else 1)
        # print(test_y)
        # print('torch.max(out,0).numpy',torch.max(out,0).numpy())
        accuracy = bina_out== test_y.numpy()
        np.save("pred_1_cos_"+str(batch), out.detach().numpy())
        # np.save("pred_"+str(batch), out.detach().numpy())
        np.save("test_1_cos_"+str(batch), test_y.numpy())
        # accuracy = torch.max(out,0).numpy() == test_y.numpy()
        suc_num = len([i for i in accuracy if i==True])
        # print('accuracy:\t',accuracy.mean())
    
    # plt.figure('PyTorch_CNN_Loss')
    # plt.plot(loss_count,label='Loss')
    # plt.legend()
    # plt.show()
    return accuracy, suc_num

def training_testing(x_train, y_train, x_test, y_test,batch):
    print('start training testing '+ str(batch)+"......")
    batch_size = 128
    model = CNNnet()
    # loss_func = torch.nn.L1Loss()
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(),lr=0.001)
    # print(x_train[0])
    train_data = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.FloatTensor(x_test),torch.Tensor(y_test))

    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_data,batch_size = len(test_data), shuffle=False)

    loss_count = []
    for epoch in range(10):
        for i,(x,y) in enumerate(train_loader):
            # print('len trainx', len(x))
            x1 = x[:,0,:].reshape(len(x),100,10,10).permute(0,3,1,2)
            x2 = x[:,1,:].reshape(len(x),100,10,10).permute(0,3,1,2)
            

            x1 = Variable(x1) # torch.Size([batch_size, 1000, 10])
            x2 = Variable(x2) ## torch.Size([batch_size, 1000, 10])
            # y = torch.LongTensor(y)
            batch_y = Variable(y) # torch.Size([batch_size])
            # 获取最后输出
            # print(batch_x)
            out = model(x1, x2) # torch.Size([128,10])
            # 获取损失
            # print(out)
            # print(batch_y)
            loss = loss_func(out,batch_y)
            # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上
            if i%20 == 0:
                loss_count.append(loss)
                print('{}:\t'.format(i), loss.item())
                torch.save(model,'cnn_model_'+str(batch))
            if i % 100 == 0:
                for test_x,test_y in test_loader:
                    print('len testx', len(test_x))
                    test_x1 = test_x[:,0,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
                    test_x2 = test_x[:,1,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
                    test_x1 = Variable(test_x1) # torch.Size([batch_size, 1000, 10])
                    test_x2 = Variable(test_x2) ## torch.Size([batch_size, 1000, 10])
                        # y = torch.LongTensor(y)
                    test_y = Variable(test_y) # torch.Size([batch_size])
                    out = model(test_x1,test_x2)
                    # print('test_out:\t',torch.max(out,1)[1])
                    # print('test_y:\t',test_y)
                    # print(out
                    # bina_out = []
                    # for i in out:
                    #     bina_out.append(0 if i<0.5 else 1)
                    # # print(test_y)
                    # # print('torch.max(out,0).numpy',torch.max(out,0).numpy())
                    # accuracy = bina_out== test_y.numpy()
                    accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
                    
                    print('accuracy:\t',accuracy.mean())
                    break

    for test_x,test_y in test_loader:
        print('len testx', len(test_x))
        test_x1 = test_x[:,0,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
        test_x2 = test_x[:,1,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
        test_x1 = Variable(test_x1) # torch.Size([batch_size, 1000, 10])
        test_x2 = Variable(test_x2) ## torch.Size([batch_size, 1000, 10])
            # y = torch.LongTensor(y)
        test_y = Variable(test_y) # torch.Size([batch_size])
        out = model(test_x1,test_x2)
        # print('test_out:\t',torch.max(out,1)[1])
        # print('test_y:\t',test_y)
        # bina_out = []
        # for i in out:
        #     bina_out.append(0 if i<0.5 else 1)
        # print(test_y)
        # print('torch.max(out,0).numpy',torch.max(out,0).numpy())
        # accuracy = bina_out== test_y.numpy()
        np.save("out_1_"+str(batch), out.detach().numpy())
        np.save("pred_1_"+str(batch), torch.max(out,1)[1].numpy())
        # np.save("pred_"+str(batch), out.detach().numpy())
        np.save("test_1_"+str(batch), test_y.numpy())
        accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
        # accuracy = torch.max(out,0).numpy() == test_y.numpy()
        suc_num = len([i for i in accuracy if i==True])
        # print('accuracy:\t',accuracy.mean())
    
    # plt.figure('PyTorch_CNN_Loss')
    # plt.plot(loss_count,label='Loss')
    # plt.legend()
    # plt.show()
    return accuracy, suc_num

def training_100000(x_train, y_train):
    print('start training testing......')
    batch_size = 128
    model = CNNnet()
    if os.path.exists('100000_mlp_model'):
        model = torch.load('100000_mlp_model')
    else:
        model = CNNnet()
    # loss_func = torch.nn.L1Loss()
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(),lr=0.001)
    # print(x_train[0])
    # print(x_train)
    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    train_data = TensorDataset(x_train, y_train)
    print(len(x_train))
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

    loss_count = []
    # for epoch in range(10):
    for i,(x,y) in enumerate(train_loader):
        # print('len trainx', len(x))
        
        x1 = x[:,0,:].reshape(len(x),100,10,10).permute(0,3,1,2)
        x2 = x[:,1,:].reshape(len(x),100,10,10).permute(0,3,1,2)
        

        x1 = Variable(x1) # torch.Size([batch_size, 1000, 10])
        x2 = Variable(x2) ## torch.Size([batch_size, 1000, 10])
        # y = torch.LongTensor(y)
        batch_y = Variable(y) # torch.Size([batch_size])
        # 获取最后输出
        # print(batch_x)
        out = model(x1, x2) # torch.Size([128,10])
        # 获取损失
        # print(out)
        # print(batch_y)
        loss = loss_func(out,batch_y)
        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward() # 误差反向传播，计算参数更新值
        opt.step() # 将参数更新值施加到net的parmeters上
        if i%20 == 0:
            loss_count.append(loss)
            print('{}:\t'.format(i), loss.item())
            torch.save(model,'100000_mlp_model')
        

def batch_100000_training():
    filelist = os.listdir(".")

    positive_file = []
    negative_file = []
    pairs = []
    for filename in filelist:
        if "5000_positive_x" in filename:
            positive_file.append(filename)
        if "5000_negative_x" in filename:
            negative_file.append(filename)
    for index, filename in enumerate(positive_file):
        if index >= len(negative_file):
            pairs.append([filename])
        else:
            pairs.append([filename, negative_file[index]])

    
    for epoch in range(3):
        count = 0
        for pair in pairs:
            if len(pair)==0:
                continue
            count += 1
            print(epoch, count, len(pairs))
            x_train = []
            y_train = []
            print(pair)
            for x_filename in pair:
                features = np.load(x_filename, allow_pickle=True)
                y_filename = x_filename.replace('_x_', '_y_')
                # labels = np.load(y_filename, allow_pickle=True)
                labels = []
                if 'negative' in y_filename:
                    for i in range(len(features)):
                        labels.append(0)
                    labels = np.array(labels)
                else:
                    labels = np.load(y_filename, allow_pickle=True)
                is_str = False
                for index, feature in enumerate(features):
                    # print(feature)
                    
                    for i in feature:
                        for j in i:
                            if type(j).__name__ == 'str':
                                print(x_filename)
                                is_str = True
                                break
                        if is_str:
                            break
                    if is_str:
                        break
                    x_train.append(feature)
                    y_train.append(labels[index])
            if len(x_train) != 10000:
                continue
            # print(x_train[0][0])
            training_100000(x_train, y_train)
            gc.collect()
            del x_train
            del y_train
            del features
            del labels
            gc.collect()

def batch_testing():
    all_suc_num = 0
    all_data = 0
    print('load data.......')
    with open("cnn_x_1.json", 'r') as f:
        x1 = json.load(f)
    with open("cnn_x_2.json", 'r') as f:
        x2 = json.load(f)
    with open("cnn_x_3.json", 'r') as f:
        x3 = json.load(f)
    with open("cnn_x_4.json", 'r') as f:
        x4 = json.load(f)

    with open("cnn_y_1.json", 'r') as f:
        y1 = json.load(f)
    with open("cnn_y_2.json", 'r') as f:
        y2 = json.load(f)
    with open("cnn_y_3.json", 'r') as f:
        y3 = json.load(f)
    with open("cnn_y_4.json", 'r') as f:
        y4 = json.load(f)
    print('load data end.......')
    x = [x1,x2,x3,x4]
    y = [y1,y2,y3,y4]
    for batch in [2]:
        # with open("x_"+str(batch)+".json", 'r') as f:
        #     x_test = json.load(f)
        # with open("y_"+str(batch)+".json", 'r') as f:
        #     y_test = json.load(f)
        x_test = x[batch-1]
        y_test = y[batch-1]
        # x_test = np.load('x_'+str(batch)+'.npy', allow_pickle=True)
        # y_test = np.load('y_'+str(batch)+'.npy', allow_pickle=True)
        all_data+=len(x_test)
        x_train_list = []
        y_train_list = []
        for index,train_batch in enumerate(list(set([1,2,3,4])-set([batch]))):
            # x_train_sub = np.load('x_'+str(train_batch)+'.npy', allow_pickle=True)
            # y_train_sub = np.load('y_'+str(train_batch)+'.npy', allow_pickle=True)
            x_train_sub = x[train_batch-1]
            y_train_sub = y[train_batch-1]
            x_train_list.append(x_train_sub)
            y_train_list.append(y_train_sub)

        x_train = np.concatenate((x_train_list[0], x_train_list[1], x_train_list[2]))
        y_train = np.concatenate((y_train_list[0], y_train_list[1], y_train_list[2]))

        accuracy, suc_num = training_testing(x_train, y_train, x_test, y_test,batch)
        # accuracy, suc_num = training_testing_cosine(x_train, y_train, x_test, y_test,batch)
        print('accuracy_'+str(batch)+':\t',accuracy.mean())
        all_suc_num+=suc_num
    print('suc_num', suc_num)
    print('all_data', all_data)
    print('all accuracy:\t', suc_num/all_data)

def look_fail():
    out4 = np.load("out_100000_3.npy", allow_pickle=True)
    pred4 = np.load("pred_100000_3.npy", allow_pickle=True)
    test4 = np.load("test_100000_3.npy", allow_pickle=True)

    fn = [out4[index] for index, i in enumerate(pred4) if i!=test4[index] and i==False]
    pprint.pprint(fn)

def cnn_evaluate(model_type=0):
    if model_type==0:
        pred1 = np.load("pred_1_cos_1.npy", allow_pickle=True)
        test1 = np.load("test_1_cos_1.npy", allow_pickle=True)
        pred2 = np.load("pred_1_cos_2.npy", allow_pickle=True)
        test2 = np.load("test_1_cos_2.npy", allow_pickle=True)
        pred3 = np.load("pred_1_cos_3.npy", allow_pickle=True)
        test3 = np.load("test_1_cos_3.npy", allow_pickle=True)
        pred4 = np.load("pred_1_cos_4.npy", allow_pickle=True)
        test4 = np.load("test_1_cos_4.npy", allow_pickle=True)
    elif model_type==1:
        pred1 = np.load("pred_100000_0.npy", allow_pickle=True)
        test1 = np.load("test_100000_0.npy", allow_pickle=True)
        pred2 = np.load("pred_100000_1.npy", allow_pickle=True)
        test2 = np.load("test_100000_1.npy", allow_pickle=True)
        pred3 = np.load("pred_100000_2.npy", allow_pickle=True)
        test3 = np.load("test_100000_2.npy", allow_pickle=True)
        pred4 = np.load("pred_100000_3.npy", allow_pickle=True)
        test4 = np.load("test_100000_3.npy", allow_pickle=True)
        print(pred1)
        print(test1)
    else:
        pred1 = np.load("pred_1_1.npy", allow_pickle=True)
        test1 = np.load("test_1_1.npy", allow_pickle=True)
        pred2 = np.load("pred_1_2.npy", allow_pickle=True)
        test2 = np.load("test_1_2.npy", allow_pickle=True)
        pred3 = np.load("pred_1_3.npy", allow_pickle=True)
        test3 = np.load("test_1_3.npy", allow_pickle=True)
        pred4 = np.load("pred_1_4.npy", allow_pickle=True)
        test4 = np.load("test_1_4.npy", allow_pickle=True)

    tp = [index for index, i in enumerate(pred1) if i==test1[index] and i==True]
    tp += [index for index, i in enumerate(pred2) if i==test2[index] and i==True]
    tp += [index for index, i in enumerate(pred3) if i==test3[index] and i==True]
    tp += [index for index, i in enumerate(pred4) if i==test4[index] and i==True]

    tn = [index for index, i in enumerate(pred1) if i==test1[index] and i==False]
    tn += [index for index, i in enumerate(pred2) if i==test2[index] and i==False]
    tn += [index for index, i in enumerate(pred3) if i==test3[index] and i==False]
    tn += [index for index, i in enumerate(pred4) if i==test4[index] and i==False]

    fp = [index for index, i in enumerate(pred1) if i!=test1[index] and i==True]
    fp += [index for index, i in enumerate(pred2) if i!=test2[index] and i==True]
    fp += [index for index, i in enumerate(pred3) if i!=test3[index] and i==True]
    fp += [index for index, i in enumerate(pred4) if i!=test4[index] and i==True]

    fn = [index for index, i in enumerate(pred1) if i!=test1[index] and i==False]
    fn += [index for index, i in enumerate(pred2) if i!=test2[index] and i==False]
    fn += [index for index, i in enumerate(pred3) if i!=test3[index] and i==False]
    fn += [index for index, i in enumerate(pred4) if i!=test4[index] and i==False]

    precision = len(tp)/(len(tp)+len(fp))
    recall = len(tp)/(len(tp)+len(fn))
    f1 = 2*(precision*recall)/(precision+recall)
    # print(batch, cosine)
    print('precision', precision)
    print('recall', recall)
    print('f1', f1)

def error_analyze():
    with open("positive_feature.json",'r') as f:
        positive_feature = json.load(f)
    with open("negative_feature.json",'r') as f:
        negative_feature = json.load(f)
    with open("positive_id.json",'r') as f:
        positive_id = json.load(f)
    with open("negative_id.json",'r') as f:
        negative_id = json.load(f)

    with open("cnn_x_4.json", 'r') as f:
        features = json.load(f)
    with open("id_2_feature.json", 'r') as f:
        id_2_feature = json.load(f)
    with open("sheet_2_id.json", 'r') as f:
        sheet_2_id = json.load(f)

    # features = np.load('cnn_x_4.npy', allow_pickle=True)
    pred_1 = np.load('pred_100000_3.npy', allow_pickle=True)
    test_1 = np.load('test_100000_3.npy', allow_pickle=True)

    re = pred_1==test_1
    # print(positive_id)
    sheet_pair_dict = {}
    str_ = ''
    for index, res in enumerate(re):
        # print(res)
        if(res==True):
            continue
        # print("res==False")
        # print(index1[index])
        feature = features[index]
        found=False

        # for one_fe in positive_feature:
        #     if feature == one_fe:
        #         found_index=True
        for index1, one_fe in enumerate(positive_feature):
            if feature == one_fe:
                found_index=index1
                found=True
                break
        if found:
            str_ += "pos#########################\n"
            indexing = positive_id[found_index]
        else:
            for index1, one_fe in enumerate(negative_feature):
                if feature == one_fe:
                    found_index=index1
                    found=True
                    break
            if found:
                str_ += "neg#########################\n"
                indexing = negative_id[found_index]
        if not found:
            # print("not found")
            continue
        # print(indexing)
        # split_index = indexing.split(':')
        id1 = indexing[0]
        id2 = indexing[1]
        for sheetname in sheet_2_id:
            for filename in sheet_2_id[sheetname]:
                if id1 == sheet_2_id[sheetname][filename]:
                    sn1 = sheetname
                    f1 = filename
                    str_ = str_ + str(sheetname) +',' + str(filename) + '\n'
                    # print(str_ + str(sheetname) +',' + str(filename) + '\n')
                    # str_ = str_ + str(x1[index]) + '\n'
                if id2 == sheet_2_id[sheetname][filename]:
                    sn2 = sheetname
                    f2 = filename
                    str_ = str_ + str(sheetname) +',' + str(filename) + '\n'
                    # print(str_ + str(sheetname) +',' + str(filename) + '\n')
        # if(split_index[0]=='pos'):
        #     str_ += "pos#########################\n"
        #     id1 = split_index[1].split(',')[0].strip()[1:]
        #     id2 = split_index[1].split(',')[1].strip()[:-1]
         
        #     for sheetname in sheet_2_id:
        #         for filename in sheet_2_id[sheetname]:
        #             if id1 == str(sheet_2_id[sheetname][filename]):
        #                 sn1 = sheetname
        #                 f1 = filename
        #                 str_ = str_ + str(sheetname) +',' + str(filename) + '\n'
        #                 # str_ = str_ + str(x1[index]) + '\n'
        #             if id2 == str(sheet_2_id[sheetname][filename]):
        #                 sn2 = sheetname
        #                 f2 = filename
        #                 str_ = str_ + str(sheetname) +',' + str(filename) + '\n'
        #                 # str_ = str_ + str(x1[index]) + '\n'
        # if(split_index[0]=='neg'):
        #     str_ += "neg#########################\n"
        #     id1 = split_index[1].split(',')[0].strip()[1:]
        #     id2 = split_index[1].split(',')[1].strip()[:-1]
         
        #     for sheetname in sheet_2_id:
        #         for filename in sheet_2_id[sheetname]:
        #             if id1 == str(sheet_2_id[sheetname][filename]):
        #                 sn1 = sheetname
        #                 f1 = filename
        #                 str_ = str_ + str(sheetname) +',' + str(filename) + '\n'
        #                 # str_ = str_ + str(x1[index]) + '\n'
        #             if id2 == str(sheet_2_id[sheetname][filename]):
        #                 sn2 = sheetname
        #                 f2 = filename
        #                 str_ = str_ + str(sheetname) +',' + str(filename) + '\n'
        #                 # str_ = str_ + str(x1[index]) + '\n'
        sp = sn1 + '------' + sn2
        if sp not in sheet_pair_dict:
            sheet_pair_dict[sp]=0
        sheet_pair_dict[sp]+=1
        with open('error_100000_4_no_feature.txt', 'w') as f:
            f.write(str_)
        with open("sheet_pair_num.json", 'w') as f:
            json.dump(sheet_pair_dict, f)
            # print(id1, id2)
    # print(negative_id.keys())
    
    # for index in re:
        # print(index1[index])
    # print((index1))

def is_same_template(dvinfo1, dvinfo2):
    with open("../AnalyzeDV/data/types/custom/change_xml_all_templates.json", 'r') as f:
        change_xml_all_templates = json.load(f)
    
    template_id1 = -1
    template_id2 = -2
    for template in change_xml_all_templates:
        if str(dvinfo1['ID'])+"---"+str(dvinfo1['batch_id']) in template['dvid_list']:
            template_id1 = template['id']
        if str(dvinfo2['ID'])+"---"+str(dvinfo2['batch_id']) in template['dvid_list']:
            template_id2 = template['id']
    return template_id1==template_id2


def get_custom_sheet_file_emb():
    with open("custom_id_2_feature.json", 'r') as f:
        id_2_feature = json.load(f)
    with open("custom_sheet_2_id.json", 'r') as f:
        sheet_2_id = json.load(f)
    print(len(sheet_2_id))
    print(len(id_2_feature))
    with open("../AnalyzeDV/data/types/custom/change_xml_cutom_list.json", 'r') as f:
        dvinfos = json.load(f)

    res = {}
    count=0
    for dvinfo in dvinfos:
        # print(count, len(dvinfos))
        count+=1
        filename = dvinfo["FileName"].replace("UnzipData/", "")
        sheetname = dvinfo["SheetName"]
        # print(list(sheet_2_id.keys())[0])
        # print(filename+"----"+sheetname)
        if filename+"----"+sheetname not in sheet_2_id:
            continue
    
        feature1 = id_2_feature[str(sheet_2_id[filename+"----"+sheetname])]
        feature1 = get_feature_vector(feature1)
        res[dvinfo["ID"]] = feature1
        # print(len(feature1))
        # id_list.append((sheet_2_id[sheetname][pair[0]], sheet_2_id[sheetname][pair[1]]))
        # break
        # positive_num+=1
    # print(len(res.keys()))
    with open('custom_feature_embed.json', 'w') as f:
        json.dump(res, f)

def get_custom_sheet_file_emb_from_model():
    with open('custom_feature_embed.json', 'r') as f:
        custom_feature_embed = json.load(f)
    print(custom_feature_embed[0])
    model = CNNnet() 
    model = torch.load('cnn_model_'+str(1))
    res = {}
    for dvid in custom_feature_embed:
        
        print(dvid)
        print(custom_feature_embed[dvid])
        embed_feature = model.get_embedding(torch.FloatTensor(custom_feature_embed[dvid]))
        res[dvid] = embed_feature
    with open('custom_feature_embed_from_model.json', 'w') as f:
        json.dump(res, f)


def check_custom_template():
    with open("custom_positive_pair.json",'r') as f:
        positive_pair = json.load(f)
    with open("../AnalyzeDV/data/types/custom/change_xml_cutom_list.json", 'r') as f:
        dvinfos = json.load(f)
    exact_match=0
    exact_correct = 0
    range_inter = 0
    range_inter_correct = 0
    most_closed=0
    most_closed_correct=0
    all_=0

    inter_list_right= []
    closed_list_right = []
    inter_list_wrong= []
    closed_list_wrong = []
    sheet_pair_dict = {}
    res_closed = [] # range不一样， 但是在表上找到了最近的其他dv
    res_not_found = [] # 找到了最近的dv，但是template不一样
    res_no_file = [] # 没有找到该文件的dv
    no_file = 0

    not_found = 0
    exact_sheet = set()
    closed_sheet = set()
    not_found_sheet = set()

    not_found_dvid_list = []
    for dvinfo in dvinfos:
        is_found=False
        is_found1=False
        is_found2=False
        # print(count, len(dvinfos))
        # count += 1  

        ltx = dvinfo['ltx']
        lty = dvinfo['lty']
        rbx = dvinfo['rbx']
        rby = dvinfo['rby']

        # if(ltx==0):
        #     continue
        for sheetname in positive_pair:
            # print(sheetname, dvinfo["SheetName"])
            if sheetname != dvinfo["SheetName"]:
                continue
            for f_pair in positive_pair[sheetname]:
                # print(f_pair)
                f1 = f_pair[0]
                f2 = f_pair[1]
                sn1 = sheetname
                sn2 = sheetname
                # print(f1, dvinfo['FileName'])
                # print(f2, dvinfo['FileName'])
                if(f1==dvinfo['FileName'] and sn1==dvinfo['SheetName']):
                    target_file = f1
                    target_sheet = sn1
                    cand_file=f2
                    cand_sheet=sn2
                    is_found1 = True
                elif(f2==dvinfo['FileName'] and sn2==dvinfo['SheetName']):
                    target_file = f2
                    target_sheet = sn2
                    cand_file=f1
                    cand_sheet=sn1
                    is_found2 = True

                if(not is_found1 or not is_found2):
                    continue
                else:
                    is_found = True
                    break
        if not is_found:
            not_found += 1
            all_+=1
            not_found_sheet.add(dvinfo["SheetName"])
            not_found_dvid_list.append(str(dvinfo['ID'])+"---"+str(dvinfo['batch_id']))
            continue
        print('found')

        is_exact = False
        is_inter = False
        is_most_closed = False
        closed_dvinfo=dvinfos[0]
        min_distance = 1000000
        is_found_dvinfo2 = False

        for dvinfo1 in dvinfos:
            if dvinfo1['ID']==dvinfo['ID']:
                continue
            if dvinfo1['FileName']==cand_file and dvinfo1['SheetName']==cand_sheet:
                ltx1 = dvinfo1['ltx']
                lty1 = dvinfo1['lty']
                rbx1 = dvinfo1['rbx']
                rby1 = dvinfo1['rby']
                
                dvinfo2 = dvinfo1
                is_found_dvinfo2 = True
                # if(ltx1==0):
                #     continue
                if(dvinfo['RangeAddress']==dvinfo1["RangeAddress"]):
                    is_exact=True
                    exact_match += 1
                    if(is_same_template(dvinfo, dvinfo1)):
                        exact_correct += 1
                    # all_ += 1
                    exact_sheet.add(dvinfo["SheetName"])
                    break
                # elif(ltx1 >= ltx and lty1>=lty and rbx1 <= rbx and rby1<=rby) or (ltx1 <= ltx and lty1<=lty and rbx1 >= rbx and rby1>=rby):
                #     is_inter=True
                #     range_inter += 1
                    
                #     if(is_same_template(dvinfo, dvinfo1)):
                #         range_inter_correct += 1
                #         inter_list_right.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                #     else:
                #         print('error:', dvinfo['ID'], dvinfo1["ID"])
                #         inter_list_wrong.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                #     # all_ += 1
                #     break
                else:
                    # if ltx1 == 0:
                    #     continue
                    distance = (ltx-ltx1)*(ltx-ltx1)+(rbx-rbx1)*(rbx-rbx1)
                    if min_distance > distance:
                        min_distance = distance
                        closed_dvinfo = dvinfo1
                        is_most_closed = True
        
        if is_found_dvinfo2:
            # if ltx1 == 0:
            #     continue
            # =True
            # if is_exact == False:
            #     print("##########################")
            #     print(dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"])
            #     print(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])   
            if is_most_closed == True and is_exact==False:
                most_closed += 1
                # .append((target_file, target_sheet, cand_file, cand_sheet))
                closed_sheet.add(dvinfo["SheetName"])
                if(is_same_template(dvinfo, dvinfo2)):
                    most_closed_correct += 1
                    res_closed.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                    print(dvinfo['ID'], dvinfo2['ID'])
                    closed_list_right.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                else:
                    closed_list_wrong.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                # else:
                    # # print('error:', dvinfo['ID'], dvinfo1["ID"])
                    # print("closed##########################")
                    # print(dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"])
                    # print(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])
                    # res_not_found.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                    
                        
                
            # elif is_exact == False:
                # print("not found##########################")
                # print(dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"])
                # print(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])  
                # res_not_found.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
        else:
            no_file+=1
            not_found_sheet.add(dvinfo["SheetName"])
            # closed_list.append((target_file, target_sheet, cand_file, cand_sheet))
        #     # if not is_exact and not is_inter:
        #     #     is_most_closed=True
        #     #     most_closed +=1
        #     #     if(is_same_template(dvinfo, closed_dvinfo)):
        #     #         most_closed_correct += 1
        #     #     else:
        #     print("########################")
        #     print(dvinfo2['ID'])
        #     print(dvinfo2['SheetName'])
        #     print(dvinfo2['FileName'])
        #     print(dvinfo['ID'])
        #     print(dvinfo['SheetName'])
        #     print(dvinfo['FileName'])
        # else:
        #     print("########################")
        #     print(dvinfo['ID'])
        #     print(dvinfo['SheetName'])
        #     print(dvinfo['FileName'])
        #     print("cand_file", cand_file)
        #     print("cand_sheet", cand_sheet)
        all_ += 1
        # break

    with open("res_closed_right.json",'w') as f:
        json.dump(closed_list_right,f)   
    with open("res_closed_wrong.json",'w') as f:
        json.dump(closed_list_wrong,f)  
    with open("exact_sheet.json",'w') as f:
        json.dump(list(exact_sheet),f)   
    with open("closed_sheet.json",'w') as f:
        json.dump(list(closed_sheet),f)
    with open("not_found_dvid_list.json",'w') as f:
        json.dump(list(not_found_dvid_list),f)    
    with open("not_found_sheet.json",'w') as f:
        json.dump(list(not_found_sheet),f)    

    print('exact_match', exact_match)
    print('exact_correct', exact_correct) 
    print('range_inter', range_inter) 
    print('range_inter_correct', range_inter_correct) 
    print('most_closed', most_closed) 
    print('most_closed_correct', most_closed_correct)
    print('no_file', no_file)    
    print('all_', all_) 
    print("not_found", not_found)

def check_boundary_template():
    with open("boundary_positive_pair.json",'r') as f:
        positive_pair = json.load(f)
    with open("../AnalyzeDV/data/types/boundary/boundary_list.json", 'r') as f:
        dvinfos = json.load(f)
    print(len(dvinfos))
    exact_match=0
    exact_correct = 0
    not_exact_match = []
    not_exact_correct = []
    range_inter = 0
    range_inter_correct = 0
    most_closed=0
    most_closed_correct=0
    all_=0

    inter_list_right= []
    closed_list_right = []
    inter_list_wrong= []
    closed_list_wrong = []
    sheet_pair_dict = {}
    res_closed = [] # range不一样， 但是在表上找到了最近的其他dv
    res_not_found = [] # 找到了最近的dv，但是template不一样
    res_no_file = [] # 没有找到该文件的dv
    no_file = 0

    not_found = 0
    exact_sheet = set()
    closed_sheet = set()
    not_found_sheet = set()

    not_found_dvid_list = []
    count =0

    
    for dvinfo in dvinfos:
        count += 1
        is_found=False
        is_found1=False
        is_found2=False
        # print(count, len(dvinfos))
        # count += 1  

        # ltx = dvinfo['ltx']
        # lty = dvinfo['lty']
        # rbx = dvinfo['rbx']
        # rby = dvinfo['rby']

        # if(ltx==0):
        #     continue
        for sheetname in positive_pair:
            # print(sheetname, dvinfo["SheetName"])
            if sheetname != dvinfo["SheetName"]:
                continue
            for f_pair in positive_pair[sheetname]:
                # print(f_pair)
                f1 = f_pair[0]
                f2 = f_pair[1]
                sn1 = sheetname
                sn2 = sheetname
                # print(f1, dvinfo['FileName'])
                # print(f2, dvinfo['FileName'])
                if(f1==dvinfo['FileName'] and sn1==dvinfo['SheetName']):
                    target_file = f1
                    target_sheet = sn1
                    cand_file=f2
                    cand_sheet=sn2
                    is_found1 = True
                elif(f2==dvinfo['FileName'] and sn2==dvinfo['SheetName']):
                    target_file = f2
                    target_sheet = sn2
                    cand_file=f1
                    cand_sheet=sn1
                    is_found2 = True

                if(not is_found1 or not is_found2):
                    continue
                else:
                    is_found = True
                    break
        print(count, len(dvinfos))
        if not is_found:
            not_found += 1
            all_+=1
            not_found_sheet.add(dvinfo["SheetName"])
            not_found_dvid_list.append(str(dvinfo['ID'])+"---"+str(dvinfo['batch_id']))
            continue
        print('found')

        is_exact = False
        is_inter = False
        is_most_closed = False
        closed_dvinfo=dvinfos[0]
        min_distance = 1000000
        is_found_dvinfo2 = False

        for dvinfo1 in dvinfos:
            if dvinfo1['ID']==dvinfo['ID']:
                continue
            if dvinfo1['FileName']==cand_file and dvinfo1['SheetName']==cand_sheet:
                # ltx1 = dvinfo1['ltx']
                # lty1 = dvinfo1['lty']
                # rbx1 = dvinfo1['rbx']
                # rby1 = dvinfo1['rby']
                
                dvinfo2 = dvinfo1
                is_found_dvinfo2 = True
                # if(ltx1==0):
                #     continue
                if(dvinfo['RangeAddress']==dvinfo1["RangeAddress"]):
                    is_exact=True
                    exact_match += 1
                    if(dvinfo['Value'] == dvinfo1['Value'] and dvinfo['Operator']==dvinfo1['Operator']):
                        exact_correct += 1
                    else:
                        not_exact_correct.append((dvinfo, dvinfo1))
                    # all_ += 1
                    exact_sheet.add(dvinfo["SheetName"])
                    break
                
                # elif(ltx1 >= ltx and lty1>=lty and rbx1 <= rbx and rby1<=rby) or (ltx1 <= ltx and lty1<=lty and rbx1 >= rbx and rby1>=rby):
                #     is_inter=True
                #     range_inter += 1
                    
                #     if(is_same_template(dvinfo, dvinfo1)):
                #         range_inter_correct += 1
                #         inter_list_right.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                #     else:
                #         print('error:', dvinfo['ID'], dvinfo1["ID"])
                #         inter_list_wrong.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                #     # all_ += 1
                #     break
                # else:
                #     # if ltx1 == 0:
                #     #     continue
                #     distance = (ltx-ltx1)*(ltx-ltx1)+(rbx-rbx1)*(rbx-rbx1)
                #     if min_distance > distance:
                #         min_distance = distance
                #         closed_dvinfo = dvinfo1
                #         is_most_closed = True
        
        if is_found_dvinfo2:
            if is_exact == False:
                not_exact_match.append((dvinfo, dvinfo2))
            # if ltx1 == 0:
            #     continue
            # =True
            # if is_exact == False:
            #     print("##########################")
            #     print(dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"])
            #     print(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])   
            if is_most_closed == True and is_exact==False:
                most_closed += 1
                # .append((target_file, target_sheet, cand_file, cand_sheet))
                closed_sheet.add(dvinfo["SheetName"])
                if(is_same_template(dvinfo, dvinfo2)):
                    most_closed_correct += 1
                    res_closed.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                    print(dvinfo['ID'], dvinfo2['ID'])
                    closed_list_right.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                else:
                    closed_list_wrong.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                # else:
                    # # print('error:', dvinfo['ID'], dvinfo1["ID"])
                    # print("closed##########################")
                    # print(dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"])
                    # print(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])
                    # res_not_found.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                    
                        
                
            # elif is_exact == False:
                # print("not found##########################")
                # print(dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"])
                # print(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])  
                # res_not_found.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
        else:
            no_file+=1
            not_found_sheet.add(dvinfo["SheetName"])
            # closed_list.append((target_file, target_sheet, cand_file, cand_sheet))
        #     # if not is_exact and not is_inter:
        #     #     is_most_closed=True
        #     #     most_closed +=1
        #     #     if(is_same_template(dvinfo, closed_dvinfo)):
        #     #         most_closed_correct += 1
        #     #     else:
        #     print("########################")
        #     print(dvinfo2['ID'])
        #     print(dvinfo2['SheetName'])
        #     print(dvinfo2['FileName'])
        #     print(dvinfo['ID'])
        #     print(dvinfo['SheetName'])
        #     print(dvinfo['FileName'])
        # else:
        #     print("########################")
        #     print(dvinfo['ID'])
        #     print(dvinfo['SheetName'])
        #     print(dvinfo['FileName'])
        #     print("cand_file", cand_file)
        #     print("cand_sheet", cand_sheet)
        all_ += 1
        # break

    with open("res_closed_right.json",'w') as f:
        json.dump(closed_list_right,f)   
    with open("res_closed_wrong.json",'w') as f:
        json.dump(closed_list_wrong,f)  
    with open("exact_sheet.json",'w') as f:
        json.dump(list(exact_sheet),f)
    with open("not_exact_match.json",'w') as f:
        json.dump(not_exact_match,f) 
    with open("not_exact_correct.json",'w') as f:
        json.dump(not_exact_correct,f)   
    with open("closed_sheet.json",'w') as f:
        json.dump(list(closed_sheet),f)
    with open("not_found_dvid_list.json",'w') as f:
        json.dump(list(not_found_dvid_list),f)    
    with open("not_found_sheet.json",'w') as f:
        json.dump(list(not_found_sheet),f)    

    print('exact_match', exact_match)
    print('exact_correct', exact_correct) 
    print('range_inter', range_inter) 
    print('range_inter_correct', range_inter_correct) 
    print('most_closed', most_closed) 
    print('most_closed_correct', most_closed_correct)
    print('no_file', no_file)    
    print('all_', all_) 
    print("not_found", not_found)

def check_template():
    with open("../AnalyzeDV/data/types/custom/change_xml_cutom_list.json", 'r') as f:
        dvinfos = json.load(f)
    with open("id_2_feature.json", 'r') as f:
        id_2_feature = json.load(f)
    with open("sheet_2_id.json", 'r') as f:
        sheet_2_id = json.load(f)
    pred_1 = np.load('pred_1_4.npy', allow_pickle=True)
    test_1 = np.load('test_1_4.npy', allow_pickle=True)
    with open("positive_id.json",'r') as f:
        positive_id = json.load(f)
    with open("negative_id.json",'r') as f:
        negative_id = json.load(f)

    re = pred_1==test_1
    res = []
    model = CNNnet() 
    model = torch.load('cnn_model_'+str(4))
                        # str_ = str_ + str(x1[index]) + '\n'
    exact_match=0
    exact_correct = 0
    range_inter = 0
    range_inter_correct = 0
    most_closed=0
    most_closed_correct=0
    all_=0

    count = 0

    fs_pairs = []    


    with open("positive_feature.json",'r') as f:
        positive_feature = json.load(f)
    with open("negative_feature.json",'r') as f:
        negative_feature = json.load(f)

    with open("cnn_x_4.json", 'r') as f:
        features = json.load(f)


    # features = np.load('cnn_x_4.npy', allow_pickle=True)
    pred_1 = np.load('pred_1_1.npy', allow_pickle=True)
    test_1 = np.load('test_1_1.npy', allow_pickle=True)
    pred_2 = np.load('pred_1_2.npy', allow_pickle=True)
    test_2 = np.load('test_1_2.npy', allow_pickle=True)
    pred_3 = np.load('pred_1_3.npy', allow_pickle=True)
    test_3 = np.load('test_1_3.npy', allow_pickle=True)
    pred_4 = np.load('pred_1_4.npy', allow_pickle=True)
    test_4 = np.load('test_1_4.npy', allow_pickle=True)

    re = list(pred_1==test_1)
    re += [pred_2 == test_2]
    re += [pred_3 == test_3]
    re += [pred_4 == test_4]
    # print(positive_id)
    sheet_pair_dict = {}
    res_closed = [] # range不一样， 但是在表上找到了最近的其他dv
    res_not_found = [] # 找到了最近的dv，但是template不一样
    res_no_file = [] # 没有找到该文件的dv
    no_file = 0
    for index,feature in enumerate(positive_feature):
        # if res == False:
        #     continue
        # feature = features[index]
        found=False

        for index1, one_fe in enumerate(positive_feature):
            if feature == one_fe:
                found_index=index1
                found=True
                break
        if found:
            indexing = positive_id[found_index]
        # else:
        # for index1, one_fe in enumerate(negative_feature):
        #     if feature == one_fe:
        #         found_index=index1
        #         found=True
        #         break
        # if found:
        #     indexing = negative_id[found_index]
        if not found:
            print("not found")
            continue
        # print(indexing)
        # split_index = indexing.split(':')
        id1 = indexing[0]
        id2 = indexing[1]
        fs_pair = []
        found1 = False
        found2 = False
        for sheetname in sheet_2_id:
            for filename in sheet_2_id[sheetname]:

            # print("#############################")
                
                # print(key)
                # filename, sheetname = key.split("----")
                # print("#################################")
                # print(sheet_2_id[sheetname][filename])
                # # print(type(id1))
                # print(id1)
                # print(id2)
                if id1 == sheet_2_id[sheetname][filename]:
                    # print('1111111')
                    sn1 = sheetname
                    f1 = filename
                    found1 = True
                    # fs_pair.append((f1,sn1))
                    # str_ = str_ + str(x1[index]) + '\n'
                if id2 == sheet_2_id[sheetname][filename]:
                    # print('2222222')
                    sn2 = sheetname
                    f2 = filename
                    found2 = True
                    # fs_pair.append((f2,sn2))
        if os.path.getsize(f1.replace('/UnzipData','')) != os.path.getsize(f2.replace('/UnzipData','')) and found1 and found2:
            fs_pair.append((f1,sn1))
            fs_pair.append((f2,sn2))
            fs_pairs.append(fs_pair)
        # if len(fs_pair) ==2:
        #     fs_pairs.append(fs_pair)
        #     if(sn1 != sn2):
        #         print("#############")
        #         print(sn1, sn2)



    
    print(len(fs_pairs))
    sheet_num = set()
    for i in fs_pairs:
        sheet_num.add(i[0][0] + "---"+i[0][1])
        sheet_num.add(i[1][0] + "---"+i[1][1])
    print('sheet num:',len(sheet_num))
    # print(sheet_num)
    for dvinfo in dvinfos:
        is_found=False
        is_found1=False
        is_found2=False
        # print(count, len(dvinfos))
        count += 1  

        ltx = dvinfo['ltx']
        lty = dvinfo['lty']
        rbx = dvinfo['rbx']
        rby = dvinfo['rby']

        if(ltx==0):
            continue
        for fs_pair in fs_pairs:
            f1 = fs_pair[0][0]
            sn1 = fs_pair[0][1]
            f2 = fs_pair[1][0]
            sn2 = fs_pair[1][1]
            # print(f1)
            # print(dvinfo['FileName'].replace("UnzipData/", ""))
            if(f1==dvinfo['FileName'] and sn1==dvinfo['SheetName']):
                target_file = f1
                target_sheet = sn1
                cand_file=f2
                cand_sheet=sn2
                is_found = True
            elif(f2==dvinfo['FileName'] and sn2==dvinfo['SheetName']):
                target_file = f2
                target_sheet = sn2
                cand_file=f1
                cand_sheet=sn1
                is_found = True
            # print('sn1',sn1)
            # print('f1',f1)
            # print('sn2',sn2)
            # print('f2',f2)
            if(not is_found):
                continue
            # print('found')

            is_exact = False
            is_inter = False
            is_most_closed = False
            closed_dvinfo=dvinfos[0]
            min_distance = 1000000
            is_found_dvinfo2 = False

            for dvinfo1 in dvinfos:
                if dvinfo1['FileName']==cand_file and dvinfo1['SheetName']==cand_sheet:
                    ltx1 = dvinfo1['ltx']
                    lty1 = dvinfo1['lty']
                    rbx1 = dvinfo1['rbx']
                    rby1 = dvinfo1['rby']
                    
                    dvinfo2 = dvinfo1
                    is_found_dvinfo2 = True
                    # if(ltx1==0):
                    #     continue
                    if(dvinfo['RangeAddress']==dvinfo1["RangeAddress"]):
                        is_exact=True
                        exact_match += 1
                        if(is_same_template(dvinfo, dvinfo1)):
                            exact_correct += 1
                        # all_ += 1
                        break
                    elif(ltx1 >= ltx and lty1>=lty and rbx1 <= rbx and rby1<=rby) or (ltx1 <= ltx and lty1<=lty and rbx1 >= rbx and rby1>=rby):
                        is_inter=True
                        range_inter += 1
                        if(is_same_template(dvinfo, dvinfo1)):
                            range_inter_correct += 1
                        else:
                            print('error:', dvinfo['ID'], dvinfo1["ID"])
                        # all_ += 1
                        break
                    else:
                        if ltx1 == 0:
                            continue
                        distance = (ltx-ltx1)*(ltx-ltx1)+(rbx-rbx1)*(rbx-rbx1)
                        if min_distance > distance:
                            min_distance = distance
                            closed_dvinfo = dvinfo1
                            is_most_closed = True
            
            if is_found_dvinfo2:
                # if ltx1 == 0:
                #     continue
                # =True
                # if is_exact == False:
                #     print("##########################")
                #     print(dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"])
                #     print(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])   
                if is_most_closed == True and is_exact==False:
                    most_closed += 1
                    if(is_same_template(dvinfo, dvinfo2)):
                        most_closed_correct += 1
                        res_closed.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                    # else:
                        # # print('error:', dvinfo['ID'], dvinfo1["ID"])
                        # print("closed##########################")
                        # print(dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"])
                        # print(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])
                        # res_not_found.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
                        
                          
                  
                # elif is_exact == False:
                    # print("not found##########################")
                    # print(dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"])
                    # print(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])  
                    # res_not_found.append(((dvinfo['ID'], dvinfo["SheetName"], dvinfo["FileName"]),(dvinfo2["ID"], dvinfo2["SheetName"], dvinfo2["FileName"])))
            else:
                no_file+=1
                res_no_file.append((target_file, target_sheet, cand_file, cand_sheet))
            #     # if not is_exact and not is_inter:
            #     #     is_most_closed=True
            #     #     most_closed +=1
            #     #     if(is_same_template(dvinfo, closed_dvinfo)):
            #     #         most_closed_correct += 1
            #     #     else:
            #     print("########################")
            #     print(dvinfo2['ID'])
            #     print(dvinfo2['SheetName'])
            #     print(dvinfo2['FileName'])
            #     print(dvinfo['ID'])
            #     print(dvinfo['SheetName'])
            #     print(dvinfo['FileName'])
            # else:
            #     print("########################")
            #     print(dvinfo['ID'])
            #     print(dvinfo['SheetName'])
            #     print(dvinfo['FileName'])
            #     print("cand_file", cand_file)
            #     print("cand_sheet", cand_sheet)
            all_ += 1
            break

    with open("res_closed.json",'w') as f:
        json.dump(res_closed,f)   
    with open("res_not_found.json",'w') as f:
        json.dump(res_not_found,f)   
    with open("res_no_file.json",'w') as f:
        json.dump(res_no_file,f)   
    print('exact_match', exact_match)
    print('exact_correct', exact_correct) 
    print('range_inter', range_inter) 
    print('range_inter_correct', range_inter_correct) 
    print('most_closed', most_closed) 
    print('most_closed_correct', most_closed_correct)
    print('no_file', no_file)    
    print('all_', all_) 
                            # str_ = str_ + str(sheetname) +','

def batch_100000_testing():
    with open("cnn_x_1.json", 'r') as f:
        x1 = json.load(f)
    with open("cnn_x_2.json", 'r') as f:
        x2 = json.load(f)
    with open("cnn_x_3.json", 'r') as f:
        x3 = json.load(f)
    with open("cnn_x_4.json", 'r') as f:
        x4 = json.load(f)

    with open("cnn_y_1.json", 'r') as f:
        y1 = json.load(f)
    with open("cnn_y_2.json", 'r') as f:
        y2 = json.load(f)
    with open("cnn_y_3.json", 'r') as f:
        y3 = json.load(f)
    with open("cnn_y_4.json", 'r') as f:
        y4 = json.load(f)
    features = [x1,x2,x3,x4]
    labels = [y1,y2,y3,y4]
    model = torch.load("100000_mlp_model")
    for batch in range(4):
        test_data = TensorDataset(torch.FloatTensor(features[batch]),torch.Tensor(labels[batch]))
        test_loader = DataLoader(test_data,batch_size = len(test_data), shuffle=False)
        for test_x,test_y in test_loader:
            print('len testx', len(test_x))
            test_x1 = test_x[:,0,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
            test_x2 = test_x[:,1,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
            test_x1 = Variable(test_x1) # torch.Size([batch_size, 1000, 10])
            test_x2 = Variable(test_x2) ## torch.Size([batch_size, 1000, 10])
                # y = torch.LongTensor(y)
            test_y = Variable(test_y) # torch.Size([batch_size])
            out = model(test_x1,test_x2)
            # print('test_out:\t',torch.max(out,1)[1])
            # print('test_y:\t',test_y)
            # bina_out = []
            # for i in out:
            #     bina_out.append(0 if i<0.5 else 1)
            # print(test_y)
            # print('torch.max(out,0).numpy',torch.max(out,0).numpy())
            # accuracy = bina_out== test_y.numpy()
            np.save("out_100000_"+str(batch), out.detach().numpy())
            np.save("pred_100000_"+str(batch), torch.max(out,1)[1].numpy())
            # np.save("pred_"+str(batch), out.detach().numpy())
            np.save("test_100000_"+str(batch), test_y.numpy())
            accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
            # accuracy = torch.max(out,0).numpy() == test_y.numpy()
            suc_num = len([i for i in accuracy if i==True])

def get_top_k_sim(topk):
    with open("custom_feature_embed.json", 'r') as f:
        custom_feature_embed = json.load(f)

    id_2_index = {}
    features = []

    index=0
    for dvid in custom_feature_embed:
        id_2_index[dvid]=index
        index+=1

        features.append(custom_feature_embed[dvid])  
    
    index = faiss.IndexFlatL2(len(features[0]))
    print(index.is_trained)
    
    index.add(np.array(features))
    print(index.ntotal)
    
    D, I = index.search(features, topk) # sanity check
    print("I: ",I)
    print("D: ",D)

if __name__ == "__main__":
    # create_features()
    # get_positive_feature()
    # get_negative_feature()
    # split_data(True)
    # split_data(False)
    # batch_100000_training()
    # batch_100000_testing()
    # get_custom_sheet_file_emb_from_model()
    # batch_testing()
    # middle_postive_pair()
    # get_10000_positive_feature()
    # middle_negative_pair()
    # get_10000_negative_feature()
    # look_positive_feature()
    # get_top_k_sim(4)
    # error_analyze()
    # check_template()
    # error_analyze()
    # get_custom_sheet_file_emb()
    # get_custom_sheet_file_emb_from_model()
    # cnn_evaluate(1)
    # cnn_evaluate(True)
    # look_fail()
    # get_positive_pair()
    # check_boundary_template()
    with open("not_exact_match.json",'r') as f:
        not_exact_match = json.load(f) 
    for item in not_exact_match:
        if item[0]['SheetName']=='Sheet1' or item[0]['SheetName'] == 'Input' or item[0]['SheetName'] == 'Journal':
            continue
        print("#############################")
        print(item[0]['SheetName'],item[0]['FileName'], item[1]['SheetName'],item[1]['FileName'], item[0]['RangeAddress'], item[1]['RangeAddress'])
    
    # get_most_positive_pair()
    # get_negative_pair()
    # create_custom_features()
    # get_positive_feature()
    # get_negative_feature()