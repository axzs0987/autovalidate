import json
import os
import random
def get_local_dict():
    file_sheet_local_dv = {}
    file_sheet_id_dict = {}
    with open("../PredictDV/ListDV/continous_batch_0_1.json", 'r', encoding='UTF-8') as f:
        dvinfos = json.load(f)
    for dvinfo in dvinfos:
        key = dvinfo["FileName"] + '-------------' + dvinfo["SheetName"] 
        if key not in file_sheet_local_dv:
            file_sheet_local_dv[key] = {}
            file_sheet_id_dict[key] = []
        file_sheet_id_dict[key].append(dvinfo["ID"])
        if dvinfo["Value"] not in file_sheet_local_dv[key]:
            file_sheet_local_dv[key][dvinfo["Value"]] = {}
            file_sheet_local_dv[key][dvinfo["Value"]]["content"] = []
            file_sheet_local_dv[key][dvinfo["Value"]]["dvid"] = []
            file_sheet_local_dv[key][dvinfo["Value"]]["number"] = 0
            # print(dvinfo.keys())
            file_sheet_local_dv[key][dvinfo["Value"]]["cornor"] = str(dvinfo["ltx"])+','+str(dvinfo["lty"])+','+str(dvinfo["rbx"])+','+str(dvinfo["rby"])
            for item in dvinfo["refers"]["List"]:
                file_sheet_local_dv[key][dvinfo["Value"]]["content"].append(item["Value"])
        file_sheet_local_dv[key][dvinfo["Value"]]["dvid"].append(dvinfo["ID"])
        file_sheet_local_dv[key][dvinfo["Value"]]["number"] += 1

    with open("file_sheet_local_dv.json", 'w') as f:
        json.dump(file_sheet_local_dv, f)
    with open("file_sheet_id_dict.json", 'w') as f:
        json.dump(file_sheet_id_dict, f)

def found_cand(dvid, cand_index):
    cand_ind = 0
    if os.path.exists("../PredictDV/delete_row_single_o_baseline/"+str(dvid)+"_search_result.json"):
        with open("../PredictDV/delete_row_single_o_baseline/"+str(dvid)+"_search_result.json", 'r', encoding='UTF-8') as f:
            range_cand = json.load(f)
            for range_c in range_cand:
                if cand_index==cand_ind:
                    return range_c
                cand_ind+=1
    if os.path.exists("../PredictDV/global_candidate_1/"+str(dvid)+".json"):
        with open("../PredictDV/global_candidate_1/"+str(dvid)+".json", 'r', encoding='UTF-8') as f:
            global_cand = json.load(f)
            for global_c in global_cand:
                if cand_index==cand_ind:
                    return global_c
                cand_ind+=1
    print('max_cand_index', cand_ind-1)
    return 'not found'
    
    

def get_local_pop(k):
    with open("file_sheet_local_dv.json", 'r', encoding='UTF-8') as f:
        file_sheet_local_dv = json.load(f)
    with open("file_sheet_id_dict.json", 'r', encoding='UTF-8') as f:
        file_sheet_id_dict = json.load(f)
    with open("../PredictDV/ListDV/c#_complete_both_features_10000.json", 'r', encoding='UTF-8') as f:
        features = json.load(f)
    with open("../PredictDV/ListDV/continous_batch_0.json", 'r', encoding='UTF-8') as f:
        dvinfos = json.load(f)

    random_sample_file_sheet_id_dict = {}
    for key in file_sheet_id_dict:
        random_sample_file_sheet_id_dict[key] = random.sample(file_sheet_id_dict[key], max(k, len(file_sheet_id_dict[key])))
            
    result = []
    for dvinfo in dvinfos:
        dvid = dvinfo["ID"]
        key = dvinfo["FileName"] + '-------------' + dvinfo["SheetName"] 
        for feature in features:
            if dvid == feature["dvid"]:
                print('dvid',dvid)
                
                print('type', feature['type'])
                look_dv_value = []
                look_dv_cornor = []
                same = 0
                print('cand_index', feature['cand_index'])
                candidate = found_cand(dvid, feature["cand_index"])
                print('candidate', candidate)
                for found_dvid in random_sample_file_sheet_id_dict[key]:
                    if found_dvid == dvid:
                        continue
                    for value in file_sheet_local_dv[key]:
                        if found_dvid in file_sheet_local_dv[key][value]['dvid']:
                            # print('found local refs')
                            look_dv_value.append(value)
                            look_dv_cornor.append(file_sheet_local_dv[key][value]['cornor'])
                if feature["type"] == 1:
                    for look_dv in look_dv_cornor:
                        cornor_split = look_dv.split(',')
                        
                        if str(candidate["left_top_x"]) == cornor_split[0] and str(candidate["left_top_y"]) == cornor_split[1] and str(candidate["right_bottom_x"]) == cornor_split[2] and str(candidate["right_bottom_y"]) == cornor_split[3]:
                            same += 1
                else:
                    for look_dv in look_dv_value:
                        split_list = look_dv.split(',')
                        res = []
                        for index, word in enumerate(split_list):
                            if index==0 and word[0]=='"':
                                res.append(word[1:])
                            elif index==len(split_list)-1 and word[-1] == '"':
                                res.append(word[0:-1])
                            else:
                                res.append(word)
                        res = set(res)
                        cand_content = set([str(cand) for cand in candidate["List"]])
                        if res==cand_content:
                            same+=1
           
                feature["local_pupolarity"] = same
                result.append(feature)
    with open("../PredictDV/ListDV/c#_local_pop_"+str(k)+"_complete_both_features_10000.json", 'w') as f:
        json.dump(result)
                
if __name__ == "__main__":
    # get_local_dict()
    get_local_pop(1)