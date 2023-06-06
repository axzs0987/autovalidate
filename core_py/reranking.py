import torch
import json
import os
import numpy as np
from cnn_fine_tune import generate_demo_features
from cnn_fine_tune import generate_one_before_feaure
from cnn_fine_tune import generate_one_after_feature
from cnn_fine_tune import euclidean
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from rerank_model import RerankModel, RerankLinearModel, RerankFinegrainModel, RerankFinegrainModelUVD

root_path = '/datadrive-2/data/top10domain_test/'
with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
    bert_dict = json.load(f)

with open("json_data/content_temp_dict_1.json", 'r') as f:
    content_tem_dict = json.load(f)

def generate_training_data(formula_token, found_formula_token, tile_path, before_path, after_path, first_save_path, second_save_path, cross=False, mask=2):
    top_k = 1 
    # model = torch.load("/datadrive-2/data/training_ref_model/epoch_20")
    model = torch.load("/datadrive-2/data/finegrained_model_16/epoch_31")
    filesheet = formula_token.split('---')[0] + '---' + formula_token.split('---')[1]
    sorted_first_block = {}
    sorted_second_block = {}
    origin_file = formula_token.split('---')[0]
    origin_sheet = formula_token.split('---')[1]
    origin_filesheet = origin_file + '---' + origin_sheet

    found_file = found_formula_token.split('---')[0]
    found_filesheet = found_formula_token.split('---')[0] + '---' +found_formula_token.split('---')[1]

    if os.path.exists("../Demo/fixed_workbook_json/" + origin_file + '.json'):
        with open("../Demo/fixed_workbook_json/" + origin_file + '.json', 'r') as f:
            origin_wbjson = json.load(f)
    # else:
    #     continue
    print('found_file', found_file)
    if os.path.exists("../Demo/fixed_workbook_json/" + found_file + '.json'):
        with open("../Demo/fixed_workbook_json/" + found_file + '.json', 'r') as f:
            found_wbjson = json.load(f)
    # else:
        # continue
    # origin_wbjson = filename2wbjson[origin_file]
    # found_wbjson = filename2wbjson[found_file]
    if not os.path.exists(root_path + 'test_refcell_position/' + found_formula_token + '.json'):
        return
    with open(root_path + 'test_refcell_position/'+found_formula_token + '.json' , 'r') as f:
        test_refcell_position = json.load(f)

    if not os.path.exists(root_path + 'tile_rows/' + origin_filesheet + '.json'):
        return
    if not os.path.exists(root_path + 'tile_cols/' + origin_filesheet + '.json'):
        return
    with open(root_path + 'tile_rows/' + origin_filesheet + '.json', 'r') as f:
        tile_rows = json.load(f)
    with open(root_path + 'tile_cols/' + origin_filesheet + '.json', 'r') as f:
        tile_cols = json.load(f)

    for refcell_item in test_refcell_position:
        if 'R' not in refcell_item or 'C' not in refcell_item:
            continue
        ref_row = refcell_item['R']
        ref_col = refcell_item['C']
        # ref_row, ref_col = ref_cell_rc.split('---') 
        ref_row = int(ref_row)
        ref_col = int(ref_col)
        
        ######## first level
            ########## check 4 nearest tile
        refcell_tile_row = int(ref_row/100)*100 + 1
        refcell_tile_col = int(ref_col/10)*10 + 1

        is_left = False
        is_up = False
        if ref_row - refcell_tile_row < refcell_tile_row + 100 - ref_row: # up
            is_up = True
        if ref_col - refcell_tile_col < refcell_tile_col + 10 - ref_col: # left
            is_left = True

        closed_four_tiles = []
        closed_four_tiles.append((refcell_tile_row, refcell_tile_col)) # first
        if is_up:
            if refcell_tile_row >= 101:
                closed_four_tiles.append((refcell_tile_row - 100, refcell_tile_col))  # second add up
                if is_left:
                    if refcell_tile_col >= 11:
                        closed_four_tiles.append((refcell_tile_row - 100, refcell_tile_col-10))  # third add left, up
                else:
                    if refcell_tile_col + 10 in tile_cols:
                        closed_four_tiles.append((refcell_tile_row - 100, refcell_tile_col+10))  # third add right, up
        else:
            if refcell_tile_col + 100 in tile_rows:
                closed_four_tiles.append((refcell_tile_row + 100, refcell_tile_col))  #0 second add down
            if is_left:
                if refcell_tile_col >= 11:
                    closed_four_tiles.append((refcell_tile_row + 100, refcell_tile_col-10))  # third add left, down
            else:
                if refcell_tile_col + 10 in tile_cols: 
                    closed_four_tiles.append((refcell_tile_row + 100, refcell_tile_col+10))  # third add right, down

        if is_left:
            if refcell_tile_col >= 11:
                closed_four_tiles.append((refcell_tile_row, refcell_tile_col-10))  # forth add left
        else:
            if refcell_tile_col + 10 in tile_cols: 
                closed_four_tiles.append((refcell_tile_row, refcell_tile_col+10))  # forth add right

            ##############find closed tile on original sheet of 4 nearest tile on found sheet
        sorted_first_block[str(ref_row) + '---' + str(ref_col)] = {}
        print('tile_rows', tile_rows)
        print('tile_cols', tile_cols)
        print('closed_four_tiles', closed_four_tiles)
        for one_found_tile in closed_four_tiles:
            one_found_row = one_found_tile[0]
            one_found_col = one_found_tile[1]
            token_npy = found_filesheet + '---' + str(one_found_row) + '---' + str(one_found_col) + '.npy'
            if not os.path.exists(tile_path + token_npy.replace('.npy', '.json')):
                # print("generate: 2882")
                generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], found_wbjson, one_found_row, one_found_col, tile_path, is_look=True, cross=cross)
            if not os.path.exists(before_path + token_npy):
                # print("generate: 2885")
                if cross:
                    res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                else:
                    res = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                if res == "invalid json features":
                    generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], found_wbjson, one_found_row, one_found_col, tile_path, is_look=True, cross=cross)
                    if cross:
                        res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                    else:
                        generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
            if not os.path.exists(after_path + token_npy):
                # print("generate: 2888")
                generate_one_after_feature(token_npy.replace('.npy', ''), model, before_path + token_npy, after_path + token_npy)     
            feature = np.load(after_path + token_npy, allow_pickle=True)
            sorted_first_block[str(ref_row) + '---' + str(ref_col)][str(one_found_row) +'---' +str(one_found_col)] = {}
            for row in tile_rows:
                for col in tile_cols:
                    token_npy =  origin_filesheet + '---' + str(row) + '---' + str(col) + '.npy'
                    if not os.path.exists(tile_path  + token_npy.replace('.npy', '.json')):
                        # print("generate: 2896")
                        generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], origin_wbjson, row, col, tile_path, is_look=True, cross=cross)
                    if not os.path.exists(before_path + token_npy):
                        # print("generate: 2899")
                        if cross:
                            res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                        else:
                            res = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                        if res == "invalid json features":
                            generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], origin_wbjson, row, col, tile_path, is_look=True, cross=cross)
                            if cross:
                                res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                            else:
                                generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                    if not os.path.exists(after_path + token_npy):
                        # print("generate: 2902")
                        generate_one_after_feature(token_npy.replace('.npy', ''), model, before_path + token_npy, after_path + token_npy)
                    other_feature = np.load(after_path + token_npy, allow_pickle=True)
                    distance = euclidean(feature, other_feature)
                    sorted_first_block[str(ref_row) + '---' + str(ref_col)][str(one_found_row) +'---' +str(one_found_col)][str(row) + '---'  + str(col)] = distance
        first_end_time = time.time()
        # sorted_first_block['time'] = first_end_time - start_time
        np.save(first_save_path + formula_token  +'.npy', sorted_first_block)
        ######### second level
        token_npy = found_filesheet + '---' + str(ref_row) + '---' + str(ref_col) + '.npy'
        if not os.path.exists(tile_path  + token_npy.replace('.npy', '.json')):
            # print("generate: 2913")
            generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], found_wbjson, ref_row, ref_col, tile_path, is_look=True, cross=cross)
        if not os.path.exists(before_path + token_npy):
            # print("generate: 2916")
            if cross:
                res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
            else:
                res = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
            if res == "invalid json features":
                generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], found_wbjson, ref_row, ref_col, tile_path, is_look=True, cross=cross)
                if cross:
                    res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                else:
                    generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
        if not os.path.exists(after_path + token_npy):
            # print("generate: 2919")
            generate_one_after_feature(token_npy.replace('.npy', ''), model, before_path + token_npy, after_path + token_npy)
            
        feature = np.load(after_path + token_npy, allow_pickle=True)

        if str(ref_row) + '---' + str(ref_col) not in sorted_first_block:
            # print('not in first', str(ref_row) + '---' + str(ref_col))
            continue
        best_row_col_list = []
        for first_row_col in sorted_first_block[str(ref_row) + '---' + str(ref_col)]:
            # print('first_row_col', first_row_col)
            distance_dict = sorted_first_block[str(ref_row) + '---' + str(ref_col)][first_row_col]
            # print('distance_dict', distance_dict)
            sorted_list = sorted(distance_dict.items(), key=lambda x: x[1])
            # print('sorted_list', [list(i)[0] for i in sorted_list[0:top_k]])
            best_row_col_list += [list(i)[0] for i in sorted_list[0:top_k]]
        best_row_col_list = list(set(best_row_col_list))
        print('best_row_col_list', best_row_col_list)
        sorted_second_block[str(ref_row) + '---' + str(ref_col)] = {}
        for best_row_col in best_row_col_list:
            best_row = int(best_row_col.split('---')[0])
            best_col = int(best_row_col.split('---')[1])
            
            for row in range(best_row, best_row + 100):
                for col in range(best_col, best_col + 10):
                    # if(best_row_col == '1---11'):
                    #     print('    ', row, col)
                    token_npy = origin_filesheet + '---' + str(row) + '---' + str(col) + '.npy'
                    if not os.path.exists(tile_path + token_npy.replace('.npy', '.json')):
                        # print("generate: 2947")
                        generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], origin_wbjson, row, col, tile_path, is_look=True, cross=cross)
                    # print('before_path + token_npy', before_path + token_npy)
                    if not os.path.exists(before_path + token_npy):
                        # print("generate: 2950")
                        if cross:
                            # print('before2cross')
                            res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                        else:
                            # print('before')
                            res = generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                        if res == "invalid json features":
                            generate_demo_features(token_npy.split('---')[0], token_npy.split('---')[1], origin_wbjson, row, col, tile_path, is_look=True, cross=cross)
                            if cross:
                                res = before2cross(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = cross_path, saved_root_path=before_path)
                            else:
                                generate_one_before_feaure(token_npy.replace('.npy', ''), bert_dict, content_tem_dict, mask, source_root_path = tile_path, saved_root_path=before_path)
                    if not os.path.exists(after_path + token_npy):
                        # print("generate: 2953")
                        generate_one_after_feature(token_npy.replace('.npy', ''), model, before_path + token_npy, after_path + token_npy)
                    other_feature = np.load(after_path + token_npy, allow_pickle=True)
                    distance = euclidean(feature, other_feature) 
                    sorted_second_block[str(ref_row) + '---' + str(ref_col)][str(row) + '---'  + str(col)] = distance
            end_time = time.time()
            # sorted_second_block['time'] = end_time - start_time
            # print('sorted_second_block',str(ref_row) + '---' + str(ref_col),sorted_second_block[str(ref_row) + '---' + str(ref_col)].keys())
    print('saving:',second_save_path + formula_token  +'.npy')
    np.save(second_save_path + formula_token  +'.npy', sorted_second_block)

def batch_generate_testfinetune():
    need_save_list = set()
    with open("fine_tune_positive.json", 'r') as f:
        dedup_positive_pair = json.load(f)

    for index, item in enumerate(dedup_positive_pair):
        print(index, len(dedup_positive_pair))
        # print(item[0])
        # print(item[1])
        # item = item[0]
        formula_token = item[0][0].split('/')[-1]

        found_formula_token = item[1][0].split('/')[-1]
        # formula_token = item[0][0]

        # found_formula_token = item[1][0]
        # need_save_list.add(formula_token)
        # need_save_list.add(found_formula_token)
        if os.path.exists(root_path + "finegrain_second_save_path/" + formula_token  +'.npy'):
            continue
        generate_training_data(formula_token, found_formula_token, tile_path=root_path + "training_demo_tile_features/", before_path=root_path + "training_before_features/", after_path=root_path + "finegrain_training_after_features", first_save_path = root_path + "finegrain_first_save_path/", second_save_path=root_path + "finegrain_second_save_path/", mask=2)
        # break
    # need_save_list = list(need_save_list)
    # with open("training_ref_formulatokens.json", 'w') as f:
    #     json.dump(need_save_list, f)

def generate_training_cases():
    training_res = []
    with open("fine_tune_positive.json", 'r') as f:
        dedup_positive_pair = json.load(f)
    for index, item in enumerate(dedup_positive_pair):
        print(index, len(dedup_positive_pair))
        # print(item)
        auchor_item = item[0]
        positive_item = item[1]
        auchor_formula_token = auchor_item[0].split('/')[-1]
        positive_formula_token = positive_item[0].split('/')[-1]
        # print('auchor_formula_token', auchor_formula_token)
        # print('positive_formula_token', positive_formula_token)
        if not os.path.exists(root_path + "finegrain_second_save_path/"+auchor_formula_token + '.npy'):
            continue
        res = np.load(root_path + "finegrain_second_save_path/"+auchor_formula_token + '.npy', allow_pickle=True).item()
        # print(res)
        found_range_path = root_path+'test_refcell_position/' + positive_formula_token + '.json'
        found_filesheet=  positive_formula_token.split('---')[0] + '---' + positive_formula_token.split('---')[1]
        gt_filesheet=  auchor_formula_token.split('---')[0] + '---' + auchor_formula_token.split('---')[1]
        gt_range_path = root_path+'test_refcell_position/' + auchor_formula_token + '.json'
        if not os.path.exists(found_range_path) or not os.path.exists(gt_range_path):
            continue
        with open(found_range_path, 'r') as f:
            found_ref_pos = json.load(f)
        with open(gt_range_path, 'r') as f:
            gt_ref_pos = json.load(f)
        # print('found_ref_pos', found_ref_pos)
        # print('gt_ref_pos', gt_ref_pos)
        ref_rc_list = list(set(res.keys()))
        # print('ref_rc_list', ref_rc_list)
        if len(gt_ref_pos) != len(found_ref_pos):
            continue
        for ref_item in ref_rc_list:
            # print('ref_item', ref_item)
            # print("found_ref_pos key", found_ref_pos)
            found = False
            for index, found_item in enumerate(found_ref_pos):
                # print('found_item', found_item)
                if 'R' not in found_item or 'C' not in found_item:
                    continue
                found_r = found_item['R']
                found_c = found_item['C']
                found_rc = str(found_r) + '---' + str(found_c)
                if found_rc == ref_item:
                    gt_item = gt_ref_pos[index]
                    gt_r = gt_item['R']
                    gt_c = gt_item['C']
                    found = True
                    break
            if not found:
                continue
            gt_rc = str(gt_r) + '---' + str(gt_c)
            pos_res = res[found_rc]
            # print(pos_res[gt_rc])
            pos_res = sorted(pos_res.items(), key=lambda x:x[1])[0:10]
            # print(pos_res)
            training_res.append([found_filesheet + '---' + found_rc, gt_filesheet + '---' + gt_rc, 1])
            for neg_item in pos_res:
                if neg_item[0] != gt_rc:
                    training_res.append([found_filesheet + '---' + found_rc, gt_filesheet + '---' + neg_item[0], 0])
            # print('training_res', training_res)
        # break 
    with open("finegrain_training_reranking_infos.json", 'w') as f:
        json.dump(training_res, f)

def generate_x_y():
    with open("finegrain_training_reranking_infos.json", 'r') as f:
        training_reranking_infos = json.load(f)
    model_path = '/datadrive-2/data/finegrained_model_16/epoch_31'
    finegrain_model = torch.load(model_path)
    new_training_reranking_infos = []
    # add_set = set()
    post_dict = {}
    for index, item in enumerate(training_reranking_infos):
        token1, token2, label = item
        if label == 1:
            # new_training_reranking_infos.append(item)
            post_dict[token1] = item
        else:
            # if token1 in add_set:
            #     continue
            # else:
            new_training_reranking_infos.append(post_dict[token1])
            new_training_reranking_infos.append(item)
            # add_set.add(token1)
    training_x = []
    training_y = []
    id_ = 1

    all_pos = 0
    all_pos_same = 0
    for index, item in enumerate(new_training_reranking_infos):
        print(index, len(new_training_reranking_infos))
        print(item)
        token1, token2, label = item
        if not os.path.exists(root_path + "training_before_features/"+token1 +'.npy') or not os.path.exists(root_path + "training_before_features/"+token2 +'.npy'):
            continue
        
        feature1 = np.load(root_path + "training_before_features/"+token1 +'.npy', allow_pickle=True)
        feature1 = feature1.reshape(1,100,10,399)
        finegrain_model.eval()
        feature1 = torch.DoubleTensor(feature1)
        feature1 = Variable(feature1).to(torch.float32)
        feature1 = finegrain_model(feature1).detach().numpy()

        feature2 = np.load(root_path + "training_before_features/"+token2 +'.npy', allow_pickle=True)
        feature2 = feature2.reshape(1,100,10,399)
        finegrain_model.eval()
        feature2 = torch.DoubleTensor(feature2)
        feature2 = Variable(feature2).to(torch.float32)
        feature2 = finegrain_model(feature2).detach().numpy()
        
        # delta_feature = feature1 - feature2
        # print('delta_feature', delta_feature)
        # print('allsame', (feature1 == feature2).all())
        if label == 1:
            all_pos += 1
            if (feature1 == feature2).all():
                all_pos_same += 1
        # if label == 1:
        #     label_onehot = [1,0]
        # else:
        #     label_onehot = [0,1]
        # if len(training_x) == 10000:
        #     # np.save("training_reranking_x_"+str(id_)+".npy", training_x)
        #     # np.save("training_reranking_y_"+str(id_)+".npy", training_y)
        #     np.save("training_reranking_finegrain_x_"+str(id_)+".npy", training_x)
        #     np.save("training_reranking_finegrain_y_"+str(id_)+".npy", training_y)
        #     training_x = []
        #     training_y = []
        #     id_ += 1
        training_x.append([feature1, feature2, feature1 - feature2])
        training_y.append(label)
        # break
    np.save("training_reranking_finegrain_x_uvd.npy", training_x)
    np.save("training_reranking_finegrain_y_uvd.npy", training_y)
    # np.save("training_reranking_x_"+str(id_)+".npy", training_x)
    # np.save("training_reranking_y_"+str(id_)+".npy", training_y)
    # np.save("training_reranking_new_x.npy", training_x)
    # np.save("training_reranking_new_y.npy", training_y)
    print('all_pos', all_pos)
    print('all_pos_same', all_pos_same)

def training():
   
    model_name = 'RerankLinearModel'
    batch_size = 32
    # if model_name == 'RerankModel':
    #     rerank_model = RerankModel()
    #     saving_model_path = "/datadrive-2/data/reranking_models/"
    # elif model_name == "RerankLinearModel":
    #     rerank_model = RerankLinearModel()
    #     saving_model_path = "/datadrive-2/data/reranking_linear_models/"
    # rerank_model = RerankFinegrainModelUVD()
    
    saving_model_path = "/datadrive-2/data/reranking_finegrain_models_uvd/"
    rerank_model = torch.load(saving_model_path + "epoch_2")
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(rerank_model.parameters(),lr=0.00001)
    logs = {}
    train_data_x = np.load("training_reranking_finegrain_x_uvd.npy", allow_pickle=True)
    train_data_y = np.load("training_reranking_finegrain_y_uvd.npy", allow_pickle=True)
    train_data_x = torch.from_numpy(train_data_x)
    train_data_y = torch.from_numpy(train_data_y)
    print(train_data_x.shape)
    for epoch in range(3, 4):
        logs[epoch] = {}
        # for id_ in range(1,8):
        
        # print("train_data_x", train_data_x.shape)
        # print("train_data_y", train_data_y.shape)
        train_data = TensorDataset(train_data_x, train_data_y)
        train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
        for i, (x, y) in enumerate(train_loader):
            # x = x[0]
            # print(x.reshape(-1,3,16000))
            # print('x.shape', x.shape)
            x = Variable(x) # torch.Size([batch_size, 1000, 10])
            output = rerank_model(x) # torch.Size([128,10])
            loss = loss_func(output,y)
            # # print('    output:', output, y)
            # print('    loss:', loss)
            logs[epoch][i] = loss
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上
        print('saving epoch_'+str(epoch))
        torch.save(rerank_model, saving_model_path + 'epoch_'+str(epoch))
        np.save(saving_model_path+'losslog.npy', logs)
        # train_data_x = np.load("training_reranking_new_x.npy", allow_pickle=True)
        # train_data_y = np.load("training_reranking_new_y.npy", allow_pickle=True)
        # train_data_x = torch.from_numpy(train_data_x)
        # train_data_y = torch.from_numpy(train_data_y)
        # # print("train_data_x", train_data_x.shape)
        # # print("train_data_y", train_data_y.shape)
        # train_data = TensorDataset(train_data_x, train_data_y)
        # train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
        # for i,(x, y) in enumerate(train_loader):
        #     # x = x[0]
        #     # print('x.shape', x.shape)
        #     feature = x.reshape(len(x),100,10,399)
            
        #     feature = Variable(feature) # torch.Size([batch_size, 1000, 10])
        #     output = rerank_model(feature.to(torch.float32)) # torch.Size([128,10])
        #     loss = loss_func(output,y)
        #     print('    loss:', loss)
        #     logs[epoch][i] = loss
        #     opt.zero_grad()  # 清空上一步残余更新参数值
        #     loss.backward() # 误差反向传播，计算参数更新值
        #     opt.step() # 将参数更新值施加到net的parmeters上
        # print('saving reranking_models/epoch_'+str(epoch))
        # torch.save(rerank_model, '/datadrive-2/data/reranking_new_models/epoch_'+str(epoch))
        # np.save('/datadrive-2/data/reranking_new_models/losslog.npy', logs)
        
# generate_training_cases()
# batch_generate_testfinetune()
# generate_x_y()
training()