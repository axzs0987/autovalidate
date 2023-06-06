from list_nn import PairNN
import torch
import random
import json
import numpy as np
import os

with open("continous_batch_0.json", 'r', encoding='UTF-8') as f:
    dvinfos = json.load(f)
    
class PairTrainer:
    def __init__(self, evaluate_range, evaluate_global, precision, batch_id, episodes_num=10, learning_rate=1e-3, batch_size=500):
        self.evaluate_range = evaluate_range
        self.evaluate_global = evaluate_global
        self.precision = precision
        self.episodes_num = episodes_num
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.res_path = "with_header_sname/nn_pair_range_"+str(self.evaluate_range).replace('-','n') + "_global_" + str(self.evaluate_global)
        self.root_path = "with_header_sname"
        self.label_path = "without_header" + "/" + "range_"+str(self.evaluate_range).replace("-","n").replace('-','n')+"_global_"+str(self.evaluate_global) + '/label'
        self.batch_id = batch_id
        if not os.path.exists(self.res_path):
            os.mkdir(self.res_path)
        

    def change_both_feature(self, is_range, feature):
        if is_range == True:
            range_bit = np.array([1])
            global_bit = np.array([0])
            global_feature = np.zeros(9)
                
            result = np.concatenate((range_bit, np.array(feature), global_bit, global_feature))
        else:
            range_bit = np.array([0])
            global_bit = np.array([1])
            range_feature = np.zeros(8)
            result = np.concatenate((range_bit, range_feature, global_bit, np.array(feature)))
        return list(result)

    def change_both_label(self, label):
        if label == 1:
            return np.array([0,1])
        else:
            return np.array([1,0])

    def load_train_data(self, batch_id):
        self.global_list = []
        self.range_list = []
       
        range_feature_dict_1 = np.load(
            self.root_path+"/range_feature_dict_1.npy", allow_pickle=True).item()
        range_feature_dict_2 = np.load(
            self.root_path+"/range_feature_dict_2.npy", allow_pickle=True).item()
        range_feature_dict_3 = np.load(
            self.root_path+"/range_feature_dict_3.npy", allow_pickle=True).item()
        range_feature_dict_4 = np.load(
            self.root_path+"/range_feature_dict_4.npy", allow_pickle=True).item()

        global_feature_dict_1 = np.load(
            self.root_path+"/global_feature_dict_1.npy", allow_pickle=True).item()
        global_feature_dict_2 = np.load(
            self.root_path+"/global_feature_dict_2.npy", allow_pickle=True).item()
        global_feature_dict_3 = np.load(
            self.root_path+"/global_feature_dict_3.npy", allow_pickle=True).item()
        global_feature_dict_4 = np.load(
            self.root_path+"/global_feature_dict_4.npy", allow_pickle=True).item()

        train_range_label_dict_1 = np.load(
            "without_header/range_0_global_0/label/range_label_dict_range_0_global_0_1.npy", allow_pickle=True).item()
        train_range_label_dict_2 = np.load(
            "without_header/range_0_global_0/label/range_label_dict_range_0_global_0_2.npy", allow_pickle=True).item()
        train_range_label_dict_3 = np.load(
            "without_header/range_0_global_0/label/range_label_dict_range_0_global_0_3.npy", allow_pickle=True).item()
        train_range_label_dict_4 = np.load(
            "without_header/range_0_global_0/label/range_label_dict_range_0_global_0_4.npy", allow_pickle=True).item()

        train_global_label_dict_1 = np.load(
            "without_header/range_0_global_0/label/global_label_dict_range_0_global_0_1.npy", allow_pickle=True).item()
        train_global_label_dict_2 = np.load(
            "without_header/range_0_global_0/label/global_label_dict_range_0_global_0_2.npy", allow_pickle=True).item()
        train_global_label_dict_3 = np.load(
            "without_header/range_0_global_0/label/global_label_dict_range_0_global_0_3.npy", allow_pickle=True).item()
        train_global_label_dict_4 = np.load(
            "without_header/range_0_global_0/label/global_label_dict_range_0_global_0_4.npy", allow_pickle=True).item()

        self.pair_train_feature_dict_1 = {}
        self.pair_train_feature_dict_2 = {}
        self.pair_train_feature_dict_3 = {}
        self.pair_train_feature_dict_4 = {}

        self.pair_train_label_dict_1 = {}
        self.pair_train_label_dict_2 = {}
        self.pair_train_label_dict_3 = {}
        self.pair_train_label_dict_4 = {}

        self.train_findex_2_cindex_dict = {}

        feature_dict_1 = {}
        feature_dict_2 = {}
        feature_dict_3 = {}
        feature_dict_4 = {}
        label_dict_1 = {}
        label_dict_2 = {}
        label_dict_3 = {}
        label_dict_4 = {}

        train_feature_dict_list = [feature_dict_1,feature_dict_2,feature_dict_3,feature_dict_4]
        train_label_dict_list = [label_dict_1,label_dict_2,label_dict_3,label_dict_4]
        train_range_feature_dict_list = [range_feature_dict_1, range_feature_dict_2, range_feature_dict_3, range_feature_dict_4, global_feature_dict_1, global_feature_dict_2, global_feature_dict_3, global_feature_dict_4]
        train_range_label_dict_list = [train_range_label_dict_1, train_range_label_dict_2, train_range_label_dict_3, train_range_label_dict_4, train_global_label_dict_1, train_global_label_dict_2, train_global_label_dict_3, train_global_label_dict_4]
        self.pair_train_feature_dict_list = [self.pair_train_feature_dict_1, self.pair_train_feature_dict_2, self.pair_train_feature_dict_3,self.pair_train_feature_dict_4]
        self.pair_train_label_dict_list = [self.pair_train_label_dict_1,self.pair_train_label_dict_2, self.pair_train_label_dict_3, self.pair_train_label_dict_4]

        for index, label_dict in enumerate(train_label_dict_list):
            feature_dict = train_feature_dict_list[index]
            range_feature_dict = train_range_feature_dict_list[index]
            global_feature_dict = train_range_feature_dict_list[index+4]
            range_label_dict = train_range_label_dict_list[index]
            global_label_dict = train_range_label_dict_list[index+4]

            for dvid in list(set(range_feature_dict.keys()) | set(global_feature_dict.keys())):
                feature_dict[dvid] = {}
                label_dict[dvid] = {}
                if dvid in range_feature_dict and dvid in range_label_dict:
                    for cand_index in range_feature_dict[dvid]:
                        if cand_index in range_label_dict[dvid]:
                            new_feature = self.change_both_feature(True, range_feature_dict[dvid][cand_index])
                            feature_dict[dvid][cand_index] = new_feature
                            label_dict[dvid][cand_index] = range_label_dict[dvid][cand_index]
                if dvid in global_feature_dict and dvid in global_label_dict:
                    for cand_index in global_feature_dict[dvid]:
                        if cand_index in global_label_dict[dvid]:
                            new_feature = self.change_both_feature(False, global_feature_dict[dvid][cand_index])
                            feature_dict[dvid][cand_index] = new_feature
                            label_dict[dvid][cand_index] = global_label_dict[dvid][cand_index]
                    

        for index, train_label_dict in enumerate(train_label_dict_list):
            self.pair_train_feature_dict = self.pair_train_feature_dict_list[index]
            train_feature_dict = train_feature_dict_list[index]
            self.pair_train_label_dict = self.pair_train_label_dict_list[index]
            for dvid in train_label_dict:
                has_positive_label = False
                positive_cand_index = 0
                findex = 0
                for cand_index in train_label_dict[dvid]:
                    if train_label_dict[dvid][cand_index] == 1:
                        has_positive_label = True
                        positive_cand_index = cand_index
                        break
                # print(train_label_dict[dvid])
                if has_positive_label == False:
                    continue
                # print("has positive")
                self.pair_train_feature_dict[dvid] = {}
                self.train_findex_2_cindex_dict[dvid] = {}
                self.pair_train_label_dict[dvid] = {}
                # print(train_label_dict[dvid])
                for cand_index in train_label_dict[dvid]:
                    if cand_index == positive_cand_index:
                        continue
                    # print("train_feature_dict[dvid][positive_cand_IN]:", train_feature_dict[dvid][positive_cand_index])
                    # print("train_feature_dict[dvid][cand_index][char]:", train_feature_dict[dvid][cand_index])
                    # print("len(train_feature_dict[dvid][positive_cand_index])",len(train_feature_dict[dvid][positive_cand_index]))
                    # print("len(train_feature_dict[dvid][cand_index])",len(train_feature_dict[dvid][cand_index]))
                    positive_feature = [train_feature_dict[dvid][positive_cand_index][char] - train_feature_dict[dvid][cand_index][char] for char in range(0,len(train_feature_dict[dvid][positive_cand_index]))]
                    # print("positive_feature", positive_feature)
                    # positive_feature = train_feature_dict[dvid][positive_cand_index] - train_feature_dict[dvid][cand_index]
                    positive_label = 1
                    self.pair_train_feature_dict[dvid][findex] = positive_feature    
                    self.pair_train_label_dict[dvid][findex] = positive_label
                    self.train_findex_2_cindex_dict[dvid][findex] = (positive_cand_index, cand_index)
                
                    findex += 1
                    negative_feature = [train_feature_dict[dvid][cand_index][char] - train_feature_dict[dvid][positive_cand_index][char] for char in range(0,len(train_feature_dict[dvid][positive_cand_index]))]
                    # negative_feature = train_feature_dict[dvid][cand_index] - train_feature_dict[dvid][positive_cand_index]
                    negative_label = 0
                    self.pair_train_feature_dict[dvid][findex] = negative_feature
                    self.pair_train_label_dict[dvid][findex] = negative_label
                    self.train_findex_2_cindex_dict[dvid][findex] = (cand_index, positive_cand_index)
                    # print("dvid:", dvid, ", findex:", findex, ", pfeature:", positive_feature, ', nfeature:', negative_feature)
                    findex += 1
        for dvid in list(set(train_range_label_dict_1) | set(train_range_label_dict_2) | set(train_range_label_dict_3) | set(train_range_label_dict_4) | set(train_global_label_dict_1) | set(train_global_label_dict_2) | set(train_global_label_dict_3) | set(train_global_label_dict_4)):
            for k in dvinfos:
                if k["ID"] == dvid:
                    if "," in k["Value"]:
                        self.global_list.append(dvid)
                    else:
                        self.range_list.append(dvid)

    def load_test_data(self,batch_id):
        self.global_list = []
        self.range_list = []
        range_feature_dict_1 = np.load(
            self.root_path+"/range_feature_dict_1.npy", allow_pickle=True).item()
        range_feature_dict_2 = np.load(
            self.root_path+"/range_feature_dict_2.npy", allow_pickle=True).item()
        range_feature_dict_3 = np.load(
            self.root_path+"/range_feature_dict_3.npy", allow_pickle=True).item()
        range_feature_dict_4 = np.load(
            self.root_path+"/range_feature_dict_4.npy", allow_pickle=True).item()

        global_feature_dict_1 = np.load(
            self.root_path+"/global_feature_dict_1.npy", allow_pickle=True).item()
        global_feature_dict_2 = np.load(
            self.root_path+"/global_feature_dict_2.npy", allow_pickle=True).item()
        global_feature_dict_3 = np.load(
            self.root_path+"/global_feature_dict_3.npy", allow_pickle=True).item()
        global_feature_dict_4 = np.load(
            self.root_path+"/global_feature_dict_4.npy", allow_pickle=True).item()
        test_range_feature_dict_list = [range_feature_dict_1, range_feature_dict_2, range_feature_dict_3, range_feature_dict_4, global_feature_dict_1, global_feature_dict_2, global_feature_dict_3, global_feature_dict_4]
        if batch_id == 1:
            range_feature_dict = test_range_feature_dict_list[3]
            global_feature_dict = test_range_feature_dict_list[3+4]
        elif batch_id == 2:
            range_feature_dict = test_range_feature_dict_list[2]
            global_feature_dict = test_range_feature_dict_list[2+4]
        elif batch_id == 3:
            range_feature_dict = test_range_feature_dict_list[1]
            global_feature_dict = test_range_feature_dict_list[1+4]
        else:
            self.pair_test_feature_dict_1 = {}
            range_feature_dict = test_range_feature_dict_list[0]
            global_feature_dict = test_range_feature_dict_list[0+4]
        
        
        print("start get test feature dict....")
        self.test_findex_2_cindex_dict = {}

        test_feature_dict = {}
        for dvid in list(set(range_feature_dict.keys()) | set(global_feature_dict.keys())):
            test_feature_dict[dvid] = {}
            if dvid in range_feature_dict:
                for cand_index in range_feature_dict[dvid]:
                    
                    new_feature = self.change_both_feature(True, range_feature_dict[dvid][cand_index])
                    test_feature_dict[dvid][cand_index] = new_feature
            if dvid in global_feature_dict:
                for cand_index in global_feature_dict[dvid]:
                    new_feature = self.change_both_feature(False, global_feature_dict[dvid][cand_index])
                    test_feature_dict[dvid][cand_index] = new_feature
    

        print("start get pair test feature dict....")
        self.pair_test_feature_dict = {}
        for dvid in test_feature_dict:
            findex = 0
            print(dvid)
            if dvid == 9632:
                print('continue')
                continue
            self.pair_test_feature_dict[dvid] = {}
        
            self.test_findex_2_cindex_dict[dvid] = {}
            for cand_index1 in test_feature_dict[dvid]:
                for cand_index2 in test_feature_dict[dvid]:
                    if cand_index1 >= cand_index2:
                        continue
                    feature_1_2 = [test_feature_dict[dvid][cand_index1][char] - test_feature_dict[dvid][cand_index2][char] for char in range(0,len(test_feature_dict[dvid][cand_index2]))]
                    # feature_1_2 = test_feature_dict[dvid][cand_index1] - test_feature_dict[dvid][cand_index2]
                    self.pair_test_feature_dict[dvid][findex] = feature_1_2
                    self.test_findex_2_cindex_dict[dvid][findex] = (cand_index1, cand_index2)
                    findex += 1

                    feature_2_1 = [test_feature_dict[dvid][cand_index2][char] - test_feature_dict[dvid][cand_index1][char] for char in range(0,len(test_feature_dict[dvid][cand_index2]))]
                    # feature_2_1 = test_feature_dict[dvid][cand_index2] - test_feature_dict[dvid][cand_index1]
                    self.pair_test_feature_dict[dvid][findex] = feature_2_1
                    self.test_findex_2_cindex_dict[dvid][findex] = (cand_index2, cand_index1)
                    findex += 1
        self.test_range_label_dict_1 = np.load(
            self.label_path+"/range_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_1.npy", allow_pickle=True).item()
        self.test_range_label_dict_2 = np.load(
            self.label_path+"/range_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_2.npy", allow_pickle=True).item()
        self.test_range_label_dict_3 = np.load(
            self.label_path+"/range_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_3.npy", allow_pickle=True).item()
        self.test_range_label_dict_4 = np.load(
            self.label_path+"/range_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_4.npy", allow_pickle=True).item()

        self.test_global_label_dict_1 = np.load(
            self.label_path+"/global_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_1.npy", allow_pickle=True).item()
        self.test_global_label_dict_2 = np.load(
            self.label_path+"/global_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_2.npy", allow_pickle=True).item()
        self.test_global_label_dict_3 = np.load(
            self.label_path+"/global_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_3.npy", allow_pickle=True).item()
        self.test_global_label_dict_4 = np.load(
            self.label_path+"/global_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_4.npy", allow_pickle=True).item()


        for dvid in list(set(self.test_range_label_dict_1) | set(self.test_range_label_dict_2) | set(self.test_range_label_dict_3) | set(self.test_range_label_dict_4) | set(self.test_global_label_dict_1) | set(self.test_global_label_dict_2) | set(self.test_global_label_dict_3) | set(self.test_global_label_dict_4)):
            for k in dvinfos:
                if k["ID"] == dvid:
                    if "," in k["Value"]:
                        self.global_list.append(dvid)
                    else:
                        self.range_list.append(dvid)

    def get_batches(self, batch_id):
        X_train = []
        y_train = []

        

        if batch_id == 1:
            for feature, label in [(self.pair_train_feature_dict_1, self.pair_train_label_dict_1), (self.pair_train_feature_dict_2, self.pair_train_label_dict_2), (self.pair_train_feature_dict_3, self.pair_train_label_dict_3)]:
                for dvid in feature.keys():
                    for candid in feature[dvid]:
                        if dvid in label:
                            if candid in label[dvid]:
                                X_train.append(feature[dvid][candid])
                                y_train.append(label[dvid][candid])
        if batch_id == 2:
            for feature, label in [(self.pair_train_feature_dict_1, self.pair_train_label_dict_1), (self.pair_train_feature_dict_2, self.pair_train_label_dict_2), (self.pair_train_feature_dict_4, self.pair_train_label_dict_4)]:
                for dvid in feature.keys():
                    for candid in feature[dvid]:
                        if dvid in label:
                            if candid in label[dvid]:
                                X_train.append(feature[dvid][candid])
                                y_train.append(label[dvid][candid])
        if batch_id == 3:
            for feature, label in [(self.pair_train_feature_dict_1, self.pair_train_label_dict_1), (self.pair_train_feature_dict_3, self.pair_train_label_dict_3), (self.pair_train_feature_dict_4, self.pair_train_label_dict_4)]:
                for dvid in feature.keys():
                    for candid in feature[dvid]:
                        if dvid in label:
                            if candid in label[dvid]:
                                X_train.append(feature[dvid][candid])
                                y_train.append(label[dvid][candid])
        if batch_id == 4:
            for feature, label in [(self.pair_train_feature_dict_2, self.pair_train_label_dict_2), (self.pair_train_feature_dict_3, self.pair_train_label_dict_3), (self.pair_train_feature_dict_4, self.pair_train_label_dict_4)]:
                for dvid in feature.keys():
                    for candid in feature[dvid]:
                        if dvid in label:
                            if candid in label[dvid]:
                                X_train.append(feature[dvid][candid])
                                y_train.append(label[dvid][candid])
    
        index = 0
        reshape_feature = []
        reshape_label = []
        while index+self.batch_size < len(X_train):
            max_index = index+self.batch_size
            if max_index >= len(X_train):
                max_index = len(X_train)
            reshape_feature.append(X_train[index:max_index])
            reshape_label.append(y_train[index:max_index])
            index+=self.batch_size

        return reshape_feature, reshape_label

    def train(self, batch_id):
        self.model = PairNN()
        self.load_train_data(self.batch_id)
        feature_train, label_train = self.get_batches(batch_id)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_func = torch.nn.CrossEntropyLoss(reduce=False, size_average=False)

        for episode in range(0, self.episodes_num):
            # print("episode", episode)
            for index, feature in enumerate(feature_train):
                range_feature = torch.from_numpy(np.array(feature))
                # print(len(range_feature))
                merge_result = self.model(range_feature)
                
                # print('merge_result', merge_result)
                # print('target', torch.from_numpy(np.array(label_train[index])))
                target_tensor = torch.tensor(torch.from_numpy(np.array(label_train[index])), dtype=torch.long)
                loss = torch.mean(loss_func(merge_result, target_tensor))
                print("episode", episode, 'loss', loss)   
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        torch.save(self.model.state_dict(), self.res_path + '/pair_model_'+str(batch_id))

    def test(self, batch_id):
        self.model = PairNN()
        self.model.load_state_dict(torch.load(self.res_path + '/pair_model_'+str(batch_id)))
        sort_cand_dict = {}
        self.load_test_data(batch_id)
        X_test = []
        test_id_dict_1 = {}
        test_id_dict = {}
        _id = 0
        # print(feature_dict)
        if batch_id == 1:
            feature_dict = self.pair_test_feature_dict
            range_label_dict = self.test_range_label_dict_4
            global_label_dict = self.test_global_label_dict_4
        if batch_id == 2:
            feature_dict = self.pair_test_feature_dict
            range_label_dict = self.test_range_label_dict_3
            global_label_dict = self.test_global_label_dict_3
        if batch_id == 3:
            feature_dict = self.pair_test_feature_dict
            range_label_dict = self.test_range_label_dict_2
            global_label_dict = self.test_global_label_dict_2
        if batch_id == 4:
            feature_dict = self.pair_test_feature_dict
            range_label_dict = self.test_range_label_dict_1
            global_label_dict = self.test_global_label_dict_1

        for dvid in feature_dict.keys():
            for findex in feature_dict[dvid]:
                if dvid not in test_id_dict_1:
                    test_id_dict_1[dvid] = {}
                test_id_dict_1[dvid][findex] = _id
                test_id_dict[_id] = [dvid, findex]
                _id += 1
                X_test.append(feature_dict[dvid][findex])

        y_pred = self.model(np.array(X_test))[:,1]

        for index, pred_res in enumerate(y_pred):
            pred_dvid, pred_findex = test_id_dict[index]
            pred_cand1, pred_cand2 = self.test_findex_2_cindex_dict[pred_dvid][pred_findex]
            # print("dvid", pred_dvid, "cand1", pred_cand1, "cand2", pred_cand2, 'pred_res', pred_res)

            if pred_res > 0.5: # pred_cand1 better than pred_cand2
                better_cand = pred_cand1
                worse_cand = pred_cand2
            else:
                better_cand = pred_cand2
                worse_cand = pred_cand1
            if pred_dvid not in sort_cand_dict:
                sort_cand_dict[pred_dvid] = {}
            if better_cand not in sort_cand_dict[pred_dvid]:
                sort_cand_dict[pred_dvid][better_cand] = 0
            if worse_cand not in sort_cand_dict[pred_dvid]:
                sort_cand_dict[pred_dvid][worse_cand] = 0
            sort_cand_dict[pred_dvid][better_cand] += 1
        best_cand_index_dict = {}
        for dvid in sort_cand_dict:
            sort_list = sorted(sort_cand_dict[dvid].items(), key=lambda x: x[1], reverse=True)
            best_cand_index_dict[dvid] = []
            for index in range(0, min(self.precision, len(sort_list))):
                best_cand_index_dict[dvid].append(sort_list[index][0])

        hitnum = 0
        suc_dvid = []
        range_true = 0
        for dvid in best_cand_index_dict:
            is_in_label = False
            if dvid in range_label_dict:
                for cad in range_label_dict[dvid]:
                    if range_label_dict[dvid][cad] == 1:
                        is_in_label = True
                        range_true += 1
                for best_cand_index in best_cand_index_dict[dvid]:
                        if best_cand_index in range_label_dict[dvid]:
                                if range_label_dict[dvid][best_cand_index] > 0.5:
                                    suc_dvid.append(dvid)
                                    hitnum += 1
                                    break
            if dvid in global_label_dict:
                for best_cand_index in best_cand_index_dict[dvid]: 
                    if best_cand_index in global_label_dict[dvid]:
                        # print("global:", global_label_dict[dvid])
                        # print('best_cand_index', best_cand_index)
                            for cad in global_label_dict[dvid]:
                                if global_label_dict[dvid][cad] == 1:
                                    print(global_label_dict[dvid])
                                    print(best_cand_index)
                                    print("global_label_dict[dvid][best_cand_index]", global_label_dict[dvid][best_cand_index])
                                    break
                            if global_label_dict[dvid][best_cand_index] > 0.5:
                                suc_dvid.append(dvid)
                                hitnum += 1
                                break
        print('range_true', range_true)
        print(len(suc_dvid))
        return suc_dvid

    def evaluate(self, rank_suc):
        all_global_list = self.global_list
        all_range_list = self.range_list
        result = {}
        rank_range_suc = set(rank_suc) & set(all_range_list)
        rank_global_suc = set(rank_suc) & set(all_global_list)
        result["len_rank_range_suc"] = len(set(rank_suc) & set(all_range_list))
        result["len_rank_range_fail"] = len(all_range_list) - result["len_rank_range_suc"]
        result["len_rank_global_suc"] = len(set(rank_suc) & set(all_global_list))
        result["len_rank_global_fail"] = len(all_global_list) - result["len_rank_global_suc"]
        result["len_global_list"] = len(all_global_list)
        result["len_range_list"] = len(all_range_list)
        result["hit@"+str(self.precision)] = (result["len_rank_range_suc"] + result["len_rank_global_suc"])/(len(all_global_list) + len(all_range_list))
        result["rank range hit@"+str(self.precision)] = result["len_rank_range_suc"] / len(all_range_list)
        result["rank global hit@"+str(self.precision)] = result["len_rank_global_suc"] / len(all_global_list)

        print('len_rank_range_suc', len(rank_range_suc))
        print('len_rank_range_fail', result["len_rank_range_fail"])
        print('len_rank_global_suc', len(rank_global_suc))
        print('len_rank_global_fail', result["len_rank_global_fail"])

        print("acc_1:", result["hit@"+str(self.precision)])
        print("rank_range_acc_1:", len(rank_range_suc) / len(all_range_list))
        print("rank_global_acc_1:", len(rank_global_suc) / len(all_global_list))

        # print("global_pred_result", global_pred_result)
        # for dvid in self.pred_result:
        #     for key in self.pred_result[dvid]:
        #         for index,point in enumerate(self.pred_result[dvid][key]):
        #             self.pred_result[dvid][key][index] = float(point)
                    
        # b = json.dumps(self.pred_result)
        # f2 = open(self.res_path+'/pred_result_'+str(self.precision)+"_.json", 'w')
        # f2.write(b)
        # f2.close()

        
        b = json.dumps(result)
        f2 = open(self.res_path+'/evaluation_'+str(self.precision)+"_NN.json", 'w')
        f2.write(b)
        f2.close()

    def run(self, need_train=False):
        # if need_train:
        #     for batch_id in [1,2,3,4]:
        #         self.train(batch_id)
        # batch_id=2
        # batch_rank_suc = self.test(batch_id)
        # np.save(self.res_path+'/suc_dvid'+str(batch_id)+"_"+str(self.precision)+"_NN", batch_rank_suc)
        # batch_id=3
        # batch_rank_suc = self.test(batch_id)
        # np.save(self.res_path+'/suc_dvid'+str(batch_id)+"_"+str(self.precision)+"_NN", batch_rank_suc)
        # batch_id=4
        # batch_rank_suc = self.test(batch_id)
        # np.save(self.res_path+'/suc_dvid'+str(batch_id)+"_"+str(self.precision)+"_NN", batch_rank_suc)
        suc_dvid1 = np.load(self.res_path+'/suc_dvid'+str(1)+"_"+str(self.precision)+"_NN.npy",allow_pickle=True)
        suc_dvid2 = np.load(self.res_path+'/suc_dvid'+str(2)+"_"+str(self.precision)+"_NN.npy",allow_pickle=True)
        suc_dvid3 = np.load(self.res_path+'/suc_dvid'+str(3)+"_"+str(self.precision)+"_NN.npy",allow_pickle=True)
        suc_dvid4 = np.load(self.res_path+'/suc_dvid'+str(4)+"_"+str(self.precision)+"_NN.npy",allow_pickle=True)
        batch_rank_suc = list(suc_dvid1)+list(suc_dvid2)+list(suc_dvid3)+list(suc_dvid4)
        self.evaluate(batch_rank_suc)

if __name__ == '__main__':
    pair_trainer = PairTrainer(evaluate_range=1, evaluate_global=1, precision=1, episodes_num=10, batch_id = 1, learning_rate=1e-3, batch_size=32)
    pair_trainer.load_train_data(1)
    pair_trainer.run(need_train=True)