from list_nn import PointNN
import torch
import random
import json
import numpy as np
import os

with open("../AnalyzeDV/continous_batch_0.json", 'r', encoding='UTF-8') as f:
    dvinfos = json.load(f)
    
class PointTrainer:
    def __init__(self, evaluate_range, evaluate_global, precision, episodes_num=10, learning_rate=1e-3, batch_size=500):
        self.evaluate_range = evaluate_range
        self.evaluate_global = evaluate_global
        self.precision = precision
        self.episodes_num = episodes_num
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.load_data()
        self.res_path = "with_header_sname/nn_range_"+str(self.evaluate_range).replace('-','n') + "_global_" + str(self.evaluate_global)
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

    def load_data(self):
        self.range_feature_dict_1 = np.load(
            "with_header_sname/range_feature_dict_1.npy", allow_pickle=True).item()
        self.range_feature_dict_2 = np.load(
            "with_header_sname/range_feature_dict_2.npy", allow_pickle=True).item()
        self.range_feature_dict_3 = np.load(
            "with_header_sname/range_feature_dict_3.npy", allow_pickle=True).item()
        self.range_feature_dict_4 = np.load(
            "with_header_sname/range_feature_dict_4.npy", allow_pickle=True).item()

        self.global_feature_dict_1 = np.load(
            "with_header_sname/global_feature_dict_1.npy", allow_pickle=True).item()
        self.global_feature_dict_2 = np.load(
            "with_header_sname/global_feature_dict_2.npy", allow_pickle=True).item()
        self.global_feature_dict_3 = np.load(
            "with_header_sname/global_feature_dict_3.npy", allow_pickle=True).item()
        self.global_feature_dict_4 = np.load(
            "with_header_sname/global_feature_dict_4.npy", allow_pickle=True).item()

        one_info = np.load("with_header_sname/one_info.npy",
                            allow_pickle=True).item()
        id_dict = one_info['id_dict']

        dvid_list = one_info["dvid_list"]
        start_1 = one_info["start_1"]
        start_2 = one_info["start_2"]
        start_3 = one_info["start_3"]
        id_dict_1 = one_info["id_dict_1"]

        self.global_list = []
        self.range_list = []
        self.test_range_label_dict_1 = np.load(
            "without_header/range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"/label/range_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_1.npy", allow_pickle=True).item()
        self.test_range_label_dict_2 = np.load(
            "without_header/range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"/label/range_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_2.npy", allow_pickle=True).item()
        self.test_range_label_dict_3 = np.load(
            "without_header/range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"/label/range_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_3.npy", allow_pickle=True).item()
        self.test_range_label_dict_4 = np.load(
            "without_header/range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"/label/range_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_4.npy", allow_pickle=True).item()
            
        self.test_global_label_dict_1 = np.load(
            "without_header/range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"/label/global_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_1.npy", allow_pickle=True).item()
        self.test_global_label_dict_2 = np.load(
            "without_header/range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"/label/global_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_2.npy", allow_pickle=True).item()
        self.test_global_label_dict_3 = np.load(
            "without_header/range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"/label/global_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_3.npy", allow_pickle=True).item()
        self.test_global_label_dict_4 = np.load(
            "without_header/range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"/label/global_label_dict_range_"+str(self.evaluate_range).replace("-","n")+"_global_"+str(self.evaluate_global)+"_4.npy", allow_pickle=True).item()

        self.train_range_label_dict_1 = np.load(
            "without_header/range_0_global_0/label/range_label_dict_range_0_global_0_1.npy", allow_pickle=True).item()
        self.train_range_label_dict_2 = np.load(
            "without_header/range_0_global_0/label/range_label_dict_range_0_global_0_2.npy", allow_pickle=True).item()
        self.train_range_label_dict_3 = np.load(
            "without_header/range_0_global_0/label/range_label_dict_range_0_global_0_3.npy", allow_pickle=True).item()
        self.train_range_label_dict_4 = np.load(
            "without_header/range_0_global_0/label/range_label_dict_range_0_global_0_4.npy", allow_pickle=True).item()

        self.train_global_label_dict_1 = np.load(
            "without_header/range_0_global_0/label/global_label_dict_range_0_global_0_1.npy", allow_pickle=True).item()
        self.train_global_label_dict_2 = np.load(
            "without_header/range_0_global_0/label/global_label_dict_range_0_global_0_2.npy", allow_pickle=True).item()
        self.train_global_label_dict_3 = np.load(
            "without_header/range_0_global_0/label/global_label_dict_range_0_global_0_3.npy", allow_pickle=True).item()
        self.train_global_label_dict_4 = np.load(
            "without_header/range_0_global_0/label/global_label_dict_range_0_global_0_4.npy", allow_pickle=True).item()

        for dvid in list(set(self.train_range_label_dict_1) | set(self.train_range_label_dict_2) | set(self.train_range_label_dict_3) | set(self.train_range_label_dict_4) | set(self.train_global_label_dict_1) | set(self.train_global_label_dict_2) | set(self.train_global_label_dict_3) | set(self.train_global_label_dict_4)):
            for k in dvinfos:
                if k["ID"] == dvid:
                    if "," in k["Value"]:
                        self.global_list.append(dvid)
                    else:
                        self.range_list.append(dvid)

    def get_batches(self, batch_id):
        range_X_train = []
        range_y_train = []
        global_X_train = []
        global_y_train = []

        range_X_test = []
        range_y_test = []
        global_X_test = []
        global_y_test = []

        if batch_id == 1:
            for feature, label in [(self.range_feature_dict_1, self.train_range_label_dict_1), (self.range_feature_dict_2, self.train_range_label_dict_2), (self.range_feature_dict_3, self.train_range_label_dict_3)]:
                for dvid in feature.keys():
                    for candid in feature[dvid]:
                        if dvid in label:
                            if candid in label[dvid]:
                                range_X_train.append(self.change_both_feature(True, feature[dvid][candid]))
                                range_y_train.append(label[dvid][candid])
            for feature, label in [(self.global_feature_dict_1, self.train_global_label_dict_1), (self.global_feature_dict_2, self.train_global_label_dict_2), (self.global_feature_dict_3, self.train_global_label_dict_3)]:
                for dvid in feature.keys():
                    for candid in feature[dvid]:
                        if dvid in label:
                            if candid in label[dvid]:
                                global_X_train.append(self.change_both_feature(False, feature[dvid][candid]))
                                global_y_train.append(label[dvid][candid])
        if batch_id == 2:
            for feature, label in [(self.range_feature_dict_4, self.train_range_label_dict_4), (self.range_feature_dict_2, self.train_range_label_dict_2), (self.range_feature_dict_3, self.train_range_label_dict_3)]:
                for dvid in feature.keys():
                    for candid in feature[dvid]:
                        if dvid in label:
                            if candid in label[dvid]:
                                range_X_train.append(self.change_both_feature(True, feature[dvid][candid]))
                                range_y_train.append(label[dvid][candid])
            for feature, label in [(self.global_feature_dict_4, self.train_global_label_dict_4), (self.global_feature_dict_2, self.train_global_label_dict_2), (self.global_feature_dict_3, self.train_global_label_dict_3)]:
                for dvid in feature.keys():
                    for candid in feature[dvid]:
                        if dvid in label:
                            if candid in label[dvid]:
                                global_X_train.append(self.change_both_feature(False, feature[dvid][candid]))
                                global_y_train.append(label[dvid][candid])
        if batch_id == 3:
            for feature, label in [(self.range_feature_dict_4, self.train_range_label_dict_4), (self.range_feature_dict_1, self.train_range_label_dict_1), (self.range_feature_dict_3, self.train_range_label_dict_3)]:
                for dvid in feature.keys():
                    for candid in feature[dvid]:
                        if dvid in label:
                            if candid in label[dvid]:
                                range_X_train.append(self.change_both_feature(True, feature[dvid][candid]))
                                range_y_train.append(label[dvid][candid])
            for feature, label in [(self.global_feature_dict_4, self.train_global_label_dict_4), (self.global_feature_dict_1, self.train_global_label_dict_1), (self.global_feature_dict_3, self.train_global_label_dict_3)]:
                for dvid in feature.keys():
                    for candid in feature[dvid]:
                        if dvid in label:
                            if candid in label[dvid]:
                                global_X_train.append(self.change_both_feature(False, feature[dvid][candid]))
                                global_y_train.append(label[dvid][candid])
        if batch_id == 4:
            for feature, label in [(self.range_feature_dict_4, self.train_range_label_dict_4), (self.range_feature_dict_2, self.train_range_label_dict_2), (self.range_feature_dict_1, self.train_range_label_dict_1)]:
                for dvid in feature.keys():
                    for candid in feature[dvid]:
                        if dvid in label:
                            if candid in label[dvid]:
                                range_X_train.append(self.change_both_feature(True, feature[dvid][candid]))
                                range_y_train.append(label[dvid][candid])
            for feature, label in [(self.global_feature_dict_4, self.train_global_label_dict_4), (self.global_feature_dict_2, self.train_global_label_dict_2), (self.global_feature_dict_1, self.train_global_label_dict_1)]:
                for dvid in feature.keys():
                    for candid in feature[dvid]:
                        if dvid in label:
                            if candid in label[dvid]:
                                global_X_train.append(self.change_both_feature(False, feature[dvid][candid]))
                                global_y_train.append(label[dvid][candid])

        feature_train = range_X_train + global_X_train
        label_train = range_y_train + global_y_train
    
        index = 0
        reshape_feature = []
        reshape_label = []
        while index+self.batch_size < len(feature_train):
            max_index = index+self.batch_size
            if max_index >= len(feature_train):
                max_index = len(feature_train)
            reshape_feature.append(feature_train[index:max_index])
            reshape_label.append(label_train[index:max_index])
            index+=self.batch_size

        return reshape_feature, reshape_label

    def train(self, batch_id):
        self.model = PointNN()
        feature_train, label_train = self.get_batches(batch_id)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_func = torch.nn.CrossEntropyLoss(reduce=False, size_average=False)

        last_feature = []
        for episode in range(0, self.episodes_num):
            # print("episode", episode)
            for index, feature in enumerate(feature_train):
                # range_feature = torch.from_numpy(np.array(feature))
                range_feature = torch.from_numpy(np.array([i[0:9] for i in feature]))
                global_feature = torch.from_numpy(np.array([i[9:] for i in feature]))
                merge_result = self.model(range_feature, global_feature)
                
                # print('merge_result', merge_result)
                # print('target', torch.from_numpy(np.array(label_train[index])))
                target_tensor = torch.tensor(torch.from_numpy(np.array(label_train[index])), dtype=torch.long)
                loss = torch.mean(loss_func(merge_result, target_tensor))
                if last_feature == feature:
                    print('same')
                    print(feature[0:5])
                else:
                    print('not same')
                print("episode", episode, 'loss', loss)   
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_feature = feature

    def test(self, batch_id):
        test_id_dict_1 = {}
        test_id_dict = {}
        _id = 0
        feature_test = []
        label_test = []
        if batch_id == 1:
            for dvid in self.range_feature_dict_4.keys():
                for candid in self.range_feature_dict_4[dvid]:
                    if dvid in self.test_range_label_dict_4:
                        if candid in self.test_range_label_dict_4[dvid]:

                            if dvid not in test_id_dict_1:
                                test_id_dict_1[dvid] = {}
                            test_id_dict_1[dvid][candid] = _id
                            test_id_dict[_id] = [dvid, candid]
                            _id += 1
                            feature_test.append(self.change_both_feature(True, self.range_feature_dict_4[dvid][candid]))
                            label_test.append(self.test_range_label_dict_4[dvid][candid])
                if dvid in self.global_feature_dict_4:
                    for candid in self.global_feature_dict_4[dvid]:
                        if dvid in self.test_global_label_dict_4:
                            if candid in self.test_global_label_dict_4[dvid]:
                                if dvid not in test_id_dict_1:
                                    test_id_dict_1[dvid] = {}
                                test_id_dict_1[dvid][candid] = _id
                                test_id_dict[_id] = [dvid, candid]
                                _id += 1
                                feature_test.append(self.change_both_feature(False, self.global_feature_dict_4[dvid][candid]))
                                label_test.append(self.test_global_label_dict_4[dvid][candid])
        elif batch_id == 2:
            for dvid in self.range_feature_dict_1.keys():
                for candid in self.range_feature_dict_1[dvid]:
                    if dvid in self.test_range_label_dict_1:
                        if candid in self.test_range_label_dict_1[dvid]:

                            if dvid not in test_id_dict_1:
                                test_id_dict_1[dvid] = {}
                            test_id_dict_1[dvid][candid] = _id
                            test_id_dict[_id] = [dvid, candid]
                            _id += 1
                            feature_test.append(self.change_both_feature(True, self.range_feature_dict_1[dvid][candid]))
                            label_test.append(self.test_range_label_dict_1[dvid][candid])
                if dvid in self.global_feature_dict_1:
                    for candid in self.global_feature_dict_1[dvid]:
                        if dvid in self.test_global_label_dict_1:
                            if candid in self.test_global_label_dict_1[dvid]:
                                if dvid not in test_id_dict_1:
                                    test_id_dict_1[dvid] = {}
                                test_id_dict_1[dvid][candid] = _id
                                test_id_dict[_id] = [dvid, candid]
                                _id += 1
                                feature_test.append(self.change_both_feature(False, self.global_feature_dict_1[dvid][candid]))
                                label_test.append(self.test_global_label_dict_1[dvid][candid])
        if batch_id == 3:
            for dvid in self.range_feature_dict_2.keys():
                for candid in self.range_feature_dict_2[dvid]:
                    if dvid in self.test_range_label_dict_2:
                        if candid in self.test_range_label_dict_2[dvid]:

                            if dvid not in test_id_dict_1:
                                test_id_dict_1[dvid] = {}
                            test_id_dict_1[dvid][candid] = _id
                            test_id_dict[_id] = [dvid, candid]
                            _id += 1
                            feature_test.append(self.change_both_feature(True, self.range_feature_dict_2[dvid][candid]))
                            label_test.append(self.test_range_label_dict_2[dvid][candid])
                if dvid in self.global_feature_dict_2:
                    for candid in self.global_feature_dict_2[dvid]:
                        if dvid in self.test_global_label_dict_2:
                            if candid in self.test_global_label_dict_2[dvid]:
                                if dvid not in test_id_dict_1:
                                    test_id_dict_1[dvid] = {}
                                test_id_dict_1[dvid][candid] = _id
                                test_id_dict[_id] = [dvid, candid]
                                _id += 1
                                feature_test.append(self.change_both_feature(False, self.global_feature_dict_2[dvid][candid]))
                                label_test.append(self.test_global_label_dict_2[dvid][candid])
        if batch_id == 4:
            for dvid in self.range_feature_dict_3.keys():
                for candid in self.range_feature_dict_3[dvid]:
                    if dvid in self.test_range_label_dict_3:
                        if candid in self.test_range_label_dict_3[dvid]:

                            if dvid not in test_id_dict_1:
                                test_id_dict_1[dvid] = {}
                            test_id_dict_1[dvid][candid] = _id
                            test_id_dict[_id] = [dvid, candid]
                            _id += 1
                            feature_test.append(self.change_both_feature(True, self.range_feature_dict_3[dvid][candid]))
                            label_test.append(self.test_range_label_dict_3[dvid][candid])
                if dvid in self.global_feature_dict_3:
                    for candid in self.global_feature_dict_3[dvid]:
                        if dvid in self.test_global_label_dict_3:
                            if candid in self.test_global_label_dict_3[dvid]:
                                if dvid not in test_id_dict_1:
                                    test_id_dict_1[dvid] = {}
                                test_id_dict_1[dvid][candid] = _id
                                test_id_dict[_id] = [dvid, candid]
                                _id += 1
                                feature_test.append(self.change_both_feature(False, self.global_feature_dict_3[dvid][candid]))
                                label_test.append(self.test_global_label_dict_3[dvid][candid])

        # range_feature = torch.from_numpy(np.array(feature_test))
        range_feature = torch.from_numpy(np.array([i[0:9] for i in feature_test]))
        global_feature = torch.from_numpy(np.array([i[9:] for i in feature_test]))
        y_pred = self.model(range_feature, global_feature)[:,1]

        dvid_list = []
        pred_list = []
        test_list = []

        rank_suc = []
        rank_fail = []

        self.pred_result = {}
        for index, i in enumerate(label_test):
            if len(dvid_list) == 0:
                dvid_list.append(test_id_dict[index][0])
            else:
                if test_id_dict[index][0] not in dvid_list:
                    self.pred_result[dvid_list[-1]
                                ] = {"test": test_list, "pred": pred_list}
                    if 1 not in test_list:
                        rank_fail.append(dvid_list[-1])
                    else:
                        prec = 0
                        temp = pred_list.copy()
                        top_index_list = []
                        while prec < self.precision:
                            top_index = np.array(temp).argmax()
                            top_index_list.append(top_index)
                            temp[top_index] = -1
                            prec += 1

                        is_suc = False
                        for top_index in top_index_list:
                            if test_list[top_index] != 0:
                                is_suc = True
                                break
                        if not is_suc:
                            rank_fail.append(dvid_list[-1])
                        else:
                            rank_suc.append(dvid_list[-1])

                    pred_list = []
                    test_list = []
                    dvid_list.append(test_id_dict[index][0])

            pred_list.append(y_pred[index])
            test_list.append(label_test[index])
        return rank_suc

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
        for dvid in self.pred_result:
            for key in self.pred_result[dvid]:
                for index,point in enumerate(self.pred_result[dvid][key]):
                    self.pred_result[dvid][key][index] = float(point)
                    
        b = json.dumps(self.pred_result)
        f2 = open(self.res_path+'/pred_result_'+str(self.precision)+"_.json", 'w')
        f2.write(b)
        f2.close()

        
        b = json.dumps(result)
        f2 = open(self.res_path+'/evaluation_'+str(self.precision)+"_NN.json", 'w')
        f2.write(b)
        f2.close()

    def run(self):
        rank_suc = []
        for batch_id in [1,2,3,4]:
            self.train(batch_id)
            batch_rank_suc = self.test(batch_id)
            rank_suc += batch_rank_suc
        self.evaluate(rank_suc)

if __name__ == '__main__':
    point_trainer = PointTrainer(evaluate_range=1, evaluate_global=1, precision=1, episodes_num=10, learning_rate=1e-3, batch_size=500)
    point_trainer.run()