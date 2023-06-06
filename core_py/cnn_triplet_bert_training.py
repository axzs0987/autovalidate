import os
import json
from cnn_model import CNNnet, CNNnet_Cosine,CNNnetTriplet, CNNnetTripletBert
import torch
import random
import multiprocessing
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pprint
import gc
from sklearn.metrics import roc_curve, auc
import faiss
from sentence_transformers import SentenceTransformer
from analyze_formula import get_feature_vector_with_bert_keyw
import sys

with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
    bert_dict = json.load(f)
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
class Logger(object):
    logfile = ""
    def __init__(self, filename=""):
        self.logfile = filename
        self.terminal = sys.stdout
        return
    def write(self, message):
        self.terminal.write(message)
        if self.logfile != "":
            try:
                self.log = open(self.logfile, "a")
                self.log.write(message)
                self.log.close()
            except:
                pass
    def flush(self):
        pass

sys.stdout = Logger('bert_training_out_1_margin20.log')
sys.stderr = Logger('bert_training_out_1_margin20.log')
# log_print = open('bert_training_out_margin20.log', 'w')
# sys.stdout = log_print
# sys.stderr = log_print

np.set_printoptions(threshold=np.inf)
embed_features1 = []
class TripletSelection():
    def __init__(self, batch_size, inner_times, epoch_nums, topk, max_inclass, margin):
        self.margin=margin
        self.batch_size = batch_size
        self.topk = topk
        self.all_list = set()
        self.auchors = []
        self.triplet_pair = []
        self.training_data = self.load_data()
        # self.content_dict = {}
        self.content_tem_dict = {}
        # self.train_loader = DataLoader(x394,batch_size=batch_size,shuffle=True)
        # if os.path.exists('cnn_dynamic_triplet_0'):
        #     self.model = torch.load('cnn_dynamic_triplet_0')
        # else:
        self.model = CNNnetTripletBert()
        self.max_inclass = max_inclass
        # self.trained_index = 0
        self.features = []
        self.inner_times = inner_times
        self.epoch_nums = epoch_nums
        self.loss_func = torch.nn.TripletMarginLoss(margin=self.margin, p=2)
        self.opt = torch.optim.Adam(self.model.parameters(),lr=0.001)
        self.embed_features = {}
        self.id2filesheet = {}
        self.class_index = 0
        self.inclass_index = 0
        self.sheet_index = 0
        self.bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def load_data(self):
        # if os.path.exists('json_data/content_dict.json'):
        #     with open("json_data/content_dict.json", 'r') as f:
        #         self.content_dict = json.load(f)
        if os.path.exists('json_data/content_temp_dict.json'):
            with open("json_data/content_temp_dict.json", 'r') as f:
                self.content_tem_dict = json.load(f)
        with open("json_data/sheetname_2_file_devided_prob_filter.json", 'r') as f:
                self.sheetname_2_file_devided = json.load(f)
                self.classes = list(self.sheetname_2_file_devided.keys())
        if os.path.exists("json_data/100000_all_list.json"):
            with open("json_data/100000_all_list.json", 'r') as f:
                self.all_list = json.load(f)
        else:
            filelist = os.listdir("../../data/sheet_features_rgb/")
            for sheetname in self.sheetname_2_file_devided:
                for same_filelist in self.sheetname_2_file_devided[sheetname]:
                    # print(same_filelist)
                    for filename in same_filelist['filenames']:
                        # print("################")
                        # print(filename)
                        # print(filename.split('/')[-1]+"---"+sheetname+".json")
                        # print(filelist[0])
                        if(filename.split('/')[-1]+"---"+sheetname+".json" in filelist):
                            self.all_list.add(filename+"---"+sheetname)
                            # print(filename+"---"+sheetname)
            # print('self.all_list', len(self.all_list))

            with open("json_data/100000_all_list.json", 'w') as f:
                json.dump(list(self.all_list), f)
        # if os.path.exists("100000_triplet_pair.json") and os.path.exists("100000_auchors.json"):
        #     with open("100000_triplet_pair.json", 'r') as f:
        #         self.triplet_pair = json.load(f)
        #     with open("100000_auchors.json", 'r') as f:
        #         self.auchors = json.load(f)
        # else:
        #     with open("100000_positive_pair.json", 'r') as f:
        #         positive_pair = json.load(f)
        #     with open("100000_positive_pair1.json", 'r') as f:
        #         positive_pair1 = json.load(f)

        # count = 0 
        # batch_list = []
        # triplet_list = []
        # for positive_pr in [positive_pair, positive_pair1]:
        #     for sheetname in positive_pair:
        #         for pair in positive_pair[sheetname]:
        #             sheetname1 = random.choice(list(self.sheetname_2_file_devided.keys()))
        #             while(sheetname1==sheetname) or len(self.sheetname_2_file_devided[sheetname1])<1:
        #                 sheetname1 = random.choice(list(self.sheetname_2_file_devided.keys()))
        #             # print(list(sheet_2_id[sheetname1].keys()))
        #             rand_filelist = random.choice(self.sheetname_2_file_devided[sheetname1])
        #             filename1 = random.choice(rand_filelist['filenames'])
        #             print(filename1)
        #             if pair[0]+"---"+sheetname+','+pair[1]+"---"+sheetname+','+filename1+'---'+sheetname1 not in triplet_list:
        #                 batch_list.append((pair[0]+"---"+sheetname,pair[1]+"---"+sheetname,filename1+'---'+sheetname1))
        #                 triplet_list.append(pair[0]+"---"+sheetname+','+pair[1]+"---"+sheetname+','+filename1+'---'+sheetname1)
        #                 self.auchors.append(pair[0]+"---"+sheetname)
        #                 count += 1
        #             if count % 5000 == 0:
        #                 self.triplet_pair.append(batch_list)
        #                 batch_list = []
        # self.triplet_pair.append(batch_list)

        # with open("100000_triplet_pair.json", 'w') as f:
        #     json.dump(triplet_list, f)
        # with open("100000_auchors.json", 'w') as f:
        #     json.dump(self.auchors, f)

    def transfer_origin_feature_with_bert(self,origin_feature):
        feature = []
        # feature = get_feature_vector(item[''])
        for one_cell_feature in origin_feature['sheetfeature']:
            new_feature = []
            for key1 in one_cell_feature:
                new_feature.append(one_cell_feature[key1])
            feature.append(new_feature)
        feature = self.get_feature_vector_with_bert_keyw(feature)
        return feature

    def load_feature(self, time):
        # gc.collect()
        # del self.features
        # gc.collect()
        print('load_feature......')
        self.features = []
        batch_list = self.triplet_pair

        temp_batch = []
        procs = []
        triplet_dict = multiprocessing.Manager().dict()
        # self.triplet_dict = {}
        features = multiprocessing.Manager().list()
        lock = multiprocessing.Lock()
        batch_len = len(batch_list)/multiprocessing.cpu_count() + 1
        for index,triplet in enumerate(batch_list):
            temp_batch.append(triplet)
            if len(temp_batch) >= batch_len:
                # self.load_one_feature_single(temp_batch, self.triplet_dict, self.features)
                procs.append(multiprocessing.Process(target=self.load_one_feature, args=(temp_batch, index, triplet_dict, features, lock)))
                temp_batch = []
        procs.append(multiprocessing.Process(target=self.load_one_feature, args=(temp_batch, index, triplet_dict, features, lock)))
        # self.load_one_feature_single(temp_batch, self.triplet_dict, self.features)
        temp_batch = []
        

        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
        
        self.features = features


    def load_one_feature(self, triplets, index, triplet_dict, features, lock):
        for triplet in triplets:

            print('triplet', triplet)
            # print(len(features), len(triplets))
            auchor_file, auchor_sheet = triplet[0].split("---")
            positive_file, positive_sheet = triplet[1].split("---")
            negative_file, negative_sheet = triplet[2].split("---")
            auchor_path = "../../data/sheet_features_rgb/"+ auchor_file.split("/")[-1]+"---"+auchor_sheet+".json"
            positive_path = "../../data/sheet_features_rgb/"+ positive_file.split("/")[-1]+"---"+positive_sheet+".json"
            negative_path = "../../data/sheet_features_rgb/"+ negative_file.split("/")[-1]+"---"+negative_sheet+".json"
            with open(auchor_path,'r') as f:
                origin_auchor_feature = json.load(f)
            with open(positive_path,'r') as f:
                origin_positive_feature = json.load(f)
            with open(negative_path,'r') as f:
                origin_negative_feature = json.load(f)
            auchor_feature = self.transfer_origin_feature_with_bert(origin_auchor_feature)
            positive_feature = self.transfer_origin_feature_with_bert(origin_positive_feature)
            negative_feature = self.transfer_origin_feature_with_bert(origin_negative_feature)
            lock.acquire()
            features.append((auchor_feature, positive_feature, negative_feature))
            # print(len(features))
            triplet_dict[len(features)-1] = index
            lock.release()
            gc.collect()
            del auchor_feature
            del positive_feature
            del negative_feature
            gc.collect()
    def load_one_feature_single(self, triplets, triplet_dict, features):
        print("load_one_feature_single........")
        for triplet in triplets:

            # print('triplet', triplet)
            # print(len(features), len(triplets))
            auchor_file, auchor_sheet = triplet[0].split("---")
            positive_file, positive_sheet = triplet[1].split("---")
            negative_file, negative_sheet = triplet[2].split("---")
            auchor_path = "../../data/sheet_features_rgb/"+ auchor_file.split("/")[-1]+"---"+auchor_sheet+".json"
            positive_path = "../../data/sheet_features_rgb/"+ positive_file.split("/")[-1]+"---"+positive_sheet+".json"
            negative_path = "../../data/sheet_features_rgb/"+ negative_file.split("/")[-1]+"---"+negative_sheet+".json"
            with open(auchor_path,'r') as f:
                origin_auchor_feature = json.load(f)
            with open(positive_path,'r') as f:
                origin_positive_feature = json.load(f)
            with open(negative_path,'r') as f:
                origin_negative_feature = json.load(f)
            auchor_feature = self.transfer_origin_feature_with_bert(origin_auchor_feature)
            positive_feature = self.transfer_origin_feature_with_bert(origin_positive_feature)
            negative_feature = self.transfer_origin_feature_with_bert(origin_negative_feature)

            features.append((auchor_feature, positive_feature, negative_feature))
            # print(len(features))
            # triplet_dict[len(features)-1] = index

            # gc.collect()
            del auchor_feature
            del positive_feature
            del negative_feature
            # gc.collect()

    def inner_training(self, time):
        self.load_feature(time)
        # self.features =np.load("features.npy")
        train_data = TensorDataset(torch.DoubleTensor(self.features))
        train_loader = DataLoader(train_data,batch_size=self.batch_size,shuffle=True)
        for epoch in range(self.inner_times):
            # for training_index in range(0, len(self.triplet_pair)):
                # self.trained_index = training_index
                
            print("inner epoch ....")
            for i,(x) in enumerate(train_loader):
                x = x[0]
                # archor = x[:,0,:].reshape(len(x),100,10,399).permute(0,3,1,2)
                # positive = x[:,1,:].reshape(len(x),100,10,399).permute(0,3,1,2)
                # negative = x[:,2,:].reshape(len(x),100,10,399).permute(0,3,1,2)
                archor = x[:,0,:].reshape(len(x),100,10,399)
                positive = x[:,1,:].reshape(len(x),100,10,399)
                negative = x[:,2,:].reshape(len(x),100,10,399)
                x1 = Variable(archor) # torch.Size([batch_size, 1000, 10])
                x2 = Variable(positive) ## torch.Size([batch_size, 1000, 10])
                x3 = Variable(negative) ## torch.Size([batch_size, 1000, 10])

                x1 = self.model(x1.to(torch.float32)) # torch.Size([128,10])
                x2 = self.model(x2.to(torch.float32))
                x3 = self.model(x3.to(torch.float32))

                self.loss = self.loss_func(x1,x2,x3)
                print('loss:', self.loss)
                self.opt.zero_grad()  # 清空上一步残余更新参数值
                self.loss.backward() # 误差反向传播，计算参数更新值
                self.opt.step() # 将参数更新值施加到net的parmeters上
            # torch.save(self.model,'cnn_dynamic_triplet_'+str(time)+"_"+str(epoch))
            torch.save(self.model,'bert_1_cnn_dynamic_triplet_margin_'+str(self.margin)+"_"+str(time)+"_"+str(epoch))
        gc.collect()
        del self.features
        del train_data
        del train_loader
        gc.collect()
        

    def batch_get(self, origin_feature_list, embed_features, id2filesheet, lock):
  
        print('start proc.....')
        count = 0
        for item in origin_feature_list:
            print(count)
            item_filename, item_sheetname = item.split("---")
            item_path = item_filename.split("/")[-1]+"---"+item_sheetname+".json"
            with open("../../data/sheet_features_rgb/"+item_path,'r') as f:
                origin_feature = json.load(f)
            print("xxxxx")
            feature = self.transfer_origin_feature_with_bert(origin_feature)
            print(len(feature))
            feature = torch.DoubleTensor(np.array(feature))
            # feature = feature.reshape(1,100,10,399).permute(0,3,1,2)
            feature.reshape(1,100,10,399)
            feature = Variable(feature)
            # print(self.model(feature))
           
            print("inferencing....")
            one_feature = self.model(feature.to(torch.float32))
            lock.acquire()
            # 
            print('after intferencing and add.....')
            embed_features.append(np.array(one_feature.detach()))
            print('infer end.....')
            # self.embed_features.append(self.model(feature))
            id2filesheet[len(embed_features)-1] = item
            print("go to release.....")

            lock.release()
            count += 1
            # print(count, len(origin_feature_list))
            # print('emb:', len(embed_features))
            gc.collect()
            del feature
            del origin_feature
            gc.collect()

    def batch_get_single(self, origin_feature_list, embed_features, id2filesheet):
        # print('start proc.....')
        count = 0
        for item in origin_feature_list:
            item_filename, item_sheetname = item.split("---")
            item_path = item_filename.split("/")[-1]+"---"+item_sheetname+".json"
            with open("../../data/sheet_features_rgb/"+item_path,'r') as f:
                origin_feature = json.load(f)
            feature = self.transfer_origin_feature_with_bert(origin_feature)
            # print(feature)
            feature = torch.DoubleTensor(np.array(feature))
            # feature = feature.reshape(1,100,10,399).permute(0,3,1,2)
            feature = feature.reshape(1,100,10,399)
            feature = Variable(feature)
            # print(self.model(feature))
           
            # print("inferencing....")
            one_feature = self.model(feature.to(torch.float32))
            # lock.acquire()
            
            # print('after intferencing and add.....')
            embed_features.append(np.array(one_feature.detach()))
            # print('infer end.....')
            # self.embed_features.append(self.model(feature))
            id2filesheet[len(embed_features)-1] = item
            # print("go to release.....")

            # lock.release()
            count += 1
            # print(count, len(origin_feature_list))
            # print('emb:', len(embed_features))
            gc.collect()
            del feature
            del origin_feature
            gc.collect()

    def load_embed_features(self):
        # count = 0
        def callback(args):
            print(args)

        def errorcallback(args):
            # 处理子进程错误的函数报错
            # a = 1/0
            print(args)
        print("load embed features......")
        self.embed_features = []
        self.id2filesheet = {}
        procs = []
        batch_len = len(self.mini_batch)/multiprocessing.cpu_count()+1
        temp_batch = []
        temp_id = []

        # pool =  multiprocessing.Pool(processes=multiprocessing.cpu_count())
        # for r in res.get():
        #     print(r)
    

        # print(batch_len)
        # print(len(self.mini_batch))
        # embed_features = multiprocessing.Manager().list()
        # id2filesheet = multiprocessing.Manager().dict()
        # lock = multiprocessing.Manager().Lock()
        # for i in self.mini_batch:
            # temp_batch.append(i)
            # print('temp_batch',i)
            # if len(temp_batch) >= batch_len:
                # res = pool.apply_async(self.batch_get,args=(temp_batch, embed_features, id2filesheet, lock,),callback=callback,error_callback=errorcallback)
                # procs.append(multiprocessing.Process(target=self.batch_get, args=(temp_batch, embed_features, id2filesheet, lock)))
        print("start load feature batch.......")
        self.batch_get_single(self.mini_batch, self.embed_features, self.id2filesheet)
        # temp_batch = []
        print("load feature end_batch.......")
        # self.batch_get_single(temp_batch, self.embed_features, self.id2filesheet)
        # procs.append(multiprocessing.Process(target=self.batch_get, args=(temp_batch, embed_features, id2filesheet, lock)))
        # res = pool.apply_async(self.batch_get,args=(temp_batch, embed_features, id2filesheet, lock,),callback=callback,error_callback=errorcallback)
        temp_batch = []
        
        # for proc in procs:
        #     proc.start()
        # for proc in procs:
        #     proc.join()
        # pool.close()
        # pool.join()
        # self.embed_features = embed_features
        # self.id2filesheet = id2filesheet
        # for batch in batch_train:

        # for item in list(self.all_list):
        #     item_filename, item_sheetname = item.split("---")
        #     item_path = item_filename.split("/")[-1]+"---"+item_sheetname+".json"
        #     with open("../../data/sheet_features_rgb/"+item_path,'r') as f:
        #         origin_feature = json.load(f)
        #     feature = self.transfer_origin_feature(origin_feature)
        #     # print(feature)
        #     feature = torch.FloatTensor(np.array(feature))
        #     feature = feature.reshape(1,100,10,399).permute(0,3,1,2)
        #     feature = Variable(feature)
        #     self.embed_features.append(self.model(feature))
        #     self.id2filesheet[count] = item
        #     count += 1
        #     print(count)
        #     gc.collect()
        #     del feature
        #     del origin_feature
        #     gc.collect()
        temp = []
        # self.embed_features = embed_features
        # print(len(self.embed_features))
        for i in self.embed_features:
            # print('add temp', len(i))
            temp.append(np.array(i[0]))
        self.embed_features = np.array(temp)
        # np.save("embed_features.npy", self.embed_features)
        # np.save("id2filesheet.npy", self.id2filesheet)
    def outer_training(self):
        # self.load_feature(time)
        # self.features =np.load("features.npy")
       
        for epoch in range(self.epoch_nums):
            # for training_index in range(0, len(self.triplet_pair)):
                # self.trained_index = training_index
        
            for time in range(self.inner_times):
                self.get_mini_batch()
                self.load_embed_features()
                # np.save('triplet_pair.npy', self.triplet_pair)
                # np.save("embed_features.npy", self.embed_features)
                # np.save("id2filesheet.npy", self.id2filesheet)
                self.get_mini_batch_triplets()
                self.load_feature(time)
                train_data = TensorDataset(torch.DoubleTensor(self.features))
                train_loader = DataLoader(train_data,batch_size=len(train_data),shuffle=True)
                print("epoach "+str(epoch)+ " batch " + str(time) + " training ....")
                for i,(x) in enumerate(train_loader):
        
                    x = x[0]
                    archor = x[:,0,:].reshape(len(x),100,10,11).permute(0,3,1,2)
                    positive = x[:,1,:].reshape(len(x),100,10,11).permute(0,3,1,2)
                    negative = x[:,2,:].reshape(len(x),100,10,11).permute(0,3,1,2)

                    x1 = Variable(archor) # torch.Size([batch_size, 1000, 10])
                    x2 = Variable(positive) ## torch.Size([batch_size, 1000, 10])
                    x3 = Variable(negative) ## torch.Size([batch_size, 1000, 10])

                    x1 = self.model(x1.to(torch.float32)) # torch.Size([128,10])
                    x2 = self.model(x2.to(torch.float32))
                    x3 = self.model(x3.to(torch.float32))

                    self.loss = self.loss_func(x1,x2,x3)
                    print('loss:', self.loss)
                    self.opt.zero_grad()  # 清空上一步残余更新参数值
                    self.loss.backward() # 误差反向传播，计算参数更新值
                    self.opt.step() # 将参数更新值施加到net的parmeters上
            # torch.save(self.model,'cnn_dynamic_triplet_'+str(time)+"_"+str(epoch))
            torch.save(self.model,'cnn_new_dynamic_triplet_margin_'+str(epoch)+"_"+str(time))
        gc.collect()
        del self.features
        del train_data
        del train_loader
        gc.collect()
    def get_sorted_list(self, auchor, candidate):
        # print('auchor',auchor)
        for id_ in self.id2filesheet:
            if self.id2filesheet[id_] == auchor:
                auchor_index = id_
                # print("auchor id:", id_)

        candidate_emb = []
        now_cand_id2filesheet = {}
        ids = []
        for id_ in self.id2filesheet:
            if self.id2filesheet[id_] in candidate:
                candidate_emb.append(self.embed_features[id_])
                now_cand_id2filesheet[len(candidate_emb)-1] = self.id2filesheet[id_]
                # ids.append(id_)
        # ids = np.array(ids)
        ids = np.array([ind for ind in range(0, len(candidate_emb))])
        
        # temp = []
        # for i in self.embed_features:
        #     if len(i)!=384:
        #         print(len(i))
        # # for i in 
        # gc.collect()
        # del temp
        # gc.collect()
        # print(self.embed_features[0])
        # print(self.embed_features.shape)
        index = faiss.IndexFlatL2(len(candidate_emb[0]))
        index2 = faiss.IndexIDMap(index)
        index2.add_with_ids(np.array(candidate_emb), ids)
        # print(index.is_trained)
        # self.embed_features = np.array(self.embed_features)
        # index.add(self.embed_features)
        # print(index.ntotal)
        search_list = [self.embed_features[auchor_index]]
        D, I = index.search(np.array(search_list), self.topk) # sanity check
        # print(I,D)
        res = []
        for id_ in I[0]:
            # print(id_)
            if id_ == -1:
                continue
            res.append(now_cand_id2filesheet[id_])
        # print("I: ",I)
        # print("D: ",D)
        return res

    # def get_new_triplet_pair(self):
    #     self.triplet_pair = []
    #     self.load_embed_features()
    #     all_sorted_list = {}
    #     added_positive_index = {}
    #     added_negative_index = {}
    #     for auchor in self.auchors:
    #         sorted_list = self.get_sorted_list(auchor)
    #         for index in sorted_list[0]:
    #             all_sorted_list[auchor] = id2filesheet[index]
    #             print(all_sorted_list[auchor])
    #     gc.collect()
    #     del self.embed_features
    #     gc.collect()

    #     count = 0
    #     batch_list = []
    #     for acthor in self.auchors:
    #         if auchor not in added_positive_index:
    #             added_positive_index[auchor] = len(all_sorted_list[auchor])-1
    #         if auchor not in added_negative_index:
    #             added_negative_index[auchor] = 0
    #         positive_file_sheet = all_sorted_list[auchor][added_positive_index[auchor]]
    #         is_found_positive = True
    #         while not self.is_positive(auchor, positive_file_sheet) and added_positive_index[auchor] >= 0:
    #             added_positive_index[auchor] -= 1
    #             positive_file_sheet = all_sorted_list[auchor][added_positive_index[auchor]]
    #         if not self.is_positive(auchor, positive_file_sheet):
    #             is_found_positive = False
            
    #         is_found_negative = True
    #         negative_file_sheet = all_sorted_list[auchor][added_negative_index[auchor]]
    #         while self.is_positive(auchor, positive_file_sheet) and added_negative_index[auchor] < len(all_sorted_list[auchor]):
    #             added_negative_index[auchor] += 1
    #             negative_file_sheet = all_sorted_list[auchor][added_negative_index[auchor]]
    #         if self.is_positive(auchor, positive_file_sheet):
    #             is_found_negative = False
            
    #         if is_found_positive and is_found_negative:
    #             batch_list.append(auchor, positive_file_sheet, negative_file_sheet)
    #         count += 1
    #         if count%5000==0:
    #             self.triplet_pair.append(batch_list)
    #             batch_list = []
    #     self.triplet_pair.append(batch_list)

    # def outer_training(self):
    #     for time in range(0, self.outer_times):
    #         self.get_mini_batch()
    #         self.load_embed_features()
    #         # np.save('triplet_pair.npy', self.triplet_pair)
    #         # np.save("embed_features.npy", self.embed_features)
    #         # np.save("id2filesheet.npy", self.id2filesheet)
    #         self.get_mini_batch_triplets()
    #         # self.triplet_pair = np.load('triplet_pair.npy')
    #         self.inner_training(time)
    #         # self.get_new_triplet_pair()


    def get_mini_batch(self, size=100):
        print("get mini batch........")
        self.mini_batch = []
        self.triplet_pair = []
        self.same_class_in_minibatch = {}
        while len(self.mini_batch) < size:
            sheetname = self.classes[self.sheet_index]
            # print(self.sheetname_2_file_devided[sheetname])
            if len(self.sheetname_2_file_devided[sheetname]) < 1:
                self.class_index = 0
                if self.sheet_index == len(self.classes)-1:
                    self.sheet_index = 0
                else:
                    self.sheet_index += 1
                self.inclass_index = 0
                continue
            if len(self.sheetname_2_file_devided[sheetname][self.class_index]) == 1 or self.inclass_index == self.max_inclass: # only one file in this class
                if self.class_index==len(self.sheetname_2_file_devided[sheetname])-1:
                    if self.sheet_index == len(self.classes)-1:
                        self.sheet_index = 0
                    else:
                        self.sheet_index += 1
                    # break
                    self.class_index = 0
                else:
                    self.class_index+=1
                self.inclass_index=0
                continue
            
            # print(self.sheetname_2_file_devided[sheetname][self.class_index]['filenames'])
            # print(self.inclass_index)
            filename = self.sheetname_2_file_devided[sheetname][self.class_index]['filenames'][self.inclass_index]
            # print(sheetname)
            if not os.path.exists('../../data/sheet_features_rgb/'+filename.split('/')[-1]+"---"+sheetname+'.json'):
                if self.inclass_index == len(self.sheetname_2_file_devided[sheetname][self.class_index]['filenames'])-1:
                    # print("to the end")
                    if self.class_index==len(self.sheetname_2_file_devided[sheetname])-1:
                        if self.sheet_index == len(self.classes)-1:
                            self.sheet_index = 0
                        else:
                            self.sheet_index += 1
                        # break
                        self.class_index = 0
                    else:
                        self.class_index+=1
                    
                    self.inclass_index=0
                else:
                    self.inclass_index += 1
                # print('not exists', '../../data/sheet_features_rgb/'+filename.split('/')[-1]+"---"+sheetname+'.json')
                continue
            
            # print(filename)
            # print(sheetname)
            self.mini_batch.append(filename+"---"+sheetname)
            # print("inclass", self.inclass_index)
            self.same_class_in_minibatch[filename+"---"+sheetname] = []
            # print('min', min(min(len(self.sheetname_2_file_devided[sheetname][self.class_index]['filenames']),self.inclass_index+size-len(self.mini_batch)+1), self.max_inclass))
            for temp_incalss_index in range(self.inclass_index+1, min(min(len(self.sheetname_2_file_devided[sheetname][self.class_index]['filenames']),self.inclass_index+size-len(self.mini_batch)+1), self.max_inclass)):
                # print(temp_incalss_index)
                if os.path.exists('../../data/sheet_features_rgb/'+self.sheetname_2_file_devided[sheetname][self.class_index]['filenames'][temp_incalss_index].split('/')[-1]+"---"+sheetname+'.json'):
                    self.triplet_pair.append([filename+"---"+sheetname, self.sheetname_2_file_devided[sheetname][self.class_index]['filenames'][temp_incalss_index]+"---"+sheetname])
                    self.same_class_in_minibatch[filename+"---"+sheetname].append(self.sheetname_2_file_devided[sheetname][self.class_index]['filenames'][temp_incalss_index]+"---"+sheetname)
                    if self.sheetname_2_file_devided[sheetname][self.class_index]['filenames'][temp_incalss_index]+"---"+sheetname not in self.same_class_in_minibatch:
                        self.same_class_in_minibatch[self.sheetname_2_file_devided[sheetname][self.class_index]['filenames'][temp_incalss_index]+"---"+sheetname] = []
                    self.same_class_in_minibatch[self.sheetname_2_file_devided[sheetname][self.class_index]['filenames'][temp_incalss_index]+"---"+sheetname].append(filename+"---"+sheetname)
            # print(len(self.sheetname_2_file_devided[sheetname][self.class_index]['filenames'])
            # print(self.inclass_index)
            # print(self.triplet_pair)
            if self.inclass_index == min(len(self.sheetname_2_file_devided[sheetname][self.class_index]['filenames'])-1, self.max_inclass-1):
                # print("to the end")
                if self.class_index==len(self.sheetname_2_file_devided[sheetname])-1:
                    if self.sheet_index == len(self.classes)-1:
                        self.sheet_index = 0
                    else:
                        self.sheet_index += 1
                    # break
                    self.class_index = 0
                else:
                    self.class_index+=1
                
                self.inclass_index=0
            else:
                self.inclass_index += 1

    def get_negative_case(self, data, sorted_list_index, sorted_list_dict, res, lock):
        # index = data[:,0]
        # triplet = data[:1]
        # print(index, triplet)
        # if len(triplet) == 3:
        #     continue
        # print(data)
        
        for item in data:
        # print(self.id2filesheet)
            index = item[0]
            triplet = item[1]
            # print("###########")
            # print('index', index)
            auchor = triplet[0]
            # print(triplet)
            # print(triplet[1] in self.mini_batch)
            # print(id2filesheet[])
            

            is_add = False
            # print("found pos and auchor")
            lock.acquire()
            # print(self.triplet_pair[index])
            # print(sorted_list_dict.keys())
            # print(sorted_list_dict[auchor])
            # print(sorted_list_index[auchor])
            # print(sorted_list_dict[auchor][sorted_list_index[auchor]])
            negative_id = sorted_list_dict[auchor][sorted_list_index[auchor]]
            # negative_distance = sorted_list_dict[auchor][sorted_list_index[auchor]]
            for feature_id in self.id2filesheet.keys():
                # print(self.id2filesheet[feature_id])
                # print('auchor', auchor)
                if self.id2filesheet[feature_id] == auchor:
                    auchor_index = feature_id
                    # print('found auchor index')
                if self.id2filesheet[feature_id] == triplet[1]:
                    positive_index = feature_id
                if self.id2filesheet[feature_id] == negative_id:
                    negative_index = feature_id
                    # print('found positive index')
            pos_distance = np.linalg.norm(self.embed_features[auchor_index]-self.embed_features[positive_index])

            neg_distance = np.linalg.norm(self.embed_features[auchor_index]-self.embed_features[negative_index])
            print('distance', pos_distance, neg_distance, auchor.split("---")[-1],triplet[1].split("---")[-1],negative_id.split("---")[-1])
            temp = self.triplet_pair[index]
            temp.append(sorted_list_dict[auchor][sorted_list_index[auchor]])
            res.append(temp)
            # print('get negative case', len(res), len(self.triplet_pair))
            if sorted_list_index[auchor] == len(sorted_list_dict[auchor])-1:
                sorted_list_index[auchor] = 0
            else:
                sorted_list_index[auchor] += 1
            # print(temp)
            lock.release()

    def get_mini_batch_triplets(self):
        sorted_list_dict = {}
        sorted_list_index = {}
        # print(len(self.id2filesheet))
        print("get mini batch triplets......")
        for triplet in self.triplet_pair: #### get all auchor sorted list
            auchor = triplet[0]
            if auchor in sorted_list_dict:
                continue
            candidate = []
            for item in self.mini_batch:
                if item not in self.same_class_in_minibatch[auchor] and item.split('---')[1] != auchor.split("---")[1]:
                    candidate.append(item)
                    # print(item.split('---')[1],  auchor.split("---")[1])
            sorted_list = self.get_sorted_list(auchor, candidate)
            sorted_list_dict[auchor] = sorted_list
            sorted_list_index[auchor] = 0
        # np.save('sorted_list_dict.npy',sorted_list_dict)
        # np.save("sorted_list_index.npy",sorted_list_index)
        # sorted_list_dict = np.load("sorted_list_dict.npy", allow_pickle=True).item()
        # sorted_list_index = np.load("sorted_list_index.npy", allow_pickle=True).item()
        temp_batch = []
        procs = []
        res = multiprocessing.Manager().list()
        # triplet_pair = multiprocessing.Manager().list()
        batch_len = len(self.triplet_pair)/multiprocessing.cpu_count()+1
        lock = multiprocessing.Lock()
        for index, triplet in enumerate(self.triplet_pair):
            temp_batch.append((index,triplet))
            if len(temp_batch) >= batch_len:
                procs.append(multiprocessing.Process(target=self.get_negative_case, args=(temp_batch, sorted_list_index, sorted_list_dict, res, lock, )))
                temp_batch = []
        procs.append(multiprocessing.Process(target=self.get_negative_case, args=(temp_batch, sorted_list_index, sorted_list_dict, res, lock, )))
        temp_batch = []
        
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
        self.triplet_pair = res
        np.save('bert_triplet_pair.npy', self.triplet_pair)
        # for index,triplet in enumerate(self.triplet_pair):
        #     print(index, triplet)
        #     if len(triplet) == 3:
        #         continue
        #     auchor = triplet[0]
        #     for id_ in self.id2filesheet:
        #         if self.id2filesheet[id_] == auchor:
        #             auchor_index = id_
        #         if self.id2filesheet[id_] == triplet[1]:
        #             positive_index = id_
        #     pos_distance = np.linalg.norm(self.embed_features[auchor_index]-self.embed_features[positive_index])
        #     # index_list1 = list(range(sorted_list_index[auchor], len(sorted_list_dict[auchor])))
        #     # index_list2 = list(range(0, sorted_list_index[auchor]))
        #     # index_list = []
        #     # index_list += index_list1
        #     # index_list += index_list2
        #     is_add = False
        #     print("found pos and auchor")
        #     # print(len(index_list))
        #     # for nega_index in index_list:
        #     #     neg = sorted_list_dict[auchor][nega_index]
        #     #     for id_ in self.id2filesheet:
        #     #         if self.id2filesheet[id_] == neg:
        #     #             neg_index = id_
        #     #             break
        #     #     print('xxxxxxxxxx')
        #     #     neg_distance = np.linalg.norm(self.embed_features[auchor_index]-self.embed_features[neg_index])
        #     #     if neg_distance < pos_distance:
        #     #         self.triplet_pair[index].append(neg)
        #     #         sorted_list_index[auchor] = nega_index +1
        #     #         is_add = True
        #     #         break
        #     # print('found neg')
        #     # if not is_add:
        #     # print(sorted_list_index[auchor])
        #     self.triplet_pair[index].append(sorted_list_dict[auchor][sorted_list_index[auchor]])
        #     if sorted_list_index[auchor] == len(sorted_list_dict[auchor])-1:
        #         sorted_list_index[auchor] = 0
        #     else:
        #         sorted_list_index[auchor] += 1
    def test_on_one_model(self, model_path, test_data):
        distance1 = []
        distance2 = []
        cos1 = []
        cos2 = []
        model = torch.load(model_path)
        new_test_data = []
        print(test_data.shape)
        # for test_x in test_data:
        #     print(np.array([test_x[0]]).shape)
        #     new_test_data.append([[test_x[0]], [test_x[1]], [test_x[2]]])
        # new_test_data = np.array(new_test_data)
        # print(new_test_data.shape)
        test_data = TensorDataset(torch.DoubleTensor(test_data))
        test_loader = DataLoader(test_data,batch_size=len(test_data))
        for test_x in test_loader:
            test_x = test_x[0]
            print('len testx', len(test_x))
            print(test_x.shape)
            # print(test_x.shape)
            print(test_x[:,0,:].shape)
            # print(test_x[0].shape)
            # print(test_x[:,0,:].reshape(len(test_x),100,10,399))
            test_x1 =test_x[:,0,:].reshape(len(test_x),100,10,399).permute(0,3,1,2)
            test_x2 = test_x[:,1, :].reshape(len(test_x),100,10,399).permute(0,3,1,2)
            test_x3 =  test_x[:,2,:].reshape(len(test_x),100,10,399).permute(0,3,1,2)
            test_x1 = Variable(test_x1).to(torch.float32) # torch.Size([batch_size, 1000, 10])
            test_x2 = Variable(test_x2).to(torch.float32) ## torch.Size([batch_size, 1000, 10])
            test_x3 = Variable(test_x3).to(torch.float32)
            # print(test_x1.type)
            # test_x1 = test_x1.double()
            # test_x2 = test_x2.double()
            # test_x3 = test_x3.double()
            print(test_x1)
            test_x1 = model(test_x1).detach().numpy() # torch.Size([128,10])
            test_x2 = model(test_x2).detach().numpy()
            test_x3 = model(test_x3).detach().numpy()
        
            

            tptn = 0
            tpfn = 0
            fptn = 0
            fpfn = 0
            best = 0
            hard = 0
            low = 0
            for index, emb in enumerate(test_x1):
                # cos1.append(test_x1[index].dot(test_x2[index]))
                # cos2.append(test_x1[index].dot(test_x3[index]))
                distance1.append(np.linalg.norm(test_x1[index]-test_x2[index]))
                distance2.append(np.linalg.norm(test_x1[index]-test_x3[index]))
            # out = [for index, emb]
            for index, dis1 in enumerate(distance1):
                # if cos1[index] >= 0.5 and cos2[index] < 0.5:
                #     tptn+=1
                # elif cos1[index] >= 0.5 and cos2[index] >= 0.5:
                #     tpfn +=1
                # elif cos1[index] < 0.5 and cos2[index] < 0.5:
                #     fptn += 1
                # else:
                #     fpfn +=1
                if distance1[index] + 1 < distance2[index]:
                    best += 1
                elif distance1[index] + 1 >= distance2[index] and distance1[index] < distance2[index]:
                    hard += 1
                else:
                    low += 1
            

                
            # print('best:\t',best)
            # print('hard:\t',hard)
            # print('low:\t',low)
            
            # print('tptn:\t',tptn)
            # print('tpfn:\t',tpfn)
            # print('fptn:\t',fptn)
            # print('fpfn:\t',fpfn)
            break
        return distance1, distance2

    def create_features(self):
        # with open("../AnalyzeDV/CNN_training_origin_dict_filter.json", 'r', encoding='utf-8') as f:
        #     cnn_feature_0 = json.load(f)
        # with open("../AnalyzeDV/CNN_training_origin_dict_filter_1.json", 'r', encoding='utf-8') as f:
        #     cnn_feature_1 = json.load(f)
        # with open("../AnalyzeDV/CNN_training_origin_dict_filter_2.json", 'r', encoding='utf-8') as f:
        #     cnn_feature_2 = json.load(f)
        with open("../AnalyzeDV/positive_training_origin_dict.json", 'r', encoding='utf-8') as f:
            cnn_feature_2 = json.load(f)
        result = {}
        sheet_2_id = {}
        id_=1
        id_2_feature = {}
        # print(list(cnn_feature_0.keys())[0])
        # print(cnn_feature_0[list(cnn_feature_0.keys())[0]][0]['sheetfeature'][0].keys())
        # all_count = len(cnn_feature_0)+ len(cnn_feature_1)+ len(cnn_feature_2)
        all_count = len(cnn_feature_2)
        count = 1
        for cnn_feature in [cnn_feature_2]:
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
                    id_2_feature[id_] = self.get_feature_vector(feature)
                    if sheetname not in sheet_2_id:
                        sheet_2_id[sheetname] = {}
                    sheet_2_id[sheetname][filename] = id_
                    id_+=1

        # print(list(cnn_feature.keys())[0])
        # print(cnn_feature[list(cnn_feature.keys())[0]][0])
        with open("json_data/dynamic_cnn_features_origin_dict.json", 'w') as f:
            json.dump(result, f)
        with open("json_data/dynamic_sheet_2_id.json", 'w') as f:
            json.dump(sheet_2_id, f)
        with open("json_data/dynamic_id_2_feature.json", 'w') as f:
            json.dump(id_2_feature, f)

    def get_triplet_feature(self):
        with open("json_data/triplet_pair.json", 'r') as f:
            triplet_pair = json.load(f)
        with open("json_data/dynamic_id_2_feature.json", 'r') as f:
            id_2_feature = json.load(f)
        with open("json_data/dynamic_sheet_2_id.json", 'r') as f:
            sheet_2_id = json.load(f)
        res = []
        id_list = []
        positive_num=0

        count=1
        for triplet in triplet_pair:
            filename1, sheetname1 = triplet[0].split('---')
            filename2, sheetname2 = triplet[1].split('---')
            filename3, sheetname3 = triplet[2].split('---')
            # filename3 = filename3[0:10]+"/UnzipData"+filename3[10:]
            print('----------')
            print(list(sheet_2_id[list(sheet_2_id.keys())[0]].keys())[0])
            print(filename1, filename2, filename3)
            print(sheetname1, sheetname2, sheetname3)
            if sheetname1 not in sheet_2_id or sheetname2 not in sheet_2_id or sheetname3 not in sheet_2_id:
                print("not have sheet")
                continue
            if filename1 not in sheet_2_id[sheetname1] or filename2 not in sheet_2_id[sheetname2] or filename3 not in sheet_2_id[sheetname3]:
                print("not have file")
                continue
            feature1 = id_2_feature[str(sheet_2_id[sheetname1][filename1])]
                # feature1.append(sheet_2_id[sheetname][pair[0]])
            feature2 = id_2_feature[str(sheet_2_id[sheetname2][filename2])]
            feature3 = id_2_feature[str(sheet_2_id[sheetname3][filename3])]
            # feature2.append(sheet_2_id[sheetname][pair[1]])
            res.append((feature1, feature2, feature3))
            id_list.append((sheet_2_id[sheetname1][filename1], sheet_2_id[sheetname2][filename2],sheet_2_id[sheetname3][filename3] ))
            # break
            positive_num+=1

        print(positive_num)
        with open("json_data/dynamic_triplet_features.json",'w') as f:
            json.dump(res, f)
        with open("json_data/dynamic_triplet_id.json",'w') as f:
            json.dump(id_list, f)
    def batch_testing(self):
        all_suc_num = 0
        all_data = 0
        print('load data.......')
        # self.create_features()
        # self.get_triplet_feature()
        # with open("triplet_features.json",'r') as f:
        #     triplet_features = json.load(f)
        with open("json_data/cnn_triplet_x_1.json", 'r') as f:
            x1 = json.load(f)
        with open("json_data/cnn_triplet_x_2.json", 'r') as f:
            x2 = json.load(f)
        with open("json_data/cnn_triplet_x_3.json", 'r') as f:
            x3 = json.load(f)
        with open("json_data/cnn_triplet_x_4.json", 'r') as f:
            x4 = json.load(f)

        print('load data end.......')
        # x = [x1,x2,x3,x4]
        # model_pathes = []
        filelist = os.listdir(".")
        model_pathes = []
        for i in range(0,5):
            for j in range(0,2):
                model_path = "bert_cnn_dynamic_triplet_margin_20_0_0" + str(i) + "_" + str(j)
            # if "cnn_dynamic_triplet_" + str(i):
            #     model_pathes.append(i)
        # for model_path in model_pathes:
            # x_test = x[batch-1]
            # all_data+=len(x_test)
            # x_train_list = []

            # for index,train_batch in enumerate(list(set([1,2,3,4])-set([batch]))):
            #     x_train_sub = x[train_batch-1]
            #     x_train_list.append(x_train_sub)

                triplet_features = np.concatenate((x1,x2, x3, x4))
                triplet_features = np.array(triplet_features)
                distance1,distance2= self.test_on_one_model(model_path, triplet_features)
                np.save("distance1_bert_"+str(i) + '_' + str(j) +'.npy', distance1)
                np.save("distance2_bert_"+str(i) + '_' + str(j) +'.npy', distance2)
            # accuracy, suc_num = training_testing_cosine(x_train, x_test,batch)
        #     print('accuracy '+str(batch)+':\t',accuracy.mean())
        #     all_suc_num+=suc_num
        # print('suc_num', suc_num)
def get_roc_curve():
    y_triplet_score = []
    y_triplet_test = []
    y_cos_score = []
    y_cos_test = []
    y_score = []
    y_test = []


    distance1_1 = np.load('distance1_1.npy', allow_pickle=True)
    distance1_2 = np.load('distance1_2.npy', allow_pickle=True)
    distance1_3 = np.load('distance1_3.npy', allow_pickle=True)
    distance1_4 = np.load('distance1_4.npy', allow_pickle=True)
    
    distance2_1 = np.load('distance2_1.npy', allow_pickle=True)
    distance2_2 = np.load('distance2_2.npy', allow_pickle=True)
    distance2_3 = np.load('distance2_3.npy', allow_pickle=True)
    distance2_4 = np.load('distance2_4.npy', allow_pickle=True)

    pred1 = np.load('pred_100000_0.npy', allow_pickle=True)
    pred2 = np.load('pred_100000_1.npy', allow_pickle=True)
    pred3 = np.load('pred_100000_2.npy', allow_pickle=True)
    pred4 = np.load('pred_100000_3.npy', allow_pickle=True)
    test1 = np.load('test_100000_0.npy', allow_pickle=True)
    test2 = np.load('test_100000_1.npy', allow_pickle=True)
    test3 = np.load('test_100000_2.npy', allow_pickle=True)
    test4 = np.load('test_100000_3.npy', allow_pickle=True)

    # pred_cos1 = np.load('pred_1_cos_1.npy', allow_pickle=True)
    # pred_cos2 = np.load('pred_1_cos_2.npy', allow_pickle=True)
    # pred_cos3 = np.load('pred_1_cos_3.npy', allow_pickle=True)
    # pred_cos4 = np.load('pred_1_cos_4.npy', allow_pickle=True)
    # test_cos1 = np.load('test_1_cos_1.npy', allow_pickle=True)
    # test_cos2 = np.load('test_1_cos_2.npy', allow_pickle=True)
    # test_cos3 = np.load('test_1_cos_3.npy', allow_pickle=True)
    # test_cos4 = np.load('test_1_cos_4.npy', allow_pickle=True)


    for distance1 in [distance1_1, distance1_2, distance1_3, distance1_4]:
        for distance in distance1:
            y_triplet_score.append(0-distance)
            y_triplet_test.append(1)

    for distance2 in [distance2_1, distance2_2, distance2_3, distance2_4]:
        for distance in distance2:
            y_triplet_score.append(0-distance)
            y_triplet_test.append(0)

    test = [test1, test2, test3,test4]
    for index,pred in enumerate([pred1, pred2, pred3, pred4]):
        for index1, item in enumerate(pred):
            y_score.append(item)
            y_test.append(test[index][index1])

    # test = [test_cos1, test_cos2, test_cos3,test_cos4]
    # for index,pred in enumerate([pred_cos1, pred_cos2, pred_cos3, pred_cos4]):
    #     for index1, item in enumerate(pred):
    #         y_cos_score.append(item)
    #         y_cos_test.append(test[index][index1])

    fpr0,tpr0,threshold = roc_curve(y_triplet_test,y_triplet_score) ###计算真正率和假正率
    roc_auc0 = auc(fpr0,tpr0) ###计算auc的值
    fpr1,tpr1,threshold1 = roc_curve(y_test,y_score) ###计算真正率和假正率
    roc_auc1 = auc(fpr1,tpr1) ###计算auc的值
    # fpr2,tpr2,threshold2 = roc_curve(y_cos_test,y_cos_score) ###计算真正率和假正率
    # roc_auc2 = auc(fpr2,tpr2) ###计算auc的值
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    for epoch in range(0,2):
        for time in range(0,19):
            y_triplet_score = []
            y_triplet_test = []
            distance1 = np.load('distance1_'+str(time)+'_'+ str(epoch)+'.npy', allow_pickle=True)
            distance2 = np.load('distance2_'+str(time)+'_'+ str(epoch)+'.npy', allow_pickle=True)
            for distance in distance1:
                y_triplet_score.append(0-distance)
                y_triplet_test.append(1)

            for distance in distance2:
                y_triplet_score.append(0-distance)
                y_triplet_test.append(0)

            fpr,tpr,threshold = roc_curve(y_triplet_test,y_triplet_score) ###计算真正率和假正率
            roc_auc = auc(fpr,tpr) ###计算auc的值
            plt.plot(fpr, tpr,
            lw=lw, label=str(epoch)+"_"+str(time)+'triplet ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线

    plt.plot(fpr0, tpr0,
            lw=lw, label='triplet ROC curve (area = %0.3f)' % roc_auc0) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot(fpr1, tpr1,
            lw=lw, label='ROC curve (area = %0.3f)' % roc_auc1) ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot(fpr2, tpr2, color='green',
    #         lw=lw, label='cos ROC curve (area = %0.3f)' % roc_auc2) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of Sheet Embedding Models')
    plt.legend(loc="lower right")

    plt.savefig("triplet_dymatic_loss_roc.png")
            
def get_threshold_p_r():
    y_triplet_score = []
    y_triplet_test = []
    y_triplet_pred = []
    distance1 = np.load('distance1_'+str(1)+'_'+ str(1)+'.npy', allow_pickle=True)
    distance2 = np.load('distance2_'+str(1)+'_'+ str(1)+'.npy', allow_pickle=True)
    for distance in distance1:
        y_triplet_score.append(0-distance)
        y_triplet_test.append(1)
        

    for distance in distance2:
        y_triplet_score.append(0-distance)
        y_triplet_test.append(0)
        # if distance <= 1:
        #     y_triplet_pred.append(1)
        # else:
        #     y_triplet_pred.append(0)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    fpr,tpr,threshold = roc_curve(y_triplet_test,y_triplet_score) ###计算真正率和假正率
    max_ = 0
    max_thresh = 0
    min_recall = 0.93
    for i in range(tpr.shape[0]):
    # if tpr[i] > _recall:
        # print("###########")
        # print(tpr[i] - fpr[i], max_)

        if tpr[i] - fpr[i] > max_:
            max_ = tpr[i] - fpr[i]  
            max_thresh = threshold[i]
            print(tpr[i] - fpr[i], max_)
            print(tpr[i], 1-fpr[i], threshold[i])
    #     # break
    print(max_thresh)
    # y_dymatic_1_1_pred = []
    for distance in distance1:
        if distance <= 0-max_thresh:
            y_triplet_pred.append(1)
        else:
            y_triplet_pred.append(0)
    for distance in distance2:
        if distance <= 0-max_thresh:
            y_triplet_pred.append(1)
        else:
            y_triplet_pred.append(0)
    # np.save("y_dymatic_1_1_pred_recall_0.93.npy", y_triplet_pred)
    # np.save("y_dymatic_1_1_test_recall_0.93.npy", y_triplet_test)
    print(y_triplet_pred)
    print(y_triplet_test)
    for index in range(len(y_triplet_pred)):
        if y_triplet_pred[index] == y_triplet_test[index] and y_triplet_test[index] == 1:
            tp += 1
        if y_triplet_pred[index] == y_triplet_test[index] and y_triplet_test[index] == 0:
            tn += 1
        if y_triplet_pred[index] != y_triplet_test[index] and y_triplet_test[index] == 1:
            fn += 1
        if y_triplet_pred[index] != y_triplet_test[index] and y_triplet_test[index] == 0:
            fp += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print('precision:', precision)
    print('recall:',recall)
    print("f1:", precision*recall*2/(precision+recall))

# def look_color():
#     with open("json_data/content_dict.json", 'r') as f:
#         content_dict = json.load(f)
#     sheetname = '3(B) HPE'
#     filename1 = '../../data/UnzipData/028/c42f7f94cc995ee27effe4938fc00247_d3d3Lmpudndhc2hpbS5nb3YuaW4JMTgyLjE4LjE0OS4xODQ=.xls.xlsx'
#     filename2 = '../../data/UnzipData/027/bb349f1609acaf8215b489dc95f0b1ac_am52Ym9sYW5naXIubmljLmluCTE2NC4xMDAuNTIuODc=.xls.xlsx'
#     with open("json_data/positive_id_2_feature.json", 'r') as f:
#         id_2_feature = json.load(f)
#     with open("json_data/positive_sheet_2_id.json", 'r') as f:
#         sheet_2_id = json.load(f)

#     # id1 = sheet_2_id[sheetname][filename1]
#     # id2 = sheet_2_id[sheetname][filename2]

#     for item in sheet_2_id:
#         for item1 in sheet_2_id[item]:
#             print("###############")
#             print(id_2_feature[str(sheet_2_id[item][item1])][0])
#     # print(id_2_feature.keys())
#     color_metric_1 = []
#     feature1 = np.array(id_2_feature[str(id1)])
#     # print(np.array(feature1).shape)
    
#     feature2 = np.array(id_2_feature[str(id2)])
#     # print(feature1[:,6])
#     # print(feature2[:,6])
#     print(feature1[0])
#     content_1 = []
#     for item in np.array(feature1)[:,9]:
#         for sentence in content_dict.keys():
#             if content_dict[sentence] == item:
#                 content_1.append(sentence)
#     content_2 = []
#     for item in np.array(feature2)[:,9]:
#         for sentence in content_dict.keys():
#             if content_dict[sentence] == item:
#                 content_2.append(sentence)
#     # print(np.array(feature1)[:,8])
#     # print(np.array(feature2)[:,8])
#     content_1 = np.array(content_1).reshape(100,10)
#     content_2 = np.array(content_1).reshape(100,10)
#     print(content_1[1])
#     print(content_2[1])
#     # pprint.pprint(np.array(feature1)[:,8]-np.array(feature2)[:,8])

def get_threshold_p_r_all():
    y_triplet_score = []
    y_triplet_test = []
    y_triplet_pred = []
    distance1 = np.load('distance1_'+str(1)+'_'+ str(1)+'.npy', allow_pickle=True)
    distance2 = np.load('distance2_'+str(1)+'_'+ str(1)+'.npy', allow_pickle=True)
    for distance in distance1:
        y_triplet_score.append(0-distance)
        y_triplet_test.append(1)
        

    for distance in distance2:
        y_triplet_score.append(0-distance)
        y_triplet_test.append(0)
        # if distance <= 1:
        #     y_triplet_pred.append(1)
        # else:
        #     y_triplet_pred.append(0)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    fpr,tpr,threshold = roc_curve(y_triplet_test,y_triplet_score) ###计算真正率和假正率
    # max_ = 0
    # max_thresh = 0
    # for i in range(tpr.shape[0]):
    # # if tpr[i] > _recall:
    #     # print("###########")
    #     # print(tpr[i] - fpr[i], max_)

    #     if tpr[i] - fpr[i] > max_:
    #         max_ = tpr[i] - fpr[i]  
    #         max_thresh = threshold[i]
    #         print(tpr[i] - fpr[i], max_)
    #         print(tpr[i], 1-fpr[i], threshold[i])
    # #     # break
    # print(max_thresh)
    # y_dymatic_1_1_pred = []
    max_f1 =0
    max_precision = 0
    max_recall = 0
    max_thresh = 0
    max_y_triplet_pred = []
    max_y_triplet_test = []
    for thresh in threshold:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        y_triplet_pred=[]
        for distance in distance1:
            if distance <= 0-thresh:
                y_triplet_pred.append(1)
            else:
                y_triplet_pred.append(0)
        for distance in distance2:
            if distance <= 0-thresh:
                y_triplet_pred.append(1)
            else:
                y_triplet_pred.append(0)
        
        # print(y_triplet_pred)
        # print(y_triplet_test)
        for index in range(len(y_triplet_pred)):
            if y_triplet_pred[index] == y_triplet_test[index] and y_triplet_test[index] == 1:
                tp += 1
            if y_triplet_pred[index] == y_triplet_test[index] and y_triplet_test[index] == 0:
                tn += 1
            if y_triplet_pred[index] != y_triplet_test[index] and y_triplet_test[index] == 1:
                fn += 1
            if y_triplet_pred[index] != y_triplet_test[index] and y_triplet_test[index] == 0:
                fp += 1
        try:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
        except:
            continue
        print('precision:', precision)
        print('recall:',recall)
        print("f1:", precision*recall*2/(precision+recall))
        if precision*recall*2/(precision+recall) > max_f1:
            max_f1 = precision*recall*2/(precision+recall)
            max_precision = precision
            max_recall = recall
            max_thresh = thresh
            
    print('max_f1:', max_f1)
    print('max_precision:',max_precision)
    print("max_recall:", max_recall)
    print("max_thresh:", max_thresh)
    
def get_threshold_p_r_all_not_dy():
    y_triplet_score = []
    y_triplet_test = []
    y_triplet_pred = []
    # distance1_1 = np.load('distance1_1.npy', allow_pickle=True)
    # distance2_1 = np.load('distance2_1.npy', allow_pickle=True)
    # distance1_2 = np.load('distance1_2.npy', allow_pickle=True)
    # distance2_2 = np.load('distance2_2.npy', allow_pickle=True)
    # distance1_3 = np.load('distance1_3.npy', allow_pickle=True)
    # distance2_3 = np.load('distance2_3.npy', allow_pickle=True)
    distance1_4 = np.load('distance1_4.npy', allow_pickle=True)
    distance2_4 = np.load('distance2_4.npy', allow_pickle=True)
    # for distance1 in [distance1_1, distance1_2, distance1_3, distance1_4]:
    for distance1 in [distance1_4]:
        for distance in distance1:
            y_triplet_score.append(0-distance)
            y_triplet_test.append(1)
        
    # for distance2 in [distance2_1, distance2_2, distance2_3, distance2_4]:
    for distance2 in [distance2_4]:
        for distance in distance2:
            y_triplet_score.append(0-distance)
            y_triplet_test.append(0)
        # if distance <= 1:
        #     y_triplet_pred.append(1)
        # else:
        #     y_triplet_pred.append(0)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    fpr,tpr,threshold = roc_curve(y_triplet_test,y_triplet_score) ###计算真正率和假正率
    # max_ = 0
    # max_thresh = 0
    # for i in range(tpr.shape[0]):
    # # if tpr[i] > _recall:
    #     # print("###########")
    #     # print(tpr[i] - fpr[i], max_)

    #     if tpr[i] - fpr[i] > max_:
    #         max_ = tpr[i] - fpr[i]  
    #         max_thresh = threshold[i]
    #         print(tpr[i] - fpr[i], max_)
    #         print(tpr[i], 1-fpr[i], threshold[i])
    # #     # break
    # print(max_thresh)
    # y_dymatic_1_1_pred = []
    max_f1 =0
    max_precision = 0
    max_recall = 0
    max_thresh = 0
    max_y_triplet_pred = []
    max_y_triplet_test = []
    for thresh in threshold:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        y_triplet_pred=[]
        for distance1 in [distance1_4]:
            for distance in distance1:
                if distance <= 0-thresh:
                    y_triplet_pred.append(1)
                else:
                    y_triplet_pred.append(0)
        for distance2 in [distance2_4]:
            for distance in distance2:
                if distance <= 0-thresh:
                    y_triplet_pred.append(1)
                else:
                    y_triplet_pred.append(0)
        # np.save("y_dymatic_1_1_pred.npy", y_triplet_pred)
        # np.save("y_dymatic_1_1_test.npy", y_triplet_test)
        # print(y_triplet_pred)
        # print(y_triplet_test)
        for index in range(len(y_triplet_pred)):
            if y_triplet_pred[index] == y_triplet_test[index] and y_triplet_test[index] == 1:
                tp += 1
            if y_triplet_pred[index] == y_triplet_test[index] and y_triplet_test[index] == 0:
                tn += 1
            if y_triplet_pred[index] != y_triplet_test[index] and y_triplet_test[index] == 1:
                fn += 1
            if y_triplet_pred[index] != y_triplet_test[index] and y_triplet_test[index] == 0:
                fp += 1
        try:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
        except:
            continue
        print('precision:', precision)
        print('recall:',recall)
        print("f1:", precision*recall*2/(precision+recall))
        if precision*recall*2/(precision+recall) > max_f1:
            max_f1 = precision*recall*2/(precision+recall)
            max_precision = precision
            max_recall = recall
            max_thresh = thresh
            max_y_triplet_pred = y_triplet_pred
            max_y_triplet_test = y_triplet_test
    print('max_f1:', max_f1)
    print('max_precision:',max_precision)
    print("max_recall:", max_recall)
    print("max_thresh:", max_thresh)
    np.save("y_triplet_loss_pred.npy", max_y_triplet_pred)
    np.save("y_triplet_loss_test.npy", max_y_triplet_test)
def error_analyze():
    # with open("positive_feature.json",'r') as f:
    #     positive_feature = json.load(f)
    # with open("triplet_features.json",'r') as f:
    #     triplet_features = json.load(f)
    # with open("triplet_id.json",'r') as f:
    #     triplet_id = json.load(f)
    # with open("negative_id.json",'r') as f:
    #     negative_id = json.load(f)

    with open("json_data/cnn_triplet_x_1.json", 'r') as f:
        features1 = json.load(f)
    with open("json_data/cnn_triplet_x_2.json", 'r') as f:
        features2 = json.load(f)
    with open("json_data/cnn_triplet_x_3.json", 'r') as f:
        features3 = json.load(f)
    with open("json_data/cnn_triplet_x_4.json", 'r') as f:
        features4 = json.load(f)
    with open("json_data/cnn_triplet_index_1.json", 'r') as f:
        index1 = json.load(f)
    with open("json_data/cnn_triplet_index_2.json", 'r') as f:
        index2 = json.load(f)
    with open("json_data/cnn_triplet_index_3.json", 'r') as f:
        index3 = json.load(f)
    with open("json_data/cnn_triplet_index_4.json", 'r') as f:
        index4 = json.load(f)
    distance1_1_1 = np.load("distance1_"+str(1) + '_' + str(1) +'.npy')
    distance2_1_1 = np.load("distance2_"+str(1) + '_' + str(1) +'.npy')
    features = np.concatenate((features1, features2, features3, features4))
    indexs = np.concatenate((index1, index2, index3, index4))
    with open("json_data/id_2_feature.json", 'r') as f:
        id_2_feature = json.load(f)
    with open("json_data/sheet_2_id.json", 'r') as f:
        sheet_2_id = json.load(f)


    # features = np.load('cnn_x_4.npy', allow_pickle=True)
    pred_1 = np.load('json_data/y_dymatic_1_1_pred.npy', allow_pickle=True)
    test_1 = np.load('json_data/y_dymatic_1_1_test.npy', allow_pickle=True)
    # pred_1 = np.load('y_triplet_loss_pred.npy', allow_pickle=True)
    # test_1 = np.load('y_triplet_loss_test.npy', allow_pickle=True)
    # print(pred_1[0:10])
    # print(test_1[0:10])
    # re = []
    # for index in range(len(pred_1)):
    #     if test_1[index] == 0 and pred_1[index] == test_1[index]:
    #         re.append(True)
    #     elif test_1[index] == 0:
    #         re.append(False)
    re = pred_1 == test_1
            # re.append(False)
    # print(len(pred_1))
    # print(len(features))
    # print(positive_id)
    # print(len(re))
    # print(len(features))
    sheet_pair_dict = {}
    str_ = ''
    all_ = 0
    failed = 0
    for index, res in enumerate(re):
        # print(res)
        # print(index%len(re))
        all_ += 1
        if(res==True):
            continue
        # print(len(indexs))
    
        # print(index%len(indexs))
        # print("false res")
        failed += 1
        # if 
    # print(all_)
    # print(failed)
        # print("res==False")
        # print(index1[index])
        # print(indexs[index%len(features)])
        auchor_id = indexs[index%len(indexs)][0]
        positive_id = indexs[index%len(indexs)][1]
        negative_id = indexs[index%len(indexs)][2]
        # one_feature = features[index%len(re)]
        # found=False
        # print(feature.shape)
        # auchor = one_feature[0]
        
        # positive = one_feature[1]
        # negative = one_feature[2]
        # for index,feature in enumerate(triplet_features):
        #     # print(feature.shape)
        #     # print(auchor.shape)
        #     if (feature == auchor).all():
        #         found_index_auchor=True
        #         auchor_id = triplet_id[index]
        #         continue
        #     if (feature == positive).all():
        #         found_index_positive=True
        #         positive_id = triplet_id[index]
        #         continue
        #     if (feature == negative).all():
        #         found_index_negative=True
        #         negative_id = triplet_id[index]
                # continue
        # for id_ in id_2_feature:
        #     if (id_2_feature[id_] == auchor).all():
        #         auchor_id = id_
        #         print("found auchor")
        #     if (id_2_feature[id_] == positive).all():
        #         positive_id = id_
        #         print("found positive_id")
        #     if (id_2_feature[id_] == negative).all():
        #         negative_id = id_
        #         print("found negative_id")
        if index < len(features):
            str_ += "pos#########################\n"
        else:
            str_ += "neg#########################\n"
        for sheetname in sheet_2_id:
            for filename in sheet_2_id[sheetname]:
                if auchor_id == sheet_2_id[sheetname][filename]:
                    sn1 = sheetname
                    f1 = filename
                    str_ = str_ + "auchor:" +str(sheetname) +',' + str(filename) + '\n'
                    # print('found auchor')
                    # print(str_ + str(sheetname) +',' + str(filename) + '\n')
                    # str_ = str_ + str(x1[index]) + '\n'
                if positive_id == sheet_2_id[sheetname][filename]:
                    sn2 = sheetname
                    f2 = filename
                    str_ = str_ + "positive:"+ str(sheetname) +',' + str(filename) + '\n'
                    # print('found positive')
                if negative_id == sheet_2_id[sheetname][filename]:
                    sn3 = sheetname
                    f3 = filename
                    str_ = str_ + "negative:"+ str(sheetname) +',' + str(filename) + '\n'
                    # print('found negative')
        if sn1 != sn2:
            print(index)
        # print(np.linalg(distance1_1_1[index]))
        # print(np.linalg(distance2_1_1[index]))
        str_ += str(np.linalg.norm(distance1_1_1[index%len(indexs)]))
        str_ += '\n'
        str_ += str(np.linalg.norm(distance2_1_1[index%len(indexs)]))
        str_ += '\n'
        with open('dynamic_triplet_error_100000_4_no_feature.txt', 'w') as f:
            f.write(str_)
    print(all_)
    print(failed)
    # with open("dynamic_triplet_sheet_pair_num.json", 'w') as f:
        # for index1, one_fe in enumerate(id_2_feature):
        #     if feature == one_fe:
        #         found_index=index1
        #         found=True
        #         break
        # if found:
        # str_ += "pos#########################\n"
        # indexing = positive_id[found_index]
        # else:
            # for index1, one_fe in enumerate(negative_feature):
            #     if feature == one_fe:
            #         found_index=index1
            #         found=True
            #         break
        #     if found:
        #         str_ += "neg#########################\n"
        #         indexing = negative_id[found_index]
        # if not found:
        #     # print("not found")
        #     continue
        # print(indexing)
        # split_index = indexing.split(':')
        # id1 = indexing[0]
        # id2 = indexing[1]
        # for sheetname in sheet_2_id:
        #     for filename in sheet_2_id[sheetname]:
        #         if id1 == sheet_2_id[sheetname][filename]:
        #             sn1 = sheetname
        #             f1 = filename
        #             str_ = str_ + str(sheetname) +',' + str(filename) + '\n'
        #             # print(str_ + str(sheetname) +',' + str(filename) + '\n')
        #             # str_ = str_ + str(x1[index]) + '\n'
        #         if id2 == sheet_2_id[sheetname][filename]:
        #             sn2 = sheetname
        #             f2 = filename
        #             str_ = str_ + str(sheetname) +',' + str(filename) + '\n'
        #             # print(str_ + str(sheetname) +',' + str(filename) + '\n')
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
        # sp = sn1 + '------' + sn2
        # if sp not in sheet_pair_dict:
            # sheet_pair_dict[sp]=0
        # sheet_pair_dict[sp]+=1
    
        # json.dump(sheet_pair_dict, f)
def pr_curve():
    y_triplet_score = []
    y_triplet_test = []
    y_triplet_pred = []
    distance1 = np.load('dynamic20_distance1_1_'+str(3)+'_'+ str(12)+'.npy', allow_pickle=True)
    distance2 = np.load('dynamic20_distance2_1_'+str(3)+'_'+ str(12)+'.npy', allow_pickle=True)
    for distance in distance1:
        y_triplet_score.append(0-distance)
        y_triplet_test.append(1)
        

    for distance in distance2:
        y_triplet_score.append(0-distance)
        y_triplet_test.append(0)
        # if distance <= 1:

    precision,recall,threshold = precision_recall_curve(y_triplet_test,y_triplet_score) ###计算真正率和假正率

    plt.clf()
    plt.plot(recall, precision, lw=2, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.savefig("triplet_dymatic_loss_pr.png")
    
def generate_sheet_level_training_before():
    with open("json_data/sheetname_2_file_devided_prob_filter.json", 'r') as f:
        sheetname_2_file_devided = json.load(f)
        classes = list(self.sheetname_2_file_devided.keys())
    for item in sheetname_2_file_devided:
        print(item)
        print(sheetname_2_file_devided[item])
        break
if __name__ == '__main__':
    # generate_sheet_level_training_before()
    # triplet_selection = TripletSelection(batch_size=128, inner_times=2, epoch_nums=20, topk=30, max_inclass=40, margin=20)
    # triplet_selection.outer_training()
    # triplet_selection.batch_testing()
    # get_roc_curve()
    # get_threshold_p_r()
    # look_color()
    # get_threshold_p_r_all_not_dy()
    # error_analyze()
    # triplet_selection.load_embed_features()
    # all_sorted_list = {}
    # triplet_selection.embed_features = np.load("embed_features.npy", allow_pickle=True)
    # temp = []
    # for i in triplet_selection.embed_features:
    #     print('add temp', len(i))
    #     temp.append(np.array(i[0].detach()))
    # triplet_selection.embed_features = np.array(temp)
    # triplet_selection.id2filesheet = np.load("id2filesheet.npy", allow_pickle=True).item()
    # num = 0
    # for index,auchor in enumerate(triplet_selection.auchors):
    #     sorted_list = triplet_selection.get_sorted_list(index)
    #     all_sorted_list[auchor] = []
    #     for ind in sorted_list[0]:
    #         all_sorted_list[auchor].append(triplet_selection.id2filesheet[ind])
    #     num += 1
    #     print(num)
    #     np.save('all_sorted_list.npy', all_sorted_list)

        # triplet_selection.get_sorted_list()