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

from sklearn.metrics import roc_curve, auc
import faiss

def look_shape():
    batch_size = 128
    with open("cnn_triplet_x_1.json", 'r') as f:
        x1 = json.load(f)
    with open("cnn_x_1.json", 'r') as f:
        x11 = json.load(f)
    with open("cnn_y_1.json", 'r') as f:
        y11 = json.load(f)
    x1 = TensorDataset(torch.FloatTensor(x1))
    x11 = TensorDataset(torch.LongTensor(y11), torch.FloatTensor(x11))
 
    train_loade = DataLoader(x11,batch_size=batch_size,shuffle=True)
    train_loader1 = DataLoader(x1,batch_size=batch_size,shuffle=True)
    for i,(x,y) in enumerate(train_loader1):
        print("############")
        print(len(x))
        print(len(x[0]))
        print(len(x[0][0]))
        print(len(x[0][0][0]))
        break

    for i,(x,y) in enumerate(train_loade):
        print("############")
        print(len(x))
        print(len(x[0]))
        print(len(x[0][0]))
        print(len(x[0][0][0]))
        break

def training_testing(x_train, x_test ,batch):
    print('start training testing '+ str(batch)+"......")
    batch_size = 128
    model = CNNnet()
    # loss_func = torch.nn.L1Loss()
    loss_func = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    opt = torch.optim.Adam(model.parameters(),lr=0.001)
    # print(x_train[0])
    train_data = TensorDataset(torch.FloatTensor(x_train))
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    test_data = TensorDataset(torch.FloatTensor(x_test))
    test_loader = DataLoader(test_data,batch_size=len(test_data),shuffle=True)

    loss_count = []
    for epoch in range(10):

        for i,(x) in enumerate(train_loader):
            # print('len trainx', len(x))
            x = x[0]
            # print(x)
            # print(x[:,0,:])
            archor = x[:,0,:].reshape(len(x),100,10,10).permute(0,3,1,2)
            positive = x[:,1,:].reshape(len(x),100,10,10).permute(0,3,1,2)
            negative = x[:,2,:].reshape(len(x),100,10,10).permute(0,3,1,2)

            x1 = Variable(archor) # torch.Size([batch_size, 1000, 10])
            x2 = Variable(positive) ## torch.Size([batch_size, 1000, 10])
            x3 = Variable(negative) ## torch.Size([batch_size, 1000, 10])

            x1 = model(x1) # torch.Size([128,10])
            x2 = model(x2)
            x3 = model(x3)

            loss = loss_func(x1,x2,x3)

            # x1_detach = x1.detach().numpy()
            # x2_detach = x2.detach().numpy()
            # x3_detach = x3.detach().numy()
            # distance1 = np.linalg.norm()
            # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上
            if i%20 == 0:
                loss_count.append(loss)
                print('{}:\t'.format(i), loss.item())
                torch.save(model,'cnn_model_'+str(batch))
            if i % 100 == 0:
                tptn = 0
                tpfn = 0
                fptn = 0
                fpfn = 0
                best = 0
                hard = 0
                low = 0
                for test_x in test_loader:
                    test_x = test_x[0]
                    print('len testx', len(test_x))
                    test_x1 = test_x[:,0,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
                    test_x2 = test_x[:,1,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
                    test_x3 = test_x[:,2,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
                    test_x1 = Variable(test_x1) # torch.Size([batch_size, 1000, 10])
                    test_x2 = Variable(test_x2) ## torch.Size([batch_size, 1000, 10])
                    test_x3 = Variable(test_x3)

                    test_x1 = model(test_x1).detach().numpy() # torch.Size([128,10])
                    test_x2 = model(test_x2).detach().numpy()
                    test_x3 = model(test_x3).detach().numpy()
                    
                    distance1 = []
                    distance2 = []
                    cos1 = []
                    cos2 = []
                    for index, emb in enumerate(test_x1):
                        cos1.append(test_x1[index].dot(test_x2[index]))
                        cos2.append(test_x1[index].dot(test_x3[index]))
                        distance1.append(np.linalg.norm(test_x1[index]-test_x2[index]))
                        distance2.append(np.linalg.norm(test_x1[index]-test_x3[index]))
                    # out = [for index, emb]
                    for index, dis1 in enumerate(distance1):
                        if cos1[index] >= 0.5 and cos2[index] < 0.5:
                            tptn+=1
                        elif cos1[index] >= 0.5 and cos2[index] >= 0.5:
                            tpfn +=1
                        elif cos1[index] < 0.5 and cos2[index] < 0.5:
                            fptn += 1
                        else:
                            fpfn +=1
                        if distance1[index] + 1 < distance2[index]:
                            best += 1
                        elif distance1[index] + 1 >= distance2[index] and distance1[index] < distance2[index]:
                            hard += 1
                        else:
                            low += 1
                    

                        
                    print('best:\t',best)
                    print('hard:\t',hard)
                    print('low:\t',low)
                    
                    print('tptn:\t',tptn)
                    print('tpfn:\t',tpfn)
                    print('fptn:\t',fptn)
                    print('fpfn:\t',fpfn)
                    break

    distance1 = []
    distance2 = []
    cos1 = []
    cos2 = []
    for test_x in test_loader:
        test_x = test_x[0]
        print('len testx', len(test_x))
        test_x1 = test_x[:,0,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
        test_x2 = test_x[:,1,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
        test_x3 = test_x[:,2,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
        test_x1 = Variable(test_x1) # torch.Size([batch_size, 1000, 10])
        test_x2 = Variable(test_x2) ## torch.Size([batch_size, 1000, 10])
        test_x3 = Variable(test_x3)

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
            cos1.append(test_x1[index].dot(test_x2[index]))
            cos2.append(test_x1[index].dot(test_x3[index]))
            distance1.append(np.linalg.norm(test_x1[index]-test_x2[index]))
            distance2.append(np.linalg.norm(test_x1[index]-test_x3[index]))
        # out = [for index, emb]
        for index, dis1 in enumerate(distance1):
            if cos1[index] >= 0.5 and cos2[index] < 0.5:
                tptn+=1
            elif cos1[index] >= 0.5 and cos2[index] >= 0.5:
                tpfn +=1
            elif cos1[index] < 0.5 and cos2[index] < 0.5:
                fptn += 1
            else:
                fpfn +=1
            if distance1[index] + 1 < distance2[index]:
                best += 1
            elif distance1[index] + 1 >= distance2[index] and distance1[index] < distance2[index]:
                hard += 1
            else:
                low += 1
        

            
        print('best:\t',best)
        print('hard:\t',hard)
        print('low:\t',low)
        
        print('tptn:\t',tptn)
        print('tpfn:\t',tpfn)
        print('fptn:\t',fptn)
        print('fpfn:\t',fpfn)
        break
    return distance1, distance2

def get_triplet_pair():
    with open("positive_pair.json", 'r') as f:
        positive_pair = json.load(f)

    triplet_pair = []
    with open("../AnalyzeDV/sheetname_2_file_devided.json", 'r') as f:
        sheetname_2_file_devided = json.load(f)
    with open("sheet_2_id.json", 'r') as f:
        sheet_2_id = json.load(f)

    count=0
    for sheetname in positive_pair:
        for pair in positive_pair[sheetname]:

            sheetname1 = random.choice(list(sheet_2_id.keys()))
            while(sheetname1==sheetname or sheetname1 not in sheet_2_id.keys()):
                sheetname1 = random.choice(list(sheetname_2_file_devided.keys()))
            # print(list(sheet_2_id[sheetname1].keys()))
            filename1 = random.choice(list(sheet_2_id[sheetname1].keys()))
            print(filename1)
            triplet_pair.append((pair[0]+"---"+sheetname,pair[1]+"---"+sheetname,filename1+'---'+sheetname1))

    with open("triplet_pair.json", 'w') as f:
        json.dump(triplet_pair, f)

def get_triplet_feature():
    with open("triplet_pair.json", 'r') as f:
        triplet_pair = json.load(f)
    with open("id_2_feature.json", 'r') as f:
        id_2_feature = json.load(f)
    with open("sheet_2_id.json", 'r') as f:
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
    with open("triplet_features.json",'w') as f:
        json.dump(res, f)
    with open("triplet_id.json",'w') as f:
        json.dump(id_list, f)

def split_data():
    print('start saving data...')
    with open("triplet_features.json",'r') as f:
        triplet_features = json.load(f)
    with open("triplet_id.json",'r') as f:
        triplet_id = json.load(f)

    x_1 = []
    x_2 = []
    x_3 = []
    x_4 = []
    index1 = []
    index2 = []
    index3 = []
    index4 = []

    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(triplet_features)
    random.seed(randnum)
    random.shuffle(triplet_id)

    for index, item in enumerate(triplet_features):
        if index<int(len(triplet_features)/4):
            x_1.append(item)
            index1.append(triplet_id[index])
        if index>=int(len(triplet_features)/4) and index<int(len(triplet_features)/2):
            x_2.append(item)
            index2.append(triplet_id[index])
        if index>=int(len(triplet_features)/2) and index<int(3*len(triplet_features)/4):
            x_3.append(item)
            index3.append(triplet_id[index])
        if index>=int(3*len(triplet_features)/4):
            x_4.append(item)
            index4.append(triplet_id[index])


    with open("shuffle_triplet_fature", 'w') as f:
        json.dump(triplet_features, f)

    with open("cnn_triplet_x_1.json", 'w') as f:
        json.dump(x_1, f)
    with open("cnn_triplet_x_2.json", 'w') as f:
        json.dump(x_2, f)
    with open("cnn_triplet_x_3.json", 'w') as f:
        json.dump(x_3, f)
    with open("cnn_triplet_x_4.json", 'w') as f:
        json.dump(x_4, f)

    with open("cnn_triplet_index_1.json", 'w') as f:
        json.dump(index1, f)
    with open("cnn_triplet_index_2.json", 'w') as f:
        json.dump(index2, f)
    with open("cnn_triplet_index_3.json", 'w') as f:
        json.dump(index3, f)
    with open("cnn_triplet_index_4.json", 'w') as f:
        json.dump(index4, f)
    return x_1,x_2,x_3,x_4

def test_on_one_model(model_path, test_data):
    distance1 = []
    distance2 = []
    cos1 = []
    cos2 = []
    model = torch.load(model_path)
    test_loader = DataLoader(test_data,batch_size=len(test_data),shuffle=True)
    for test_x in test_loader:
        test_x = test_x[0]
        print('len testx', len(test_x))
        test_x1 = test_x[:,0,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
        test_x2 = test_x[:,1,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
        test_x3 = test_x[:,2,:].reshape(len(test_x),100,10,10).permute(0,3,1,2)
        test_x1 = Variable(test_x1) # torch.Size([batch_size, 1000, 10])
        test_x2 = Variable(test_x2) ## torch.Size([batch_size, 1000, 10])
        test_x3 = Variable(test_x3)

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
            cos1.append(test_x1[index].dot(test_x2[index]))
            cos2.append(test_x1[index].dot(test_x3[index]))
            distance1.append(np.linalg.norm(test_x1[index]-test_x2[index]))
            distance2.append(np.linalg.norm(test_x1[index]-test_x3[index]))
        # out = [for index, emb]
        for index, dis1 in enumerate(distance1):
            if cos1[index] >= 0.5 and cos2[index] < 0.5:
                tptn+=1
            elif cos1[index] >= 0.5 and cos2[index] >= 0.5:
                tpfn +=1
            elif cos1[index] < 0.5 and cos2[index] < 0.5:
                fptn += 1
            else:
                fpfn +=1
            if distance1[index] + 1 < distance2[index]:
                best += 1
            elif distance1[index] + 1 >= distance2[index] and distance1[index] < distance2[index]:
                hard += 1
            else:
                low += 1
        

            
        print('best:\t',best)
        print('hard:\t',hard)
        print('low:\t',low)
        
        print('tptn:\t',tptn)
        print('tpfn:\t',tpfn)
        print('fptn:\t',fptn)
        print('fpfn:\t',fpfn)
        break
    return distance1, distance2


def batch_testing():
    all_suc_num = 0
    all_data = 0
    print('load data.......')
    with open("cnn_triplet_x_1.json", 'r') as f:
        x1 = json.load(f)
    with open("cnn_triplet_x_2.json", 'r') as f:
        x2 = json.load(f)
    with open("cnn_triplet_x_3.json", 'r') as f:
        x3 = json.load(f)
    with open("cnn_triplet_x_4.json", 'r') as f:
        x4 = json.load(f)

    print('load data end.......')
    x = [x1,x2,x3,x4]
    for batch in [1]:
        x_test = x[batch-1]
        all_data+=len(x_test)
        x_train_list = []

        for index,train_batch in enumerate(list(set([1,2,3,4])-set([batch]))):
            x_train_sub = x[train_batch-1]
            x_train_list.append(x_train_sub)

        x_train = np.concatenate((x_train_list[0], x_train_list[1], x_train_list[2]))
        distance1,distance2= test_on_one_model(model_path, x_test)
        np.save("distance1_"+str(batch)+'.npy', distance1)
        np.save("distance2_"+str(batch)+'.npy', distance2)
        # accuracy, suc_num = training_testing_cosine(x_train, x_test,batch)
    #     print('accuracy '+str(batch)+':\t',accuracy.mean())
    #     all_suc_num+=suc_num
    # print('suc_num', suc_num)
    # print('all_data', all_data)
    # print('all accuracy:\t', suc_num/all_data)
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

    pred1 = np.load('pred_1_1.npy', allow_pickle=True)
    pred2 = np.load('pred_1_2.npy', allow_pickle=True)
    pred3 = np.load('pred_1_3.npy', allow_pickle=True)
    pred4 = np.load('pred_1_4.npy', allow_pickle=True)
    test1 = np.load('test_1_1.npy', allow_pickle=True)
    test2 = np.load('test_1_2.npy', allow_pickle=True)
    test3 = np.load('test_1_3.npy', allow_pickle=True)
    test4 = np.load('test_1_4.npy', allow_pickle=True)

    pred_cos1 = np.load('pred_1_cos_1.npy', allow_pickle=True)
    pred_cos2 = np.load('pred_1_cos_2.npy', allow_pickle=True)
    pred_cos3 = np.load('pred_1_cos_3.npy', allow_pickle=True)
    pred_cos4 = np.load('pred_1_cos_4.npy', allow_pickle=True)
    test_cos1 = np.load('test_1_cos_1.npy', allow_pickle=True)
    test_cos2 = np.load('test_1_cos_2.npy', allow_pickle=True)
    test_cos3 = np.load('test_1_cos_3.npy', allow_pickle=True)
    test_cos4 = np.load('test_1_cos_4.npy', allow_pickle=True)


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

    test = [test_cos1, test_cos2, test_cos3,test_cos4]
    for index,pred in enumerate([pred_cos1, pred_cos2, pred_cos3, pred_cos4]):
        for index1, item in enumerate(pred):
            y_cos_score.append(item)
            y_cos_test.append(test[index][index1])

    fpr,tpr,threshold = roc_curve(y_triplet_test,y_triplet_score) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    fpr1,tpr1,threshold1 = roc_curve(y_test,y_score) ###计算真正率和假正率
    roc_auc1 = auc(fpr1,tpr1) ###计算auc的值
    fpr2,tpr2,threshold2 = roc_curve(y_cos_test,y_cos_score) ###计算真正率和假正率
    roc_auc2 = auc(fpr2,tpr2) ###计算auc的值
 
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,19))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='triplet ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot(fpr1, tpr1, color='pink',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc1) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot(fpr2, tpr2, color='green',
             lw=lw, label='cos ROC curve (area = %0.3f)' % roc_auc2) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of Sheet Embedding Models')
    plt.legend(loc="lower right")
 
    plt.savefig("triplet_loss_roc.png")


if __name__ == '__main__':
    # get_triplet_pair()
    # get_triplet_feature()
    # split_data()
    batch_testing()
    # get_roc_curve()
    # look_shape()