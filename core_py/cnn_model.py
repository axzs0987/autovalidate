import torch

class CNNRCModel(torch.nn.Module):
    def __init__(self):
        super(CNNRCModel,self).__init__()
        self.conv_c = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=35,
                            out_channels=16,
                            kernel_size=(12,3),
                            stride=1,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv_r = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=35,
                            out_channels=16,
                            kernel_size=(3,12),
                            stride=1,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv2_c = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv2_r = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv3_c = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv3_r = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv4_c = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,3,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv4_r = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,3,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )
        self.mlp = torch.nn.Linear(in_features=384, out_features=100)
        self.mlp1_b = torch.nn.Linear(in_features=100, out_features=50)
        self.mlp2_b = torch.nn.Linear(in_features=50, out_features=20)

        self.mlp = torch.nn.Linear(in_features=384, out_features=100)
        self.mlp1 = torch.nn.Linear(in_features=100, out_features=50)
        self.mlp2 = torch.nn.Linear(in_features=50, out_features=20)
 
    def forward(self, x1):
        # print("model 1")
        # print(x1.shape)
        # # print(x1)
        # print(x1.shape)
        # print(x1)
        bert_feature = x1[:,:,:,13:-2]
        # print(bert_feature.shape)
        bert_feature = self.mlp(bert_feature)
        # print(bert_feature.shape)
        bert_feature = self.mlp1_b(bert_feature)
        # print(bert_feature.shape)
        bert_feature = self.mlp2_b(bert_feature)
        # print(bert_feature.shape)
        concate_fature = torch.cat([x1[:,:,:,0:13],bert_feature,x1[:,:,:,-2:]],3)

        # print(concate_fature.shape)
        concate_fature = concate_fature.permute(0,3,1,2)
        x1 = concate_fature
        # print(x1.shape)
        row_feature = self.conv_r(x1) #torch.Size([1, 16, 100, 3])
        # print(row_feature.shape)
        row_feature = self.conv2_r(row_feature) #torch.Size([1, 32, 50, 2])
        # print(row_feature.shape)
        row_feature = self.conv3_r(row_feature) #torch.Size([1, 64, 25, 1])
        # print(row_feature.shape)
        row_feature = row_feature.view(row_feature.size(0),-1)#torch.Size([1, 1600])
        # print(row_feature.shape)

        column_feature = self.conv_c(x1)#orch.Size([1, 16, 3, 10])
        # print(column_feature.shape)
        column_feature = self.conv2_c(column_feature)#torch.Size([1, 32, 2, 5])
        # print(column_feature.shape)
        column_feature = self.conv3_c(column_feature)#torch.Size([1, 64, 1, 3])
        # print(column_feature.shape)
        column_feature = column_feature.view(column_feature.size(0),-1)#torch.Size([1, 192])
        # print(column_feature.shape)

        concat_feature = torch.cat([row_feature,column_feature], 1)#torch.Size([1, 1792])
        # print('concat_feature', concat_feature.shape)
        

        
        concat_feature = self.mlp(concat_feature)
        concat_feature = self.mlp1(concat_feature)
        concat_feature = self.mlp2(concat_feature)
  
        return concat_feature#384

class CNNnetTriplet(torch.nn.Module):
    def __init__(self):
        super(CNNnetTriplet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )
 
    def forward(self, x1):
        # print("model 1")
        # print(x1.shape)
        # print(x1)
        x1 = self.conv1(x1)
        # print("model 2")
        x1 = self.conv2(x1)
        # print("model 3")/
        x1 = self.conv3(x1)
        # print("model 4")
        x1 = self.conv4(x1)
        # print("model 5")
        x1 = x1.view(x1.size(0),-1)
        # print("model 6")
        return x1

class CNNnetTripletBert(torch.nn.Module):
    def __init__(self):
        super(CNNnetTripletBert,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=35,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128,3,2,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128,128,3,2,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )
        self.mlp = torch.nn.Linear(in_features=384, out_features=100)
        self.mlp1 = torch.nn.Linear(in_features=100, out_features=50)
        self.mlp2 = torch.nn.Linear(in_features=50, out_features=20)

        self.last_mlp = torch.nn.Linear(in_features=896, out_features=640)
        self.last_mlp1 = torch.nn.Linear(in_features=640, out_features=384)

    def forward(self, x1):
        # print("model 1")
        print(x1.shape)
        # print(x1)
        bert_feature = x1[:,:,:,13:-2]
        # print(bert_feature.shape)
        bert_feature = self.mlp(bert_feature)
        # print(bert_feature.shape)
        bert_feature = self.mlp1(bert_feature)
        # print(bert_feature.shape)
        bert_feature = self.mlp2(bert_feature)
        # print(bert_feature.shape)
        concate_fature = torch.cat([x1[:,:,:,0:13],bert_feature,x1[:,:,:,-2:]],3)

        # print(concate_fature.shape)
        concate_fature = concate_fature.permute(0,3,1,2)
        # print(concate_fature.shape)
        concate_fature = self.conv1(concate_fature)
        # x1 = self.conv1(x1)
        # print("model 2")
        concate_fature = self.conv2(concate_fature)
        # print("model 3")/
        concate_fature = self.conv3(concate_fature)
        # print("model 4")
        concate_fature = self.conv4(concate_fature)
        # print("model 5")
        concate_fature = concate_fature.view(concate_fature.size(0),-1)
        # print(concate_fature.shape)

        last_feature = self.last_mlp(concate_fature)
        last_feature = self.last_mlp1(last_feature)
        # print("model 6")
        return concate_fature

class CNNnetTripletBert1010(torch.nn.Module):
    def __init__(self):
        super(CNNnetTripletBert1010,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=35,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128,3,2,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128,128,3,2,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )
        self.mlp = torch.nn.Linear(in_features=384, out_features=100)
        self.mlp1 = torch.nn.Linear(in_features=100, out_features=50)
        self.mlp2 = torch.nn.Linear(in_features=50, out_features=20)

        self.last_mlp = torch.nn.Linear(in_features=128, out_features=640)
        self.last_mlp1 = torch.nn.Linear(in_features=640, out_features=384)

    def forward(self, x1):
        # print("model 1")
        print(x1.shape)
        # print(x1)
        bert_feature = x1[:,:,:,13:-2]
        # print(bert_feature.shape)
        bert_feature = self.mlp(bert_feature)
        # print(bert_feature.shape)
        bert_feature = self.mlp1(bert_feature)
        # print(bert_feature.shape)
        bert_feature = self.mlp2(bert_feature)
        # print(bert_feature.shape)
        concate_fature = torch.cat([x1[:,:,:,0:13],bert_feature,x1[:,:,:,-2:]],3)

        # print('concate_fature', concate_fature.shape)
        concate_fature = concate_fature.permute(0,3,1,2)
        # print('concate_fature', concate_fature.shape)
        concate_fature = self.conv1(concate_fature)
        # x1 = self.conv1(x1)
        # print("model 2")
        concate_fature = self.conv2(concate_fature)
        # print("model 3")
        concate_fature = self.conv3(concate_fature)
        print("model 4")
        print('concate_fature', concate_fature.shape)
        concate_fature = self.conv4(concate_fature)
        print("model 5")
        
        concate_fature = concate_fature.view(concate_fature.size(0),-1)
        # print(concate_fature.shape)

        last_feature = self.last_mlp(concate_fature)
        last_feature = self.last_mlp1(last_feature)
        # print("model 6")
        return concate_fature

class CNNnetTripletBertL2Norm(torch.nn.Module):
    def __init__(self):
        super(CNNnetTripletBertL2Norm,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=35,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128,3,2,1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128,128,3,2,1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )
        self.mlp = torch.nn.Linear(in_features=384, out_features=100)
        self.mlp1 = torch.nn.Linear(in_features=100, out_features=50)
        self.mlp2 = torch.nn.Linear(in_features=50, out_features=20)

        self.last_mlp = torch.nn.Linear(in_features=896, out_features=640)
        self.last_mlp1 = torch.nn.Linear(in_features=640, out_features=384)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x1):
        # print("model 1")
        # print(x1.shape)
        # print(x1)
        bert_feature = x1[:,:,:,13:-2]
        # print(bert_feature.shape)
        bert_feature = self.mlp(bert_feature)
        # print(bert_feature.shape)
        bert_feature = self.mlp1(bert_feature)
        # print(bert_feature.shape)
        bert_feature = self.mlp2(bert_feature)
        # print(bert_feature.shape)
        concate_fature = torch.cat([x1[:,:,:,0:13],bert_feature,x1[:,:,:,-2:]],3)

        # print(concate_fature.shape)
        concate_fature = concate_fature.permute(0,3,1,2)
        # print(concate_fature.shape)
        concate_fature = self.conv1(concate_fature)
        # x1 = self.conv1(x1)
        # print("model 2")
        concate_fature = self.conv2(concate_fature)
        # print("model 3")/
        concate_fature = self.conv3(concate_fature)
        # print("model 4")
        concate_fature = self.conv4(concate_fature)
        # print("model 5")
        concate_fature = concate_fature.view(concate_fature.size(0),-1)
        # print(concate_fature.shape)
        concate_fature = torch.nn.functional.normalize(concate_fature, p=2)

        last_feature = self.last_mlp(concate_fature)
        last_feature = self.last_mlp1(last_feature)
        last_feature = self.l2_norm(last_feature)
        # print("model 6")
        return last_feature

class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=10,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )
       
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(768,400),
            torch.nn.ReLU(),
            torch.nn.Linear(400,100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,2),
            # torch.nn.ReLU()
            torch.nn.Softmax()
        )
    def get_embedding(self, x1):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = x1.view(x1.size(0),-1)
        return x1
        
    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = x1.view(x1.size(0),-1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        # print("################")
        x2 = x2.view(x2.size(0),-1)
        # print(x1)
        # print(x2)
        # x = torch.cosine_similarity(x1, x2)
        x = torch.cat([x1,x2],1)
        x = self.mlp(x)
        # x = self.mlp2(x)
        return x

class CNNnet_Cosine(torch.nn.Module):
    def __init__(self):
        super(CNNnet_Cosine,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=10,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2),
        )
       
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(768,400),
            torch.nn.ReLU(),
            torch.nn.Linear(400,100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,2),
            # torch.nn.ReLU()
            torch.nn.Softmax()
        )
    def get_embedding(self, x1):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = x1.view(x.size(0),-1)
        return x1
        
    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = x1.view(x1.size(0),-1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        # print("################")
        x2 = x2.view(x2.size(0),-1)
        # print(x1)
        # print(x2)
        x = torch.cosine_similarity(x1, x2)
        # x = torch.cat([x1,x2],1)
        # x = self.mlp(x)
        # x = self.mlp2(x)
        return x


