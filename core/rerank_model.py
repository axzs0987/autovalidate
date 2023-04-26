import torch
class RerankModel(torch.nn.Module):
    def __init__(self):
        super(RerankModel,self).__init__()
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
        self.last_relu = torch.nn.ReLU()
        self.last_mlp1 = torch.nn.Linear(in_features=640, out_features=384)
        self.last_relu1 = torch.nn.ReLU()
        self.last_mlp2 = torch.nn.Linear(in_features=384, out_features=120)
        self.last_relu2 = torch.nn.ReLU()
        self.last_mlp3 = torch.nn.Linear(in_features=120, out_features=50)
        self.last_relu3 = torch.nn.ReLU()
        self.last_mlp4 = torch.nn.Linear(in_features=50, out_features=20)
        self.last_relu4 = torch.nn.ReLU()
        self.last_mlp5 = torch.nn.Linear(in_features=20, out_features=2)
        self.last_softmax = torch.nn.Softmax(dim=1)

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
        last_feature = self.last_relu(last_feature)
        last_feature = self.last_mlp1(last_feature)
        last_feature = self.last_relu1(last_feature)
        last_feature = self.last_mlp2(last_feature)
        last_feature = self.last_relu2(last_feature)
        last_feature = self.last_mlp3(last_feature)
        last_feature = self.last_relu3(last_feature)
        last_feature = self.last_mlp4(last_feature)
        last_feature = self.last_relu4(last_feature)
        last_feature = self.last_mlp5(last_feature)
        last_feature = self.last_softmax(last_feature)

        
        # print("model 6")
        return last_feature

class RerankLinearModel(torch.nn.Module):
    def __init__(self):
        super(RerankLinearModel,self).__init__()
        self.mlp = torch.nn.Linear(in_features=384, out_features=100)
        self.mlp1 = torch.nn.Linear(in_features=100, out_features=50)
        self.mlp2 = torch.nn.Linear(in_features=50, out_features=20)

        self.last_mlp = torch.nn.Linear(in_features=35000, out_features=10000)
        self.last_relu = torch.nn.LeakyReLU()
        self.last_mlp1 = torch.nn.Linear(in_features=10000, out_features=2000)
        self.last_relu1 = torch.nn.LeakyReLU()
        self.last_mlp2 = torch.nn.Linear(in_features=2000, out_features=1000)
        self.last_relu2 = torch.nn.LeakyReLU()
        self.last_mlp3 = torch.nn.Linear(in_features=1000, out_features=500)
        self.last_relu3 = torch.nn.LeakyReLU()
        self.last_mlp4 = torch.nn.Linear(in_features=500, out_features=200)
        self.last_relu4 = torch.nn.LeakyReLU()
        self.last_mlp5= torch.nn.Linear(in_features=200, out_features=100)
        self.last_relu5 = torch.nn.LeakyReLU()
        self.last_mlp6 = torch.nn.Linear(in_features=100, out_features=2)
        self.last_softmax = torch.nn.Softmax(dim=1)

    def forward(self, x1):
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
        concate_fature = concate_fature.view(concate_fature.size(0),-1)
        # print('concate_fature', concate_fature.shape)
        # print(concate_fature.shape)

        last_feature = self.last_mlp(concate_fature)
        last_feature = self.last_relu(last_feature)
        last_feature = self.last_mlp1(last_feature)
        last_feature = self.last_relu1(last_feature)
        last_feature = self.last_mlp2(last_feature)
        last_feature = self.last_relu2(last_feature)
        last_feature = self.last_mlp3(last_feature)
        last_feature = self.last_relu3(last_feature)
        last_feature = self.last_mlp4(last_feature)
        last_feature = self.last_relu4(last_feature)
        last_feature = self.last_mlp5(last_feature)
        last_feature = self.last_relu5(last_feature)
        last_feature = self.last_mlp6(last_feature)
        last_feature = self.last_softmax(last_feature)

        
        # print("model 6")
        return last_feature


class RerankLinearModel1(torch.nn.Module):
    def __init__(self):
        super(RerankLinearModel,self).__init__()
        self.mlp = torch.nn.Linear(in_features=384, out_features=100)
        self.mlp1 = torch.nn.Linear(in_features=100, out_features=50)
        self.mlp2 = torch.nn.Linear(in_features=50, out_features=20)

        self.last_mlp = torch.nn.Linear(in_features=35000, out_features=20000)
        self.last_relu = torch.nn.LeakyReLU()
        self.last_mlp1 = torch.nn.Linear(in_features=20000, out_features=10000)
        self.last_relu1 = torch.nn.LeakyReLU()
        self.last_mlp2 = torch.nn.Linear(in_features=10000, out_features=2000)
        self.last_relu2 = torch.nn.LeakyReLU()
        self.last_mlp3 = torch.nn.Linear(in_features=2000, out_features=1000)
        self.last_relu3 = torch.nn.LeakyReLU()
        self.last_mlp4 = torch.nn.Linear(in_features=1000, out_features=500)
        self.last_relu4 = torch.nn.LeakyReLU()
        self.last_mlp5 = torch.nn.Linear(in_features=500, out_features=200)
        self.last_relu5 = torch.nn.LeakyReLU()
        self.last_mlp6= torch.nn.Linear(in_features=200, out_features=100)
        self.last_relu6 = torch.nn.LeakyReLU()
        self.last_mlp7 = torch.nn.Linear(in_features=100, out_features=2)
        self.last_softmax = torch.nn.Softmax(dim=1)

    def forward(self, x1):
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
        concate_fature = concate_fature.view(concate_fature.size(0),-1)
        # print('concate_fature', concate_fature.shape)
        # print(concate_fature.shape)

        last_feature = self.last_mlp(concate_fature)
        last_feature = self.last_relu(last_feature)
        last_feature = self.last_mlp1(last_feature)
        last_feature = self.last_relu1(last_feature)
        last_feature = self.last_mlp2(last_feature)
        last_feature = self.last_relu2(last_feature)
        last_feature = self.last_mlp3(last_feature)
        last_feature = self.last_relu3(last_feature)
        last_feature = self.last_mlp4(last_feature)
        last_feature = self.last_relu4(last_feature)
        last_feature = self.last_mlp5(last_feature)
        last_feature = self.last_relu5(last_feature)
        last_feature = self.last_mlp6(last_feature)
        last_feature = self.last_relu6(last_feature)
        last_feature = self.last_mlp7(last_feature)
        last_feature = self.last_softmax(last_feature)

        
        # print("model 6")
        return last_feature

class RerankFinegrainModel(torch.nn.Module):
    def __init__(self):
        super(RerankFinegrainModel,self).__init__()
        self.last_mlp = torch.nn.Linear(in_features=16000, out_features=16000)
        self.last_relu = torch.nn.LeakyReLU()
        self.last_mlp1 = torch.nn.Linear(in_features=16000, out_features=10000)
        self.last_relu1 = torch.nn.LeakyReLU()
        self.last_mlp2 = torch.nn.Linear(in_features=10000, out_features=2000)
        self.last_relu2 = torch.nn.LeakyReLU()
        self.last_mlp3 = torch.nn.Linear(in_features=2000, out_features=1000)
        self.last_relu3 = torch.nn.LeakyReLU()
        self.last_mlp4 = torch.nn.Linear(in_features=1000, out_features=500)
        self.last_relu4 = torch.nn.LeakyReLU()
        self.last_mlp5 = torch.nn.Linear(in_features=500, out_features=200)
        self.last_relu5 = torch.nn.LeakyReLU()
        self.last_mlp6= torch.nn.Linear(in_features=200, out_features=100)
        self.last_relu6 = torch.nn.LeakyReLU()
        self.last_mlp7 = torch.nn.Linear(in_features=100, out_features=2)
        self.last_softmax = torch.nn.Softmax(dim=1)

    def forward(self, x1):
        # delta = u - v
        x1 = x1.view(x1.size(0),-1)
        last_feature = self.last_mlp(x1)
        last_feature = self.last_relu(last_feature)
        last_feature = self.last_mlp1(last_feature)
        last_feature = self.last_relu1(last_feature)
        last_feature = self.last_mlp2(last_feature)
        last_feature = self.last_relu2(last_feature)
        last_feature = self.last_mlp3(last_feature)
        last_feature = self.last_relu3(last_feature)
        last_feature = self.last_mlp4(last_feature)
        last_feature = self.last_relu4(last_feature)
        last_feature = self.last_mlp5(last_feature)
        last_feature = self.last_relu5(last_feature)
        last_feature = self.last_mlp6(last_feature)
        last_feature = self.last_relu6(last_feature)
        last_feature = self.last_mlp7(last_feature)
        last_feature = self.last_softmax(last_feature)

        
        # print("model 6")
        return last_feature

class RerankFinegrainModelUVD(torch.nn.Module):
    def __init__(self):
        super(RerankFinegrainModelUVD,self).__init__()
        self.last_mlp = torch.nn.Linear(in_features=48000, out_features=32000)
        self.last_relu = torch.nn.LeakyReLU()
        self.last_mlp0 = torch.nn.Linear(in_features=32000, out_features=16000)
        self.last_relu0 = torch.nn.LeakyReLU()
        self.last_mlp1 = torch.nn.Linear(in_features=16000, out_features=10000)
        self.last_relu1 = torch.nn.LeakyReLU()
        self.last_mlp2 = torch.nn.Linear(in_features=10000, out_features=2000)
        self.last_relu2 = torch.nn.LeakyReLU()
        self.last_mlp3 = torch.nn.Linear(in_features=2000, out_features=1000)
        self.last_relu3 = torch.nn.LeakyReLU()
        self.last_mlp4 = torch.nn.Linear(in_features=1000, out_features=500)
        self.last_relu4 = torch.nn.LeakyReLU()
        self.last_mlp5 = torch.nn.Linear(in_features=500, out_features=200)
        self.last_relu5 = torch.nn.LeakyReLU()
        self.last_mlp6= torch.nn.Linear(in_features=200, out_features=100)
        self.last_relu6 = torch.nn.LeakyReLU()
        self.last_mlp7 = torch.nn.Linear(in_features=100, out_features=2)
        self.last_softmax = torch.nn.Softmax(dim=1)

    def forward(self, x1):
        # delta = u - v
        x1 = x1.view(x1.size(0),-1)
        last_feature = self.last_mlp(x1)
        last_feature = self.last_relu(last_feature)
        last_feature = self.last_mlp0(last_feature)
        last_feature = self.last_relu0(last_feature)
        last_feature = self.last_mlp1(last_feature)
        last_feature = self.last_relu1(last_feature)
        last_feature = self.last_mlp2(last_feature)
        last_feature = self.last_relu2(last_feature)
        last_feature = self.last_mlp3(last_feature)
        last_feature = self.last_relu3(last_feature)
        last_feature = self.last_mlp4(last_feature)
        last_feature = self.last_relu4(last_feature)
        last_feature = self.last_mlp5(last_feature)
        last_feature = self.last_relu5(last_feature)
        last_feature = self.last_mlp6(last_feature)
        last_feature = self.last_relu6(last_feature)
        last_feature = self.last_mlp7(last_feature)
        last_feature = self.last_softmax(last_feature)

        
        # print("model 6")
        return last_feature