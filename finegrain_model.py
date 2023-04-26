import torch

class FinegrainedModel(torch.nn.Module):
    def __init__(self):
        super(FinegrainedModel,self).__init__()
        
        self.mlp = torch.nn.Linear(in_features=384, out_features=100)
        self.mlp1 = torch.nn.Linear(in_features=100, out_features=50)
        self.mlp2 = torch.nn.Linear(in_features=50, out_features=20)

        self.last_mlp = torch.nn.Linear(in_features=35, out_features=32)
        self.last_relu = torch.nn.LeakyReLU()
        self.last_mlp1 = torch.nn.Linear(in_features=32, out_features=32)
        self.last_relu1 = torch.nn.LeakyReLU()
        self.last_mlp2 = torch.nn.Linear(in_features=32, out_features=32)
        self.last_relu2 = torch.nn.LeakyReLU()
        self.last_mlp3 = torch.nn.Linear(in_features=32, out_features=16)
        self.last_relu3 = torch.nn.LeakyReLU()
        self.last_mlp4 = torch.nn.Linear(in_features=16, out_features=16)
        self.last_relu4 = torch.nn.LeakyReLU()
        self.last_mlp5= torch.nn.Linear(in_features=16, out_features=16)
        self.last_relu5 = torch.nn.LeakyReLU()

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
        # print('concate_fature', concate_fature.shape)
        cell_feature = concate_fature[:,:,:,0:]
        # print('cell_feature', cell_feature.shape)
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
        last_feature = torch.reshape(last_feature,(-1,100,10))
        # print('last_feature', last_feature.shape)
        return last_feature