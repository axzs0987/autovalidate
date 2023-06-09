from torch import nn
import torch
import numpy as np
torch.set_printoptions(threshold=np.inf)
class PointNN(nn.Module):
    def __init__(self):
        super(PointNN, self).__init__()
        self.r_d1 = nn.Linear(9, 9) 
        self.r_a1 = nn.LeakyReLU()
        self.r_d2 = nn.Linear(9, 9)
        self.r_a2 = nn.LeakyReLU()
        self.r_d3 = nn.Linear(9, 9)
        self.r_a3 = nn.LeakyReLU()
        self.r_d4 = nn.Linear(9, 9)
        self.r_a4 = nn.LeakyReLU()
        self.r_d5 = nn.Linear(9, 9)
        self.r_a5 = nn.LeakyReLU()
        self.r_d6 = nn.Linear(9, 5)
        self.r_a6 = nn.LeakyReLU()
        self.r_d7 = nn.Linear(5, 5)
        self.r_a7 = nn.LeakyReLU()
        self.r_d8 = nn.Linear(5, 2)
        self.r_softmax = nn.Softmax(dim=1) 

        self.g_d1 = nn.Linear(10, 10) 
        self.g_a1 = nn.LeakyReLU()
        self.g_d2 = nn.Linear(10, 10)
        self.g_a2 = nn.LeakyReLU()
        self.g_d3 = nn.Linear(10, 10)
        self.g_a3 = nn.LeakyReLU()
        self.g_d4 = nn.Linear(10, 10)
        self.g_a4 = nn.LeakyReLU()
        self.g_d5 = nn.Linear(10, 10)
        self.g_a5 = nn.LeakyReLU()
        self.g_d6 = nn.Linear(10, 5)
        self.g_a6 = nn.LeakyReLU()
        self.g_d7 = nn.Linear(5, 5)
        self.g_a7 = nn.LeakyReLU()
        self.g_d8 = nn.Linear(5, 2)
        self.g_softmax = nn.Softmax(dim=1) 

        self.mlp_d1 = nn.Linear(4, 3) 
        self.mlp_a1 = nn.LeakyReLU()
        self.mlp_d2 = nn.Linear(3, 2) 
        self.mlp_a2 = nn.LeakyReLU()
        self.mlp_d3 = nn.Linear(2, 2) 
        self.mlp_softmax = nn.Softmax(dim=1)

    def forward(self, range_feature, global_feature):
        range_feature = torch.tensor(range_feature, dtype=torch.float32)
        global_feature = torch.tensor(global_feature, dtype=torch.float32)
        range_feature = self.r_d1(range_feature)
        range_feature = self.r_a1(range_feature)
        range_feature = self.r_d2(range_feature)
        range_feature = self.r_a2(range_feature)
        range_feature = self.r_d3(range_feature)
        range_feature = self.r_a3(range_feature)
        range_feature = self.r_d4(range_feature)
        range_feature = self.r_a4(range_feature)
        range_feature = self.r_d5(range_feature)
        range_feature = self.r_a5(range_feature)
        range_feature = self.r_d6(range_feature)
        range_feature = self.r_a6(range_feature)
        range_feature = self.r_d7(range_feature)
        range_feature = self.r_a7(range_feature)
        range_feature = self.r_d8(range_feature)
        range_feature = self.r_softmax(range_feature)

        global_feature = self.g_d1(global_feature)
        global_feature = self.g_a1(global_feature)
        global_feature = self.g_d2(global_feature)
        global_feature = self.g_a2(global_feature)
        global_feature = self.g_d3(global_feature)
        global_feature = self.g_a3(global_feature)
        global_feature = self.g_d4(global_feature)
        global_feature = self.g_a4(global_feature)
        global_feature = self.g_d5(global_feature)
        global_feature = self.g_a5(global_feature)
        global_feature = self.g_d6(global_feature)
        global_feature = self.g_a6(global_feature)
        global_feature = self.g_d7(global_feature)
        global_feature = self.g_a7(global_feature)
        global_feature = self.g_d8(global_feature)
        global_feature = self.g_softmax(global_feature)

        merge_feature = torch.cat([range_feature,global_feature],1)
        merge_feature = self.mlp_d1(merge_feature)
        merge_feature = self.mlp_a1(merge_feature)
        merge_feature = self.mlp_d2(merge_feature)
        merge_feature = self.mlp_a2(merge_feature)
        merge_feature = self.mlp_d3(merge_feature)
        merge_feature = self.mlp_softmax(merge_feature)
        return merge_feature


class PairNN(nn.Module):
    def __init__(self):
        super(PairNN, self).__init__()
        self.d1 = nn.Linear(19, 19) 
        self.a1 = nn.LeakyReLU()
        self.d2 = nn.Linear(19, 19)
        self.a2 = nn.LeakyReLU()
        self.d3 = nn.Linear(19, 10)
        self.a3 = nn.LeakyReLU()
        self.d4 = nn.Linear(10, 5)
        self.a4 = nn.LeakyReLU()
        self.d5 = nn.Linear(5, 5)
        self.a5 = nn.LeakyReLU()
        self.d6 = nn.Linear(5, 5)
        self.a6 = nn.LeakyReLU()
        self.d7 = nn.Linear(5, 2)
        self.softmax = nn.Softmax(dim=1) 


    def forward(self, delta_feature):
        delta_feature = torch.tensor(delta_feature, dtype=torch.float32)
        delta_feature = self.d1(delta_feature)
        delta_feature = self.a1(delta_feature)
        delta_feature = self.d2(delta_feature)
        delta_feature = self.a2(delta_feature)
        delta_feature = self.d3(delta_feature)
        delta_feature = self.a3(delta_feature)
        delta_feature = self.d4(delta_feature)
        delta_feature = self.a4(delta_feature)
        delta_feature = self.d5(delta_feature)
        delta_feature = self.a5(delta_feature)
        delta_feature = self.d6(delta_feature)
        delta_feature = self.a6(delta_feature)
        delta_feature = self.d7(delta_feature)
        delta_feature = self.softmax(delta_feature)

        return delta_feature