import json 
import matplotlib.pyplot as plt
import pprint
from random import sample

class Analyzer:
    def __init__(self):
        # with open("../share/dvinfoWithRef.json",'r', encoding='UTF-8') as f:
        self.dv_infos = []
        self.wn_dv_infos = [] # 1
        self.f_dv_infos = [] # 2
        self.d_dv_infos = [] # 3
        self.t_dv_infos = [] # 4
        self.tl_dv_infos = [] # 5
        self.l_dv_infos = [] # 6

    def count_content_number(self):
        with open("../AnalyzeDV/new_continous_batch_0.json",'r', encoding='UTF-8') as f:
            dvinfos0 = json.load(f)
        with open("../AnalyzeDV/new_continous_batch_1.json",'r', encoding='UTF-8') as f:
            dvinfos1 = json.load(f)
        with open("../AnalyzeDV/new_continous_batch_2.json",'r', encoding='UTF-8') as f:
            dvinfos2 = json.load(f)
        with open("../AnalyzeDV/new_continous_batch_3.json",'r', encoding='UTF-8') as f:
            dvinfos3 = json.load(f)       


        result_dic = {}
        for i in dvinfos0:
            if len(i["content"]) not in result_dic:
                result_dic[len(i["content"])] = 0
            result_dic[len(i["content"])] += 1
        for i in dvinfos1:
            if len(i["content"]) not in result_dic:
                result_dic[len(i["content"])] = 0
            result_dic[len(i["content"])] += 1
        for i in dvinfos2:
            if len(i["content"]) not in result_dic:
                result_dic[len(i["content"])] = 0
            result_dic[len(i["content"])] += 1
        for i in dvinfos3:
            if len(i["content"]) not in result_dic:
                result_dic[len(i["content"])] = 0
            result_dic[len(i["content"])] += 1
        x = result_dic.keys()
        y = result_dic.values()

        plt.scatter(x,y)
        # plt.xlim(0,100)
        # plt.ylim(0,10000)
        pprint.pprint(result_dic)
        plt.savefig("list_number_distribution.png")
        plt.cla()

    def get_popular_refer_list(self):
        with open("../AnalyzeDV/global_refer_list_number.json",'r', encoding='UTF-8') as f:
            global_refer_list_number = json.load(f)
        with open("../AnalyzeDV/refer_dictionary.json",'r', encoding='UTF-8') as f:
            refer_dictionary = json.load(f)
        a1 = sorted(global_refer_list_number.items(),key = lambda x:x[1],reverse = True)
        result = []
        print(a1)
        # print(refer_dictionary)
        
        count = 0
        for i in a1:
            if count == 100:
                break
            result.append([refer_dictionary[i[0]], i[1]])
            count += 1
        pprint.pprint(result)

    def plot_boundary(self, type_id, prex):
        boundary_et = {}
        boundary_net = {}
        boundary_gt = {}
        boundary_lt = {}
        boundary_eogt = {}
        boundary_eolt = {}
        boundary_b = {}
        boundary_b_min = {}
        boundary_b_max = {}
        boundary_nbt = {}
        boundary_et_number = 0
        boundary_net_number = 0
        boundary_gt_number = 0
        boundary_lt_number = 0
        boundary_eogt_number = 0
        boundary_eolt_number = 0
        boundary_b_number = 0
        boundary_nbt_number = 0

        all_number = 0
        file_set = set()
        dv_list = []
        for dv_info in self.dv_infos:
            if dv_info["Type"] == type_id:
                all_number += 1
                # self.boundary_dv_infos.append(dv_info)
                if dv_info["Operator"] == 0: # whole number equal to
                    if dv_info["MinValue"] not in boundary_et.keys():
                        boundary_et[dv_info["MinValue"]] = 0
                    # if dv_info["MinValue"] == "1":
                    #     print(dv_info["FileName"], dv_info["SheetName"], dv_info["RangeAddress"])
                    boundary_et[dv_info["MinValue"]] += 1
                    boundary_et_number += 1
                elif dv_info["Operator"] == 1: # whole number not equal to
                    if dv_info["MinValue"] not in boundary_net.keys():
                        boundary_net[dv_info["MinValue"]] = 0
                    # if dv_info["MinValue"] == "0":
                    #     print(dv_info["FileName"], dv_info["SheetName"], dv_info["RangeAddress"])
                    boundary_net[dv_info["MinValue"]] += 1
                    boundary_net_number += 1
                elif dv_info["Operator"] == 2: # whole number greater than
                    if dv_info["MinValue"] not in boundary_gt.keys():
                        boundary_gt[dv_info["MinValue"]] = 0
                    # if dv_info["MinValue"] == "0":
                    #     print(dv_info["FileName"], dv_info["SheetName"], dv_info["RangeAddress"])
                    boundary_gt[dv_info["MinValue"]] += 1
                    boundary_gt_number += 1
        
                elif dv_info["Operator"] == 3: # whole number less than
                    if dv_info["MinValue"] not in boundary_lt.keys():
                        boundary_lt[dv_info["MinValue"]] = 0
                    # if dv_info["MinValue"] == "0":
                    #     print(dv_info["FileName"], dv_info["SheetName"], dv_info["RangeAddress"])
                    boundary_lt[dv_info["MinValue"]] += 1
                    boundary_lt_number += 1
                    
                elif dv_info["Operator"] == 4: # whole number equal or greater than
                    if dv_info["MinValue"] not in boundary_eogt.keys():
                        boundary_eogt[dv_info["MinValue"]] = 0
                    # if dv_info["MinValue"] == "A2":
                    #     if dv_info["FileName"] not in file_set:
                    #         dv_list.append(dv_info)
                    #         print(dv_info["FileName"], dv_info["SheetName"], dv_info["RangeAddress"])
                    #     file_set.add(dv_info["FileName"])
                    boundary_eogt[dv_info["MinValue"]] += 1
                    boundary_eogt_number += 1
                    # if dv_info["MinValue"] == "0":
                    #     print(dv_info["ID"], dv_info["FileName"], dv_info["SheetName"], dv_info["RangeAddress"])
                elif dv_info["Operator"] == 5: # whole number equal or less than
                    if dv_info["MinValue"] not in boundary_eolt.keys():
                        boundary_eolt[dv_info["MinValue"]] = 0
                    boundary_eolt[dv_info["MinValue"]] += 1
                    boundary_eolt_number += 1
                elif dv_info["Operator"] == 6: # whole number between
                    if dv_info["MinValue"]+","+dv_info["MaxValue"] not in boundary_b.keys():
                        boundary_b[dv_info["MinValue"]+","+dv_info["MaxValue"]] = 0
                    if dv_info["MinValue"] not in boundary_b_min.keys():
                        boundary_b_min[dv_info["MinValue"]] = 0
                    if dv_info["MaxValue"] not in boundary_b_max.keys():
                        boundary_b_max[dv_info["MaxValue"]] = 0
                    boundary_b[dv_info["MinValue"]+","+dv_info["MaxValue"]] += 1
                    boundary_b_min[dv_info["MinValue"]] += 1
                    boundary_b_max[dv_info["MaxValue"]] += 1
                    boundary_b_number += 1
                    # if dv_info["MinValue"] == "-0.25":
                    # if dv_info["FileName"] not in file_set:
                    #     dv_list.append(dv_info)
                    # file_set.add(dv_info["FileName"])
                elif dv_info["Operator"] == 7: # whole number not between
                    if dv_info["MinValue"]+","+dv_info["MaxValue"] not in boundary_nbt.keys():
                        boundary_nbt[dv_info["MinValue"]+","+dv_info["MaxValue"]] = 0
                    boundary_nbt[dv_info["MinValue"]+","+dv_info["MaxValue"]] += 1
                    boundary_nbt_number += 1
                    # if dv_info["MinValue"] == "0":
                    #     if dv_info["FileName"] not in file_set:
                    #         dv_list.append(dv_info)
                    #     file_set.add(dv_info["FileName"])
        # print("file_set_len", len(file_set))
        # result = sample(dv_list, 10)
        # pprint.pprint(result)
        print("###################################################################")
        x = boundary_et.keys()
        y = boundary_et.values()
        print("equal to: ")
        print("number:", boundary_et_number)
        # pprint.pprint(boundary_et)
        plt.plot(x,y)
        plt.savefig(prex + "_et.png")
        plt.cla()

        print("###################################################################")
        x = boundary_net.keys()
        y = boundary_net.values()
        print("not equal to: ")
        print("number:", boundary_net_number)
        # pprint.pprint(boundary_net)
        plt.plot(x,y)
        plt.savefig(prex + "_net.png")
        plt.cla()

        print("###################################################################")
        x = boundary_gt.keys()
        y = boundary_gt.values()
        print("greater than: ")
        print("number:", boundary_gt_number)
        # pprint.pprint(boundary_gt)
        plt.plot(x,y)
        plt.savefig(prex + "_gt.png")
        plt.cla()

        print("###################################################################")
        x = boundary_eogt.keys()
        y = boundary_eogt.values()
        print("equal or greater than: ")
        print("number:", boundary_eogt_number)
        # pprint.pprint(boundary_eogt)
        plt.plot(x,y)
        plt.savefig(prex + "_eogt.png")
        plt.cla()

        print("###################################################################")
        x = boundary_lt.keys()
        y = boundary_lt.values()
        print("less than: ")
        print("number:", boundary_lt_number)
        # pprint.pprint(boundary_lt)
        plt.plot(x,y)
        plt.savefig(prex + "_lt.png")
        plt.cla()
        
        print("###################################################################")
        x = boundary_eolt.keys()
        y = boundary_eolt.values()
        print("equal or less than: ")
        print("number:", boundary_eolt_number)
        # pprint.pprint(boundary_eolt)
        plt.plot(x,y)
        plt.savefig(prex + "_eolt.png")
        plt.cla()
        
        print("###################################################################")
        x = boundary_nbt.keys()
        y = boundary_nbt.values()
        print("not between to: ")
        print("number:", boundary_nbt_number)
        # pprint.pprint(boundary_nbt)
        plt.plot(x,y)
        plt.savefig(prex + "_nbt.png")
        plt.cla()

        print("###################################################################")
        x = boundary_b.keys()
        y = boundary_b.values()
        print("between to: ")
        print("number:", boundary_b_number)
        # pprint.pprint(boundary_b)
        # pprint.pprint(boundary_b_min)
        # pprint.pprint(boundary_b_max)
        try:
            plt.plot(x,y)
            plt.savefig(prex + "_b.png")
            plt.cla()
        except:
            print("error")

        print("all_number:", all_number)


if __name__ == "__main__":
    analyzer = Analyzer()
    analyzer.get_popular_refer_list()

