import numpy as np
import os
import torch
import json
import faiss
from sentence_transformers import SentenceTransformer
from torch.autograd import Variable
import _thread
from multiprocessing import Process

# with open('/datadrive/data/bert_dict/bert_dict.json', 'r') as f:
#     bert_dict = json.load(f)
# bert_dict = {}
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

class End2End:
    def __init__(self, dvid2dtop_name='dvid2dtop.json', dvid2stop_name='dvid2stop.json', l2 = False):
        self.l2 = l2
        if not self.l2:
            if os.path.exists('196model/cnn_new_dynamic_triplet_margin_1_3_12'):
                self.model = torch.load('196model/cnn_new_dynamic_triplet_margin_1_3_12')
            else:
                self.model = CNNnetTriplet()
        else:
            if os.path.exists('196model/model_l2norm_3_2'):
                self.model = torch.load('196model/model_l2norm_3_2')
            else:
                self.model = CNNnetTripletBertL2Norm()

        self.bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # if os.path.exists('json_data/content_dict_1.json'):
        #     with open("json_data/content_dict_1.json", 'r') as f:
        #         self.content_dict = json.load(f)
        if os.path.exists('json_data/content_temp_dict_1.json'):
            with open("json_data/content_temp_dict_1.json", 'r') as f:
                self.content_tem_dict = json.load(f)
        self.dvid2stop_name = dvid2stop_name
        self.dvid2dtop_name = dvid2dtop_name

        self.custom_dvinfos = None
        self.dvid2sheetfeature = {}
        self.dvid2dvsheetfeature = {}
        self.dvid2dvstartrc = {}
        self.dvid2filesheet = {}
        self.dvid2dvvalue = {}

        self.dvid2stop = {}
        self.dvid2dtop = {}
        self.custom_dvinfos = []

    def distance2cos(self, distance):
        return (2-distance)/2

    def get_feature_vector_with_bert(self,feature):
        result = []
        for index, item1 in enumerate(feature):
            one_cel_feature = []
            for index1, item in enumerate(feature[index]):
                if index1==0:#background color r
                    one_cel_feature.append(item)
                if index1==1:#background color b
                    one_cel_feature.append(item)
                if index1==2:#background color g
                    one_cel_feature.append(item)
                if index1==3:#font color r
                    one_cel_feature.append(item)
                if index1==4:#font color b
                    one_cel_feature.append(item)
                if index1==5:#font color g
                    one_cel_feature.append(item)
                                
                
                        
                if index1==6:#font size
                    one_cel_feature.append(item)

                if index1==7:#font_strikethrough
                    if(item==True):
                        one_cel_feature.append(1)
                    else:
                        one_cel_feature.append(0)
                if index1==8:#font_shadow
                    if(item==True):
                        one_cel_feature.append(1)
                    else:
                        one_cel_feature.append(0)
                if index1==9:#font_ita
                    if(item==True):
                        one_cel_feature.append(1)
                    else:
                        one_cel_feature.append(0)
                if index1==10:#font_bold
                    if(item==True):
                        one_cel_feature.append(1)
                    else:
                        one_cel_feature.append(0)
                if index1==11:#height
                    one_cel_feature.append(item)
                if index1==12:#width
                    one_cel_feature.append(item)
                if index1==13:#content\\
                    if is_number(str(item)):
                        cell_type = 1
                    elif str(item) == '':
                        cell_type = 0
                    else:
                        cell_type = 2
                    # print("manma?")
                    # print(str(item) in bert_dict)
                    # if str(item) in bert_dict:
                        # bert_feature = bert_dict[str(item)]
                    # else:
                    bert_feature = self.bert_model.encode(str(item)).tolist()
                    # print("mana------")
                    for i in bert_feature:
                        one_cel_feature.append(i)
                    # if not str(item) in self.content_dict:
                    #     self.content_dict[str(item)] = len(self.content_dict)+1

                    # one_cel_feature.append(self.content_dict[str(item)])
                if index1==14:#content_template
                    if not str(item) in self.content_tem_dict:
                        self.content_tem_dict[str(item)] = len(self.content_tem_dict)+1
                    one_cel_feature.append(self.content_tem_dict[str(item)])
                    one_cel_feature.append(cell_type)
            result.append(one_cel_feature)
            
        # print('len(one feature)', len(result[0]))
        return result

    def get_feature_vector_with_bert_keyw(self,feature):
        result = []

        for index, item1 in enumerate(feature):
            one_cel_feature = []
            if type(feature[index]).__name__ == 'list':
                feature[index] = feature[index][0]
            for index1, item in enumerate(feature[index]):

                # #print('index1', index1, 'item', item, 'feature[index][item]', feature[index][item])
                if index1==0:#background color r
                    #print('index0')
                    one_cel_feature.append(feature[index][item])
                if index1==1:#background color b
                    #print('index1')
                    one_cel_feature.append(feature[index][item])
                if index1==2:#background color g
                    #print('index2')
                    one_cel_feature.append(feature[index][item])
                if index1==3:#font color r
                    #print('index3')
                    one_cel_feature.append(feature[index][item])
                if index1==4:#font color b
                    #print('index4')
                    one_cel_feature.append(feature[index][item])
                if index1==5:#font color g
                    #print('index5')
                    one_cel_feature.append(feature[index][item])
                                
                
                        
                if index1==6:#font size
                    #print('index6')
                    one_cel_feature.append(feature[index][item])

                if index1==7:#font_strikethrough
                    #print('index7')
                    if(feature[index][item]==True):
                        one_cel_feature.append(1)
                    else:
                        one_cel_feature.append(0)
                if index1==8:#font_shadow
                    #print('index8')
                    if(feature[index][item]==True):
                        one_cel_feature.append(1)
                    else:
                        one_cel_feature.append(0)
                if index1==9:#font_ita
                    #print('index9')
                    if(feature[index][item]==True):
                        one_cel_feature.append(1)
                    else:
                        one_cel_feature.append(0)
                if index1==10:#font_bold
                    #print('index10')
                    if(feature[index][item]==True):
                        one_cel_feature.append(1)
                    else:
                        one_cel_feature.append(0)
                if index1==11:#height
                    #print('index11')
                    one_cel_feature.append(feature[index][item])
                if index1==12:#width
                    #print('index12')
                    one_cel_feature.append(feature[index][item])
                if index1==13:#content\\
                    #print('index13')
                    if is_number(str(feature[index][item])):
                        cell_type = 1
                    elif str(feature[index][item]) == '':
                        cell_type = 0
                    else:
                        cell_type = 2
                    # print("manma?")
                    # print(str(item) in bert_dict)
                    if str(item) in bert_dict:
                        bert_feature = bert_dict[str(item)]
                    else:
                        bert_feature = self.bert_model.encode(str(feature[index][item])).tolist()
                    # print("mana------")
                    for i in bert_feature:
                        one_cel_feature.append(i)

                    # if not str(item) in self.content_dict:
                    #     self.content_dict[str(item)] = len(self.content_dict)+1

                    # one_cel_feature.append(self.content_dict[str(item)])
                if index1==14:#content_template
                    #print('index14')
                    if not str(feature[index][item]) in self.content_tem_dict:
                        self.content_tem_dict[str(feature[index][item])] = len(self.content_tem_dict)+1
                    one_cel_feature.append(self.content_tem_dict[str(feature[index][item])])
                    one_cel_feature.append(cell_type)
            result.append(one_cel_feature)
            
        # print('len(one feature)', len(result[0]))
        return result
    def get_new_color_feature_vector(self,feature, is_test=False):
        result = []
        for index, item1 in enumerate(feature):
            one_cel_feature = []
            cell_type = 0
            for index1, item in enumerate(feature[index]):
                if index1==0:#background color r
                    one_cel_feature.append(item)
                if index1==1:#background color g
                    one_cel_feature.append(item)
                if index1==2:#background color b
                    one_cel_feature.append(item)
                if index1==3:#font color r
                    one_cel_feature.append(item)
                if index1==4:#font color g
                    one_cel_feature.append(item)
                if index1==5:#font color b
                    one_cel_feature.append(item)
                        
                if index1==6:#font size
                    one_cel_feature.append(item)

                if index1==7:#font_strikethrough
                    if(item==True):
                        one_cel_feature.append(1)
                    else:
                        one_cel_feature.append(0)
                if index1==8:#font_shadow
                    if(item==True):
                        one_cel_feature.append(1)
                    else:
                        one_cel_feature.append(0)
                if index1==9:#font_ita
                    if(item==True):
                        one_cel_feature.append(1)
                    else:
                        one_cel_feature.append(0)
                if index1==10:#font_bold
                    if(item==True):
                        one_cel_feature.append(1)
                    else:
                        one_cel_feature.append(0)
                if index1==11:#height
                    one_cel_feature.append(item)
                if index1==12:#width
                    one_cel_feature.append(item)
                if index1==13:#content
                    if is_number(str(item)):
                        cell_type = 1
                    elif str(item) == '':
                        cell_type = 0
                    else:
                        cell_type = 2
                    if not str(item) in self.content_dict:
                        self.content_dict[str(item)] = len(self.content_dict)+1
                    one_cel_feature.append(self.content_dict[str(item)])


                if index1==14:#content_template
                    if not str(item) in self.content_tem_dict:
                        self.content_tem_dict[str(item)] = len(self.content_tem_dict)+1
                    one_cel_feature.append(self.content_tem_dict[str(item)])

                    one_cel_feature.append(cell_type)
            result.append(one_cel_feature)
  
        with open("json_data/content_dict_2.json", 'w') as f:
            json.dump(self.content_dict, f)
        with open("json_data/content_temp_dict_2.json", 'w') as f:
            json.dump(self.content_tem_dict, f)
        
        return result

    def transfer_origin_feature(self,origin_feature):
        feature = []
        for one_cell_feature in origin_feature['sheetfeature']:
            new_feature = []
            for key1 in one_cell_feature:
                new_feature.append(one_cell_feature[key1])
            feature.append(new_feature)
        feature = self.get_new_color_feature_vector(feature)
        return feature

    # def generate_dvinfos(self, path, save_path):
    #     with open(path, 'r') as f:
    #         c_dvinfos = json.load(f)

      
    #     for index,dvinfo in enumerate(c_dvinfos):
    #         print(index, len(c_dvinfos))
    #         self.dvid2filesheet[dvinfo['ID']] = dvinfo['FileName'] + '---' + dvinfo['SheetName']
    #         self.dvid2dvvalue[dvinfo['ID']] = dvinfo['Value']
    #         if os.path.exists('../AnalyzeDV/DVFeaturesDictionary/'+str(dvinfo['ID']) + '.json') and os.path.exists('../AnalyzeDV/StartRowColumn/'+str(dvinfo['ID']) + '.json') and os.path.exists('../AnalyzeDV/FeaturesDictionary/'+str(dvinfo['ID']) + '.json'):
    #             with open('../AnalyzeDV/DVFeaturesDictionary/'+str(dvinfo['ID']) + '.json', 'r') as f:
    #                 origin_feature = json.load(f)
    #                 # print(origin_feature.keys())
    #                 # print(np.array(origin_feature['sheetfeature']).shape)
    #                 self.dvid2dvsheetfeature[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = self.get_feature_vector_with_bert(origin_feature['sheetfeature'])
    #                 # self.dvid2dvsheetfeature[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = self.transfer_origin_feature(origin_feature)
    #                 print('dvid2dvsheetfeature shape', np.array(self.dvid2dvsheetfeature[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])]).shape)
    #             with open('../AnalyzeDV/FeaturesDictionary/'+str(dvinfo['ID']) + '.json', 'r') as f:
    #                 origin_feature = json.load(f)
    #                 # self.dvid2sheetfeature[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = self.transfer_origin_feature(origin_feature)
    #                 self.dvid2sheetfeature[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = self.get_feature_vector_with_bert(origin_feature['sheetfeature'])
    #             with open('../AnalyzeDV/StartRowColumn/'+ str(dvinfo['ID']) + '.json', 'r') as f:
    #                 self.dvid2dvstartrc[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = json.load(f)
    #             self.custom_dvinfos.append(dvinfo)
    #     print(len(self.custom_dvinfos))
    #     with open(save_path + '_dvid2dvsheetfeature.json', 'w') as f:
    #         json.dump(self.dvid2dvsheetfeature ,f)
    #     with open(save_path + '_dvid2sheetfeature.json', 'w') as f:
    #         json.dump(self.dvid2sheetfeature ,f)
    #     with open(save_path + '_dvid2dvstartrc.json', 'w') as f:
    #         json.dump(self.dvid2dvstartrc ,f)
    #     with open(save_path + '_custom_dvinfos.json', 'w') as f:
    #         json.dump(self.custom_dvinfos ,f)

    def generate_l2_formulafeatures(self):
        self.model = torch.load('196model/model_l2norm_3_2')
        filenames = os.listdir('deal00formula2beforebertfeature/')
        saved_filenames = os.listdir('l2_formula2afterfeature')

        all_files = os.listdir('formula2beforebertfeature')
        other_filenames = list(set(all_files) - set(filenames))
        for filename in filenames:
            # if filename.split('.')[0] + '.npy' not in saved_filenames:
            if fileanme in saved_filenames:
                continue
            feature = np.load('deal00formula2beforebertfeature/'+filename, allow_pickle=True)
            feature = Variable(feature).to(torch.float32)
            feature = self.model(feature).detach().numpy()
            np.save('l2_filename2afterbertfeature/' + filename.replace('.json',''), feature)

        for filename in other_filenames:
            if fileanme in saved_filenames:
                continue
            feature = np.load('formula2beforebertfeature/'+filename, allow_pickle=True)
            feature = Variable(feature).to(torch.float32)
            feature = self.model(feature).detach().numpy()
            np.save('l2_filename2afterbertfeature/' + filename.replace('.json',''), feature)

    def save_bert_feature_for_1900(self):
        filenames = os.listdir('../AnalyzeDV/SheetFeatures/')
        saved_filenames = os.listdir('filename2bertfeature')
        for filename in filenames:
            # if filename.split('.')[0] + '.npy' not in saved_filenames:
            with open("../AnalyzeDV/SheetFeatures/" + filename, 'r') as f:
                origin_feature = json.load(f)
            print(filename)
            # print(self.get_feature_vector_with_bert(origin_feature['sheetfeature']))
            feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
            print(feature_nparray.shape)
            res =  torch.DoubleTensor(feature_nparray)
            feature = res.reshape(1,100,10,399)
            if self.l2:
                np.save('l2_filename2beforebertfeature/' + filename.replace('.json',''), feature)
            else:
                np.save('filename2beforebertfeature/' + filename.replace('.json',''), feature)
            feature = Variable(feature).to(torch.float32)
            feature = self.model(feature).detach().numpy()
            if self.l2:
                np.save('l2_filename2afterbertfeature/' + filename.replace('.json',''), feature)
            else:
                np.save('filename2afterbertfeature/' + filename.replace('.json','') + '.npy', feature)

    # def save_bert_feature_for_fine_tune(self):
    #     with open('fine_tune_training_pair.json', 'r') as f:
    #         training_pairs = json.load(f)
    #     for index, triple in enumerate(training_pairs):
    #         print('training', index, len(training_pairs))
    #         formula_token1 = triple[0].split('/')[-1]
    #         formula_token2 = triple[1].split('/')[-1]
    #         if not os.path.exists('input_feature_finetune/' + formula_token1+'.npy'):
    #             with open("fixed_formulas_training/" + formula_token1 + '.json', 'r') as f:
    #                 origin_feature = json.load(f)
    #             feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
    #             np.save('input_feature_finetune/' + formula_token1 + '.npy', feature_nparray)
    #         if not os.path.exists('input_feature_finetune/' + formula_token2+'.npy'):
    #             with open("fixed_formulas_training/" + formula_token2+ '.json', 'r') as f:
    #                 origin_feature = json.load(f)
    #             feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
    #             np.save('input_feature_finetune/' + formula_token2 + '.npy', feature_nparray)

    #     with open('fine_tune_testing_pair.json', 'r') as f:
    #         testing_pairs = json.load(f)
    #     for triple in testing_pairs:
    #         formula_token1 = triple[0].split('/')[-1]
    #         formula_token2 = triple[1].split('/')[-1]
    #         print('testing', index, len(testing_pairs))
    #         if not os.path.exists('input_feature_finetune/' + formula_token1+'.npy'):
    #             with open("fixed_formulas_training/" + formula_token1 + '.json', 'r') as f:
    #                 origin_feature = json.load(f)
    #             feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
    #             np.save('input_feature_finetune/' + formula_token1 + '.npy', feature_nparray)
    #         if not os.path.exists('input_feature_finetune/' + formula_token2+'.npy'):
    #             with open("fixed_formulas_training/" + formula_token2+ '.json', 'r') as f:
    #                 origin_feature = json.load(f)
    #             feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
    #             np.save('input_feature_finetune/' + formula_token2 + '.npy', feature_nparray)

    def save_bert_feature_for_fine_tune(self):
        with open('saved_finetune_triplet_sampled.json', 'r') as f:
            training_pairs = json.load(f)

        new_res = []
        for index, triple in enumerate(training_pairs):
            print(index, len(training_pairs))
            formula_token1 = triple[0].split('/')[-1]
            formula_token2 = triple[1].split('/')[-1]
            formula_token3 = triple[2].split('/')[-1]
            if not os.path.exists('input_feature_finetune/' + formula_token1+'.npy'):
                if not os.path.exists("fixed_formulas_training/" + formula_token1 + '.json'):
                    continue
                with open("fixed_formulas_training/" + formula_token1 + '.json', 'r') as f:
                    origin_feature = json.load(f)
                feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
                np.save('input_feature_finetune/' + formula_token1 + '.npy', feature_nparray)
            if not os.path.exists('input_feature_finetune/' + formula_token2+'.npy'):
                if not os.path.exists("fixed_formulas_training/" + formula_token2 + '.json'):
                    continue
                with open("fixed_formulas_training/" + formula_token2+ '.json', 'r') as f:
                    origin_feature = json.load(f)
                feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
                np.save('input_feature_finetune/' + formula_token2 + '.npy', feature_nparray)
            if not os.path.exists('input_feature_finetune/' + formula_token3+'.npy'):
                if not os.path.exists("fixed_formulas_training/" + formula_token3 + '.json'):
                    continue
                with open("fixed_formulas_training/" + formula_token3+ '.json', 'r') as f:
                    origin_feature = json.load(f)
                feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
                np.save('input_feature_finetune/' + formula_token3 + '.npy', feature_nparray)
            new_res.append(triple)
        with open('saved_finetune_triplet_sampled.json', 'w') as f:
            json.dump(new_res, f)

    
            
    def save_bert_feature_for_neighbor_fine_tune(self, thread_id, batch_num):
        with open('saved_finetune_neighbor_triplet_sampled.json', 'r') as f:
            training_pairs = json.load(f)

        new_res = []
        batch_len = len(training_pairs)/batch_num
        for index, triple in enumerate(training_pairs):
            
            if thread_id != batch_num:
                if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                    continue
            else:
                if index <= batch_len * (thread_id - 1 ):
                    continue
            if index == 51865:
                continue
            print(index, len(training_pairs))
            formula_token1 = triple[0].split('/')[-1]
            formula_token2 = triple[1].split('/')[-1]
            formula_token3 = triple[2].split('/')[-1]
            # if not os.path.exists('input_feature_neighbor_finetune/' + formula_token1+'.npy'):
            #     if not os.path.exists("fixed_formulas_training/" + formula_token1 + '.json'):
            #         print('no fixed neighbors')
            #         continue
            #     with open("fixed_formulas_training/" + formula_token1 + '.json', 'r') as f:
            #         origin_feature = json.load(f)
            #     feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
            #     np.save('input_feature_neighbor_finetune/' + formula_token1 + '.npy', feature_nparray)
            #     print('1:saving...')
            # else:
            #     print("1:exists...")
            # if not os.path.exists('input_feature_neighbor_finetune/' + formula_token2+'.npy'):
            #     if not os.path.exists("fixed_formulas_training/" + formula_token2 + '.json'):
            #         print('no fixed neighbors')
            #         continue
            #     with open("fixed_formulas_training/" + formula_token2+ '.json', 'r') as f:
            #         origin_feature = json.load(f)
            #     feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
            #     np.save('input_feature_neighbor_finetune/' + formula_token2 + '.npy', feature_nparray)
            #     print('2:saving...')
            # else:
            #     print("2:exists...")
            if not os.path.exists('input_feature_neighbor_finetune/' + formula_token3+'.npy'):
                if not os.path.exists("fixed_neighbors_training/" + formula_token3 + '.json'):
                    print('no fixed neighbors')
                    continue
                with open("fixed_neighbors_training/" + formula_token3+ '.json', 'r') as f:
                    origin_feature = json.load(f)
                feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
                np.save('input_feature_neighbor_finetune/' + formula_token3 + '.npy', feature_nparray)
                print('3:saving...')
            else:

                print("3:exists...")
            new_res.append(triple)
 
            
    def save_bert_feature_for_fine_tune_triplet(self):
        with open('saved_finetune_triplet_sampled.json', 'r') as f:
            finetune_triplet = json.load(f)
        res = []
        for index, triple in enumerate(finetune_triplet):
            print('training', index, len(finetune_triplet))
            formula_token1 = triple[0].split('/')[-1]
            formula_token2 = triple[1].split('/')[-1]
            formula_token3 = triple[2].split('/')[-1]
            print('#####')
            print('formula_token1', formula_token1)
            print('formula_token2', formula_token2)
            print('formula_token3', formula_token3)
            if not os.path.exists('input_feature_finetune_more/' + formula_token1+'.npy'):
                if not os.path.exists("fixed_formulas_training/" + formula_token1 + '.json'):
                    continue
                with open("fixed_formulas_training/" + formula_token1 + '.json', 'r') as f:
                    origin_feature = json.load(f)
                feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
                count = 0
                feature_1010 = []
                for row in range(0,100):
                    for col in range(0,10):
                        if row < 45 or row > 54:
                            count += 1
                            continue
                        feature_1010.append(feature_nparray[count])
                        count += 1

                np.save('input_feature_finetune_more/' + formula_token1 + '.npy', feature_1010)
            if not os.path.exists('input_feature_finetune_more/' + formula_token2+'.npy'):
                if not os.path.exists("fixed_formulas_training/" + formula_token2 + '.json'):
                    continue
                with open("fixed_formulas_training/" + formula_token2+ '.json', 'r') as f:
                    origin_feature = json.load(f)
                feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
                count = 0
                feature_1010 = []
                for row in range(0,100):
                    for col in range(0,10):
                        if row < 45 or row > 54:
                            count += 1
                            continue
                        feature_1010.append(feature_nparray[count])
                        count += 1
                np.save('input_feature_finetune_more/' + formula_token2 + '.npy', feature_nparray)
            if not os.path.exists('input_feature_finetune_more/' + formula_token3+'.npy'):
                if not os.path.exists("fixed_formulas_training/" + formula_token3 + '.json'):
                    continue
                with open("fixed_formulas_training/" + formula_token3+ '.json', 'r') as f:
                    origin_feature = json.load(f)
                feature_nparray = np.array(self.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
                count = 0
                feature_1010 = []
                for row in range(0,100):
                    for col in range(0,10):
                        if row < 45 or row > 54:
                            count += 1
                            continue
                        feature_1010.append(feature_nparray[count])
                        count += 1
                np.save('input_feature_finetune_more/' + formula_token3 + '.npy', feature_nparray)
            if os.path.exists("fixed_formulas_training/" + formula_token1 + '.json') and os.path.exists("fixed_formulas_training/" + formula_token2 + '.json') and os.path.exists("fixed_formulas_training/" + formula_token3 + '.json') :
                res.append(triple)
        with open('saved_finetune_triplet_more.json', 'w') as f:
            json.dump(res, f)
    def generate_dvinfos(self, path, save_path):
        with open(path, 'r') as f:
            c_dvinfos = json.load(f)

        with open(save_path + '_notmask_dvid2dvsheetfeature.json', 'r') as f:
            self.dvid2dvsheetfeature = json.load(f)
        with open(save_path + '_notmask_dvid2sheetfeature.json', 'r') as f:
            self.dvid2sheetfeature = json.load(f)
        with open(save_path + '_notmask_dvid2dvstartrc.json', 'r') as f:
            self.dvid2dvstartrc = json.load(f)
        with open(save_path + '_notmask_custom_dvinfos.json', 'r') as f:
            self.custom_dvinfos = json.load(f)
        ids = [i['ID'] for i in self.custom_dvinfos]
        for index,dvinfo in enumerate(c_dvinfos):
            # if index >= 5500 and index <= 8000:
                # continue
            if dvinfo['ID'] in ids:
                continue
            if index == 5506 or index == 3402:
                continue
            print(index, len(c_dvinfos))
            self.dvid2filesheet[dvinfo['ID']] = dvinfo['FileName'] + '---' + dvinfo['SheetName']
            self.dvid2dvvalue[dvinfo['ID']] = dvinfo['Value']
            if os.path.exists('../AnalyzeDV/NMDVFeaturesDictionary/'+str(dvinfo['ID']) + '.json') and os.path.exists('../AnalyzeDV/StartRowColumn/'+str(dvinfo['ID']) + '.json') and os.path.exists('../AnalyzeDV/FeaturesDictionary/'+str(dvinfo['ID']) + '.json'):
                with open('../AnalyzeDV/NMDVFeaturesDictionary/'+str(dvinfo['ID']) + '.json', 'r') as f:
                    origin_feature = json.load(f)
                    # print(origin_feature.keys())
                    # print(np.array(origin_feature['sheetfeature']).shape)
                    self.dvid2dvsheetfeature[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = self.get_feature_vector_with_bert(origin_feature['sheetfeature'])
                    # self.dvid2dvsheetfeature[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = self.transfer_origin_feature(origin_feature)
                    print('dvid2dvsheetfeature shape', np.array(self.dvid2dvsheetfeature[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])]).shape)
                with open('../AnalyzeDV/FeaturesDictionary/'+str(dvinfo['ID']) + '.json', 'r') as f:
                    origin_feature = json.load(f)
                    # self.dvid2sheetfeature[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = self.transfer_origin_feature(origin_feature)
                    self.dvid2sheetfeature[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = self.get_feature_vector_with_bert(origin_feature['sheetfeature'])
                with open('../AnalyzeDV/StartRowColumn/'+ str(dvinfo['ID']) + '.json', 'r') as f:
                    self.dvid2dvstartrc[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = json.load(f)
                self.custom_dvinfos.append(dvinfo)
            if index % 1000 == 0:
                print(len(self.custom_dvinfos))
                with open(save_path + '_notmask_dvid2dvsheetfeature.json', 'w') as f:
                    json.dump(self.dvid2dvsheetfeature ,f)
                with open(save_path + '_notmask_dvid2sheetfeature.json', 'w') as f:
                    json.dump(self.dvid2sheetfeature ,f)
                with open(save_path + '_notmask_dvid2dvstartrc.json', 'w') as f:
                    json.dump(self.dvid2dvstartrc ,f)
                with open(save_path + '_notmask_custom_dvinfos.json', 'w') as f:
                    json.dump(self.custom_dvinfos ,f)
        with open(save_path + '_notmask_dvid2dvsheetfeature.json', 'w') as f:
            json.dump(self.dvid2dvsheetfeature ,f)
        with open(save_path + '_notmask_dvid2sheetfeature.json', 'w') as f:
            json.dump(self.dvid2sheetfeature ,f)
        with open(save_path + '_notmask_dvid2dvstartrc.json', 'w') as f:
            json.dump(self.dvid2dvstartrc ,f)
        with open(save_path + '_notmask_custom_dvinfos.json', 'w') as f:
            json.dump(self.custom_dvinfos ,f)

    def load_mix_dvinfos(self):
        with open("../AnalyzeDV/data/types/custom/custom_list.json") as f:
            custom_dvinfos = json.load(f)
        with open("sampled_list_list.json") as f:
            sampled_list_list = json.load(f)
        with open("sampled_boundary_list.json") as f:
            sampled_boundary_list = json.load(f)

        for c_dvinfos in [custom_dvinfos, sampled_list_list, sampled_boundary_list]:
            for index,dvinfo in enumerate(c_dvinfos):
                self.dvid2filesheet[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = dvinfo['FileName'] + '---' + dvinfo['SheetName']
                self.dvid2dvvalue[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = dvinfo['Value']

        with open('mix_dvid2dvsheetfeature.json', 'r') as f:
            self.dvid2dvsheetfeature = json.load(f)
        with open('mix_dvid2sheetfeature.json', 'r') as f:
            self.dvid2sheetfeature = json.load(f)
        with open('mix_dvid2dvstartrc.json', 'r') as f:
            self.dvid2dvstartrc = json.load(f)
        with open('mix_dvinfos.json', 'r') as f:
            self.custom_dvinfos = json.load(f)

    def look_custom(self):
        with open('custom_dvid2sheetfeature.json', 'r') as f:
            self.dvid2sheetfeature = json.load(f)
        with open("../AnalyzeDV/data/types/custom/custom_list.json") as f:
            self.custom_dvinfos = json.load(f)
        count =0
        for dvinfo in self.custom_dvinfos:
            print(str(dvinfo['ID'])+'---'+str(dvinfo['batch_id']))
            
            if str(dvinfo['ID'])+'---'+str(dvinfo['batch_id']) in self.dvid2sheetfeature.keys():
                count += 1
        print(self.dvid2sheetfeature.keys())
        print(count)
    
    def load_custom_dvinfos(self):
        with open("../AnalyzeDV/data/types/custom/custom_list.json") as f:
            c_dvinfos = json.load(f)

        for index,dvinfo in enumerate(c_dvinfos):
            self.dvid2filesheet[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = dvinfo['FileName'] + '---' + dvinfo['SheetName']
            self.dvid2dvvalue[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = dvinfo['Value']


        
        with open('custom_dvid2dvsheetfeature.json', 'r') as f:
            self.dvid2dvsheetfeature = json.load(f)
        with open('custom_dvid2sheetfeature.json', 'r') as f:
            self.dvid2sheetfeature = json.load(f)
        with open('custom_dvid2dvstartrc.json', 'r') as f:
            self.dvid2dvstartrc = json.load(f)
        # with open('custom_custom_dvinfos.json', 'r') as f:
        self.custom_dvinfos = c_dvinfos

    def norm(self, batches):
        length = 0
        new_batch = []
        for vec in batches:
            for i in vec:
                # print(i)
                length += i*i
            length = length ** 0.5
            res = []
            for i in vec:
                res.append(i/length)
            new_batch.append(np.array(res))
        return np.array(new_batch)
    def find_closed_sheet(self, l2=True):
        count = 0
        for dvinfo in self.custom_dvinfos:
            print(count, len(self.custom_dvinfos))
            count += 1
            # if str(dvinfo['ID'])+'---'+str(dvinfo['batch_id']) not in self.dvid2sheetfeature:
            #     continue
            result_sheet = []
            add_sheet = []
            custom_dvinfos_embeddings = []
            ids = []
            count_ = 1
            id2dvid = {}
            if str(dvinfo['ID'])+'---'+str(dvinfo['batch_id']) not in self.dvid2sheetfeature:
                continue
            # print(list(self.dvid2filesheet.keys())[0])
            for id_ in self.dvid2sheetfeature:

                if id_ not in self.dvid2filesheet.keys():
                    continue
                # print(self.dvid2filesheet[id_])
                # print(self.dvid2filesheet[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])])
                if self.dvid2filesheet[id_] in add_sheet or self.dvid2filesheet[id_] == self.dvid2filesheet[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])]:
                    continue
                if l2:
                    # print(self.dvid2sheetfeature[str(id_)][0])
                    feature = torch.DoubleTensor(np.array([self.dvid2sheetfeature[str(id_)]]))
                    feature = feature.reshape(1,100,10,16).permute(0,3,1,2)
                    embeded_feature = self.model(feature)
                    print(embeded_feature.shape)
                    custom_dvinfos_embeddings.append(embeded_feature)
                else:
                    custom_dvinfos_embeddings.append(self.dvid2sheetfeature[str(id_)])
                ids.append(count_)
                id2dvid[count_] = id_
                count_ += 1
                add_sheet.append(self.dvid2filesheet[id_])
            
            custom_dvinfos_embeddings = np.array(custom_dvinfos_embeddings).reshape((len(custom_dvinfos_embeddings), 16000))
            print(custom_dvinfos_embeddings[0])
            ids =  np.array(ids)
            custom_dvinfos_embeddings = custom_dvinfos_embeddings.astype('float32')

            index = faiss.IndexFlatL2(len(custom_dvinfos_embeddings[0]))
            index2 = faiss.IndexIDMap(index)
            index2.add_with_ids(custom_dvinfos_embeddings, ids)
    
            search_list = np.array(self.dvid2sheetfeature[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])]).reshape((1,16000)).astype('float32')
      
            D, I = index.search(np.array(search_list), 20) # sanity check
     
            top_k = []
            for index,i in enumerate(I[0]):
                top_k.append((self.dvid2filesheet[id2dvid[ids[i]]], float(D[0][index])))
    
            self.dvid2stop[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = top_k
    
        with open(self.dvid2stop_name, 'w') as f:
            json.dump(self.dvid2stop,f)

    def find_most_simular_dv(self, l2=True):
        with open(self.dvid2stop_name, 'r') as f:
            self.dvid2stop = json.load(f)

        c = 0
        
        for dvinfo in self.custom_dvinfos:
            print(c, len(self.custom_dvinfos))
            c+=1 
            if str(dvinfo['ID'])+'---'+str(dvinfo['batch_id']) not in self.dvid2stop:
                continue
            candidates_id = []
            candidates_emb = []
            result_cand = []
        
            id2dvid = {}
            count_ = 1

            for cand_id in self.dvid2filesheet:
                # if count_ == 0:
                #     count_ += 1
                #     continue
                
                if cand_id not in list(self.dvid2filesheet.keys()):
                    # print(list(self.dvid2stop.keys())[0][0])
                    print(cand_id)
                    continue
                # count_ += 1
                # print("found")
                # print(np.array(self.dvid2stop[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])])[:, 0])
                # print(self.dvid2dvsheetfeature.keys())
                if self.dvid2filesheet[cand_id] in np.array(self.dvid2stop[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])])[:, 0]:
                    if cand_id not in self.dvid2dvsheetfeature:
                        continue
                    candidates_id.append(count_)
                    id2dvid[count_] = cand_id
                    count_ += 1
                    if l2:
                        candidates_emb.append(self.norm(self.dvid2dvsheetfeature[str(cand_id)]))
                    else:
                        candidates_emb.append(self.dvid2dvsheetfeature[str(cand_id)])

            candidates_emb = np.array(candidates_emb).reshape((len(candidates_emb), 16000)).astype('float32')
            candidates_id =  np.array(candidates_id)
    
            if len(candidates_emb) == 0:
                self.dvid2dtop[str(dvinfo['ID'])+"---"+str(dvinfo['batch_id'])] = []
                continue
            index = faiss.IndexFlatL2(len(candidates_emb[0]))
            index2 = faiss.IndexIDMap(index)
            
            candidates_emb = candidates_emb.reshape((len(candidates_emb), 16000)).astype('float32')
            index2.add_with_ids(candidates_emb, candidates_id)
    
            search_list = np.array(self.dvid2dvsheetfeature[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])]).reshape((1,16000)).astype('float32')
            D, I = index.search(np.array(search_list), 20) # sanity check
            top_k = []
            for index,i in enumerate(I[0]):
                top_k.append((id2dvid[candidates_id[i]], float(D[0][index])))
            self.dvid2dtop[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])] = top_k
        
        with open(self.dvid2dtop_name, 'w') as f:
            json.dump(self.dvid2dtop,f)

    def evaluate(self, eval_type='last_only', thr=20):
        # print('self.dvid2dtop', self.dvid2dtop)
        # print('self.dvid2stop', self.dvid2stop)
        dvid2rank = {}
        if eval_type=='last_only': 
            for dvid in self.dvid2dtop:
                dvid2rank[dvid] = []
                for cand in self.dvid2dtop[dvid]:
                    if thr == 0:
                        dvid2rank[dvid].append(cand[0])
                    else:
                        if cand[1] < thr:
                            # print(cand[0])
                            dvid2rank[dvid].append(cand[0])
        elif eval_type=='multi_cos':
            dvid2cos = {}
            for dvid in self.dvid2dtop:
                temp = []
                for cand in self.dvid2dtop[dvid]:
                    cand_dvid = cand[0]
                    cand_sheet = self.dvid2filesheet[cand_dvid]
                    ddis = cand[1]
                    for sheet in self.dvid2stop[dvid]:
                        if sheet[0] == cand_sheet:
                            sdis = sheet[1]

                    dcos = self.distance2cos(ddis)
                    scos = self.distance2cos(sdis)
                    temp.append([cand_dvid, dcos*scos])
                temp = sorted(temp, key=lambda e: e[1])
                dvid2cos[dvid] = temp
                
            for dvid in dvid2cos:
                dvid2rank[dvid] = []
                for cand in dvid2cos[dvid]:
                    if thr == 0:
                        dvid2rank[dvid].append(cand[0])
                    else:
                        if cand[1] > thr:
                            print(cand[0], cand[1])
                            dvid2rank[dvid].append(cand[0])

        has_cand_dv = []
        for item in dvid2rank:
            if len(dvid2rank[item]) != 0:
                has_cand_dv.append(item)


        hit = 0
        all_ = 0
        not_hit = []
        for dvinfo in self.custom_dvinfos:
            result_value = []
            if str(dvinfo['ID'])+"---"+str(dvinfo['batch_id']) not in dvid2rank:
                continue
            for cand_id in dvid2rank[str(dvinfo['ID'])+"---"+str(dvinfo['batch_id'])]:
                result_value.append(self.dvid2dvvalue[cand_id])
            # print(dvinfo['Value'])
            # print(result_value)
            # print()
            if dvinfo['Value'] in result_value:
                hit+=1
            else:
                not_hit.append(dvinfo)
            all_ += 1
       
        print('thr', thr)
        print("has candidate dv", len(has_cand_dv))
        print('hit', hit)
        print('all_', all_)

        # for dvinfo in not_hit:
        #     if len(dvid2rank[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])]) == 0:
        #         continue
        #     print('##################')
        #     print(str(dvinfo['ID'])+'---'+str(dvinfo['batch_id']), dvinfo['SheetName'], dvinfo["FileName"])
        #     print(dvid2rank[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])][0], self.dvid2filesheet[self.dvid2dtop[str(dvinfo['ID'])+'---'+str(dvinfo['batch_id'])][0]])

    def pipeline(self, data_type='mix', l2=True):
        if data_type == 'custom':
            self.load_custom_dvinfos()
        if data_type == 'mix':
            self.load_mix_dvinfos()
        self.find_closed_sheet(l2=l2)
        self.find_most_simular_dv(l2=l2)
        # self.evaluate()


def look_custom_at_least_coverage():
    with open("../AnalyzeDV/data/types/custom/change_xml_custom_list.json") as f:
        c_dvinfos = json.load(f)

    same_dvinfo = {}
    for dvinfo in c_dvinfos:
        same_dvinfo[dvinfo['ID']] = []
        for another_dvinfo in c_dvinfos:
            if dvinfo['Value'] == another_dvinfo['Value'] and dvinfo['SheetName'] == another_dvinfo['SheetName'] and dvinfo['FileName'] != another_dvinfo['FileName']:
                same_dvinfo[dvinfo['ID']].append(another_dvinfo)

    has_same_dvinfo = 0
    all_ = 0
    for item in same_dvinfo:
        all_ += 1
        if len(same_dvinfo[item]) > 0:
            has_same_dvinfo += 1
    print('has_same_dvinfo', has_same_dvinfo)
    print('all_', all_)

def look_boudnary_at_least_coverage():
    # with open("../AnalyzeDV/data/types/boundary/boundary_list.json") as f:
    # with open("sampled_boundary_list.json") as f:
    with open("sampled_list_list.json") as f:
        c_dvinfos = json.load(f)

    same_dvinfo = {}
    for index, dvinfo in enumerate(c_dvinfos):
        print(index, len(c_dvinfos))
        same_dvinfo[dvinfo['ID']] = []
        for another_dvinfo in c_dvinfos:
            if dvinfo['Type'] == another_dvinfo['Type'] and dvinfo['Operator'] == another_dvinfo['Operator'] and dvinfo['Value'] == another_dvinfo['Value'] and dvinfo['SheetName'] == another_dvinfo['SheetName'] and dvinfo['FileName'] != another_dvinfo['FileName']:
                same_dvinfo[dvinfo['ID']].append(another_dvinfo)

    has_same_dvinfo = 0
    all_ = 0
    for item in same_dvinfo:
        all_ += 1
        if len(same_dvinfo[item]) > 0:
            has_same_dvinfo += 1
    print('has_same_dvinfo', has_same_dvinfo)
    print('all_', all_)

def run_version(data_type='mix', l2=True):
    if l2:
        dvid2stop_name=data_type+'_dvid2stop20_l2norm.json'
        dvid2dtop_name=data_type+'_dvid2dtop20_l2norm.json'
    else:
        dvid2stop_name=data_type+'_dvid2stop20.json'
        dvid2dtop_name=data_type+'_dvid2dtop20.json'
    e2e = End2End(dvid2dtop_name, dvid2stop_name)
    e2e.pipeline(data_type, l2)

def run_eval(data_type):
    dvid2stop_name=data_type+'_dvid2stop20.json'
    dvid2dtop_name=data_type+'_dvid2dtop20.json'
    e2e = End2End(dvid2dtop_name, dvid2stop_name)
    e2e.load_custom_dvinfos()
    with open(self.dvid2dtop_name, 'r') as f:
        self.dvid2dtop = json.load(f)
    with open(self.dvid2stop_name, 'r') as f:
        self.dvid2stop = json.load(f)
    
    dlen = 0
    for thr in [0,0.002,0.004,0.006,0.008,0.01,0.2,0.4,0.6,0.8,1]:
        print('##############')
        e2e.evaluate(eval_type='multi_cos', thr=thr)

def generate_dvinfos_boundary_list():
    e2e = End2End()
    # e2e.generate_dvinfos("sampled_list_list.json", "list_bert")
    # e2e.generate_dvinfos("sampled_boundary_list.json", "boundary_bert")
    e2e.generate_dvinfos("../AnalyzeDV/data/types/custom/custom_list.json", "custom_bert")
    # with open('custom_bert_dvid2dvsheetfeature.json', 'r') as f:
    #     custom_dvid2dvsheetfeature = json.load(f)
    # with open('custom_bert_dvid2sheetfeature.json', 'r') as f:
    #     custom_dvid2sheetfeature = json.load(f)
    # with open('custom_bert_dvid2dvstartrc.json', 'r') as f:
    #     custom_dvid2dvstartrc = json.load(f)
    # with open('custom_bert_custom_dvinfos.json', 'r') as f:
    #     custom_custom_dvinfos = json.load(f)

    # with open('boundary_bert_dvid2dvsheetfeature.json', 'r') as f:
    #     boundary_dvid2dvsheetfeature = json.load(f)
    # with open('boundary_bert_dvid2sheetfeature.json', 'r') as f:
    #     boundary_dvid2sheetfeature = json.load(f)
    # with open('boundary_bert_dvid2dvstartrc.json', 'r') as f:
    #     boundary_dvid2dvstartrc = json.load(f)
    # with open('boundary_bert_custom_dvinfos.json', 'r') as f:
    #     boundary_custom_dvinfos = json.load(f)

    # with open('list_bert_dvid2dvsheetfeature.json', 'r') as f:
    #     list_dvid2dvsheetfeature = json.load(f)
    # with open('list_bert_dvid2sheetfeature.json', 'r') as f:
    #     list_dvid2sheetfeature = json.load(f)
    # with open('list_bert_dvid2dvstartrc.json', 'r') as f:
    #     list_dvid2dvstartrc = json.load(f)
    # with open('list_bert_custom_dvinfos.json', 'r') as f:
    #     list_custom_dvinfos = json.load(f)

    # dvid2dvsheetfeature = {}
    # for key in custom_dvid2dvsheetfeature:
    #     dvid2dvsheetfeature[key] = custom_dvid2dvsheetfeature[key]
    # for key in boundary_dvid2dvsheetfeature:
    #     dvid2dvsheetfeature[key] = boundary_dvid2dvsheetfeature[key]
    # for key in list_dvid2dvsheetfeature:
    #     dvid2dvsheetfeature[key] = list_dvid2dvsheetfeature[key]

    # dvid2sheetfeature = {}
    # for key in custom_dvid2sheetfeature:
    #     dvid2sheetfeature[key] = custom_dvid2sheetfeature[key]
    # for key in boundary_dvid2sheetfeature:
    #     dvid2sheetfeature[key] = boundary_dvid2sheetfeature[key]
    # for key in list_dvid2sheetfeature:
    #     dvid2sheetfeature[key] = list_dvid2sheetfeature[key]
    
    # dvid2dvstartrc = {}
    # for key in custom_dvid2dvstartrc:
    #     dvid2dvstartrc[key] = custom_dvid2dvstartrc[key]
    # for key in boundary_dvid2dvstartrc:
    #     dvid2dvstartrc[key] = boundary_dvid2dvstartrc[key]
    # for key in list_dvid2dvstartrc:
    #     dvid2dvstartrc[key] = list_dvid2dvstartrc[key]

    # custom_dvinfos = []
    # for key in custom_custom_dvinfos:
    #     custom_dvinfos.append(key)
    # for key in boundary_custom_dvinfos:
    #     custom_dvinfos.append(key)
    # for key in list_custom_dvinfos:
    #     custom_dvinfos.append(key)

    # with open('mix_bert_dvid2dvsheetfeature.json', 'w') as f:
    #     json.dump(dvid2dvsheetfeature ,f)
    # with open('mix_bert_dvid2sheetfeature.json', 'w') as f:
    #     json.dump(dvid2sheetfeature ,f)
    # with open('mix_bert_dvid2dvstartrc.json', 'w') as f:
    #     json.dump(dvid2dvstartrc ,f)
    # with open('mix_bert_dvinfos.json', 'w') as f:
    #     json.dump(custom_dvinfos ,f)

def split_formulas():
    filenames = os.listdir('../AnalyzeDV/FormulaFeatures77772/')
    print('filenames', len(filenames))
    filenames.sort()
    saved_filenames = os.listdir('formula2afterbertfeature')
    
    result = []
    for index,filename in enumerate(filenames):
    
        print(index, len(filenames))

        if filename.replace('.json', '.npy') in saved_filenames:
            continue
        result.append(filename)

    list197 = os.listdir('formulas197')
    list196 = os.listdir('formulas196')
    for index,filename in enumerate(result):
        if filename in list197 or filename in list196:
            continue
        # print(filename)
        # print('mv ../AnalyzeDV/FormulaFeatures77772/'+filename + ' formulas197/')

        if len(list196) > len(list197):
            os.system('mv ../AnalyzeDV/FormulaFeatures77772/'+filename.replace(' ', '\ ').replace('(', '\(').replace(')', '\)') + ' formulas197/')
        else:
            os.system('mv ../AnalyzeDV/FormulaFeatures77772/'+filename.replace(' ', '\ ').replace('(', '\(').replace(')', '\)') + ' formulas196/')



def save_bert_feature_for_77721(thread_id, batch_num, save_path='deal00formula2afterbertfeature/', model_path = '196model/cnn_new_dynamic_triplet_margin_1_3_12', load_path = 'deal00formula2beforebertfeature' ):#load_path = 'deal00formula2beforebertfeature'):
    # e2e = End2End()
    # if model_path:
    model = torch.load(model_path)
    # with open('need_rerun_list.json','r') as f:
    #     need_rerun_list = json.load(f)
    # print('thread_id', thread_id, batch_num)
    filenames = os.listdir('formulas197/')
    filenames1 = os.listdir('formulas196/')
    filenames += filenames1
    # with open('res197.json', 'r') as f:
    #     res197 = json.load(f)
    
    saved_filenames1 = os.listdir(save_path)
    filenames = list(set(filenames) - set(saved_filenames1)) 
    # filenames = res197
    print('filenames', len(filenames))
    filenames.sort()
    print('saved_filenames1', len(saved_filenames1))
    
    filename_list = []
    feature_list = []
    batch_len = len(filenames)/batch_num
    for index,filename in enumerate(filenames):
        # if filename != '00142e01-dc91-41a7-854f-59bf0a8ea170_aHR0cHM6Ly93d3cuYmsubXVmZy5qcC9lYnVzaW5lc3MvZS9ncGx1cy9wZGYvY3MvQ1NfQVBQMjAyXzIwMTkwMl9DT01TVUlURV9BcHBsaWNhdGlvbl9mb3JfVXNlcl9NYWludGVuYW5jZS54bHN4.xlsx---02_User(Common) (Additional)---8---10.json':
        #     continue
        # print(index, batch_len * (thread_id - 1 ), batch_len * (thread_id ))
        if thread_id != batch_num:
            if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                continue
        else:
            if(index <= batch_len * (thread_id - 1 )):
                continue
    
        print(index, len(filenames))

        # if filename.replace('.json', '.npy') in saved_filenames1:
        #     continue
        if os.path.exists(save_path + filename.replace('.json','') + '.npy'):
            print('exists.......')
            continue
        # print(filename)
        # print('load origin feature....')
        # if not os.path.exists(load_path + '/' + filename.replace('.json','.npy')):
        #     if not os.path.exists('formulas_deal_00/'+filename):
        #         continue
        #     with open("formulas_deal_00/" + filename, 'r') as f:
        #         origin_feature = json.load(f)
        #     # print('origin_feature', origin_feature)
        #     print(origin_feature.keys())
            
        #     # print(self.get_feature_vector_with_bert(origin_feature['sheetfeature']))
        #     print('transform to np array....')
        #     feature_nparray = np.array(e2e.get_feature_vector_with_bert_keyw(origin_feature['sheetfeature']))
        #     print(feature_nparray.shape)
        #     print('before model predict....')
        #     res =  torch.DoubleTensor(feature_nparray)
        #     feature = res.reshape(1,100,10,399)
        #     np.save(load_path + '/' + filename.replace('.json',''), feature)
        feature_nparray = np.load(load_path + '/' + filename.replace('.json','.npy'))
        feature_nparray = feature_nparray.reshape(1,100,10,399)
        model.eval()

        feature_nparray = torch.DoubleTensor(feature_nparray)
        feature_nparray = Variable(feature_nparray).to(torch.float32)
        feature_nparray = model(feature_nparray).detach().numpy()
        print('after model predict....')
            # print('feature_list', feature_list)
           
        np.save(save_path + filename.replace('.json','') + '.npy', feature_nparray)
       


def generate_1010_testing_data():

    formula_tokens = os.listdir('deal00formula2beforebertfeature')
    for index, formula_token in enumerate(formula_tokens):
        print(index, len(formula_tokens))
        if os.path.exists('model2_formula2beforebertfeature/'+formula_token):
            continue
        feature_10010 = np.load('deal00formula2beforebertfeature/'+formula_token, allow_pickle=True)[0]
        count = 0
        feature_1010 = []
        # print('feature_10010', feature_10010.shape)/
        for row in range(0,100):
            temp = []
            for col in range(0,10):
                if row < 45 or row > 54:
                    count += 1
                    continue
                temp.append(feature_10010[row][col])
                count += 1
            feature_1010.append(temp)

        np.save('model2_formula2beforebertfeature/'+formula_token, feature_1010)
        # break
def generate_1010_training_data():

    formula_tokens = os.listdir('input_feature_finetune')
    for index, formula_token in enumerate(formula_tokens):
        print(index, len(formula_tokens))
        if os.path.exists('input_feature_finetune_1010/'+formula_token):
            continue
        feature_10010 = np.load('input_feature_finetune/'+formula_token, allow_pickle=True)
        count = 0
        feature_1010 = []
        for row in range(0,100):
            for col in range(0,10):
                if row < 45 or row > 54:
                    count += 1
                    continue
                feature_1010.append(feature_10010[count])
                count += 1

        np.save('input_feature_finetune_1010/'+formula_token, feature_1010)
        # break

def para_save_input_neighbor():
    e2e = End2End()
    process = [Process(target=e2e.save_bert_feature_for_neighbor_fine_tune, args=(1,5)),
        Process(target=e2e.save_bert_feature_for_neighbor_fine_tune, args=(2,5)), 
        Process(target=e2e.save_bert_feature_for_neighbor_fine_tune, args=(3,5)),
        Process(target=e2e.save_bert_feature_for_neighbor_fine_tune, args=(4,5)), 
        Process(target=e2e.save_bert_feature_for_neighbor_fine_tune, args=(5,5)),
        # Process(target=look_positive, args=(6,20)), 
        # Process(target=look_positive, args=(7,20)),
        # Process(target=look_positive, args=(8,20)), 
        # Process(target=look_positive, args=(9,20)),
        # Process(target=look_positive, args=(10,20)), 
        # Process(target=look_positive, args=(11,20)),
        # Process(target=look_positive, args=(12,20)), 
        # Process(target=look_positive,args=(13,20)),
        # Process(target=look_positive, args=(14,20)), 
        # Process(target=look_positive, args=(15,20)),
        # Process(target=look_positive, args=(16,20)), 
        # Process(target=look_positive, args=(17,20)),
        # Process(target=look_positive, args=(18,20)), 
        # Process(target=look_positive, args=(19,20)), 
        # Process(target=look_positive, args=(20,20)), 
    ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束

def look_input1010():
    res = np.load("input_feature_finetune_1010/0114dc82a3c3130ef362db3f601b9fac_c3RyZWFtLW1lY2hhbmljcy5jb20JNTAuNjIuMTcyLjExMw==.xlsx---Monitoring Data---3---6.npy", allow_pickle=True)
    print(res)
    print(res.shape)

def saved_res():
    with open('finetune_neighbor_triplet_sampled.json', 'r') as f:
        training_pairs = json.load(f)

    res = []
    print('len(training_pairs)', len(training_pairs))
    filelist = os.listdir('fixed_neighbors_training')

    num =0
    for index,triple in enumerate(training_pairs):
        if triple[2].split("/")[-1] + '.json' not in filelist:
            continue
        num += 1
        print('num', num)
        res.append(triple)
    with open('saved_finetune_neighbor_triplet_sampled.json', 'w') as f:
        json.dump(res,f)
def move_196():
    filenames = os.listdir('formulas196')
    for filename in filenames:
        os.system('cp formulas_deal_00/'+filename.replace(' ', '\ ').replace('(', '\(').replace(')', '\)')  + ' formulas_deal_00_196/')

def para_save_after():
    process = [Process(target=save_bert_feature_for_77721, args=(1,20)),
        Process(target=save_bert_feature_for_77721, args=(2,20)), 
        Process(target=save_bert_feature_for_77721, args=(3,20)),
        Process(target=save_bert_feature_for_77721, args=(4,20)), 
        Process(target=save_bert_feature_for_77721, args=(5,20)),
        Process(target=save_bert_feature_for_77721, args=(6,20)), 
        Process(target=save_bert_feature_for_77721, args=(7,20)),
        Process(target=save_bert_feature_for_77721, args=(8,20)), 
        Process(target=save_bert_feature_for_77721, args=(9,20)),
        Process(target=save_bert_feature_for_77721, args=(10,20)), 
        Process(target=save_bert_feature_for_77721, args=(11,20)),
        Process(target=save_bert_feature_for_77721, args=(12,20)), 
        Process(target=save_bert_feature_for_77721,args=(13,20)),
        Process(target=save_bert_feature_for_77721, args=(14,20)), 
        Process(target=save_bert_feature_for_77721, args=(15,20)),
        Process(target=save_bert_feature_for_77721, args=(16,20)), 
        Process(target=save_bert_feature_for_77721, args=(17,20)),
        Process(target=save_bert_feature_for_77721, args=(18,20)), 
        Process(target=save_bert_feature_for_77721, args=(19,20)), 
        Process(target=save_bert_feature_for_77721, args=(20,20)), 
    ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束

def generate_test_formulas():
    with open("Formulas_")
if __name__ == '__main__':
    # split_formulas()
    # generate_1010_training_data()
    # look_input1010()
    
    # save_bert_feature_for_77721(e2e,1,1)
    # e2e = End2End()
    
    # save_bert_feature_for_77721(save_path='model2_formula2afterfeature/', model_path = 'model2/epoch_55', load_path = 'model2_formula2beforebertfeature',thread_id = 1,batch_num = 1)
    # e2e.save_bert_feature_for_fine_tune()
    # e2e.save_bert_feature_for_fine_tune_triplet()
    # e2e.save_bert_feature_for_neighbor_fine_tune(1,1)
    # saved_res()
    para_save_after()
    # with open('/datadrive/data/bert_dict/bert_dict_2.json', 'w') as f:
        # json.dump(bert_dict, f)
    # move_196()
    # e2e.save_bert_feature_for_77721(1, 8)
    
        
    # e2e.save_bert_feature_for_1900()
    # generate_l2_formulafeatures()
    # e2e.look_custom()
    # e2e.pipeline()
    # e2e.load_custom_dvinfos()
    # e2e.evaluate()
    # generate_dvinfos_boundary_list()
    # run_version(data_type='custom', l2=True)
    # run_eval(data_type='custom')
    # look_boudnary_at_least_coverage()
    # look_boudnary_at_least_coverage()
    # generate_1010_testing_data()
    # para_save_input_neighbor()
