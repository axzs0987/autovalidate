import json
import os
import numpy as np
import faiss
import pprint
import shutil
import difflib
from cnn_fine_tune import generate_demo_features, generate_one_after_feature, generate_one_before_feaure
# root_path = '/datadrive-2/data/top10domain_test/'
root_path = '/datadrive-2/data/fortune500_test/'
# root_path = '/datadrive-2/data/middle10domain_test/'
# import _thread
from multiprocessing import Process
def find_l2_closed_sheet(need_save):
    with open('Formula_160000.json','r') as f:
        formulas = json.load(f)
    
    if need_save:
        all_sheet_features = {}
        feature_files = os.listdir('l2_filename2afterbertfeature')
        feature_files.sort()
        for filename in feature_files:
            all_sheet_features[filename[:-4]] = np.load('l2_filename2afterbertfeature/' + filename, allow_pickle=True)[0]
        np.save('all_sheet_features.npy', all_sheet_features)
    else:
        all_sheet_features = np.load('all_sheet_features.npy', allow_pickle=True).item()

    res = {}
    
    count = 0
    print(len(formulas))
    for formula in formulas:
        # filesheet = formula['filesheet']
        filesheet = formula['filesheet'].split('/')[5]
        # print(filesheet)
        if filesheet in res:
            continue
        count += 1
        print('count', count)
        all_features = []
        id2filesheet = {}
        id_ = 1
        ids = []
        for one_filesheet in all_sheet_features:
            if one_filesheet != filesheet:
                id2filesheet[id_] = one_filesheet
                ids.append(id_)
                id_ += 1
                all_features.append(all_sheet_features[one_filesheet])
            else:
                target_feature = all_sheet_features[one_filesheet]

        all_features = np.array(all_features)
        ids = np.array(ids)

        index = faiss.IndexFlatL2(len(all_features[0]))
        index2 = faiss.IndexIDMap(index)
        index2.add_with_ids(all_features, ids)

        search_list = np.array([target_feature])
    
        D, I = index.search(np.array(search_list), 20) # sanity check
    
        top_k = []
        for index,i in enumerate(I[0]):
            top_k.append((id2filesheet[ids[i]], float(D[0][index])))

        res[filesheet] = top_k
    
    with open('l2_most_simular_sheet_1900.json', 'w') as f:
        json.dump(res,f)

def find_closed_sheet(need_save):
    with open('Formula_160000.json','r') as f:
        formulas = json.load(f)
    
    if need_save:
        all_sheet_features = {}
        feature_files = os.listdir('filename2afterbertfeature')
        feature_files.sort()
        for filename in feature_files:
            all_sheet_features[filename[:-4]] = np.load('filename2afterbertfeature/' + filename, allow_pickle=True)[0]
        np.save('all_sheet_features.npy', all_sheet_features)
    else:
        all_sheet_features = np.load('all_sheet_features.npy', allow_pickle=True).item()

    res = {}
    
    count = 0
    print(len(formulas))
    for formula in formulas:
        # filesheet = formula['filesheet']
        filesheet = formula['filesheet'].split('/')[5]
        # print(filesheet)
        if filesheet in res:
            continue
        count += 1
        print('count', count)
        all_features = []
        id2filesheet = {}
        id_ = 1
        ids = []
        for one_filesheet in all_sheet_features:
            if one_filesheet != filesheet:
                id2filesheet[id_] = one_filesheet
                ids.append(id_)
                id_ += 1
                all_features.append(all_sheet_features[one_filesheet])
            else:
                target_feature = all_sheet_features[one_filesheet]

        all_features = np.array(all_features)
        ids = np.array(ids)

        index = faiss.IndexFlatL2(len(all_features[0]))
        index2 = faiss.IndexIDMap(index)
        index2.add_with_ids(all_features, ids)

        search_list = np.array([target_feature])
    
        D, I = index.search(np.array(search_list), 20) # sanity check
    
        top_k = []
        for index,i in enumerate(I[0]):
            top_k.append((id2filesheet[ids[i]], float(D[0][index])))

        res[filesheet] = top_k
    
    with open('most_simular_sheet_1900.json', 'w') as f:
        json.dump(res,f)

def para_most_similar_sheet(is_save=False):
    files = os.listdir('.')
    files = [item for item in files if 'reduced_formulas_' in item]
    for filename in files:
        company_name = filename.split('_')[2].replace(".json", '')
        if company_name not in ['pge','cisco','ibm','ti']:
            continue
        with open(filename, 'r') as f:
            workbooks = json.load(f)
        with open("fortune500_company2workbook.json", 'r') as f:
            constrained_workbooks = json.load(f)[company_name]
        find_middle10domain_closed_sheet(is_save, 1,1,constrained_workbooks)
        # process = [Process(target=find_middle10domain_closed_sheet, args=(is_save, 1,20, constrained_workbooks)),
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 2,20, constrained_workbooks)), 
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 3,20, constrained_workbooks)),
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 4,20, constrained_workbooks)), 
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 5,20, constrained_workbooks)),
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 6,20, constrained_workbooks)), 
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 7,20, constrained_workbooks)),
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 8,20, constrained_workbooks)), 
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 9,20, constrained_workbooks)),
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 10,20, constrained_workbooks)), 
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 11,20, constrained_workbooks)),
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 12,20, constrained_workbooks)), 
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 13,20, constrained_workbooks)),
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 14,20, constrained_workbooks)), 
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 15,20, constrained_workbooks)),
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 16,20, constrained_workbooks)), 
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 17,20, constrained_workbooks)),
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 18,20, constrained_workbooks)), 
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 19,20, constrained_workbooks)), 
        #         Process(target=find_middle10domain_closed_sheet, args=(is_save, 20,20, constrained_workbooks)), 
        #         ]
        # [p.start() for p in process]  # 开启了两个进程
        # [p.join() for p in process]   # 等待两个进程依次结束
        break
    # res1 = {}
    # for thread_id in range(1,21):
    #     with open(root_path + 'top10domain_most_simular_sheet_'+str(thread_id)+'.json', 'r') as f:
    #         temp1 = json.load(f)
    #     for item in temp1:
    #         res1[item] = temp1[item]

    # with open(root_path + 'top10domain_most_simular_sheet.json', 'w') as f:
    #     json.dump(res1, f)

    

def find_middle10domain_closed_sheet(need_save, thread_id, batch_num, constrain_workbooks, save_path=root_path + 'company_model1_similar_sheet'):
    # with open('Formulas_middle10domain_with_id.json','r') as f:
    with open('Formulas_fortune500_with_id.json','r') as f:
        formulas = json.load(f)
    
    # if need_save:
    all_sheet_features = {}
    feature_files = os.listdir(root_path + 'sheet_after_features')
    feature_files.sort()
    # print('len(sheetfeature2afterfeature', len(feature_files))
    for filename in feature_files:
        wb_name = filename.split('---')[0]
        if wb_name not in constrain_workbooks:
            continue
        # print('filename', filename.replace('.npy',''))
        all_sheet_features[filename.replace('---1---1.npy','')] = np.load(root_path + 'sheet_after_features/' + filename, allow_pickle=True)[0]
        # all_sheet_features[filename.replace('.npy','')] = np.load(root_path + 'sheetfeature2afterfeature/' + filename, allow_pickle=True)[0]
    # np.save('middle10domain_sheet_features.npy', all_sheet_features)
    # np.save('middle10domain_sheet_features.npy', all_sheet_features)
    # else:
        # all_sheet_features = np.load('middle10domain_sheet_features.npy', allow_pickle=True).item()

    res = {}
    
    print('len(all_sheet_features', len(all_sheet_features))
    count = 0
    # print(len(formulas))
    batch_len = len(all_sheet_features) / batch_num
    num = set()
    for index,filesheet in enumerate(all_sheet_features):
        # if index != batch_num:
        #     if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
        #         continue
        # else:
        #     if index <= batch_len * (thread_id - 1 ):
        #         continue
        # print("XXXX")
        # filesheet = formula['filesheet']
        # filesheet = formula['filesheet'].split('/')[-1]
        # if os.path.exists(save_path + '/'+filesheet+'.json'):
        #     continue
        # print("XXXX")
        # if filesheet != "012dcc10c41410112265d9556ec89bb1_d3d3LmFlci5nb3YuYXUJMTUyLjkxLjUzLjE5Mw==.xls.xlsx---Analysis":
        #     continue
        # if filesheet != '23938280411769767155305058958183774721-ltd_trading_authority_2022.xlsx---Summary':
        #     continue
        num.add(filesheet)
        print(filesheet)
        if filesheet in res:
            continue
        count += 1
        print( index, len(all_sheet_features))
        all_features = []
        id2filesheet = {}
        id_ = 0
        ids = []
        for one_filesheet in all_sheet_features:
            # print('one filesheet', one_filesheet)
            # print('filesheet', filesheet)
            if one_filesheet != filesheet:
                id2filesheet[id_] = one_filesheet
                ids.append(id_)
                id_ += 1
                all_features.append(all_sheet_features[one_filesheet])
            else:
                target_feature = all_sheet_features[one_filesheet]

        # try:
        #     target_feature
        # except:
        #     print('not exists')
        #     continue
        
        all_features = np.array(all_features)
        ids = np.array(ids)

        res = {}
        for i_ind, other_feature in enumerate(all_features):
            distance = euclidean(target_feature, other_feature)
            res[id2filesheet[ids[i_ind]]] = distance

        res = sorted(res.items(), key=lambda x: x[1])
        print(res[0:10])
        faiss_index = faiss.IndexFlatL2(len(all_features[0]))
        faiss_index2 = faiss.IndexIDMap(faiss_index)
        faiss_index2.add_with_ids(all_features, ids)

        search_list = np.array([target_feature])
    
        D, I = faiss_index.search(np.array(search_list), 0) # sanity check
    
        top_k = []
        # print('I[0]', I[0])
        for i_index,i in enumerate(I[0]):
            other_feature = all_features[i_index]
            print('id2filesheet[ids[i]]', id2filesheet[ids[i]])
            print(euclidean(target_feature, other_feature))
            print('float(D[0][i_index]))', float(D[0][i_index]))
            # print("id2filesheet[ids[i]]", id2filesheet[ids[i]])
            top_k.append((id2filesheet[ids[i]], float(D[0][i_index])))
        with open(save_path + '/'+filesheet+'.json','w') as f:
            json.dump(top_k, f)
        break
    # with open(root_path +'10domain_most_simular_sheet_'+str(thread_id)+'.json', 'w') as f:
    #     json.dump(res,f)
    print(len(num))
def cos(a,b):
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    return np.matmul(a,b) / (ma*mb)

def find_dis_only_closed_formula(thread_id, batch_num):
    with open('Formula_77772_with_id.json','r') as f:
        formulas = json.load(f)
    with open('most_simular_sheet_1900.json', 'r') as f:
        similar_sheets = json.load(f)

    if os.path.exists("deal00_dis_only_most_similar_formula_1900_"+str(thread_id)+".npy"):
        res = np.load("deal00_dis_only_most_similar_formula_1900_"+str(thread_id)+".npy", allow_pickle=True).item()
    else:
        res = {}

    count = 0
    print(len(formulas))

    batch_len = len(formulas)/batch_num
    # print('thread_id', thread_id)
    # print('batch_len * (thread_id - 1 )', batch_len * (thread_id - 1 ), 'batch_len * thread_id', batch_len * thread_id)
    for index,formula in enumerate(formulas):
        if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
            continue
        formula_token = formula['filesheet'].split('/')[5] + '---' + str(formula['fr']) + '---' +  str(formula['fc'])
        if formula_token in res:
            continue
        # if not os.path.exists('formula2afterbertfeature/' + formula_token + '.npy'):
            # continue
        filesheet = formula['filesheet'].split('/')[5]
        
        # filesheet_vec = np.load('filename2afterbertfeature/'+filesheet+'.npy', allow_pickle=True)[0]
        
        print('count', index)
        need_contine = False
        
        res[formula_token] = {}
        id2filesheet = {}
        id_ = 1
        ids = []
        all_features = []
        for similar_sheet_pair in similar_sheets[filesheet]:
            similar_sheet = similar_sheet_pair[0]
            if similar_sheet == filesheet:
                continue
            # similar_sheet_vec = np.load('filename2afterbertfeature/'+similar_sheet+'.npy', allow_pickle=True)[0]
 
            for other_formula in formulas:
                if other_formula['id'] == formula['id']:
                    if os.path.exists("deal00formula2afterbertfeature/"+formula_token+'.npy'):   
                        # print('deal 00')
                        target_feature = np.load('deal00formula2afterbertfeature/'+formula_token+'.npy', allow_pickle=True)[0]
                    else:
                        # print(formula_token)
                        # print('deal 00')
                        target_feature = np.load('formula2afterbertfeature/' + formula_token + '.npy', allow_pickle=True)[0]
                    # continue
            for other_formula in formulas:
                if other_formula['filesheet'].split('/')[5] == similar_sheet:
                    other_formula_token = similar_sheet + '---' + str(other_formula['fr']) + '---' + str(other_formula['fc'])
                    if not os.path.exists('formula2afterbertfeature/' + other_formula_token + '.npy'):
                        # need_contine = True
                        continue
                    # print('formula2afterbertfeature/' + other_formula_token + '.npy')
                    if os.path.exists("deal00formula2afterbertfeature/"+other_formula_token+'.npy'):
                        other_formula_feature = np.load('deal00formula2afterbertfeature/'+other_formula_token+'.npy', allow_pickle=True)[0]
                    else:
                        # print(other_formula_token)
                        # print('deal 001')
                        other_formula_feature = np.load('formula2afterbertfeature/' + other_formula_token + '.npy', allow_pickle=True)[0]
                    
             
  
                    id2filesheet[id_] = other_formula_token
                    ids.append(id_)
                    id_ += 1
                    all_features.append(other_formula_feature)
            
            all_features = np.array(all_features)
            ids = np.array(ids)

            index = faiss.IndexFlatL2(len(all_features[0]))
            index2 = faiss.IndexIDMap(index)
            index2.add_with_ids(all_features, ids)

            search_list = np.array([target_feature])
        
            D, I = index.search(np.array(search_list), 20) # sanity check
        
            top_k = []
            for index,i in enumerate(I[0]):
                top_k.append((id2filesheet[ids[i]], float(D[0][index])))

            res[formula_token] = top_k
                    # id2filesheet[id_] = other_formula_token
                    # ids.append(id_)
                    # id_ += 1
                    # all_features.append(other_formula_feature)
    
        # if need_contine == True:
            # continue

        # all_features = np.array(all_features)
        # ids = np.array(ids)

        # index = faiss.IndexFlatL2(len(all_features[0]))
        # index2 = faiss.IndexIDMap(index)
        # index2.add_with_ids(all_features, ids)

        # search_list = np.array([target_feature])
    
        # D, I = index.search(np.array(search_list), 20) # sanity check
    
        # top_k = []
        # for index,i in enumerate(I[0]):
        #     top_k.append((id2filesheet[ids[i]], float(D[0][index])))

        # res[formula_token] = top_k
        # print(formula_token)
        # pprint.pprint(res[formula_token])
        # break
    # with open('most_simular_formula_1900.json', 'w') as f:
    #     json.dump(res,f)
        if index % 500 == 0:
            np.save("deal00_dis_only_most_similar_formula_1900_"+str(thread_id)+".npy",res)
    np.save("deal00_dis_only_most_similar_formula_1900_"+str(thread_id)+".npy",res)

def euclidean(x, y):
    return np.sqrt(np.sum((x - y)**2))
    
# def find_only_closed_formula(thread_id, batch_num, similar_sheets='most_simular_sheet_1900.json', save_filepath='model1_formulas_dis',testpath='model1_top10domain_formula2afterfeature_test', filepath='model1_top10domain_formula2afterfeature'):
def find_only_closed_formula(thread_id, batch_num, similar_sheets=root_path + 'company_model1_similar_sheet/', save_filepath=root_path+'company_model1_formulas_dis',testpath=root_path + 'afterfeature_test', filepath=root_path + 'afterfeature'):
    

    #save_filepath='model2_formulas_dis',filepath='model2_formula2afterfeature', ):
    
    # with open('Formula_77772_with_id.json','r') as f:
    #     formulas = json.load(f)
    # with open('Formula_hasfunc_with_id.json','r') as f:
    #     formulas = json.load(f)
    # with open('Formulas_test_top10domain_with_id.json','r') as f:
    #     formulas = json.load(f)
    with open('Formulas_middle10domain_with_id.json','r') as f:
    # with open('Formulas_fortune500_with_id.json','r') as f:
        formulas = json.load(f)

    # with open('most_simular_sheet_1900.json', 'r') as f:
    # with open(similar_sheets, 'r') as f:
        # similar_sheets = json.load(f)
    ne_count = 0
    filesheet2token = {}
    for formula in formulas:
        if formula['filesheet'].split('/')[-1] not in filesheet2token:
            filesheet2token[formula['filesheet'].split('/')[-1]] = []
        # print("xxxx", formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' +  str(formula['fc']))
        filesheet2token[formula['filesheet'].split('/')[-1]].append(formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' +  str(formula['fc']))
    # if os.path.exists(save_filepath+"_"+str(thread_id)+".npy"):
    #     try:
    #         res = np.load(save_filepath+"_"+str(thread_id)+".npy", allow_pickle=True).item()
    #     except:
    #         res = {}
    # else:
    #     res = {}
    # res = {}
    count = 0
    print(len(formulas))
    resset = set()

    batch_len = len(formulas)/batch_num
    # print('thread_id', thread_id)
    # print('batch_len * (thread_id - 1 )', batch_len * (thread_id - 1 ), 'batch_len * thread_id', batch_len * thread_id)
    for index,formula in enumerate(formulas):
        if index != batch_num:
            if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1 ):
                continue
        formula_token = formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' +  str(formula['fc'])
        # formula_token = formula_token.replace(".xls.xlsx", '.xlsx')
        
        if not os.path.exists(testpath+'/'+formula_token+'.npy'):
            # print('formula_token', formula_token)
            ne_count += 1
            continue
        if os.path.exists(save_filepath+'/'+formula_token+'.npy'):
            continue
        
        # if os.path.exists(save_filepath +'/' + formula_token + '.npy'):
            # continue
        # if formula_token != '012dcc10c41410112265d9556ec89bb1_d3d3LmFlci5nb3YuYXUJMTUyLjkxLjUzLjE5Mw==.xls.xlsx---Analysis---10---7':
            # continue
        print('count', index)
        # if index == 9:
        #     continue
        # if not os.path.exists('formula2afterbertfeature/' + formula_token + '.npy'):
            # continue
        filesheet = formula['filesheet'].split('/')[-1]
        
        # filesheet_vec = np.load('filename2afterbertfeature/'+filesheet+'.npy', allow_pickle=True)[0]
        # print('time0')
        
        # need_contine = False
        # is_exist = False
        # res[formula_token] = {}
        # try:
        res = {}
        
        # if filesheet not in similar_sheets:
            # continue
        # if not os.path.exists(similar_sheets + filesheet + '.json'):
        if not os.path.exists(similar_sheets + filesheet + '.json'):
            continue
        # with open(similar_sheets + filesheet + '.json', 'r') as f:
        with open(similar_sheets + filesheet + '.json', 'r') as f:
            similar_sheet_feature = json.load(f)
        target_feature = np.load(filepath+'/'+formula_token+'.npy', allow_pickle=True)
        if len(target_feature) == 1:
            target_feature = target_feature[0]

        # print('similar_sheet_feature', similar_sheet_feature)
        # print('similar_sheets[filesheet]', similar_sheets[filesheet])
        for similar_sheet_pair in similar_sheet_feature:
            # print('time1')
            # print('similar_sheet_pair', similar_sheet_pair)
            similar_sheet = similar_sheet_pair[0].replace('---1---1','')
            # print('similar_sheet',similar_sheet)
            if similar_sheet == filesheet:
                continue
            # print("similar_sheet#############", similar_sheet)
            # similar_sheet_vec = np.load('filename2afterbertfeature/'+similar_sheet+'.npy', allow_pickle=True)[0]
            

            
            # if not is_exist:
            #     break
            # print("##################")
            if similar_sheet not in filesheet2token:
                continue
            for other_formula_token in filesheet2token[similar_sheet]:
                # if 'OP2-Project Cost' in other_formula['filesheet'].split('/')[-1]:
                #     print('similar_sheet', similar_sheet)
                #     print("other_formula['filesheet'].split('/')[-1]", other_formula['filesheet'].split('/')[-1])
                    # resset.add(other_formula['filesheet'].split('/')[-1])
                # if other_formula['filesheet'].split('/')[-1] == similar_sheet:
                    # print("?????")
                    # print("other_formula['filesheet'].split('/')[-1]", other_formula['filesheet'].split('/')[-1])
                    # other_formula_token = similar_sheet + '---' + str(other_formula['fr']) + '---' + str(other_formula['fc'])
                    # if not os.path.exists(filepath+'/'+other_formula_token+'.npy'):
                    #     # print('other formula not in')
                    #     continue
                    # print('other formula in')
                    # print('other_formula_token', other_formula_token)
                    # print('time2')
                    other_formula_token = other_formula_token.replace(".xls.xlsx", '.xlsx')
                    # print('other_formula_token', other_formula_token)
                    try:
                        # print("filepath + '/'+other_formula_token+'.npy.npy'", filepath + '/'+other_formula_token+'.npy.npy')
                        # print(filepath + '/'+other_formula_token+'.npy')
                        other_formula_feature = np.load(filepath + '/'+other_formula_token+'.npy', allow_pickle=True)
                    except:
                        # print("not exists", other_formula_token)
                        continue
                        # print("not exists", other_formula_token)
                 
                    print("Xxxxxxxxx")
                    if len(other_formula_feature) == 1:
                        other_formula_feature = other_formula_feature[0]
                    # print('target_feature', target_feature)
                    # print('other_formula_feature', other_formula_feature)
                    # formula_cos = cos(target_feature, other_formula_feature)
                    formula_dis = euclidean(target_feature, other_formula_feature)
                    # print('other_formula_token....', other_formula_token)
                    res[other_formula_token] = formula_dis
                    
                    # print('simialr sheet', similar_sheet)
                    # print('other formula token', other_formula_token)
                # print(res)
        # print('res', res)
        np.save(save_filepath +'/' + formula_token + '.npy', res)
    #     print('len(res', len(res))
    #     # except:
    #     #     continue
    #     if index % 500 == 0:
    #         np.save(save_filepath + "_"+str(thread_id)+".npy",res)
    # print(len(res))
    # np.save(save_filepath + "_"+str(thread_id)+".npy",res)
    # print("----")
    # print('similar_sheets[filesheet]', similar_sheets['019ac86c566c52abade9a8ff47968c92_ZWRnZS5yaXQuZWR1CTEyOS4yMS4xOTguMTYz.xls.xlsx---House of Quality 1'])
    # print("\n\n\n")
    # # print('similar_sheet', similar_sheet)
    # print('resset', resset)
    # print("----")
    print('ne_count', ne_count)

def save_formula_dis(save_filepath=root_path+'top10domain_model1_formulas_dis'):
    for thread_id in range(1,11):
        print('thread_id', thread_id)
        if not os.path.exists(save_filepath + "_"+str(thread_id)+".npy"):
            continue
        res = np.load(save_filepath + "_"+str(thread_id)+".npy", allow_pickle=True).item()
        for formula_token in res:
            if os.path.exists(root_path + 'top10domain_model1_formulas_dis' + formula_token + '.npy'):
                continue
            np.save(root_path + 'top10domain_model1_formulas_dis/' + formula_token + '.npy', res[formula_token])
    # filelist = os.listdir(root_path)
    # for filename in filelist:
    #     if 'top10domain_model1_formulas_dis' in filename and '.npy' in filename:
    #         shutil.move(root_path + filename, root_path + 'top10domain_model1_formulas_dis/' + filename.replace('top10domain_model1_formulas_dis',''))
def naive_find_formula(thread_id, batch_num, save_path=root_path + 'dedup_model1_naive_res', testpath=root_path + 'afterfeature_test' ):
    # with open('Formula_77772_with_id.json','r') as f:
    #     formulas = json.load(f)
    # with open('Formula_hasfunc_with_id.json','r') as f:
    #     formulas = json.load(f)
    # with open('most_simular_sheet_1900.json', 'r') as f:
    #     similar_sheets = json.load(f)
    # with open('Formulas_middle10domain_with_id.json','r') as f:
    # with open('Formulas_top10domain_with_id.json','r') as f:
        # formulas = json.load(f)
    with open(root_path + 'dedup_workbooks.json','r') as f:
        dedup_workbooks = json.load(f)
    with open('Formulas_fortune500_with_id.json','r') as f:
        formulas = json.load(f)
    # with open('fortune500_formulatoken2r1c1.json','r') as f:
    # with open('top10domain_formulatoken2r1c1.json','r') as f:
        # formulatoken2r1c1 = json.load(f)
    # with open(root_path + 'top10domain_most_simular_sheet.json', 'r') as f:
    #     similar_sheets = json.load(f)
    found_true_res = []
    not_found_res = []
    found_false_res = []
    formulatoken2r1c1 = {}
    res = {}
    batch_len = len(formulas)/batch_num
    ne_count = 0
    ne_count1 = 0

    # file2sheet2rc2r1c1 = {}
    sheet2file2rc2r1c1 = {}
    for index,formula in enumerate(formulas):
        filename = formula['filesheet'].split('---')[0]
        sheetname = formula['filesheet'].split('---')[1]
        # print(formula['filesheet'])
        rc = str(formula['fr']) + '---' + str(formula['fc'])
        formula_token = filename + '---' + sheetname + '---' + rc
        formulatoken2r1c1[formula_token] = formula['r1c1']
        # if formula_token not in formulatoken2r1c1:
        #     continue
        # if sheetname not in sheet2file2rc2r1c1:
        #     sheet2file2rc2r1c1[sheetname] = {}
        # if filename not in sheet2file2rc2r1c1:
        #     sheet2file2rc2r1c1[sheetname][filename] = {}
        # sheet2file2rc2r1c1[sheetname][filename][rc] = {
        #     'r1c1': formulatoken2r1c1[formula_token],
        #     'id': formula['id']
        #     }

    for index,formula in enumerate(formulas):
        if thread_id != batch_num:
            if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1 ):
                continue
        
        # print("formula['filesheet']", formula['filesheet'])
        filesheet = formula['filesheet'].split('/')[-1]
        found = False
        fname = filesheet.split('---')[0]
        if fname not in dedup_workbooks:
            continue
        formula_token = filesheet + '---' + str(formula['fr']) + '---' + str(formula['fc'])
        formula_rc = str(formula['fr']) + '---' + str(formula['fc'])
        # print(save_path+'/' + filesheet+ '---' + str(formula['fr']) + '---' + str(formula['fc'])+'.json')
        # if os.path.exists(save_path+'/' + filesheet+ '---' + str(formula['fr']) + '---' + str(formula['fc'])+'.json'):
        #     continue
        if not os.path.exists(root_path + 'dedup_model1_res/' + formula_token + '.json'):
            continue
        # if filesheet not in similar_sheets:
        #     continue
        ne_count += 1
        # print('xxxxxxx')
        # print('filesheet', filesheet)
        if not os.path.exists(root_path + 'model1_similar_sheet/' + filesheet + '.json'):
            continue
        # print("sssssss")
        with open(root_path + 'model1_similar_sheet/' + filesheet + '.json', 'r') as f:
            similar_sheet_feature = json.load(f)
        # print('formula_token', formula_token)
        if not os.path.exists(testpath+'/'+formula_token+'.npy'):
            ne_count1 += 1
            continue
        print(index, len(formulas))
        for similar_sheet_pair in similar_sheet_feature:
            similar_sheet = similar_sheet_pair[0].replace('---1---1','')
            print('   similar shet')
            # print('similar_sheet', similar_sheet)
            # print('filesheet', filesheet)
            if similar_sheet == filesheet:
                continue
            # print('sss')
            # for other_formula in formulas:
            other_fname = similar_sheet.split('---')[0]
            # print("similar_sheet + '---' + formula_rc", similar_sheet + '---' + formula_rc)
            if other_fname not in dedup_workbooks:
                continue
            # print("similar_sheet + '---' + formula_rc", similar_sheet + '---' + formula_rc)
            # print("formula_rc", formula_token)
            if similar_sheet + '---' + formula_rc not in formulatoken2r1c1:
                print('    not in formulatoken2r1c1')
                continue
            found = True
            # print('xxxxxx')
            if formula['r1c1'] == formulatoken2r1c1[similar_sheet + '---' + formula_rc]:
                # print('dddd')
                found_true_res.append([similar_sheet + '---' + formula_rc, formulatoken2r1c1[similar_sheet + '---' + formula_rc]])
                # res[filesheet+ '---' + str(formula['fr']) + '---' + str(formula['fc'])] = similar_sheet + '---' + str(other_formula['fr']) + '---' + str(other_formula['fc'])
                with open(save_path+'/' + filesheet+ '---' + formula_rc + '.json','w') as f:
                    json.dump([similar_sheet + '---' + formula_rc, True], f)
            else:
                found_false_res.append(formula['id'])
                with open(save_path+'/' + filesheet+ '---' + formula_rc+ '.json','w') as f:
                    json.dump([similar_sheet + '---' + formula_rc, False], f)
            if found:
                break
        if not found:
            with open(save_path+'/' + filesheet+ '---' + str(formula['fr']) + '---' + str(formula['fc']) + '.json','w') as f:
                json.dump(['not found'], f)
        print('ne_count1', ne_count1)
        print('ne_count', ne_count)
            # not_found_res.append(formula['id'])
    # with open("naive_hasfunc_found_true_res_"+str(thread_id)+".json", 'w') as f:
    #     json.dump(found_true_res, f)
    # with open("naive_hasfunc_found_false_res_"+str(thread_id)+".json", 'w') as f:
    #     json.dump(found_false_res, f)
    # with open("naive_hasfunc_not_found_res_"+str(thread_id)+".json", 'w') as f:
        # json.dump(not_found_res, f)
    # with open("naive_res_"+str(thread_id)+".json",'w') as f:
    #     json.dump(res, f)

def para_naive():
    process = [Process(target=naive_find_formula, args=(1,20)),
               Process(target=naive_find_formula, args=(2,20)), 
               Process(target=naive_find_formula, args=(3,20)),
               Process(target=naive_find_formula, args=(4,20)), 
               Process(target=naive_find_formula, args=(5,20)),
               Process(target=naive_find_formula, args=(6,20)), 
               Process(target=naive_find_formula, args=(7,20)),
               Process(target=naive_find_formula, args=(8,20)), 
               Process(target=naive_find_formula, args=(9,20)),
               Process(target=naive_find_formula, args=(10,20)), 
               Process(target=naive_find_formula, args=(11,20)),
               Process(target=naive_find_formula, args=(12,20)), 
               Process(target=naive_find_formula, args=(13,20)),
               Process(target=naive_find_formula, args=(14,20)), 
               Process(target=naive_find_formula, args=(15,20)),
               Process(target=naive_find_formula, args=(16,20)), 
               Process(target=naive_find_formula, args=(17,20)),
               Process(target=naive_find_formula, args=(18,20)), 
               Process(target=naive_find_formula, args=(19,20)), 
               Process(target=naive_find_formula, args=(20,20)), 
            ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束

    res1 = []
    res2 = []
    res3 = []
    res4 = {}

def clean_multi_same(thread_id, batch_num):
    with open('Formula_77772_with_id.json','r') as f:
        formulas = json.load(f)
    with open('most_simular_sheet_1900.json', 'r') as f:
        similar_sheets = json.load(f)

    result = []
    batch_len = len(formulas)/batch_num
    for index,formula in enumerate(formulas):
        if thread_id != batch_num:
            if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1 ):
                continue
        print(index, len(formulas))
        filesheet = formula['filesheet'].split('/')[5]
        need_remove = False

        origin_features = []

        for similar_sheet_pair in similar_sheets[filesheet]:
            similar_sheet = similar_sheet_pair[0]
            if similar_sheet == filesheet:
                continue
            for other_formula in formulas:
                if other_formula['filesheet'].split('/')[5] == similar_sheet and other_formula['r1c1'] == formula['r1c1']:
                    other_formula_token = other_formula['filesheet'].split('/')[5] + '---' + str(other_formula['fr']) + '---' +  str(other_formula['fc'])
                    feature = np.load('model2_formula2beforebertfeature/'+other_formula_token + '.npy', allow_pickle=True)
                    for other_feature in origin_features:
                        if (other_feature == feature).all():
                            need_remove = True
                            break
                    if need_remove:
                        break
                    origin_features.append(feature)
            if need_remove:
                break
        if not need_remove:
            result.append(formula)
    print(len(result))
    with open("Formula_clean_with_id_"+str(thread_id)+".json",'w') as f:
        json.dump(result,f)

def para_multi_clean():
    # process = [Process(target=clean_multi_same, args=(1,20)),
    #            Process(target=clean_multi_same, args=(2,20)), 
    #            Process(target=clean_multi_same, args=(3,20)),
    #            Process(target=clean_multi_same, args=(4,20)), 
    #            Process(target=clean_multi_same, args=(5,20)),
    #            Process(target=clean_multi_same, args=(6,20)), 
    #            Process(target=clean_multi_same, args=(7,20)),
    #            Process(target=clean_multi_same, args=(8,20)), 
    #            Process(target=clean_multi_same, args=(9,20)),
    #            Process(target=clean_multi_same, args=(10,20)), 
    #            Process(target=clean_multi_same, args=(11,20)),
    #            Process(target=clean_multi_same, args=(12,20)), 
    #            Process(target=clean_multi_same, args=(13,20)),
    #            Process(target=clean_multi_same, args=(14,20)), 
    #            Process(target=clean_multi_same, args=(15,20)),
    #            Process(target=clean_multi_same, args=(16,20)), 
    #            Process(target=clean_multi_same, args=(17,20)),
    #            Process(target=clean_multi_same, args=(18,20)), 
    #            Process(target=clean_multi_same, args=(19,20)), 
    #            Process(target=clean_multi_same, args=(20,20)), 
    #         ]
    # [p.start() for p in process]  # 开启了两个进程
    # [p.join() for p in process]   # 等待两个进程依次结束

    res = []
    for thread_id in range(1,21):
        with open('Formula_clean_with_id_'+str(thread_id)+'.json', 'r') as f:
            temp_formulas = json.load(f)
        for item in temp_formulas:
            res.append(item)

    print('all', len(res))
    with open("Formula_clean_with_id.json", 'w') as f:
        json.dump(res, f)

def find_only_closed_formula_finetune(thread_id, batch_num):
    with open('Formula_77772_with_id.json','r') as f:
        formulas = json.load(f)
    with open('most_simular_sheet_1900.json', 'r') as f:
        similar_sheets = json.load(f)


    # if os.path.exists("finetune_only_most_similar_formula_1900_"+str(thread_id)+".npy"):
    if os.path.exists("triplet_finetune_only_most_similar_formula_1900_"+str(thread_id)+".npy"):
        try:
            # res = np.load("finetune_only_most_similar_formula_1900_"+str(thread_id)+".npy", allow_pickle=True).item()
            res = np.load("triplet_finetune_only_most_similar_formula_1900_"+str(thread_id)+".npy", allow_pickle=True).item()
        except:
            res = {}
    else:
        res = {}

    count = 0
    print(len(formulas))

    batch_len = len(formulas)/batch_num
    # print('thread_id', thread_id)
    # print('batch_len * (thread_id - 1 )', batch_len * (thread_id - 1 ), 'batch_len * thread_id', batch_len * thread_id)
    for index,formula in enumerate(formulas):
        if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
            continue
        formula_token = formula['filesheet'].split('/')[5] + '---' + str(formula['fr']) + '---' +  str(formula['fc'])
        if formula_token in res:
            continue
        # if not os.path.exists('formula2afterbertfeature/' + formula_token + '.npy'):
            # continue
        filesheet = formula['filesheet'].split('/')[5]
        
        # filesheet_vec = np.load('filename2afterbertfeature/'+filesheet+'.npy', allow_pickle=True)[0]
        
        print('count', index)
        need_contine = False
        
        res[formula_token] = {}
        for similar_sheet_pair in similar_sheets[filesheet]:
            similar_sheet = similar_sheet_pair[0]
            if similar_sheet == filesheet:
                continue
            # similar_sheet_vec = np.load('filename2afterbertfeature/'+similar_sheet+'.npy', allow_pickle=True)[0]
 
            for other_formula in formulas:
                if other_formula['id'] == formula['id']:
                    target_feature = np.load('finetune_formuals/'+formula_token+'.npy', allow_pickle=True)[0]
                   
            for other_formula in formulas:
                if other_formula['filesheet'].split('/')[5] == similar_sheet:
                    other_formula_token = similar_sheet + '---' + str(other_formula['fr']) + '---' + str(other_formula['fc'])
                    if not os.path.exists('finetune_formuals/' + other_formula_token + '.npy'):
                        # need_contine = True
                        continue

                    other_formula_feature = np.load('finetune_formuals/'+other_formula_token+'.npy', allow_pickle=True)[0]

                    formula_cos = cos(target_feature, other_formula_feature)
                    res[formula_token][other_formula_token] = formula_cos

        if index % 500 == 0:
            np.save("triplet_finetune_only_most_similar_formula_1900_"+str(thread_id)+".npy",res)
    #         np.save("finetune_only_most_similar_formula_1900_"+str(thread_id)+".npy",res)
    np.save("triplet_finetune_only_most_similar_formula_1900_"+str(thread_id)+".npy",res)
    # np.save("finetune_only_most_similar_formula_1900_"+str(thread_id)+".npy",res)

def no_l2_simsheet_l2_simformu():
    with open('Formula_77772_with_id.json','r') as f:
        formulas = json.load(f)
    with open('most_simular_sheet_1900.json', 'r') as f:
        similar_sheets = json.load(f)

    # if os.path.exists("deal00_most_similar_formula_1900_"+str(thread_id)+".npy"):
    #     res = np.load("deal00_most_similar_formula_1900_"+str(thread_id)+".npy", allow_pickle=True).item()
    # else:
    res = {}

    count = 0
    print(len(formulas))
    batch_len = len(formulas)/batch_num
    for index,formula in enumerate(formulas):
        if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
            continue
        formula_token = formula['filesheet'].split('/')[5] + '---' + str(formula['fr']) + '---' +  str(formula['fc'])
        if formula_token in res:
            continue
        # if not os.path.exists('formula2afterbertfeature/' + formula_token + '.npy'):
            # continue
        filesheet = formula['filesheet'].split('/')[5]
        
        filesheet_vec = np.load('filename2afterbertfeature/'+filesheet+'.npy', allow_pickle=True)[0]
        
        print('count', index)
        need_contine = False
        
        res[formula_token] = {}
        for similar_sheet_pair in similar_sheets[filesheet]:
            similar_sheet = similar_sheet_pair[0]
            if similar_sheet == filesheet:
                continue
            similar_sheet_vec = np.load('filename2afterbertfeature/'+similar_sheet+'.npy', allow_pickle=True)[0]
            # print(filesheet_vec)
            # print(similar_sheet_vec)
            sheet_cos = cos(filesheet_vec, similar_sheet_vec)
            for other_formula in formulas:
                if other_formula['id'] == formula['id']:
                    if os.path.exists("deal00formula2afterbertfeature/"+formula_token+'.npy'):   
                        # print('deal 00')
                        target_feature = np.load('deal00formula2afterbertfeature/'+formula_token+'.npy', allow_pickle=True)[0]
                    else:
                        # print(formula_token)
                        # print('deal 00')
                        target_feature = np.load('formula2afterbertfeature/' + formula_token + '.npy', allow_pickle=True)[0]
                    # continue
            for other_formula in formulas:
                if other_formula['filesheet'].split('/')[5] == similar_sheet:
                    other_formula_token = similar_sheet + '---' + str(other_formula['fr']) + '---' + str(other_formula['fc'])
                    if not os.path.exists('formula2afterbertfeature/' + other_formula_token + '.npy'):
                        # need_contine = True
                        continue
                    # print('formula2afterbertfeature/' + other_formula_token + '.npy')
                    if os.path.exists("deal00formula2afterbertfeature/"+other_formula_token+'.npy'):
                        other_formula_feature = np.load('deal00formula2afterbertfeature/'+other_formula_token+'.npy', allow_pickle=True)[0]
                    else:
                        # print(other_formula_token)
                        # print('deal 001')
                        other_formula_feature = np.load('formula2afterbertfeature/' + other_formula_token + '.npy', allow_pickle=True)[0]
                    formula_cos = cos(target_feature, other_formula_feature)
                    res[formula_token][other_formula_token] = sheet_cos*formula_cos

        if index % 500 == 0:
            np.save("deal00_most_similar_formula_1900_"+str(thread_id)+".npy",res)
    np.save("deal00_most_similar_formula_1900_"+str(thread_id)+".npy",res)

    
def find_closed_formula(thread_id, batch_num):
    with open('Formula_77772_with_id.json','r') as f:
        formulas = json.load(f)
    with open('most_simular_sheet_1900.json', 'r') as f:
        similar_sheets = json.load(f)

    # if os.path.exists("deal00_most_similar_formula_1900_"+str(thread_id)+".npy"):
    #     res = np.load("deal00_most_similar_formula_1900_"+str(thread_id)+".npy", allow_pickle=True).item()
    # else:
    res = {}

    count = 0
    print(len(formulas))
    batch_len = len(formulas)/batch_num
    for index,formula in enumerate(formulas):
        if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
            continue
        formula_token = formula['filesheet'].split('/')[5] + '---' + str(formula['fr']) + '---' +  str(formula['fc'])
        if formula_token in res:
            continue
        # if not os.path.exists('formula2afterbertfeature/' + formula_token + '.npy'):
            # continue
        filesheet = formula['filesheet'].split('/')[5]
        
        filesheet_vec = np.load('filename2afterbertfeature/'+filesheet+'.npy', allow_pickle=True)[0]
        
        print('count', index)
        need_contine = False
        
        res[formula_token] = {}
        for similar_sheet_pair in similar_sheets[filesheet]:
            similar_sheet = similar_sheet_pair[0]
            if similar_sheet == filesheet:
                continue
            similar_sheet_vec = np.load('filename2afterbertfeature/'+similar_sheet+'.npy', allow_pickle=True)[0]
            # print(filesheet_vec)
            # print(similar_sheet_vec)
            sheet_cos = cos(filesheet_vec, similar_sheet_vec)
            for other_formula in formulas:
                if other_formula['id'] == formula['id']:
                    if os.path.exists("deal00formula2afterbertfeature/"+formula_token+'.npy'):   
                        # print('deal 00')
                        target_feature = np.load('deal00formula2afterbertfeature/'+formula_token+'.npy', allow_pickle=True)[0]
                    else:
                        # print(formula_token)
                        # print('deal 00')
                        target_feature = np.load('formula2afterbertfeature/' + formula_token + '.npy', allow_pickle=True)[0]
                    # continue
            for other_formula in formulas:
                if other_formula['filesheet'].split('/')[5] == similar_sheet:
                    other_formula_token = similar_sheet + '---' + str(other_formula['fr']) + '---' + str(other_formula['fc'])
                    if not os.path.exists('formula2afterbertfeature/' + other_formula_token + '.npy'):
                        # need_contine = True
                        continue
                    # print('formula2afterbertfeature/' + other_formula_token + '.npy')
                    if os.path.exists("deal00formula2afterbertfeature/"+other_formula_token+'.npy'):
                        other_formula_feature = np.load('deal00formula2afterbertfeature/'+other_formula_token+'.npy', allow_pickle=True)[0]
                    else:
                        # print(other_formula_token)
                        # print('deal 001')
                        other_formula_feature = np.load('formula2afterbertfeature/' + other_formula_token + '.npy', allow_pickle=True)[0]
                    formula_cos = cos(target_feature, other_formula_feature)
                    res[formula_token][other_formula_token] = sheet_cos*formula_cos
                    # id2filesheet[id_] = other_formula_token
                    # ids.append(id_)
                    # id_ += 1
                    # all_features.append(other_formula_feature)
    
        # if need_contine == True:
            # continue

        # all_features = np.array(all_features)
        # ids = np.array(ids)

        # index = faiss.IndexFlatL2(len(all_features[0]))
        # index2 = faiss.IndexIDMap(index)
        # index2.add_with_ids(all_features, ids)

        # search_list = np.array([target_feature])
    
        # D, I = index.search(np.array(search_list), 20) # sanity check
    
        # top_k = []
        # for index,i in enumerate(I[0]):
        #     top_k.append((id2filesheet[ids[i]], float(D[0][index])))

        # res[formula_token] = top_k
        # print(formula_token)
        # pprint.pprint(res[formula_token])
        # break
    # with open('most_simular_formula_1900.json', 'w') as f:
    #     json.dump(res,f)
        if index % 500 == 0:
            np.save("deal00_most_similar_formula_1900_"+str(thread_id)+".npy",res)
    np.save("deal00_most_similar_formula_1900_"+str(thread_id)+".npy",res)

def save_not_found_true_false():
    with open('Formula_160000_with_id.json','r') as f:
        formulas = json.load(f)
    with open("naive_not_found_1900.json", 'r') as f:
        not_found = json.load(f)
    with open("naive_found_false_1900.json", 'r') as f:
        found_false = json.load(f)
    with open("naive_found_true_1900.json", 'r') as f:
        found_true = json.load(f)
    not_found_res = []
    found_false_res = []
    found_true_res = []


    for formula in formulas:
        print(formula['id'])
        if formula['id'] in not_found:
            not_found_res.append(formula)
        elif formula['id'] in found_false:
            found_false_res.append(formula)
        elif formula['id'] in found_true:
            found_true_res.append(formula)
    
    with open("naive_found_true_1900_formulas.json", 'w') as f:
        json.dump(found_true_res, f)
    with open("naive_found_false_1900_formulas.json", 'w') as f:
        json.dump(found_false_res, f)
    with open("naive_not_found_1900_formulas.json", 'w') as f:
        json.dump(not_found_res, f)
            

def print_one_most_similar(filename):
    with open('most_simular_sheet_1900.json','r') as f:
        sheet_distance = json.load(f)
    pprint.pprint(sheet_distance[filename])

def print_one_filesheet_r1c1(filename, r1c1):
    with open('Formula_160000_with_id.json','r') as f:
        formulas = json.load(f)
    for formula in formulas:
        if formula['filesheet'] == filename and formula['r1c1'] == r1c1:
            print("#######")
            pprint.pprint(formula)

def look_same_sheet_name():
    with open("naive_found_true_1900.json", 'r') as f:
        found_true_ids = json.load(f)
    with open('Formula_160000_with_id.json','r') as f:
        formulas = json.load(f)
    with open('most_simular_sheet_1900.json','r') as f:
        sheet_distance = json.load(f)

    same_name =0
    for id_ in found_true_ids:
        for formula in formulas:
            if formula['id'] == id_:
                # print("####")
                # print(formula['filesheet'].split('/')[5] )
                # print([i[0] for i in sheet_distance[formula['filesheet'].split('/')[5]]])
                if formula['filesheet'].split('---')[1] in [i[0].split('---')[1] for i in sheet_distance[formula['filesheet'].split('/')[5]]]:
                    same_name += 1
                    print(same_name)
                # break
            
    print(same_name)

def reranking():
    with open('Formula_77772_with_id.json','r') as f:
        formulas = json.load(f)
    with open('most_simular_sheet_1900.json', 'r') as f:
        similar_sheets = json.load(f)
    with open('most_simular_formula_1900.json', 'r') as f:
        similar_formulas = json.load(f)

def para_run():
    
    process = [Process(target=find_closed_formula, args=(1,20)),
               Process(target=find_closed_formula, args=(2,20)), 
               Process(target=find_closed_formula, args=(3,20)),
               Process(target=find_closed_formula, args=(4,20)), 
               Process(target=find_closed_formula, args=(5,20)),
               Process(target=find_closed_formula, args=(6,20)), 
               Process(target=find_closed_formula, args=(7,20)),
               Process(target=find_closed_formula, args=(8,20)), 
               Process(target=find_closed_formula, args=(9,20)),
               Process(target=find_closed_formula, args=(10,20)), 
               Process(target=find_closed_formula, args=(11,20)),
               Process(target=find_closed_formula, args=(12,20)), 
               Process(target=find_closed_formula, args=(13,20)),
               Process(target=find_closed_formula, args=(14,20)), 
               Process(target=find_closed_formula, args=(15,20)),
               Process(target=find_closed_formula, args=(16,20)), 
               Process(target=find_closed_formula, args=(17,20)),
               Process(target=find_closed_formula, args=(18,20)), 
               Process(target=find_closed_formula, args=(19,20)), 
               Process(target=find_closed_formula, args=(20,20)), 
            ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束

def para_run_template_eval():
    
    process = [
            Process(target=sort_only_template_most_similar_formula, args=(1,1)),
            Process(target=sort_only_template_most_similar_formula, args=(1,2)), 
            Process(target=sort_only_template_most_similar_formula, args=(1,3)),
            Process(target=sort_only_template_most_similar_formula, args=(1,4)), 
            Process(target=sort_only_template_most_similar_formula, args=(1,5)),
            Process(target=sort_only_template_most_similar_formula, args=(1,6)), 
            Process(target=sort_only_template_most_similar_formula, args=(1,7)),
            Process(target=sort_only_template_most_similar_formula, args=(1,8)), 
            Process(target=sort_only_template_most_similar_formula, args=(1,9)),
            Process(target=sort_only_template_most_similar_formula, args=(1,10)), 
            Process(target=sort_only_template_most_similar_formula, args=(1,11)),
            Process(target=sort_only_template_most_similar_formula, args=(1,12)), 
            Process(target=sort_only_template_most_similar_formula, args=(1,13)),
            Process(target=sort_only_template_most_similar_formula, args=(1,14)), 
            Process(target=sort_only_template_most_similar_formula, args=(1,15)),
            Process(target=sort_only_template_most_similar_formula, args=(1,16)), 
            Process(target=sort_only_template_most_similar_formula, args=(1,17)),
            Process(target=sort_only_template_most_similar_formula, args=(1,18)), 
            Process(target=sort_only_template_most_similar_formula, args=(1,19)), 
            Process(target=sort_only_template_most_similar_formula, args=(1,20)), 
            ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束
    res1 = []
    res2 = []
    for thread_id in range(1,11):
        print(thread_id)
        with open("found_res_"+str(thread_id)+'.json','r') as f:
            temp1 = json.load(f)
        with open("not_found_res_"+str(thread_id)+'.json','r') as f:
            temp2 = json.load(f)
        print('found', len(temp1))
        print('not found', len(temp2))
        for item in temp1:
            res1.append(item)
        for item in temp2:
            res2.append(item)

    print('found', len(res1))
    print('not found', len(res2))
    with open("model1_hasfunc_formulas_dis_sketch_found_res.json",'w') as f:
        json.dump(res1, f) 
    with open("model1_hasfunc__formulas_dis_sketch_not_found_res.json",'w') as f:
        json.dump(res2, f) 

# def generate_formula_features():
#     files = os.listdir('.')
#     files = [item for item in files if 'reduced_formulas_' in item]
#     for filename in files:
#         company_name = filename.split('_')[2].replace(".json", '')
#         if company_name not in ['cisco','ibm','ti','pge']:
#             continue
#         with open(filename, 'r') as f:
#             formulas = json.load(f)
#         with open("fortune500_company2workbook.json", 'r') as f:
#             constrain_workbooks = json.load(f)[company_name]
#         print('workbooks', company_name, len(workbooks))
#         for formula_token in formulas:
#             generate_demo_features(formula_token.split('---')[0], formula_token.split('---')[1], origin_wbjson, row, col, tile_path, is_look=True, cross=cross)


# def para_only_run_eval(filepath = root_path + 'model1_formulas_dis', load_path=root_path+'model1_formulas_dis', save_path = root_path + 'dedup_model1_res'):
def para_only_run_eval(filepath = root_path + 'model1_formulas_dis', load_path=root_path+'model1_formulas_dis', save_path = root_path + 'company_model1_res'):
    # sort_only_most_similar_formula(1,1,1,filepath,load_path)
    files = os.listdir('.')
    files = [item for item in files if 'reduced_formulas_' in item]
    for filename in files:
        company_name = filename.split('_')[2].replace(".json", '')
        if company_name not in ['cisco','ibm','ti','pge']:
            continue
        with open(filename, 'r') as f:
            workbooks = json.load(f)
        with open("fortune500_company2workbook.json", 'r') as f:
            constrain_workbooks = json.load(f)[company_name]
        print('workbooks', company_name, len(workbooks))
        sort_only_most_similar_formula(1,1,1,filepath,load_path,save_path, constrain_workbooks, workbooks)
        # process = [
        #     Process(target=sort_only_most_similar_formula, args=(1,1,10,filepath,load_path,save_path, constrain_workbooks, workbooks)),
        #     Process(target=sort_only_most_similar_formula, args=(1,2,10,filepath,load_path,save_path, constrain_workbooks, workbooks)), 
        #     Process(target=sort_only_most_similar_formula, args=(1,3,10,filepath,load_path,save_path, constrain_workbooks, workbooks)),
        #     Process(target=sort_only_most_similar_formula, args=(1,4,10,filepath,load_path,save_path, constrain_workbooks, workbooks)), 
        #     Process(target=sort_only_most_similar_formula, args=(1,5,10,filepath,load_path,save_path, constrain_workbooks, workbooks)),
        #     Process(target=sort_only_most_similar_formula, args=(1,6,10,filepath,load_path,save_path, constrain_workbooks, workbooks)), 
        #     Process(target=sort_only_most_similar_formula, args=(1,7,10,filepath,load_path,save_path, constrain_workbooks, workbooks)),
        #     Process(target=sort_only_most_similar_formula, args=(1,8,10,filepath,load_path,save_path, constrain_workbooks, workbooks)), 
        #     Process(target=sort_only_most_similar_formula, args=(1,9,10,filepath,load_path,save_path, constrain_workbooks, workbooks)),
        #     Process(target=sort_only_most_similar_formula, args=(1,10,10,filepath,load_path,save_path, constrain_workbooks, workbooks)), 
        #     # Process(target=sort_only_most_similar_formula, args=(1,11)),
        #     # Process(target=sort_only_most_similar_formula, args=(1,12)), 
        #     # Process(target=sort_only_most_similar_formula, args=(1,13)),
        #     # Process(target=sort_only_most_similar_formula, args=(1,14)), 
        #     # Process(target=sort_only_most_similar_formula, args=(1,15)),
        #     # Process(target=sort_only_most_similar_formula, args=(1,16)), 
        #     # Process(target=sort_only_most_similar_formula, args=(1,17)),
        #     # Process(target=sort_only_most_similar_formula, args=(1,18)), 
        #     # Process(target=sort_only_most_similar_formula, args=(1,19)), 
        #     # Process(target=sort_only_most_similar_formula, args=(1,20)), 
        # ]
        # [p.start() for p in process]  # 开启了两个进程
        # [p.join() for p in process]   # 等待两个进程依次结束
    # res1 = []
    # res2 = []
    # res = {}
    # for thread_id in range(1,11):
        
    #     print(thread_id)
    #     with open(filepath + "_not_found_res_" + str(thread_id)+'.json','r') as f:
    #         temp1 = json.load(f)
    #     # model1_formulas_disfound_res_
    #     # with open("model2_formulas_disfound_res_"+str(thread_id)+'.json','r') as f:
    #     #     temp1 = json.load(f)
    #     with open(filepath + "_found_res_"+str(thread_id)+'.json','r') as f:
    #         temp2 = json.load(f)
    #     print('found', len(temp1))
    #     print('not found', len(temp2))
    #     for item in temp1:
    #         res1.append(item)
    #     for item in temp2:
    #         res2.append(item)
    
    # print('found', len(res1))
    # print('not found', len(res2))
    # with open(filepath + "_hasfunc_formulas_dis_found_res.json",'w') as f:
    #     json.dump(res1, f) 
    # with open(filepath + "_hasfunc_formulas_dis_not_found_res.json",'w') as f:
    #     json.dump(res2, f) 

def look_test_data():
    with open('Formula_hasfunc_with_id.json','r') as f:
        formulas = json.load(f)
    sheetname2num = {}
    r1c12num = {}

    for formula in formulas:
        sheetname = formula['filesheet'].split('---')[1]
        r1c1 = formula['r1c1']
        if sheetname not in sheetname2num:
            sheetname2num[sheetname] = 0
        if r1c1 not in r1c12num:
            r1c12num[r1c1] = 0

        sheetname2num[sheetname] += 1
        r1c12num[r1c1] += 1
    
    r1c12num_tuple = sorted(r1c12num.items(), key=lambda x: x[1], reverse=True)    
    sheetname2num_tuple = sorted(sheetname2num.items(), key=lambda x: x[1], reverse=True)    

    new_sheetname2num = {}
    new_r1c12num = {}
    for tuple in r1c12num_tuple:
        new_r1c12num[tuple[0]] = tuple[1]
    for tuple in sheetname2num_tuple:
        new_sheetname2num[tuple[0]] = tuple[1]
    with open('testdata_sheetname2num.json', 'w') as f:
        json.dump(new_sheetname2num, f)
    with open('testdata_r1c12num.json', 'w') as f:
        json.dump(new_r1c12num, f)

def sort_only_most_similar_formula(topk, thread_id, batch_num, filepath, load_path, save_path, constrain_workbooks, workbooks): # l2_most_simular_sheet_1900
    # filepath = 'deal00_only_most_similar_formula_1900'. model2_formulas_dis. model1_formulas_dis
    #  filepath = deal00_only_most_similar_formula_1900

    # with open('Formula_77772_with_id.json','r') as f:
    #     formulas = json.load(f)
    # with open('Formulas_top10domain_with_id.json','r') as f:
    
    with open('fortune500_formulatoken2r1c1.json','r') as f:
    # with open('top10domain_formulatoken2r1c1.json','r') as f:
        top10domain_formulatoken2r1c1 = json.load(f)
    with open(root_path + 'dedup_workbooks.json','r') as f:
        dedup_workbooks = json.load(f)
    
    
    # with open('Formulas_middle10domain_with_id.json','r') as f:
    #     formulas = json.load(f)
    with open('Formulas_fortune500_with_id.json','r') as f:
        formulas = json.load(f)
    # with open('Formulas_top10domain_with_id.json','r') as f:
    #     formulas = json.load(f)


    # res = np.load(filepath + "_"+str(thread_id)+".npy", allow_pickle=True).item()
    found_res = []
    not_found_res = []
    # print('start', thread_id, len(res))
    # print('len res', len(res))

    num = 0
    batch_len = len(formulas)/batch_num
    count = 0
    ne_cout = 0
    ne_cout1 = 0
    ne_cout2 = 0
    # print(len(formulas))
    for index,formula in enumerate(formulas):
        if index != batch_num:
            if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                continue
        else:
            if index <= batch_len * (thread_id - 1 ):
                continue
        formula_token = formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
        if formula_token not in workbooks:
            continue
        # if formula_token != '370f0a2c8e9e3329dcf56c7be0949ef7_d3d3LmFlci5nb3YuYXUJMTUyLjkxLjUzLjE5Mw==.xlsx---OP2-Project Cost---10---2':
            # continue
        
        # print('res[0]', list(res.keys())[0])
        # if formula_token not in res:
            # continue
        # num += 1
        # print('num', num)
        # formula_token = formula_token.replace('.xls.xlsx', '.xlsx')
        
        if not os.path.exists(root_path + 'afterfeature_test/' + formula_token + '.npy'):
            if not os.path.exists(root_path + 'afterfeature/' + formula_token + '.npy'):
                ne_cout2 += 1
                continue
            shutil.copy(root_path + 'afterfeature/' + formula_token + '.npy', root_path + 'afterfeature_test/' + formula_token +'.npy')
            if not os.path.exists(root_path + 'afterfeature_test/' + formula_token + '.npy'):
                # print('not exists:afterfeature_test')
            
                # if not os.path.exists(root_path + 'afterfeature_test/' + formula_token + '.npy'):
                    # print(root_path + 'afterfeature_test/' + formula_token + '.npy')
                    # ne_cout += 1
                ne_cout2 += 1
                continue
        # formula_token = formula_token.replace('.xls.xlsx', '.xlsx')
        # print(load_path + '/' + formula_token)
        if not os.path.exists(load_path + '/' + formula_token + '.npy'):
            # print('not exists', load_path + '/' + formula_token + '.npy')
            # ne_cout += 1 
            ne_cout1 += 1
            continue
        count += 1
        # print(count)
        if os.path.exists(save_path + '/' + formula_token + '.json'):
            ne_cout += 1 
            continue
        print(root_path + 'afterfeature_test/' + formula_token + '.npy')
        print("Xxxxxx")
       
        # print('formula_token',formula_token)
        # try:
        formula_item = np.load(load_path + '/' + formula_token+ '.npy', allow_pickle=True).item()
        # except:
            # continue
        # print('formula_item',formula_item)
        # print('formula token', formula_token)
        # res_list = sorted(res[formula_token].items(), key=lambda x: x[1], reverse=False)[0:topk]
        res_list = sorted(formula_item.items(), key=lambda x: x[1], reverse=False)
        new_res = {}
        for tuple in res_list:
            new_res[tuple[0]] = tuple[1]

        res_list = new_res
        found_ = False
        found_formula_token = ''
        found_r1c1 = ''
        # print('formula_item', formula_item)
        print('len res_list', len(res_list))
        if len(res_list) == 0:
            continue
        for other_formula_token in res_list:
            # print('item[0', item, res_list[item])
            # for other_formula in formulas:
                # other_formula_token = other_formula['filesheet'].split('/')[-1] + '---' + str(other_formula['fr']) + '---' + str(other_formula['fc'])
                # other_formula_token = other_formula_token
                # itemfile = item.split('---')[0]
                other_file = other_formula_token.split('---')[0]
                other_filename = other_formula_token.split('---')[0]
                
                if other_filename not in dedup_workbooks and other_filename not in constrain_workbooks:
                    continue
                
                
                # print('itemfile', itemfile)
                # print('other_file', other_file)
                # if other_file == itemfile:
                    # print('other_formula_token', other_formula_token)
                    # print('item', item)
                # print("other_formula['filesheet']", other_formula['filesheet'])
                # if other_formula_token != item:
                #     continue
                # print('candidate')
                print("formula['r1c1']", formula['r1c1'])
                print("other_formula['r1c1']",found_r1c1)
                found_formula_token = other_formula_token
                if found_formula_token in top10domain_formulatoken2r1c1:
                    found_r1c1 = top10domain_formulatoken2r1c1[found_formula_token]
                else:
                    found_r1c1 = ''
                if formula['r1c1'] ==found_r1c1:
                    found_ = True
                    
                    
                    break

                    # print("formula['r1c1']", formula['r1c1'])
                    # print("other_formula['r1c1']", other_formula['r1c1'])
        # print('found_', found_)
        if found_:
            print("\033[1;32m found \033[0m")
            with open(save_path + '/' + formula_token + '.json', 'w') as f:
                json.dump([formula_token, found_formula_token,formula['r1c1'],found_r1c1,True], f)
            # found_res.append(formula['id'])
        else:
            print("\033[1;33m not found \033[0m")
            with open(save_path + '/' + formula_token + '.json', 'w') as f:
                json.dump([formula_token, found_formula_token,formula['r1c1'],found_r1c1,False], f)
        # break
            # not_found_res.append(formula['id'])
    print('count', count)
    print('exists', ne_cout)
    print('no load', ne_cout1)
    print('no after feature', ne_cout2)
    # print('len(dedup_workbooks', len(dedup_workbooks))
    # print('found_res', thread_id,  len(found_res))
    # print('not_found_res', thread_id, len(not_found_res))
    # with open(filepath + "_found_res_"+str(thread_id)+".json",'w') as f:
    #     json.dump(found_res,f)
    # with open(filepath + "_not_found_res_"+str(thread_id)+".json",'w') as f:
    #     json.dump(not_found_res,f)

def sort_only_template_most_similar_formula(topk, thread_id, filepath='model1_formulas_dis'): # l2_most_simular_sheet_1900
    with open('Formula_hasfunc_with_id.json','r') as f:
        formulas = json.load(f)
    with open('formula_token_2_template_id_custom.json', 'r') as f:
        formula_token_2_template_id = json.load(f)
    res = np.load(filepath + "_"+str(thread_id)+".npy", allow_pickle=True).item()
    found_res = []
    not_found_res = []
    print('start', thread_id, len(res))
    print('len res', len(res))
    all_not_found_temp = 0
    num = 0
    for index,formula in enumerate(formulas):
        
        formula_token = formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
        # if formula_token != '0a94a6ef9551a96e60082fbd8c71df8b_d3d3LmFlci5nb3YuYXUJMTUyLjkxLjUzLjE5Mw==.xls.xlsx---Input---177---5':
        #     continue
        if formula_token not in res:
            continue
        # print(index)
        num += 1
        # print('num', num)
        res_list = sorted(res[formula_token].items(), key=lambda x: x[1], reverse=False)[0:topk]
        found_ = False
        for item in res_list:
            for other_formula in formulas:
                other_formula_token = other_formula['filesheet'].split('/')[-1] + '---' + str(other_formula['fr']) + '---' + str(other_formula['fc'])
                if other_formula_token != item[0]:
                    continue
                if formula_token not in formula_token_2_template_id:
                    formula_temp_id = -1
                else:
                    formula_temp_id = formula_token_2_template_id[formula_token]

                if other_formula_token not in formula_token_2_template_id:
                    other_formula_temp_id = -1
                else:
                    other_formula_temp_id = formula_token_2_template_id[other_formula_token]

                
                if  formula_temp_id == other_formula_temp_id:
                    if formula_temp_id == -1 and other_formula_temp_id == -1:
                        all_not_found_temp += 1
                    found_ = True
                    break
            if found_:
                break
        # print('found_', found_)
        if found_:
            found_res.append(formula['id'])
        else:
            not_found_res.append(formula['id'])
    print('all_not_found_temp', all_not_found_temp)
    print('found_res', thread_id,  len(found_res))
    print('not_found_res', thread_id, len(not_found_res))
    with open("found_res_"+str(thread_id)+".json",'w') as f:
        json.dump(found_res,f)
    with open("not_found_res_"+str(thread_id)+".json",'w') as f:
        json.dump(not_found_res,f)

def sort_most_similar_formula(topk, thread_id):
    with open('Formula_77772_with_id.json','r') as f:
        formulas = json.load(f)
    res = np.load("deal00_most_similar_formula_1900_"+str(thread_id)+".npy", allow_pickle=True).item()
    found_res = []
    not_found_res = []
    print('start', thread_id)
    for index,formula in enumerate(formulas):
        
        formula_token = formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
        if formula_token != '0a94a6ef9551a96e60082fbd8c71df8b_d3d3LmFlci5nb3YuYXUJMTUyLjkxLjUzLjE5Mw==.xls.xlsx---Input---177---5':
            continue
        if formula_token not in res:
            continue
        # print(index)
        res_list = sorted(res[formula_token].items(), key=lambda x: x[1], reverse=True)[0:topk]
        found_ = False
        for item in res_list:
            for other_formula in formulas:
                other_formula_token = other_formula['filesheet'].split('/')[-1] + '---' + str(other_formula['fr']) + '---' + str(other_formula['fc'])
                if other_formula_token != item[0]:
                    continue
                if formula['r1c1'] == other_formula['r1c1']:
                    found_ = True
        print('found_', found_)
        if found_:
            found_res.append(formula['id'])
        else:
            not_found_res.append(formula['id'])
    # print('found_res', len(found_res))
    # print('not_found_res', len(not_found_res))
    # with open("found_res_"+str(thread_id)+".json",'w') as f:
    #     json.dump(found_res,f)
    # with open("not_found_res_"+str(thread_id)+".json",'w') as f:
    #     json.dump(not_found_res,f)

def look_res_len():
    num = 0
    new_res = {}
    for i in range(1,21):
        res = np.load('most_similar_formula_1900_'+str(i)+'.npy',allow_pickle=True).item()
        for key in res:
            new_res[key] = res[key]
        num += len(res)
        print(i, num)
    # np.save('most_similar_formula_1900.npy', new_res)


def naive_suc_model2_fail():
    with open('formula_token_2_template_id.json', 'r') as f:
        formula_token_2_template_id = json.load(f)
    with open('formula_token_2_r1c1.json', 'r') as f:
        formula_token_2_r1c1 = json.load(f)
    print('naive true, model2 false')
    with open("naive_hasfunc_found_true_res.json", 'r') as f:
        naive_found_true_1900 = json.load(f)
        # naive_hasfunc_found_true_res, naive_hasfunc_found_false_res, naive_hasfunc_not_found_res
        # naive_found_true_1900_formulas
    with open('Formula_77772_with_id.json','r') as f:
        formulas = json.load(f)

    count = 0
    for thread_id in range(1,21):
    # for thread_id in range(1,21):
        res = np.load("model2_formulas_dis_"+str(thread_id)+".npy", allow_pickle=True).item()
        
        with open("model2_formulas_disnot_found_res_"+str(thread_id)+".json", 'r') as f:
            not_found_res_1 = json.load(f)
        found_res = []
        not_found_res = []
        print('start', thread_id)

        all_fail = 0
        same_template = 0

        for index,formula in enumerate(formulas):
            # print(thread_id, index, len(formulas))
            if formula['id'] not in not_found_res_1 or not formula['id'] in naive_found_true_1900:
                continue
            all_fail += 1
            formula_token = formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
            if formula_token not in res:
                continue
            if len(res[formula_token]) == 0:
                all_fail += 1
                count += 1
                continue
            res_list = sorted(res[formula_token].items(), key=lambda x: x[1], reverse=False)[0]
            top1_formula_token = res_list[0]
            
            if formula_token not in formula_token_2_template_id:
                template_id_1 = -1
            else:
                template_id_1 = formula_token_2_template_id[formula_token]
            
            if top1_formula_token not in formula_token_2_template_id:
                template_id_2 = -1
            else:
                template_id_2 = formula_token_2_template_id[top1_formula_token]
            if template_id_1 == template_id_2:
                same_template += 1

            count += 1
    print(count)
    print('count', count)

def deal_00(thread_id, batch_num):
    invalid_cell_feature = {
            "background_color_r": 0,
            "background_color_g": 0,
            "background_color_b": 0,
            "font_color_r": 0,
            "font_color_g": 0,
            "font_color_b": 0,
            "font_size": 0.0,
            "font_strikethrough": False,
            "font_shadow": False,
            "font_ita": False,
            "font_bold": False,
            "height": 0.0,
            "width": 0.0,
            "content": '',
            "content_template": '',
        },

    feature_list1 = os.listdir('formulas196/')
    num = 0

    need_rerun_list = []

    batch_len = len(feature_list1)/batch_num
    for index,filename in enumerate(feature_list1):
        if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
            continue
        if os.path.exists('formulas_deal_00/' + filename):
            continue
        print(index, len(feature_list1))
        filename_ = filename.replace('.json', '')
        info_list = filename_.split('---')
        formula_token = info_list[0] + '---' + info_list[1]
        fr = int(info_list[2])
        fc = int(info_list[3])
        
        need_rerun = False
        if fr-50 >= 1:
            start_row =  fr-50 
        else:
            start_row = 1
            need_rerun = True

        if fc-5 >= 1:
            start_column = fc-5
        else:
            start_column = 1
            need_rerun = True

        if need_rerun:
            need_rerun_list.append(filename)
        # if os.path.exists('formulas_deal_00/'+filename):
        #     continue
        # if filename != 'f7b86a0a61fa3220c638c0e37064ce11_d3d3LmtldHNvLmNvbQkyMTMuMTM4LjExMy4yMTI=.xls.xlsx---Workshop_Outputs---3---2.json' :
            # continue
        with open('formulas196/' + filename, 'r') as f:
            origin_feature = json.load(f)
        feature_list = origin_feature['sheetfeature']

        table_feature = {}
        count = 0
        for row in range(start_row, start_row + 100):
            table_feature[row]= {}
            for col in range(start_column, start_column + 10):
                table_feature[row][col] = feature_list[count]
                count += 1

        new_start_row = fr - 50
        new_start_column = fc - 5

        new_feature_list = []
        # pprint.pprint(table_feature)
        for row in range(new_start_row, new_start_row + 100):
            for col in range(new_start_column, new_start_column + 10):
                # print(row, col)
                if row in table_feature:
                    if col in table_feature[row]:
                        new_feature_list.append(table_feature[row][col])
                        # print('valid')
                        continue
                # print('invalid')
                new_feature_list.append(invalid_cell_feature)
        origin_feature['sheetfeature'] = new_feature_list
        # print(len(new_feature_list))
        with open('formulas_deal_00/' + filename, 'w') as f:
            json.dump(origin_feature, f)

def para_only_run():
    process = [
        Process(target=find_only_closed_formula, args=(1,10)),
        Process(target=find_only_closed_formula, args=(2,10)), 
        Process(target=find_only_closed_formula, args=(3,10)),
        Process(target=find_only_closed_formula, args=(4,10)), 
        Process(target=find_only_closed_formula, args=(5,10)),
        Process(target=find_only_closed_formula, args=(6,10)), 
        Process(target=find_only_closed_formula, args=(7,10)),
        Process(target=find_only_closed_formula, args=(8,10)), 
        Process(target=find_only_closed_formula, args=(9,10)),
        Process(target=find_only_closed_formula, args=(10,10)), 
        # Process(target=find_only_closed_formula, args=(11,20)),
        # Process(target=find_only_closed_formula, args=(12,20)), 
        # Process(target=find_only_closed_formula, args=(13,20)),
        # Process(target=find_only_closed_formula, args=(14,20)), 
        # Process(target=find_only_closed_formula, args=(15,20)),
        # Process(target=find_only_closed_formula, args=(16,20)), 
        # Process(target=find_only_closed_formula, args=(17,20)),
        # Process(target=find_only_closed_formula, args=(18,20)), 
        # Process(target=find_only_closed_formula, args=(19,20)), 
        # Process(target=find_only_closed_formula, args=(20,20)), 
    ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束
def para_only_finetune_run():
    process = [Process(target=find_only_closed_formula_finetune, args=(1,20)),
        Process(target=find_only_closed_formula_finetune, args=(2,20)), 
        Process(target=find_only_closed_formula_finetune, args=(3,20)),
        Process(target=find_only_closed_formula_finetune, args=(4,20)), 
        Process(target=find_only_closed_formula_finetune, args=(5,20)),
        Process(target=find_only_closed_formula_finetune, args=(6,20)), 
        Process(target=find_only_closed_formula_finetune, args=(7,20)),
        Process(target=find_only_closed_formula_finetune, args=(8,20)), 
        Process(target=find_only_closed_formula_finetune, args=(9,20)),
        Process(target=find_only_closed_formula_finetune, args=(10,20)), 
        Process(target=find_only_closed_formula_finetune, args=(11,20)),
        Process(target=find_only_closed_formula_finetune, args=(12,20)), 
        Process(target=find_only_closed_formula_finetune, args=(13,20)),
        Process(target=find_only_closed_formula_finetune, args=(14,20)), 
        Process(target=find_only_closed_formula_finetune, args=(15,20)),
        Process(target=find_only_closed_formula_finetune, args=(16,20)), 
        Process(target=find_only_closed_formula_finetune, args=(17,20)),
        Process(target=find_only_closed_formula_finetune, args=(18,20)), 
        Process(target=find_only_closed_formula_finetune, args=(19,20)), 
        Process(target=find_only_closed_formula_finetune, args=(20,20)), 
    ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束

def para_deal00():
    process = [Process(target=deal_00, args=(1,20)),
               Process(target=deal_00, args=(2,20)), 
               Process(target=deal_00, args=(3,20)),
               Process(target=deal_00, args=(4,20)), 
               Process(target=deal_00, args=(5,20)),
               Process(target=deal_00, args=(6,20)), 
               Process(target=deal_00, args=(7,20)),
               Process(target=deal_00, args=(8,20)), 
               Process(target=deal_00, args=(9,20)),
               Process(target=deal_00, args=(10,20)), 
               Process(target=deal_00, args=(11,20)),
               Process(target=deal_00, args=(12,20)), 
               Process(target=deal_00, args=(13,20)),
               Process(target=deal_00, args=(14,20)), 
               Process(target=deal_00, args=(15,20)),
               Process(target=deal_00, args=(16,20)), 
               Process(target=deal_00, args=(17,20)),
               Process(target=deal_00, args=(18,20)), 
               Process(target=deal_00, args=(19,20)), 
               Process(target=deal_00, args=(20,20)), 
            ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束

def para_dis_only_run():
    process = [Process(target=find_dis_only_closed_formula, args=(1,20)),
        Process(target=find_dis_only_closed_formula, args=(2,20)), 
        Process(target=find_dis_only_closed_formula, args=(3,20)),
        Process(target=find_dis_only_closed_formula, args=(4,20)), 
        Process(target=find_dis_only_closed_formula, args=(5,20)),
        Process(target=find_dis_only_closed_formula, args=(6,20)), 
        Process(target=find_dis_only_closed_formula, args=(7,20)),
        Process(target=find_dis_only_closed_formula, args=(8,20)), 
        Process(target=find_dis_only_closed_formula, args=(9,20)),
        Process(target=find_dis_only_closed_formula, args=(10,20)), 
        Process(target=find_dis_only_closed_formula, args=(11,20)),
        Process(target=find_dis_only_closed_formula, args=(12,20)), 
        Process(target=find_dis_only_closed_formula, args=(13,20)),
        Process(target=find_dis_only_closed_formula, args=(14,20)), 
        Process(target=find_dis_only_closed_formula, args=(15,20)),
        Process(target=find_dis_only_closed_formula, args=(16,20)), 
        Process(target=find_dis_only_closed_formula, args=(17,20)),
        Process(target=find_dis_only_closed_formula, args=(18,20)), 
        Process(target=find_dis_only_closed_formula, args=(19,20)), 
        Process(target=find_dis_only_closed_formula, args=(20,20)), 
    ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束

def count_same_template():
    all_all_fail = 0
    all_same_template = 0
    all_count = 0
    for id_ in range(1, 21):
        all_fail, same_template, count = look_faile(id_,1)
        all_all_fail += all_fail
        same_template += same_template
        all_count += count

    print(all_all_fail)
    print(same_template)
    print(all_count)

def compare_2_before_feature(token1, token2):
    feature1 = np.load('model2_formula2beforebertfeature/' + token1 + '.npy', allow_pickle=True)
    feature2 = np.load('model2_formula2beforebertfeature/' + token2 + '.npy', allow_pickle=True)
    print((feature1 == feature2).all())

def count_fail():
    sheetname2num = {}
    r1c12num = {}
    with open('Formula_hasfunc_with_id.json','r') as f:
        formulas = json.load(f)
    for thread_id in range(1, 21):
        with open("model2_formulas_disnot_found_res_"+str(thread_id)+".json", 'r') as f:
            not_found_res_1 = json.load(f)
        for id_ in not_found_res_1:
            for formula in formulas:
                if id_ == formula['id']:
                    sheetname = formula['filesheet'].split('---')[1]
                    r1c1 = formula['r1c1']
                    if sheetname not in sheetname2num:
                        sheetname2num[sheetname] =0
                    if r1c1 not in r1c12num:
                        r1c12num[r1c1] =0
                    sheetname2num[sheetname] += 1
                    r1c12num[r1c1] += 1

    with open('fail_model2_sheetname.json', 'w') as f:
        json.dump(sheetname2num, f)
    with open('fail_r1c12num.json', 'w') as f:
        json.dump(r1c12num, f)

def look_sheetnum():
    filesheets = set()
    with open('Formulas_top10domain_with_id.json','r') as f:
        formulas = json.load(f)

    for formula in formulas:
        filesheets.add(formula['filesheet'])

    print(len(filesheets))

    with open(root_path + "top10domain_most_simular_sheet.json", 'r') as f:
        res = json.load(f)

    print(len(res))
    
def para_simple():
    # with open('middle10domain_formulatoken2r1c1.json','r') as f:
    with open('fortune500_formulatoken2r1c1.json','r') as f:
        top10domain_formulatoken2r1c1 = json.load(f)
    with open('r1c12template_fortune500.json','r') as f:
        r1c12template_top10domain = json.load(f)
    suc_sheets = os.listdir(root_path + 'model1_similar_sheet')
    suc_sheets = [i.replace('.json','') for i in suc_sheets]
    print('len(suc_sheets', len(suc_sheets))
    with open('/datadrive-2/data/fortune500_test/dedup_workbooks.json','r') as f:
        dedup_workbooks = json.load(f)
    def simple(thread_id, batch_num):
        print('start', thread_id)
        filelist = os.listdir(root_path + 'model1_res')
        filelist.sort()
        simple_found_true = 0
        simple_found_false = 0
        simple_template_found_true = 0
        simple_template_found_false = 0
        batch_len = len(filelist)/batch_num
        for index,filename in enumerate(filelist):
            
            if index != batch_num:
                if(index <= batch_len * (thread_id - 1 ) or index > batch_len * thread_id):
                    continue
            else:
                if index <= batch_len * (thread_id - 1 ):
                    continue
            # print(index, len(filelist))
            
            filename = filename.replace('.json','')
            fname = filename.split('---')[0]
            if fname not in dedup_workbooks:
                continue
            sheetname = filename.split("---")[1]
            filesheet = fname + '---' + sheetname
            if os.path.exists(root_path + 'dedup_simple_res/' + filename + '.json'):
                print('exists')
                continue
            if not os.path.exists(root_path + 'dedup_model1_res/' + filename + '.json'):
                continue
            if os.path.exists(root_path + 'dedup_simple_res/' + filename + '.json'):
                continue
            print('filesheet', filesheet)
            if filesheet not in suc_sheets:
                print('not in suc_sheets')
                # print('suc_sheets', suc_sheets[0])
                continue
            print(index, len(filelist))
            with open(root_path + 'model1_res/' + filename+'.json', 'r') as f:
                res = json.load(f)
            
            
            fr = filename.split('---')[2]
            fc = filename.split('---')[3]
            simple_found = False
        
            for other_formulatoken in top10domain_formulatoken2r1c1:
                other_filename = other_formulatoken.split("---")[0]
                other_sheetname = other_formulatoken.split("---")[1]
                other_fr = other_formulatoken.split("---")[2]
                other_fc = other_formulatoken.split("---")[3]
                if other_filename not in dedup_workbooks:
                    continue
                # print('other_filename', other_filename)
                # print('filename', filename)
                # print('other_sheetname', other_sheetname)
                # print('sheetname', sheetname)
               
                # if other_sheetname == sheetname:
                #     print('other_formulatoken', other_formulatoken)
                #     print('filename', filename)
                #     print('other_fc', type(other_fc))
                #     print('fc', fc)
                #     print('other_fr', other_fr)
                #     print('fr', fr)
                #     print('other_fc == fc and other_fr == fr', other_fc == fc and other_fr == fr)
                if other_filename != fname and other_sheetname == sheetname and other_fc == fc and other_fr == fr:
                    found_formulatoken = other_formulatoken
                    res_r1c1 = top10domain_formulatoken2r1c1[other_formulatoken]
                    simple_found=  True
                    with open(root_path + 'dedup_simple_res/' + filename + '.json', 'w') as f:
                        json.dump([found_formulatoken], f)
                    break
            print('simple_found', simple_found)
            
        #     if not simple_found:
        #         simple_found_false += 1
        #         simple_template_found_false += 1
        #         continue

        #     target_r1c1 = res[2]
            
        #     if target_r1c1 == res_r1c1:
        #         simple_found_true += 1
        #         simple_template_found_true += 1
        #     else:
        #         if target_r1c1 not in r1c12template_top10domain:
        #             simple_target_templateid = -1
        #         else:
        #             simple_target_templateid = r1c12template_top10domain[target_r1c1]
        #         if res_r1c1 not in r1c12template_top10domain:
        #             simple_res_templateid = -1
        #         else:
        #             simple_res_templateid = r1c12template_top10domain[res_r1c1]

        #         if simple_target_templateid == simple_res_templateid:
        #             simple_template_found_true += 1
        #         else:
        #             simple_template_found_false += 1
        #         simple_found_false += 1
        # print('simple_found_false', simple_found_false)
        # print('simple_found_true', simple_found_true)
        # print('simple_template_found_false', simple_template_found_false)
        # print('simple_template_found_true', simple_template_found_true)
    # simple(1,1)
    process = [Process(target=simple, args=(1,30)),
               Process(target=simple, args=( 2,30)), 
               Process(target=simple, args=( 3,30)),
               Process(target=simple, args=( 4,30)), 
               Process(target=simple, args=( 5,30)),
               Process(target=simple, args=( 6,30)), 
               Process(target=simple, args=( 7,30)),
               Process(target=simple, args=( 8,30)), 
               Process(target=simple, args=( 9,30)),
               Process(target=simple, args=( 10,30)), 
               Process(target=simple, args=( 11,30)),
               Process(target=simple, args=( 12,30)), 
               Process(target=simple, args=( 13,30)),
               Process(target=simple, args=( 14,30)), 
               Process(target=simple, args=( 15,30)),
               Process(target=simple, args=( 16,30)), 
               Process(target=simple, args=( 17,30)),
               Process(target=simple, args=( 18,30)), 
               Process(target=simple, args=( 19,30)), 
               Process(target=simple, args=( 20,30)), 
               Process(target=simple, args=( 21,30)), 
               Process(target=simple, args=( 22,30)),
               Process(target=simple, args=( 23,30)), 
               Process(target=simple, args=( 24,30)),
               Process(target=simple, args=( 25,30)), 
               Process(target=simple, args=( 26,20)),
               Process(target=simple, args=( 27,30)), 
               Process(target=simple, args=( 28,30)),
               Process(target=simple, args=( 29,30)), 
               Process(target=simple, args=( 30,30)), 

            ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束
    # simple(1,1)
def look_fail():
    # with open('r1c12template_top10domain_constant.json','r') as f:
        # r1c12template_top10domain = json.load(f)
    with open('r1c12template_fortune500_constant.json','r') as f:
        r1c12template_top10domain = json.load(f)
    # with open('r1c12template_middle10domain.json','r') as f:
        # r1c12template_top10domain = json.load(f)
    # with open('top10domain_formulatoken2r1c1.json','r') as f:
    with open('fortune500_formulatoken2r1c1.json','r') as f:
    # with open('middle10domain_formulatoken2r1c1.json','r') as f:
        top10domain_formulatoken2r1c1 = json.load(f)
    # with open('/datadrive-2/data/top10domain_test/dedup_workbooks.json','r') as f:
    with open('/datadrive-2/data/fortune500_test/dedup_workbooks.json','r') as f:
        dedup_workbooks = json.load(f)
    with open('fortune500_test_formula_token.json', 'r') as f:
        test_formulas = json.load(f)
    suc_sheets = os.listdir(root_path + 'model1_similar_sheet')
    suc_sheets = [i.replace('---1---1.json','') for i in suc_sheets]
    # test_formulas = os.listdir(root_path + 'model1_top10domain_formula2afterfeature_test')
    # test_formulas = os.listdir(root_path + 'afterfeature_test')
    # test_formulas = [i.replace('.npy','') for i in test_formulas]
    fail_formulas = []

    count = 0
    print('len(test_formulas', len(test_formulas))
    new_top10domain_formulatoken2r1c1 = {}
    for item in top10domain_formulatoken2r1c1:
        new_top10domain_formulatoken2r1c1[item] = top10domain_formulatoken2r1c1[item]
    top10domain_formulatoken2r1c1 = new_top10domain_formulatoken2r1c1
    # print('top10domain_formulatoken2r1c1', list(top10domain_formulatoken2r1c1.keys()))
    with open('/datadrive-2/data/fortune500_test/dedup_workbooks.json','r') as f:
        dedup_workbooks = json.load(f)
    
    for index,formula_token in enumerate(test_formulas):

        count += 1
        # print(count)
        filename = formula_token.split('---')[0]
        
        sheetname = formula_token.split('---')[1]
        filesheet = filename +'---' + sheetname
        # print('filesheet', filesheet)
        # print('suc_sheets', suc_sheets[0])
        if filesheet not in suc_sheets:
            # print('formula_token',formula_token)
            fail_formulas.append(filesheet)
    # print('in suc_sheets', "012dcc10c41410112265d9556ec89bb1_d3d3LmFlci5nb3YuYXUJMTUyLjkxLjUzLjE5Mw==.xls.xlsx---Analysis" in suc_sheets)
    # print('in fail_formulas', "012dcc10c41410112265d9556ec89bb1_d3d3LmFlci5nb3YuYXUJMTUyLjkxLjUzLjE5Mw==.xls.xlsx---Analysis" in fail_formulas)
    # print('len fail_formulas', len(fail_formulas))
    filelist = os.listdir(root_path + 'dedup_model1_res')
    filelist.sort()
    file_sheet_num = {}
    
    found_true = 0
    found_false = 0
    template_found_true = 0
    template_found_false = 0
    count =0

    naive_found_true = 0
    naive_found_false = 0
    naive_template_found_true = 0
    naive_template_found_false = 0

    simple_found_true = 0
    simple_found_false = 0
    simple_template_found_true = 0
    simple_template_found_false = 0

    model1_suc_but_fail = 0
    model1_fail_but_suc = 0

    naive_true_found_false = 0

    download = set()

    sheetnames = set()

    ne_count = 0

    all_fail = 0
    all_suc = 0

    copy_fail_same_table = []
    copy_fail_multitable = []
    for index, formula_token in enumerate(test_formulas):
        filename = formula_token + '.json'
        model1_found = False
        naive_found = False
        print(index, len(test_formulas))
        fname = filename.split("---")[0]
        sheetname = filename.split("---")[1]
        fr = filename.split('---')[2]
        fc = filename.split('---')[3]
        filesheet = fname + '---' + sheetname
        if fname not in dedup_workbooks:
            continue
        # if sheetname == 'Analysis':
            # continue
        # print("########")
        # print(filename)
        same_position = False
        model1 = False
        with open(root_path + 'dedup_model1_res/' + filename, 'r') as f:
            res = json.load(f)
        if 'RC[-1]+ROUND(10*R' in res[2]:
            continue
        if not os.path.exists(root_path + 'dedup_model1_naive_res/' + filename):
            print(root_path + 'dedup_model1_naive_res/' + filename)
            ne_count += 1
            continue
        with open(root_path + 'dedup_model1_naive_res/' + filename, 'r') as f:
            naive_res = json.load(f)

        ############### simple 
        if not os.path.exists(root_path + 'dedup_simple_res/' + filename):
            simple_found_false += 1
            simple_template_found_false += 1
            # continue
            # print('not exists')
        else:
            # print('exists')
            with open(root_path + 'dedup_simple_res/' + filename, 'r') as f:
                simple_res = json.load(f)

            target_r1c1 = res[2]
            simple_res[0] = simple_res[0]
            res_r1c1 = top10domain_formulatoken2r1c1[simple_res[0]]
            # print('target_r1c1', target_r1c1)
            # print('res_r1c1', res_r1c1)
            if target_r1c1 not in r1c12template_top10domain:
                simple_target_templateid = -1
            else:
                simple_target_templateid = r1c12template_top10domain[target_r1c1]
            if res_r1c1 not in r1c12template_top10domain:
                simple_res_templateid = -1
            else:
                simple_res_templateid = r1c12template_top10domain[res_r1c1]
            if target_r1c1 == res_r1c1:
                simple_found_true += 1
            else:
                simple_found_false += 1
            if simple_target_templateid == simple_res_templateid:
                simple_template_found_true += 1
                simple_found = True
            else:
                simple_template_found_false += 1
            

        # distance = np.load(root_path + 'top10domain_model1_formulas_dis/' + filename.replace('.json', '.npy'), allow_pickle=True)

        



        ############### naive

        # print(res)
        # print(res[])
        if naive_res[0] == 'not found':
            naive_found_false += 1
            naive_template_found_false += 1
        else:
            # print('naive_res', naive_res)
            if naive_res[1] == True:
                naive_found_true += 1
                naive_template_found_true += 1
                naive_found = True
            else:
                # target_r1c1 = top10domain_formulatoken2r1c1[r1c1]
                res_r1c1 = top10domain_formulatoken2r1c1[naive_res[0]]
                if target_r1c1 not in r1c12template_top10domain:
                    naive_target_templateid = -1
                else:
                    naive_target_templateid = r1c12template_top10domain[target_r1c1]
                if res_r1c1 not in r1c12template_top10domain:
                    naive_res_templateid = -1
                else:
                    naive_res_templateid = r1c12template_top10domain[res_r1c1]

                if naive_target_templateid == naive_res_templateid:
                    naive_template_found_true += 1
                    naive_found = True
                else:
                    naive_template_found_false += 1
                naive_found_false += 1

        if res[4] == False:
            target_r1c1 = res[2]
            res_r1c1 = res[3]
            if target_r1c1 not in r1c12template_top10domain:
                target_templateid = -1
            else:
                target_templateid = r1c12template_top10domain[target_r1c1]
            if res_r1c1 not in r1c12template_top10domain:
                res_templateid = -1
            else:
                res_templateid = r1c12template_top10domain[res_r1c1]

            if target_templateid == res_templateid:
                template_found_true += 1
                model1_found = True
                # print(res)
                if '!' in target_r1c1:
                    copy_fail_multitable.append(res)
                else:
                    copy_fail_same_table.append(res)
            else:
                template_found_false += 1
            # count += 1
            # sheetname = res[0].split('---')[1]
            # # if count == 2:
            # dis = np.load(root_path + 'top10domain_model1_formulas_dis/' + filename.replace('.json','.npy'), allow_pickle=True).item()
            
            # filename = res[0].split('---')[0]
        
            # if sheetname != 'Analysis':
            #     continue
            # print('res0', res[0])
            # print("distance...................")
            # dis = sorted(dis.items(), key = lambda x: x[1])
            # # for item in dis:
            #     # print(item[0], item[1])
            # print("similar sheets...............")
            # print("filename + '---' + sheetname", filename + '---' + sheetname)
            # print(similar_sheets[filename + '---' + sheetname])
            
            # print("res.......................")
            # print(res)
            # break
            # if sheetname not in file_sheet_num:
            #     file_sheet_num[sheetname] = 0
            # file_sheet_num[sheetname] += 1
            found_false += 1
        else:
            model1_found = True
            found_true += 1
            template_found_true += 1

        # print('filesheet', filesheet)
        # print('fail_formulas[0]', fail_formulas[0])
        # print('filename in fail_formulas', filesheet not in fail_formulas)
        if model1_found and not naive_found:
            model1_suc_but_fail += 1
        if not model1_found and naive_found:
            model1_fail_but_suc += 1
        if not model1_found and not naive_found:
            all_fail += 1
        if model1_found and naive_found:
            all_suc += 1

        # if not naive_found:
        # if not model1_found and not naive_found and filesheet not in fail_formulas:
             
        #     # print(similar_sheets[fname + '---' + sheetname])
        #     if sheetname in sheetnames:
        #         continue
        #     print("###########")
        #     print(filename)
        #     naive_true_found_false += 1
        #     print(res)
        #     print(naive_res)  
        #     # print(distance)
        #     sheetnames.add(sheetname)
        #     download.add(res[0])
        #     download.add(res[1])
        #     download.add(naive_res[0])
        #     break
            
    download = list(download)

    datalist = os.listdir('/datadrive/data')
    datalist = [i for i in datalist if len(i) == 3]
    add = []
    # for filesheet in filelist:
    #     if filesheet.split('---')[0] in add:
    #         continue
    #     add.append(filesheet.split('---')[0])
    #     for item in datalist:
    #         subdatalist = os.listdir('/datadrive/data/' + item)
    #         if filesheet.split('---')[0] in subdatalist:
    #             shutil.copy('/datadrive/data/'+item + '/' + filesheet.split('---')[0] , '/datadrive-2/data/middle10domain_test/data/')
    #             break
    print('copy_fail_same_table', len(copy_fail_same_table))
    print('copy_fail_multitable', len(copy_fail_multitable))
    with open("copy_fail_same_table.json", 'w') as f:
        json.dump(copy_fail_same_table, f)
    with open("copy_fail_multitable.json", 'w') as f:
        json.dump(copy_fail_multitable, f)
    print('len failformulas', len(fail_formulas))
    print('naive_true_found_false', naive_true_found_false)
    print('model1 r1c1', 'false:', found_false, 'true:',found_true)
    print('model1 template','false:',template_found_false, 'true:',template_found_true)

    print('naive r1c1','false:',naive_found_false, 'true:',naive_found_true)
    print('naive template','false:',naive_template_found_false, 'true:',naive_template_found_true)

    print('simple r1c1','false:',simple_found_false, 'true:',simple_found_true)
    print('simple template', 'false:',simple_template_found_false, 'true:',simple_template_found_true)
    print('model1_suc_but_fail', model1_suc_but_fail)
    print('model1_fail_but_suc', model1_fail_but_suc)

    print('all fail', all_fail)
    print('all suc', all_suc)
    # with open("file_sheet_num.json", 'w') as f:
        # json.dump(file_sheet_num, f)
    print('ne_count', ne_count)

def analyze_nodelist_length():
    with open('r1c12template_top10domain.json','r') as f:
        r1c12template_middle10domain = json.load(f)
    with open('top10domain_formulatoken2r1c1.json','r') as f:
        middle10domain_formulatoken2r1c1 = json.load(f)
    with open('formula_r1c1_top10domain_template.json', 'r') as f:
        formula_r1c1_middle10domain_template = json.load(f)

    id2nodelistlen = {}
    for item in formula_r1c1_middle10domain_template:
        id2nodelistlen[item['id']] = len(item['node_list'])
    nodelistlen2formulatoken = {}

    for formula_token in middle10domain_formulatoken2r1c1:
        r1c1 = middle10domain_formulatoken2r1c1[formula_token]
        if r1c1 not in r1c12template_middle10domain:
            continue
        template_id = r1c12template_middle10domain[r1c1]
        nodelist_len = id2nodelistlen[template_id]
        if nodelist_len not in nodelistlen2formulatoken:
            nodelistlen2formulatoken[nodelist_len] = []
        nodelistlen2formulatoken[nodelist_len].append(formula_token)
    
    nodelistlen2res = {}
    nodelistlen2num = {}
    for index,nodelistlen in enumerate(nodelistlen2formulatoken):
        
        template_found_true  = 0
        template_found_false = 0
        nodelistlen2res[nodelistlen] = []
        for formula_token in nodelistlen2formulatoken[nodelistlen]:
            model1_found = False
            naive_found = False
            # print(index, len(filelist))
            fname = formula_token.split("---")[0]
            sheetname = formula_token.split("---")[1]
            fr = formula_token.split('---')[2]
            fc = formula_token.split('---')[3]
            filesheet = fname + '---' + sheetname
            if not os.path.exists(root_path + 'top10domain_model1_res/' + formula_token + '.json'):
                continue
        
            with open(root_path + 'top10domain_model1_res/' + formula_token + '.json', 'r') as f:
                res = json.load(f)
            with open(root_path + 'model1_naive_res/' + formula_token + '.json', 'r') as f:
                naive_res = json.load(f)

            ############### simple 
            target_r1c1 = res[2]
            # print(res)
            # print('xxxx')
            if res[4] == False:
                target_r1c1 = res[2]
                res_r1c1 = res[3]
                if target_r1c1 not in r1c12template_middle10domain:
                    target_templateid = -1
                else:
                    target_templateid = r1c12template_middle10domain[target_r1c1]
                if res_r1c1 not in r1c12template_middle10domain:
                    res_templateid = -1
                else:
                    res_templateid = r1c12template_middle10domain[res_r1c1]

                if target_templateid == res_templateid:
                    template_found_true += 1
                    nodelistlen2res[nodelistlen].append(True)
                else:
                    template_found_false += 1
                    nodelistlen2res[nodelistlen].append(False)

            else:
                model1_found = True
                template_found_true += 1
                nodelistlen2res[nodelistlen].append(True)
        # print('template_found_true', template_found_false)
        # print('template_found_false', template_found_false)
        if template_found_false == 0 and template_found_false == 0:
            continue
        
        nodelistlen2num[nodelistlen] = template_found_true + template_found_false
    # with open(root_path + 'nodelistlen2res.json', 'w') as f:
    #     json.dump(nodelistlen2res, f)
    with open(root_path + 'nodelistlen2num.json', 'w') as f:
        json.dump(nodelistlen2num, f)

    index_list = []
    for length in nodelistlen2num:
        for item in range(0, nodelistlen2num[length]):
            index_list.append(length)

    first_point = np.percentile(index_list, 25)
    second_point = np.percentile(index_list, 50)
    last_point = np.percentile(index_list, 75)

    print(np.percentile(index_list, 25))
    print(np.percentile(index_list, 50))
    print(np.percentile(index_list, 75))
    print(np.percentile(index_list, 100))

    first_true = 0
    first_all = 0
    second_true = 0
    second_all = 0
    third_true = 0
    third_all = 0
    last_true = 0
    last_all = 0
    for length in nodelistlen2res:
        if length < first_point:
            first_all += len(nodelistlen2res[length])
            first_true += len([item for item in nodelistlen2res[length] if item == True])
        elif length >= first_point and length < second_point:
            second_all += len(nodelistlen2res[length])
            second_true += len([item for item in nodelistlen2res[length] if item == True])
        elif length >= second_point and length < last_point:
            third_all += len(nodelistlen2res[length])
            third_true += len([item for item in nodelistlen2res[length] if item == True])
        if length >= last_point:
            last_all += len(nodelistlen2res[length])
            last_true += len([item for item in nodelistlen2res[length] if item == True])

    print('first:', first_true / first_all)
    print('second:', second_true / second_all)
    print('third:', third_true / third_all)
    print('last:', last_true / last_all)

    print('first_num', first_all)
    print('second_num', second_all)
    print('third_num', third_all)
    print('last_num', last_all)
def generate_formula_token_2_r1c1():
    # with open("Formulas_middle10domain_with_id.json",'r') as f:
    #     formulas = json.load(f)
    with open("Formulas_fortune500_with_id.json",'r') as f:
        formulas = json.load(f)

    formula_token_2_r1c1 = {}

    for formula in formulas:
        formula_token = formula['filesheet'].split("/")[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc'])
        formula_token_2_r1c1[formula_token] = formula['r1c1']
        # if formula_token == "23938280411769767155305058958183774721-ltd_trading_authority_2022.xlsx---Q1-2022-trans_summary---12---5":
            # print('found')


    # with open("middle10domain_formulatoken2r1c1.json",'w') as f:
    #     json.dump(formula_token_2_r1c1, f)
    with open("fortune500_formulatoken2r1c1.json",'w') as f:
        json.dump(formula_token_2_r1c1, f)

def analyze_similar_sheet_notfound():
    # fail_sheets = []
    suc_sheets = os.listdir(root_path + 'model1_similar_sheet')
    suc_sheets = [i.replace('.json','') for i in suc_sheets]
    test_formulas = os.listdir(root_path + 'model1_top10domain_formula2afterfeature_test')
    test_formulas = [i.replace('.npy','') for i in test_formulas]
    fail_formulas = []

    count = 0

    for index,formula_token in enumerate(test_formulas):

        count += 1
        print(count)
        filename = formula_token.split('---')[0]
        sheetname = formula_token.split('---')[1]
        filesheet = filename +'---' + sheetname

        if filesheet not in suc_sheets:
            fail_formulas.append(formula_token)
    print(fail_formulas)

def r1c12templateid():
    # with open('formula_r1c1_top10domain_template_constant.json','r') as f:
    with open('formula_r1c1_top10domain_template.json','r') as f:
        templates = json.load(f)
    res = {}
    for template in templates:
        template_id = template['id']
        formulas = template['formulas']
        for formula in formulas:
            res[formula] = template_id

    with open('r1c12template_top10domain_constant.json', 'w') as f:
        json.dump(res, f)
    

def look_how_shift():
    # with open('r1c12template_top10domain.json','r') as f:
        # r1c12template_top10domain = json.load(f)
    with open('r1c12template_top10domain_constant.json','r') as f:
        r1c12template_top10domain = json.load(f)
    with open('top10domain_formulatoken2r1c1.json','r') as f:
        top10domain_formulatoken2r1c1 = json.load(f)

    suc_sheets = os.listdir(root_path + 'model1_similar_sheet')
    suc_sheets = [i.replace('---1---1.json','') for i in suc_sheets]
    test_formulas = os.listdir(root_path + 'model1_top10domain_formula2afterfeature_test')
    test_formulas = [i.replace('.npy','') for i in test_formulas]
    fail_formulas = []

    count = 0
    print('len(test_formulas', len(test_formulas))

    res_pair = []
    all_target_set = set()
    for index,formula_token in enumerate(test_formulas):

        count += 1
        # print(count)
        filename = formula_token.split('---')[0]
        sheetname = formula_token.split('---')[1]
        filesheet = filename +'---' + sheetname
        # print('filesheet', filesheet)
        # print('suc_sheets', suc_sheets[0])
        if filesheet not in suc_sheets:
            # print('formula_token',formula_token)
            fail_formulas.append(filesheet)
    # print('in suc_sheets', "012dcc10c41410112265d9556ec89bb1_d3d3LmFlci5nb3YuYXUJMTUyLjkxLjUzLjE5Mw==.xls.xlsx---Analysis" in suc_sheets)
    # print('in fail_formulas', "012dcc10c41410112265d9556ec89bb1_d3d3LmFlci5nb3YuYXUJMTUyLjkxLjUzLjE5Mw==.xls.xlsx---Analysis" in fail_formulas)
    # print('len fail_formulas', len(fail_formulas))
    filelist = os.listdir(root_path + 'top10domain_model1_res')
    filelist.sort()
    file_sheet_num = {}
    
    found_true = 0
    found_false = 0
    template_found_true = 0
    template_found_false = 0
    count =0

    naive_found_true = 0
    naive_found_false = 0
    naive_template_found_true = 0
    naive_template_found_false = 0

    simple_found_true = 0
    simple_found_false = 0
    simple_template_found_true = 0
    simple_template_found_false = 0

    naive_true_found_false = 0

    download = set()

    sheetnames = set()

    count = 0

    shift_fail_template2num = {}
    for index,filename in enumerate(filelist):

        model1_found = False
        naive_found = False
        # print(index, len(filelist))
        fname = filename.split("---")[0]
        sheetname = filename.split("---")[1]
        fr = filename.split('---')[2]
        fc = filename.split('---')[3]
        filesheet = fname + '---' + sheetname

        with open(root_path + 'top10domain_model1_res/' + filename, 'r') as f:
            res = json.load(f)


        if res[4] == False:
            target_r1c1 = res[2]
            res_r1c1 = res[3]
            if target_r1c1 not in r1c12template_top10domain:
                target_templateid = -1
            else:
                target_templateid = r1c12template_top10domain[target_r1c1]
            if res_r1c1 not in r1c12template_top10domain:
                res_templateid = -1
            else:
                res_templateid = r1c12template_top10domain[res_r1c1]

            if res_r1c1 != target_r1c1 and target_templateid == res_templateid and res_templateid != -1:
                
                res_pair.append((res_r1c1, target_r1c1))
                if res_templateid not in all_target_set:
                    all_target_set.add(res_templateid)
                    print("##############")
                    print(res[2])
                    print(res[3])
                    d=difflib.Differ()
                    diff=d.compare(res_r1c1,target_r1c1)
                    print('diff', list(diff))
                count += 1
                if res_templateid not in shift_fail_template2num:
                    shift_fail_template2num[res_templateid] = 0
                shift_fail_template2num[res_templateid] += 1

    # with open("shift_fail_template2num_constant.json", 'w') as f:
        # json.dump(shift_fail_template2num, f)
    with open("shift_fail_r1c1_pairs.json", 'w') as f:
        json.dump(res_pair, f)
    print('count', count)

def analyze_differ():
    with open("shift_fail_differ_pairs.json", 'r') as f:
        res = json.load(f)

    rf_num = 0
    rf_str = 0
    sheet = 0
    all_ = 0

    shift_distance = []

    for listlist in res:
        res_list = listlist[0]
        target_list = listlist[1]
        for index,item in enumerate(res_list):
            print("########")
            print(item)
            print( target_list[index])
            if '[' == item[0] and ']' == item[-1]:
                res_content = item[1:-1]
                taret_content = target_list[index][1:-1]
                try:
                    res_num = int(res_content)
                    target_num = int(taret_content)
                    print('abs:', abs(res_num - target_num))
                    shift_distance.append(abs(res_num - target_num))
                    count += 1
                    rf_num += 1
                except:
                    rf_str += 1
            elif '!' == item[-1]:
                sheet += 1

            else:
                # print("####")
                print(item)
                print(target_list[index])

                res_split_temp = item.split("R")
                res_split_temp1 = res_split_temp[-1].split('C')

                if len(res_split_temp) > 1:
                    res_split_list = [res_split_temp[0]] + res_split_temp1
                else:
                    res_split_list = res_split_temp1
                print("res_split_list", res_split_list)
                target_split_temp = target_list[index].split("R")
                target_split_temp1 = target_split_temp[-1].split('C')
                if len(target_split_temp) > 1:
                    target_split_list = [target_split_temp[0]] + target_split_temp1
                else:
                    target_split_list = target_split_temp1
                print("target_split_list", target_split_list)

                for index1, item1 in enumerate(res_split_list):
                    if len(item1) == 0:
                        continue
                    res_num = int(item1)
                    target_num = int(target_split_list[index1])
                    if res_num == target_num:
                        continue
                    # print('Abs',abs(res_num - target_num))
                    shift_distance.append(abs(res_num - target_num))
                rf_num += 1
            all_+= 1
    print("rf_num", rf_num)
    print("rf_str", rf_str)
    print('sheet', sheet)
    print('all_', all_)
    print('shift_distance', shift_distance)
    print('len shift_distance', len(shift_distance))
    print('avg', np.array(shift_distance).mean())

    n2n = {}
    for item in shift_distance:
        if item not in n2n:
            n2n[item] = 0
        n2n[item] += 1

    n2n = sorted(n2n.items(), key=lambda x: x[0])

    new_n2n = {}
    for tuple in n2n:
        new_n2n[tuple[0]] = tuple[1]
    with open('rf_num_distance2num.json', 'w') as f:
        json.dump(new_n2n, f)

def deal_fortune500():
    with open('/datadrive/data_fortune500/crawled_index_fortune500.txt', 'r') as f:
        str_ = f.read()
    
    lines = str_.split('\n')
    res = {}

    domian2num = {}

    company2num = {}
    for line in lines:
        print("#####")
        print(line)
        if len(line.split('\t')) < 2:
            continue
        target_file = line.split('\t')[0]
        source_file = line.split('\t')[1]

        source_domain = '/'.join(source_file.split('/')[0:3])
        res[target_file] = source_domain
        print('source_domain', source_domain)
        company = source_domain.split('.')[-2]

        if source_domain not in domian2num:
            domian2num[source_domain] = 0
        if company not in company2num:
            company2num[company] = 0
        domian2num[source_domain] += 1
        company2num[company] += 1
    domian2num = sorted(domian2num.items(), key=lambda x: x[1], reverse=True)
    company2num = sorted(company2num.items(), key=lambda x: x[1], reverse=True)

    new_domain2num = {}
    for tuple in domian2num:
        new_domain2num[tuple[0]] = tuple[1]
    with open("fortune500_domain2num.json", 'w') as f:
        json.dump(new_domain2num, f)

    new_domain2num = {}
    for tuple in company2num:
        new_domain2num[tuple[0]] = tuple[1]
    with open("fortune500_company2num.json", 'w') as f:
        json.dump(new_domain2num, f)
        # break
    # with open("fortune500_file2domain.json", 'w') as f:
    #     json.dump(res, f)
    
def save_fortune500_files():
    files = []

    with open("fortune500_company2num.json", 'r') as f:
        fortune500_company2num = json.load(f)
    
    company_list = []
    for index,key in enumerate(fortune500_company2num):
        if index < 1 or index > 10:
            continue
        company_list.append(key)

    with open('/datadrive/data_fortune500/crawled_index_fortune500.txt', 'r') as f:
        str_ = f.read()
    
    lines = str_.split('\n')

    for line in lines:
        if len(line.split('\t')) < 2:
            continue
        target_file = line.split('\t')[0]
        source_file = line.split('\t')[1]
        
        source_domain = '/'.join(source_file.split('/')[0:3])
        company = source_domain.split('.')[-2]
        
        if company not in company_list:
            continue
        print(source_domain)
        files.append(target_file)

    print(len(files))
    with open("fortune500_filenames.json", 'w') as f:
        json.dump(files, f)
        
    
    

def move_xls_xlsx():
    filelist = os.listdir('/datadrive-2/data/middle10domain_test/model1_similar_sheet')
    for filename in filelist:
        if '.xls.xlsx' in filename and '.json' in filename:
            shutil.move('/datadrive-2/data/middle10domain_test/model1_similar_sheet/' + filename, '/datadrive-2/data/middle10domain_test/model1_similar_sheet/' + filename.replace('.xls.xlsx', '.xlsx'))

def get_ref():
    filelist = os.listdir(root_path + "afterfeature_test")
    with open('fortune500_formulatoken2r1c1.json','r') as f:
        fortune500_formulatoken2r1c1 = json.load(f)

    for formula_token_file in filelist:
        formula_token = formula_token_file.replace(".json", "")
        r1c1 = fortune500_formulatoken2r1c1[formula_token] 

def find_closed_tile():
    tile_files_list = os.listdir(root_path + 'tile_after_features/')
    index_dict = {}
    for filename_npy in tile_files_list:
        tile_token = filename_npy.replace('.npy', '')
        splited_list = tile_token.split('---')
        filesheet = splited_list[0] + '---' + splited_list[1]
        if filesheet not in index_dict:
            index_dict[filesheet] = []
        index_dict[filesheet].append(splited_list[2] + '---' + splited_list[3])

    formulas_list = os.listdir(root_path + 'model1_res')
    for index, formula_token_json in enumerate(formulas_list):
        print('formula_token_npy', formula_token_json)
        formula_token = formula_token_json.replace('.json','')
        origin_split_list = formula_token.split('---')
        origin_filesheet = origin_split_list[0] + '---' + origin_split_list[1]
        with open(root_path + 'model1_res/' + formula_token_json, 'r') as f:
            res = json.load(f)
        other_formula_token = res[1]
        print('formula_token', formula_token)
        print('other_formula_token', other_formula_token)
        other_split_list = other_formula_token.split('---')
        other_filesheet = other_split_list[0] + '---' + other_split_list[1]

        with open(root_path + 'test_refcell_position/' + formula_token_json, 'r') as f:
            refcell_position = json.load(f)
        for pos in refcell_position:
            row = pos['R']
            col = pos['C']
            refcell_feature = np.load(root_path + 'refcell_after_features/' + origin_filesheet + '---'+ str(row) + '---' + str(col) + '.npy' , allow_pickle=True)
            best_distance = np.inf
            best_position = ''
            print('# of tile', len(index_dict[filesheet]))
            print('filesheet', filesheet)
            for index1, position in enumerate(index_dict[filesheet]):
                # print(index1, len(index_dict[filesheet]))
                tile_token = other_filesheet + '---' + position
                tile_feature = np.load(root_path + 'tile_after_features/' + tile_token + '.npy', allow_pickle=True)
                distance = euclidean(refcell_feature, tile_feature)
                if distance < best_distance:
                    best_distance = distance
                    best_position = position
            np.save(root_path + 'best_tile/' + origin_filesheet + '---'+str(row) + '---' + str(col) + '.npy', {'best_position': best_position, 'best_distance': best_distance, 'other_file': other_filesheet})

        print('filesheet', filesheet)

def find_closed_tile():
    tile_files_list = os.listdir(root_path + 'tile_rows/')
    filesheet2tilenum = {}
    for filename_json in tile_files_list:
        with open(root_path + 'tile_rows/' + filename_json, 'r') as f:
            tile_rows = json.load(f)
            row_num = len(tile_rows)
        with open(root_path + 'tile_cols/' + filename_json, 'r') as f:
            tile_cols = json.load(f)
            col_num = len(tile_cols)
        # print('res', res)
        tile_token = filename_json.replace('.npy', '')
        splited_list = tile_token.split('---')
        # break
        filesheet = splited_list[0] + '---' + splited_list[1]
        # if filesheet not in index_dict:
        filesheet2tilenum[filesheet] = row_num*col_num
            
        # index_dict[filesheet].append(splited_list[2] + '---' + splited_list[3])
        
    # for filesheet in index_dict:
        # index_dict_num[filesheet] = len(index_dict[filesheet])

    filesheet2tilenum = sorted(filesheet2tilenum.items(), key=lambda x:x[1], reverse=True)

    new_res = {}
    for pair in filesheet2tilenum:
        new_res[pair[0]] = pair[1]
    with open('filesheet2tilenum.json', 'w') as f:
        json.dump(new_res, f)
#     refcell_files_list = os.listdir(root_path + 'refcell_after_features')

#     for index, filename_npy in enumerate(refcell_files_list):
#         print(index, len(refcell_files_list))
#         refcell_token = filename_npy.replace('.npy', '')
#         if os.path.exists(root_path + 'best_tile/' + refcell_token + '.npy'):
#             continue

#         splited_list = refcell_token.split('---')
#         filesheet = splited_list[0] + '---' + splited_list[1]
#         if filesheet not in index_dict:
#             continue
#         refcell_feature = np.load(root_path + 'refcell_after_features/' + filename_npy, allow_pickle=True)
#         best_distance = np.inf
#         best_position = ''
#         print('# of tile', len(index_dict[filesheet]))
#         print('filesheet', filesheet)
#         for index1, position in enumerate(index_dict[filesheet]):
#             # print(index1, len(index_dict[filesheet]))
#             tile_token = filesheet + '---' + position
#             tile_feature = np.load(root_path + 'tile_after_features/' + tile_token + '.npy', allow_pickle=True)
#             distance = euclidean(refcell_feature, tile_feature)
#             if distance < best_distance:
#                 best_distance = distance
#                 best_position = position
#         np.save(root_path + 'best_tile/' + refcell_token + '.npy', {'best_position': best_position, 'best_distance': best_distance })
        
def refcell_filesheets_num():
    refcell_files_list = os.listdir(root_path + 'refcell_after_features')
    filesheet2num = {}    

    for index, filename_npy in enumerate(refcell_files_list):
        refcell_token = filename_npy.replace('.npy', '')
        splited_list = refcell_token.split('---')
        filesheet = splited_list[0] + '---' + splited_list[1]
        if filesheet not in filesheet2num:
            filesheet2num[filesheet] = 0
        filesheet2num[filesheet] += 1
    # print(filesheet2num)
    with open('filesheet2num.json', 'w') as f:
        json.dump(filesheet2num, f)
        
def generate_second_tile():
    filelist = os.listdir(root_path + "/best_tile/")
    for index, filename_npy in enumerate(filelist):
        print(index, len(filelist))
        refcell_token = filename_npy.replace('.npy', '')
        best_file = np.load(root_path + "/best_tile/" + filename_npy, allow_pickle=True).item()
        min_dis = best_file['best_distance']
        best_position = best_file['best_position']

        cols = []
        rows = []
        start_row, start_col = best_position.split('---')
        start_row = int(start_row)
        start_col = int(start_col)
                
        for col in [0,2,4,6,8]:
            cols.append(start_col + col)
        for row in [0,20,40,60,80]:
            rows.append(start_row + row)

        with open(root_path + 'second_tile_position/' + refcell_token + '.json', 'w') as f:
            json.dump([cols, rows], f)

def clean_dedup_model1_res():
    with open(root_path + 'dedup_workbooks.json','r') as f:
        dedup_workbooks = json.load(f)
    filelist = os.listdir(root_path + 'dedup_model1_res')
    for filename in filelist:
        fname = filename.split('---')[0]
        if fname not in dedup_workbooks:
            os.remove(root_path + 'dedup_model1_res/' + filename)
if __name__ == "__main__":
    # move_xls_xlsx()
    # find_top10domain_closed_sheet(True,1,1)
    para_most_similar_sheet(is_save=True)
    # clean_multi_same()
    # para_multi_clean()
    # para_only_finetune_run()
    # para_finetune_run_eval()
    # para_run_template_eval()
    # para_run()

    # find_only_closed_formula(1,20)
    # step1
    # para_only_run()
    # find_only_closed_formula(1,1)
    # save_formula_dis()
    # step2
    # para_only_run_eval()
    # look_fail()
    # para_simple()
    # clean_dedup_model1_res()
    # analyze_similar_sheet_notfound()
    # generate_formula_token_2_r1c1()
    # find_top10domain_closed_sheet(False,1,1)
    # look_test_data()
    # sort_only_most_similar_formula(1,1,1,filepath = root_path + 'model1_formulas_dis', load_path=root_path+'model1_formulas_dis', save_path = root_path + 'model1_res')
    # look_sheetnum()
    # find_only_closed_formula(15,20)
    # para_dis_only_run()

    # para_run_eval()
    # count = 0
    # look_faile(1)
    # count_fail()
    # naive_suc_model2_fail()
    # compare_2_before_feature( \
    #     'bf47292856ad3b7c4c90c18fc62b4e18_d3d3Lm5pcHBvbnB1bHNlLmNvbQkxNjIuMjQyLjIxNS4yMA==.xls.xlsx---Smart---63---121', \
    #     '0841e16db277406ef66c64bf72ee8298_d3d3Lm5pcHBvbnB1bHNlLmNvbQkxNjIuMjQyLjIxNS4yMA==.xlsx---Smart---29---65')
    # count_same_template()
    # print(count)
    # deal_00()
    # para_deal00()
    # find_closed_formula(1,20)
    # sort_most_similar_formula(1)
    # find_closed_formula(20,20)
    # look_res_len()
    # find_closed_formula(1,20)
# find_closed_sheet(need_save=True)
    # find_l2_closed_sheet(need_save = True)
# add_formula_id()
    # naive_find_formula(1,1)
    # para_naive()
    # find_closed_tile()
    # generate_second_tile()
    # refcell_filesheets_num()
# save_not_found_true_false()
# print_one_most_similar("c69ac3334272c4215c9d61f30255a402_ZHNocy50ZXhhcy5nb3YJNTIuMjYuOTguMjM0.xlsx---Sheet1")
# print_one_filesheet_r1c1("../../data/UnzipData/000/023a1e3935ae240d74a6ca050622f654_a2V0c28uY29tCTIxMy4xMzguMTEzLjIxMg==.xls.xlsx---Workshop_Outputs","IF(COUNTA(RC[-8]:RC[-2])>0,IF(COUNTA(RC[-12]:RC[-11])<>2,1,\"\"),\"\")")
# print_one_filesheet_r1c1("../../data/UnzipData/006/2c0e90a75d82f925136504f10a0d4443_a2V0c28uY29tCTIxMy4xMzguMTEzLjIxMg==.xls.xlsx---Workshop_Outputs","IF(COUNTA(RC[-8]:RC[-2])>0,IF(COUNTA(RC[-12]:RC[-11])<>2,1,\"\"),\"\")")


# opt_naive_not_found()
# look_same_sheet_name()
    # analyze_nodelist_length()
    # look_how_shift()
    # analyze_differ()
    # deal_fortune500()
    # save_fortune500_files()
    # r1c12templateid()

    # generate_formula_features()