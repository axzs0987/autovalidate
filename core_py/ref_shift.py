import os
import json
import re
import random
import copy
import time
version = 0
if version == 0:
    root_path = '/datadrive-2/data/fortune500_test/'
elif version == 1:
    root_path = '/datadrive-2/data/top10domain_test/'

def generate_ref_position(formula_token, r1c1):
    if os.path.exists(root_path + 'formula_template/' + formula_token + '.json'):
        with open(root_path + 'formula_template/' + formula_token + '.json', 'r') as f:
            node_list = json.load(f)
    else:
        return
    res = []
    target_row = int(formula_token.split('---')[2])
    target_col = int(formula_token.split('---')[3])
    now_nametoken = ''

    new_item = {}
    print('r1c1', r1c1)
    # print( node_list)
    is_end = False
    is_rc_token = False
    is_vrange = False
    for index,node in enumerate(node_list):
        print('node', node)
        if 'R' in new_item and 'C' in new_item:
            res.append(new_item)
            new_item = {}
        elif is_rc_token and ( not is_vrange and (node['Term'] != 'FileNameEnclosedInBracketsToken' and node['Term'] != 'StructuredReferenceElement') and node['Term'] != 'Argument'):
            if 'R' in new_item:
                new_item['C'] = target_col
                new_item['C_dollar'] = False
            elif 'C' in new_item:
                new_item['R'] = target_row
                new_item['R_dollar'] = False
            print('append new_item 2', new_item)
            res.append(new_item)
            new_item = {}
            is_rc_token = False
            is_vrange = False

        if node['Term'] == 'CellToken':
            print("node['Token']", node['Token'])
            if not re.match(r'R\d+C\d+', node['Token']) and not re.match(r'R\d+', node['Token']) and not re.match(r'C\d+', node['Token']):
                # print('r1c1', r1c1)
                print('token', node['Token'])
                continue
            row_num = ''
            if len(node['Token'].split('R')) > 1:
                row_num = node['Token'].split('R')[1].split('C')[0]
            if row_num != '':
                new_item['R'] = int(row_num)
                new_item['R_dollar'] = True
            col_num = ''

            if len(node['Token'].split('C')) > 1:
                col_num = node['Token'].split("C")[1]
            print('col_num', col_num)
            if col_num != '':
                new_item['C'] = int(col_num)
                new_item['C_dollar'] = True

        if node['Term'] == 'VRangeToken':
            if node['Token'] == 'C:R':
                new_item['C'] = target_col
                new_item['C_dollar'] = False
                res.append(new_item)
                new_item = {}
                now_nametoken = "R"
                is_rc_token = True
                is_vrange = True
                
                

        if node['Term'] == "NamedRangeCombinationToken":
            if not re.match(r'R\d+C\d+', node['Token']):
                # print('r1c1', r1c1)
                print('token NamedRangeCombinationToken', node['Token'])
                continue
                
            new_item = {}
            r_num = int(node['Token'].split("R")[-1].split('C')[0])
            c_num = int(node['Token'].split('C')[1])
            new_item['R'] = r_num
            new_item['C'] = c_num
            new_item['R_dollar'] = True
            new_item['C_dollar'] = True
            if index == len(node_list)-1:
                is_end = True
            # print('1:new_item', new_item)
        # if node['Term'] == 'VRangeToken':
        #     two_token = node['Token'].split(':')
        #     if two_token[0]
        if node['Term'] == 'NameToken':
            now_nametoken = node['Token']
            print('2:now_nametoken', now_nametoken)
            print('2:new_item', new_item)
            if now_nametoken == 'R' or now_nametoken == 'C':
                is_rc_token = True
            if index == len(node_list)-1:
                is_end = True
                if now_nametoken == 'R':
                    new_item['R_dollar'] = False
                if now_nametoken == 'C':
                    new_item['C_dollar'] = False
        if node['Term'] == "FileNameEnclosedInBracketsToken":
            try:
                num = int(node['Token'][1:-1])
            except:
                # print("####")
                # print(node['Token'])
                # print(r1c1)
                continue
            now_key = now_nametoken
            if now_key == 'RC':
                new_item['R'] = target_row
                new_item['R_dollar'] = False
                now_key = 'C'
                new_item['C_dollar'] = False
            if now_key == 'R':
                num += target_row
                new_item['R_dollar'] = False
            if now_key == 'C':
                num += target_col
                new_item['C_dollar'] = False
            
            new_item[now_key] = num
            now_nametoken = ''
            if index == len(node_list)-1:
                is_end = True
            print('3:new_item', new_item)
            is_rc_token = False
            is_vrange = False
    print('is_rc_token', is_rc_token)
    print('is_end', is_end)
    if is_end:
        if 'R' in new_item and 'C' in new_item:
            res.append(new_item)
            new_item = {}
        elif is_rc_token:
            if 'R' in new_item:
                new_item['C'] = target_col
            elif 'C' in new_item:
                new_item['R'] = target_row
            print('append new_item 1', new_item)
            res.append(new_item)
            new_item = {}
            is_rc_token = False
            is_vrange = False
    # if now_nametoken == 'R':
    #     new_item[now_nametoken] = taret_row 
    # if now_nametoken == 'C':
    #     new_item[now_nametoken] = target_col 
    # if 'R' in new_item and 'C' in new_item:
    #     res.append(new_item)
    #     new_item = {}
    invalid = False
    for new_item in res:
        if 'R' not in new_item or 'C' not in new_item:
            invalid = True
            break

    # if invalid:
    #     count += 1
    #     if count <= 2:
    #         continue
    #     print('res', res)
    #     # if len(res) == 0:
    #     print(formula['r1c1'])
    #     res0_list[formula['r1c1']] = res
    #     print('xxxx')
    #     break
    print('res', res)
    with open(root_path + 'test_refcell_position/'+formula_token + '.json', 'w') as f:
        json.dump(res, f)

def generate_training_refcell():
    with open('fine_tune_positive.json', 'r') as f:
        fine_tune_positive = json.load(f)

    start_time = time.time()
    all_count = 0
    for index, pair in enumerate(fine_tune_positive):
        auchor_formula_token, auchor_r1c1 = pair[0]
        # positive_formula_token, positive_r1c1 = pair[1]
        auchor_formula_token = auchor_formula_token.split('/')[-1]
        # positive_formula_token = positive_formula_token.split('/')[-1]
        # if positive_formula_token != '1b8ba579545e8c428178e6a3d5c37dc5_c3RyZWFtLW1lY2hhbmljcy5jb20JNTAuNjIuMTcyLjExMw==.xlsx---Monitoring Data---3---3':
        #     continue
        generate_ref_position(auchor_formula_token, auchor_r1c1)
        all_count += 1
        if all_count == 50:
            break
        # generate_ref_position(positive_formula_token, positive_r1c1)
        # break
    end_time = time.time()
    print('all_time:', end_time - start_time)
    print('avg_time:', (end_time - start_time) / all_count)
        
        
def genearte_training_triples(generate_positive):
    if generate_positive:
        with open('fine_tune_positive.json', 'r') as f:
            fine_tune_positive = json.load(f)
        positive_pair = []
        for index, pair in enumerate(fine_tune_positive):
            # print(index, len(fine_tune_positive))
            auchor_formula_token, auchor_r1c1 = pair[0]
            positive_formula_token, positive_r1c1 = pair[1]
            auchor_formula_token = auchor_formula_token.split('/')[-1]
            positive_formula_token = positive_formula_token.split('/')[-1]
            auchor_filesheet = auchor_formula_token.split('---')[0] + '---' + auchor_formula_token.split('---')[1]
            positive_filesheet = positive_formula_token.split('---')[0] + '---' + positive_formula_token.split('---')[1]
            # if(auchor_formula_token != '011ade4d28969ac6e1e8d045f28dbe64_bGV0aXBwaW5nLmNvbS5hdQkyMDIuNzIuMTg0LjI3.xls.xlsx---All Tipsters---27---47'):
            #     continue
            # if auchor_filesheet != '011ade4d28969ac6e1e8d045f28dbe64_bGV0aXBwaW5nLmNvbS5hdQkyMDIuNzIuMTg0LjI3.xls.xlsx---All Tipsters':
            #     continue
            if not os.path.exists(root_path+'test_refcell_position/' + auchor_formula_token + '.json') or not os.path.exists(root_path+'test_refcell_position/' + positive_formula_token + '.json'):
                continue
            with open(root_path+'test_refcell_position/' + auchor_formula_token + '.json', 'r') as f:
                res = json.load(f)
            with open(root_path+'test_refcell_position/' + positive_formula_token + '.json', 'r') as f:
                res1 = json.load(f)
            print("######")
            print(res)
            print(auchor_r1c1)
            print(auchor_formula_token)
            print(res1)
            print(positive_r1c1)
            print(positive_formula_token)
            if len(res) != len(res1):
                continue
            for index1, position in enumerate(res):
                pos_pair = []
                if not 'R' in position or not 'C' in position:
                    continue
                pos_pair.append(auchor_filesheet + '---' + str(position['R']) + '---' + str(position['C']))
                pos_pair.append(positive_filesheet + '---' + str(res1[index1]['R']) + '---' + str(res1[index1]['C']))
                positive_pair.append(pos_pair)
            # break
        
        dedup_positive_pair = []
        for pair in positive_pair:
            exist = False
            for ex_pair in dedup_positive_pair:
                if pair[0] == ex_pair[0] or pair[1] == ex_pair[1]:
                    exist = True
                    break
            if not exist:
                dedup_positive_pair.append(pair)
        with open('dedup_positive_pair.json', 'w') as f:
            json.dump(dedup_positive_pair, f)
    else:
        with open('dedup_positive_pair.json', 'r') as f:
            dedup_positive_pair = json.load(f)
        range_list = [5,10,15,20]
        new_list = []
        for index, pair in enumerate(dedup_positive_pair):
            print(index, len(dedup_positive_pair))
            positive_formula_token = pair[1]
            positive_fr = int(positive_formula_token.split('---')[2])
            positive_fc = int(positive_formula_token.split('---')[3])
            
            for shift_range in range_list:
                one_pair = copy.copy(pair)
                negative_formula_token = one_pair[1].split('---')[0] + '---' + one_pair[1].split('---')[1]
                negative_fr = random.choice(list(range(max(positive_fr-shift_range, 1), positive_fr + shift_range)))
                negative_fc = random.choice(list(range(max(positive_fc-shift_range, 1), positive_fc + shift_range)))
                negative_formula_token = negative_formula_token + '---' + str(negative_fr) + '---' + str(negative_fc)
                one_pair.append(negative_formula_token)
                new_list.append(one_pair)
                one_pair = copy.copy(pair)
                negative_formula_token = one_pair[1].split('---')[0] + '---' + one_pair[1].split('---')[1]
                negative_fr = random.choice(list(range(max(positive_fr-shift_range, 1), positive_fr + shift_range)))
                negative_fc = random.choice(list(range(max(positive_fc-shift_range, 1), positive_fc + shift_range)))
                negative_formula_token = negative_formula_token + '---' + str(negative_fr) + '---' + str(negative_fc)
                one_pair.append(negative_formula_token)
                new_list.append(one_pair)
        with open('dedup_training_triples.json', 'w') as f:
            json.dump(new_list, f)
def generate_r1c1_2_refcell():
    filelist = os.listdir(root_path + 'formula_template')
    if version == 0:
        with open('fortune500_formulatoken2r1c1.json', 'r') as f:
            fortune500_formulatoken2r1c1 = json.load(f)
        with open("Formulas_fortune500_with_id.json",'r') as f:
            formulas = json.load(f)
    elif version == 1:
        with open('top10domain_formulatoken2r1c1.json', 'r') as f:
            fortune500_formulatoken2r1c1 = json.load(f)
        with open("Formulas_top10domain_with_id.json",'r') as f:
            formulas = json.load(f)

    count = 0

    res0_list = {}

    count = 0
    for index, formula in enumerate(formulas):
        formula_token_file = formula['filesheet'].split('/')[-1] + '---' + str(formula['fr']) + '---' + str(formula['fc']) + '.json'
        # if formula_token_file != "124581666438778705534848216950560361712-area-surcharge-zips-us-preview.xlsx---US 48 Zip---224---2.json":
            # continue
        # if formula_token_file != "108138953513719859234385013398650948303-obf-lighting-workbook-032022.xlsx---2. UTILITY INFO---14---5.json":
            # continue
        print(index, len(formulas))
        # print('formula_token_file', formula_token_file)
        # print('r1c1', formula['r1c1'])
        if os.path.exists(root_path + 'test_refcell_position/'+formula_token_file):
            continue
        if not os.path.exists(root_path + 'formula_template/' + formula_token_file):
            continue
        # if formula['r1c1'] != "(MEDIAN(R[-13]C:R[-6]C))":
        # if formula['r1c1'] != "SUM(RC[-8]:RC[-1])*R45C4":
        # if formula['r1c1'] != "BIN2HEX(CONCATENATE(DEC2BIN(VLOOKUP(RC[3],Lists!R[-8]C[14]:R[-7]C[15],2,0)),DEC2BIN(VLOOKUP(R[1]C[3],Lists!R[-25]C:R[-18]C[1],2,0),3),DEC2BIN(VLOOKUP(R[2]C[3],Lists!R[-25]C[2]:R[-22]C[3],2,0),2),DEC2BIN(VLOOKUP(R[3]C[3],Lists!R[-25]C[4]:R[-22]C[5],2,0),2)),2)":
        #     continue
        count += 1
        print('formula_token_file', formula_token_file)
        

        formula_token = formula_token_file.replace('.json','')
        
        r1c1 = fortune500_formulatoken2r1c1[formula_token]
        generate_ref_position(formula_token, r1c1)
        
        # break
    # with open('res0_list.json', 'w') as f:
    #     json.dump(res0_list, f)
    print('count', count)

def generate_training_ref():

    formula_pairs = []
    with open("fine_tune_positive.txt", 'r') as f:
        positive_txt = f.read()
        res_lines = positive_txt.split('\n')
        for line in res_lines:
            split_list = line.split('   ')
            if len(split_list) != 8:
                continue
            auchor_filesheet, positive_filesheet, auchor_r1c1, positive_r1c1, auchor_row, auchor_col, positive_row, positive_col = split_list
            pair = []
            pair.append([auchor_filesheet + '---' + auchor_row + '---' + auchor_col, auchor_r1c1])
            pair.append([positive_filesheet + '---' + positive_row + '---' + positive_col, positive_r1c1])
            formula_pairs.append(pair)
    with open("fine_tune_positive.json", 'w') as f:
        json.dump(formula_pairs, f)
generate_r1c1_2_refcell()
# generate_training_ref()
# generate_training_refcell()
# genearte_training_triples(generate_positive=False)
# generate_ref_position("71830531429966026667023738346702335760-lm5160_2d00_ver1.xlsx---Parts---4---63", "(RC[-52]*R24C58)+(RC[-46]*R23C58)+(RC[-44]*R[424]C58)+(RC[-39]*R24C58)+(RC[-32]*R23C58)+(RC[-31]*R23C58)+(RC[-15]*R23C58)+(RC[-12]*R24C58)+(RC[-11]*R23C58)+(RC[-10]*R27C58)+(RC[-9]*R27C58)+(RC[-8]*R22C58)+(RC[-7]*R25C58)+(RC[-17]*R23C58)+(RC[-18]*R24C58)+R[18]C[-2]")