import json
import os
import numpy as np
import pprint


######eidt distance baseline#####
def edit_distance(str1, str2):
    edit = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            edit[i][j] = min(edit[i - 1][j] + 1, edit[i][j - 1] + 1, edit[i - 1][j - 1] + d)

    return edit[len(str1)][len(str2)]

def find_closed_sheet_by_editdistance():
    with open('Formula_160000.json','r') as f:
        formulas = json.load(f)
    res = {}
    count = 0
    sheetname_distance = {}

    for formula in formulas:
        count += 1
        print(count, len(formulas))
        filesheet = formula['filesheet'].split('/')[5]
        if filesheet in sheetname_distance:
            continue
        sheetname_distance[filesheet] = []
        add_sheets = []
        for other_formula in formulas:
            other_filesheet = other_formula['filesheet'].split('/')[5]
            if other_filesheet in add_sheets:
                continue
            add_sheets.append(other_filesheet)
            sheetname_distance[filesheet].append((other_filesheet, edit_distance(filesheet.split('---')[1], other_filesheet.split('---')[1])))
        sheetname_distance[filesheet] = sorted(sheetname_distance[filesheet], key = lambda x: x[1])[0:20]
    with open('edit_distance_sheet_similarity.json','w') as f:
        json.dump(sheetname_distance, f)


def edit_distance_baseline_find_formula():
    with open('Formula_77772_with_id.json','r') as f:
        formulas = json.load(f)
    with open('edit_distance_sheet_similarity.json','r') as f:
        sheet_distance = json.load(f)

    found_false = []
    not_found = []
    found_true = []

    count = 0
    for formula in formulas:
        count += 1
        print(count, len(formulas))
        filesheet = formula['filesheet'].split('/')[5]
        similar_sheets = [sheet_pair[0] for sheet_pair in sheet_distance[filesheet]]
        found = False
        for similar_sheet in similar_sheets:
            if filesheet == similar_sheet:
                continue
            for other_formula in formulas:
                other_filesheet = other_formula['filesheet'].split('/')[5]
                # print('other_filesheet', other_filesheet)
                # print('similar_sheet', similar_sheet)
                if other_filesheet != similar_sheet:
                    continue
                # print('xxxx')
                if formula['fr'] >= other_formula['fr'] and formula['fc'] >= other_formula['fc'] and formula['lr'] <= other_formula['lr'] and formula['lc'] <= other_formula['lc'] :
                    found = True
                    if other_formula['r1c1'] == formula['r1c1']:
                        found_true.append(formula['id'])
                        print('found_true')
                    else:
                        found_false.append(formula['id'])
                        print('found_false')
                    break
            if found:
                
                break
        if not found:
            not_found.append(formula['id'])
            print('not_found')
    print('len found_true', len(found_true))
    print('len found_false', len(found_false))
    print('len not_found', len(not_found))
    with open("edit_distance_baseline_found_true_1900.json", 'w') as f:
        json.dump(found_true, f)
    with open("edit_distance_baseline_found_false_1900.json", 'w') as f:
        json.dump(found_false, f)
    with open("edit_distance_baseline_not_found_1900.json", 'w') as f:
        json.dump(not_found, f)

#######most popular baseline#########
def get_most_popular_r1c1(topk=100):
    with open('Formula_77772_with_id.json','r') as f:
        formulas = json.load(f)

    r1c12num = {}
    for formula in formulas:
        r1c1 = formula['r1c1']
        if formula['r1c1'] not in r1c12num:
            r1c12num[r1c1] = 0
        r1c12num[r1c1] += 1

    res = [i[0] for i in sorted(r1c12num.items(), key=lambda x:x[1], reverse=True)]
    return res[0:topk]

def baseline_most_popular_r1c1(topk=100):
    topk_r1c1 = get_most_popular_r1c1(topk)
    with open('Formula_77772_with_id.json','r') as f:
        formulas = json.load(f)
    found_true = 0
    found_false = 0
    for formula in formulas:
        if formula['r1c1'] in topk_r1c1:
            found_true += 1
        else:
            found_false += 1

    print('found_true', found_true)
    print('found_false', found_false)

if __name__ == '__main__':
    ### edit distance
    # find_closed_sheet_by_editdistance()
    # edit_distance_baseline_find_formula()

    ### most popular
    # get_most_popular_r1c1()
    baseline_most_popular_r1c1()
    
