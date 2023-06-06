using System;
using ClosedXML.Excel;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using XLParser;

namespace AnalyzeDV
{
    class CellFeature
    {
        public byte background_color_r;
        public byte background_color_g;
        public byte background_color_b;
        public byte font_color_r;
        public byte font_color_g;
        public byte font_color_b;
        public double font_size;
        public bool font_strikethrough;
        public bool font_shadow;
        public bool font_ita;
        public bool font_bold;
        public double height;
        public double width;
        public string content;
        public string content_template;

    }
    class SameTempFiles{
        public List<string> sheetnames;
        public List<string> filenames;
    }
    class SheetFeature{
        public string filename;
        public string sheetname;
        public List<CellFeature> sheetfeature;
    }
    class SheetSim{
        public void saveAsJson(object need_save_content, string file_name){
            string jsonData = JsonConvert.SerializeObject(need_save_content);
            File.WriteAllText(file_name, jsonData);
        }
        
        public bool is_same(IXLCell cell1, IXLCell cell2, int metric){// 0: content; 1: font-color; 2: fill-color; 3: height; 4: width; 5: data-type;
            if(metric == 0){
                // var a = cell1.Value.ToString();
                // var b = cell2.Value.ToString();
                if(cell1.Value.ToString()==cell2.Value.ToString()){
                    return true;
                }
                else{
                    return false;
                }
            }else if(metric == 1){
                if(cell1.Style.Font.FontColor.GetHashCode() == cell2.Style.Font.FontColor.GetHashCode()){
                    return true;
                }else{
                    return false;
                }
            }else if(metric==2){
                if(cell1.Style.Fill.BackgroundColor.GetHashCode() == cell2.Style.Fill.BackgroundColor.GetHashCode()){
                    return true;
                }else{
                    return false;
                }
            }else if(metric==3){
                if(cell1.WorksheetRow().Height == cell2.WorksheetRow().Height){
                    return true;
                }else{
                    return false;
                }
            }else if(metric==4){
                if(cell1.WorksheetColumn().Width == cell2.WorksheetColumn().Width){
                    return true;
                }else{
                    return false;
                }
            }else if(metric==5){
                if(cell1.DataType == cell2.DataType){
                    return true;
                }else{
                    return false;
                }
            }
            Console.WriteLine("invalid metric");
            return false;
        }

        public bool is_all_blank(IXLCell cell1, IXLCell cell2){
            if(cell1.IsEmpty() && cell2.IsEmpty()){
                return true;
            }
            return false;
        }

        public string get_cell_content_template(IXLCell cell){
            string result = "";
            if(cell.DataType == XLDataType.Text){
                foreach(var cha in cell.Value.ToString()){
                    result += 'S';
                }
            }else if(cell.DataType == XLDataType.Number){
                foreach(var cha in cell.Value.ToString()){
                    result += 'N';
                }
            }else if(cell.DataType == XLDataType.Boolean){
                foreach(var cha in cell.Value.ToString()){
                    result += 'B';
                }
            }else if(cell.DataType == XLDataType.DateTime){
                foreach(var cha in cell.Value.ToString()){
                    result += 'D';
                }
            }else if(cell.DataType == XLDataType.TimeSpan){
                foreach(var cha in cell.Value.ToString()){
                    result += 'T';
                }
            }
            return result;
        }
        public CellFeature get_cell_features(IXLCell cell){
        // public int background_color;
        // public int font_color;
        // public double font_size;
        // public bool font_strikethrough;
        // public bool font_shadow;
        // public bool font_ita;
        // public bool font_bold;
        // public double height;
        // public double width;
        // public string content;
        // public string content_template;
            CellFeature cell_feature = new CellFeature();
            // cell_feature.background_color = cell.Style.Fill.BackgroundColor.GetHashCode();
            // cell_feature.font_color = cell.Style.Font.FontColor.GetHashCode();
            cell_feature.font_size = cell.Style.Font.FontSize;
            cell_feature.font_ita = cell.Style.Font.Italic;
            cell_feature.font_bold = cell.Style.Font.Bold;
            cell_feature.font_strikethrough = cell.Style.Font.Strikethrough;
            cell_feature.font_shadow = cell.Style.Font.Shadow;
            cell_feature.height = cell.WorksheetRow().Height; 
            cell_feature.width = cell.WorksheetColumn().Width; 
            cell_feature.content = cell.Value.ToString();
            cell_feature.content_template = get_cell_content_template(cell);
            return cell_feature;
        }

        public List<double> get_simularity_of_sheet(IXLWorksheet sheet1, IXLWorksheet sheet2){
            int content_hit = 0;
            int font_hit = 0;
            int type_hit = 0;
            int fill_hit = 0;
            int height_hit = 0;
            int width_hit = 0;
            int all_ = 0;
            int both_not_exist_number = 0;
            int both_blank_number = 0;
            List<double> result = new List<double>();
            // Console.WriteLine("&&&&&&&&&&&&&&&&&&&&&&&&&&&");
            for(int row=1;row<=100;row+=1){
                for(int column=1;column<=10;column+=1){
                    all_ += 1;
                    bool exist1 = true;
                    bool exist2 = true;
                    // Console.Write("load_cell1");
                    IXLCell cell1 = sheet1.Cell(1,1);
                    // Console.WriteLine("load_cell2");
                    IXLCell cell2 = sheet2.Cell(1,1);
                    // Console.WriteLine("row:"+row.ToString()+", column:"+column.ToString());
                    try{
                        cell1 = sheet1.Cell(row,column);
                        var value1 = cell1.Value.ToString();
                    }catch{
                        exist1 = false;
                    }
                    try{
                        cell2 = sheet2.Cell(row,column);
                        var value2 = cell2.Value.ToString();
                    }catch{
                        exist2 = false;
                    }
                    // Console.WriteLine("load_cell end");
                    // Console.WriteLine("cell1:"+exist1.ToString()+", cell1:"+exist2.ToString());
                    if(exist1 && exist2){
                        bool both_blank = is_all_blank(cell1, cell2);
                        if(both_blank){
                            both_blank_number += 1;
                        }else{
                            // Console.WriteLine("cell content");
                            if(is_same(sheet1.Cell(row,column), sheet2.Cell(row, column), 0)){
                                content_hit += 1;
                                
                            }
                            // Console.WriteLine("cell font");
                            if(is_same(sheet1.Cell(row,column), sheet2.Cell(row, column), 1)){
                                font_hit += 1;
                                
                            }
                            // Console.WriteLine("cell fill");
                            if(is_same(sheet1.Cell(row,column), sheet2.Cell(row, column), 2)){
                                fill_hit += 1;
                                
                            }
                            // Console.WriteLine("cell height");
                            if(is_same(sheet1.Cell(row,column), sheet2.Cell(row, column), 3)){
                                height_hit += 1;
                                
                            }
                            // Console.WriteLine("cell weight");
                            if(is_same(sheet1.Cell(row,column), sheet2.Cell(row, column), 4)){
                                width_hit += 1;
                                
                            }
                            // Console.WriteLine("cell type");
                            if(is_same(sheet1.Cell(row,column), sheet2.Cell(row, column), 5)){
                                type_hit += 1;
                                
                            }
                        }
                    }else if(exist1 || exist2){
                        continue;
                    }else{
                        both_not_exist_number+=1;
                    }
                }
            }
            result.Add((double)content_hit/(all_-both_blank_number-both_not_exist_number));
            result.Add((double)font_hit/(all_-both_blank_number-both_not_exist_number));
            result.Add((double)fill_hit/(all_-both_blank_number-both_not_exist_number));
            result.Add((double)height_hit/(all_-both_blank_number-both_not_exist_number));
            result.Add((double)width_hit/(all_-both_blank_number-both_not_exist_number));
            result.Add((double)type_hit/(all_-both_blank_number-both_not_exist_number));
            return result;
        }

        public void get_same_sheet_in_metric(){
            string jsonstring = File.ReadAllText("data/types/custom/change_xml_all_templates.json");
            var templates = JsonConvert.DeserializeObject<List<Template>>(jsonstring);

            List<Template> new_temp;
            if(File.Exists("data/types/custom/change_xml_all_template_with_simularity.json")){
                string jsonstring1 = File.ReadAllText("data/types/custom/change_xml_all_template_with_simularity.json");
                new_temp = JsonConvert.DeserializeObject<List<Template>>(jsonstring1);
            }else{
                new_temp = new List<Template>();
            }
            List<Dictionary<string, List<int>>> result;
            if(File.Exists("data/types/custom/change_xml_template_splited_by_style.json")){
                string jsonstring2 = File.ReadAllText("data/types/custom/change_xml_template_splited_by_style.json");
                result = JsonConvert.DeserializeObject<List<Dictionary<string, List<int>>>>(jsonstring2);
            }
            else{
                result = new List<Dictionary<string, List<int>>>();
            }
            
            int count = 0;
            foreach(var template in templates){
                template.value_similarity_list = new List<Similarity>();
                template.type_similarity_list = new List<Similarity>();
                template.fill_color_similarity_list = new List<Similarity>();
                template.font_color_similarity_list = new List<Similarity>();
                template.height_similarity_list = new List<Similarity>();
                template.width_similarity_list = new List<Similarity>();
                count+=1;
                
                // if(template.id!=303){
                //     continue;
                // }
                bool is_saved = false;
                Console.WriteLine("Count:"+count.ToString()+"/"+templates.Count().ToString());
                foreach(var saved_template in new_temp){
                    if(saved_template.id == template.id){
                        is_saved = true;
                        break;
                    }
                }
                if(is_saved){
                    continue;
                }
                if(template.id==3){
                    new_temp.Add(template);
                    continue;
                }
                if(count==31 || count==176){
                    new_temp.Add(template);
                    continue;
                }
                // List<string> distinct_file_list = new List<string>();
                // foreach(string file in template.file_sheet_name_list){
                //     bool is_in=false;
                //     foreach(string add_file in distinct_file_list){
                //         if(file==add_file){
                //             is_in=true;
                //             break;
                //         }
                //     }
                //     if(!is_in){
                //         distinct_file_list.Add(file);
                //     }
                // }
                // if(distinct_file_list.Count()==1){
                //     new_temp.Add(template);
                //     continue;
                // }
                // Console.WriteLine("Count:"+count.ToString()+"/"+templates.Count().ToString());
                Dictionary<string, List<int>> temp_dic = new Dictionary<string, List<int>>();
                int cluster = 0;
                
                // var worksheets = new List<IXLWorksheet>{};
                Dictionary<string, IXLWorksheet> worksheets = new Dictionary<string, IXLWorksheet>();
                Console.WriteLine("load sheets");
                foreach(var file_sheet in template.file_sheet_name_list){
                    if(worksheets.ContainsKey(file_sheet)){
                        continue;
                    }
                    string file1 = file_sheet.Split("-------------")[0];
                    string sheet1 = file_sheet.Split("-------------")[1];
                    var worksheet = new XLWorkbook(file1).Worksheet(sheet1);
                    worksheets.Add(file_sheet, worksheet);
                }
                Console.WriteLine("load end");
                for(int index1=0;index1<template.dvid_list.Count();index1++){
                    List<string> add_file = new List<string>();
                    string found_key = "";
                    bool index1_is_in = false;
                    
                    foreach(var one_res in temp_dic.Keys){
                        foreach(var one_index in temp_dic[one_res]){
                            if(one_index == index1){
                                index1_is_in = true;
                                found_key = one_res;
                                break;
                            }
                        }
                    }
                    // Console.WriteLine("index1:"+index1.ToString()+" " + index1_is_in);
                    // Console.WriteLine("found_key:"+index1.ToString()+" " + found_key);
                    if(!index1_is_in){
                        string new_key = template.id.ToString() + "---"+cluster.ToString();
                        List<int> new_list = new List<int>();
                        new_list.Add(index1);
                        temp_dic.Add(new_key, new_list);
                        // Console.WriteLine("temp_dic add "+index1.ToString() + " " + new_key);
                        found_key = new_key;
                        cluster += 1;
                    }else{
                        // Console.WriteLine("index1:"+index1.ToString()+" continue");
                        continue;
                    }
                    // Console.WriteLine("load workbook1");
                    string file1 = template.file_sheet_name_list[index1].Split("-------------")[0];
                    string sheet1 = template.file_sheet_name_list[index1].Split("-------------")[1];
                    // using(XLWorkbook workbook = new XLWorkbook(file1)){
                    // Console.WriteLine("load worksheet1");
                    // var worksheet1 = workbook.Worksheet(sheet1);
                    var worksheet1 = worksheets[template.file_sheet_name_list[index1]];
                    for(int index2=index1;index2<template.dvid_list.Count();index2++){
                        if(index1 == index2){
                            continue;
                        }
                        bool is_in_add = false;
                        foreach(string filename in add_file){
                            if(filename==template.file_sheet_name_list[index2]){
                                is_in_add=true;
                                break;
                            }
                        }
                        if(is_in_add){
                            temp_dic[found_key].Add(index2);
                            // Console.WriteLine("is_in_add temp_dic add "+index1.ToString()+"   "+index2.ToString());
                            // Console.WriteLine("index"+index2.ToString()+" is add");
                            continue;
                        }
                        add_file.Add(template.file_sheet_name_list[index2]);
                        
                        string file2 = template.file_sheet_name_list[index2].Split("-------------")[0];
                        string sheet2 = template.file_sheet_name_list[index2].Split("-------------")[1];
                        
                        
                        if(template.file_sheet_name_list[index1] ==  template.file_sheet_name_list[index2]){
                            // Console.WriteLine("file same temp_dic add "+index1.ToString()+"   "+index2.ToString());
                            temp_dic[found_key].Add(index2);
                            // Console.WriteLine("index"+index2.ToString()+" is same as index1");
                            continue;
                        }
                        Console.WriteLine("index1:"+index1.ToString()+"/"+template.dvid_list.Count().ToString()+", index2:"+index2.ToString());
                        // Console.WriteLine("load worksheet2");
                        // var worksheet2 = new XLWorkbook(file2).Worksheet(sheet2);
                        var worksheet2 = worksheets[template.file_sheet_name_list[index2]];
                        // Console.WriteLine("load finish");
                        bool is_same_style = true;

                        // Console.WriteLine("count_simu");
                        List<double> score = get_simularity_of_sheet(worksheet1, worksheet2);
                        // Console.WriteLine("count_end");
                        Similarity value_sim = new Similarity();
                        value_sim.score = score[0];
                        value_sim.index1 = index1;
                        value_sim.index2 = index2;

                        Similarity font_sim = new Similarity();
                        font_sim.score = score[1];
                        font_sim.index1 = index1;
                        font_sim.index2 = index2;
                        if(font_sim.score < 0.94){
                            is_same_style = false;
                        }

                        Similarity fill_sim = new Similarity();
                        fill_sim.score = score[2];
                        fill_sim.index1 = index1;
                        fill_sim.index2 = index2;
                        if(score[2] < 0.94){
                            is_same_style = false;
                        }

                        Similarity height_sim = new Similarity();
                        height_sim.score = score[3];
                        height_sim.index1 = index1;
                        height_sim.index2 = index2;
                        if(score[3] < 0.94){
                            is_same_style = false;
                        }

                        Similarity width_sim = new Similarity();
                        width_sim.score = score[4];
                        width_sim.index1 = index1;
                        width_sim.index2 = index2;
                        if(score[4] < 0.94){
                            is_same_style = false;
                        }

                        Similarity type_sim = new Similarity();
                        type_sim.score = score[5];
                        type_sim.index1 = index1;
                        type_sim.index2 = index2;
                        // Console.WriteLine("add simu");
                        template.value_similarity_list.Add(value_sim);
                        template.font_color_similarity_list.Add(font_sim);
                        template.fill_color_similarity_list.Add(fill_sim);
                        template.height_similarity_list.Add(height_sim);
                        template.width_similarity_list.Add(width_sim);
                        template.type_similarity_list.Add(type_sim);
                        
                        
                        // if(is_same_style){
                        //     temp_dic[found_key].Add(index2);
                        //     // break;
                        // }
                        // }
                        
                    }
                }
                new_temp.Add(template);
                result.Add(temp_dic);
                saveAsJson(new_temp, "data/types/custom/change_xml_all_template_with_simularity.json");
                saveAsJson(result, "data/types/custom/change_xml_template_splited_by_style.json");
                
                // break;
            }
            saveAsJson(new_temp, "data/types/custom/change_xml_all_template_with_simularity.json");
            saveAsJson(result, "data/types/custom/change_xml_template_splited_by_style.json");
        }

        public void get_sheet_number(){
            List<string> path_list = new List<string>();
            path_list.Add("/datadrive/data/dvinfoWithRef.json");
            path_list.Add("/datadrive/data/dvinfoWithRef1.json");
            path_list.Add("/datadrive/data/dvinfoWithRef2.json");
            path_list.Add("/datadrive/data/dvinfoWithRef3.json");
            int index = 0;
            int count= 0;

            Dictionary<string, List<string>> result = new Dictionary<string, List<string>>();
            foreach(var path in path_list){
                string jsonstring = File.ReadAllText(path);
                List<DVInfo> dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
                foreach(DVInfo dv_info in dv_infos){
                    if(dv_info.Type == XLAllowedValues.Custom || dv_info.Type == XLAllowedValues.AnyValue || dv_info.Type == XLAllowedValues.List ){
                        continue;
                    }
                    if(!result.ContainsKey(dv_info.SheetName)){
                        List<string> file_list = new List<string>();
                        result.Add(dv_info.SheetName, file_list);
                        result[dv_info.SheetName].Add(dv_info.FileName);
                    }
                    else{
                        bool is_found=false;
                        foreach(var filename in result[dv_info.SheetName]){
                            if(filename == dv_info.FileName){
                                is_found = true;
                                break;
                            }
                        }
                        if(!is_found){
                            result[dv_info.SheetName].Add(dv_info.FileName);
                        }
                    }
                }
            }
            saveAsJson(result, "boundary_sheetname_2_file.json");
            Dictionary<string, int> sheet_num = new Dictionary<string, int>();
            foreach(var sheetname in result.Keys){
                sheet_num.Add(sheetname, result[sheetname].Count());
            }
            saveAsJson(sheet_num, "boundary_sheetname_2_num.json");
        }

        public void get_most_sheet_number(){
            List<string> path_list = new List<string>();
            path_list.Add("../share/dvinfoWithRef.json");
            path_list.Add("../share/dvinfoWithRef1.json");
            path_list.Add("../share/dvinfoWithRef2.json");
            path_list.Add("../share/dvinfoWithRef3.json");
            int index = 0;
            int count= 0;

            Dictionary<string, List<string>> result = new Dictionary<string, List<string>>();
            foreach(var path in path_list){
                string jsonstring = File.ReadAllText(path);
                List<DVInfo> dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
                foreach(DVInfo dv_info in dv_infos){
                    if(!result.ContainsKey(dv_info.SheetName)){
                        List<string> file_list = new List<string>();
                        result.Add(dv_info.SheetName, file_list);
                        result[dv_info.SheetName].Add(dv_info.FileName);
                    }
                    else{
                        bool is_found=false;
                        foreach(var filename in result[dv_info.SheetName]){
                            if(filename == dv_info.FileName){
                                is_found = true;
                                break;
                            }
                        }
                        if(!is_found){
                            result[dv_info.SheetName].Add(dv_info.FileName);
                        }
                    }
                }
            }
            saveAsJson(result, "most_sheetname_2_file.json");
            Dictionary<string, int> sheet_num = new Dictionary<string, int>();
            foreach(var sheetname in result.Keys){
                sheet_num.Add(sheetname, result[sheetname].Count());
            }
            saveAsJson(sheet_num, "most_sheetname_2_num.json");
        }

        public bool is_all_same(string file1, string file2, string sheetname){
            var worksheet1 = new XLWorkbook(file1).Worksheet(sheetname);
            var worksheet2 = new XLWorkbook(file2).Worksheet(sheetname);
            for(int row=1;row<=100;row+=1){
                for(int column=1;column<=10;column+=1){
                    // all_ += 1;
                    bool exist1 = true;
                    bool exist2 = true;
                    string value1="";
                    string value2="";
                    // Console.Write("load_cell1");
                    // CellFeature cell_feature = new CellFeature();
                    IXLCell cell1 = worksheet1.Cell(1,1);
                    IXLCell cell2 = worksheet2.Cell(1,1);
                    // Console.WriteLine("row:"+row.ToString()+", column:"+column.ToString());
                    try{
                        cell1 = worksheet1.Cell(row,column);
                        value1 = cell1.Value.ToString();
                    }catch{
                        exist1 = false;
                    }
                    try{
                        cell2 = worksheet2.Cell(row,column);
                        value2 = cell2.Value.ToString();
                    }catch{
                        exist2 = false;
                    }
                    if(!exist1 && !exist2){
                        continue;
                    }
                    else if(!exist1 || !exist2){
                        return false;
                    }else{
                        if(value1==value2){
                            continue;
                        }
                        else{
                            return false;
                        }
                    }
                }
            }
            return true;
        }
        public void check_not_same(){
            string jsonstring = File.ReadAllText("sheetname_2_file_devided.json");
            var sheet2file = JsonConvert.DeserializeObject<Dictionary<string, List<List<string>>>>(jsonstring);
            Dictionary<string, List<List<string>>> result = new Dictionary<string, List<List<string>>>();
            int count=1;
            foreach(string sheetname in sheet2file.Keys){
                Console.WriteLine(count.ToString()+"/"+sheet2file.Keys.Count().ToString());
                count+=1;
                if(count <=251){
                    continue;
                }
                foreach(var filelist in sheet2file[sheetname]){
                    if(filelist.Count()==1){
                        continue;
                    }

                    foreach(var file1 in filelist){
                        bool need_continue=true;
                        foreach(var file2 in filelist){
                            if(file1==file2){
                                need_continue=false;
                                continue;
                            }
                            if(need_continue){
                                continue;
                            }
                            bool is_similarity = is_all_same(file1, file2, sheetname);
                            if(!is_similarity){
                                List<string> pair = new List<string>();
                                pair.Add(file1);
                                pair.Add(file2);
                                if(!result.ContainsKey(sheetname)){
                                    List<List<string>> newlist = new List<List<string>>();
                                    result.Add(sheetname, newlist);
                                }
                                result[sheetname].Add(pair);
                            }
                        }
                    }
                }
                saveAsJson(result, "not_all_same_file_1.json");
            }
        }

        public void get_custom_faetures(){
            // string jsonstring = File.ReadAllText("training_sheet.json");
            // var sheetnames = JsonConvert.DeserializeObject<List<string>>(jsonstring);
            // string jsonstring1 = File.ReadAllText("sheetname_2_file.json");
            // var sheet2file = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring1);
            // string jsonstring2 = File.ReadAllText("training_sheet.json");
            // var training_sheet = JsonConvert.DeserializeObject<List<string>>(jsonstring2);
            string jsonstring = File.ReadAllText("data/types/custom/change_xml_cutom_list.json");
            var dvinfos = JsonConvert.DeserializeObject<List<CustomDVInfo>>(jsonstring);
            Dictionary<string, List<string>> sheet_files = new Dictionary<string, List<string>>();
            Dictionary<int, List<SheetFeature>> final_result = new Dictionary<int, List<SheetFeature>>();
            foreach(var dvinfo in dvinfos){
                if(!sheet_files.ContainsKey(dvinfo.SheetName)){
                    List<string> new_files = new List<string>();
                    sheet_files.Add(dvinfo.SheetName, new_files);
                }
                sheet_files[dvinfo.SheetName].Add(dvinfo.FileName);
            }

            Dictionary<string, List<SheetFeature>> result = new Dictionary<string, List<SheetFeature>>();
            // if(File.Exists("CNN_training_origin_dict_filter.json")){
            //     string jsonstring3 = File.ReadAllText("CNN_training_origin_dict_filter_1.json");
            //     result = JsonConvert.DeserializeObject<Dictionary<string, List<SheetFeature>>>(jsonstring1);
            // }   
            
            int count = 1;

            Dictionary<string, List<string>> file2sheets = new Dictionary<string, List<string>>();

            foreach(var sheetname in sheet_files.Keys){
        
                foreach(var file in sheet_files[sheetname]){
                    if(!file2sheets.ContainsKey(file)){
                        List<string> sheetname_list = new List<string>();
                        file2sheets.Add(file, sheetname_list);
                    }
                    file2sheets[file].Add(sheetname);
                }
            }
        
            // Console.WriteLine("start");
            foreach(var file in file2sheets.Keys){
                // bool saved = false;
                // foreach(var saved_sheet in result.Keys){
                //     foreach(var sheet_f in result[saved_sheet])
                //         if(sheet_f.filename==file){
                //             saved = true;
                //             break;
                //         }
                // }
                // if(saved){
                //     count+=1;
                //     continue;
                // }
                // if(count<=2899){
                //     count+=1;
                //     continue;
                // }
                List<string> sheets = file2sheets[file];
                List<SheetFeature> all_sheet_features_per_sheetname = new List<SheetFeature>();
                int count_1 = 1;
                string file_path = file;
                // string file_path = file.Replace("/UnzipData/","/");
                // XLWorkbook workbook;
                Console.WriteLine(file_path);
                // if(file_path == "../../data/061/f0fc1ace-2b15-48c6-981d-22976b45085c_Ly93d3cuY2l0eW9mbm9ydGhsYXN2ZWdhcy5jb20vRGVwYXJ0bWVudHMvRmluYW5jZS9GaWxlcy9TdGF0ZW1lbnRvZkZpbmFuY2VzLzIwMTUvU0I2NS1Nb2RpZmllZEFjY3J1YWxSZXBvcnQtMjAxNS1RMi54bHN4.xlsx"|| file_path=="../../data/079/aaed3cc4-89c2-4e7e-8ece-e231d1feb0ae_aHR0cDovL3d3dy53c2RvdC53YS5nb3YvcHVibGljYXRpb25zL2Z1bGx0ZXh0L2Rlc2lnbi9BU0RFL1BETVNHL1BETVNHLU1hdHJpeC54bHN4.xlsx" || file_path=="../../data/UnzipData/037/3df88066-b900-4063-a466-1dfa709d2960_aHR0cDovL3d3dy5saGNoLm5ocy51ay9tZWRpYS81ODI0L3N0YWZmaW5nLW1hcmNoLTE4Lnhsc3g=.xlsx"|| file_path=="4bd02493-8f88-453a-8f33-d83f0ee469f4_aHR0cDovL3d3dy5saGNoLm5ocy51ay9tZWRpYS81ODc5L3N0YWZmaW5nLWFwcmlsLTIwMTgueGxzeA==.xlsx"){
                //     count+=1;
                //     continue;
                // }
                // try{
                    var workbook = new XLWorkbook(file_path);
                    Console.WriteLine("load succeed");
                    foreach(var sheetname in sheets){
                    Console.WriteLine(count.ToString()+"/"+file2sheets.Keys.Count().ToString()+"   "+count_1.ToString()+"/"+sheets.Count().ToString());
                    SheetFeature sheet_feature = new SheetFeature();
                    sheet_feature.filename = file;
                    sheet_feature.sheetname = sheetname;
                    var worksheet = workbook.Worksheet(sheetname);
                    List<CellFeature> one_sheet_feature = new List<CellFeature>();
                    for(int row=1;row<=100;row+=1){
                        for(int column=1;column<=10;column+=1){
                            // all_ += 1;
                            bool exist1 = true;
                            // Console.Write("load_cell1");
                            CellFeature cell_feature = new CellFeature();
                            IXLCell cell1 = worksheet.Cell(1,1);
                            // Console.WriteLine("row:"+row.ToString()+", column:"+column.ToString());
                            try{
                                cell1 = worksheet.Cell(row,column);
                                var value1 = cell1.Value.ToString();
                            }catch{
                                exist1 = false;
                            }
                            if(exist1){
                                cell_feature = get_cell_features(cell1);
                            }
                            one_sheet_feature.Add(cell_feature);
                        }
                    }
                    sheet_feature.sheetfeature = one_sheet_feature;
                    if(!result.ContainsKey(file_path+"----"+sheetname)){
                        List<SheetFeature> sheet_features = new List<SheetFeature>();
                        result.Add(file_path+"----"+sheetname, sheet_features);
                    }
                    result[file_path+"----"+sheetname].Add(sheet_feature);
                    count_1 += 1;
                }
                count+=1;
                // }catch{
                //     count+=1;
                //     continue;
                // }
                if(count%100==0){
                    saveAsJson(result, "custom_all_training_origin_dict_filter_2.json");
                }
            }

            
            saveAsJson(result, "custom_all_training_origin_dict_filter_2.json");

            foreach(var dvinfo in dvinfos){
                if(!result.ContainsKey(dvinfo.FileName+"----"+dvinfo.SheetName)){
                    continue;
                }
                final_result.Add(dvinfo.ID, result[dvinfo.FileName+"----"+dvinfo.SheetName]);
            }
            saveAsJson(result, "all_custom_dv_sheet_feature.json");
        }

        public void batch_get_sheet_features(){
            // string jsonstring = File.ReadAllText("training_sheet.json");
            // var sheetnames = JsonConvert.DeserializeObject<List<string>>(jsonstring);
            string jsonstring1 = File.ReadAllText("sheetname_2_file.json");
            var sheet2file = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring1);
            string jsonstring2 = File.ReadAllText("training_sheet.json");
            var training_sheet = JsonConvert.DeserializeObject<List<string>>(jsonstring2);
            Dictionary<string, List<SheetFeature>> result = new Dictionary<string, List<SheetFeature>>();
            if(File.Exists("CNN_training_origin_dict_filter_2.json")){
                string jsonstring3 = File.ReadAllText("CNN_training_origin_dict_filter_2.json");
                result = JsonConvert.DeserializeObject<Dictionary<string, List<SheetFeature>>>(jsonstring3);
            }   
            
            int count = 1;

            Dictionary<string, List<string>> file2sheets = new Dictionary<string, List<string>>();

            foreach(var sheetname in training_sheet){
        
                foreach(var file in sheet2file[sheetname]){
                    if(!file2sheets.ContainsKey(file)){
                        List<string> sheetname_list = new List<string>();
                        file2sheets.Add(file, sheetname_list);
                    }
                    file2sheets[file].Add(sheetname);
                }
            }
        
            // Console.WriteLine("start");
            foreach(var file in file2sheets.Keys){
                bool saved = false;
                foreach(var saved_sheet in result.Keys){
                    foreach(var sheet_f in result[saved_sheet])
                        if(sheet_f.filename==file){
                            saved = true;
                            break;
                        }
                }
                if(saved){
                    count+=1;
                    continue;
                }
            
                // if(count<=2899){
                //     count+=1;
                //     continue;
                // }
                List<string> sheets = file2sheets[file];
                List<SheetFeature> all_sheet_features_per_sheetname = new List<SheetFeature>();
                int count_1 = 1;
                // string file_path = file;
                string file_path = file.Replace("/UnzipData/","/");
                // XLWorkbook workbook;
                Console.WriteLine(file_path);
                if(file_path == "../../data/061/f0fc1ace-2b15-48c6-981d-22976b45085c_Ly93d3cuY2l0eW9mbm9ydGhsYXN2ZWdhcy5jb20vRGVwYXJ0bWVudHMvRmluYW5jZS9GaWxlcy9TdGF0ZW1lbnRvZkZpbmFuY2VzLzIwMTUvU0I2NS1Nb2RpZmllZEFjY3J1YWxSZXBvcnQtMjAxNS1RMi54bHN4.xlsx"|| file_path=="../../data/079/aaed3cc4-89c2-4e7e-8ece-e231d1feb0ae_aHR0cDovL3d3dy53c2RvdC53YS5nb3YvcHVibGljYXRpb25zL2Z1bGx0ZXh0L2Rlc2lnbi9BU0RFL1BETVNHL1BETVNHLU1hdHJpeC54bHN4.xlsx" || file_path=="../../data/UnzipData/037/3df88066-b900-4063-a466-1dfa709d2960_aHR0cDovL3d3dy5saGNoLm5ocy51ay9tZWRpYS81ODI0L3N0YWZmaW5nLW1hcmNoLTE4Lnhsc3g=.xlsx"|| file_path=="4bd02493-8f88-453a-8f33-d83f0ee469f4_aHR0cDovL3d3dy5saGNoLm5ocy51ay9tZWRpYS81ODc5L3N0YWZmaW5nLWFwcmlsLTIwMTgueGxzeA==.xlsx"){
                    count+=1;
                    continue;
                }
                if(count >= 860 && count < 900 || count <= 2270){
                    count += 1;
                    continue;
                }
                try{
                    var workbook = new XLWorkbook(file_path);
                    Console.WriteLine("load succeed");
                    foreach(var sheetname in sheets){
                    Console.WriteLine(count.ToString()+"/"+file2sheets.Keys.Count().ToString()+"   "+count_1.ToString()+"/"+sheets.Count().ToString());
                    SheetFeature sheet_feature = new SheetFeature();
                    sheet_feature.filename = file;
                    sheet_feature.sheetname = sheetname;
                    var worksheet = workbook.Worksheet(sheetname);
                    List<CellFeature> one_sheet_feature = new List<CellFeature>();
                    for(int row=1;row<=100;row+=1){
                        for(int column=1;column<=10;column+=1){
                            // all_ += 1;
                            bool exist1 = true;
                            // Console.Write("load_cell1");
                            CellFeature cell_feature = new CellFeature();
                            IXLCell cell1 = worksheet.Cell(1,1);
                            // Console.WriteLine("row:"+row.ToString()+", column:"+column.ToString());
                            try{
                                cell1 = worksheet.Cell(row,column);
                                var value1 = cell1.Value.ToString();
                            }catch{
                                exist1 = false;
                            }
                            if(exist1){
                                cell_feature = get_cell_features(cell1);
                            }
                            one_sheet_feature.Add(cell_feature);
                        }
                    }
                    sheet_feature.sheetfeature = one_sheet_feature;
                    if(!result.ContainsKey(sheetname)){
                        List<SheetFeature> sheet_features = new List<SheetFeature>();
                        result.Add(sheetname, sheet_features);
                    }
                    result[sheetname].Add(sheet_feature);
                    count_1 += 1;
                }
                count+=1;
                }catch{
                    count+=1;
                    continue;
                }
                if(count%1==0){
                    Console.WriteLine("saving...........");
                    saveAsJson(result, "CNN_training_origin_dict_filter_2.json");
                }
            }

            
            saveAsJson(result, "CNN_training_origin_dict_filter_2.json");
        }
        public void negative_batch_200000_get_sheet_features(){
            string jsonstring = File.ReadAllText("../analyze-dv-1/400000_negative_need_feature.json");
            var negative_sheet_2_file = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring);
            int count = 1;
            Dictionary<string, List<string>> negative_file2sheets = new Dictionary<string, List<string>>();
            Dictionary<string, List<SheetFeature>> negative_result = new Dictionary<string, List<SheetFeature>>();
            foreach(var sheetname in negative_sheet_2_file.Keys){
                foreach(var file in negative_sheet_2_file[sheetname]){
                    if(!negative_file2sheets.ContainsKey(file)){
                        List<string> sheetname_list = new List<string>();
                        negative_file2sheets.Add(file, sheetname_list);
                    }
                    negative_file2sheets[file].Add(sheetname);
                }
            }
            foreach(var file in negative_file2sheets.Keys){
                List<string> sheets = negative_file2sheets[file];
                List<SheetFeature> all_sheet_features_per_sheetname = new List<SheetFeature>();
                int count_1 = 1;
                string file_path = file;
                Console.WriteLine(file_path);
                if(count==1){
                    count+=1;
                    continue;
                }
                try{
                    var workbook = new XLWorkbook(file_path);
                    Console.WriteLine("load succeed");
                    foreach(var sheetname in sheets){
                    Console.WriteLine(count.ToString()+"/"+negative_sheet_2_file.Keys.Count().ToString()+"   "+count_1.ToString()+"/"+sheets.Count().ToString());
                    SheetFeature sheet_feature = new SheetFeature();
                    sheet_feature.filename = file;
                    sheet_feature.sheetname = sheetname;
                    var worksheet = workbook.Worksheet(sheetname);
                
                    List<CellFeature> one_sheet_feature = new List<CellFeature>();
                    for(int row=1;row<=100;row+=1){
                        for(int column=1;column<=10;column+=1){
                            bool exist1 = true;
                            CellFeature cell_feature = new CellFeature();
                            IXLCell cell1 = worksheet.Cell(1,1);
                            try{
                                cell1 = worksheet.Cell(row,column);
                                var value1 = cell1.Value.ToString();
                            }catch{
                                exist1 = false;
                            }
                            if(exist1){
                                cell_feature = get_cell_features(cell1);
                            }
                            one_sheet_feature.Add(cell_feature);
                        }
                    }
                    sheet_feature.sheetfeature = one_sheet_feature;
                    if(!negative_result.ContainsKey(sheetname)){
                        List<SheetFeature> sheet_features = new List<SheetFeature>();
                        negative_result.Add(sheetname, sheet_features);
                    }
                    negative_result[sheetname].Add(sheet_feature);
                    count_1 += 1;
                }
                count+=1;
                }catch{
                    count+=1;
                    continue;
                }
                if(count%50==0){
                    saveAsJson(negative_result, "negative_training_origin_dict.json");
                }
            }
            saveAsJson(negative_result, "negative_training_origin_dict.json");

        }
        public void positive_batch_200000_get_sheet_features(){
            string jsonstring1 = File.ReadAllText("../analyze-dv-1/100000_positive_need_feature.json");
            var positive_sheet_2_file = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring1);
            
            int count = 1;
            
            Dictionary<string, List<string>> positive_file2sheets = new Dictionary<string, List<string>>();
            Dictionary<string, List<SheetFeature>> positive_result = new Dictionary<string, List<SheetFeature>>();
            if(File.Exists("positive_training_origin_dict.json")){
                string jsonstring2 = File.ReadAllText("positive_training_origin_dict.json");
                positive_result = JsonConvert.DeserializeObject<Dictionary<string, List<SheetFeature>>>(jsonstring2);
            }
            foreach(var sheetname in positive_sheet_2_file.Keys){
                foreach(var file in positive_sheet_2_file[sheetname]){
                    if(!positive_file2sheets.ContainsKey(file)){
                        List<string> sheetname_list = new List<string>();
                        positive_file2sheets.Add(file, sheetname_list);
                    }
                    positive_file2sheets[file].Add(sheetname);
                }
            }
            foreach(var file in positive_file2sheets.Keys){
                if(count<=48){
                    count+=1;
                    continue;
                }
                List<string> sheets = positive_file2sheets[file];
                List<SheetFeature> all_sheet_features_per_sheetname = new List<SheetFeature>();
                int count_1 = 1;
                string file_path = file;
                Console.WriteLine(file_path);
                
                try{
                    var workbook = new XLWorkbook(file_path);
                    Console.WriteLine("load succeed");
                    foreach(var sheetname in sheets){
                        Console.WriteLine(count.ToString()+"/"+positive_file2sheets.Keys.Count().ToString()+"   "+count_1.ToString()+"/"+sheets.Count().ToString());
                        SheetFeature sheet_feature = new SheetFeature();
                        sheet_feature.filename = file;
                        sheet_feature.sheetname = sheetname;
                        var worksheet = workbook.Worksheet(sheetname);
                        List<CellFeature> one_sheet_feature = new List<CellFeature>();
                        for(int row=1;row<=100;row+=1){
                            for(int column=1;column<=10;column+=1){
                                bool exist1 = true;
                                CellFeature cell_feature = new CellFeature();
                                IXLCell cell1 = worksheet.Cell(1,1);
                                try{
                                    cell1 = worksheet.Cell(row,column);
                                    var value1 = cell1.Value.ToString();
                                }catch{
                                    exist1 = false;
                                }
                                if(exist1){
                                    cell_feature = get_cell_features(cell1);
                                }
                                one_sheet_feature.Add(cell_feature);
                            }
                        }
                        sheet_feature.sheetfeature = one_sheet_feature;
                        if(!positive_result.ContainsKey(sheetname)){
                            List<SheetFeature> sheet_features = new List<SheetFeature>();
                            positive_result.Add(sheetname, sheet_features);
                        }
                        positive_result[sheetname].Add(sheet_feature);
                        count_1 += 1;
                    }
                    count+=1;
                }catch{
                    count+=1;
                    continue;
                }
                if(count%50==0){
                    saveAsJson(positive_result, "positive_training_origin_dict.json");
                }
            }
            saveAsJson(positive_result, "positive_training_origin_dict.json");
        }
        // public bool check_not_same(){

        // }
        public bool is_same_template(string file_name_1, string file_name_2, Dictionary<string, int> sheet_num, double threshold){
            // Console.WriteLine("##################");
            
            if(!File.Exists("/datadrive-2/data/fortune500_test/semi_super_prob/"+file_name_1 + "---" + file_name_2 + ".json")){
                // Console.WriteLine("not exists: calculate");
                List<string> sheet_list_1 = new List<string>();
                List<string> sheet_list_2 = new List<string>();
                int all_sheet = 0;
                file_name_1 = file_name_1.Replace("UnzipData/","");
                file_name_2 = file_name_2.Replace("UnzipData/","");
                foreach(var sheetname in sheet_num.Keys){
                    all_sheet+=sheet_num[sheetname];
                }
                // Console.WriteLine(file_name_1);
                using(var workbook1 = new XLWorkbook("/datadrive/data_fortune500/crawled_xlsx_fortune500/" + file_name_1)){
                    foreach(var sheetname in workbook1.Worksheets.ToArray()){
                        sheet_list_1.Add(sheetname.Name);
                    }
                }
                // Console.WriteLine(file_name_2);
                using(var workbook2 = new XLWorkbook("/datadrive/data_fortune500/crawled_xlsx_fortune500/" + file_name_2)){
                    foreach(var sheetname in workbook2.Worksheets.ToArray()){
                        sheet_list_2.Add(sheetname.Name);
                    }  
                }
                
                if(sheet_list_1.Count()!=sheet_list_2.Count()){
                    // Console.WriteLine("not same sheetlist len");
                    return false;
                }

                // foreach(var sheetname in sheet_list_1){
                //     Console.Write(sheetname);
                //     Console.Write(",");
                // }
                // Console.WriteLine(";");
                // foreach(var sheetname in sheet_list_2){
                //     Console.Write(sheetname);
                //     Console.Write(",");
                // }
                // Console.WriteLine(";");
                foreach(var sheetname1 in sheet_list_1){
                    bool found=false;
                    foreach(var sheetname2 in sheet_list_2){
                        if(sheetname1==sheetname2){
                            found=true;
                        }
                    }
                    if(!found){
                        // Console.WriteLine("not same sheetlist");
                        return false;
                    }
                }
                // return true;
                double prob = 1;
                foreach(var one_sheetname in sheet_list_1){
                    // if()
                    // Console.WriteLine(one_sheetname);
                    prob *= (double)sheet_num[one_sheetname]/all_sheet;
                }
                List<double> result = new List<double>();
                result.Add(prob);
                saveAsJson(result, "/datadrive-2/data/fortune500_test/semi_super_prob/"+file_name_1 + "---" + file_name_2 + ".json");
                if(prob < threshold){
                    // Console.WriteLine("true");
                    return true;
                }
                // Console.WriteLine("false");
                return false;
            }
            else{
                // Console.WriteLine("exists: load");
                string jsonstring = File.ReadAllText("/datadrive-2/data/fortune500_test/semi_super_prob/"+file_name_1 + "---" + file_name_2 + ".json");
                List<double> result= JsonConvert.DeserializeObject<List<double>>(jsonstring);
                double prob = result[0];
                if(prob < threshold){
                    // Console.WriteLine("true");
                    return true;
                }
                // Console.WriteLine("false");
                return false;
            }
            
        }

        public void generate_sheetname_2_num(){
            
        }

        // public filename2sheetnames(){
        //     string jsonstring1 = File.ReadAllText("/datadrive-2/data/fortune500_test/sheet2filenames.json");
        //     Dictionary<string, List<string>> sheetname2filename = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring1);

        // }
        public void devide_by_file_template(){
            string jsonstring = File.ReadAllText("/datadrive-2/data/fortune500_test/sheet2num.json");
            Dictionary<string, int> sheet_num = JsonConvert.DeserializeObject<Dictionary<string, int>>(jsonstring);
            // string jsonstring1 = File.ReadAllText("all_file_sheet.json");
            // Dictionary<string, List<string>> filename2sheetname = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring1);
            //  string jsonstring = File.ReadAllText("data/types/custom/custom_sheet_2_num.json");
            // Dictionary<string, int> sheet_num = JsonConvert.DeserializeObject<Dictionary<string, int>>(jsonstring);
            // string jsonstring1 = File.ReadAllText("boundary_sheetname_2_file.json");
            string jsonstring1 = File.ReadAllText("/datadrive-2/data/fortune500_test/sheet2filenames.json");
            Dictionary<string, List<string>> sheetname2filename = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring1);
            
            // string jsonstring2 = File.ReadAllText("/datadrive-2/data/fortune500_test/sheet2filenames.json");
            // Dictionary<string, List<string>> sheetname2filename = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring2);
            // string jsonstring2 = File.ReadAllText("all_training_sheet.json");
            // List<string> training_sheets = JsonConvert.DeserializeObject<List<string>>(jsonstring2);
            List<double> thresholds = new List<double>(); 
            thresholds.Add(0.00001);
            thresholds.Add(0.0001);
            thresholds.Add(0.001);
            thresholds.Add(0.01);
            thresholds.Add(0.1);
            thresholds.Add(0.2);
            thresholds.Add(0.4);
            thresholds.Add(0.6);
            thresholds.Add(0.8);

            var filename2sheetname = new Dictionary<string, List<string>>();
            foreach(var sheetname in sheetname2filename.Keys){
                foreach(var filename in sheetname2filename[sheetname]){
                    if(!filename2sheetname.ContainsKey(filename)){
                        List<string> files = new List<string>();
                        filename2sheetname.Add(filename, files);
                    }
                    filename2sheetname[filename].Add(sheetname);
                    // Console.WriteLine(sheetname+"   "+ filename);
                }
            }
            saveAsJson(filename2sheetname, "fortune500_filename2sheetname.json");
            // foreach(var threshold in thresholds){
            //     Dictionary<string, List<SameTempFiles>> result = new Dictionary<string, List<SameTempFiles>>();
            //     // if(File.Exists("sheetname_2_file_devided.json")){
            //     //     string jsonstring3 = File.ReadAllText("sheetname_2_file_devided.json");
            //     //     result = JsonConvert.DeserializeObject<Dictionary<string, List<List<string>>>>(jsonstring3);
            //     // }
                
            //     int count=1;
                
            //     // var sheetname2filename = new Dictionary<string, List<string>>();
            //     // foreach(var filename in filename2sheetname.Keys){
            //         // foreach(var sheetname in filename2sheetname[filename]){
            //             // if(!sheetname2filename.ContainsKey(sheetname)){
            //                 // List<string> files = new List<string>();
            //                 // sheetname2filename.Add(sheetname, files);
            //             // }
            //             // sheetname2filename[sheetname].Add(filename);
            //             // Console.WriteLine(sheetname+"   "+ filename);
            //         // }
            //     // }
                
            //     // return;
                
            //     foreach(var sheetname in sheetname2filename.Keys){
    
            //         Console.WriteLine(sheetname + ":" + count.ToString()+"/"+sheetname2filename.Count().ToString());
            //         count+=1;
            //         if(result.ContainsKey(sheetname)){
            //             int file_num = 0;
            //             foreach(var list in result[sheetname]){
            //                 file_num += list.filenames.Count();
            //             }
                        
            //             if(file_num == sheetname2filename[sheetname].Count()){
            //                 continue;
            //             }else{
            //                 Console.WriteLine("file_num: "+file_num.ToString());
            //                 Console.WriteLine("sheetname2filename[sheetname].Count(): "+sheetname2filename[sheetname].Count().ToString());
            //             }
            //         }
            //         List<SameTempFiles> one_sheetname = new List<SameTempFiles>();
            //         if(result.ContainsKey(sheetname)){
            //             one_sheetname = result[sheetname];
            //         }
                    
            //         // bool is_found=false;
            //         // foreach(var training_sheetname in training_sheets){
            //         //     if(training_sheetname==sheetname){
            //         //         is_found=true;
            //         //         break;
            //         //     }
            //         // }
            //         // if(!is_found){
            //         //     continue;
            //         // }
            //         Console.WriteLine("sheetname: "+sheetname);
                    
            //         foreach(var file_name_1 in sheetname2filename[sheetname]){
            //             if(file_name_1 in )
            //             Console.WriteLine("file_name_1:" + file_name_1);
            //             bool is_in_1 = false;
            //             int found_index_1 =0;
            //             // Console.WriteLine("))))))))))))))))");
            //             for(int one_index=0; one_index<one_sheetname.Count();one_index++){
            //             // foreach(var per_list in one_sheetname){
        
            //                 List<string> per_list = one_sheetname[one_index].filenames;
            //                 foreach(var one_file in per_list){
            //                     if(file_name_1==one_file){
            //                         is_in_1=true;
            //                         found_index_1 = one_index;
            //                         // break;
            //                     }
            //                 }
            //             }
            //             if(!is_in_1){
            //                 SameTempFiles new_list = new SameTempFiles();
            //                 new_list.sheetnames = filename2sheetname[file_name_1];
            //                 new_list.filenames = new List<string>();
            //                 new_list.filenames.Add(file_name_1);
            //                 one_sheetname.Add(new_list);
            //                 found_index_1 = one_sheetname.Count()-1;
            //             }
            //             // Console.WriteLine("found_:"+found_index_1.ToString());
            //             // int temp_index=0;
            //             bool need_continue = true;
            //             foreach(var file_name_2 in sheetname2filename[sheetname]){ 
            //                 Console.WriteLine("file_name_2:" + file_name_2);
            //                 // Console.WriteLine("file_name1:"+file_name_1);
            //                 // Console.WriteLine("file_name2:"+temp_index);
            //                 // temp_index+=1;
            //                 if(file_name_1==file_name_2){
            //                     need_continue= false;
            //                     continue;
            //                 }
            //                 if(need_continue){
            //                     continue;
            //                 }
            //                 // Console.WriteLine("*****");
            //                 bool is_in_2 = false;
            //                 for(int one_index=0; one_index<one_sheetname.Count();one_index++){
            //                     List<string> per_list = one_sheetname[one_index].filenames;
            //                     foreach(var one_file in per_list){
            //                         if(file_name_2==one_file){
            //                             is_in_2=true;
            //                             break;
            //                         }
            //                     }
            //                     if(is_in_2){
            //                         break;
            //                     }
            //                 }
            //                 if(is_in_2){
            //                     continue;
            //                 }
            //                 // Console.WriteLine(file_name_1 + "     " + file_name_2);
            //                 bool same_template = false;
            //                 if(filename2sheetname[file_name_1].Count() == filename2sheetname[file_name_2].Count()){
            //                     same_template = true;
            //                     foreach(var sheetname_1 in filename2sheetname[file_name_1]){
            //                         bool found_sheet_1 = false;
            //                         foreach(var sheetname_2 in filename2sheetname[file_name_2]){
            //                             if(sheetname_1 == sheetname_2){
            //                                 found_sheet_1 = true;
            //                             }
            //                         }
            //                         if(! found_sheet_1){
            //                             same_template  = false;
            //                             break;
            //                         }
            //                     }
            //                 }

            //                 try{
            //                     // Console.WriteLine("start is_same_template");
            //                     // Console.WriteLine("file_name_1 " + file_name_1);
            //                     // Console.WriteLine("file_name_2 " + file_name_2);
            //                     same_template = is_same_template(file_name_1, file_name_2, sheet_num, threshold);
            //                     // Console.WriteLine("end is_same_template");
            //                 }catch{
            //                     same_template = false;
            //                 }
            //                 if(same_template == false){
            //                     try{
            //                         List<double> result_false = new List<double>();
            //                         result_false.Add(100);
            //                         saveAsJson(result_false, "/datadrive-2/data/fortune500_test/semi_super_prob/"+file_name_1 + "---" + file_name_2 + ".json");
            //                     }
            //                     catch{
            //                         continue;
            //                     }
            //                 }
            //                 if(same_template){
            //                     one_sheetname[found_index_1].filenames.Add(file_name_2);
            //                 }
            //             }
            //         }
            //         // if(result.ContainsKey(sheetname)){
            //         //     foreach(var one_list in one_sheetname){
            //         //         result[sheetname].Add(one_list);
            //         //     }
            //         // }else{
            //         result.Add(sheetname, one_sheetname);
            //         // break;
            //         // }
                    
                    
            //         // break;
            //     }
            //     saveAsJson(result, "fortune500_sheetname_2_file_devided_"+threshold.ToString()+".json");
            }
            
        // }
        public void get_training_sheet_name(){
            string jsonstring = File.ReadAllText("data/types/custom/custom_sheet_2_num.json");
            Dictionary<string, int> sheet_num = JsonConvert.DeserializeObject<Dictionary<string, int>>(jsonstring);
            string jsonstring1 = File.ReadAllText("training_sheet.json");
            List<string> sheet_num1 = JsonConvert.DeserializeObject<List<string>>(jsonstring1);
            List<string> result = new List<string>();
            int count = 0;
            foreach(var sheet_name in sheet_num.Keys){
                count += sheet_num[sheet_name];
                // bool isdigit = false;
                // bool isletter = false;
                // bool issymol = false;
                // bool ispunc = false;
                // bool iswhitespace = false;
                // List<int> add_type = new List<int>();
                // if(sheet_num[sheet_name]>2 && sheet_num[sheet_name]<50){
                //     foreach(char cha in sheet_name){
                //         if(Char.IsDigit(cha)){
                //             isdigit=true;
                //             bool is_in = false;
                //             foreach(int i in add_type){
                //                 if(i==1){
                //                     is_in=true;
                //                     break;
                //                 }
                //             }
                //             if(!is_in){
                //                 add_type.Add(1);
                //             }
                            
                //             // char_type
                //         }
                //         if(Char.IsLetter(cha)){
                //             isletter=true;
                //             bool is_in = false;
                //             foreach(int i in add_type){
                //                 if(i==2){
                //                     is_in=true;
                //                     break;
                //                 }
                //             }
                //             if(!is_in){
                //                 add_type.Add(2);
                //             }
                //         }
                //         if(Char.IsSymbol(cha)){
                //             issymol=true;
                //             bool is_in = false;
                //             foreach(int i in add_type){
                //                 if(i==3){
                //                     is_in=true;
                //                     break;
                //                 }
                //             }
                //             if(!is_in){
                //                 add_type.Add(3);
                //             }
                //         }
                //         if(Char.IsPunctuation(cha)){
                //             ispunc=true;
                //             bool is_in = false;
                //             foreach(int i in add_type){
                //                 if(i==4){
                //                     is_in=true;
                //                     break;
                //                 }
                //             }
                //             if(!is_in){
                //                 add_type.Add(4);
                //             }
                //         }
                //         if(Char.IsWhiteSpace(cha)){
                //             iswhitespace=true;
                //             bool is_in = false;
                //             foreach(int i in add_type){
                //                 if(i==5){
                //                     is_in=true;
                //                     break;
                //                 }
                //             }
                //             if(!is_in){
                //                 add_type.Add(5);
                //             }
                //         }
                //     }
                //     if(add_type.Count()>=3){
                //         result.Add(sheet_name);
                //     }
                //     // break;
                // }
            }
            Console.WriteLine(count);
            // saveAsJson(result, "custom_training_sheet.json");
            Console.WriteLine(sheet_num1.Count());
        }
    }
}