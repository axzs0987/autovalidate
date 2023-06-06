using System;
using ClosedXML.Excel;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using XLParser;

namespace AnalyzeDV
{
    class CustomMetaInfo{
        public int all_number; 
        public int negative_number;
        public int error_type_number;
        public int mean_height;
        public int mean_width;
        public Dictionary<string, int> sheet_duplicated_formular_number;
        public Dictionary<string, int> duplicated_formular_number;
        public Dictionary<string, int> content_type_number_dic;
        public Dictionary<string, int> formular_number_dict;
        public Dictionary<string, int> sheet_formular_number_dict;
        public Dictionary<string, List<int>> content_type_dic;
        public Dictionary<string, List<int>> formular_dict;
        public Dictionary<string, List<int>> sheet_formular_dict;
    }
    class Analyzer{
        private List<DVInfo> dv_infos;
        private List<Dictionary<string, string>> error_dict_list;
        private const string dv_info_file_name = "/datadrive/data/dvinfoWithRef.json";
        private const string error_path = "../extractDV-master/new_data/error.json";
        private const string default_sheet = "default_sheet";
        private const string content_path ="../extractDV-master/new_data/content/";

        private const string meta_info_file_name = "data/metainfo.json";
        public Analyzer() {
            loadDVInfos(dv_info_file_name);
            loadErrorList();
        }

        public void saveAsJson(object need_save_content, string file_name){
            string jsonData = JsonConvert.SerializeObject(need_save_content);
            File.WriteAllText(file_name, jsonData);
        }
        public void loadDVInfos(string dv_info_file_path){
            if(!File.Exists(@dv_info_file_path)){
                dv_infos = new List<DVInfo>{};
            } else{
                string jsonstring = File.ReadAllText(dv_info_file_path);
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
        }

        public void loadErrorList(){
            if(!File.Exists(@error_path)){
                error_dict_list = new List<Dictionary<string, string>>{};
            } else{
                string jsonstring = File.ReadAllText(error_path);
                error_dict_list = JsonConvert.DeserializeObject<List<Dictionary<string, string>>>(jsonstring);
            }
        }
        
        public void analyze_formula(){
            string jsonstring1 = File.ReadAllText("custom_sampled_file.json");
            List<string> sampled_file = JsonConvert.DeserializeObject<List<string>>(jsonstring1);
            Dictionary<string, List<CellFormula>> result = new Dictionary<string, List<CellFormula>>();
            int count = 0;
            int batch_id = 2;
            foreach(var filesheet in sampled_file){
                try{
                    count += 1;
                    Console.WriteLine(count.ToString()+"/"+sampled_file.Count().ToString());
                    // if(count == 1802){
                    //     continue;
                    // }
                    // if(count <= 2500){
                    //     continue;
                    // }
                  
                    var filesheetlist= filesheet.Split("---");
                    string filename = filesheetlist[0];
                    string sheetname = filesheetlist[1];
                    filename = filename.Replace("/UnzipData","");
                    Console.WriteLine(filesheet);
                    var workbook = new XLWorkbook(filename);
                    IXLWorksheet sheet = workbook.Worksheet(sheetname);
                    int firstcolumn = sheet.FirstColumnUsed().ColumnNumber();
                    int lastcolumn = sheet.LastColumnUsed().ColumnNumber();
                    int firstrow = sheet.FirstRowUsed().RowNumber();
                    int lastrow = sheet.LastRowUsed().RowNumber();
                    List<CellFormula> forlist = new List<CellFormula>();
                    result.Add(filesheet, forlist);
                    Console.WriteLine("result add!");
                    Console.WriteLine("result lengh:" + result.Count());
                    for(int row =firstrow; row <= lastrow; row++){
                        for(int col = firstcolumn; col <= lastcolumn; col++){
                            // Console.WriteLine("before cell row:"+row.ToString()+",col:"+col.ToString());
                            var cell = sheet.Cell(row, col);
                            if(cell.FormulaA1.Length==0){
                                continue;
                            }
                            
                            // Console.WriteLine("row:"+row.ToString()+",col:"+col.ToString());
                            CellFormula formula = new CellFormula();
                            formula.row = row;
                            formula.column = col;
                            formula.formulaA1 = cell.FormulaA1;
                            formula.formulaR1C1 = cell.FormulaR1C1;
                            try{
                                if(cell.FormulaReference.Worksheet.Name != sheetname){
                                    continue;
                                }
                                // Console.WriteLine("before as range."); 
                                var refrange = cell.FormulaReference.AsRange();
                                // Console.WriteLine("as range.");                            formula.formulaReferenceFC = refrange.FirstColumn().ColumnNumber();
                                formula.formulaReferenceFR = refrange.FirstRow().RowNumber();
                                formula.formulaReferenceLC = refrange.LastColumn().ColumnNumber();
                                formula.formulaReferenceLR = refrange.LastRow().RowNumber();
                            }catch{

                                // Console.WriteLine("Error0");
                            }
                            
                            result[filesheet].Add(formula);
                        }
                    }
                    
                }catch{
                    Console.WriteLine("Error1");
                    continue;
                }
                if(result.Count() %500 == 0){
                        saveAsJson(result, "Formulas_20000sheets_custom.json");
                        // saveAsJson(result, "Formulas_20000sheets_recheck_"+batch_id.ToString()+".json");
                        batch_id += 1;
                        // result = new Dictionary<string, List<Formula>>();
                    }
            }
            saveAsJson(result, "Formulas_20000sheets_custom.json");
            // saveAsJson(result, "Formulas_20000sheets_recheck_"+batch_id.ToString()+".json");
        }
        public void analyze_training_formula(){
            string jsonstring1 = File.ReadAllText("all_file_sheet.json");
            Dictionary<string, List<string>> filename2sheetname = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring1);
            // string jsonstring2 = File.ReadAllText("TrainingFormulas.json");
            // Dictionary<string, List<CellFormula>> result = JsonConvert.DeserializeObject<Dictionary<string, List<CellFormula>>>(jsonstring2);
            // Dictionary<string, List<string>> filename2sheetname = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring1);
            // List<string> sampled_file = JsonConvert.DeserializeObject<List<string>>(jsonstring1);
            Dictionary<string, List<CellFormula>> result = new Dictionary<string, List<CellFormula>>();
            
            int batch_id = 2;
            int wb_count = 0;
            int save_id = 50;
            foreach(var fname in filename2sheetname.Keys){ 
                wb_count += 1;        
                Console.WriteLine(wb_count.ToString()+"/"+filename2sheetname.Keys.Count().ToString());
                int count = 0;
                if(wb_count <= 5328){
                    continue;
                }
                foreach(var sheetname in filename2sheetname[fname]){
                    try{
                        string filesheet = fname + "---" + sheetname;
                        if(result.ContainsKey(filesheet)){
                            continue;
                        }
                        // else{
                        //     result = new Dictionary<string, List<CellFormula>>();
                        // }
                        count += 1;
                        Console.WriteLine(count.ToString()+"/"+filename2sheetname[fname].Count().ToString());
                        string filename = fname.Replace("/UnzipData","");
                        var workbook = new XLWorkbook(filename);
                        IXLWorksheet sheet = workbook.Worksheet(sheetname);
                        int firstcolumn = sheet.FirstColumnUsed().ColumnNumber();
                        int lastcolumn = sheet.LastColumnUsed().ColumnNumber();
                        int firstrow = sheet.FirstRowUsed().RowNumber();
                        int lastrow = sheet.LastRowUsed().RowNumber();
                        List<CellFormula> forlist = new List<CellFormula>();
                        
                        result.Add(filesheet, forlist);
                        Console.WriteLine("result add!");
                        Console.WriteLine("result lengh:" + result.Count());
                        for(int row =firstrow; row <= lastrow; row++){
                            for(int col = firstcolumn; col <= lastcolumn; col++){
                                // Console.WriteLine("before cell row:"+row.ToString()+",col:"+col.ToString());
                                var cell = sheet.Cell(row, col);
                                if(cell.FormulaA1.Length==0){
                                    continue;
                                }
                                
                                // Console.WriteLine("row:"+row.ToString()+",col:"+col.ToString());
                                CellFormula formula = new CellFormula();
                                formula.row = row;
                                formula.column = col;
                                formula.formulaA1 = cell.FormulaA1;
                                formula.formulaR1C1 = cell.FormulaR1C1;
                                try{
                                    if(cell.FormulaReference.Worksheet.Name != sheetname){
                                        continue;
                                    }
                                    // Console.WriteLine("before as range."); 
                                    var refrange = cell.FormulaReference.AsRange();
                                    // Console.WriteLine("as range.");                            formula.formulaReferenceFC = refrange.FirstColumn().ColumnNumber();
                                    formula.formulaReferenceFR = refrange.FirstRow().RowNumber();
                                    formula.formulaReferenceLC = refrange.LastColumn().ColumnNumber();
                                    formula.formulaReferenceLR = refrange.LastRow().RowNumber();
                                }catch{

                                    // Console.WriteLine("Error0");
                                }
                                
                                result[filesheet].Add(formula);
                            }
                        }
                        
                    }catch{
                        Console.WriteLine("Error1");
                        continue;
                    }
                    if(result.Count() >= 1000){
                        
                            saveAsJson(result, "TrainingFormulas_"+save_id.ToString()+".json");
                            // saveAsJson(result, "Formulas_20000sheets_recheck_"+batch_id.ToString()+".json");
                            // batch_id += 1;
                            save_id += 1;
                            result = new Dictionary<string, List<CellFormula>>();
            
                        }
                }
            }
            saveAsJson(result, "TrainingFormulas_"+save_id.ToString()+".json");
            // saveAsJson(result, "Formulas_20000sheets_recheck_"+batch_id.ToString()+".json");
        }
        public void generate_sampled_files(){
            string jsonstring = File.ReadAllText("clusterd_custom_sheets.json");
            Dictionary<string, List<string>> custom_sampled_file = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring);
            string jsonstring1 = File.ReadAllText("clusterd_boundary_sheets.json");
            Dictionary<string, List<string>> boundary_sampled_file = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring1);

            List<string> sampled_file = new List<string>();
            foreach(string dvtemp in custom_sampled_file.Keys){
                foreach(string filename in custom_sampled_file[dvtemp]){
                    bool isin = false;
                    foreach(string infile in sampled_file){
                        if(infile == filename){
                            isin = true;
                        }
                    }
                    if(!isin){
                        sampled_file.Add(filename);
                    }
                }
            }

            // foreach(string dvtemp in boundary_sampled_file.Keys){
            //     foreach(string filename in boundary_sampled_file[dvtemp]){
            //         bool isin = false;
            //         foreach(string infile in sampled_file){
            //             if(infile == filename){
            //                 isin = true;
            //             }
            //         }
            //         if(!isin){
            //             sampled_file.Add(filename);
            //         }
            //     }
            //     if(sampled_file.Count() > 20000){
            //         Console.WriteLine(sampled_file.Count());
            //         break;
            //     }
            // }
            saveAsJson(sampled_file, "custom_sampled_file.json");
        }
        public void get_all_custom_sheets(){
            string jsonstring = File.ReadAllText("data/types/custom/custom_list.json");
            List<DVInfo> custom_dvinfos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            Dictionary<string, List<string>> sampled_file = new Dictionary<string, List<string>>();
            foreach(var custom_dvinfo in custom_dvinfos){
                string dvtemp = custom_dvinfo.Type.ToString()+"---"+custom_dvinfo.Operator.ToString()+"---"+custom_dvinfo.MinValue.ToString()+"---"+custom_dvinfo.MaxValue.ToString();
                if(!sampled_file.ContainsKey(dvtemp)){
                    List<string> new_filesheet = new List<string>();
                    sampled_file.Add(dvtemp, new_filesheet);
                }
                bool isin = false;
                foreach(string filesheetname in sampled_file[dvtemp]){
                    if(filesheetname  == custom_dvinfo.FileName + "---" + custom_dvinfo.SheetName){
                        isin = true;
                        break;
                    }
                }
                if(!isin){
                    sampled_file[dvtemp].Add(custom_dvinfo.FileName + "---" + custom_dvinfo.SheetName);
                }
            }
            // saveAsJson(result, "clusterd_boundary_list.json");
            saveAsJson(sampled_file, "clusterd_custom_sheets.json");
        }
        public void cluster_boundary(){
            Dictionary<string, List<DVInfo>> result = new Dictionary<string, List<DVInfo>>();
            Dictionary<string, List<string>> sampled_file = new Dictionary<string, List<string>>();
            string jsonstring = File.ReadAllText("data/types/boundary/boundary_list.json");
            List<DVInfo> boundary_dvinfos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
                     
            foreach(var boundary_dvinfo in boundary_dvinfos){
                string dvtemp = boundary_dvinfo.Type.ToString()+"---"+boundary_dvinfo.Operator.ToString()+"---"+boundary_dvinfo.MinValue.ToString()+"---"+boundary_dvinfo.MaxValue.ToString();
                if(!result.ContainsKey(dvtemp)){
                    List<DVInfo> new_list = new List<DVInfo>();
                    List<string> new_filesheet = new List<string>();
                    result.Add(dvtemp, new_list);
                    sampled_file.Add(dvtemp, new_filesheet);
                }
                result[dvtemp].Add(boundary_dvinfo);
                bool isin = false;
                foreach(string filesheetname in sampled_file[dvtemp]){
                    if(filesheetname  == boundary_dvinfo.FileName + "---" + boundary_dvinfo.SheetName){
                        isin = true;
                        break;
                    }
                }
                if(!isin){
                    sampled_file[dvtemp].Add(boundary_dvinfo.FileName + "---" + boundary_dvinfo.SheetName);
                }
            }
            // saveAsJson(result, "clusterd_boundary_list.json");
            saveAsJson(sampled_file, "clusterd_boundary_sheets.json");
        }

        public void cluster_list(){
            Dictionary<string, List<DVInfo>> result = new Dictionary<string, List<DVInfo>>();
            List<List<DVInfo>> dvinfos_list = new List<List<DVInfo>>();
            string jsonstring = File.ReadAllText("continous_batch_0.json");
            List<DVInfo> list_dvinfos_0 = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            dvinfos_list.Add(list_dvinfos_0);
            string jsonstring1 = File.ReadAllText("continous_batch_1.json");
            List<DVInfo> list_dvinfos_1 = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring1);
            dvinfos_list.Add(list_dvinfos_1);
            string jsonstring2 = File.ReadAllText("continous_batch_2.json");
            List<DVInfo> list_dvinfos_2 = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring2);
            dvinfos_list.Add(list_dvinfos_2);
            string jsonstring3 = File.ReadAllText("continous_batch_3.json");
            List<DVInfo> list_dvinfos_3 = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring3);
            dvinfos_list.Add(list_dvinfos_3);        
                      
            int batch_id = 0;
            foreach(var list_dvinfos in dvinfos_list){
                foreach(var list_dvinfo in list_dvinfos){
                    string dvtemp = list_dvinfo.Type.ToString()+"---"+list_dvinfo.Operator.ToString()+"---"+list_dvinfo.MinValue.ToString();
                    if(!result.ContainsKey(dvtemp)){
                        if(result.Count()>=1000){
                            saveAsJson(result, "clusterd_list_list_"+batch_id.ToString()+".json");
                            batch_id += 1;
                            result.Clear();
                        }
                        List<DVInfo> new_list = new List<DVInfo>();
                        result.Add(dvtemp, new_list);
                    }
                    result[dvtemp].Add(list_dvinfo);
                }
            }
            
            
        }

        public void getMetaInfo(){
            Dictionary<string, object> meta_info = new Dictionary<string, object>();

            int any_number = 0;
            int custom_number = 0;
            int int_number = 0;
            int float_number = 0;
            int date_number = 0;
            int time_number = 0;
            int list_number = 0;
            int text_len_number = 0;

            List<string> success_file_list = new List<string>{};
            List<string> fail_file_list = new List<string>{};

            List<DVInfo> custom_list = new List<DVInfo>{};
            List<DVInfo> int_list = new List<DVInfo>{};
            List<DVInfo> float_list = new List<DVInfo>{};
            List<DVInfo> boundary_list = new List<DVInfo>{};
            List<DVInfo> date_list = new List<DVInfo>{};
            List<DVInfo> time_list = new List<DVInfo>{};
            List<DVInfo> any_list = new List<DVInfo>{};
            List<DVInfo> list_list = new List<DVInfo>{};
            List<DVInfo> list_list1 = new List<DVInfo>{};
            List<DVInfo> list_list2 = new List<DVInfo>{};
            List<DVInfo> list_list3 = new List<DVInfo>{};
            List<DVInfo> text_len_list = new List<DVInfo>{};

            int height_all = 0;
            int width_all = 0;
            int[] max_height_width = {0,0};
            int[] min_height_width = {20000,20000};
            int negtive_height = 0;
            List<string> path_list = new List<string>();
            path_list.Add("/datadrive/data/dvinfoWithRef.json");
            path_list.Add("/datadrive/data/dvinfoWithRef1.json");
            path_list.Add("/datadrive/data/dvinfoWithRef2.json");
            path_list.Add("/datadrive/data/dvinfoWithRef3.json");
            int index = 0;
            int count= 0;
            foreach(var path in path_list){
                loadDVInfos(path);
                
                foreach(DVInfo dv_info in dv_infos){
                    count += 1;
                    // if(dv_info.Type != XLAllowedValues.Custom){
                    //     continue;
                    // }
                    dv_info.batch_id = index;

                    Console.WriteLine(dv_info.ID);
                    if(dv_info.Height >= 0){
                        height_all += dv_info.Height;
                        width_all += dv_info.Width;
                    } else{
                        dv_info.Height = 0;
                        negtive_height += 1;
                        continue;
                    }


                    if(dv_info.Height > max_height_width[0] ){
                        max_height_width[0] = dv_info.Height;
                    }
                    if(dv_info.Width > max_height_width[1]){
                        max_height_width[1] = dv_info.Width;
                    }
                    if(dv_info.Height < min_height_width[0] ){
                        min_height_width[0] = dv_info.Height;
                    }
                    if(dv_info.Width < min_height_width[1]){
                        min_height_width[1] = dv_info.Width;
                    }

                    if(dv_info.Type == XLAllowedValues.AnyValue){
                        any_list.Add(dv_info);
                        any_number += 1;
                    } else if(dv_info.Type == XLAllowedValues.Custom){
                        custom_list.Add(dv_info);
                        custom_number += 1;
                       
                    } else if(dv_info.Type == XLAllowedValues.Date){
                        date_list.Add(dv_info);
                        date_number += 1;
                        boundary_list.Add(dv_info);
                    } else if(dv_info.Type == XLAllowedValues.Decimal){
                        float_list.Add(dv_info);
                        float_number += 1;
                        boundary_list.Add(dv_info);
                    } else if(dv_info.Type == XLAllowedValues.List){
                        list_list.Add(dv_info);
                        list_number += 1;
                    } else if(dv_info.Type == XLAllowedValues.TextLength){
                        text_len_list.Add(dv_info);
                        text_len_number += 1;
                        boundary_list.Add(dv_info);
                    } else if(dv_info.Type == XLAllowedValues.WholeNumber){
                        int_list.Add(dv_info);
                        int_number += 1;
                        boundary_list.Add(dv_info);
                    } else if(dv_info.Type == XLAllowedValues.Time){
                        time_list.Add(dv_info);
                        time_number += 1;
                        boundary_list.Add(dv_info);
                    }

                    var is_in_suc = false;
                    foreach(string filename in success_file_list){
                        if(dv_info.FileName == filename){
                            is_in_suc = true;
                            break;
                        }
                    }
                    if(is_in_suc == false){
                        success_file_list.Add(dv_info.FileName);
                    }

                }

                Dictionary<string, int> error_number_dic = new Dictionary<string, int>();
                foreach(Dictionary<string, string> error in error_dict_list){
                    var is_in_fail = false;
                    foreach(string filename in fail_file_list){
                        if(error["FileName"] == filename){
                            is_in_fail = true;
                            break;
                        }
                    }
                    if(is_in_fail == false){
                        fail_file_list.Add(error["FileName"]);
                    }
                    if(!error_number_dic.ContainsKey(error["Message"])){
                        error_number_dic.Add(error["Message"], 0);
                        
                    } 
                    error_number_dic[error["Message"]] += 1;
                }

                index += 1;
            }
           
            // meta_info.Add("success_file_number", success_file_list.Count());
            // meta_info.Add("fail_file_number", fail_file_list.Count());

            // meta_info.Add("all_number", dv_infos.Count());
            
            // meta_info.Add("negative_height_number", negtive_height);
            // meta_info.Add("mean_height", height_all / (dv_infos.Count() - negtive_height));
            // meta_info.Add("mean_width", width_all / (dv_infos.Count() - negtive_height));
            // meta_info.Add("max_height_width", max_height_width);
            // meta_info.Add("min_height_width", min_height_width);

            // meta_info.Add("any_number", any_number);
            // meta_info.Add("custom_number", custom_number);
            // meta_info.Add("int_number", int_number);
            // meta_info.Add("float_number", float_number);
            // meta_info.Add("date_number", date_number);
            // meta_info.Add("time_number", time_number);
            // meta_info.Add("text_len_number", text_len_number);
            // meta_info.Add("list_number", list_number);

            // var all_number = any_number + custom_number + int_number + float_number + date_number + time_number + text_len_number + list_number;
            // meta_info.Add("any_ratio", (double)any_number / all_number);
            // meta_info.Add("custom_ratio", (double)custom_number / all_number);
            // meta_info.Add("int_ratio", (double)int_number / all_number);
            // meta_info.Add("float_ratio", (double)float_number / all_number);
            // meta_info.Add("date_ratio", (double)date_number / all_number);
            // meta_info.Add("time_ratio", (double)time_number / all_number);
            // meta_info.Add("text_len_ratio", (double)text_len_number / all_number);
            // meta_info.Add("list_ratio", (double)list_number / all_number);

            // meta_info.Add("error_count", error_number_dic);
            // saveAsJson(meta_info, meta_info_file_name);
            saveAsJson(custom_list, "data/types/custom/custom_list.json");
            // saveAsJson(boundary_list, "data/types/boundary/boundary_list.json");
            // saveAsJson(any_list, "data/types/other/any_list.json");
            // saveAsJson(int_list, "data/types/boundary/int_list.json");
            // saveAsJson(float_list, "data/types/boundary/float_list.json");
            // saveAsJson(date_list, "data/types/boundary//date_list.json");
            // saveAsJson(text_len_list, "data/types/boundary//text_len_list.json");
            // saveAsJson(time_list, "data/types/boundary//time_list.json");
            // for(var list_index=0;list_index<=list_list.Count();list_index++){
                // if(list_index >= 0 && list_index < 1000000){
                    // list_list1.Add(list_list[list_index]);
                // }
                // if(list_index >= 1000000 && list_index < 2000000){
                    // list_list2.Add(list_list[list_index]);
                // }
                // if(list_index >= 2000000){
                    // list_list3.Add(list_list[list_index]);
                // }
            // }
            // saveAsJson(list_list1, "data//types/list/list_list1.json");
            // saveAsJson(list_list2, "data//types/list/list_list2.json");
            // saveAsJson(list_list3, "data//types/list/list_list3.json");
            // Console.WriteLine("all_:"+count.ToString());
            // Console.WriteLine("boundary:"+boundary_list.Count().ToString());
            // Console.WriteLine("list:"+list_list.Count().ToString());
        }

        public void getCustomMetaInfo(){
            Dictionary<string, object> custom_meta_info = new Dictionary<string, object>();
            List<DVInfo> custom_dv_infos;
            
            if(!File.Exists(@dv_info_file_name)){
                Console.WriteLine("no custom list. Please run getMetaInfo first！");
                return;
            } else{
                string jsonstring = File.ReadAllText("data/types/custom/custom_list.json");
                custom_dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }

            int all_custom_number = custom_dv_infos.Count();
            Dictionary<string, List<List<string>>> formular_dict;
            Dictionary<string, int> formular_number_dict;
            if(!File.Exists("data/types/custom/formular_number_dict.json")){
               formular_number_dict = new Dictionary<string, int>();
            } else{
                string jsonstring = File.ReadAllText("data/types/custom/formular_number_dict.json");
                formular_number_dict = JsonConvert.DeserializeObject<Dictionary<string, int>>(jsonstring);
            }
       
            if(!File.Exists("data/types/custom/formular_dict.json")){
                formular_dict = new Dictionary<string, List<List<string>>>();
            } else{
                string jsonstring = File.ReadAllText("data/types/custom/formular_dict.json");
                formular_dict = JsonConvert.DeserializeObject<Dictionary<string, List<List<string>>>>(jsonstring);
            }
       
            Dictionary<string, Dictionary<string, int>> sheet_formular_number_dict = new Dictionary<string, Dictionary<string, int>>();
            Dictionary<string, Dictionary<string, List<int>>> sheet_formular_dict = new Dictionary<string, Dictionary<string, List<int>>>();
            Dictionary<string, int> content_type_number_dic = new Dictionary<string, int>();
            Dictionary<string, List<int>> content_type_dic = new Dictionary<string, List<int>>();

            int duplicated_formular_number = 0;
            int sheet_duplicated_formular_number = 0;
            int negative_number = 0;
            int error_type_number = 0;

            int all_height = 0;
            int all_width = 0;
            string content_type = string.Empty;
            foreach(var custom_dv_info in custom_dv_infos){
                if(custom_dv_info.Height < 0){
                    negative_number += 1;
                    continue;
                }
                // content_type = getCustomContentType(custom_dv_info.ID);
                // if(content_type == "Error"){
                //     error_type_number += 1;
                //     continue;
                // }
                all_height += custom_dv_info.Height;
                all_width += custom_dv_info.Width;
                string sheet_name = custom_dv_info.FileName + "/" + custom_dv_info.SheetName;
                if(!sheet_formular_number_dict.ContainsKey(sheet_name)){
                    sheet_formular_number_dict.Add(sheet_name, new Dictionary<string, int>{});
                    sheet_formular_dict.Add(sheet_name, new Dictionary<string, List<int>>{});
                }
                if(!sheet_formular_number_dict[sheet_name].ContainsKey(custom_dv_info.Value)){
                    List<int> new_list = new List<int>{};
                    sheet_formular_number_dict[sheet_name].Add(custom_dv_info.Value, 0);
                    sheet_formular_dict[sheet_name].Add(custom_dv_info.Value, new_list);
                    sheet_duplicated_formular_number += 1;
                }
                if(!formular_dict.ContainsKey(custom_dv_info.Value)){
                    List<List<string>> new_list = new List<List<string>>{};
                    formular_dict.Add(custom_dv_info.Value,new_list);
                    duplicated_formular_number += 1;
                }
                if(!formular_number_dict.ContainsKey(custom_dv_info.Value)){
                    formular_number_dict.Add(custom_dv_info.Value, 0);
                }
                if(!content_type_dic.ContainsKey(content_type)){
                    List<int> new_list = new List<int>{};
                    content_type_number_dic.Add(content_type, 0);
                    content_type_dic.Add(content_type, new_list); 
                }
                List<string> temp_list = new List<string>{};
                temp_list.Add("1");
                temp_list.Add(custom_dv_info.ID.ToString());
                temp_list.Add(custom_dv_info.Value);
                formular_dict[custom_dv_info.Value].Add(temp_list);
                formular_number_dict[custom_dv_info.Value] += 1;
                sheet_formular_dict[sheet_name][custom_dv_info.Value].Add(custom_dv_info.ID);
                sheet_formular_number_dict[sheet_name][custom_dv_info.Value] += 1;
                content_type_number_dic[content_type] += 1;
                content_type_dic[content_type].Add(custom_dv_info.ID);
            }
            
            custom_meta_info.Add("all_number", custom_dv_infos.Count());
            custom_meta_info.Add("negative_number", negative_number);
            custom_meta_info.Add("error_type_number", error_type_number);
  
            custom_meta_info.Add("mean_height", all_height / (custom_dv_infos.Count() - negative_number));
            custom_meta_info.Add("mean_width", all_width / (custom_dv_infos.Count() - negative_number));
            custom_meta_info.Add("sheet_duplicated_formular_number", sheet_duplicated_formular_number);
            custom_meta_info.Add("duplicated_formular_number", duplicated_formular_number);
            custom_meta_info.Add("content_type_number_dic", content_type_number_dic);
            custom_meta_info.Add("formular_number_dict", formular_number_dict);
            custom_meta_info.Add("sheet_formular_number_dict", sheet_formular_number_dict);
            
            custom_meta_info.Add("content_type_dic", content_type_dic);
            custom_meta_info.Add("formular_dict", formular_dict);
            custom_meta_info.Add("sheet_formular_dict", sheet_formular_dict);
            


            saveAsJson(custom_meta_info, "data/types/custom/custom_meta_info.json");
            saveAsJson(formular_number_dict, "data/types/custom/formular_number_dict.json");
            saveAsJson(formular_dict, "data/types/custom/formular_dict.json");
        }


        public void getBoundary(XLAllowedValues type){
            Dictionary<string, object> boundary_meta_info = new Dictionary<string, object>();
            Dictionary<XLOperator, int> operator_number = new Dictionary<XLOperator, int>();
            Dictionary<XLOperator, List<int>> operator_list = new Dictionary<XLOperator, List<int>>();
            var dv_list_file_name = string.Empty;
            var save_path = string.Empty;
            if(type == XLAllowedValues.WholeNumber){
                dv_list_file_name = "data/types/boundary/int_list.json";
                save_path = "data/types/boundary/int_meta_info.json";
            } else if(type == XLAllowedValues.Time){
                dv_list_file_name = "data/types/boundary/time_list.json";  
                save_path = "data/types/boundary/time_meta_info.json";  
            } else if(type == XLAllowedValues.TextLength){
                dv_list_file_name = "data/types/boundary/text_len_list.json";  
                save_path = "data/types/boundary/text_len_meta_info.json";  
            } else if(type == XLAllowedValues.Decimal){
                dv_list_file_name = "data/types/boundary/float_list.json";    
                save_path = "data/types/boundary/float_meta_info.json";
            } else if(type == XLAllowedValues.Date){
                dv_list_file_name = "data/types/boundary/date_list.json";    
                save_path = "data/types/boundary/date_meta_info.json";
            } else{
                Console.WriteLine("the type is not correct!");
                return;
            }

            if(!File.Exists(dv_list_file_name)){
                Console.WriteLine("the dv list file not exists. Please run getMetaInfo first!");
            } else{
                string jsonstring = File.ReadAllText(dv_list_file_name);
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }

            int negatve_number = 0;
            int all_height = 0;
            int all_width = 0;

            int fail_min_number = 0;
            int fail_max_number = 0;
            int without_max_number = 0;

            double all_min_number = 0;
            double all_max_number = 0;
            int min_count = 0;
            int max_count = 0;
            double min_min = 0;
            double min_max = 0;
            double max_min = 10000000;
            double max_max = 0;
            foreach(DVInfo dv_info in dv_infos){
                if(dv_info.Height < 0){
                    negatve_number += 1;

                    continue;
                }
                if(!operator_number.ContainsKey(dv_info.Operator)){
                    operator_number.Add(dv_info.Operator, 0);
                    List<int> new_list = new List<int>{};
                    operator_list.Add(dv_info.Operator, new_list);
                }
                operator_number[dv_info.Operator] += 1;
                operator_list[dv_info.Operator].Add(dv_info.ID);

                all_height += dv_info.Height;
                all_width += dv_info.Width;
                double min_value = 0;
                double max_value = 0;
                if(double.TryParse(dv_info.MinValue, out min_value) == true){
                    all_min_number += min_value;
                    min_count += 1;
                    if(min_value < min_min){
                        min_min = min_value;
                    }
                    if(min_value > min_max){
                        min_max = min_value;
                    }
                } else{
                    fail_min_number += 1;
                }
                if(dv_info.MinValue == ""){
                    without_max_number += 1;
                } else{
                    if(double.TryParse(dv_info.MaxValue, out max_value) == true){
                        all_max_number += max_value;
                        max_count+= 1;
                        if(max_value < max_min){
                            max_min = max_value;
                        }
                        if(max_value > max_max){
                            max_max = max_value;
                        }
                    } else{
                        fail_max_number += 1;
                    }
                }
            }
            boundary_meta_info.Add("all_number", dv_infos.Count());
            boundary_meta_info.Add("negative_number", negatve_number);
            boundary_meta_info.Add("mean_height", all_height / (dv_infos.Count() - negatve_number));
            boundary_meta_info.Add("mean_width", all_width / (dv_infos.Count() - negatve_number));
            boundary_meta_info.Add("operator_number", operator_number);
            boundary_meta_info.Add("operator_list", operator_list);

            boundary_meta_info.Add("min_mean", all_min_number / min_count);
            boundary_meta_info.Add("max_mean", all_max_number / max_count);
            boundary_meta_info.Add("min_count", min_count);
            boundary_meta_info.Add("max_count", max_count);
            boundary_meta_info.Add("min_min", min_min);
            boundary_meta_info.Add("min_max", min_max);
            boundary_meta_info.Add("max_min", max_min);
            boundary_meta_info.Add("max_max", max_max);
            boundary_meta_info.Add("fail_min_number", fail_min_number);
            boundary_meta_info.Add("fail_max_number", fail_max_number);
            boundary_meta_info.Add("without_max_number", without_max_number);
            saveAsJson(boundary_meta_info, save_path);
        }


        public string getCustomContentType(int id){
            List<Tuple> custom_content;
            if(!File.Exists(content_path + id.ToString()+ ".json")){
                Console.WriteLine("no custom content. May be it not extracted！");
                return "Error";
            } else{
                string jsonstring = File.ReadAllText(content_path + id.ToString()+ ".json");
                custom_content = JsonConvert.DeserializeObject<List<Tuple>>(jsonstring);
            }

            List<XLDataType> type_list = new List<XLDataType>{};
            foreach(Tuple tup in custom_content){
                var is_in_type_list = false;
                foreach(XLDataType item in type_list){
                    if(tup.DataType == item){
                        is_in_type_list = true;
                        break;
                    }
                }
                if(is_in_type_list == false){
                    type_list.Add(tup.DataType);
                }
            }

            if(type_list.Count() > 1){
                return "Rich";
            } else if(type_list.Count() == 1){
                return type_list[0].ToString();
            } else{
                return "Error";
            }
        }

        public void sheetfile(){
            // string second_res_path = "/datadrive-2/data/fortune500_test/second_res_shift_epoch3_c2/";
            string second_res_path = "/datadrive/data_fortune500/crawled_xlsx_fortune500/";
            DirectoryInfo origin_root_path_info = new DirectoryInfo(second_res_path);
            FileInfo[] sub_file_names = origin_root_path_info.GetFiles();
            Dictionary<string, List<string>> filename2sheets = new Dictionary<string, List<string>>{};
            Dictionary<string, List<string>> sheet2filenames = new Dictionary<string, List<string>>{};
            Dictionary<string, int> sheet2count = new Dictionary<string, int>{};
            var count = 0;
            List<string> wblist = new List<string>{};
            foreach(var file_info in sub_file_names){
                // string wbname = file_info.Name.Split("---")[0];
                string wbname = file_info.Name;
                if(!wblist.Contains(wbname)){
                    wblist.Add(wbname);
                }
            }
            foreach(var wb_name in wblist){
                count += 1;
                Console.WriteLine(count.ToString() + " " + wb_name+" "+wblist.Count().ToString());
                string file_path = "/datadrive/data_fortune500/crawled_xlsx_fortune500/"+ wb_name;

                try{
                    var workbook = new XLWorkbook(file_path);
                    foreach(var ws in workbook.Worksheets.ToArray()){
                        if (!sheet2count.ContainsKey(ws.Name)){
                            sheet2count.Add(ws.Name, 1);
                        } else{
                            sheet2count[ws.Name] += 1;
                        }

                        if(!sheet2filenames.ContainsKey(ws.Name)){
                            List<string> temp = new List<string>();
                            sheet2filenames.Add(ws.Name,temp);
                        }
                        sheet2filenames[ws.Name].Add(wb_name);
                        if(!filename2sheets.ContainsKey(wb_name)){
                            List<string> sheetname = new List<string>();
                            filename2sheets.Add(wb_name,sheetname);
                        }
                        filename2sheets[wb_name].Add(ws.Name);
                    }
                }catch{
                    continue;
                }
               
            }
            saveAsJson(filename2sheets,  "/datadrive-2/data/fortune500_test/filename2sheets.json");
            saveAsJson(sheet2filenames,  "/datadrive-2/data/fortune500_test/sheet2filenames.json");
            saveAsJson(sheet2count,  "/datadrive-2/data/fortune500_test/sheet2num.json");
        }
        public void countAllSheet(){
            string origin_root_path = "../../data/";
            DirectoryInfo origin_root_path_info = new DirectoryInfo(origin_root_path);
            DirectoryInfo[] sub_path_infos = origin_root_path_info.GetDirectories();;
            var file_number = 0;
            var sheet_number = 0;
            Dictionary<string, int> result = new Dictionary<string, int>();
            Dictionary<string, List<string>> result1 = new Dictionary<string, List<string>>();
            int file_count = 1;
            if(File.Exists("all_sheetname_2_num.json")){
                string jsonstring = File.ReadAllText("all_sheetname_2_num.json");
                result = JsonConvert.DeserializeObject<Dictionary<string, int>>(jsonstring);
                string jsonstring1 = File.ReadAllText("all_file_sheet.json");
                result1 = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring1);
            }
            foreach(var sub_path_info in sub_path_infos){
                FileInfo[] sub_file_names = sub_path_info.GetFiles();
                Console.Write("file numbers: ");
                Console.WriteLine(sub_file_names.Count());
                
                file_number += sub_file_names.Count();
                foreach(var file_info in sub_file_names){
                    string file_path = origin_root_path + sub_path_info.Name + '/' + file_info.Name;
                    Console.WriteLine("file: "+file_count.ToString() + "," + sub_file_names.Count().ToString());
                    file_count += 1;
                    int sheet_count = 1;
                    if(file_count==5405||file_count==731||file_count==9197){
                        // Console.WriteLine("OCc");
                        continue;
                    }
                    try{
                        if(result1.ContainsKey(file_path)){
                            continue;
                        }
                        // using(var workbook = new XLWorkbook(file_path)){
                        var workbook = new XLWorkbook(file_path);
                        sheet_number += workbook.Worksheets.ToArray().Count();
                        foreach(var ws in workbook.Worksheets.ToArray()){
                            Console.WriteLine("    sheet: "+sheet_count.ToString() + "," + workbook.Worksheets.ToArray().Count().ToString());
                            if(!result.ContainsKey(ws.Name)){
                                result.Add(ws.Name,0);
                            }
                            result[ws.Name]+=1;
                            if(!result1.ContainsKey(file_path)){
                                List<string> sheetname = new List<string>();
                                result1.Add(file_path,sheetname);
                            }
                            result1[file_path].Add(ws.Name);
                            sheet_count += 1;
                            // }
                        }
                    }catch{
                        Console.WriteLine("error");
                        continue;
                    }
                    // if(file_count%100==0){
                    Console.WriteLine("Saving.......");
                    saveAsJson(result, "all_sheetname_2_num.json");
                    saveAsJson(result1, "all_file_sheet.json");
                    // }
                    Console.WriteLine("end");
                }
            }
            Console.WriteLine("file_number:" + file_number.ToString());
            Console.WriteLine("sheet_number:" + sheet_number.ToString());
            saveAsJson(result, "all_sheetname_2_num.json");
            saveAsJson(result1, "all_file_sheet.json");
        }
        public void getFunctionCount(){
            Dictionary<string, int> formular_number_dict;
            if(!File.Exists("data/types/custom/formular_number_dict.json")){
                Console.WriteLine("no formular number dictionalry exists. Please run getDictionary<string, int> first！");
                return;
            } else{
                string jsonstring = File.ReadAllText("data/types/custom/formular_number_dict.json");
                formular_number_dict = JsonConvert.DeserializeObject<Dictionary<string, int>>(jsonstring);
            }
            Dictionary<string, int> all_function_number = new Dictionary<string, int>();
            Dictionary<string, int> duplicated_function_number = new Dictionary<string, int>();
            
            foreach (KeyValuePair<string, int> kvp in formular_number_dict)
            {
                var formular = kvp.Key;
                if(formular != "AND(J7<>\"\",OR(J7=\"n/a\",AND(J7>=0,J7<=100)))"){
                    continue;
                } 
                Console.WriteLine(formular);
                var root_node = ExcelFormulaParser.Parse(formular);
                var all_nodes = ExcelFormulaParser.AllNodes(root_node);
                List<string> functions = new List<string>();
                foreach(var node in all_nodes){
                    Console.WriteLine("#############");
                    Console.WriteLine(node.Tag);
                    Console.WriteLine(node.Term);
                    Console.WriteLine(node.Token);
                }
                foreach(var func in functions){
                    if(!all_function_number.ContainsKey(func)){
                        all_function_number[func] = 0;
                        duplicated_function_number[func] = 0;
                    }
                    all_function_number[func] += kvp.Value;
                    duplicated_function_number[func] += 1;
                }
                break;
            }
            // saveAsJson(all_function_number, "all_func_number.json");
            // saveAsJson(duplicated_function_number, "duplicated_function_number.json");
        }

        public void getFunctionFiles(string function){
            List<DVInfo> custom_list = new List<DVInfo>{};
            if(!File.Exists("data/types/custom/custom_list.json")){
                Console.WriteLine("no formular number dictionalry exists. Please run getDictionary<string, int> first！");
                return;
            } else{
                string jsonstring = File.ReadAllText("data/types/custom/custom_list.json");
                custom_list = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
            Dictionary<string, int> all_function_number = new Dictionary<string, int>();
            Dictionary<string, int> duplicated_function_number = new Dictionary<string, int>();
            List<DVInfo> result = new List<DVInfo>{};
            foreach (DVInfo dv_info in custom_list)
            {
                var formular = dv_info.MinValue;
                var root_node = ExcelFormulaParser.Parse(formular);
                var all_nodes = ExcelFormulaParser.AllNodes(root_node);
    
                foreach(var node in all_nodes){
                    if(ExcelFormulaParser.IsFunction(node)){
                        string _function = ExcelFormulaParser.GetFunction(node);
                        if(_function == function){
                            result.Add(dv_info);
                        }
                    }
                }
            }
            foreach(var dv_info in result){
                Console.Write("ID: ");
                Console.Write(dv_info.ID);
                Console.Write("\tformula: ");
                Console.Write(dv_info.MinValue);
                Console.Write("\tFileName: ");
                Console.Write(dv_info.FileName);
                Console.Write("\tSheetName: ");
                Console.Write(dv_info.SheetName);
                Console.Write("\tRange: ");
                Console.WriteLine(dv_info.RangeAddress);
            }
        }
        public void filtTextLength(){
            var dv_file_list = new List<string>{};
            dv_file_list.Add("/datadrive/data/dvinfoWithRef.json");
            dv_file_list.Add("/datadrive/data/dvinfoWithRef1.json");
            dv_file_list.Add("/datadrive/data/dvinfoWithRef2.json");
            dv_file_list.Add("/datadrive/data/dvinfoWithRef3.json");
        
            var count = 0;
            foreach(var filename in dv_file_list){
                List<DVInfo> result = new List<DVInfo>{};
                List<DVInfo> dv_infos_batches = new List<DVInfo>{};
                if(File.Exists(@filename)){
                    string jsonstring = File.ReadAllText(filename);
                    dv_infos_batches = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
                }
            
                foreach(DVInfo dvinfo in dv_infos_batches){
                    if(dvinfo.Type == XLAllowedValues.TextLength){
                        result.Add(dvinfo);
                    }
                }
                saveAsJson(result, "data/types/boundary/text_len_dv_" + count.ToString() + ".json");
                count += 1;
            }

            

        }

        public void filtList(){
            var dv_file_list = new List<string>{};
            dv_file_list.Add("/datadrive/data/dvinfoWithRef.json");
            dv_file_list.Add("/datadrive/data/dvinfoWithRef1.json");
            dv_file_list.Add("/datadrive/data/dvinfoWithRef2.json");
            dv_file_list.Add("/datadrive/data/dvinfoWithRef3.json");
            
            var count = 0;
            var batch = 0;
            List<DVInfo> result = new List<DVInfo>{};
            foreach(var filename in dv_file_list){
                List<DVInfo> dv_infos_batches = new List<DVInfo>{};
                if(File.Exists(@filename)){
                    string jsonstring = File.ReadAllText(filename);
                    dv_infos_batches = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
                }
            
                foreach(DVInfo dvinfo in dv_infos_batches){
                    if(dvinfo.Type == XLAllowedValues.List){
                        result.Add(dvinfo);
                        count += 1;
                    }
                    if(count == 1000){
                        batch += 1;
                        count = 0;
                        saveAsJson(result, "data/types/boundary/list_" + batch.ToString()+ ".json");
                        result = new List<DVInfo>{};
                        if(batch == 10){break;}
                    }
                }
                break;
            }
        }
        public void filtCustomFunction(){
            var dv_file_list = new List<string>{};
            dv_file_list.Add("/datadrive/data/dvinfoWithRef.json");
            dv_file_list.Add("/datadrive/data/dvinfoWithRef1.json");
            dv_file_list.Add("/datadrive/data/dvinfoWithRef2.json");
            dv_file_list.Add("/datadrive/data/dvinfoWithRef3.json");
            

            List<DVInfo> COUNTIF = new List<DVInfo>{};
            List<DVInfo> SEARCH = new List<DVInfo>{};
            List<DVInfo> VLOOKUP = new List<DVInfo>{};
            List<DVInfo> MONTH = new List<DVInfo>{};
            List<DVInfo> MATCH = new List<DVInfo>{};
            List<DVInfo> LEN = new List<DVInfo>{};
            List<DVInfo> HEX2DEC = new List<DVInfo>{};
            List<DVInfo> CHAR = new List<DVInfo>{};
            List<DVInfo> ROUND = new List<DVInfo>{};
            List<DVInfo> CODE = new List<DVInfo>{};
            List<DVInfo> MID = new List<DVInfo>{};
            List<DVInfo> UPPER = new List<DVInfo>{};
            List<DVInfo> ROW = new List<DVInfo>{};
            List<DVInfo> INDIRECT = new List<DVInfo>{};
            List<DVInfo> RIGHT = new List<DVInfo>{};
            List<DVInfo> SUMPRODUCT = new List<DVInfo>{};
            List<DVInfo> LEFT = new List<DVInfo>{};
            List<DVInfo> VALUE = new List<DVInfo>{};
            List<DVInfo> TRUNC = new List<DVInfo>{};
            List<DVInfo> EXACT = new List<DVInfo>{};
            List<DVInfo> GTE = new List<DVInfo>{};
            List<DVInfo> SUBSTITUTE = new List<DVInfo>{};
            List<DVInfo> DATEVALUE = new List<DVInfo>{};
            List<DVInfo> LOWER = new List<DVInfo>{};
            List<DVInfo> ISNONTEXT = new List<DVInfo>{};
            List<DVInfo> T = new List<DVInfo>{};
            List<DVInfo> FIND = new List<DVInfo>{};
            List<DVInfo> LENB = new List<DVInfo>{};
            List<DVInfo> ASC = new List<DVInfo>{};
            List<DVInfo> COUNTA = new List<DVInfo>{};
            List<DVInfo> WEEKDAY = new List<DVInfo>{};
            List<DVInfo> ISEMAIL = new List<DVInfo>{};
            List<DVInfo> ROUNDDOWN = new List<DVInfo>{};
            List<DVInfo> TRIM = new List<DVInfo>{};

            foreach(var filename in dv_file_list){
                List<DVInfo> dv_infos_batches = new List<DVInfo>{};
                if(File.Exists(@filename)){
                    string jsonstring = File.ReadAllText(filename);
                    dv_infos_batches = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
                }
            
                foreach(DVInfo dvinfo in dv_infos_batches){
                    if(dvinfo.Type == XLAllowedValues.Custom){
                        string formular = dvinfo.Value;
                        if(formular == ""){
                            continue;
                        } 
                        Console.WriteLine(formular);
                        var root_node = ExcelFormulaParser.Parse(formular);
                        var all_nodes = ExcelFormulaParser.AllNodes(root_node);
                        List<string> functions = new List<string>();
                    
                        foreach(var node in all_nodes){
                            if(ExcelFormulaParser.IsFunction(node)){
                                string function = ExcelFormulaParser.GetFunction(node);
                                var is_in_functions = false;
                                foreach(var func in functions){
                                    if(func == function){
                                        is_in_functions = true;
                                        break;
                                    }
                                }
                                if(!is_in_functions){
                                    functions.Add(function);
                                    if(function == "COUNTIF"){
                                        COUNTIF.Add(dvinfo);
                                    } else if(function == "SEARCH"){
                                        SEARCH.Add(dvinfo);
                                    } else if(function == "VLOOKUP"){
                                        VLOOKUP.Add(dvinfo);
                                    } else if(function == "MONTH"){
                                        MONTH.Add(dvinfo);
                                    } else if(function == "MATCH"){
                                        MATCH.Add(dvinfo);
                                    } else if(function == "LEN"){
                                        LEN.Add(dvinfo);
                                    } else if(function == "HEX2DEC"){
                                        HEX2DEC.Add(dvinfo);
                                    } else if(function == "CHAR"){
                                        CHAR.Add(dvinfo);
                                    } else if(function == "ROUND"){
                                        ROUND.Add(dvinfo);
                                    } else if(function == "CODE"){
                                        CODE.Add(dvinfo);
                                    } else if(function == "MID"){
                                        MID.Add(dvinfo);
                                    } else if(function == "UPPER"){
                                        UPPER.Add(dvinfo);
                                    } else if(function == "ROW"){
                                        ROW.Add(dvinfo);
                                    } else if(function == "INDIRECT"){
                                        INDIRECT.Add(dvinfo);
                                    } else if(function == "RIGHT"){
                                        RIGHT.Add(dvinfo);
                                    } else if(function == "SUMPRODUCT"){
                                        SUMPRODUCT.Add(dvinfo);
                                    } else if(function == "LEFT"){
                                        LEFT.Add(dvinfo);
                                    } else if(function == "VALUE"){
                                        VALUE.Add(dvinfo);
                                    } else if(function == "TRUNC"){
                                        TRUNC.Add(dvinfo);
                                    } else if(function == "EXACT"){
                                        EXACT.Add(dvinfo);
                                    } else if(function == "GTE"){
                                        GTE.Add(dvinfo);
                                    } else if(function == "SUBSTITUTE"){
                                        SUBSTITUTE.Add(dvinfo);
                                    } else if(function == "DATEVALUE"){
                                        DATEVALUE.Add(dvinfo);
                                    } else if(function == "LOWER"){
                                        LOWER.Add(dvinfo);
                                    } else if(function == "ISNONTEXT"){
                                        ISNONTEXT.Add(dvinfo);
                                    } else if(function == "T"){
                                        T.Add(dvinfo);
                                    } else if(function == "FIND"){
                                        FIND.Add(dvinfo);
                                    } else if(function == "LENB"){
                                        LENB.Add(dvinfo);
                                    } else if(function == "ASC"){
                                        ASC.Add(dvinfo);
                                    } else if(function == "COUNTA"){
                                        COUNTA.Add(dvinfo);
                                    } else if(function == "WEEKDAY"){
                                        WEEKDAY.Add(dvinfo);
                                    } else if(function == "ISEMAIL"){
                                        ISEMAIL.Add(dvinfo);
                                    } else if(function == "ROUNDDOWN"){
                                        ROUNDDOWN.Add(dvinfo);
                                    } else if(function == "TRIM"){
                                        TRIM.Add(dvinfo);
                                    } 
                                }
                            }
                        }
                    }
                }
            }
            saveAsJson(COUNTIF, "data/types/custom/COUNTIF.json");
            saveAsJson(SEARCH, "data/types/custom/SEARCH.json");
            saveAsJson(VLOOKUP, "data/types/custom/VLOOKUP.json");
            saveAsJson(MONTH, "data/types/custom/MONTH.json");
            saveAsJson(MATCH, "data/types/custom/MATCH.json");
            saveAsJson(LEN, "data/types/custom/LEN.json");
            saveAsJson(HEX2DEC, "data/types/custom/HEX2DEC.json");
            saveAsJson(CHAR, "data/types/custom/CHAR.json");
            saveAsJson(ROUND, "data/types/custom/ROUND.json");
            saveAsJson(CODE, "data/types/custom/CODE.json");
            saveAsJson(MID, "data/types/custom/MID.json");
            saveAsJson(UPPER, "data/types/custom/UPPER.json");
            saveAsJson(ROW, "data/types/custom/ROW.json");
            saveAsJson(INDIRECT, "data/types/custom/INDIRECT.json");
            saveAsJson(RIGHT, "data/types/custom/RIGHT.json");
            saveAsJson(SUMPRODUCT, "data/types/custom/SUMPRODUCT.json");
            saveAsJson(LEFT, "data/types/custom/LEFT.json");
            saveAsJson(VALUE, "data/types/custom/VALUE.json");
            saveAsJson(TRUNC, "data/types/custom/TRUNC.json");
            saveAsJson(EXACT, "data/types/custom/EXACT.json");
            saveAsJson(GTE, "data/types/custom/GTE.json");
            saveAsJson(SUBSTITUTE, "data/types/custom/SUBSTITUTE.json");
            saveAsJson(DATEVALUE, "data/types/custom/DATEVALUE.json");
            saveAsJson(LOWER, "data/types/custom/LOWER.json");
            saveAsJson(ISNONTEXT, "data/types/custom/ISNONTEXT.json");
            saveAsJson(T, "data/types/custom/T.json");
            saveAsJson(FIND, "data/types/custom/FIND.json");
            saveAsJson(LENB, "data/types/custom/LENB.json");
            saveAsJson(ASC, "data/types/custom/ASC.json");
            saveAsJson(COUNTA, "data/types/custom/COUNTA.json");
            saveAsJson(WEEKDAY, "data/types/custom/WEEKDAY.json");
            saveAsJson(ISEMAIL, "data/types/custom/ISEMAIL.json");
            saveAsJson(ROUNDDOWN, "data/types/custom/ROUNDDOWN.json");
            saveAsJson(TRIM, "data/types/custom/TRIM.json");
        }

        public void merge_batch_list(){
            string jsonstring = File.ReadAllText("continous_batch_0.json");
            List<DVInfo> list0 = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            string jsonstring1 = File.ReadAllText("continous_batch_0.json");
            List<DVInfo> list1 = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring1);
            string jsonstring2 = File.ReadAllText("continous_batch_0.json");
            List<DVInfo> list2= JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring2);
            string jsonstring3 = File.ReadAllText("continous_batch_0.json");
            List<DVInfo> list3 = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring3);

            List<DVInfo> result = new List<DVInfo>{};
            var count = 0;
            foreach(var dvinfo in list0){
                result.Add(dvinfo);
                count += 1;
            }
            foreach(var dvinfo in list1){
                result.Add(dvinfo);
                count += 1;
            }
            foreach(var dvinfo in list2){
                result.Add(dvinfo);
                count += 1;
            }
            foreach(var dvinfo in list3){
                result.Add(dvinfo);
                count += 1;
            }
            Console.WriteLine(count);
            saveAsJson(result, "40List.json");
        }

        public void count_list_type(){
            List<DVInfo> dv_infos;
            List<DVInfo> result = new List<DVInfo>{};
            var count = 0;
            var error_count = 0;
            var all = 0;
            if(!File.Exists("/datadrive/data/dvinfoWithRef.json")){
                Console.WriteLine("no file path！");
                return;
            } else{
                string jsonstring = File.ReadAllText("/datadrive/data/dvinfoWithRef.json");
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
            foreach(var dvinfo in dv_infos){
                if(dvinfo.Type != XLAllowedValues.List){
                    continue;
                }
                all += 1;
                if(dvinfo.refers.Type == 1){
                    count += 1;
                    continue;
                } else if(dvinfo.refers.Type == 2){
                    List<ReferItem> new_refer_list = new List<ReferItem>{};
                    var index = 0;
                    foreach(var refer_item in dvinfo.refers.List){
                        if(index == 0){
                            var temp_item = refer_item;
                            temp_item.Value = refer_item.Value.ToString().Substring(1).Trim();
                            new_refer_list.Add(temp_item);
                        } else if(index == dvinfo.refers.List.Count()-1){
                            var temp_item = refer_item;
                            temp_item.Value = refer_item.Value.ToString().Substring(0, refer_item.Value.ToString().Count()-1).Trim();
                            new_refer_list.Add(temp_item);
                        } else{
                            refer_item.Value = refer_item.Value.ToString().Trim();
                            new_refer_list.Add(refer_item);
                        }
                        index += 1;
                    }
                    dvinfo.refers.List = new_refer_list;
                }
                var is_error = false;
                foreach(var content in dvinfo.content){
                    var is_in_ref = false;
                    // Console.WriteLine(",,,,,,");
                    // Console.WriteLine(content.Value.ToString());
                    // Console.WriteLine(content.Value().Count());

                    foreach(var refer in dvinfo.refers.List){
                        // Console.WriteLine("xxxx");
                        // Console.WriteLine(refer.Value);
                        // Console.WriteLine(content.Value.ToString() == refer.Value);
                        if(content.Value.ToString() == refer.Value.ToString()){
                            // Console.WriteLine(content.Value.ToString());
                            is_in_ref = true;
                            break;
                        }
                    }
                    if(is_in_ref == false){
                        is_error = true;
                        break;
                    }
                }
                if(is_error == true){
                    error_count += 1;
                } 
            }


            if(!File.Exists("/datadrive/data/dvinfoWithRef1.json")){
                Console.WriteLine("no file path！");
                return;
            } else{
                string jsonstring = File.ReadAllText("/datadrive/data/dvinfoWithRef1.json");
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
            foreach(var dvinfo in dv_infos){
                if(dvinfo.Type != XLAllowedValues.List){
                    continue;
                }
                all += 1;
                if(dvinfo.refers.Type == 1){
                    count += 1;
                    continue;
                } else if(dvinfo.refers.Type == 2){
                    List<ReferItem> new_refer_list = new List<ReferItem>{};
                    var index = 0;
                    foreach(var refer_item in dvinfo.refers.List){
                        if(index == 0){
                            var temp_item = refer_item;
                            temp_item.Value = refer_item.Value.ToString().Substring(1).Trim();
                            new_refer_list.Add(temp_item);
                        } else if(index == dvinfo.refers.List.Count()-1){
                            var temp_item = refer_item;
                            temp_item.Value = refer_item.Value.ToString().Substring(0, refer_item.Value.ToString().Count()-1).Trim();
                            new_refer_list.Add(temp_item);
                        } else{
                            refer_item.Value = refer_item.Value.ToString().Trim();
                            new_refer_list.Add(refer_item);
                        }
                        index += 1;
                    }
                    dvinfo.refers.List = new_refer_list;
                }
                var is_error = false;
                foreach(var content in dvinfo.content){
                    var is_in_ref = false;
                    // Console.WriteLine(",,,,,,");
                    // Console.WriteLine(content.Value.ToString());
                    // Console.WriteLine(content.Value().Count());

                    foreach(var refer in dvinfo.refers.List){
                        // Console.WriteLine("xxxx");
                        // Console.WriteLine(refer.Value);
                        // Console.WriteLine(content.Value.ToString() == refer.Value);
                        if(content.Value.ToString() == refer.Value.ToString()){
                            // Console.WriteLine(content.Value.ToString());
                            is_in_ref = true;
                            break;
                        }
                    }
                    if(is_in_ref == false){
                        is_error = true;
                        break;
                    }
                }
                if(is_error == true){
                    error_count += 1;
                } 
            }

            if(!File.Exists("/datadrive/data/dvinfoWithRef2.json")){
                Console.WriteLine("no file path！");
                return;
            } else{
                string jsonstring = File.ReadAllText("/datadrive/data/dvinfoWithRef2.json");
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
            foreach(var dvinfo in dv_infos){
                if(dvinfo.Type != XLAllowedValues.List){
                    continue;
                }
                all += 1;
                if(dvinfo.refers.Type == 1){
                    count += 1;
                    continue;
                } else if(dvinfo.refers.Type == 2){
                    List<ReferItem> new_refer_list = new List<ReferItem>{};
                    var index = 0;
                    foreach(var refer_item in dvinfo.refers.List){
                        if(index == 0){
                            var temp_item = refer_item;
                            temp_item.Value = refer_item.Value.ToString().Substring(1).Trim();
                            new_refer_list.Add(temp_item);
                        } else if(index == dvinfo.refers.List.Count()-1){
                            var temp_item = refer_item;
                            temp_item.Value = refer_item.Value.ToString().Substring(0, refer_item.Value.ToString().Count()-1).Trim();
                            new_refer_list.Add(temp_item);
                        } else{
                            refer_item.Value = refer_item.Value.ToString().Trim();
                            new_refer_list.Add(refer_item);
                        }
                        index += 1;
                    }
                    dvinfo.refers.List = new_refer_list;
                }
                var is_error = false;
                foreach(var content in dvinfo.content){
                    var is_in_ref = false;
                    // Console.WriteLine(",,,,,,");
                    // Console.WriteLine(content.Value.ToString());
                    // Console.WriteLine(content.Value().Count());

                    foreach(var refer in dvinfo.refers.List){
                        // Console.WriteLine("xxxx");
                        // Console.WriteLine(refer.Value);
                        // Console.WriteLine(content.Value.ToString() == refer.Value);
                        if(content.Value.ToString() == refer.Value.ToString()){
                            // Console.WriteLine(content.Value.ToString());
                            is_in_ref = true;
                            break;
                        }
                    }
                    if(is_in_ref == false){
                        is_error = true;
                        break;
                    }
                }
                if(is_error == true){
                    error_count += 1;
                } 
            }


            if(!File.Exists("/datadrive/data/dvinfoWithRef3.json")){
                Console.WriteLine("no file path！");
                return;
            } else{
                string jsonstring = File.ReadAllText("/datadrive/data/dvinfoWithRef3.json");
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
            foreach(var dvinfo in dv_infos){
                if(dvinfo.Type != XLAllowedValues.List){
                    continue;
                }
                all += 1;
                if(dvinfo.refers.Type == 1){
                    count += 1;
                    continue;
                } else if(dvinfo.refers.Type == 2){
                    List<ReferItem> new_refer_list = new List<ReferItem>{};
                    var index = 0;
                    foreach(var refer_item in dvinfo.refers.List){
                        if(index == 0){
                            var temp_item = refer_item;
                            temp_item.Value = refer_item.Value.ToString().Substring(1).Trim();
                            new_refer_list.Add(temp_item);
                        } else if(index == dvinfo.refers.List.Count()-1){
                            var temp_item = refer_item;
                            temp_item.Value = refer_item.Value.ToString().Substring(0, refer_item.Value.ToString().Count()-1).Trim();
                            new_refer_list.Add(temp_item);
                        } else{
                            refer_item.Value = refer_item.Value.ToString().Trim();
                            new_refer_list.Add(refer_item);
                        }
                        index += 1;
                    }
                    dvinfo.refers.List = new_refer_list;
                }
                var is_error = false;
                foreach(var content in dvinfo.content){
                    var is_in_ref = false;
                    // Console.WriteLine(",,,,,,");
                    // Console.WriteLine(content.Value.ToString());
                    // Console.WriteLine(content.Value().Count());

                    foreach(var refer in dvinfo.refers.List){
                        // Console.WriteLine("xxxx");
                        // Console.WriteLine(refer.Value);
                        // Console.WriteLine(content.Value.ToString() == refer.Value);
                        if(content.Value.ToString() == refer.Value.ToString()){
                            // Console.WriteLine(content.Value.ToString());
                            is_in_ref = true;
                            break;
                        }
                    }
                    if(is_in_ref == false){
                        is_error = true;
                        break;
                    }
                }
                if(is_error == true){
                    error_count += 1;
                } 
            }

            Console.WriteLine(count);
            Console.WriteLine(error_count);
            Console.WriteLine(all);
        }

        public void count_filter_list_type(){
            List<DVInfo> dv_infos;
            var all = 0;
            List<string> sheet_name_list = new List<string>{};
            List<string> file_name_list = new List<string>{};
            if(!File.Exists("../extractDV-master/new_data_1/dvinfoWithRef.json")){
                Console.WriteLine("no file path！");
                return;
            } else{
                string jsonstring = File.ReadAllText("../extractDV-master/new_data_1/dvinfoWithRef.json");
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }


            foreach(DVInfo dvinfo in dv_infos){
                all += 1;
                var is_in_sheet_list = false;
                var is_in_file_list = false;
                // Console.WriteLine(dvinfo.FileName);
                // Console.WriteLine(dvinfo.SheetName);
                foreach(var sheetname in sheet_name_list){
                    if(sheetname == dvinfo.FileName+":"+dvinfo.SheetName){
                        is_in_sheet_list = true;
                        break;
                    }
                }
                foreach(var filename in file_name_list){
                    if(filename == dvinfo.FileName){
                        is_in_file_list = true;
                        // Console.WriteLine("############################");
                        // Console.WriteLine("is in");
                        // Console.WriteLine(filename);
                        // Console.WriteLine(dvinfo.SheetName);
                        // Console.WriteLine("############################");
                        break;
                    }
                }
                if(is_in_sheet_list == false){
                    sheet_name_list.Add(dvinfo.FileName+":"+dvinfo.SheetName);
                }
                if(is_in_file_list == false){
                    // Console.WriteLine("add filename");
                    file_name_list.Add(dvinfo.FileName);
                }
            }
            // Console.WriteLine(file_name_list.Count());

            if(!File.Exists("../extractDV-master/new_data_1/dvinfoWithRef1.json")){
                Console.WriteLine("no file path！");
                return;
            } else{
                string jsonstring = File.ReadAllText("../extractDV-master/new_data_1/dvinfoWithRef1.json");
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
            foreach(var dvinfo in dv_infos){
                all += 1;
                var is_in_sheet_list = false;
                var is_in_file_list = false;
                foreach(var sheetname in sheet_name_list){
                    if(sheetname == dvinfo.FileName+":"+dvinfo.SheetName){
                        is_in_sheet_list = true;
                        break;
                    }
                }
                foreach(var filename in file_name_list){
                    if(filename == dvinfo.FileName){
                        is_in_file_list = true;
                        break;
                    }
                }
                if(is_in_sheet_list == false){
                    sheet_name_list.Add(dvinfo.FileName+":"+dvinfo.SheetName);
                }
                if(is_in_file_list == false){
                    file_name_list.Add(dvinfo.FileName);
                }
            }
            // Console.WriteLine(file_name_list.Count());

            if(!File.Exists("../extractDV-master/new_data_1/dvinfoWithRef2.json")){
                Console.WriteLine("no file path！");
                return;
            } else{
                string jsonstring = File.ReadAllText("../extractDV-master/new_data_1/dvinfoWithRef2.json");
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
            foreach(var dvinfo in dv_infos){
                all += 1;
                var is_in_sheet_list = false;
                var is_in_file_list = false;
                foreach(var sheetname in sheet_name_list){
                    if(sheetname == dvinfo.FileName+":"+dvinfo.SheetName){
                        is_in_sheet_list = true;
                        break;
                    }
                }
                foreach(var filename in file_name_list){
                    if(filename == dvinfo.FileName){
                        is_in_file_list = true;
                        break;
                    }
                }
                if(is_in_sheet_list == false){
                    sheet_name_list.Add(dvinfo.FileName+":"+dvinfo.SheetName);
                }
                if(is_in_file_list == false){
                    file_name_list.Add(dvinfo.FileName);
                }
            }
            // Console.WriteLine(file_name_list.Count());

            if(!File.Exists("../extractDV-master/new_data_1/dvinfoWithRef3.json")){
                Console.WriteLine("no file path！");
                return;
            } else{
                string jsonstring = File.ReadAllText("../extractDV-master/new_data_1/dvinfoWithRef3.json");
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
            foreach(var dvinfo in dv_infos){
                all += 1;
                var is_in_sheet_list = false;
                var is_in_file_list = false;
                foreach(var sheetname in sheet_name_list){
                    if(sheetname == dvinfo.FileName+":"+dvinfo.SheetName){
                        is_in_sheet_list = true;
                        break;
                    }
                }
                foreach(var filename in file_name_list){
                    if(filename == dvinfo.FileName){
                        is_in_file_list = true;
                        break;
                    }
                }
                if(is_in_sheet_list == false){
                    sheet_name_list.Add(dvinfo.FileName+":"+dvinfo.SheetName);
                }
                if(is_in_file_list == false){
                    file_name_list.Add(dvinfo.FileName);
                }
            }
            // Console.WriteLine(count);
            // Console.WriteLine(error_count);
            Console.WriteLine(all);
            Console.WriteLine(sheet_name_list.Count());
            // foreach(var sheet_name in file_name_list){
            //     Console.WriteLine(sheet_name);
            // }
            Console.WriteLine(file_name_list.Count());
        }

        public void sample_fifty_dvs(){
            string jsonstring = File.ReadAllText("new_continous_batch_0.json");
            dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            Random rand = new Random();
            var dv_infos_shuffle = dv_infos.OrderBy(c => rand.Next()).Select(c => c).ToList();
            List<string> filelist = new List<string>{};
            List<DVInfo> result = new List<DVInfo>{};
            foreach(var dvinfo in dv_infos_shuffle){
                var is_in_file_list = false;
                foreach(var filename in filelist){
                    if(filename == dvinfo.FileName){
                        is_in_file_list = true;
                        break;
                    }
                }
                if(is_in_file_list == true){
                    continue;
                }
                XLWorkbook workbook = new XLWorkbook();
                var worksheet = workbook.AddWorksheet();
                IXLRange range_a = worksheet.Range(dvinfo.RangeAddress);
                int left_top_y_a = range_a.RangeAddress.FirstAddress.ColumnNumber;
                int right_bottom_y_a = range_a.RangeAddress.LastAddress.ColumnNumber;
                int left_top_x_a = range_a.RangeAddress.FirstAddress.RowNumber;
                int right_bottom_x_a = range_a.RangeAddress.LastAddress.RowNumber;

                if(left_top_x_a == right_bottom_x_a && left_top_y_a == right_bottom_y_a){ // one cell range
                    result.Add(dvinfo);
                }
                if(result.Count() == 50){
                    break;
                }
            }
            saveAsJson(result, "sample50singlecell.json");
        }

        public void check_distinct_refer_list(){
            string jsonstring = File.ReadAllText("/datadrive/data/dvinfoWithRef.json");
            dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            var is_distinct = 0;
            var has_same_value = 0;
            foreach(var dvinfo in dv_infos){
                List<ReferItem> distinct_refer_list = new List<ReferItem>{};
                foreach(ReferItem referitem in dvinfo.refers.List){
                    var is_in_distinct_list = false;
                    foreach(ReferItem distinct_refer_item in distinct_refer_list){
                        if(referitem.DataType == distinct_refer_item.DataType){
                            if(referitem.Value == distinct_refer_item.Value){
                                is_in_distinct_list = true;
                                break;
                            }
                        }
                    }
                    if(is_in_distinct_list == false){
                        distinct_refer_list.Add(referitem);
                    }
                }
                if(distinct_refer_list.Count() == dvinfo.refers.List.Count()){
                    is_distinct += 1;
                }
                else{
                    has_same_value += 1;
                }
            }
            Console.WriteLine(is_distinct);
            Console.WriteLine(has_same_value);
        }

        public void count_type2_list_number_and_coverage(){
            List<float> coverage_list = new List<float>();
             for(int i=0; i<=3;i++){
                string jsonstring = File.ReadAllText("new_continous_batch_"+i.ToString()+".json");
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
                foreach(var dvinfo in dv_infos){
                    if(dvinfo.refers.Type == 2){
                        List<string> distinct_content = new List<string>();
                        foreach(var content_item in dvinfo.content){
                            var is_in_distinct_list = false;
                            foreach(var distinct_content_item in distinct_content){
                                if(content_item.Value.ToString() == distinct_content_item){
                                    is_in_distinct_list = true;
                                    break;
                                }
                            }
                            // Console.WriteLine(is_in_distinct_list);
                            if(is_in_distinct_list == false){
                                Console.WriteLine(content_item);
                                distinct_content.Add(content_item.Value.ToString());
                            }
                        }

                        List<string> distinct_refer = new List<string>();
                        foreach(var content_item in dvinfo.refers.List){
                            var is_in_distinct_list = false;
                            foreach(var distinct_content_item in distinct_refer){
                                if(content_item.Value.ToString() == distinct_content_item){
                                    // Console.WriteLine("******");
                                    // Console.WriteLine(content_item);
                                    // Console.WriteLine(distinct_content_item);
                                    is_in_distinct_list = true;
                                    break;
                                }
                            }
                            // Console.WriteLine(is_in_distinct_list);
                            if(is_in_distinct_list == false){
                                distinct_refer.Add(content_item.Value.ToString());
                            }
                        }
                        Console.WriteLine("############");
                        Console.WriteLine(distinct_content.Count());
                        Console.WriteLine(distinct_refer.Count());
                        Console.WriteLine("############");
                        coverage_list.Add(distinct_content.Count()/distinct_refer.Count());
                        Console.WriteLine(dvinfo.refers.List.Count());
                    }
                }
             }
            Console.WriteLine(coverage_list.Count());
            float mean = 0;
            foreach(var item in coverage_list){
                mean += item;
            }
            Console.WriteLine(mean/coverage_list.Count());
        }
        // public bool is_same_list(List<ReferItem> refer_list_a,  List<ReferItem> refer_list_b){
        //     if(refer_list_a.Count() != refer_list_b.Count()){
        //         return false;
        //     }
        //     foreach(ReferItem item_a in refer_list_a){
        //         foreach(ReferItem itemb in refer_list_b){
        //             if(item_a.DataType )
        //         }
        //     }
        // }
        // public void duplicateErrorList(){
        //     List<string> add_name = new List<string>{};
        //     foreach(Dictionary<string, string> error in error_dict_list){
        //         var is_duplicated = false;
        //         foreach(var name in add_name){
        //             if(error["FileName"] == name){
        //                 is_duplicated = true;
        //                 break;
        //             }
        //         }
        //         if(is_duplicated == true){
        //             error_dict_list.remove(error);
        //         }
        //         else{
        //             add_name.Add(error["Filename"]);
        //         }
        //     }
        // }
        public bool is_same_refer_list(List<ReferItem> a_refer_list,  List<ReferItem> b_refer_list){
            if(a_refer_list.Count() != b_refer_list.Count()){
                return false;
            }
            Dictionary<string, int> a_list_dic = new Dictionary<string, int>();
            Dictionary<string, int> b_list_dic = new Dictionary<string, int>();

            foreach(var refer_item in a_refer_list){
                if(!a_list_dic.ContainsKey(refer_item.Value.ToString())){
                    a_list_dic[refer_item.Value.ToString()] = 0;
                }
                a_list_dic[refer_item.Value.ToString()] += 1;
            }
            foreach(var refer_item in b_refer_list){
                if(!b_list_dic.ContainsKey(refer_item.Value.ToString())){
                    b_list_dic[refer_item.Value.ToString()] = 0;
                }
                b_list_dic[refer_item.Value.ToString()] += 1;
            }

            foreach(var a_key in a_list_dic.Keys){
                var is_in_b = false;
                foreach(var b_key in b_list_dic.Keys){
                    if(a_key == b_key){
                        is_in_b = true;
                        if(a_list_dic[a_key] != b_list_dic[b_key]){
                            return false;
                        }
                        break;
                    }
                }
                if(is_in_b == false){
                    return false;
                }
            }
            return true;
        }
        public Dictionary<string, int> get_refer_list_dic(List<ReferItem> a_refer_list){
            Dictionary<string, int> a_list_dic = new Dictionary<string, int>();
            foreach(var refer_item in a_refer_list){
                if(!a_list_dic.ContainsKey(refer_item.Value.ToString())){
                    a_list_dic[refer_item.Value.ToString()] = 0;
                }
                a_list_dic[refer_item.Value.ToString()] += 1;
            }
            return a_list_dic;
        }
        public void count_global_refer_list(){
            Dictionary<List<ReferItem>, int> global_list_dic = new Dictionary<List<ReferItem>, int>(); 
            Dictionary<int, int> result_list_dic = new Dictionary<int, int>(); 
            Dictionary<int, List<ReferItem>> refer_dictionary = new Dictionary<int, List<ReferItem>>();
            var all_id = 1;
            for(int i=0; i<=3;i++){
                string jsonstring = File.ReadAllText("new_continous_batch_"+i.ToString()+".json");
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
                foreach(var dvinfo in dv_infos){
                    var refer_list = dvinfo.refers.List;
                    var is_in_dic = false;
                    foreach(int refer_id in refer_dictionary.Keys){
                        var refer_key = refer_dictionary[refer_id];
                        if(is_same_refer_list(refer_list, refer_key)){
                            is_in_dic = true;
                            result_list_dic[refer_id] += 1;
                        }
                    }
                    if(is_in_dic == false){
                        result_list_dic[all_id] = 0;
                        result_list_dic[all_id] += 1;
                        refer_dictionary[all_id] = refer_list;
                        all_id += 1;
                    }
                }
            }
            saveAsJson(result_list_dic, "global_refer_list_number.json");
            saveAsJson(refer_dictionary, "refer_dictionary.json");
        }

        public void count_error_type(){
            Dictionary<int, int> error_type_number_dic = new Dictionary<int, int>(); 
            var all_number = 0;
            for(int i=0; i<=3;i++){
                string jsonstring = File.ReadAllText("error_new_filter_list_batch_"+i.ToString()+".json");
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
                all_number += dv_infos.Count();
                foreach(var dvinfo in dv_infos){
                    if(dvinfo.refers.Type == 2){
                        if(!error_type_number_dic.ContainsKey(2)){
                            error_type_number_dic[2] = 0;
                        }
                        error_type_number_dic[2] += 1;
                    }
                    else{
                        if(dvinfo.Value.Contains(":")){
                            if(!error_type_number_dic.ContainsKey(0)){
                                error_type_number_dic[0] = 0;
                            }
                            error_type_number_dic[0] += 1;
                        }
                        else{
                            if(!error_type_number_dic.ContainsKey(1)){
                                error_type_number_dic[1] = 0;
                            }
                            error_type_number_dic[1] += 1;
                        }
                    }

                }
            }
            Console.WriteLine(all_number);
            saveAsJson(error_type_number_dic, "error_type_number.json");
        }

         public void analyze_formula_fortune500(){
            // string jsonstring1 = File.ReadAllText("../analyze-dv-1/top10_domain_filenames.json");
            // string jsonstring1 = File.ReadAllText("../analyze-dv-1/middle10_domain_filenames.json");
            string jsonstring1 = File.ReadAllText("../analyze-dv-1/small_data_set/small_data_set_workbook.json");
            List<string> sampled_file = JsonConvert.DeserializeObject<List<string>>(jsonstring1);
            // string jsonstring2 = File.ReadAllText("origin_middle10domain/saved_filesheet.json");
            // List<string> saved_sampled_file = JsonConvert.DeserializeObject<List<string>>(jsonstring2);
            Dictionary<string, List<SimpleCellFormula>> result = new Dictionary<string, List<SimpleCellFormula>>();
            int count = 0;
            int batch_id = 58;
            foreach(var fname in sampled_file){
                try{
                    count += 1;
                    Console.WriteLine(count.ToString()+"/"+sampled_file.Count().ToString());
                    string source_filename = "data_set/xls_data_set/" + fname;
                    bool need_continue = false;
                    // foreach(var saved_filesheet in saved_sampled_file){
                    //     string filename = saved_filesheet.Split("---")[0];
                    //     // Console.WriteLine("filename:"+ filename);
                    //     // Console.WriteLine("fname:"+ fname);
                    //     if(filename == fname){
                    //         need_continue = true;
                    //         break;
                    //     }
                    // }
                    // if(need_continue){
                    //     continue;
                    // }
                    var workbook = new XLWorkbook(source_filename);
                    var worksheets = workbook.Worksheets.ToArray();
                    
                    foreach(var sheet in worksheets){
                        string filesheet = fname + "---" + sheet;    
                        string sheetname = sheet.Name;
                        try{
                            int firstcolumn = sheet.FirstColumnUsed().ColumnNumber();
                            int lastcolumn = sheet.LastColumnUsed().ColumnNumber();
                            int firstrow = sheet.FirstRowUsed().RowNumber();
                            int lastrow = sheet.LastRowUsed().RowNumber();
                            List<SimpleCellFormula> forlist = new List<SimpleCellFormula>();
                            result.Add(filesheet, forlist);
                            Console.WriteLine("result add!");
                            Console.WriteLine("result lengh:" + result.Count());
                            for(int row =firstrow; row <= lastrow; row++){
                                for(int col = firstcolumn; col <= lastcolumn; col++){
                        
                                    var cell = sheet.Cell(row, col);
                                    if(cell.FormulaA1.Length==0){
                                        continue;
                                    }
                                
                            
                                    SimpleCellFormula formula = new SimpleCellFormula();
                                    formula.row = row;
                                    formula.column = col;
                                    formula.formulaR1C1 = cell.FormulaR1C1;
                                    
                                
                                    result[filesheet].Add(formula);
                                }
                            }
                        }
                        catch{
                            Console.WriteLine("error:no column or row.");
                            continue;
                        }
                    }
                }catch{
                    Console.WriteLine("Error1");
                    continue;
                }
                if(result.Count() >= 500){
                        saveAsJson(result, "../data_set/formula_data_set/origin_data_formulas_"+batch_id.ToString()+".json");
                        // saveAsJson(result, "Formulas_20000sheets_recheck_"+batch_id.ToString()+".json");
                        batch_id += 1;
                        result = new Dictionary<string, List<SimpleCellFormula>>();
                    }
            }
            saveAsJson(result, "../data_set/formula_data_set/origin_data_formulas_"+batch_id.ToString()+".json");
            // saveAsJson(result, "Formulas_20000sheets_recheck_"+batch_id.ToString()+".json");
        }
        
        public void get_cell_coverage(){
            string origin_root_path = "../../Data/UnzipData/";
            DirectoryInfo origin_root_path_info = new DirectoryInfo(origin_root_path);
            DirectoryInfo[] sub_path_infos = origin_root_path_info.GetDirectories();;
            var file_number = 0;
            var sheet_number = 0;
            int zero_dv_cells = 134697;
            int zero_all_cells = 265617408;
 
            foreach(var sub_path_info in sub_path_infos){
                FileInfo[] sub_file_names = sub_path_info.GetFiles();
                Console.Write("file numbers: ");
                Console.WriteLine(sub_file_names.Count());
                file_number += sub_file_names.Count();
                int now_file = 0;
                foreach(var file_info in sub_file_names){
                    // filenumber += 1;
                    now_file += 1;
                    if(now_file <= 15){
                        continue;
                    }
                    string file_path = origin_root_path + sub_path_info.Name + '/' + file_info.Name;
                    using(var workbook = new XLWorkbook(file_path)){
                        sheet_number += workbook.Worksheets.ToArray().Count();
                        Console.WriteLine("now file:"+now_file.ToString()+'/' + file_number.ToString());
                        int now_sheet = 0;
                        foreach(var sheet in workbook.Worksheets.ToArray()){
                            now_sheet +=1;
                            int all_cell = sheet.Rows().Cells().Count();
                            int dv_cells = 0;
                            Console.WriteLine("now sheet:"+now_sheet.ToString()+'/' + workbook.Worksheets.ToArray().Count().ToString());
                            foreach(var dv in sheet.DataValidations){
                                int one_dv_cells = 0;
                                foreach(var range in dv.Ranges){
                                    one_dv_cells += range.Cells().Count();
                                }
                                dv_cells += one_dv_cells;
                            }
                            zero_dv_cells += dv_cells;
                            zero_all_cells += all_cell;
                    
                        }
                    }
                    
                    Console.WriteLine("zero_dv_cells:" + zero_dv_cells.ToString());
                    Console.WriteLine("zero_all_cells:" + zero_all_cells.ToString());
                }
                break;
            }
            Console.WriteLine("file_number:" + file_number.ToString());
            // Console.WriteLine("sheet_number:" + sheet_number.ToString());
            Console.WriteLine("zero_dv_cells:"+zero_dv_cells.ToString());
            Console.WriteLine("zero_all_cells:" + zero_all_cells.ToString());
        }
        static void Main(string[] args)
        {
            // 调用 analyze_formula_fortune500 方法
            analyze_formula_fortune500();
            // 可选：等待控制台按键后退出程序
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }
}