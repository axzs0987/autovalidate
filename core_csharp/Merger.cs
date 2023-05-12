using System;
using ClosedXML.Excel;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using XLParser;

namespace AnalyzeDV
{
    class MergeResult{
        public bool is_merge;
        public List<DVInfo> result_list;
    }
    class Merger{
        public Analyzer analyzer = new Analyzer();
        public int is_consecutive_range(DVInfo dv_a, DVInfo dv_b){
            XLWorkbook workbook = new XLWorkbook();
            var worksheet = workbook.AddWorksheet();
            IXLRange range_a = worksheet.Range(dv_a.RangeAddress);
            IXLRange range_b = worksheet.Range(dv_b.RangeAddress);
            int left_top_y_a = range_a.RangeAddress.FirstAddress.ColumnNumber;
            int right_bottom_y_a = range_a.RangeAddress.LastAddress.ColumnNumber;
            int left_top_x_a = range_a.RangeAddress.FirstAddress.RowNumber;
            int right_bottom_x_a = range_a.RangeAddress.LastAddress.RowNumber;

            int left_top_y_b = range_b.RangeAddress.FirstAddress.ColumnNumber;
            int right_bottom_y_b = range_b.RangeAddress.LastAddress.ColumnNumber;
            int left_top_x_b = range_b.RangeAddress.FirstAddress.RowNumber;
            int right_bottom_x_b = range_b.RangeAddress.LastAddress.RowNumber;
            
            var column_row_matrix_a = 0;
            if(left_top_y_a == right_bottom_y_a){
                column_row_matrix_a = 0;
            } else if(left_top_x_a == right_bottom_x_a){
                column_row_matrix_a = 1;
            } else{
                column_row_matrix_a = 2;
            }

            var column_row_matrix_b = 0;
            if(left_top_y_b == right_bottom_y_b){
                column_row_matrix_b = 0;
            } else if(left_top_x_b == right_bottom_x_b){
                column_row_matrix_b = 1;
            } else{
                column_row_matrix_b = 2;
            }

            if(column_row_matrix_a != column_row_matrix_b){
                return -1;
            } else if(column_row_matrix_a == 0){ // column
                if(left_top_y_a == left_top_y_b && (left_top_x_a - right_bottom_x_b <= 1 || left_top_x_b - right_bottom_x_a <= 1)){
                    return column_row_matrix_b;
                }
            } else if(column_row_matrix_a == 1){ // row
                if(left_top_x_a == left_top_x_b && (left_top_y_a - right_bottom_y_b <= 1 || left_top_y_b - right_bottom_y_a <= 1)){
                    return column_row_matrix_b;
                }
            } else if(column_row_matrix_a == 2){
                if(right_bottom_x_a == right_bottom_x_b && left_top_x_a == left_top_x_b){ // matrixes with same rows, column need be consecutive
                    if(left_top_y_a - right_bottom_y_b <= 1 || left_top_y_b - right_bottom_y_a <= 1){
                        return 2;
                    }
                }
                if(right_bottom_y_a == right_bottom_y_b && left_top_y_a == left_top_y_b){ // matrixes with same columns, row need be consecutive
                    if(left_top_x_a - right_bottom_x_b <= 1 || left_top_x_b - right_bottom_x_a <= 1){
                        return 3;
                    }
                }
            }
            return -1;
        }

        public bool is_same_dv(DVInfo dv_a, DVInfo dv_b){
            if(dv_a.Type != dv_b.Type){
                return false;
            } if(dv_a.Operator != dv_b.Operator){
                return false;
            } if(dv_a.MinValue != dv_b.MinValue || dv_a.MaxValue != dv_b.MaxValue || dv_a.Value != dv_b.Value){
                return false;
            } if(dv_a.FileName != dv_b.FileName || dv_a.SheetName != dv_b.SheetName){
                return false;
            } if (dv_a.InputTitle != dv_b.InputTitle || dv_a.InputMessage != dv_b.InputMessage || dv_a.ErrorTitle != dv_b.ErrorTitle || dv_a.ErrorMessage != dv_b.ErrorMessage || dv_a.ErrorStyle != dv_b.ErrorStyle){
                return false;
            }
            return true;
        }
        public bool is_same_dvinfo(DVInfo dv_a, DVInfo dv_b){
            if(dv_a.Type != dv_b.Type){
                return false;
            } if(dv_a.Operator != dv_b.Operator){
                return false;
            } if(dv_a.MinValue != dv_b.MinValue || dv_a.MaxValue != dv_b.MaxValue || dv_a.Value != dv_b.Value){
                return false;
            } if (dv_a.InputTitle != dv_b.InputTitle || dv_a.InputMessage != dv_b.InputMessage || dv_a.ErrorTitle != dv_b.ErrorTitle || dv_a.ErrorMessage != dv_b.ErrorMessage || dv_a.ErrorStyle != dv_b.ErrorStyle){
                return false;
            }
            return true;
        }
        public void get_same_dictinfo(){
            List<DVInfo> all_dv_infos = new List<DVInfo>{};
            string jsonstring = File.ReadAllText("continous_batch_0.json");
            var one_dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            foreach(var dvinfo in one_dv_infos){
                all_dv_infos.Add(dvinfo);
            }

            jsonstring = File.ReadAllText("continous_batch_1.json");
            one_dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            foreach(var dvinfo in one_dv_infos){
                all_dv_infos.Add(dvinfo);
            }

            jsonstring = File.ReadAllText("continous_batch_2.json");
            one_dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            foreach(var dvinfo in one_dv_infos){
                all_dv_infos.Add(dvinfo);
            }

            jsonstring = File.ReadAllText("continous_batch_3.json");
            one_dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            foreach(var dvinfo in one_dv_infos){
                all_dv_infos.Add(dvinfo);
            }


            var count =0;
            List<List<DVInfo>> same_set = new List<List<DVInfo>>{};
            foreach(DVInfo dvinfo in all_dv_infos){
                var is_in_same_set = false;
                var set_index = 0;
                foreach(List<DVInfo> set_list in same_set){
                    if(is_same_dvinfo(set_list[0], dvinfo)){
                        same_set[set_index].Add(dvinfo);
                        is_in_same_set = true;
                        break;
                    }
                    set_index += 1;
                }
                if(is_in_same_set == false){
                    List<DVInfo> new_list = new List<DVInfo>{};
                    new_list.Add(dvinfo);
                    same_set.Add(new_list);
                    count += 1;
                    // Console.Write("set number:");
                    Console.WriteLine(count);
                }
            }
            Console.WriteLine(same_set.Count);
            // analyzer.saveAsJson(same_set, save_path);
        }

        public void get_same_dict(XLAllowedValues type,string filename, string save_path){
            List<DVInfo> all_dv_infos = new List<DVInfo>{};
            string jsonstring = File.ReadAllText(filename);
            var one_dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);

            var count =0;
            List<List<DVInfo>> same_set = new List<List<DVInfo>>{};
            foreach(DVInfo dvinfo in one_dv_infos){
                if(dvinfo.Type != type){
                    continue;
                }
                var is_in_same_set = false;
                var set_index = 0;
                foreach(List<DVInfo> set_list in same_set){
                    if(is_same_dvinfo(set_list[0], dvinfo)){
                        same_set[set_index].Add(dvinfo);
                        is_in_same_set = true;
                        break;
                    }
                    set_index += 1;
                }
                if(is_in_same_set == false){
                    List<DVInfo> new_list = new List<DVInfo>{};
                    new_list.Add(dvinfo);
                    same_set.Add(new_list);
                    count += 1;
                    // Console.Write("set number:");
                    Console.WriteLine(count);
                }
            }
            // Console.WriteLine(same_set.Count);
            analyzer.saveAsJson(same_set, save_path);
        }
        public DVInfo merge(DVInfo dv_a, DVInfo dv_b, int consecutive_type){
            DVInfo result = dv_a;
            XLWorkbook workbook = new XLWorkbook();
            var worksheet = workbook.AddWorksheet();
            IXLRange range_a = worksheet.Range(dv_a.RangeAddress);
            IXLRange range_b = worksheet.Range(dv_b.RangeAddress);
            int left_top_y_a = range_a.RangeAddress.FirstAddress.ColumnNumber;
            int right_bottom_y_a = range_a.RangeAddress.LastAddress.ColumnNumber;
            int left_top_x_a = range_a.RangeAddress.FirstAddress.RowNumber;
            int right_bottom_x_a = range_a.RangeAddress.LastAddress.RowNumber;

            int left_top_y_b = range_b.RangeAddress.FirstAddress.ColumnNumber;
            int right_bottom_y_b = range_b.RangeAddress.LastAddress.ColumnNumber;
            int left_top_x_b = range_b.RangeAddress.FirstAddress.RowNumber;
            int right_bottom_x_b = range_b.RangeAddress.LastAddress.RowNumber;

            foreach(Tuple b_content in dv_b.content){
                result.content.Add(b_content);
            }
            foreach(var b_header in dv_b.header){
                result.header.Add(b_header);
            }
            if(consecutive_type == 0 || consecutive_type == 3){ // column
                result.Height = dv_a.Height + dv_b.Height;
                // Console.WriteLine(result.Height);
                int top = left_top_x_a;
                int bottom = right_bottom_x_a;
                if(left_top_x_a > left_top_x_b){
                    top = left_top_x_b;
                } 
                if(right_bottom_x_a < right_bottom_x_b){
                    top = left_top_x_b;
                } 
                IXLRange merged_range_0 = worksheet.Range(top, left_top_y_a, bottom, right_bottom_y_a);
                result.RangeAddress = merged_range_0.ToString();
                return result;
            } else if(consecutive_type == 1 || consecutive_type == 2) {// row{
                result.Width = dv_a.Width + dv_b.Width;
                int left = left_top_y_a;
                int right = right_bottom_y_a;
                if(left_top_y_a > left_top_y_b){
                    left = left_top_x_b;
                } 
                if(right_bottom_y_a < right_bottom_y_b){
                    right = left_top_x_b;
                } 
                IXLRange merged_range_1 = worksheet.Range(left_top_x_a, left, right_bottom_x_a, right);
                result.RangeAddress = merged_range_1.ToString();
                return result;
            } 

            Console.WriteLine("wrong consecutive_type");
            return result;
        }
        public MergeResult check_pair(List<DVInfo> dvinfos){
            MergeResult res = new MergeResult();
            List<DVInfo> result = new List<DVInfo>{};
            var is_merged = false;
            var merged_address = string.Empty;
            foreach(DVInfo a in dvinfos){
                var can_merge = false;
                if(is_merged == true && a.RangeAddress != merged_address){
                    result.Add(a);
                    continue;
                }
                if(is_merged == true){
                    continue;
                }
                foreach(DVInfo b in dvinfos){
                    if(a.RangeAddress == b.RangeAddress){
                        continue;
                    }
                    var consecutive_type = is_consecutive_range(a,b);
                    if(consecutive_type != -1){
                        // Console.WriteLine(a.ID);
                        DVInfo merged_dv = merge(a, b, consecutive_type);
                        // result.Add(merged_dv);
                        can_merge = true;
                        is_merged = true;
                        merged_address = b.RangeAddress;
                        break;
                    }
                }
                if(can_merge == false){
                    result.Add(a);
                }
            }
            res.is_merge = is_merged;
            res.result_list = result;
            return res;
        }
        public void get_continous(XLAllowedValues type, string same_dict_path, string save_file_path){
            List<List<DVInfo>> same_set;
            if(!File.Exists(@same_dict_path)){
                Console.WriteLine("no file path！");
                return;
            } else{
                string jsonstring = File.ReadAllText(@same_dict_path);
                same_set = JsonConvert.DeserializeObject<List<List<DVInfo>>>(jsonstring);
            }
            List<DVInfo> result = new List<DVInfo>{};
            foreach(var one_set in same_set){
                List<List<DVInfo>> same_set_with_range_check = new List<List<DVInfo>>{};
                var temp_same_one_set = new List<DVInfo>{};
                foreach(var dvinfo in one_set){
                    temp_same_one_set.Add(dvinfo);
                }

                MergeResult checked_result = check_pair(temp_same_one_set);
                temp_same_one_set = checked_result.result_list;
                while(checked_result.is_merge && temp_same_one_set.Count() > 1){
                    Console.Write("is_merge: ");
                    Console.WriteLine(checked_result.is_merge);
                    Console.Write("count: ");
                    Console.WriteLine(temp_same_one_set.Count());
                    checked_result = check_pair(temp_same_one_set);
                    temp_same_one_set = checked_result.result_list;
                }
                foreach(var dvinfo in temp_same_one_set){
                    result.Add(dvinfo);
                }
            }
            Console.Write("result count:");
            Console.WriteLine(result.Count());
            analyzer.saveAsJson(result, save_file_path);
        }

        public void recheck(string file_path, string save_path){
            List<DVInfo> dv_infos;
            List<DVInfo> new_dv_infos = new List<DVInfo>{};
            if(!File.Exists(@file_path)){
                Console.WriteLine("no file path！");
                return;
            } else{
                string jsonstring = File.ReadAllText(file_path);
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
            int count = 0;
            foreach(var dvinfo in dv_infos){
                Console.WriteLine(count.ToString()+"/" + dv_infos.Count().ToString());
                if(dvinfo.Type == XLAllowedValues.List){
                    try{
                        if(dvinfo.refers.Type != 2){
                            // Console.WriteLine(dvinfo.Value);
                            var is_range = true;
            
                            var workbook = new XLWorkbook(dvinfo.FileName);
                            var worksheet = workbook.Worksheet(dvinfo.SheetName);
                            IXLRange ixlrange = worksheet.Range(dvinfo.Value);
                            Refer res = new Refer();
                            List<ReferItem> res_list = new List<ReferItem>{};
                            for(int row = 1; row <= ixlrange.RowCount(); row++){
                                for(int col = 1; col <= ixlrange.ColumnCount(); col++){
                                    ReferItem temp_tuple = new ReferItem();
                                    // Console.WriteLine(ixlrange.Cell(row, col).Value.ToString());
                                    if(ixlrange.Cell(row, col).Value == ""){
                                        continue;
                                    }
                                    temp_tuple.Value = ixlrange.Cell(row, col).Value;
                                    temp_tuple.DataType = ixlrange.Cell(row, col).DataType;
                                    res_list.Add(temp_tuple);
                                }
                            }
                            res.Type = 0;
                            res.List = res_list;
                            if(res_list.Count() != 0){
                                dvinfo.refers = res;
                            }
                        }
                
                    
                        // else{
                        //     Dictionary<string, object> res1 = new Dictionary<string, object>();
                        //     List<Dictionary<string, object>> res_list1 = new List<Dictionary<string, object>>{};

                        //     // var worksheet = workbook.Worksheet(dvinfo.SheetName);
                        //     IXLNamedRange ixnamedrange = workbook.NamedRange(dvinfo.Value);

                            
                        //     // Console.WriteLine(ixnamedrange.Scope);
                        //     // Console.WriteLine(ixnamedrange.Name);
                        //     try{
                        //         Console.WriteLine(ixnamedrange.RefersTo);
                        //     }
                        //     catch{
                        //         Console.Write("error: ");
                        //         Console.WriteLine(dvinfo.Value);
                        //     }
                        
                        
                        //     // var worksheet = workbook.Worksheet(10);
                        //     // Console.WriteLine(worksheet.Name);
                        //     IXLRange ixlrange1 = ixnamedrange.Ranges.ToArray()[0];
                        //     for(int row = 1; row <= ixlrange1.RowCount(); row++){
                        //         for(int col = 1; col <= ixlrange1.ColumnCount(); col++){
                        //             ReferItem temp_tuple = new ReferItem();
                        //             Console.WriteLine(ixlrange1.Cell(row, col).Value);
                        //             if(ixlrange1.Cell(row, col).Value == ""){
                        //                 continue;
                        //             }
                        //             temp_tuple.Value = ixlrange1.Cell(row, col).Value;
                        //             temp_tuple.DataType = ixlrange1.Cell(row, col).DataType;
                        //             res_list.Add(temp_tuple);
                        //         }
                        //     }
                        // }

                        

                    }catch{
                        // Console.WriteLine("XXXXXXXXXXXXXXXXXXXXXXXXX");
                        Refer res = new Refer();
                        res.Type = 1;
                        dvinfo.refers = res;
            
                    }
                }
                new_dv_infos.Add(dvinfo);
                count += 1;
                // break;
            }
            analyzer.saveAsJson(new_dv_infos, save_path);
        }

        public void test_type_1(string file_path){
            List<DVInfo> dv_infos;
            List<DVInfo> result = new List<DVInfo>{};
            List<DVInfo> error_result = new List<DVInfo>{};
            if(!File.Exists(@file_path)){
                Console.WriteLine("no file path！");
                return;
            } else{
                string jsonstring = File.ReadAllText(file_path);
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
            foreach(var dvinfo in dv_infos){
                if(dvinfo.Type != XLAllowedValues.List){
                    continue;
                }
                if(dvinfo.refers.Type != 1){
                    continue;
                }
                // Console.WriteLine(dvinfo.FileName);
                // Console.WriteLine(dvinfo.SheetName);
                // Console.WriteLine(dvinfo.Value);
                // Console.WriteLine(dvinfo.RangeAddress);
                Dictionary<string, object> res = new Dictionary<string, object>();
                List<Dictionary<string, object>> res_list = new List<Dictionary<string, object>>{};
                var workbook = new XLWorkbook(dvinfo.FileName);

                // var worksheet = workbook.Worksheet(dvinfo.SheetName);
                IXLNamedRange ixnamedrange = workbook.NamedRange(dvinfo.Value);

                
                // Console.WriteLine(ixnamedrange.Scope);
                // Console.WriteLine(ixnamedrange.Name);
                try{
                    Console.WriteLine(ixnamedrange.RefersTo);
                }
                catch{
                    Console.Write("error: ");
                    Console.WriteLine(dvinfo.Value);
                }
            
            
                // var worksheet = workbook.Worksheet(10);
                // Console.WriteLine(worksheet.Name);
                IXLRange ixlrange = ixnamedrange.Ranges.ToArray()[0];
                for(var r=1;r<=ixlrange.RowCount();r++){
                    for(var c=1;c<=ixlrange.ColumnCount();c++){
                        Console.WriteLine(ixlrange.Cell(r,c).Value);
                    }
                }
                
                // // break;

                // Console.WriteLine(ixlrange.ColumnCount());
                // Console.WriteLine(ixlrange.RowCount());
                // Console.WriteLine(ixlrange.RangeAddress.FirstAddress.ColumnNumber);
                // Console.WriteLine(ixlrange.RangeAddress.LastAddress.ColumnNumber);
                // int left_top_y = ixlrange.RangeAddress.FirstAddress.ColumnNumber;
                // int right_bottom_y = ixlrange.RangeAddress.LastAddress.ColumnNumber;
                // int left_top_x = ixlrange.RangeAddress.FirstAddress.RowNumber;
                
                // int right_bottom_x = 16384;
                // if(ixlrange.RangeAddress.LastAddress.RowNumber < 16384){
                //     right_bottom_x = ixlrange.RangeAddress.LastAddress.RowNumber;
                // }
                // Console.WriteLine(ixlrange.RangeAddress.FirstAddress.RowNumber);
                // Console.WriteLine(ixlrange.RangeAddress.LastAddress.RowNumber);
                // for(int row = left_top_x; row <= right_bottom_x; row++){
                //     for(int col = left_top_y; col <= right_bottom_y; col++){
                //         Dictionary<string, object> temp_tuple = new Dictionary<string, object>();
                //         Console.WriteLine(worksheet.Cell(row, col).Value);
                //         if(worksheet.Cell(row, col).Value == ""){
                //             continue;
                //         }
                //         temp_tuple["Value"] = worksheet.Cell(row, col).Value;
                //         temp_tuple["DataType"] = worksheet.Cell(row, col).DataType;
                //         res_list.Add(temp_tuple);
                //         Console.WriteLine(temp_tuple["Value"]);
                //     }
                // }
                // res["Type"] = 0;
                // res["List"] = res_list;

                break;
            }
        }

        public void filt_list(string file_path, string save_file_path){
            List<DVInfo> dv_infos;
            List<DVInfo> result = new List<DVInfo>{};
            List<DVInfo> error_result = new List<DVInfo>{};
            if(!File.Exists(@file_path)){
                Console.WriteLine("no file path！");
                return;
            } else{
                string jsonstring = File.ReadAllText(file_path);
                dv_infos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
            foreach(var dvinfo in dv_infos){
                if(dvinfo.Type != XLAllowedValues.List){
                    continue;
                }
                if(dvinfo.refers.Type == 1){
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
                if(is_error == false){
                    result.Add(dvinfo);
                }
                else{
                    error_result.Add(dvinfo);
                }
                // break;
            }
            Console.Write("after filter count:");
            Console.WriteLine(result.Count());
            analyzer.saveAsJson(result, save_file_path);
            analyzer.saveAsJson(error_result, "error_"+ save_file_path);
        }
    }
}
