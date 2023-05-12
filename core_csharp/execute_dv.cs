using System;
using ClosedXML.Excel;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using XLParser;

namespace AnalyzeDV
{
    class Excuter{
        public void saveAsJson(object need_save_content, string file_name){
            string jsonData = JsonConvert.SerializeObject(need_save_content);
            File.WriteAllText(file_name, jsonData);
        }
        public List<int> get_runner_cell(CustomDVInfo dvinfo){
            List<int> result = new List<int>();
            int row = 10000;
            int column = 10000;
            while(row>=dvinfo.ltx && row<=dvinfo.rbx){
                row--;
            }
            while(column>=dvinfo.lty && column<=dvinfo.rby){
                column--;
            }
            bool is_same = true;
            while(is_same){
                bool has_same = false;
                // Console.WriteLine(dvinfo.ID);
                // Console.WriteLine(dvinfo.shift);
                foreach(var shift in dvinfo.shift){
                    if(shift.column==column && shift.row==row){
                        has_same=true;
                        break;
                    }
                }
                if(!has_same){
                    is_same=false;
                }else{
                    row--;
                }
            }

            result.Add(column);
            result.Add(row);
            return result;
        }

        public string change_formula(CustomDVInfo dvinfo, List<string> range_cell, int range_col, int range_row){
            var root_node = ExcelFormulaParser.Parse(dvinfo.Value);
            var all_nodes = ExcelFormulaParser.AllNodes(root_node);
            List<string> functions = new List<string>();

            List<Node> new_temp_content = new List<Node>();

            List<string> origin_ref = new List<string>();
            foreach(var node in all_nodes){
                if(node.Term.Name=="CellToken"){
                    origin_ref.Add(node.Token.ValueString);
                }
            }
            string formula = dvinfo.Value;
            // Console.WriteLine("origin_ref:"+origin_ref.Count().ToString());
            // Console.WriteLine("range_cell:"+range_cell.Count().ToString());
            List<string> temp_placeholder = new List<string>();
            
            for(var index=0;index<origin_ref.Count();index++){
                temp_placeholder.Add("temp_place_holder_"+index.ToString());
            }
            for(var index=0;index<origin_ref.Count();index++){
                // Console.WriteLine("origin_ref:"+origin_ref[index]);
                // Console.WriteLine("range_cell:"+range_cell[index]);
                if (formula.IndexOf(origin_ref[index]) > -1)
                {
                    formula = formula.Remove(formula.IndexOf(origin_ref[index]), origin_ref[index].Count()).Insert(formula.IndexOf(origin_ref[index]), temp_placeholder[index]);
                }
                // Console.WriteLine(formula);
            }
            for(var index=0;index<temp_placeholder.Count();index++){
                // Console.WriteLine("origin_ref:"+origin_ref[index]);
                // Console.WriteLine("range_cell:"+range_cell[index]);
                if (formula.IndexOf(temp_placeholder[index]) > -1)
                {
                    formula = formula.Remove(formula.IndexOf(temp_placeholder[index]), temp_placeholder[index].Count()).Insert(formula.IndexOf(temp_placeholder[index]), range_cell[index]);
                }
                // Console.WriteLine(formula);
            }
            return formula;
        }
        public int evaluate_one_dv(CustomDVInfo dvinfo, List<int> runner_index){
            int result = 1;
            using(var workbook = new XLWorkbook(dvinfo.FileName)){
                var worksheet = workbook.Worksheet(dvinfo.SheetName);
                var runner = worksheet.Cell(runner_index[1], runner_index[0]);

                int index=0;
                for(int range_col = dvinfo.lty; range_col<=dvinfo.rby;range_col++){
                    for(int range_row = dvinfo.ltx; range_row<=dvinfo.rbx; range_row++){
                        
                        var cell = worksheet.Cell(range_row,range_col);
                        if(cell.Value.ToString()==""){
                            index+=1;
                            continue;
                        }
                        List<string> one_range_cells = new List<string>();
                        foreach(var shift in dvinfo.shift){
                            string cell_token = "";
                            try{
                                var shifted_cell = worksheet.Cell(shift.cells[index].row, shift.cells[index].column);
                                // Console.WriteLine("shifted_cell:"+shift.cells[index].column.ToString()+","+shift.cells[index].row.ToString());
                                cell_token = shifted_cell.WorksheetColumn().ColumnLetter() + shifted_cell.WorksheetRow().RowNumber().ToString();
                            }catch{
                                cell_token = "ERROR";
                            }
                            
                            // Console.WriteLine(cell_token);
                            one_range_cells.Add(cell_token);
                        }
                        string changed_formula = change_formula(dvinfo, one_range_cells, range_col, range_row);
                        index+=1;
                        runner.FormulaA1 =changed_formula;

                        workbook.CalculateMode = ClosedXML.Excel.XLCalculateMode.Auto;
                        try{
                            Console.WriteLine(runner.Value);
                            // Console.WriteLine(runner.Value.ToString());
                        }catch{
                            // Console.WriteLine("run fail");
                            result = 2;
                            return result;
                        }
                        
                        if(runner.Value.ToString() != "True" && runner.Value.ToString() != "False" ){
                            
                            if(cell.Value.ToString() != runner.Value.ToString()){
                                // Console.WriteLine("False");
                                result = 0;
                            }
                            // else{
                            //     // Console.WriteLine("True");
                            // }
                        }else if(runner.Value.ToString()=="False"){
                            result = 0;
                            Console.WriteLine(changed_formula);
                            Console.WriteLine("range:"+range_col.ToString()+','+range_row.ToString());
                            // Console.WriteLine("False");
                        }
                    }
                }
            }
            return result;
        }

        public void test_evaluate(){
            List<CustomDVInfo> dvinfos;

            List<CustomDVInfo> sucdvinfos = new List<CustomDVInfo>();
            List<CustomDVInfo> faildvinfos = new List<CustomDVInfo>();
            List<CustomDVInfo> errordvinfos = new List<CustomDVInfo>();
            if(!File.Exists("data/types/custom/dedup_shifted_custom_info.json")){
                Console.WriteLine("no formular number dictionalry exists. Please run getDictionary<string, int> first！");
                return;
            } else{
                string jsonstring = File.ReadAllText("data/types/custom/dedup_shifted_custom_info.json");
                dvinfos = JsonConvert.DeserializeObject<List<CustomDVInfo>>(jsonstring);
            }
            // string jsonstring1 = File.ReadAllText("data/types/custom/execute_suc_dvinfos.json");
            // var suc_dvinfos = JsonConvert.DeserializeObject<List<CustomDVInfo>>(jsonstring1);
            // string jsonstring2 = File.ReadAllText("data/types/custom/execute_fail_dvinfos.json");
            // var fail_dvinfos = JsonConvert.DeserializeObject<List<CustomDVInfo>>(jsonstring2);
            // string jsonstring3 = File.ReadAllText("data/types/custom/execute_error_dvinfos.json");
            // var error_dvinfos = JsonConvert.DeserializeObject<List<CustomDVInfo>>(jsonstring2);
            
            int suc = 0;
            int fail=0;
            int error=0;

            // List<int> exist_dv = new List<int>();
            // foreach(var dvinfo in suc_dvinfos){
            //     sucdvinfos.Add(dvinfo);
            //     exist_dv.Add(dvinfo.ID);
            // }
            // foreach(var dvinfo in fail_dvinfos){
            //     faildvinfos.Add(dvinfo);
            //     exist_dv.Add(dvinfo.ID);
            // }
            // foreach(var dvinfo in error_dvinfos){
            //     errordvinfos.Add(dvinfo);
            //     exist_dv.Add(dvinfo.ID);
            // }
            Console.WriteLine("suc_:" + sucdvinfos.Count());
            Console.WriteLine("fail:" + faildvinfos.Count());
            Console.WriteLine("error:" + errordvinfos.Count());
            Console.WriteLine("dvinfos:"+dvinfos.Count());
            foreach(var dvinfo in dvinfos){
                // bool found=false;
                // foreach(var id_ in exist_dv){
                //     if(id_==dvinfo.ID){
                //         found=true;
                //         break;
                //     }
                // }
                // if(found==true){
                //     continue;
                // }
                // if(dvinfo.ID != 59705){
                //     continue;
                // }
                List<int> runner_index = get_runner_cell(dvinfo);
                int res = evaluate_one_dv(dvinfo, runner_index);
                if(res==1){
                    suc+=1;
                    sucdvinfos.Add(dvinfo);
                }
                if(res==0){
                    fail+=1;
                    faildvinfos.Add(dvinfo);
                }
                if(res==2){
                    error+=1;
                    errordvinfos.Add(dvinfo);
                }
                // break;
            }
            Console.WriteLine("suc:"+suc.ToString());
            Console.WriteLine("fail:"+fail.ToString());
            Console.WriteLine("error:"+error.ToString());
 
            saveAsJson(sucdvinfos, "data/types/custom/execute_suc_dvinfos.json");  
            saveAsJson(faildvinfos, "data/types/custom/execute_fail_dvinfos.json");   
            saveAsJson(errordvinfos, "data/types/custom/execute_error_dvinfos.json");            
        }
    }
}