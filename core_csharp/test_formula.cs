using System;
using ClosedXML.Excel;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using XLParser;

namespace AnalyzeDV
{
    class XMLDVInfo{
        public string range;
        public string value;
    }
    class Shift{
        public int column;
        public bool is_abs_column;
        public bool is_abs_row;
        public int row;
        public List<Cell> cells;
    }
    // class Shift{
    //     public List<int> shift_column;
    //     public List<int> shift_row;
    //     public List<List<Cell>> shifted_cell;
    //     public List<List<Cell>> abs_cell;
    //     public List<bool> is_abs_column;
    //     public List<bool> is_abs_row;
    // }
    class Cell{
        public int column;
        public int row;
    }
    class Node{
        public string Term;
        public string Token;
    }
    class Similarity{
        public int index1;
        public int index2;
        public double score;
    }
    class R1C1Template{
        public int id;
        public List<Node> node_list;
        public List<string> formulas;
    }
    class Template{
        public int id;
        public List<Node> node_list;
        public int number;
        public List<string> formulas_list;
        public List<string> dvid_list;
        public List<string> file_sheet_name_list;

        public List<Similarity> value_similarity_list;
        public List<Similarity> fill_color_similarity_list;
        public List<Similarity> font_color_similarity_list;
        public List<Similarity> height_similarity_list;
        public List<Similarity> width_similarity_list;
        public List<Similarity> type_similarity_list;
    }
    class TestFormula{
        public void id2sheetnames(){
            List<string> dedup_workbooks = new List<string>();
            Dictionary<int, List<string>> id2sheetnames = new Dictionary<int, List<string>>();
            Dictionary<int, List<string>> id2workbook = new Dictionary<int, List<string>>();
            Dictionary<int, List<long>> id2size = new Dictionary<int, List<long>>();

            // string workbook_path = "/datadrive/data_fortune500/crawled_xlsx_fortune500/";
            string workbook_path_list = "/datadrive/data/";
            DirectoryInfo root = new DirectoryInfo(workbook_path_list);
            // DirectoryInfo root = new DirectoryInfo(workbook_path);
            // FileInfo[] files=root.GetFiles();
            DirectoryInfo[] dicts=root.GetDirectories();
            int now_id = 1;
            foreach(var dict in dicts){
                FileInfo[] files=dict.GetFiles();
                Console.WriteLine("###############");
                Console.WriteLine(dict.Name);
                if(dict.Name.Count() != 3){
                    continue;
                }
                foreach(var fileinfo in files){
                    Console.WriteLine("xxxxxxxxx");
                    try{
                        Console.WriteLine("fileinfo:" + fileinfo.Name);
                        string file_name = fileinfo.Name;
                        string file_path = "/datadrive/data/" + dict.Name + "/" + file_name;
                        // string file_path = workbook_path + file_name;f
                        if(file_name.Contains("d3d3Lmdvb2RseS5jby5pbgkxMDQuMTguMzQuNjI=") | file_name.Contains("d3d3Lmdvb2RseS5jby5pbgkxMDQuMTguMzUuNjI=")){
                            continue;
                        }
                        XLWorkbook workbook = new XLWorkbook(file_path);
                        List<string> sheet_names = new List<string>();

                        foreach(var worksheet in workbook.Worksheets){
                            string sheet_name = worksheet.Name;
                            sheet_names.Add(sheet_name);
                        }
                        sheet_names.Sort();
                        bool found = false;
                        Console.WriteLine("start iter id....");
                        foreach(int i in id2sheetnames.Keys){

                            if(id2sheetnames[i].SequenceEqual(sheet_names)){
                                found =true;
                                id2workbook[i].Add(file_name);
                                if (!id2size[i].Contains(fileinfo.Length)){
                                    id2size[i].Add(fileinfo.Length);
                                    dedup_workbooks.Add(file_name);
                                    Console.WriteLine("1 add new size");
                                }   
                                break;                     
                            }
                        }
                        Console.WriteLine("found:"+ found.ToString());
                        if(found == false){
                            id2sheetnames[now_id] = sheet_names;
                            id2workbook[now_id] = new List<string>();
                            id2workbook[now_id].Add(file_name);
                            id2size[now_id] = new List<long>();
                            id2size[now_id].Add(fileinfo.Length);
                            dedup_workbooks.Add(file_name);
                            Console.WriteLine(dedup_workbooks.Count());
                            Console.WriteLine("2 add new size");
                            now_id++;
                        }
                        Console.WriteLine("end iter....");
                    }catch{
                        Console.WriteLine("error");
                        continue;
                    }
                    Console.WriteLine("1 dict name: " + dict.Name);
                }
                Console.WriteLine("sssssss");
                Console.WriteLine("dict name: " + dict.Name);
            }
            Console.WriteLine("wwwwwww");
            Console.WriteLine(id2sheetnames);
            string json = JsonConvert.SerializeObject(id2sheetnames);
            File.WriteAllText("/datadrive-2/data/top10domain_test/id2sheetnames.json", json);
            Console.WriteLine(id2workbook);
            json = JsonConvert.SerializeObject(id2workbook);
            File.WriteAllText("/datadrive-2/data/top10domain_test/id2workbook.json", json);
            Console.WriteLine(dedup_workbooks);
            Console.WriteLine(dedup_workbooks.Count());
            json = JsonConvert.SerializeObject(dedup_workbooks);
            File.WriteAllText("/datadrive-2/data/top10domain_test/dedup_workbooks.json", json);
        }
        public int column_id(string column_cha){
            int index=1;
            int result = 0;
            while(index<=column_cha.Count()){
                int cha_num = (int)column_cha[column_cha.Count()-index]-64;
                int sub_index=0;
                int di = 1;
                while(sub_index<index-1){
                    di*=26;
                    sub_index+=1;
                }
                result+=di*cha_num;
                index+=1;
            }
            return result;
        }
        public bool range_is_in(string formula_range, string dv_range){
            string[] split_dv_range = dv_range.Split(':');
            string range_start_cell = split_dv_range[0];
            string range_end_cell = split_dv_range[1];
            int range_start_number_index = 0;
            foreach(var cha in range_start_cell){
                if(!Char.IsNumber(cha)){
                    range_start_number_index += 1;
                    continue;
                }
                break;
            }
            int range_end_number_index = 0;
            foreach(var cha in range_end_cell){
                if(!Char.IsNumber(cha)){
                    range_end_number_index += 1;
                    continue;
                }
                break;
            }
            string start_column = range_start_cell.Substring(0,range_start_number_index);
            string start_row = range_start_cell.Substring(range_start_number_index);
            string end_column = range_end_cell.Substring(0,range_end_number_index);
            string end_row = range_end_cell.Substring(range_end_number_index);
            
            bool is_range = false;
            foreach(var cha in formula_range){
                if(cha == ':'){
                    is_range = true;
                    break;
                }
            }
            if(is_range == true){
                string[] split_formula_range = formula_range.Split(':');
                string formula_start_cell = split_dv_range[0];
                string formula_end_cell = split_dv_range[1];
                int formula_start_number_index = 0;
                foreach(var cha in formula_start_cell){
                    if(!Char.IsNumber(cha)){
                        formula_start_number_index += 1;
                        continue;
                    }
                    break;
                }
                int formula_end_number_index = 0;
                foreach(var cha in formula_end_cell){
                    if(!Char.IsNumber(cha)){
                        formula_end_number_index += 1;
                        continue;
                    }
                    break;
                }
                string formula_start_column = formula_start_cell.Substring(0,formula_start_number_index);
                string formula_start_row = formula_start_cell.Substring(formula_start_number_index);
                string formula_end_column = formula_end_cell.Substring(0,formula_end_number_index);
                string formula_end_row = formula_end_cell.Substring(formula_end_number_index);

                int formula_column = column_id(formula_start_column);
                int formula_row = int.Parse(formula_start_row);
                bool is_in = true;
                for(int column=column_id(start_column); column<=column_id(end_column); column++){
                    for(int row=int.Parse(start_row); row<=int.Parse(end_row); row++){
                        int temp_for_col = formula_column;
                        int temp_for_row = formula_row;
                        bool found=false;
                        for(int for_c = temp_for_col; for_c <= column_id(formula_end_column); for_c++){
                            if(for_c == column){
                                for(int for_r = temp_for_row; for_r <= int.Parse(formula_end_row); for_r++){
                                    if(for_r==row){
                                        formula_column = column;
                                        formula_row = row;
                                        found = true;
                                    }
                                }
                            }
                        }
                        if(found==false){
                            is_in=true;
                            break;
                        }
                    }   
                }
                return is_in;
            }else{
                int formula_start_number_index = 0;
                foreach(var cha in formula_range){
                    if(!Char.IsNumber(cha)){
                        formula_start_number_index += 1;
                        continue;
                    }
                    break;
                }
                string formula_start_column = formula_range.Substring(0,formula_start_number_index);
                string formula_start_row = formula_range.Substring(formula_start_number_index);
                bool is_in = false;
                Console.WriteLine(formula_start_column);
                Console.WriteLine(formula_start_row);
                for(int column=column_id(start_column); column<=column_id(end_column); column++){
                    for(int row=int.Parse(start_row); row<=int.Parse(end_row); row++){
                        if(column==column_id(formula_start_column) && row==int.Parse(formula_start_row)){
                            is_in=true;
                        }
                    }   
                }
                return is_in;
            }
            
        }

        public List<string> get_refs(string formula, bool need_jump){
            
            var root_node = ExcelFormulaParser.Parse(formula);
            var all_nodes = ExcelFormulaParser.AllNodes(root_node);
    
            List<string> result = new List<string>();
            int index=0;
            foreach(var node in all_nodes){
                Node new_node = new Node();
                new_node.Term = node.Term.Name;
                if(node.Term.Name=="key symbol" && need_jump){
                    var last_node = all_nodes.ToArray()[index-1];
                    var next_node = all_nodes.ToArray()[index+1];
                    string last_cha = last_node.Token.ValueString;
                    string next_cha = last_node.Token.ValueString;
                    string range_string = last_cha+':'+next_cha;
                    result.Add(range_string);
                }
                if(node.Term.Name=="CellToken"){
                    if(need_jump){
                        if(index!=0){
                            if(all_nodes.ToArray()[index-1].Term.Name=="key symbol"){
                                continue;
                            }
                        }
                        if(index!=all_nodes.ToArray().Count()-1){
                            if(all_nodes.ToArray()[index+1].Term.Name=="key symbol"){
                                continue;
                            }
                        }
                    }
                    result.Add(node.Token.ValueString);
                }
        
                index+=1;
            }
            return result;
        }

        public void check_range(){
            List<DVInfo> custom_list;
            if(!File.Exists("data/types/custom/custom_list.json")){
                Console.WriteLine("no formular number dictionalry exists. Please run getDictionary<string, int> first！");
                return;
            } else{
                string jsonstring = File.ReadAllText("data/types/custom/custom_list.json");
                custom_list = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
            int is_in = 0;
            int not_in = 0;
            int null_dv = 0;
            foreach(var custom_dv in custom_list){
                if(custom_dv.Value==""){
                    null_dv+=1;
                    continue;
                }
                List<string> form_refs = get_refs(custom_dv.Value, true);
                bool is_in_range = false;
                Console.WriteLine("XXXXXXXXXXXXXXXXXXXXXXXXXXXX");
                Console.WriteLine(custom_dv.ID);
                Console.WriteLine(custom_dv.Value);
                Console.WriteLine(custom_dv.RangeAddress);
                foreach(var one_ref in form_refs){
                    is_in_range = range_is_in(one_ref,custom_dv.RangeAddress);
                    if(is_in_range){
                        break;
                    }
                }
                if(is_in_range || form_refs.Count()==0){
                    is_in+=1;
                }else{
                    not_in+=1;
                }
                Console.WriteLine(is_in_range);
            }
            Console.WriteLine(is_in);
            Console.WriteLine(not_in);
            Console.WriteLine(null_dv);
        }
        public Shift get_shift_of_one_ref(string dv_ranges, string one_ref){
            Console.WriteLine("######################");
            Shift result = new Shift();
            string[] ranges = dv_ranges.Split(' ');

            string[] split_dv_range_0 = ranges[0].Split(':');
            string range_start_cell_0 = split_dv_range_0[0];
            string range_end_cell_0 = split_dv_range_0[1];
            int range_start_number_index_0 = 0;
            
            foreach(var cha in range_start_cell_0){
                if(!Char.IsNumber(cha)){
                    range_start_number_index_0 += 1;
                    continue;
                }
                break;
            }
            int range_end_number_index_0 = 0;
            foreach(var cha in range_end_cell_0){
                if(!Char.IsNumber(cha)){
                    range_end_number_index_0 += 1;
                    continue;
                }
                break;
            }
            int formula_start_number_index = 0;

            result.is_abs_column=false;
            result.is_abs_row= false;
            if(one_ref[0] == '$'){
                result.is_abs_column = true;
                one_ref = one_ref.Substring(1);
            }
            foreach(var cha in one_ref){
                if(!Char.IsNumber(cha) && cha != '$'){
                    formula_start_number_index += 1;
                    continue;
                }
                break;
            }
            if(one_ref[formula_start_number_index] == '$'){
                result.is_abs_row = true;
            }

            string formula_start_column = one_ref.Substring(0,formula_start_number_index);
            string formula_start_row;
            if(result.is_abs_row){
                formula_start_row = one_ref.Substring(formula_start_number_index+1);
            }
            else{
                formula_start_row = one_ref.Substring(formula_start_number_index);
            }


            foreach(var dv_range in ranges){
                string[] split_dv_range = dv_range.Split(':');
                string range_start_cell = split_dv_range[0];
                string range_end_cell = split_dv_range[1];
                int range_start_number_index = 0;
                
                foreach(var cha in range_start_cell){
                    if(!Char.IsNumber(cha)){
                        range_start_number_index += 1;
                        continue;
                    }
                    break;
                }
                int range_end_number_index = 0;
                foreach(var cha in range_end_cell){
                    if(!Char.IsNumber(cha)){
                        range_end_number_index += 1;
                        continue;
                    }
                    break;
                }
                string dvrange_start_column = range_start_cell.Substring(0,range_start_number_index);
                string dvrange_start_row = range_start_cell.Substring(range_start_number_index);
                string dvrange_end_column = range_end_cell.Substring(0,range_end_number_index);
                string dvrange_end_row = range_end_cell.Substring(range_end_number_index);
                Console.WriteLine("dvrange_start_column:"+column_id(dvrange_start_column));
                Console.WriteLine("dvrange_start_row:"+dvrange_start_row);
                Console.WriteLine("dvrange_end_column:"+column_id(dvrange_end_column));
                Console.WriteLine("dvrange_end_row:"+dvrange_end_row);
                

                int shift_column = column_id(formula_start_column)-column_id(dvrange_start_column);
                int shift_row = int.Parse(formula_start_row) - int.Parse(dvrange_start_row);

                result.cells = new List<Cell>();
                for(int column=column_id(dvrange_start_column); column<= column_id(dvrange_end_column); column++ ){
                    for(int row=int.Parse(dvrange_start_row); row <= int.Parse(dvrange_end_row); row++){
                        Cell shifted_cell = new Cell();
                        if(result.is_abs_column){
                            shifted_cell.column = column_id(formula_start_column);
                        }else{
                            shifted_cell.column = column+shift_column;
                        }
                        Console.WriteLine("dvrange_column:"+column.ToString());
                        Console.WriteLine("dvrange_row:"+row.ToString());
                        Console.WriteLine("shifted_cell.column:"+shifted_cell.column.ToString());
                        if(result.is_abs_row){
                            shifted_cell.row = int.Parse(formula_start_row);
                        }
                        else{
                            shifted_cell.row = row+shift_row;
                        }
                        
                        result.cells.Add(shifted_cell);
                        Console.WriteLine("shifted_cell.row:"+shifted_cell.column.ToString());
                    }
                }
                if(result.is_abs_column){
                    result.column = column_id(formula_start_column);
                }else{
                    result.column = shift_column;
                }
                if(result.is_abs_row){
                    result.column = column_id(formula_start_row);
                    result.row = shift_row;
                }
            }
            
            return result;
        }

        public void get_range_shift(){
            List<CustomDVInfo> custom_list;
            List<CustomDVInfo> result = new List<CustomDVInfo>();
            if(!File.Exists("data/types/custom/change_xml_cutom_list.json")){
                Console.WriteLine("no formular number dictionalry exists. Please run getDictionary<string, int> first！");
                return;
            } else{
                string jsonstring = File.ReadAllText("data/types/custom/change_xml_cutom_list.json");
                custom_list = JsonConvert.DeserializeObject<List<CustomDVInfo>>(jsonstring);
            }
            int is_in = 0;
            int not_in = 0;
            int null_dv = 0;

            foreach(var custom_dv in custom_list){
                if(custom_dv.Value==""){
                    null_dv+=1;
                    continue;
                }
                // if(custom_dv.ID!=11271){
                //     continue;
                // }
                if(custom_dv.shift != null){
                    result.Add(custom_dv);
                    continue;
                }
                List<string> form_refs = get_refs(custom_dv.Value, false);

                string[] split_dv_range_1 = custom_dv.RangeAddress.Split(' ');
                List<string> ranges = new List<string>();

                int min_column = 1000000;
                int min_row = 1000000;
                int max_column = 0;
                int max_row = 0;
                foreach(string split_dv in split_dv_range_1){
                    string[] split_dv_range = split_dv.Split(':');
                    
                    string range_start_cell = split_dv_range[0];
                    string range_end_cell = split_dv_range[1];
                    int range_start_number_index = 0;

                    foreach(var cha in range_start_cell){
                        if(!Char.IsNumber(cha)){
                            range_start_number_index += 1;
                            continue;
                        }
                        break;
                    }
                    int range_end_number_index = 0;
                    foreach(var cha in range_end_cell){
                        if(!Char.IsNumber(cha)){
                            range_end_number_index += 1;
                            continue;
                        }
                        break;
                    }
                    if(min_column > column_id(range_start_cell.Substring(0,range_start_number_index))){
                        min_column = column_id(range_start_cell.Substring(0,range_start_number_index));
                    }
                    if(min_row > int.Parse(range_start_cell.Substring(range_start_number_index))){
                        min_row = int.Parse(range_start_cell.Substring(range_start_number_index));
                    }
                    if(max_column < column_id(range_end_cell.Substring(0,range_end_number_index))){
                        max_column = column_id(range_end_cell.Substring(0,range_end_number_index));
                    }
                    if(max_row > int.Parse(range_end_cell.Substring(range_end_number_index))){
                        max_row = int.Parse(range_end_cell.Substring(range_end_number_index));
                    }
                }
                custom_dv.lty = min_column;
                custom_dv.ltx = min_row;
                custom_dv.rby = max_column;
                custom_dv.rbx = max_row;
                

                Console.WriteLine("XXXXXXXXXXXXXXXXXXXXXXXXXXXX");
                Console.WriteLine(custom_dv.ID);
                Console.WriteLine(custom_dv.Value);
                Console.WriteLine(custom_dv.RangeAddress);
                custom_dv.shift = new List<Shift>();
                foreach(var one_ref in form_refs){
                    Shift one_ref_result = get_shift_of_one_ref(custom_dv.RangeAddress, one_ref);
                    custom_dv.shift.Add(one_ref_result);
                }
                result.Add(custom_dv);
                // break;
            }
            
            Console.WriteLine(is_in);
            Console.WriteLine(not_in);
            Console.WriteLine(null_dv);
            saveAsJson(result, "data/types/custom/change_xml_shift_custom_list.json");
        }

        public void saveAsJson(object need_save_content, string file_name){
            string jsonData = JsonConvert.SerializeObject(need_save_content);
            File.WriteAllText(file_name, jsonData);
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
                if(formular != "FIND(\".\",D14:BW298,1)>=LEN(D14:BW298)-2"){
                    continue;
                } 
                // if(formular != "IF(K6=\"2012-2013\",\"2011-2012\",IF(K6=\"2011-2012\", \"2010-2011\",\"Academic Year\"))"){
                //     continue;
                // }
                // if(formular != "ISLOGICAL(J31)"){
                //     continue;
                // }
                
                // if(formular == ""){
                //     continue;
                // } 
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
                // break;
            }
            // saveAsJson(all_function_number, "all_func_number.json");
            // saveAsJson(duplicated_function_number, "duplicated_function_number.json");
        }
        public void test_parse_refence(){
            List<string> crosstable = new List<string>();
            List<List<string>> formula_lists;
            if(!File.Exists("../analyze-dv-1/fortune500_formula_list.json")){
                Console.WriteLine("no formular number dictionalry exists. Please run getDictionary<string, int> first！");
                return;
            } else{
                string jsonstring = File.ReadAllText("../analyze-dv-1/fortune500_formula_list.json");
                formula_lists = JsonConvert.DeserializeObject<List<List<string>>>(jsonstring);
            }
            Dictionary<string, int> all_function_number = new Dictionary<string, int>();
            Dictionary<string, int> duplicated_function_number = new Dictionary<string, int>();
            
            foreach (List<string> formula_tuple in formula_lists)
            {
                string origin_sheetname = formula_tuple[0];
                string formular = formula_tuple[1];
                // var formular = kvp.Key;
                // if(formular != "'Design Regulator'!R[-66]C[2]*1000"){
                //     continue;
                // } 
                // if(formular != "IF(K6=\"2012-2013\",\"2011-2012\",IF(K6=\"2011-2012\", \"2010-2011\",\"Academic Year\"))"){
                //     continue;
                // }
                // if(formular != "ISLOGICAL(J31)"){
                //     continue;
                // }
                
                // if(formular == ""){
                //     continue;
                // } 
                // string formular = "'Design Regulator'!R[-66]C[2]*1000";
                Console.WriteLine(formular);
                try{
                    var root_node = ExcelFormulaParser.Parse(formular);
                    var all_nodes = ExcelFormulaParser.AllNodes(root_node);
            
                    List<string> functions = new List<string>();
                    bool is_crosstable = false;
                    bool is_all_same = true;
                    foreach(var node in all_nodes){
                        // Console.WriteLine("#############");
                        // Console.WriteLine("TAG");
                        // Console.WriteLine(node.Tag);
                        // Console.WriteLine("Term");
                        // Console.WriteLine(node.Term);
                        if(node.Term.ToString() == "SheetNameQuotedToken"){
                            is_crosstable = true;
                            Console.WriteLine(node.Token.GetType().GetProperties());
                            string sheetname = node.Token.ToString().Split("'!")[0];
                            Console.WriteLine("sheetname: " + sheetname);
                            Console.WriteLine("origin_sheetname: " + origin_sheetname);
                            if(origin_sheetname != sheetname){
                                is_all_same = false;
                            }
                        }
                        // Console.WriteLine("Term");
                        // Console.WriteLine(node.Token);
                    }
                    if(is_crosstable && !is_all_same){
                        crosstable.Add(formular);
                    }
                    foreach(var func in functions){
                        if(!all_function_number.ContainsKey(func)){
                            all_function_number[func] = 0;
                            duplicated_function_number[func] = 0;
                        }
                        // all_function_number[func] += kvp.Value;
                        // duplicated_function_number[func] += 1;
                    }
                // break;
                }catch{
                    continue;
                }
                
                
            }
            saveAsJson(crosstable, "../analyze-dv-1/crosstable_formulas.json");
        }

        public void get_training_refcell(){
            string jsonstring = File.ReadAllText("../analyze-dv-1/fine_tune_positive.json");
            List<List<List<string>>> fine_tune_positive = JsonConvert.DeserializeObject<List<List<List<string>>>>(jsonstring);
            int count=1;
             DateTime start_time = DateTime.Now;	//获取当前时间
    
                
            foreach(var pair in fine_tune_positive){
                Console.WriteLine(count.ToString() + "/" + fine_tune_positive.Count().ToString());
                List<string> auchor_info = pair[0];
                List<string> positive_info = pair[1];
                if(count == 50){
                    break;
                }
                get_all_refcell(auchor_info[1], auchor_info[0].Split('/')[auchor_info[0].Split('/').Length - 1]);
                // if(positive_info[0].Split('/')[positive_info[0].Split('/').Length - 1] == "1b8ba579545e8c428178e6a3d5c37dc5_c3RyZWFtLW1lY2hhbmljcy5jb20JNTAuNjIuMTcyLjExMw==.xlsx---Monitoring Data---3---3"){
                //     // continue;
                //     Console.WriteLine(positive_info[0].Split('/')[positive_info[0].Split('/').Length - 1]);
                //     Console.WriteLine(positive_info[1]);
                // }
                // get_all_refcell(positive_info[1], positive_info[0].Split('/')[positive_info[0].Split('/').Length - 1]);
                count += 1;
                // break;
            }
            DateTime end_time = DateTime.Now;	//获取当前时间
            TimeSpan ts = end_time - start_time;	//计算时间差
            double time = ts.TotalSeconds;	//将时间差转换为秒
            Console.WriteLine("all time:" + time.ToString());
            Console.WriteLine("avg time:" + (time / count).ToString());
        }
        public void batch_get_all_refcell(){
            // DirectoryInfo root = new DirectoryInfo("/datadrive-2/data/fortune500_test/afterfeature_test/");
            // DirectoryInfo root = new DirectoryInfo("/datadrive-2/data/top10domain_test/afterfeature_test/");
            DirectoryInfo root = new DirectoryInfo("/datadrive-2/data/fortune500_test/company_model1_res/");
            FileInfo[] files=root.GetFiles();
            string jsonstring = File.ReadAllText("../analyze-dv-1/fortune500_formulatoken2r1c1.json");
            // string jsonstring = File.ReadAllText("../analyze-dv-1/top10domain_formulatoken2r1c1.json");
            Dictionary<string, string> formulatoken2r1c1 = JsonConvert.DeserializeObject<Dictionary<string, string>>(jsonstring);
            int count = 0;
            foreach(var fileinfo in files){
            // foreach(string formula_token in formulatoken2r1c1.Keys){
                string filename = fileinfo.Name;
                string formula_token = filename.Replace(".json", "");
                // Console.WriteLine(formula_token);
                // if(formula_token != "266755824202847486563786796188818448742-ldc_5f00_tools_2d00_ext16_5f00_pcb-thick-_3d00_-1.5mm.xlsx---Spiral_Inductor_Designer---36---21"){
                //     continue;
                // }
                // Console.WriteLine(formula_token);
                string r1c1 = formulatoken2r1c1[formula_token];
                Console.WriteLine("count:" + count.ToString());
                count += 1;
                try{
                    // if(!File.Exists("/datadrive-2/data/top10domain_test/model1_res/"+formula_token+".json")){
                    // // if(!File.Exists("/datadrive-2/data/fortune500_test/model1_res/"+formula_token+".json")){
                    //     continue;
                    // }
                    // string jsonstring1 = File.ReadAllText("/datadrive-2/data/fortune500_test/model1_res/"+formula_token+".json");
                    string jsonstring1 = File.ReadAllText("/datadrive-2/data/fortune500_test/company_model1_res/"+formula_token+".json");
                    List<string> mode1_res = JsonConvert.DeserializeObject<List<string>>(jsonstring1);
                    string found_formula_token = mode1_res[1];
                    // if(found_formula_token != "266755824202847486563786796188818448742-ldc_5f00_tools_2d00_ext16_5f00_pcb-thick-_3d00_-1.5mm.xlsx---Spiral_Inductor_Designer---36---21"){
                    //     continue;
                    // }
                    get_all_refcell(r1c1, formula_token);
                    get_all_refcell(formulatoken2r1c1[found_formula_token], found_formula_token);
                    // break;
                }catch{
                    Console.WriteLine("not exists");
                    continue;
                }
            }
        }

        public void get_all_refcell(string r1c1, string formula_token){
        
            try{
                var root_node = ExcelFormulaParser.Parse(r1c1);
                var all_nodes = ExcelFormulaParser.AllNodes(root_node);
                List<Node> new_temp_content = new List<Node>();
                foreach(var node in all_nodes){
                    Node new_node = new Node();
                    // Console.WriteLine("#############");
                    // new_node.Term = node.Term.ToString();

                    // new_node.Token = node.Token.ToString();
                    // Console.WriteLine(node.Tag.ToString());
                    // Console.WriteLine(node.Term.Name);
                    new_node.Term = node.Term.Name;
                    try{
                        new_node.Token = node.Token.ValueString;
                    }catch{
                        new_node.Token = "NULL";
                    }
                    // if(node.Term.Name == "ExcelFunction" || node.Term.Name == "BoolToken"){
                    //     // Console.WriteLine(node.Token.ValueString);
                    //     new_node.Token = node.Token.ValueString;
                    // }
                    // if(node.Term.Name=="CellToken"){
                    //     Console.WriteLine(node.Token.ValueString);
                    //     new_node.Token = "cell token";
                    // }

                    // if(node.Term.Name=="NumberToken"){

                    //     Console.WriteLine(node.Token.Value);
                    //     new_node.Token = node.Token.Value.ToString();
                    // }
                    // if(node.Term.Name=="TextToken"){
                    //     Console.WriteLine(node.Token.Value);
                    //     new_node.Token = node.Token.Value.ToString();
                    // }
                    // if(node.Term.Name=="NameToken"){
                    //     Console.WriteLine(node.Token.Value);
                    //     new_node.Token = node.Token.Value.ToString();
                    // }
                    // if(node.Term.Name == "StructuredReferenceElement"){
                    //     Console.WriteLine(node.Token);
                    //     new_node.Token = node.Token.Value.ToString();
                    // }
                    
                    new_temp_content.Add(new_node);
                }
                Console.WriteLine("save:" + "/datadrive-2/data/fortune500_test/formula_template/"+formula_token + ".json");
                // saveAsJson(new_temp_content, "/datadrive-2/data/fortune500_test/formula_template/"+formula_token + ".json");
                saveAsJson(new_temp_content, "/datadrive-2/data/fortune500_test/formula_template/"+formula_token + ".json");
            }catch{
                Console.WriteLine("Error");
                return;
            }
            
            
        }

        public void analyze_differ(){
            string jsonstring = File.ReadAllText("../analyze-dv-1/shift_fail_r1c1_pairs.json");
            List<List<string>> shift_fail_r1c1_pairs = JsonConvert.DeserializeObject<List<List<string>>>(jsonstring);
            // [[res_r1c1, formula_r1c1], ....]
            List<List<List<string>>> res = new List<List<List<string>>>();
            // [[[string, string,...], [string, string, ...]], ]
            foreach(var pair in shift_fail_r1c1_pairs){
                string res_r1c1 = pair[0];
                string target_r1c1 = pair[1];
                
                var res_root_node = ExcelFormulaParser.Parse(res_r1c1);
                var res_all_nodes = ExcelFormulaParser.AllNodes(res_root_node);
                var target_root_node = ExcelFormulaParser.Parse(target_r1c1);
                var target_all_nodes = ExcelFormulaParser.AllNodes(target_root_node);

                List<string> target_list = new List<string>();
                List<string> res_list = new List<string>();
                List<string> differ_target_list = new List<string>();
                List<string> differ_res_list = new List<string>();
                List<List<string>> tuple_list = new List<List<string>>();
                foreach(var node in res_all_nodes){
                    // Type type = typeof(node.Term);
                    // var property = type.GetProperty("Value");
                    // Console.WriteLine("property:"+property.ToString());
                    // Console.WriteLine("node.Term.Value:"+node.Term.ToString());
                    // res_list.Add(node.Term.Value);
                    // if (property == null)
                    // {
                    //     res_list.Add("NULL");
                    // }
                    // else{
                    // var info = node.Term.GetType().GetProperty("Value");
                
                    // Console.WriteLine("######");
                    // Console.WriteLine(node.Term.GetType());
                    // Console.WriteLine(info);
                    try{
                        res_list.Add(node.Token.Value.ToString());
                    }catch{
                        res_list.Add("NULL");
                    }
                    
                    // }
                
                }
                foreach(var node in target_all_nodes){
                    // var property = node.Term.GetProperty("Value");
                    // if (property == null)
                    // {
                    //     target_list.Add("NULL");
                    // }
                    // else{
                    try{
                        target_list.Add(node.Token.Value.ToString());
                    }catch{
                        target_list.Add("NULL");
                    }
                    
                    // }
                
                }
                Console.WriteLine("################");
                Console.WriteLine("res_node:"+res_r1c1);
                Console.WriteLine("target_node:"+target_r1c1);
                for(var i=0; i<res_list.Count(); i++){
                    var res_node = res_list[i];
                    var target_node = target_list[i];
                    // Console.WriteLine("res_node:"+res_node);
                    // Console.WriteLine("target_node:"+target_node);
                    if(res_node != target_node){
                        Console.WriteLine("res_node:"+res_node);
                        Console.WriteLine("target_node:"+target_node);
                        differ_target_list.Add(target_node);
                        differ_res_list.Add(res_node);
                        
                    }
                }
                tuple_list.Add(differ_res_list);
                tuple_list.Add(differ_target_list);

                res.Add(tuple_list);
            }
        saveAsJson(res, "../analyze-dv-1/shift_fail_differ_pairs.json");

        }

        public void save_all_formula_template(bool change_constant){
            List<string> formulas_20000sheets;
            // if(!File.Exists("../analyze-dv-1/top10domain_formulas_list.json")){
            //     Console.WriteLine("no formular number dictionalry exists. Please run getDictionary<string, int> first！");
            //     return;
            // } else{
            // string jsonstring = File.ReadAllText("../analyze-dv-1/top10domain_formulas_list.json");
            string jsonstring = File.ReadAllText("../analyze-dv-1/fortune500_formulas_list.json");
            formulas_20000sheets = JsonConvert.DeserializeObject<List<string>>(jsonstring);
            // }
           
            Dictionary<string, int> r1c12template = new Dictionary<string, int>();
            int all_num = 0;
            List<R1C1Template> result = new List<R1C1Template>();
            foreach (string r1c1formula in formulas_20000sheets)
            {
                var formular = r1c1formula;
                
                // if(formular != "AVERAGE(R[2]C:R[396]C)"){
                //     continue;
                // }
            
                all_num += 1;
                Console.WriteLine(formular);
                try{
                    // if(formular != "WMS-EZ-B22-50P"){
                    //     continue;
                    // }
                    var root_node = ExcelFormulaParser.Parse(formular);
                    var all_nodes = ExcelFormulaParser.AllNodes(root_node);
                    List<string> functions = new List<string>();

                    List<Node> new_temp_content = new List<Node>();
                    foreach(var node in all_nodes){
                        Node new_node = new Node();
                        // Console.WriteLine("#############");
                        // new_node.Term = node.Term.ToString();

                        // new_node.Token = node.Token.ToString();
                        // Console.WriteLine(node.Tag.ToString());
                        // Console.WriteLine(node.Term.Name);
                        new_node.Term = node.Term.Name;
                        if(node.Term.Name == "ExcelFunction" || node.Term.Name == "BoolToken"){
                            // Console.WriteLine(node.Token.ValueString);
                            new_node.Token = node.Token.ValueString;
                        }
                        if(node.Term.Name=="CellToken"){
                            Console.WriteLine(node.Token.ValueString);
                            new_node.Token = "cell token";
                        }
                        if(change_constant){
                            // if(node.Term.Name.IndexOf("Token")>-1){
                            //      //     Console.WriteLine(node.Token.Value);
                            //     new_node.Token = node.Token.Value.ToString();
                
                            // }
                            if(node.Term.Name=="NumberToken"){

                                Console.WriteLine(node.Token.Value);
                                new_node.Token = node.Token.Value.ToString();
                            }
                            if(node.Term.Name=="TextToken"){
                                Console.WriteLine(node.Token.Value);
                                new_node.Token = node.Token.Value.ToString();
                            }
                            if(node.Term.Name=="NameToken"){
                                Console.WriteLine(node.Token.Value);
                                new_node.Token = node.Token.Value.ToString();
                            }
                            // if(node.Term.Name == "StructuredReferenceElement"){
                            //     Console.WriteLine(node.Token.Value);
                            //     new_node.Token = node.Token.Value.ToString();
                            // }
                        }
                        new_temp_content.Add(new_node);
                    }

                    bool is_found = false;
                    R1C1Template found_template = new R1C1Template();
                    int found_index = 0;
                    int max_id = 0;
                    foreach(var tempelte in result){
                        bool is_same = true;
                        if(tempelte.id > max_id){
                            max_id=tempelte.id;
                        }
                        if(tempelte.node_list.Count() != new_temp_content.Count()){
                            is_same = false;
                            found_index += 1;
                            continue;
                        }
                        for(var index=0;index<new_temp_content.Count();index++){
                            if(tempelte.node_list[index].Term != new_temp_content[index].Term || tempelte.node_list[index].Token != new_temp_content[index].Token){
                                is_same = false;
                                break;
                            }
                        }
                        if(is_same == false){
                            found_index += 1;
                            continue;
                        }
                        is_found=true;
                        found_template = tempelte;
                        break;
                    }
                    Console.WriteLine("xxx");
                    if(is_found == true){
                        found_template.formulas.Add(formular);
                        r1c12template.Add(formular, found_template.id);
                        // continue;
                    }else{
                        
                        found_template.id = max_id+1;
                        r1c12template.Add(formular, found_template.id);
                        found_template.node_list = new_temp_content;
                        Console.WriteLine("found_template:" + found_template.ToString());
                        List<string> formulas = new List<string>();
                        formulas.Add(formular);
                        found_template.formulas = formulas;
                        result.Add(found_template);
                        
                    }   
                }
                catch{
                    continue;
                }
                        
            }
            Console.WriteLine("all_dv_number:"+all_num.ToString());
            if(change_constant){
                // saveAsJson(result, "../analyze-dv-1/formula_r1c1_top10domain_template_constant.json");
                // saveAsJson(r1c12template, "../analyze-dv-1/r1c12template_top10domain_constant.json");
                saveAsJson(result, "../analyze-dv-1/formula_r1c1_fortune500_template_constant.json");
                saveAsJson(r1c12template, "../analyze-dv-1/r1c12template_fortune500_constant.json");
            }else{
                // saveAsJson(result, "../analyze-dv-1/formula_r1c1_top10domain_template.json");
                // saveAsJson(r1c12template, "../analyze-dv-1/r1c12template_top10domain.json");
                saveAsJson(result, "../analyze-dv-1/formula_r1c1_fortune500_template.json");
                saveAsJson(r1c12template, "../analyze-dv-1/r1c12template_fortune500.json");
            }
            
            // saveAsJson(duplicated_function_number, "duplicated_function_number.json");
        }
        
        public void save_all_template(bool change_constant){
            List<DVInfo> dvinfos;
            if(!File.Exists("data/types/custom/dedup_shifted_custom_info.json")){
                Console.WriteLine("no formular number dictionalry exists. Please run getDictionary<string, int> first！");
                return;
            } else{
                string jsonstring = File.ReadAllText("data/types/custom/change_xml_cutom_list.json");
                dvinfos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
           
            int all_num = 0;
            List<Template> result = new List<Template>();
            foreach (DVInfo dvinfo in dvinfos)
            {
                var formular = dvinfo.Value;
                
                if(formular == ""){
                    continue;
                }
                all_num += 1;
                Console.WriteLine(formular);
                var root_node = ExcelFormulaParser.Parse(formular);
                var all_nodes = ExcelFormulaParser.AllNodes(root_node);
                List<string> functions = new List<string>();

                List<Node> new_temp_content = new List<Node>();
                foreach(var node in all_nodes){
                    Node new_node = new Node();
                    // Console.WriteLine("#############");
                    // new_node.Term = node.Term.ToString();

                    // new_node.Token = node.Token.ToString();
                    // Console.WriteLine(node.Tag.ToString());
                    // Console.WriteLine(node.Term.Name);
                    new_node.Term = node.Term.Name;
                    if(node.Term.Name == "ExcelFunction" || node.Term.Name == "BoolToken"){
                        // Console.WriteLine(node.Token.ValueString);
                        new_node.Token = node.Token.ValueString;
                    }
                    if(node.Term.Name=="CellToken"){
                        Console.WriteLine(node.Token.ValueString);
                        new_node.Token = "cell token";
                    }
                    if(change_constant){
                        if(node.Term.Name=="NumberToken"){

                            Console.WriteLine(node.Token.Value);
                            new_node.Token = node.Token.Value.ToString();
                        }
                        if(node.Term.Name=="TextToken"){
                            Console.WriteLine(node.Token.Value);
                            new_node.Token = node.Token.Value.ToString();
                        }
                    }
                    new_temp_content.Add(new_node);
                }

                bool is_found = false;
                Template found_template = new Template();
                int found_index = 0;
                int max_id = 0;
                foreach(var tempelte in result){
                    bool is_same = true;
                    if(tempelte.id > max_id){
                        max_id=tempelte.id;
                    }
                    if(tempelte.node_list.Count() != new_temp_content.Count()){
                        is_same = false;
                        found_index += 1;
                        continue;
                    }
                    for(var index=0;index<new_temp_content.Count();index++){
                        if(tempelte.node_list[index].Term != new_temp_content[index].Term || tempelte.node_list[index].Token != new_temp_content[index].Token){
                            is_same = false;
                            break;
                        }
                    }
                    if(is_same == false){
                        found_index += 1;
                        continue;
                    }
                    is_found=true;
                    found_template = tempelte;
                    break;
                }
                if(is_found == true){
                    found_template.dvid_list.Add(dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString());
                    found_template.formulas_list.Add(formular);
                    found_template.file_sheet_name_list.Add(dvinfo.FileName+"-------------"+dvinfo.SheetName);
                    found_template.number+=1;
                }else{
                    found_template.id = max_id+1;
                    found_template.node_list = new_temp_content;
                    found_template.dvid_list = new List<string>();
                    found_template.dvid_list.Add(dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString());
                    found_template.formulas_list = new List<string>();
                    found_template.formulas_list.Add(formular);
                    found_template.file_sheet_name_list = new List<string>();
                    found_template.file_sheet_name_list.Add(dvinfo.FileName+"-------------"+dvinfo.SheetName);
                    found_template.number =1;
                    result.Add(found_template);
                    
                }           
            }
            Console.WriteLine("all_dv_number:"+all_num.ToString());
            if(change_constant){
                saveAsJson(result, "data/types/custom/all_templates_specific_constant.json");
            }else{
                saveAsJson(result, "data/types/custom/change_xml_all_templates.json");
            }
            
            // saveAsJson(duplicated_function_number, "duplicated_function_number.json");
        }

        public void save_all_boundary_template(bool change_constant){
            List<DVInfo> dvinfos;
            if(!File.Exists("data/types/boundary/boundary_list.json")){
                Console.WriteLine("no formular number dictionalry exists. Please run getDictionary<string, int> first！");
                return;
            } else{
                string jsonstring = File.ReadAllText("data/types/boundary/boundary_list.json");
                dvinfos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring);
            }
           
            int all_num = 0;
            List<Template> result = new List<Template>();
            foreach (DVInfo dvinfo in dvinfos)
            {
                var formular = dvinfo.Value;
                
                if(formular == ""){
                    continue;
                }
                all_num += 1;
                Console.WriteLine(formular);
                var root_node = ExcelFormulaParser.Parse(formular);
                var all_nodes = ExcelFormulaParser.AllNodes(root_node);
                List<string> functions = new List<string>();

                List<Node> new_temp_content = new List<Node>();
                foreach(var node in all_nodes){
                    Node new_node = new Node();
                    // Console.WriteLine("#############");
                    // new_node.Term = node.Term.ToString();

                    // new_node.Token = node.Token.ToString();
                    // Console.WriteLine(node.Tag.ToString());
                    // Console.WriteLine(node.Term.Name);
                    new_node.Term = node.Term.Name;
                    if(node.Term.Name == "ExcelFunction" || node.Term.Name == "BoolToken"){
                        // Console.WriteLine(node.Token.ValueString);
                        new_node.Token = node.Token.ValueString;
                    }
                    if(node.Term.Name=="CellToken"){
                        Console.WriteLine(node.Token.ValueString);
                        new_node.Token = "cell token";
                    }
                    if(change_constant){
                        if(node.Term.Name=="NumberToken"){

                            Console.WriteLine(node.Token.Value);
                            new_node.Token = node.Token.Value.ToString();
                        }
                        if(node.Term.Name=="TextToken"){
                            Console.WriteLine(node.Token.Value);
                            new_node.Token = node.Token.Value.ToString();
                        }
                    }
                    new_temp_content.Add(new_node);
                }

                bool is_found = false;
                Template found_template = new Template();
                int found_index = 0;
                int max_id = 0;
                foreach(var tempelte in result){
                    bool is_same = true;
                    if(tempelte.id > max_id){
                        max_id=tempelte.id;
                    }
                    if(tempelte.node_list.Count() != new_temp_content.Count()){
                        is_same = false;
                        found_index += 1;
                        continue;
                    }
                    for(var index=0;index<new_temp_content.Count();index++){
                        if(tempelte.node_list[index].Term != new_temp_content[index].Term || tempelte.node_list[index].Token != new_temp_content[index].Token){
                            is_same = false;
                            break;
                        }
                    }
                    if(is_same == false){
                        found_index += 1;
                        continue;
                    }
                    is_found=true;
                    found_template = tempelte;
                    break;
                }
                if(is_found == true){
                    found_template.dvid_list.Add(dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString());
                    found_template.formulas_list.Add(formular);
                    found_template.file_sheet_name_list.Add(dvinfo.FileName+"-------------"+dvinfo.SheetName);
                    found_template.number+=1;
                }else{
                    found_template.id = max_id+1;
                    found_template.node_list = new_temp_content;
                    found_template.dvid_list = new List<string>();
                    found_template.dvid_list.Add(dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString());
                    found_template.formulas_list = new List<string>();
                    found_template.formulas_list.Add(formular);
                    found_template.file_sheet_name_list = new List<string>();
                    found_template.file_sheet_name_list.Add(dvinfo.FileName+"-------------"+dvinfo.SheetName);
                    found_template.number =1;
                    result.Add(found_template);
                    
                }           
            }
            Console.WriteLine("all_dv_number:"+all_num.ToString());
            if(change_constant){
                saveAsJson(result, "data/types/custom/all_templates_specific_constant.json");
            }else{
                saveAsJson(result, "data/types/custom/all_templates.json");
            }
            
            // saveAsJson(duplicated_function_number, "duplicated_function_number.json");
        }

  

        public void get_new_custom_json(){
            string jsonstring = File.ReadAllText("data/types/custom/dedup_shifted_custom_info.json");
            List<CustomDVInfo> dedup_shifted_custom_info = JsonConvert.DeserializeObject<List<CustomDVInfo>>(jsonstring);

            string jsonstring1 = File.ReadAllText("../analyze-dv-1/dedup_xml_dvinfo.json");
            Dictionary<string,List<XMLDVInfo>> xml_dvinfos = JsonConvert.DeserializeObject<Dictionary<string, List<XMLDVInfo>>>(jsonstring1);

            List<CustomDVInfo> new_custom_info = new List<CustomDVInfo>();

            int max_id = 0;
            Dictionary<string, int> file_batch = new Dictionary<string, int>();
            foreach(var custom_dvinfo in dedup_shifted_custom_info){
                bool need_add = true;
                foreach(var filesheet in xml_dvinfos.Keys){
                    // string filesheet = xml_dvinfo.Keys.ToArray()[0];
                    string[] file_sheet = filesheet.Split("------");
                    string file = file_sheet[0];
                    string sheet = file_sheet[1];
                    if(file==custom_dvinfo.FileName && sheet == custom_dvinfo.SheetName){
                        need_add = false;
                        if(!file_batch.ContainsKey(file)){  
                            file_batch.Add(file, custom_dvinfo.batch_id);
                        }
                        break;
                    }
                }
                if(need_add){
                    new_custom_info.Add(custom_dvinfo);
                    if(max_id < custom_dvinfo.ID){
                        max_id = custom_dvinfo.ID;
                    }
                }
            }

            foreach(var file_sheet_key in xml_dvinfos.Keys){
                // string file_sheet_key = xml_dvinfo.Keys.ToArray()[0];
                string[] file_sheet = file_sheet_key.Split("------");
                string file = file_sheet[0];
                string sheet = file_sheet[1];
                foreach(var xml_dvinfo in xml_dvinfos[file_sheet_key]){
                    bool is_found = false;
                    var found_dvinfo = dedup_shifted_custom_info[0];
                    foreach(var origin_dvinfo in dedup_shifted_custom_info){
                        if(origin_dvinfo.Value == xml_dvinfo.value && origin_dvinfo.RangeAddress == xml_dvinfo.range && origin_dvinfo.FileName == file && origin_dvinfo.SheetName==sheet){
                            is_found = true;
                            found_dvinfo = origin_dvinfo;
                            break;
                        }
                    }
                    if(is_found){
                        new_custom_info.Add(found_dvinfo);
                    }else{
                        CustomDVInfo new_dvinfo = new CustomDVInfo();
                        new_dvinfo.Type = XLAllowedValues.Custom;
                        new_dvinfo.Operator = XLOperator.Between;
                        
                        Console.WriteLine(xml_dvinfo.value);
                        Console.WriteLine(xml_dvinfo.range);
                        new_dvinfo.Value = xml_dvinfo.value;
                        new_dvinfo.MinValue = xml_dvinfo.value;
                        
                        new_dvinfo.RangeAddress = xml_dvinfo.range;
                        new_dvinfo.FileName = file;
                        new_dvinfo.SheetName = sheet;
                        new_dvinfo.batch_id = file_batch[file];
                        new_dvinfo.ID = max_id;
                        max_id += 1;
                        new_custom_info.Add(new_dvinfo);
                    }
                    
                }
                
                // break;
            }
            saveAsJson(new_custom_info, "data/types/custom/change_xml_cutom_list.json");
        }
    }
}